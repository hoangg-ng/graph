"""Complexity checks and output regression against pinned Git commit 4c6854b.

Historical tests require that commit locally. Trained-checkpoint regression is
skipped explicitly when the twelve seed-0 checkpoints or prepared data are absent.
"""
import contextlib
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import numpy as np
import torch

import profile_models
import run_eval
import run_followup_eval
import train_eval
from proposal_model import (
    ARMS,
    ExpertPool,
    LinearScaleEncoder,
    ProposalMAGE,
    parameter_count,
)
from profile_models import _applications, _count_flops
from report_complexity import build_report
from run_followup_eval import plan_jobs

ROOT = Path(__file__).resolve().parent
REFERENCE_COMMIT = "4c6854bfa98c2e3830fe82e24c592399e422cf67"
REFERENCE_FILES = (
    "proposal_model.py", "train_proposal.py", "models.py", "prep_proposal.py",
    "official_repro/src/models/MAGE.py", "official_repro/src/base/model.py",
)
# A fresh interpreter imports only the historical model source tree. In
# particular, its models and src modules cannot reuse current cached imports.
REFERENCE_RUNNER = """
import sys
from types import SimpleNamespace
import torch
sys.path.insert(0, sys.argv[1])
import train_proposal
torch.set_num_threads(3)
cases = torch.load(sys.argv[2], map_location='cpu', weights_only=False)
outputs = []
for case in cases:
    x = case['input']
    model, *_ = train_proposal._build_model(
        case['arm'], SimpleNamespace(**case['args']), x.shape[2], x.shape[3])
    model.load_state_dict(case['state'], strict=True)
    model.eval()
    with torch.inference_mode():
        outputs.append([model(x[:batch], bias=case['bias']) for batch in (1, 2)])
torch.save(outputs, sys.argv[3])
"""


def worker_args(arm):
    return train_eval.make_parser().parse_args([
        "--arm", arm, "--device", "cpu", "--out", "unused.json",
        "--model-dim", "16", "--experts", "4", "--topk", "2",
        "--graph-dim", "5", "--pdm-width", "8", "--pdm-depth", "1", "--bs", "2",
    ])


def make_args(nodes=7, model_dim=16, experts=4, topk=2, graph_dim=5):
    return SimpleNamespace(
        model_dim=model_dim,
        recur_num=experts,
        topk=topk,
        node_num=nodes,
        feature_dim=1,
        graph_dim=graph_dim,
        input_dim=3,
        output_dim=1,
        seq_len=96,
        horizon=12,
        minute=288,
        day=7,
    )


def make_input(batch=2, nodes=7):
    flow = torch.randn(batch, 96, nodes, 1)
    tod = torch.arange(96).remainder(288).float() / 288
    calendar = torch.cat(
        (
            tod[None, :, None, None].expand(batch, -1, nodes, -1),
            torch.zeros(batch, 96, nodes, 1),
        ),
        dim=-1,
    )
    return torch.cat((flow, calendar), dim=-1)


class ComplexityTests(unittest.TestCase):
    def build(self, arm):
        return ProposalMAGE(
            make_args(), arm, pdm_width=8, pdm_depth=1, dropout=0
        )

    def test_every_registered_proposal_arm_has_finite_expected_outputs(self):
        for arm in ARMS:
            model = self.build(arm).eval()
            for batch in (1, 2):
                with self.subTest(arm=arm, batch=batch), torch.inference_mode():
                    output = model(make_input(batch=batch))
                    self.assertEqual(tuple(output.shape), (batch, 12, 7, 1))
                    self.assertTrue(torch.isfinite(output).all())
            self.assertEqual(parameter_count(model), sum(p.numel() for p in model.parameters()))

    def test_existing_and_new_graph_application_budgets(self):
        expected = {
            "proposed": (45, 12, 57),
            "reversed": (45, 12, 57),
            "shared": (12, 12, 24),
            "independent": (12, 12, 24),
            "uniform": (0, 0, 0),
            "no_spatial": (0, 0, 0),
            "grouped": (3, 3, 6),
            "proposed_no_pdm": (45, 12, 57),
        }
        for arm, counts in expected.items():
            with self.subTest(arm=arm):
                model = self.build(arm)
                self.assertEqual(
                    (
                        model.positive_propagation_count,
                        model.negative_propagation_count,
                        model.propagation_count,
                    ),
                    counts,
                )
                self.assertEqual(_applications(arm, model)[:3], counts)

    def test_no_spatial_and_uniform_do_not_apply_graph_operators(self):
        original_apply = ExpertPool.apply
        calls = 0

        def counted_apply(pool, x, route, sign=0):
            nonlocal calls
            calls += 1
            return original_apply(pool, x, route, sign)

        for arm in ("uniform", "no_spatial"):
            model = self.build(arm).eval()
            with patch.object(ExpertPool, "apply", counted_apply), torch.inference_mode():
                model(make_input(batch=1))
            self.assertEqual(calls, 0)
            self.assertTrue(all(block.router is None and len(block.pools) == 0 for block in model.blocks))

    def test_grouped_and_no_pdm_have_their_declared_temporal_paths(self):
        grouped = self.build("grouped")
        self.assertEqual(len(grouped.temporal(make_input(batch=1))), 4)
        self.assertEqual(grouped.group_projection.in_features, 4 * grouped.heads[0].in_features)
        self.assertEqual(len(grouped.heads), 1)
        self.assertTrue(all(len(block.pools) == 1 for block in grouped.blocks))

        no_pdm = self.build("proposed_no_pdm")
        self.assertIsInstance(no_pdm.temporal, LinearScaleEncoder)
        self.assertEqual(len(no_pdm.temporal.encoders), 4)
        self.assertFalse(hasattr(no_pdm.temporal, "blocks"))
        self.assertEqual(no_pdm.positive_propagation_count, 45)

    def test_flops_are_nonzero_for_uniform_and_scale_with_batch_size(self):
        for arm in ("uniform", "proposed", "reversed"):
            model = self.build(arm).eval()
            bias = torch.zeros(3, 4)
            flops = []
            for batch in (1, 2):
                count, _, error = _count_flops(
                    model, make_input(batch=batch), bias, torch.device("cpu")
                )
                self.assertIsNone(error)
                self.assertIsNotNone(count)
                flops.append(count)
            self.assertGreater(flops[0], 0, arm)
            self.assertAlmostEqual(flops[1] / flops[0], 2.0, places=5)
            if arm == "proposed":
                proposed_flops = flops[0]
            elif arm == "reversed":
                self.assertEqual(flops[0], proposed_flops)

    def test_mage_features_profile_uses_raw_windows_and_restores_eight_channel_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_dir = root / "data"
            prepared = data_dir / "PEMS04"
            prepared.mkdir(parents=True)
            data = torch.cat((make_input(batch=1)[0], make_input(batch=1)[0]), dim=0)
            np.savez_compressed(prepared / "his.npz", data=data.numpy(), mean=0., std=1.)
            for split, origins in (("train", [95]), ("val", [108, 109]), ("test", [121])):
                np.save(prepared / f"idx_{split}.npy", np.array(origins, dtype=np.int64))
            args = worker_args("mage_features")
            args.data_dir = str(data_dir)
            with patch.object(train_eval, "make_window_features", wraps=train_eval.make_window_features) as features:
                _, _, raw = profile_models._load_profile_inputs(
                    args, "PEMS04", "mage_features", data_dir, 2
                )
                self.assertEqual(tuple(raw.shape), (2, 96, 7, 3))
                features.assert_not_called()
            model, *_ = train_eval.legacy._build_model("mage_features", args, 7, 8)
            checkpoint = root / "PEMS04_mage_features_s0.pt"
            torch.save({
                "state": model.state_dict(), "bias": torch.zeros(3, 4),
                "args": vars(args), "data_digest": train_eval.data_digest("PEMS04", str(data_dir)),
            }, checkpoint)
            row = profile_models.profile_one(
                "PEMS04", "mage_features", 0, checkpoint, data_dir,
                torch.device("cpu"), [1, 2], warmup=0, repeats=1,
            )
            self.assertEqual(row["status"], "completed")
            self.assertGreater(row["flops_counted_per_forward"], 0)
            self.assertEqual(set(row["feature_preparation_timings"]), {"1", "2"})

    def test_cuda_timers_record_on_the_requested_device_stream(self):
        device = torch.device("cuda:1")
        stream = object()
        for timing in ("model", "features"):
            with self.subTest(timing=timing):
                event = MagicMock()
                event.elapsed_time.return_value = 2.5
                with (
                    patch.object(profile_models, "_sync"),
                    patch.object(torch.cuda, "device") as context,
                    patch.object(torch.cuda, "current_stream", return_value=stream) as current_stream,
                    patch.object(torch.cuda, "Event", return_value=event),
                ):
                    if timing == "model":
                        result = profile_models._latencies(
                            MagicMock(), make_input(), torch.zeros(3, 4), 0, 2, device
                        )
                    else:
                        result = profile_models._feature_latencies(make_input(), 0, 2, device)
                context.assert_called_once_with(device)
                current_stream.assert_called_once_with(device)
                self.assertEqual(event.record.call_args_list, [call(stream)] * 4)
                self.assertEqual(event.synchronize.call_count, 2)
                self.assertEqual(result["median_ms"], 2.5)

    def test_report_outputs_join_profiles_and_mae(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile_dir = root / "profile"
            mae_dir = root / "mae"
            profile_dir.mkdir()
            mae_dir.mkdir()
            profile = {
                "status": "completed",
                "dataset": "PEMS04",
                "arm": "proposed",
                "seed": 0,
                "params_total": 100,
                "params_trainable": 100,
                "params_non_trainable": 0,
                "positive_graph_applications": 45,
                "negative_graph_applications": 12,
                "total_graph_applications": 57,
                "gflops_counted_per_forward": 1.25,
                "latency_ms_batch1_median": 2.0,
                "latency_ms_batch1_p95": 2.1,
                "latency_ms_batch64_median": 5.0,
                "latency_ms_batch64_p95": 5.2,
                "throughput_samples_per_s_batch64": 12800.0,
                "inference_peak_allocated_mib_batch1": 10.0,
                "inference_peak_allocated_mib_batch64": 50.0,
                "parameter_memory_mib_fp32": 0.001,
                "flops_count_method": "test counter",
                "gpu_model": "test GPU",
                "profile_batch_sizes": [1, 64],
            }
            (profile_dir / "profile_PEMS04_proposed_s0.json").write_text(
                json.dumps(profile), encoding="utf-8"
            )
            (mae_dir / "PEMS04_proposed_s0.json").write_text(
                json.dumps({
                    "completed": True,
                    "dataset": "PEMS04",
                    "arm": "proposed",
                    "seed": 0,
                    "test_mae": 18.0,
                }),
                encoding="utf-8",
            )
            outputs = build_report(mae_dir, profile_dir)
            self.assertTrue(all(path.is_file() for path in outputs))
            with (profile_dir / "model_complexity_summary.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["positive_graph_applications"], "45")
            self.assertEqual(row["test_mae_mean"], "18.0")
            summary = json.loads((profile_dir / "model_complexity_summary.json").read_text())
            self.assertEqual(summary["summaries"][0]["gflops_counted_per_forward"], 1.25)
            self.assertEqual(summary["profile_environment"][0]["gpu_model"], "test GPU")
            self.assertEqual(summary["profile_environment"][0]["seed"], 0)
            self.assertEqual(summary["profile_environment"][0]["batch_sizes"], [1, 64])
            markdown = (profile_dir / "model_complexity_table.md").read_text()
            self.assertIn("`profile_environment` in `model_complexity_summary.json`", markdown)
            self.assertNotIn("See `model_complexity_per_run.csv`", markdown)
            with self.assertRaises(FileExistsError):
                build_report(mae_dir, profile_dir)

    def test_followup_plan_is_exactly_40_runs(self):
        jobs = plan_jobs()
        self.assertEqual(len(jobs), 40)
        self.assertEqual(len({job.name for job in jobs}), 40)
        self.assertEqual(
            {job.arm for job in jobs},
            {"mage_features", "no_spatial", "grouped", "proposed_no_pdm"},
        )
        self.assertEqual(
            {job.dataset for job in jobs},
            {"PEMS04", "PEMS08"},
        )

    def test_followup_worker_preserves_recipe_provenance(self):
        args = run_eval.parser().parse_args(["--dry-run"])
        args.recipe_path = Path("/study/selected_recipe.json")
        args.recipe_digest = "recipe-sha"
        args.recipe_source_code_digest = "original-code-sha"
        args.lr, args.wd, args.dropout = 0.005, 0.0001, 0.3
        options = run_followup_eval._worker_args(
            args, plan_jobs()[0], Path("/study/eval_followup96/result.json")
        )
        worker_args = train_eval.make_parser().parse_args(options)
        self.assertEqual(worker_args.lr, 0.005)
        self.assertEqual(worker_args.recipe_source, str(args.recipe_path))
        self.assertEqual(worker_args.recipe_digest, "recipe-sha")
        self.assertEqual(worker_args.recipe_source_code_digest, "original-code-sha")


class HistoricalModelRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(directory.cleanup)
        cls.reference_dir = Path(directory.name)
        for name in REFERENCE_FILES:
            result = subprocess.run(
                ["git", "show", f"{REFERENCE_COMMIT}:MAGE_MS/{name}"],
                cwd=ROOT, capture_output=True, timeout=30,
            )
            if result.returncode:
                raise RuntimeError(
                    f"Historical regression requires local Git commit {REFERENCE_COMMIT}: "
                    f"{result.stderr.decode(errors='replace')}"
                )
            destination = cls.reference_dir / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(result.stdout)

    def compare_with_reference(self, cases):
        inputs = self.reference_dir / "cases.pt"
        outputs = self.reference_dir / "outputs.pt"
        torch.save(cases, inputs)
        result = subprocess.run(
            [sys.executable, "-I", "-c", REFERENCE_RUNNER,
             str(self.reference_dir), str(inputs), str(outputs)],
            cwd=self.reference_dir, capture_output=True, text=True, timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        expected = torch.load(outputs, map_location="cpu", weights_only=True)
        self.assertEqual(len(expected), len(cases))
        for case, predictions in zip(cases, expected):
            x = case["input"]
            with contextlib.redirect_stdout(io.StringIO()):
                model, *_ = train_eval.legacy._build_model(
                    case["arm"], SimpleNamespace(**case["args"]), x.shape[2], x.shape[3]
                )
            model.load_state_dict(case["state"], strict=True)
            model.eval()
            for batch, previous in zip((1, 2), predictions):
                with self.subTest(case=case["name"], batch=batch), torch.inference_mode():
                    actual = model(x[:batch], bias=case["bias"])
                    self.assertEqual(tuple(actual.shape), (batch, 12, x.shape[2], 1))
                    self.assertTrue(torch.isfinite(actual).all())
                    torch.testing.assert_close(actual, previous, atol=1e-6, rtol=1e-5)

    def test_six_original_arms_match_pinned_commit(self):
        cases = []
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(2026)
            x = make_input()
            for arm in profile_models.DEFAULT_ARMS:
                args = worker_args(arm)
                with contextlib.redirect_stdout(io.StringIO()):
                    model, *_ = train_eval.legacy._build_model(arm, args, 7, 3)
                cases.append({
                    "name": arm, "arm": arm, "args": vars(args), "input": x,
                    "state": model.state_dict(), "bias": torch.linspace(-0.2, 0.2, 12).reshape(3, 4),
                })
        self.compare_with_reference(cases)

    def test_saved_seed_zero_checkpoints_match_pinned_commit(self):
        results_dir = Path(os.environ.get("MAGE_REGRESSION_RESULTS_DIR", ROOT / "results/eval_proposal96"))
        data_dir = Path(os.environ.get("MAGE_REGRESSION_DATA_DIR", ROOT / "data_proposal96"))
        paths = [
            (dataset, arm, results_dir / f"{dataset}_{arm}_s0.pt")
            for dataset in ("PEMS04", "PEMS08") for arm in profile_models.DEFAULT_ARMS
        ]
        required = [path for _, _, path in paths] + [
            data_dir / dataset / name
            for dataset in ("PEMS04", "PEMS08")
            for name in ("his.npz", "idx_train.npy", "idx_val.npy", "idx_test.npy")
        ]
        if any(not path.is_file() for path in required):
            self.skipTest("Trained regression requires all twelve seed-0 checkpoints and prepared PEMS04/08 data")
        cases, windows = [], {}
        for dataset, arm, path in paths:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            args = SimpleNamespace(**checkpoint["args"])
            args.data_dir = str(data_dir)
            if dataset not in windows:
                _, _, windows[dataset] = profile_models._load_profile_inputs(
                    args, dataset, arm, data_dir, 2
                )
            cases.append({
                "name": f"{dataset}/{arm}/s0", "arm": arm, "args": vars(args),
                "input": windows[dataset], "state": checkpoint["state"], "bias": checkpoint["bias"],
            })
        self.compare_with_reference(cases)


if __name__ == "__main__":
    unittest.main()
