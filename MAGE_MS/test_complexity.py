"""CPU checks for model complexity instrumentation and follow-up arms."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

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


if __name__ == "__main__":
    unittest.main()
