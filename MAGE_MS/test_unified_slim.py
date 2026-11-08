"""CPU structural, arithmetic, gradient, cost and prior-arm regression checks."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import types
import tempfile
import unittest

import torch

import proposal_model
from proposal_model import ProposalMAGE, SlimBlock, SLIM_ARMS, parameter_count
from profile_models import _count_flops
from profile_models import _load_profile_inputs
from run_unified_slim_eval import plan_jobs, make_parser
from test_proposal import make_args, make_input
import train_proposal
import train_eval
import report_unified_slim

BASE_COMMIT = "b5d4d818ff1e0673396ec1c74dc59cb32c2f7e55"


class UnifiedSlimTests(unittest.TestCase):
    def build(self, arm):
        return ProposalMAGE(make_args(), arm, pdm_width=8, pdm_depth=1, dropout=0)

    def test_shapes_registration_parameter_budget_and_zero_bias_update(self):
        a, b, baseline = (self.build(arm) for arm in (*SLIM_ARMS, "grouped"))
        self.assertEqual(parameter_count(a), parameter_count(b))
        self.assertLess(parameter_count(a), parameter_count(baseline))
        self.assertEqual(set(a.state_dict()), set(b.state_dict()))
        for model in (a, b):
            self.assertEqual(model.slim_coefficient_count, 3)
            self.assertEqual(model.graph_parameter_count, 0)
            self.assertEqual(model.propagation_count, 0)
            self.assertEqual(len(model.heads), 1)
            self.assertEqual(len(model.blocks), 3)
            self.assertEqual(len({id(block.theta) for block in model.blocks}), 3)
            self.assertTrue(all(block.theta.item() == 0.5 for block in model.blocks))
            self.assertFalse(any(name in key for key in model.state_dict()
                                 for name in ("E1", "E2", "router", "lambda_logits")))
            for batch in (1, 2):
                x = make_input(batch=batch)
                encoded = model.temporal(x)
                self.assertEqual(len(encoded), 4)
                self.assertTrue(all(h.shape == (batch, 7, 16) for h in encoded))
                model.train()
                prediction, counts = model(x)
                self.assertEqual(tuple(prediction.shape), (batch, 12, 7, 1))
                self.assertEqual(tuple(counts.shape), (3, 4))
                self.assertEqual(counts.count_nonzero().item(), 0)
                bias = torch.randn(3, 4)
                previous = bias.clone()
                train_proposal._bias_counts(bias, counts, types.SimpleNamespace(lb=1, lb_lr=0.001, arm=model.arm))
                torch.testing.assert_close(bias, previous)
                model.eval()
                with torch.inference_mode():
                    self.assertTrue(torch.isfinite(model(x)).all())

    def test_slim_formula_uses_node_mean_tanh_and_ordered_residuals(self):
        torch.manual_seed(31)
        block = SlimBlock(5, dropout=0)
        hidden = torch.randn(2, 7, 5)
        normalized = block.norm1(hidden)
        residual = hidden + torch.tanh(block.theta) * normalized.mean(dim=1, keepdim=True)
        expected = residual + block.ffn(block.norm2(residual))
        torch.testing.assert_close(block(hidden), expected)
        wrong = hidden + block.theta * normalized.mean(dim=2, keepdim=True)
        self.assertFalse(torch.allclose(residual, wrong))

    def test_forward_order_and_shared_instances(self):
        for arm in SLIM_ARMS:
            model = self.build(arm).eval()
            x = make_input()
            events, first_inputs, last_outputs, fusion_inputs, fusion_outputs = [], [], [], [], []
            handles = [model.group_projection.register_forward_hook(
                lambda module, args, out: (events.append("fusion"), fusion_inputs.append(args[0].clone()), fusion_outputs.append(out.clone())) and None)]
            for index, block in enumerate(model.blocks):
                def hook(module, args, out, index=index):
                    events.append(index)
                    if index == 0:
                        first_inputs.append(args[0].clone())
                    if index == 2:
                        last_outputs.append(out.clone())
                handles.append(block.register_forward_hook(hook))
            try:
                with torch.inference_mode():
                    encoded, prompt = model.temporal(x), model.prompt(x)
                    model(x)
            finally:
                for handle in handles:
                    handle.remove()
            if arm == "grouped_slim":
                self.assertEqual(events, ["fusion", 0, 1, 2])
                torch.testing.assert_close(fusion_inputs[0], torch.cat(encoded, dim=-1))
                torch.testing.assert_close(first_inputs[0], fusion_outputs[0] + prompt)
            else:
                self.assertEqual(events, [0, 1, 2] * 4 + ["fusion"])
                self.assertEqual(len(first_inputs), 4)
                for actual, value in zip(first_inputs, encoded):
                    torch.testing.assert_close(actual, value + prompt)
                torch.testing.assert_close(fusion_inputs[0], torch.cat(last_outputs, dim=-1))
                self.assertEqual(len(list(model.blocks.parameters())), len({id(p) for p in model.blocks.parameters()}))

    def test_gradients_reach_theta_pdm_fusion_and_ffn(self):
        for arm in SLIM_ARMS:
            model = self.build(arm).train()
            prediction, _ = model(make_input())
            prediction.square().mean().backward()
            parameters = [model.temporal.embedding.weight, model.group_projection.weight]
            parameters += [p for block in model.blocks for p in (block.theta, *block.ffn.parameters())]
            for parameter in parameters:
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_equal_weights_have_different_predictions_and_flops(self):
        a, b = (self.build(arm).eval() for arm in SLIM_ARMS)
        b.load_state_dict(a.state_dict(), strict=True)
        x, bias = make_input(), torch.zeros(3, 4)
        with torch.inference_mode():
            self.assertFalse(torch.allclose(a(x), b(x)))
        ca, _, ea = _count_flops(a, x, bias, torch.device("cpu"))
        cb, _, eb = _count_flops(b, x, bias, torch.device("cpu"))
        self.assertIsNone(ea)
        self.assertIsNone(eb)
        self.assertGreater(cb, ca)

    def test_all_preexisting_arms_match_base_commit(self):
        source = subprocess.check_output(
            ["git", "show", f"{BASE_COMMIT}:MAGE_MS/proposal_model.py"], cwd=Path(__file__).resolve().parent
        ).decode()
        reference = types.ModuleType("unified_slim_base_proposal")
        reference.__file__ = proposal_model.__file__
        exec(compile(source, reference.__file__, "exec"), reference.__dict__)
        for arm in reference.ARMS:
            with self.subTest(arm=arm), contextlib.redirect_stdout(io.StringIO()):
                current = self.build(arm).eval()
                original = reference.ProposalMAGE(make_args(), arm, pdm_width=8, pdm_depth=1, dropout=0).eval()
                original.load_state_dict(current.state_dict(), strict=True)
                for batch in (1, 2):
                    x, bias = make_input(batch=batch), torch.randn(3, 4)
                    with torch.inference_mode():
                        torch.testing.assert_close(current(x, bias), original(x, bias), atol=1e-6, rtol=1e-5)

    def test_trained_grouped_outputs_match_base_commit(self):
        root = Path(__file__).resolve().parent
        results = Path(os.environ.get("MAGE_GROUPED_REGRESSION_RESULTS_DIR", root / "results/eval_followup96"))
        data_dir = Path(os.environ.get("MAGE_REGRESSION_DATA_DIR", root / "data_proposal96"))
        paths = [results / f"{dataset}_grouped_s0.pt" for dataset in ("PEMS04", "PEMS08")]
        if any(not path.is_file() for path in paths):
            self.skipTest("Set MAGE_GROUPED_REGRESSION_RESULTS_DIR to the two trained Grouped seed-0 checkpoints")
        source = subprocess.check_output(["git", "show", f"{BASE_COMMIT}:MAGE_MS/proposal_model.py"], cwd=root).decode()
        reference = types.ModuleType("trained_grouped_base_proposal")
        reference.__file__ = proposal_model.__file__
        exec(compile(source, reference.__file__, "exec"), reference.__dict__)
        for dataset, path in zip(("PEMS04", "PEMS08"), paths):
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            args = types.SimpleNamespace(**checkpoint["args"])
            args.data_dir = str(data_dir)
            data, _, x = _load_profile_inputs(args, dataset, "grouped", data_dir, 2)
            with contextlib.redirect_stdout(io.StringIO()):
                current, *_ = train_proposal._build_model("grouped", args, data.shape[1], 3)
                original = reference.ProposalMAGE(
                    make_args(nodes=data.shape[1], model_dim=args.model_dim, experts=args.experts,
                              topk=args.topk, graph_dim=args.graph_dim), "grouped",
                    pdm_width=args.pdm_width, pdm_depth=args.pdm_depth,
                    moving_average=args.moving_average, dropout=args.dropout,
                )
            current.load_state_dict(checkpoint["state"], strict=True)
            original.load_state_dict(checkpoint["state"], strict=True)
            current.eval()
            original.eval()
            for batch in (1, 2):
                with self.subTest(dataset=dataset, batch=batch), torch.inference_mode():
                    torch.testing.assert_close(current(x[:batch], checkpoint["bias"]),
                                               original(x[:batch], checkpoint["bias"]), atol=1e-6, rtol=1e-5)

    def test_pilot_and_formal_plans_and_dry_run_default(self):
        self.assertEqual(len(plan_jobs(seeds=[0, 1])), 16)
        self.assertEqual(len(plan_jobs()), 40)
        self.assertFalse(make_parser().parse_args([]).execute)
        self.assertEqual({job.arm for job in plan_jobs()}, {"grouped", *SLIM_ARMS, "grouped_uniform"})


class UnifiedSlimReportTests(unittest.TestCase):
    def test_validation_report_uses_only_validation_metrics_and_matching_costs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, profiles, output = (root / name for name in ("runs", "profiles", "report"))
            source.mkdir()
            profiles.mkdir()
            for arm, values in (("grouped", (12., 13.)), ("grouped_slim", (10., 12.)), ("slim_then_grouped", (11., 11.))):
                for seed, value in enumerate(values):
                    args = train_eval.make_parser().parse_args(["--out", "unused", "--arm", arm, "--seed", str(seed), "--validation-only"])
                    row = {
                        "completed": True, "dataset": "PEMS04", "arm": arm, "seed": seed,
                        "protocol": "96to12_v1_validation_only", "best_val_mae": value,
                        "test_mae": -999., "params": 100 if arm == "grouped" else 50,
                        "code_digest": "source", "data_digest": "data", "signature": f"{arm}-{seed}",
                        "config": {k: getattr(args, k) for k in train_eval.CONFIG_KEYS},
                    }
                    (source / f"PEMS04_{arm}_s{seed}.json").write_text(json.dumps(row))
            (profiles / "profile_PEMS04_grouped_s0.json").write_text(json.dumps({
                "status": "completed", "dataset": "PEMS04", "arm": "grouped", "seed": 0,
                "data_digest": "data", "checkpoint_code_digest": "source", "profile_code_digest": "source",
                "checkpoint_signature": "grouped-0", "gflops_counted_per_forward": 1.0,
            }))
            records, missing = report_unified_slim.read_runs(source, ["PEMS04"], report_unified_slim.ARMS, [0, 1], True)
            costs = report_unified_slim.read_profiles(profiles, records)
            rows, differences = report_unified_slim.summarize(records, costs, ["PEMS04"], report_unified_slim.ARMS, [0, 1], True)
            by_arm = {row["arm"]: row for row in rows}
            self.assertEqual(by_arm["grouped_slim"]["mae_mean"], 11.)
            self.assertEqual(by_arm["grouped_slim"]["paired_wins_vs_grouped"], 2)
            self.assertEqual(by_arm["grouped_slim"]["parameter_reduction_pct"], 50.)
            self.assertIsNone(by_arm["grouped_slim"]["rmse_mean"])
            self.assertEqual(by_arm["grouped_uniform"]["completed_seeds"], 0)
            self.assertEqual(len(missing), 2)
            self.assertEqual(differences[0]["A_minus_B_mae_mean"], 0.)
            self.assertEqual(differences[0]["A_wins"], 1)
            report_unified_slim.build_report(source, output, profiles, ["PEMS04"], seeds=[0, 1])
            with self.assertRaises(FileExistsError):
                report_unified_slim.build_report(source, output, profiles, ["PEMS04"], seeds=[0, 1])
            profile_path = profiles / "profile_PEMS04_grouped_s0.json"
            profile = json.loads(profile_path.read_text())
            profile["data_digest"] = "wrong-split"
            profile_path.write_text(json.dumps(profile))
            with self.assertRaisesRegex(ValueError, "data mismatch"):
                report_unified_slim.read_profiles(profiles, records)

    def test_report_rejects_test_protocol_in_validation_mode(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            (source / "PEMS04_grouped_s0.json").write_text(json.dumps({
                "completed": True, "dataset": "PEMS04", "arm": "grouped", "seed": 0,
                "protocol": "96to12_v1_window_local_features",
            }))
            with self.assertRaisesRegex(ValueError, "Wrong evaluation protocol"):
                report_unified_slim.read_runs(source, ["PEMS04"], ["grouped"], [0], True)


if __name__ == "__main__":
    unittest.main()
