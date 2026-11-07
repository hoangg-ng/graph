"""Fast CPU-only checks for evaluation utilities (requires project source imports)."""
import json
from pathlib import Path
import tempfile
import unittest

import torch

from train_eval import make_window_features
from run_eval import planned_jobs
from report_eval import ordering_decision, paired
from power_diag import distances


class EvaluationToolsTests(unittest.TestCase):
    def test_full_plan_has_43_unique_runs(self):
        runs = planned_jobs("full")
        self.assertEqual(len(runs), 43)
        self.assertEqual(len({run.name for run in runs}), 43)
        self.assertEqual(len([r for r in runs if r.dataset == "PEMS04"]), 31)
        self.assertEqual(len([r for r in runs if r.dataset == "PEMS08"]), 12)
        self.assertEqual(len(planned_jobs("core")), 37)
        self.assertEqual(len(planned_jobs("pilot")), 3)

    def test_engineered_features_are_window_causal_and_calendar_last(self):
        x = torch.zeros(2, 96, 3, 3)
        x[:, :, :, 0] = torch.arange(96, dtype=torch.float32).view(1, 96, 1)
        x[:, :, :, 1] = torch.arange(96).view(1, 96, 1) / 288
        x[:, :, :, 2] = 2 / 7
        x.requires_grad_(True)
        f = make_window_features(x)
        self.assertEqual(tuple(f.shape), (2, 96, 3, 8))
        self.assertTrue(torch.allclose(f[0, 0, 0, :4], torch.tensor([0., 0., 0., 0.])))
        self.assertAlmostEqual(f[0, 1, 0, 1].item(), 0.5)
        self.assertAlmostEqual(f[0, 11, 0, 2].item(), 5.5)
        self.assertAlmostEqual(f[0, 12, 0, 2].item(), 6.5)
        self.assertTrue(torch.allclose(f[..., 6:], x[..., 1:]))
        # Changing inputs after step 5 cannot change any features through step 5.
        altered = x.detach().clone()
        altered[:, 6:, :, 0] += 10000
        f2 = make_window_features(altered)
        torch.testing.assert_close(f[:, :6], f2[:, :6])
        f.sum().backward()
        self.assertIsNotNone(x.grad)

    def test_power_distance_identical_for_uniform_averaging(self):
        p = torch.full((1, 5, 5), 0.2)
        result = distances(p)
        for k, v in result.items():
            self.assertAlmostEqual(v, 0.0, places=5, msg=k)

    def test_paired_rule_requires_both_4of5_and_1percent(self):
        def build(candidate, shared=100.0, reversed_mae=100.0):
            results = {}
            for s, m in enumerate(candidate):
                for arm, mae in (("proposed", m), ("shared", shared), ("reversed", reversed_mae)):
                    results[("PEMS04", arm, s)] = {"test_mae": float(mae)}
            return results

        yes = build([98., 98., 98., 98., 101.])
        decision, *_ = ordering_decision(yes, [])
        self.assertEqual(decision, "SUPPORTED")
        insufficient_percent = build([99.8, 99.8, 99.8, 99.8, 100.4])
        self.assertEqual(ordering_decision(insufficient_percent, [])[0],
                         "NOT_SUPPORTED_BY_PRESPECIFIED_RULE")
        weak_seed_count = build([97., 97., 97., 101., 101.])
        self.assertEqual(ordering_decision(weak_seed_count, [])[0],
                         "NOT_SUPPORTED_BY_PRESPECIFIED_RULE")
        self.assertEqual(ordering_decision(yes, ["mixed data"])[0], "INVALID_COMPARISON")
        self.assertEqual(ordering_decision(build([98.]*4), [])[0], "INCOMPLETE")


if __name__ == "__main__":
    unittest.main()
