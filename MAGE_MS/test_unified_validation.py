"""Supplemental validation metrics must never evaluate the test split."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import torch

import eval_unified_slim_validation as evaluator
import report_unified_slim as report


class SupplementalValidationTests(unittest.TestCase):
    def result(self):
        return {
            "completed": True, "protocol": "96to12_v1_validation_only",
            "config": {"validation_only": True}, "dataset": "PEMS04",
            "arm": "grouped_slim", "seed": 0, "signature": "signature",
            "code_digest": "code", "data_digest": "data", "best_val_mae": 10., "params": 100,
        }

    def test_evaluator_calls_validation_only_and_preserves_training_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "PEMS04_grouped_slim_s0.json"
            path.write_text(json.dumps(self.result()))
            before = path.read_bytes()
            checkpoint = {"signature": "signature", "code_digest": "code", "data_digest": "data",
                          "state": {}, "bias": torch.zeros(3, 16)}
            data = torch.zeros(101, 7, 3)
            prediction, target = torch.zeros(1, 12, 7), torch.ones(1, 12, 7)
            with (
                patch.object(torch, "load", return_value=checkpoint),
                patch.object(evaluator.train_eval, "code_digest", return_value="code"),
                patch.object(evaluator.train_eval, "data_digest", return_value="data"),
                patch.object(evaluator.profile_models, "_checkpoint_args", return_value=SimpleNamespace(tf32=0)),
                patch.object(evaluator.train_eval.legacy, "_load_data", return_value=(data, torch.zeros(101, 7), 0., 1., {"val": [95]})),
                patch.object(evaluator.train_eval.legacy, "_build_model", return_value=(MagicMock(), 0, 0, 0)),
                patch.object(evaluator.train_eval, "_evaluate", return_value=(prediction, target)) as evaluate,
                patch.object(evaluator.train_eval, "_metrics", return_value=[10., 11., 12.]),
            ):
                row = evaluator.evaluate_one(path, Path(temporary), torch.device("cpu"))
            self.assertEqual(evaluate.call_count, 1)
            self.assertEqual(evaluate.call_args.args[-1], "val")
            self.assertEqual(row["val_rmse"], 11.)
            self.assertEqual(len(row["per_horizon"]), 12)
            self.assertNotIn("test_mae", row)
            self.assertEqual(path.read_bytes(), before)

    def test_report_accepts_only_matching_validation_metric_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            key = ("PEMS04", "grouped_slim", 0)
            records = {key: self.result()}
            row = {**self.result(), "protocol": "96to12_v1_validation_best_checkpoint_metrics",
                   "val_mae": 10., "val_rmse": 11., "val_mape": 12., "per_horizon": [[10., 11., 12.]] * 12}
            path = folder / "validation_PEMS04_grouped_slim_s0.json"
            path.write_text(json.dumps(row))
            metrics = report.read_validation_metrics(folder, records)
            rows, _ = report.summarize(records, {}, ["PEMS04"], ["grouped_slim"], [0], True, metrics)
            self.assertEqual(rows[0]["rmse_mean"], 11.)
            self.assertEqual(rows[0]["mae_15m"], 10.)
            row["signature"] = "different-checkpoint"
            path.write_text(json.dumps(row))
            with self.assertRaisesRegex(ValueError, "provenance mismatch"):
                report.read_validation_metrics(folder, records)


if __name__ == "__main__":
    unittest.main()
