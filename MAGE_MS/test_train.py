import unittest
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'official_repro'))
from train import masked_metrics, optional_metrics


class MaskedMetricsTests(unittest.TestCase):
    def test_metrics_use_only_finite_positive_targets(self):
        pred = torch.tensor([2.0, 4.0, 100.0, 8.0])
        target = torch.tensor([1.0, 2.0, 0.0, float('nan')])
        mae, rmse, mape = masked_metrics(pred, target)

        self.assertAlmostEqual(mae.item(), 1.5)
        self.assertAlmostEqual(rmse.item(), (2.5 ** 0.5))
        self.assertAlmostEqual(mape.item(), 100.0)

    def test_empty_valid_target_set_raises_or_reports_unmeasurable(self):
        pred = torch.tensor([1.0, 2.0])
        target = torch.tensor([0.0, float('nan')])

        with self.assertRaisesRegex(ValueError, 'No valid targets'):
            masked_metrics(pred, target)
        self.assertEqual(optional_metrics(pred, target), [None, None, None])


if __name__ == '__main__':
    unittest.main()