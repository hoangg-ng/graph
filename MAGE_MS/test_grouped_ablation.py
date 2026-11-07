"""CPU checks for the Grouped ablation arms."""
import unittest
from unittest.mock import patch

import torch

from proposal_model import ExpertPool, LinearScaleEncoder, ProposalMAGE
from test_proposal import make_args, make_input


COUNTS = {
    "grouped": (3, 3),
    "grouped_no_pdm": (3, 3),
    "grouped_no_spatial": (0, 0),
    "grouped_uniform": (0, 0),
}


class GroupedAblationTests(unittest.TestCase):
    def test_output_and_graph_counts(self):
        x = make_input(batch=2)
        for arm, (positive, negative) in COUNTS.items():
            with self.subTest(arm=arm):
                torch.manual_seed(17)
                model = ProposalMAGE(
                    make_args(), arm, pdm_width=8, pdm_depth=1, dropout=0
                ).eval()
                with torch.inference_mode():
                    output = model(x)
                self.assertEqual(tuple(output.shape), (2, 12, 7, 1))
                self.assertTrue(torch.isfinite(output).all())
                self.assertEqual(len(model.heads), 1)
                self.assertIsNotNone(model.group_projection)
                self.assertEqual(len(model.temporal(x)), 4)
                self.assertEqual(model.positive_propagation_count, positive)
                self.assertEqual(model.negative_propagation_count, negative)
                self.assertEqual(model.propagation_count, positive + negative)

    def test_pdm_removal_and_graph_removal(self):
        no_pdm = ProposalMAGE(make_args(), "grouped_no_pdm", pdm_width=8, pdm_depth=1)
        self.assertIsInstance(no_pdm.temporal, LinearScaleEncoder)
        self.assertEqual(len(no_pdm.temporal.encoders), 4)
        self.assertEqual(len(no_pdm.blocks[0].pools), 1)

        for arm in ("grouped_no_spatial", "grouped_uniform"):
            with self.subTest(arm=arm):
                model = ProposalMAGE(make_args(), arm, pdm_width=8, pdm_depth=1).eval()
                self.assertTrue(all(block.router is None for block in model.blocks))
                self.assertTrue(all(len(block.pools) == 0 for block in model.blocks))
                self.assertEqual(model.graph_parameter_count, 0)
                with patch.object(ExpertPool, "apply", side_effect=AssertionError("graph called")):
                    with torch.inference_mode():
                        model(make_input(batch=1))


if __name__ == "__main__":
    unittest.main()
