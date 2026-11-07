import json
import os
import tempfile
import unittest

from run_proposal import experiment_jobs, paired_mae_results


class ProposalRunnerTests(unittest.TestCase):
    def test_pems08_runs_include_mage_reference_for_each_seed(self):
        jobs = experiment_jobs()
        pems08_jobs = {
            (arm, seed) for dataset, arm, seed in jobs if dataset == "PEMS08"
        }
        self.assertEqual(len(jobs), 46)
        for seed in range(3):
            self.assertIn(("mage", seed), pems08_jobs)
            self.assertIn(("proposed", seed), pems08_jobs)

    def test_paired_mae_comparison_uses_matching_dataset_and_seed(self):
        with tempfile.TemporaryDirectory() as results_dir:
            for arm, mae in (("mage", 13.5), ("proposed", 13.2)):
                path = os.path.join(results_dir, f"PEMS08_{arm}_s0.json")
                with open(path, "w", encoding="utf-8") as result_file:
                    json.dump({"test_mae": mae}, result_file)

            comparisons = paired_mae_results(results_dir)

        self.assertEqual(comparisons["PEMS04"], [])
        paired = comparisons["PEMS08"]
        self.assertEqual(len(paired), 1)
        self.assertEqual(paired[0]["seed"], 0)
        self.assertEqual(paired[0]["mage_mae"], 13.5)
        self.assertEqual(paired[0]["proposed_mae"], 13.2)
        self.assertAlmostEqual(paired[0]["delta_mae"], -0.3)


if __name__ == "__main__":
    unittest.main()
