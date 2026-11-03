import json
import os
import tempfile
import unittest

import numpy as np

from prep_proposal import (
    build_mage_feature_data,
    calendar_channels,
    prepare_dataset,
    split_indices,
)


class ProposalPrepTests(unittest.TestCase):
    def test_flow_only_preparation_metadata_and_disjoint_targets(self):
        length, nodes = 600, 3
        raw = np.zeros((length, nodes, 3), dtype=np.float32)
        raw[..., 0] = np.arange(length, dtype=np.float32)[:, None] + 1
        raw[..., 1] = 50
        raw[..., 2] = 30

        with tempfile.TemporaryDirectory() as temp_dir:
            source = os.path.join(temp_dir, "PEMS04.npz")
            output = os.path.join(temp_dir, "prepared")
            np.savez(source, data=raw)
            metadata = prepare_dataset("04", source, output, seq_len=96, horizon=12)

            with np.load(os.path.join(output, "his.npz")) as archive:
                prepared = archive["data"]
                mean, std = float(archive["mean"]), float(archive["std"])
            self.assertEqual(prepared.shape, (length, nodes, 3))
            self.assertAlmostEqual(mean, float(raw[:360, :, 0].mean()), places=5)
            self.assertAlmostEqual(std, float(raw[:360, :, 0].std()), places=5)
            self.assertTrue(np.allclose(prepared[..., 0], (raw[..., 0] - mean) / std))
            self.assertTrue(np.all((prepared[..., 1:] >= 0) & (prepared[..., 1:] < 1)))
            self.assertEqual(metadata["source_shape"], [length, nodes, 3])
            self.assertEqual(len(metadata["source_sha256"]), 64)
            with open(os.path.join(output, "metadata.json"), encoding="utf-8") as source_file:
                self.assertEqual(json.load(source_file)["origin_counts"], metadata["origin_counts"])

            splits = [
                np.load(os.path.join(output, f"idx_{name}.npy"))
                for name in ("train", "val", "test")
            ]
            target_sets = [
                {
                    target
                    for origin in split
                    for target in range(int(origin) + 1, int(origin) + 13)
                }
                for split in splits
            ]
            self.assertFalse(target_sets[0] & target_sets[1])
            self.assertFalse(target_sets[1] & target_sets[2])

    def test_reference_split_uses_96_to_12_origins(self):
        train, val, test = split_indices(1000)
        boundary1, boundary2 = 600, 800
        self.assertLess(train[-1] + 12, boundary1)
        self.assertGreaterEqual(val[0] + 1, boundary1)
        self.assertLess(val[-1] + 12, boundary2)
        self.assertGreaterEqual(test[0] + 1, boundary2)
        self.assertLess(test[-1] + 12, 1000)

    def test_standard_dataset_origin_counts(self):
        counts04 = [len(indices) for indices in split_indices(16992)]
        counts08 = [len(indices) for indices in split_indices(17856)]
        self.assertEqual(counts04, [10088, 3387, 3388])
        self.assertEqual(counts08, [10606, 3560, 3561])

    def test_causal_engineered_features_keep_calendar_channels(self):
        length, nodes = 20, 2
        tod, dow = calendar_channels(length, "2018-01-01")
        flow = np.arange(length * nodes, dtype=np.float32).reshape(length, nodes)
        data = np.stack(
            (
                flow,
                np.broadcast_to(tod[:, None], (length, nodes)),
                np.broadcast_to(dow[:, None], (length, nodes)),
            ),
            axis=-1,
        )
        augmented = build_mage_feature_data(data)
        self.assertEqual(augmented.shape, (length, nodes, 8))
        self.assertTrue(np.allclose(augmented[..., 0], flow))
        rolling3 = np.stack(
            [flow[max(0, index - 2) : index + 1].mean(axis=0) for index in range(length)]
        )
        rolling12 = np.stack(
            [flow[max(0, index - 11) : index + 1].mean(axis=0) for index in range(length)]
        )
        self.assertTrue(np.allclose(augmented[..., 1], rolling3))
        self.assertTrue(np.allclose(augmented[..., 2], rolling12))
        self.assertTrue(np.all(augmented[0, :, 3] == 0))
        self.assertTrue(np.allclose(augmented[1:, :, 3], flow[1:] - flow[:-1]))
        self.assertTrue(np.allclose(augmented[..., 6], data[..., 1]))
        self.assertTrue(np.allclose(augmented[..., 7], data[..., 2]))

        changed_future = data.copy()
        changed_future[10:] += 1000
        augmented_future = build_mage_feature_data(changed_future)
        self.assertTrue(np.allclose(augmented[:10], augmented_future[:10]))


if __name__ == "__main__":
    unittest.main()
