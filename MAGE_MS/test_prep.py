import unittest

from prep import split_indices


class SplitIndicesTests(unittest.TestCase):
    def test_target_windows_do_not_cross_or_overlap_split_boundaries(self):
        T, max_seq, horizon = 1000, 48, 12
        train, val, test = split_indices(T, max_seq, horizon)
        b1, b2 = int(T * 0.6), int(T * 0.8)

        self.assertLess(train[-1] + horizon, b1)
        self.assertGreaterEqual(val[0] + 1, b1)
        self.assertLess(val[-1] + horizon, b2)
        self.assertGreaterEqual(test[0] + 1, b2)
        self.assertLess(test[-1] + horizon, T)

        target_sets = [
            {target for index in split for target in range(index + 1, index + horizon + 1)}
            for split in (train, val, test)
        ]
        self.assertFalse(target_sets[0] & target_sets[1])
        self.assertFalse(target_sets[1] & target_sets[2])
        self.assertLess(val[0] - max_seq + 1, b1)


if __name__ == '__main__':
    unittest.main()