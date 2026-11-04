import unittest

import numpy as np

from modules.dataset_tools.coordinates import Overlap


class MatchingToleranceTests(unittest.TestCase):
    def test_exactly_twelve_pixels_is_a_match(self):
        overlap = Overlap(
            np.asarray([[0.0, 0.0]]),
            np.asarray([[12.0, 0.0]]),
            tolerance=12,
        )

        self.assertEqual(overlap.A_in_B()[0].tolist(), [0])
        self.assertEqual(overlap.B_in_A()[0].tolist(), [0])

    def test_more_than_twelve_pixels_is_not_a_match(self):
        overlap = Overlap(
            np.asarray([[0.0, 0.0]]),
            np.asarray([[12.000001, 0.0]]),
            tolerance=12,
        )

        self.assertEqual(overlap.A_not_B()[0].tolist(), [0])
        self.assertEqual(overlap.B_not_A()[0].tolist(), [0])


if __name__ == "__main__":
    unittest.main()
