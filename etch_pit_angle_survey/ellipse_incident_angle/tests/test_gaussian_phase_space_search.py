from __future__ import annotations

import sys
import unittest
from pathlib import Path


PIPELINE_DIR = Path(__file__).resolve().parents[1]
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from search_gaussian_phase_space import (  # noqa: E402
    GaussianPoint,
    SearchConfig,
    adaptive_search,
    neighborhood,
)


class GaussianPhaseSpaceSearchTests(unittest.TestCase):
    def test_one_pixel_proposals_skip_even_opencv_kernels(self) -> None:
        points, skipped = neighborhood(
            GaussianPoint(5, 1.1),
            SearchConfig(maximum_iterations=2),
        )

        self.assertEqual({point.kernel_px for point in points}, {3, 5, 7})
        self.assertEqual({point.sigma_px for point in points}, {0.9, 1.1, 1.3})
        self.assertEqual(skipped, [4, 6])

    def test_search_changes_direction_and_stops_at_local_minimum(self) -> None:
        def evaluator(point: GaussianPoint, iteration: int) -> dict[str, object]:
            loss = (point.kernel_px - 9) ** 2 + (point.sigma_px - 1.7) ** 2
            return {"provisional_weighted_total_loss": loss, "iteration": iteration}

        evaluated, path, _, stop_reason = adaptive_search(
            GaussianPoint(5, 1.1),
            SearchConfig(maximum_iterations=10),
            evaluator,
        )

        self.assertEqual(path[-1], GaussianPoint(9, 1.7))
        self.assertEqual(stop_reason, "local_minimum_reached")
        self.assertIn(GaussianPoint(7, 1.3), path)
        self.assertGreater(len(evaluated), len(path))


if __name__ == "__main__":
    unittest.main()

