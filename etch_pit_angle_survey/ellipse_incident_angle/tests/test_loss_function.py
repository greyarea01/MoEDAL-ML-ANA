from __future__ import annotations

import sys
import unittest
from pathlib import Path


PIPELINE_DIR = Path(__file__).resolve().parents[1]
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from loss_function import (  # noqa: E402
    LossConfig,
    PitLossInput,
    SurfaceLossInput,
    evaluate_loss,
    evaluate_surface_stability,
)


class LossFunctionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = LossConfig(
            perturbation_offsets_xy=((-1, 0), (1, 0), (0, -1), (0, 1)),
            minimum_passing_perturbations=3,
            stability_failure_penalty_deg=20.0,
            tail_percentile_weights=(1.0, 1.0, 0.5),
            allowed_low_angle_fraction=0.25,
        )

    def test_majority_failure_is_robust_qc_failure_without_stability_cost(self) -> None:
        result = evaluate_surface_stability(
            SurfaceLossInput(
                nominal_qc_passed=True,
                nominal_angle_deg=30.0,
                perturbed_angles_deg=(31.0, None, None, None),
            ),
            self.config,
        )

        self.assertEqual(result.perturbation_pass_count, 1)
        self.assertEqual(result.perturbation_fail_count, 3)
        self.assertFalse(result.robust_qc_passed)
        self.assertIsNone(result.mean_stability_error_deg)

    def test_failed_perturbations_are_penalized_after_majority_passes(self) -> None:
        result = evaluate_surface_stability(
            SurfaceLossInput(
                nominal_qc_passed=True,
                nominal_angle_deg=10.0,
                perturbed_angles_deg=(11.0, 9.0, 10.0, None),
            ),
            self.config,
        )

        self.assertTrue(result.robust_qc_passed)
        self.assertAlmostEqual(result.mean_stability_error_deg, 5.5)

    def test_all_loss_terms_use_only_robust_paired_passes(self) -> None:
        pits = [
            PitLossInput(
                clean_id="A",
                c1=SurfaceLossInput(True, 10.0, (11.0, 9.0, 10.0, None)),
                c2=SurfaceLossInput(True, 14.0, (15.0, 13.0, 14.0, 14.0)),
            ),
            PitLossInput(
                clean_id="B",
                c1=SurfaceLossInput(True, 2.0, (2.0, 2.0, 2.0, 2.0)),
                c2=SurfaceLossInput(True, 4.0, (4.0, 4.0, 4.0, 4.0)),
            ),
            PitLossInput(
                clean_id="C",
                c1=SurfaceLossInput(True, 30.0, (31.0, None, None, None)),
                c2=SurfaceLossInput(True, 31.0, (31.0, 31.0, 31.0, 31.0)),
            ),
        ]

        summary, per_pit = evaluate_loss(pits, self.config)
        terms = summary["raw_loss_terms"]

        self.assertEqual(summary["nominal_paired_pass_count"], 3)
        self.assertEqual(summary["robust_paired_pass_count"], 2)
        self.assertAlmostEqual(terms["coverage_penalty"], 1.0 / 3.0)
        self.assertAlmostEqual(terms["central_absolute_error_deg"], 3.0)
        self.assertAlmostEqual(terms["tail_error_deg"], 3.876)
        self.assertAlmostEqual(terms["signed_bias_deg"], 3.0)
        self.assertAlmostEqual(terms["mean_stability_error_deg"], 1.5)
        self.assertAlmostEqual(terms["low_angle_penalty"], 0.25)
        self.assertFalse(per_pit[2].robust_paired_passed)
        self.assertIsNone(per_pit[2].mean_stability_error_deg)


if __name__ == "__main__":
    unittest.main()
