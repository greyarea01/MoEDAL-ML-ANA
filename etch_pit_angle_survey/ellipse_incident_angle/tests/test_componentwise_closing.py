"""Regression tests for component-wise morphological closing."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np


TEST_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = TEST_DIR.parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from ellipse_fit import (  # noqa: E402
    FitConfig,
    _close_components_independently,
    fit_etch_pit,
)
from run_pipeline import (  # noqa: E402
    C2_IMAGE_PATH,
    DEFAULT_INPUT_PATH,
    FIRST_IMAGE_NUMBER,
    load_manifest,
)


class ComponentwiseClosingTests(unittest.TestCase):
    def test_closing_does_not_join_disconnected_components(self) -> None:
        mask = np.zeros((25, 25), dtype=np.uint8)
        mask[7:18, 3:10] = 255
        mask[7:18, 12:19] = 255
        config = FitConfig(closing_kernel_px=3)

        globally_closed = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            np.ones((3, 3), dtype=np.uint8),
        )
        global_contours, _ = cv2.findContours(
            globally_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        _, component_contours = _close_components_independently(
            mask, np.ones(mask.shape, dtype=bool), config
        )

        self.assertEqual(len(global_contours), 1)
        self.assertEqual(len(component_contours), 2)

    def test_c0074_c2_scratch_remains_separate(self) -> None:
        row = next(
            row
            for row in load_manifest(DEFAULT_INPUT_PATH)
            if row["clean_id"] == "C0074"
        )
        images = np.load(C2_IMAGE_PATH, mmap_mode="r")
        source = images[int(row["c2_image"]) - FIRST_IMAGE_NUMBER]
        result = fit_etch_pit(
            source,
            float(row["c2_local_x"]),
            float(row["c2_local_y"]),
            FitConfig(gaussian_kernel_px=3, gaussian_sigma_px=0.9),
        )

        self.assertTrue(result.qc_passed)
        self.assertAlmostEqual(result.incident_angle_deg, 45.33721710999816)
        self.assertGreaterEqual(result.solidity, 0.92)
        self.assertLessEqual(result.ellipse_residual_p90, 0.15)


if __name__ == "__main__":
    unittest.main()
