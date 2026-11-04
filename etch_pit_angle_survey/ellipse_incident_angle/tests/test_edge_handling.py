from __future__ import annotations

import csv
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np


PIPELINE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PIPELINE_DIR.parents[1]
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from ellipse_fit import FitConfig, extract_valid_crop, fit_etch_pit  # noqa: E402


class EdgeHandlingTests(unittest.TestCase):
    def test_crop_masks_missing_pixels_without_reflection(self) -> None:
        source = np.arange(12 * 12, dtype=np.uint8).reshape(12, 12)
        config = FitConfig(crop_radius_px=4)
        crop = extract_valid_crop(source, local_x=1, local_y=6, config=config)

        self.assertEqual(crop.image.shape, (9, 9))
        self.assertTrue(np.all(crop.image[:, :3] == 255))
        self.assertFalse(np.any(crop.valid_pixels[:, :3]))
        np.testing.assert_array_equal(crop.image[:, 3:], source[2:11, 0:6])
        self.assertTrue(np.all(crop.valid_pixels[:, 3:]))

    def test_truncated_synthetic_pit_fails_source_boundary_qc(self) -> None:
        source = np.full((80, 80), 245, dtype=np.uint8)
        cv2.ellipse(source, (3, 40), (8, 13), 20, 0, 360, 20, -1)
        result = fit_etch_pit(source, local_x=3, local_y=40)

        self.assertTrue(result.candidate_found)
        self.assertIn("F", result.qc_failure_codes)
        self.assertTrue(result.touches_source_image_boundary)
        points = result.contour[:, 0, :]
        self.assertTrue(np.all(result.crop.valid_pixels[points[:, 1], points[:, 0]]))

    def test_c0464_has_no_mirrored_contour_points(self) -> None:
        manifest_path = PIPELINE_DIR / "inputs/clean_surface_locations.csv"
        with manifest_path.open(newline="", encoding="utf-8") as input_file:
            row = next(
                record
                for record in csv.DictReader(input_file)
                if record["clean_id"] == "C0464"
            )

        image_paths = {
            "c1": REPO_ROOT / "numpy_data_eval/preprocessed_clean_eval.npy",
            "c2": REPO_ROOT / "numpy_data_eval/clean_2_eval.npy",
        }
        for surface in ("c1", "c2"):
            images = np.load(image_paths[surface], mmap_mode="r")
            image_number = int(float(row[f"{surface}_image"]))
            result = fit_etch_pit(
                images[image_number - 251],
                float(row[f"{surface}_local_x"]),
                float(row[f"{surface}_local_y"]),
            )
            self.assertFalse(np.all(result.crop.valid_pixels))
            self.assertTrue(result.candidate_found)
            self.assertIn("F", result.qc_failure_codes)
            points = result.contour[:, 0, :]
            self.assertTrue(
                np.all(result.crop.valid_pixels[points[:, 1], points[:, 0]])
            )


if __name__ == "__main__":
    unittest.main()
