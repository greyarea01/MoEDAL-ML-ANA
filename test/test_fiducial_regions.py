import json
import unittest
from pathlib import Path

from modules.clean_pit_registry import load_clean_id_registry, select_clean_truth
from modules.fiducial_regions import (
    FiducialRegionEvaluator,
    load_fiducial_regions,
)
from modules.pipeline_paths import PROJECT_ROOT


CONFIG = PROJECT_ROOT / "fiducial_cuts/analysis_v2_regions.json"
BLACKLIST = PROJECT_ROOT / "fiducial_cuts/analysis_v2_truth_blacklist.json"
REGISTRY = PROJECT_ROOT / "clean_pit_id_registries/paperV1.json"


class FiducialRegionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, cls.config, _ = load_fiducial_regions(CONFIG)
        cls.evaluator = FiducialRegionEvaluator(cls.config)

    def test_alignment_hole_uses_inclusive_184_pixel_radius(self):
        center = self.config["alignment_holes"]["centers"][0]["coordinate"]
        on_boundary = [center[0] + 184.0, center[1]]
        outside = [center[0] + 184.000001, center[1]]
        self.assertIn(
            "alignment_hole", self.evaluator.classify(on_boundary)["reasons"]
        )
        self.assertNotIn(
            "alignment_hole", self.evaluator.classify(outside)["reasons"]
        )

    def test_physical_region_is_an_inclusive_eight_by_eight_square(self):
        feature = self.config["physical_inconsistencies"]["features"][0]
        center = feature["front_center"]
        corner = [center[0] + 4.0, center[1] - 4.0]
        outside = [center[0] + 4.000001, center[1]]
        self.assertIn(
            "physical_inconsistency", self.evaluator.classify(corner)["reasons"]
        )
        self.assertNotIn(
            "physical_inconsistency", self.evaluator.classify(outside)["reasons"]
        )

    def test_edge_region_matches_existing_strict_four_pixel_convention(self):
        point = [7000.0, 1000.0]
        hit = self.evaluator.classify(
            point, side="front", local_coordinate=[300, 3.999, 200]
        )
        boundary = self.evaluator.classify(
            point, side="front", local_coordinate=[300, 4.0, 200]
        )
        self.assertEqual(hit["reasons"], ["image_edge"])
        self.assertEqual(boundary["reasons"], [])

    def test_registered_truth_cut_is_id_preserving_and_reproducible(self):
        _, records, _ = load_clean_id_registry(REGISTRY)
        selection = select_clean_truth(
            [record["coordinate"] for record in records],
            registry_path=REGISTRY,
            spatial_fiducial_file=CONFIG,
        )
        self.assertEqual(selection["active_count"], 1233)
        self.assertEqual(
            [record["id"] for record in selection["excluded_records"]],
            [
                "C0188",
                "C0243",
                "C0300",
                "C0874",
                "C0929",
                "C0932",
                "C0952",
                "C0997",
                "C1120",
            ],
        )
        active_ids = [record["id"] for record in selection["active_records"]]
        self.assertEqual(active_ids[0], "C0000")
        self.assertEqual(active_ids[-1], "C1241")
        self.assertNotIn("C0929", active_ids)

    def test_analysis_blacklist_combines_with_spatial_cuts_without_renumbering(self):
        _, records, _ = load_clean_id_registry(REGISTRY)
        selection = select_clean_truth(
            [record["coordinate"] for record in records],
            registry_path=REGISTRY,
            fiducial_cut_files=(BLACKLIST,),
            spatial_fiducial_file=CONFIG,
        )
        excluded = {
            record["id"]: record["exclusions"]
            for record in selection["excluded_records"]
        }
        self.assertEqual(selection["active_count"], 1232)
        self.assertEqual(selection["excluded_count"], 10)
        self.assertIn("C0123", excluded)
        self.assertEqual(len(excluded["C0123"]), 1)
        self.assertEqual(len(excluded["C0300"]), 2)
        self.assertEqual(len(excluded["C0997"]), 2)
        active_ids = [record["id"] for record in selection["active_records"]]
        self.assertEqual(active_ids[122], "C0122")
        self.assertEqual(active_ids[123], "C0124")


if __name__ == "__main__":
    unittest.main()
