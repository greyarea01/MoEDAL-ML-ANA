import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from modules.clean_pit_registry import (
    load_clean_id_registry,
    select_clean_truth,
    validate_clean_truth_configuration,
)
from modules.pipeline_paths import PROJECT_ROOT


REGISTRY = PROJECT_ROOT / "clean_pit_id_registries/paperV1.json"
HISTORICAL = (
    PROJECT_ROOT
    / "pipeline_runs/paperV1_baseline/predicted_list/ids/Aligned_clean_with_ids.txt"
)


class CleanPitRegistryTests(unittest.TestCase):
    def test_paper_v1_preserves_historical_ids_and_coordinates(self):
        _, records, _ = load_clean_id_registry(REGISTRY)
        self.assertEqual(len(records), 1242)
        self.assertEqual(len({record["id"] for record in records}), 1242)
        self.assertEqual(records[0], {"id": "C0000", "coordinate": [6171, -206]})
        self.assertEqual(
            records[-1], {"id": "C1241", "coordinate": [11953, 10708]}
        )
        historical_serialization_hash = (
            "689d93a983530692ddeaa9099aebe4c344d9aba3593660f9c6fe054fee9cde5b"
        )
        self.assertEqual(
            hashlib.sha256(json.dumps(records).encode()).hexdigest(),
            historical_serialization_hash,
        )
        if HISTORICAL.is_file():
            historical = json.loads(HISTORICAL.read_text(encoding="utf-8"))
            self.assertEqual(records, historical)

    def test_no_cut_preserves_every_record(self):
        _, records, _ = load_clean_id_registry(REGISTRY)
        selection = select_clean_truth(
            [record["coordinate"] for record in records],
            registry_path=REGISTRY,
        )
        self.assertEqual(selection["active_records"], records)
        self.assertEqual(selection["excluded_records"], [])

    def test_registry_must_live_in_the_registry_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            outside = Path(directory) / "registry.json"
            outside.write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must be stored below"):
                validate_clean_truth_configuration(outside)

    def test_cut_excludes_ids_without_renumbering_survivors(self):
        _, records, registry_hash = load_clean_id_registry(REGISTRY)
        with tempfile.TemporaryDirectory() as directory:
            cut_path = Path(directory) / "cut.json"
            cut_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "clean_id_registry_sha256": registry_hash,
                        "exclude_clean_ids": [
                            {"id": "C0001", "reason": "test cut"},
                            {"id": "C0123", "reason": "second test cut"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with patch("modules.clean_pit_registry.CUT_ROOT", Path(directory)):
                selection = select_clean_truth(
                    [record["coordinate"] for record in records],
                    registry_path=REGISTRY,
                    fiducial_cut_files=(cut_path,),
                )

        active_ids = [record["id"] for record in selection["active_records"]]
        self.assertEqual(selection["active_count"], 1240)
        self.assertNotIn("C0001", active_ids)
        self.assertNotIn("C0123", active_ids)
        self.assertEqual(active_ids[:3], ["C0000", "C0002", "C0003"])
        self.assertIn("C0124", active_ids)
        self.assertEqual(
            [record["id"] for record in selection["excluded_records"]],
            ["C0001", "C0123"],
        )

    def test_changed_base_truth_is_rejected_instead_of_reidentified(self):
        _, records, _ = load_clean_id_registry(REGISTRY)
        coordinates = [record["coordinate"][:] for record in records]
        coordinates[2][0] += 1
        with self.assertRaisesRegex(ValueError, r"row 2 \(C0002\)"):
            select_clean_truth(coordinates, registry_path=REGISTRY)

    def test_cut_must_target_selected_registry_hash(self):
        _, records, _ = load_clean_id_registry(REGISTRY)
        with tempfile.TemporaryDirectory() as directory:
            cut_path = Path(directory) / "wrong_registry.json"
            cut_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "clean_id_registry_sha256": hashlib.sha256(b"wrong").hexdigest(),
                        "exclude_clean_ids": [],
                    }
                ),
                encoding="utf-8",
            )
            with patch("modules.clean_pit_registry.CUT_ROOT", Path(directory)):
                with self.assertRaisesRegex(ValueError, "targets registry SHA-256"):
                    select_clean_truth(
                        [record["coordinate"] for record in records],
                        registry_path=REGISTRY,
                        fiducial_cut_files=(cut_path,),
                    )


if __name__ == "__main__":
    unittest.main()
