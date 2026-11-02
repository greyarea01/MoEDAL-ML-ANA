import json
import os
import tempfile
import unittest
from pathlib import Path

from modules.dataset_tools import mhio
from modules.pipeline_paths import (
    RUN_DIR_ENV,
    RUN_MANIFEST,
    artifact_path,
    copy_source_artifact,
    create_pipeline_run,
    record_source_inputs,
    resolve_pipeline_path,
    seed_artifacts,
    sha256_file,
)


class PipelinePathTests(unittest.TestCase):
    def setUp(self):
        self.previous_run = os.environ.pop(RUN_DIR_ENV, None)
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        os.environ.pop(RUN_DIR_ENV, None)
        if self.previous_run is not None:
            os.environ[RUN_DIR_ENV] = self.previous_run
        self.temporary.cleanup()

    def test_run_and_artifacts_are_exclusive(self):
        run = create_pipeline_run(self.root / "run")
        output = artifact_path("plots/example.png", for_write=True)
        self.assertEqual(output, run / "plots/example.png")
        output.write_bytes(b"example")
        with self.assertRaises(FileExistsError):
            artifact_path("plots/example.png", for_write=True)
        with self.assertRaises(FileExistsError):
            create_pipeline_run(run)

    def test_seed_records_hash_and_relative_layout(self):
        source = self.root / "source"
        relative = Path("predicted_etch_pits/FCN/final_pr.txt")
        source_file = source / relative
        source_file.parent.mkdir(parents=True)
        source_file.write_text("[[1, 2]]\n", encoding="utf-8")
        run = create_pipeline_run(self.root / "run")

        [destination] = seed_artifacts([relative], source_root=source)

        self.assertEqual(destination, run / relative)
        self.assertEqual(destination.read_bytes(), source_file.read_bytes())
        manifest = json.loads((run / RUN_MANIFEST).read_text(encoding="utf-8"))
        self.assertEqual(manifest["seeded_artifacts"][0]["sha256"], sha256_file(source_file))

    def test_source_can_be_copied_to_a_named_run_artifact(self):
        source = self.root / "raw_front.txt"
        source.write_text("raw model output\n", encoding="utf-8")
        run = create_pipeline_run(self.root / "run")

        destination = copy_source_artifact(
            source, "model_outputs/DCC/DirtyTestCentroid.txt"
        )

        self.assertEqual(
            destination, run / "model_outputs/DCC/DirtyTestCentroid.txt"
        )
        self.assertEqual(destination.read_bytes(), source.read_bytes())
        manifest = json.loads((run / RUN_MANIFEST).read_text(encoding="utf-8"))
        self.assertEqual(manifest["seeded_artifacts"][0]["source"], str(source))

    def test_mhio_routes_generated_json_to_run(self):
        run = create_pipeline_run(self.root / "run")
        mhio.save_json([[3, 4]], "Aligned_clean.txt")
        self.assertEqual(mhio.load_json("Aligned_clean.txt"), [[3, 4]])
        self.assertTrue((run / "Aligned_clean.txt").is_file())

    def test_source_input_hash_is_recorded(self):
        from modules.pipeline_paths import PROJECT_ROOT

        run = create_pipeline_run(self.root / "run")
        relative = "modules/dataset_tools/metadata/Image_location_metadata.txt"
        [record] = record_source_inputs([relative])
        manifest = json.loads((run / RUN_MANIFEST).read_text(encoding="utf-8"))
        self.assertEqual(
            record["sha256"], sha256_file(PROJECT_ROOT / relative)
        )
        self.assertEqual(manifest["source_inputs"], [record])

    def test_generated_write_requires_active_run(self):
        with self.assertRaises(RuntimeError):
            artifact_path("predicted_list/result.txt", for_write=True)

    def test_static_inputs_remain_readable_without_a_run(self):
        from modules.pipeline_paths import PROJECT_ROOT

        relative = "modules/dataset_tools/metadata/Image_location_metadata.txt"
        self.assertEqual(
            resolve_pipeline_path(relative, must_exist=True),
            PROJECT_ROOT / relative,
        )

    def test_archived_kanik_inputs_are_read_from_active_run(self):
        run = create_pipeline_run(self.root / "run")
        archived = artifact_path("kanik/clean_aligned.txt", for_write=True)
        archived.write_text("[]\n", encoding="utf-8")

        self.assertEqual(
            resolve_pipeline_path("kanik/clean_aligned.txt", must_exist=True),
            run / "kanik/clean_aligned.txt",
        )

    def test_run_cannot_be_nested_in_legacy_results(self):
        from modules.pipeline_paths import PROJECT_ROOT

        with self.assertRaises(ValueError):
            create_pipeline_run(PROJECT_ROOT / "plots/new_run")


if __name__ == "__main__":
    unittest.main()
