import json
import os
import tempfile
import unittest
from pathlib import Path

from modules.compare.kanik_to_mhio import convert_dcc_predictions
from modules.pipeline_paths import RUN_DIR_ENV, create_pipeline_run, sha256_file
from modules.pipeline_start import (
    CNN_AUXILIARY_ARTIFACTS,
    CNN_COORDINATE_ARTIFACTS,
    CNN_STARTING_ARTIFACTS,
    DCC_BACK_MODEL_OUTPUT,
    DCC_BACK_KANIK_OUTPUT,
    DCC_FRONT_MODEL_OUTPUT,
    DCC_FRONT_KANIK_OUTPUT,
    FCN_BACK_MODEL_OUTPUT,
    FCN_COORDINATE_ARTIFACTS,
    FCN_FRONT_MODEL_OUTPUT,
    KANIK_CLEAN_TRUTH,
    PREDICTION_COORDINATE_ARTIFACTS,
    seed_cnn_prediction_coordinates,
    seed_dcc_model_outputs,
    seed_fcn_prediction_coordinates,
    seed_fcn_model_outputs,
    seed_kanik_archive,
    seed_preprocessed_prediction_coordinates,
    validate_starting_point_inputs,
)


class PipelineStartingPointTests(unittest.TestCase):
    def setUp(self):
        self.previous_run = os.environ.pop(RUN_DIR_ENV, None)
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        os.environ.pop(RUN_DIR_ENV, None)
        if self.previous_run is not None:
            os.environ[RUN_DIR_ENV] = self.previous_run
        self.temporary.cleanup()

    def _write_source_files(self, relative_paths):
        source = self.root / "source"
        for index, relative in enumerate(relative_paths):
            path = source / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"artifact {index}\n", encoding="utf-8")
        return source

    def test_coordinate_start_copies_six_coordinates_and_three_cnn_results(self):
        source = self._write_source_files(
            [
                *PREDICTION_COORDINATE_ARTIFACTS,
                *CNN_AUXILIARY_ARTIFACTS,
                KANIK_CLEAN_TRUTH,
            ]
        )
        provenance = source / "predicted_etch_pits/DCC/conversion_metadata.json"
        provenance.write_text(json.dumps({"historical": True}), encoding="utf-8")
        validate_starting_point_inputs(
            start_from="preprocessed_predicted_coordinates",
            input_source=source,
        )
        run = create_pipeline_run(self.root / "run")

        kanik = seed_kanik_archive(source)
        result = seed_preprocessed_prediction_coordinates(source)

        self.assertEqual(kanik["file_count"], 1)
        self.assertEqual(
            (run / KANIK_CLEAN_TRUTH).read_bytes(),
            (source / KANIK_CLEAN_TRUTH).read_bytes(),
        )
        self.assertEqual(len(result["coordinate_artifacts"]), 6)
        self.assertEqual(len(result["cnn_auxiliary_artifacts"]), 3)
        for relative in PREDICTION_COORDINATE_ARTIFACTS:
            self.assertEqual((run / relative).read_bytes(), (source / relative).read_bytes())
        for relative in CNN_AUXILIARY_ARTIFACTS:
            self.assertEqual((run / relative).read_bytes(), (source / relative).read_bytes())
        self.assertEqual(
            (run / "predicted_etch_pits/DCC/conversion_metadata.json").read_bytes(),
            provenance.read_bytes(),
        )

    def test_model_output_start_copies_fcn_dcc_and_cnn_inputs(self):
        source = self._write_source_files(
            [
                *CNN_STARTING_ARTIFACTS,
                FCN_FRONT_MODEL_OUTPUT,
                FCN_BACK_MODEL_OUTPUT,
                KANIK_CLEAN_TRUTH,
                DCC_FRONT_KANIK_OUTPUT,
                DCC_BACK_KANIK_OUTPUT,
            ]
        )
        validate_starting_point_inputs(
            start_from="model_output",
            input_source=source,
        )
        run = create_pipeline_run(self.root / "run")

        seed_kanik_archive(source)
        seed_cnn_prediction_coordinates(source)
        seed_fcn_model_outputs(source)
        seed_dcc_model_outputs(source)

        for relative in CNN_STARTING_ARTIFACTS:
            self.assertEqual((run / relative).read_bytes(), (source / relative).read_bytes())
        self.assertEqual(
            (run / FCN_FRONT_MODEL_OUTPUT).read_bytes(),
            (source / FCN_FRONT_MODEL_OUTPUT).read_bytes(),
        )
        self.assertEqual(
            (run / FCN_BACK_MODEL_OUTPUT).read_bytes(),
            (source / FCN_BACK_MODEL_OUTPUT).read_bytes(),
        )
        self.assertEqual(
            (run / DCC_FRONT_MODEL_OUTPUT).read_bytes(),
            (source / DCC_FRONT_KANIK_OUTPUT).read_bytes(),
        )
        self.assertEqual(
            (run / DCC_BACK_MODEL_OUTPUT).read_bytes(),
            (source / DCC_BACK_KANIK_OUTPUT).read_bytes(),
        )

    def test_hybrid_coordinate_start_can_seed_fcn_without_historical_dcc(self):
        source = self._write_source_files(FCN_COORDINATE_ARTIFACTS)
        run = create_pipeline_run(self.root / "run")

        result = seed_fcn_prediction_coordinates(source)

        self.assertEqual(len(result["coordinate_artifacts"]), 2)
        for relative in FCN_COORDINATE_ARTIFACTS:
            self.assertEqual((run / relative).read_bytes(), (source / relative).read_bytes())

    def test_coordinate_start_rejects_a_missing_method_side(self):
        source = self._write_source_files(
            [
                *PREDICTION_COORDINATE_ARTIFACTS[:-1],
                *CNN_AUXILIARY_ARTIFACTS,
                KANIK_CLEAN_TRUTH,
            ]
        )
        with self.assertRaisesRegex(FileNotFoundError, "DCC/final_reverse_pr.txt"):
            validate_starting_point_inputs(
                start_from="preprocessed_predicted_coordinates",
                input_source=source,
            )

    def test_current_dcc_model_output_path_retains_four_pixel_cut_and_presets(self):
        from modules.pipeline_paths import PROJECT_ROOT

        run = create_pipeline_run(self.root / "run")
        copied = seed_dcc_model_outputs(PROJECT_ROOT / "pipeline_runs/paperV1_baseline")

        report = convert_dcc_predictions(
            front_source=copied["front"],
            back_source=copied["back"],
            edge_threshold=4,
        )

        self.assertEqual(report["edge_threshold"], 4)
        self.assertEqual(report["mapping"]["front"]["preset"], "dc")
        self.assertEqual(report["mapping"]["front"]["offset"], [10, -7])
        self.assertEqual(report["mapping"]["back"]["preset"], "drc")
        self.assertEqual(report["mapping"]["back"]["offset"], [10, 0])
        self.assertEqual(report["front"]["count"], 1531)
        self.assertEqual(report["back"]["count"], 1504)
        self.assertEqual(
            len(
                json.loads(
                    (
                        run
                        / "predicted_etch_pits/DCC/front_local_coordinates.txt"
                    ).read_text(encoding="utf-8")
                )
            ),
            1531,
        )
        self.assertEqual(
            len(
                json.loads(
                    (
                        run
                        / "predicted_etch_pits/DCC/back_local_coordinates.txt"
                    ).read_text(encoding="utf-8")
                )
            ),
            1504,
        )
        self.assertEqual(
            sha256_file(run / "predicted_etch_pits/DCC/final_pr.txt"),
            "d159d931c283ac8ed5c1f2df33245787ac19e3f469be80aad94187ded26689f4",
        )
        self.assertEqual(
            sha256_file(run / "predicted_etch_pits/DCC/final_reverse_pr.txt"),
            "28619e2983077a8fcd7e973ec850060e19426cf268c2832c2d88a806cda14983",
        )


if __name__ == "__main__":
    unittest.main()
