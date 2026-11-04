#!/usr/bin/env python3
"""Run the comparison pipeline without writing into historical result paths."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

from modules.clean_pit_registry import validate_clean_truth_configuration  # noqa: E402
from modules.pipeline_start import (  # noqa: E402
    CNN_AUXILIARY_ARTIFACTS,
    FCN_BACK_MODEL_OUTPUT,
    FCN_FRONT_MODEL_OUTPUT,
    DCC_BACK_KANIK_OUTPUT,
    DCC_FRONT_KANIK_OUTPUT,
    STARTING_POINTS,
    resolve_input_source,
    seed_cnn_prediction_coordinates,
    seed_dcc_model_outputs,
    seed_fcn_prediction_coordinates,
    seed_fcn_model_outputs,
    seed_kanik_archive,
    seed_preprocessed_prediction_coordinates,
    validate_starting_point_inputs,
)
from modules.pipeline_paths import (  # noqa: E402
    RUN_MANIFEST,
    active_run_root,
    artifact_path,
    create_pipeline_run,
    record_source_inputs,
)


def _update_manifest(**values) -> None:
    manifest_path = active_run_root() / RUN_MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(values)
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, manifest_path)


def _run_stage(name, function) -> object:
    import matplotlib.pyplot as plt

    print(f"\n=== {name} ===", flush=True)
    started = time.time()
    result = function()
    plt.close("all")
    print(f"=== {name} complete ({time.time() - started:.1f} s) ===", flush=True)
    return result


def _configure_plot_style() -> None:
    """Match the plotting defaults used by the historical analysis notebook."""
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 14,
            "font.family": "serif",
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "figure.dpi": 150,
        }
    )


def _write_pipeline_summary() -> dict:
    from modules.dataset_tools import mhio

    summary = {"methods": {}}
    for method in ("CNN", "FCN", "DCC"):
        summary["methods"][method] = {}
        for view in ("front", "back", "aligned"):
            base = f"predicted_list/{method}/{view}"
            summary["methods"][method][view] = {
                "predictions": len(mhio.load_json(f"{base}/final_pred_list.txt")),
                "true_positives": len(mhio.load_json(f"{base}/final_true_pos.txt")),
                "false_positives": len(mhio.load_json(f"{base}/final_false_pos.txt")),
                "false_negatives": len(mhio.load_json(f"{base}/final_false_neg.txt")),
                "fragments": len(mhio.load_json(f"{base}/final_fragments.txt")),
            }
    destination = artifact_path("pipeline_summary.json", for_write=True)
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
        stream.write("\n")
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="New, non-existing run directory")
    parser.add_argument(
        "--start-from",
        choices=STARTING_POINTS,
        required=True,
        help="Scientific stage at which FCN/DCC processing begins",
    )
    parser.add_argument(
        "--input-source",
        required=True,
        help="Root containing CNN coordinates and selected FCN/DCC starting inputs",
    )
    parser.add_argument(
        "--clean-id-registry",
        required=True,
        help="Explicit clean-pit ID registry selected for this analysis",
    )
    parser.add_argument(
        "--fiducial-cut-file",
        action="append",
        default=[],
        help="Optional ID-preserving clean-truth cut file; may be repeated",
    )
    parser.add_argument(
        "--spatial-fiducial-file",
        help=(
            "Optional spatial regions applied to both predictions and registered "
            "clean truth"
        ),
    )
    parser.add_argument("--model", default="models/FCN/unet_ROC.keras")
    parser.add_argument(
        "--fcn-front-model-output",
        default=FCN_FRONT_MODEL_OUTPUT,
        help="FCN front probability map path relative to --input-source",
    )
    parser.add_argument(
        "--fcn-back-model-output",
        default=FCN_BACK_MODEL_OUTPUT,
        help="FCN back probability map path relative to --input-source",
    )
    parser.add_argument(
        "--dcc-front-model-output",
        default=DCC_FRONT_KANIK_OUTPUT,
        help="Kanik/DCC raw front centroid output relative to --input-source",
    )
    parser.add_argument(
        "--dcc-back-model-output",
        default=DCC_BACK_KANIK_OUTPUT,
        help="Kanik/DCC raw reverse centroid output relative to --input-source",
    )
    parser.add_argument(
        "--dcc-from-model-output",
        action="store_true",
        help=(
            "With a preprocessed-coordinate start, regenerate DCC coordinates "
            "from archived Kanik model output using current presets"
        ),
    )
    parser.add_argument("--front-images", default="numpy_data_eval/preprocessed_dirty_eval.npy")
    parser.add_argument(
        "--back-images", default="numpy_data_eval/preprocessed_dirty_reverse_eval.npy"
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument(
        "--invert-fcn-model-output",
        action="store_true",
        help=(
            "Store 1 minus the FCN's native output before coordinate reconstruction; "
            "required when a pit-positive model is used with the historical "
            "background-probability extractor"
        ),
    )
    parser.add_argument("--front-threshold", type=float)
    parser.add_argument("--back-threshold", type=float)
    parser.add_argument("--sigma", type=float)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--beta", type=float)
    parser.add_argument(
        "--skip-spatial-plots",
        action="store_true",
        help="Skip the spatial-resolution plots included in the main analysis",
    )
    parser.add_argument("--include-legacy-curves", action="store_true")
    parser.add_argument(
        "--skip-gallery",
        action="store_true",
        help="Skip the full clean-pit and false-positive panel gallery",
    )
    parser.add_argument(
        "--skip-example-panels",
        action="store_true",
        help="Skip the selected manuscript example panels",
    )
    return parser.parse_args()


def _validate_reconstruction_parameters(args: argparse.Namespace) -> None:
    if (
        args.dcc_from_model_output
        and args.start_from != "preprocessed_predicted_coordinates"
    ):
        raise ValueError(
            "--dcc-from-model-output is only valid with "
            "--start-from preprocessed_predicted_coordinates"
        )
    if args.start_from == "preprocessed_predicted_coordinates":
        return
    required = ("front_threshold", "back_threshold", "sigma", "alpha", "beta")
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        options = ", ".join("--" + name.replace("_", "-") for name in missing)
        raise ValueError(
            f"{args.start_from} requires explicit FCN reconstruction parameters: "
            f"{options}"
        )


def main() -> None:
    args = _parse_args()
    _validate_reconstruction_parameters(args)
    validate_clean_truth_configuration(
        args.clean_id_registry,
        tuple(args.fiducial_cut_file),
        args.spatial_fiducial_file,
    )
    validate_starting_point_inputs(
        start_from=args.start_from,
        input_source=args.input_source,
        fcn_front_model_output=args.fcn_front_model_output,
        fcn_back_model_output=args.fcn_back_model_output,
        model_path=args.model,
        front_images=args.front_images,
        back_images=args.back_images,
        dcc_front_model_output=args.dcc_front_model_output,
        dcc_back_model_output=args.dcc_back_model_output,
    )

    _configure_plot_style()

    create_pipeline_run(
        args.run_dir,
        metadata={
            "entry_point": str(Path(__file__).resolve()),
            "arguments": vars(args),
        },
    )
    source_inputs = [
            "modules/dataset_tools/metadata/Image_location_metadata.txt",
            "modules/dataset_tools/metadata/aligned_Global_locations.txt",
            "modules/dataset_tools/metadata/all_aligned_Global_locations.txt",
            args.clean_id_registry,
            *args.fiducial_cut_file,
    ]
    if args.spatial_fiducial_file:
        source_inputs.append(args.spatial_fiducial_file)
    record_source_inputs(source_inputs)
    dcc_actual_start = (
        "model_output"
        if args.start_from == "model" or args.dcc_from_model_output
        else args.start_from
    )
    starting_point = {
        "requested": args.start_from,
        "input_source": str(resolve_input_source(args.input_source)),
        "methods": {
            "CNN": {
                "actual": "preprocessed_predicted_coordinates",
                "auxiliary_results": CNN_AUXILIARY_ARTIFACTS,
            },
            "FCN": {"actual": args.start_from},
            "DCC": {"actual": dcc_actual_start},
        },
    }
    if args.start_from == "model":
        starting_point["methods"]["DCC"]["fallback"] = (
            "DCC model integration is not available; starting from Kanik's raw "
            "model output instead"
        )
    if (
        args.start_from != "preprocessed_predicted_coordinates"
        or args.dcc_from_model_output
    ):
        starting_point["methods"]["DCC"]["processing"] = {
            "edge_threshold_pixels": 4,
            "coordinate_configuration": "current repository presets",
        }
    _update_manifest(
        status="running",
        started_epoch=time.time(),
        completed_stages=[],
        starting_point=starting_point,
    )

    completed = []
    try:
        _run_stage(
            "copy complete archived Kanik tree",
            lambda: seed_kanik_archive(args.input_source),
        )
        completed.append("kanik_archive")
        if args.start_from == "preprocessed_predicted_coordinates":
            if args.dcc_from_model_output:
                _run_stage(
                    "copy preprocessed CNN coordinates and auxiliary results",
                    lambda: seed_cnn_prediction_coordinates(args.input_source),
                )
                _run_stage(
                    "copy preprocessed FCN coordinates",
                    lambda: seed_fcn_prediction_coordinates(args.input_source),
                )
                dcc_outputs = _run_stage(
                    "copy Kanik/DCC model outputs",
                    lambda: seed_dcc_model_outputs(
                        args.input_source,
                        front_relative_path=args.dcc_front_model_output,
                        back_relative_path=args.dcc_back_model_output,
                    ),
                )
                from modules.compare.kanik_to_mhio import convert_dcc_predictions

                _run_stage(
                    "DCC preprocessed coordinate conversion",
                    lambda: convert_dcc_predictions(
                        front_source=dcc_outputs["front"],
                        back_source=dcc_outputs["back"],
                        edge_threshold=4,
                    ),
                )
                completed.extend(
                    [
                        "cnn_coordinates",
                        "fcn_coordinates",
                        "dcc_model_outputs",
                        "dcc_coordinates",
                    ]
                )
            else:
                _run_stage(
                    "copy preprocessed coordinates and auxiliary CNN results",
                    lambda: seed_preprocessed_prediction_coordinates(args.input_source),
                )
                completed.append("preprocessed_prediction_coordinates")
        else:
            from modules.compare.kanik_to_mhio import convert_dcc_predictions
            from modules.compare.reconstruct_unet_coordinates import (
                reconstruct_unet_coordinates,
            )

            _run_stage(
                "copy CNN coordinates and auxiliary results",
                lambda: seed_cnn_prediction_coordinates(args.input_source),
            )
            completed.append("cnn_coordinates")
            dcc_outputs = _run_stage(
                "copy Kanik/DCC model outputs",
                lambda: seed_dcc_model_outputs(
                    args.input_source,
                    front_relative_path=args.dcc_front_model_output,
                    back_relative_path=args.dcc_back_model_output,
                ),
            )
            completed.append("dcc_model_outputs")

            if args.start_from == "model":
                from modules.compare.predict_from_model import (
                    generate_probability_masks,
                )

                _run_stage(
                    "FCN model inference",
                    lambda: generate_probability_masks(
                        model_path=args.model,
                        front_input=args.front_images,
                        back_input=args.back_images,
                        batch_size=args.batch_size,
                        require_gpu=args.require_gpu,
                        invert_output=args.invert_fcn_model_output,
                    ),
                )
                completed.append("fcn_model_outputs")
            else:
                _run_stage(
                    "copy FCN model outputs",
                    lambda: seed_fcn_model_outputs(
                        args.input_source,
                        front_relative_path=args.fcn_front_model_output,
                        back_relative_path=args.fcn_back_model_output,
                    ),
                )
                completed.append("fcn_model_outputs")

            _run_stage(
                "FCN preprocessed coordinate reconstruction",
                lambda: reconstruct_unet_coordinates(
                    front_threshold=args.front_threshold,
                    back_threshold=args.back_threshold,
                    sigma=args.sigma,
                    alpha=args.alpha,
                    beta=args.beta,
                ),
            )
            completed.append("fcn_coordinates")
            _run_stage(
                "DCC preprocessed coordinate conversion",
                lambda: convert_dcc_predictions(
                    front_source=dcc_outputs["front"],
                    back_source=dcc_outputs["back"],
                    edge_threshold=4,
                ),
            )
            completed.append("dcc_coordinates")

        if args.spatial_fiducial_file:
            from modules.fiducial_regions import apply_prediction_fiducials

            _run_stage(
                "apply spatial fiducials to predictions",
                lambda: apply_prediction_fiducials(args.spatial_fiducial_file),
            )
            completed.append("prediction_fiducials")

        from modules.compare.common_aligned_clean_list import clean_common_aligned
        from modules.compare.compare_CNN_aligned import compare_CNN_aligned
        from modules.compare.compare_DCC_aligned import compare_DCC_aligned
        from modules.compare.compare_FCN_aligned import compare_FCN_aligned
        from modules.compare.compare_clean_aligned import compare_clean_aligned
        from modules.compare.test_CNN_aligned import evaluate_CNN_aligned
        from modules.compare.test_CNN_back import evaluate_CNN_pits_back
        from modules.compare.test_CNN_front import evaluate_CNN_pits
        from modules.compare.test_DCC_aligned import evaluate_DCC_aligned
        from modules.compare.test_DCC_back import evaluate_DCC_back
        from modules.compare.test_DCC_front import evaluate_DCC_front
        from modules.compare.test_FCN_aligned import evaluate_FCN_aligned
        from modules.compare.test_FCN_back import evaluate_FCN_back
        from modules.compare.test_FCN_front import evaluate_FCN_front
        from pit_id import generate_pit_ids

        stages = [
            ("compare clean truth lists", compare_clean_aligned),
            (
                "build aligned clean truth with stable IDs",
                lambda: clean_common_aligned(
                    clean_id_registry=args.clean_id_registry,
                    fiducial_cut_files=tuple(args.fiducial_cut_file),
                    spatial_fiducial_file=args.spatial_fiducial_file,
                ),
            ),
            ("align CNN front/back", compare_CNN_aligned),
            ("align FCN front/back", compare_FCN_aligned),
            ("align DCC front/back", compare_DCC_aligned),
            ("evaluate CNN front", evaluate_CNN_pits),
            ("evaluate CNN back", evaluate_CNN_pits_back),
            ("evaluate CNN aligned", evaluate_CNN_aligned),
            ("evaluate FCN front", evaluate_FCN_front),
            ("evaluate FCN back", evaluate_FCN_back),
            ("evaluate FCN aligned", evaluate_FCN_aligned),
            ("evaluate DCC front", evaluate_DCC_front),
            ("evaluate DCC back", evaluate_DCC_back),
            ("evaluate DCC aligned", evaluate_DCC_aligned),
            ("assign run-specific false-positive IDs", generate_pit_ids),
        ]
        for name, function in stages:
            _run_stage(name, function)
            completed.append(name)
            _update_manifest(completed_stages=completed)

        _run_stage("write pipeline summary", _write_pipeline_summary)
        completed.append("pipeline_summary")

        if not args.skip_spatial_plots:
            from modules.compare.get_spacial_reso_CNN_front import (
                get_spacial_reso_CNN_front,
            )
            from modules.compare.get_spacial_reso_DCC_front import (
                get_spacial_reso_DCC_front,
            )
            from modules.compare.get_spacial_reso_FCN_front import (
                get_spacial_reso_FCN_front,
            )

            for name, function in (
                ("CNN spatial resolution", get_spacial_reso_CNN_front),
                ("FCN spatial resolution", get_spacial_reso_FCN_front),
                ("DCC spatial resolution", get_spacial_reso_DCC_front),
            ):
                _run_stage(name, function)
                completed.append(name)

        if args.include_legacy_curves:
            from modules.compare.eff_purity_CNN import evaluate_efficiency_purity_CNN
            from modules.compare.eff_purity_DCC import evaluate_efficiency_purity_DCC
            from modules.compare.eff_purity_FCN import evaluate_efficiency_purity_FCN

            for name, function in (
                ("CNN legacy efficiency/purity", evaluate_efficiency_purity_CNN),
                ("FCN legacy efficiency/purity", evaluate_efficiency_purity_FCN),
                ("DCC legacy efficiency/purity", evaluate_efficiency_purity_DCC),
            ):
                _run_stage(name, function)
                completed.append(name)

        if not args.skip_gallery:
            from gallery_generation import generate_gallery

            _run_stage("full gallery", generate_gallery)
            completed.append("gallery")
        if not args.skip_example_panels:
            from generate_example_panels import generate_example_panels

            _run_stage("example panels", generate_example_panels)
            completed.append("example_panels")

        _update_manifest(
            status="complete",
            completed_epoch=time.time(),
            completed_stages=completed,
        )
        print(f"\nPipeline artifacts: {active_run_root()}")
    except Exception as error:
        _update_manifest(
            status="failed",
            failed_epoch=time.time(),
            completed_stages=completed,
            error={"type": type(error).__name__, "message": str(error)},
        )
        raise


if __name__ == "__main__":
    main()
