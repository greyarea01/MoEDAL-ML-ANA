#!/usr/bin/env python3
"""Run the comparison pipeline without writing into historical result paths."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

from modules.pipeline_paths import (  # noqa: E402
    PROJECT_ROOT,
    RUN_MANIFEST,
    active_run_root,
    artifact_path,
    create_pipeline_run,
    record_source_inputs,
    seed_artifacts,
)


PREDICTION_ARTIFACTS = [
    f"predicted_etch_pits/{method}/{side}.txt"
    for method in ("CNN", "FCN", "DCC")
    for side in ("final_pr", "final_reverse_pr")
]
CNN_PREDICTION_ARTIFACTS = [
    "predicted_etch_pits/CNN/final_pr.txt",
    "predicted_etch_pits/CNN/final_reverse_pr.txt",
]


def _update_manifest(**values) -> None:
    manifest_path = active_run_root() / RUN_MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(values)
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, manifest_path)


def _run_stage(name, function) -> None:
    import matplotlib.pyplot as plt

    print(f"\n=== {name} ===", flush=True)
    started = time.time()
    function()
    plt.close("all")
    print(f"=== {name} complete ({time.time() - started:.1f} s) ===", flush=True)


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
        "--prediction-mode",
        choices=("seed", "unet"),
        default="seed",
        help="Copy existing coordinate predictions or start from U-Net inference",
    )
    parser.add_argument(
        "--prediction-source",
        required=True,
        help="Root containing existing prediction artifacts used by seed mode",
    )
    parser.add_argument("--model", default="model/unet_ROC.keras")
    parser.add_argument("--front-images", default="numpy_data_eval/preprocessed_dirty_eval.npy")
    parser.add_argument(
        "--back-images", default="numpy_data_eval/preprocessed_dirty_reverse_eval.npy"
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--require-gpu", action="store_true")
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
    parser.add_argument("--generate-gallery", action="store_true")
    parser.add_argument("--generate-example-panels", action="store_true")
    return parser.parse_args()


def _validate_unet_parameters(args: argparse.Namespace) -> None:
    required = ("front_threshold", "back_threshold", "sigma", "alpha", "beta")
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        options = ", ".join("--" + name.replace("_", "-") for name in missing)
        raise ValueError(f"U-Net mode requires explicit parameters: {options}")


def main() -> None:
    args = _parse_args()
    if args.prediction_mode == "unet":
        _validate_unet_parameters(args)

    _configure_plot_style()

    create_pipeline_run(
        args.run_dir,
        metadata={
            "entry_point": str(Path(__file__).resolve()),
            "arguments": vars(args),
        },
    )
    record_source_inputs(
        [
            "kanik/clean_aligned.txt",
            "modules/dataset_tools/metadata/Image_location_metadata.txt",
            "modules/dataset_tools/metadata/aligned_Global_locations.txt",
            "modules/dataset_tools/metadata/all_aligned_Global_locations.txt",
        ]
    )
    _update_manifest(status="running", started_epoch=time.time(), completed_stages=[])

    completed = []
    try:
        if args.prediction_mode == "seed":
            _run_stage(
                "seed coordinate predictions",
                lambda: seed_artifacts(
                    PREDICTION_ARTIFACTS, source_root=args.prediction_source
                ),
            )
            completed.append("seed_predictions")
        else:
            from modules.compare.kanik_to_mhio import convert_dcc_predictions
            from modules.compare.predict_from_model import generate_probability_masks
            from modules.compare.reconstruct_unet_coordinates import (
                reconstruct_unet_coordinates,
            )

            _run_stage(
                "U-Net probability masks",
                lambda: generate_probability_masks(
                    model_path=args.model,
                    front_input=args.front_images,
                    back_input=args.back_images,
                    batch_size=args.batch_size,
                    require_gpu=args.require_gpu,
                ),
            )
            completed.append("probability_masks")
            _run_stage(
                "U-Net coordinate reconstruction",
                lambda: reconstruct_unet_coordinates(
                    front_threshold=args.front_threshold,
                    back_threshold=args.back_threshold,
                    sigma=args.sigma,
                    alpha=args.alpha,
                    beta=args.beta,
                ),
            )
            completed.append("fcn_coordinates")
            _run_stage("DCC coordinate conversion", convert_dcc_predictions)
            completed.append("dcc_coordinates")
            _run_stage(
                "seed CNN coordinate predictions",
                lambda: seed_artifacts(
                    CNN_PREDICTION_ARTIFACTS,
                    source_root=args.prediction_source,
                ),
            )
            completed.append("cnn_coordinates")

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
            ("build aligned clean truth", clean_common_aligned),
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
            ("assign pit IDs", generate_pit_ids),
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

        if args.generate_gallery:
            from gallery_generation import generate_gallery

            _run_stage("full gallery", generate_gallery)
            completed.append("gallery")
        if args.generate_example_panels:
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
