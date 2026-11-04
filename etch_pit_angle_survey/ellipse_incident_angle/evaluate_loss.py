#!/usr/bin/env python3
"""Evaluate the configurable robust C1-C2 incident-angle loss."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
SURVEY_ROOT = SCRIPT_DIR.parent
REPO_ROOT = SURVEY_ROOT.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from ellipse_fit import FitConfig, FitResult, fit_etch_pit  # noqa: E402
from loss_function import (  # noqa: E402
    LossConfig,
    LossWeights,
    PitLossInput,
    PitLossResult,
    SurfaceLossInput,
    evaluate_loss,
)
from run_pipeline import (  # noqa: E402
    C1_IMAGE_PATH,
    C2_IMAGE_PATH,
    DEFAULT_INPUT_PATH,
    FIRST_IMAGE_NUMBER,
    load_manifest,
    run_fits,
    validate_manifest_truth,
)


DEFAULT_SUMMARY_OUTPUT = SCRIPT_DIR / "data/current_config_loss_summary.json"
DEFAULT_PER_PIT_OUTPUT = SCRIPT_DIR / "data/current_config_loss_per_pit.csv"


def _comma_floats(value: str) -> tuple[float, ...]:
    try:
        result = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if not result:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated number")
    return result


def _offsets(value: str) -> tuple[tuple[int, int], ...]:
    try:
        offsets = tuple(
            tuple(int(coordinate.strip()) for coordinate in item.split(":"))
            for item in value.split(",")
            if item.strip()
        )
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if not offsets or any(len(offset) != 2 for offset in offsets):
        raise argparse.ArgumentTypeError(
            "Use comma-separated x:y offsets, for example -1:-1,0:-1,1:-1"
        )
    return offsets


def parse_args() -> argparse.Namespace:
    fit_defaults = FitConfig()
    loss_defaults = LossConfig()
    weight_defaults = loss_defaults.weights
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_OUTPUT)
    parser.add_argument("--per-pit-output", type=Path, default=DEFAULT_PER_PIT_OUTPUT)

    fit_group = parser.add_argument_group("ellipse fit configuration")
    fit_group.add_argument(
        "--crop-radius-px", type=int, default=fit_defaults.crop_radius_px
    )
    fit_group.add_argument(
        "--gaussian-kernel-px", type=int, default=fit_defaults.gaussian_kernel_px
    )
    fit_group.add_argument(
        "--gaussian-sigma-px", type=float, default=fit_defaults.gaussian_sigma_px
    )
    fit_group.add_argument(
        "--closing-kernel-px", type=int, default=fit_defaults.closing_kernel_px
    )
    fit_group.add_argument(
        "--minimum-contour-points",
        type=int,
        default=fit_defaults.minimum_contour_points,
    )
    fit_group.add_argument(
        "--minimum-contour-area-px2",
        type=float,
        default=fit_defaults.minimum_contour_area_px2,
    )
    fit_group.add_argument(
        "--minimum-minor-axis-px",
        type=float,
        default=fit_defaults.minimum_minor_axis_px,
    )
    fit_group.add_argument(
        "--maximum-major-axis-px",
        type=float,
        default=fit_defaults.maximum_major_axis_px,
    )
    fit_group.add_argument(
        "--maximum-center-offset-px",
        type=float,
        default=fit_defaults.maximum_center_offset_px,
    )
    fit_group.add_argument(
        "--minimum-solidity", type=float, default=fit_defaults.minimum_solidity
    )
    fit_group.add_argument(
        "--maximum-residual-p90",
        type=float,
        default=fit_defaults.maximum_residual_p90,
    )
    fit_group.add_argument(
        "--candidate-score-solidity-reference",
        type=float,
        default=fit_defaults.candidate_score_solidity_reference,
    )
    fit_group.add_argument(
        "--candidate-score-shape-weight",
        type=float,
        default=fit_defaults.candidate_score_shape_weight,
    )

    weight_group = parser.add_argument_group("provisional loss weights")
    weight_group.add_argument(
        "--k1-coverage", type=float, default=weight_defaults.coverage
    )
    weight_group.add_argument(
        "--k2-central-error",
        type=float,
        default=weight_defaults.central_absolute_error,
    )
    weight_group.add_argument(
        "--k3-tail-error", type=float, default=weight_defaults.tail_error
    )
    weight_group.add_argument(
        "--k4-signed-bias", type=float, default=weight_defaults.signed_bias
    )
    weight_group.add_argument(
        "--k5-stability", type=float, default=weight_defaults.stability_error
    )
    weight_group.add_argument(
        "--k6-low-angle", type=float, default=weight_defaults.low_angle
    )
    weight_group.add_argument(
        "--weights-final",
        action="store_true",
        help="Mark the supplied loss weights as selected rather than provisional.",
    )

    robust_group = parser.add_argument_group("robustness and loss configuration")
    robust_group.add_argument(
        "--perturbation-offsets",
        type=_offsets,
        default=loss_defaults.perturbation_offsets_xy,
        metavar="X:Y,...",
    )
    robust_group.add_argument(
        "--minimum-passing-perturbations",
        type=int,
        default=loss_defaults.minimum_passing_perturbations,
    )
    robust_group.add_argument(
        "--stability-failure-penalty-deg",
        type=float,
        default=loss_defaults.stability_failure_penalty_deg,
    )
    robust_group.add_argument(
        "--tail-percentiles",
        type=_comma_floats,
        default=loss_defaults.tail_percentiles,
    )
    robust_group.add_argument(
        "--tail-percentile-weights",
        type=_comma_floats,
        default=loss_defaults.tail_percentile_weights,
    )
    robust_group.add_argument(
        "--low-angle-threshold-deg",
        type=float,
        default=loss_defaults.low_angle_threshold_deg,
    )
    robust_group.add_argument(
        "--allowed-low-angle-fraction",
        type=float,
        default=loss_defaults.allowed_low_angle_fraction,
    )
    robust_group.add_argument(
        "--minimum-robust-paired-coverage-fraction",
        type=float,
        default=loss_defaults.minimum_robust_paired_coverage_fraction,
    )
    return parser.parse_args()


def build_fit_config(args: argparse.Namespace) -> FitConfig:
    return FitConfig(
        crop_radius_px=args.crop_radius_px,
        gaussian_kernel_px=args.gaussian_kernel_px,
        gaussian_sigma_px=args.gaussian_sigma_px,
        closing_kernel_px=args.closing_kernel_px,
        minimum_contour_points=args.minimum_contour_points,
        minimum_contour_area_px2=args.minimum_contour_area_px2,
        minimum_minor_axis_px=args.minimum_minor_axis_px,
        maximum_major_axis_px=args.maximum_major_axis_px,
        maximum_center_offset_px=args.maximum_center_offset_px,
        minimum_solidity=args.minimum_solidity,
        maximum_residual_p90=args.maximum_residual_p90,
        candidate_score_solidity_reference=(
            args.candidate_score_solidity_reference
        ),
        candidate_score_shape_weight=args.candidate_score_shape_weight,
    )


def build_loss_config(args: argparse.Namespace) -> LossConfig:
    return LossConfig(
        weights=LossWeights(
            coverage=args.k1_coverage,
            central_absolute_error=args.k2_central_error,
            tail_error=args.k3_tail_error,
            signed_bias=args.k4_signed_bias,
            stability_error=args.k5_stability,
            low_angle=args.k6_low_angle,
        ),
        perturbation_offsets_xy=args.perturbation_offsets,
        minimum_passing_perturbations=args.minimum_passing_perturbations,
        stability_failure_penalty_deg=args.stability_failure_penalty_deg,
        tail_percentiles=args.tail_percentiles,
        tail_percentile_weights=args.tail_percentile_weights,
        low_angle_threshold_deg=args.low_angle_threshold_deg,
        allowed_low_angle_fraction=args.allowed_low_angle_fraction,
        minimum_robust_paired_coverage_fraction=(
            args.minimum_robust_paired_coverage_fraction
        ),
        weights_are_provisional=not args.weights_final,
    )


def perturbed_angles(
    source_image: np.ndarray,
    local_x: float,
    local_y: float,
    fit_config: FitConfig,
    offsets: Sequence[tuple[int, int]],
) -> tuple[float | None, ...]:
    """Return accepted perturbed angles; a QC failure is represented by None."""

    output: list[float | None] = []
    for offset_x, offset_y in offsets:
        result = fit_etch_pit(
            source_image,
            local_x + offset_x,
            local_y + offset_y,
            fit_config,
        )
        output.append(
            float(result.incident_angle_deg) if result.qc_passed else None
        )
    return tuple(output)


def build_loss_inputs(
    rows: Sequence[dict[str, object]],
    images: dict[str, np.ndarray],
    nominal_fits: dict[str, Sequence[FitResult]],
    fit_config: FitConfig,
    loss_config: LossConfig,
) -> list[PitLossInput]:
    inputs: list[PitLossInput] = []
    for index, row in enumerate(rows):
        surfaces: dict[str, SurfaceLossInput] = {}
        for surface in ("C1", "C2"):
            nominal = nominal_fits[surface][index]
            if nominal.qc_passed:
                key = surface.lower()
                image_index = int(row[f"{key}_image"]) - FIRST_IMAGE_NUMBER
                source_image = images[surface][image_index]
                perturbations = perturbed_angles(
                    source_image,
                    float(row[f"{key}_local_x"]),
                    float(row[f"{key}_local_y"]),
                    fit_config,
                    loss_config.perturbation_offsets_xy,
                )
                nominal_angle = float(nominal.incident_angle_deg)
            else:
                perturbations = ()
                nominal_angle = (
                    None
                    if nominal.incident_angle_deg is None
                    else float(nominal.incident_angle_deg)
                )
            surfaces[surface] = SurfaceLossInput(
                nominal_qc_passed=nominal.qc_passed,
                nominal_angle_deg=nominal_angle,
                perturbed_angles_deg=perturbations,
            )
        inputs.append(
            PitLossInput(
                clean_id=str(row["clean_id"]),
                c1=surfaces["C1"],
                c2=surfaces["C2"],
            )
        )
    return inputs


def _format_float(value: float | None) -> str:
    return "" if value is None else f"{value:.9f}"


def write_per_pit_csv(
    path: Path,
    rows: Sequence[dict[str, object]],
    nominal_fits: dict[str, Sequence[FitResult]],
    results: Sequence[PitLossResult],
) -> None:
    records: list[dict[str, object]] = []
    for index, (row, result) in enumerate(zip(rows, results)):
        record: dict[str, object] = {
            "clean_id": result.clean_id,
            "robust_paired_passed": result.robust_paired_passed,
            "signed_error_deg": _format_float(result.signed_error_deg),
            "absolute_error_deg": _format_float(result.absolute_error_deg),
            "paired_angle_deg": _format_float(result.paired_angle_deg),
            "below_low_angle_threshold": (
                ""
                if result.below_low_angle_threshold is None
                else result.below_low_angle_threshold
            ),
            "mean_stability_error_deg": _format_float(
                result.mean_stability_error_deg
            ),
        }
        for surface in ("C1", "C2"):
            key = surface.lower()
            stability = getattr(result, key)
            nominal = nominal_fits[surface][index]
            record.update(
                {
                    f"{key}_image": row[f"{key}_image"],
                    f"{key}_local_x": row[f"{key}_local_x"],
                    f"{key}_local_y": row[f"{key}_local_y"],
                    f"{key}_crop_crosses_source_boundary": not bool(
                        np.all(nominal.crop.valid_pixels)
                    ),
                    f"{key}_nominal_qc_passed": stability.nominal_qc_passed,
                    f"{key}_nominal_failed_qc_codes": "+".join(
                        nominal.qc_failure_codes
                    ),
                    f"{key}_nominal_angle_deg": _format_float(
                        getattr(result, f"{key}_nominal_angle_deg")
                    ),
                    f"{key}_perturbation_pass_count": (
                        stability.perturbation_pass_count
                    ),
                    f"{key}_perturbation_fail_count": (
                        stability.perturbation_fail_count
                    ),
                    f"{key}_robust_qc_passed": stability.robust_qc_passed,
                    f"{key}_mean_stability_error_deg": _format_float(
                        stability.mean_stability_error_deg
                    ),
                }
            )
        records.append(record)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main() -> int:
    args = parse_args()
    fit_config = build_fit_config(args)
    loss_config = build_loss_config(args)
    rows = load_manifest(args.input.resolve())
    validate_manifest_truth(rows)
    images = {
        "C1": np.load(C1_IMAGE_PATH, mmap_mode="r"),
        "C2": np.load(C2_IMAGE_PATH, mmap_mode="r"),
    }
    nominal_fits = run_fits(rows, images, fit_config)
    loss_inputs = build_loss_inputs(
        rows, images, nominal_fits, fit_config, loss_config
    )
    evaluation, pit_results = evaluate_loss(loss_inputs, loss_config)

    summary = {
        "method": "robust paired ellipse incident-angle configuration loss",
        "fit_config": asdict(fit_config),
        "loss_config": asdict(loss_config),
        "definitions": {
            "robust_surface_pass": (
                "The nominal fit passes QC and at least the configured minimum "
                "number of coordinate perturbations pass QC."
            ),
            "robust_paired_pass": "Both C1 and C2 are robust surface passes.",
            "signed_error_deg": "C2 nominal angle minus C1 nominal angle.",
            "paired_angle_deg": "Mean of the nominal C1 and C2 angles.",
            "failed_perturbation_handling": (
                "If the perturbation majority gate passes, each failed "
                "perturbation contributes stability_failure_penalty_deg. If the "
                "gate fails, the surface is a robust QC failure and does not "
                "enter the remaining loss terms."
            ),
        },
        "evaluation": evaluation,
    }
    summary_output = args.summary_output.resolve()
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    per_pit_output = args.per_pit_output.resolve()
    write_per_pit_csv(per_pit_output, rows, nominal_fits, pit_results)

    print(f"Evaluated {len(rows)} clean IDs")
    print(
        f"Nominal paired passes: {evaluation['nominal_paired_pass_count']}"
    )
    print(
        f"Robust paired passes: {evaluation['robust_paired_pass_count']} "
        f"({evaluation['robust_paired_coverage_fraction']:.6f})"
    )
    for name, value in evaluation["raw_loss_terms"].items():
        formatted = "undefined" if value is None else f"{value:.9f}"
        print(f"{name}: {formatted}")
    total = evaluation["provisional_weighted_total_loss"]
    print(
        "Provisional weighted total: "
        + ("undefined" if total is None else f"{total:.9f}")
    )
    print(f"Summary: {summary_output}")
    print(f"Per-pit audit: {per_pit_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
