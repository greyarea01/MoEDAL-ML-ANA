#!/usr/bin/env python3
"""Adaptively search Gaussian kernel and sigma using the robust loss."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


SCRIPT_DIR = Path(__file__).resolve().parent
SURVEY_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from ellipse_fit import FitConfig  # noqa: E402
from evaluate_loss import build_loss_inputs  # noqa: E402
from loss_function import LossConfig, evaluate_loss  # noqa: E402
from run_pipeline import (  # noqa: E402
    C1_IMAGE_PATH,
    C2_IMAGE_PATH,
    DEFAULT_INPUT_PATH,
    load_manifest,
    run_fits,
    validate_manifest_truth,
)


DEFAULT_CSV_OUTPUT = SCRIPT_DIR / "data/gaussian_phase_space_search.csv"
DEFAULT_JSON_OUTPUT = SCRIPT_DIR / "data/gaussian_phase_space_search.json"
DEFAULT_REPORT_OUTPUT = SCRIPT_DIR / "reports/gaussian_phase_space_search.pdf"


@dataclass(frozen=True)
class SearchConfig:
    """Adaptive-neighborhood search controls."""

    requested_kernel_step_px: int = 1
    sigma_step_px: float = 0.2
    minimum_kernel_px: int = 1
    maximum_kernel_px: int = 49
    minimum_sigma_px: float = 0.1
    maximum_sigma_px: float = 5.1
    maximum_iterations: int = 12
    minimum_loss_improvement: float = 1e-9

    def __post_init__(self) -> None:
        if self.requested_kernel_step_px <= 0:
            raise ValueError("requested_kernel_step_px must be positive")
        if self.sigma_step_px <= 0:
            raise ValueError("sigma_step_px must be positive")
        if self.minimum_kernel_px <= 0:
            raise ValueError("minimum_kernel_px must be positive")
        if self.maximum_kernel_px < self.minimum_kernel_px:
            raise ValueError("maximum_kernel_px must not be below the minimum")
        if self.minimum_sigma_px <= 0:
            raise ValueError("minimum_sigma_px must be positive")
        if self.maximum_sigma_px < self.minimum_sigma_px:
            raise ValueError("maximum_sigma_px must not be below the minimum")
        if self.maximum_iterations <= 0:
            raise ValueError("maximum_iterations must be positive")
        if self.minimum_loss_improvement < 0:
            raise ValueError("minimum_loss_improvement must be nonnegative")


@dataclass(frozen=True, order=True)
class GaussianPoint:
    kernel_px: int
    sigma_px: float


def _rounded_sigma(value: float) -> float:
    return round(float(value), 9)


def _adjacent_valid_kernel(
    kernel_px: int,
    direction: int,
    config: SearchConfig,
) -> tuple[int | None, list[int]]:
    """Walk by the requested step until OpenCV's next odd kernel is reached."""

    proposal = kernel_px + direction * config.requested_kernel_step_px
    skipped: list[int] = []
    while config.minimum_kernel_px <= proposal <= config.maximum_kernel_px:
        if proposal % 2 == 1:
            return proposal, skipped
        skipped.append(proposal)
        proposal += direction * config.requested_kernel_step_px
    return None, skipped


def neighborhood(
    center: GaussianPoint,
    config: SearchConfig,
) -> tuple[set[GaussianPoint], list[int]]:
    """Return the valid local 2D neighborhood and skipped even kernels."""

    lower_kernel, lower_skipped = _adjacent_valid_kernel(center.kernel_px, -1, config)
    upper_kernel, upper_skipped = _adjacent_valid_kernel(center.kernel_px, 1, config)
    kernels = [value for value in (lower_kernel, center.kernel_px, upper_kernel) if value]
    sigmas = [
        _rounded_sigma(center.sigma_px - config.sigma_step_px),
        _rounded_sigma(center.sigma_px),
        _rounded_sigma(center.sigma_px + config.sigma_step_px),
    ]
    sigmas = [
        value
        for value in sigmas
        if config.minimum_sigma_px <= value <= config.maximum_sigma_px
    ]
    points = {
        GaussianPoint(kernel_px=kernel, sigma_px=sigma)
        for kernel in kernels
        for sigma in sigmas
    }
    return points, lower_skipped + upper_skipped


def adaptive_search(
    start: GaussianPoint,
    config: SearchConfig,
    evaluator: Callable[[GaussianPoint, int], dict[str, object]],
) -> tuple[
    dict[GaussianPoint, dict[str, object]],
    list[GaussianPoint],
    list[dict[str, object]],
    str,
]:
    """Move to the best local neighbor until reaching a local minimum."""

    if start.kernel_px % 2 != 1:
        raise ValueError("The starting Gaussian kernel must be odd")
    if not config.minimum_kernel_px <= start.kernel_px <= config.maximum_kernel_px:
        raise ValueError("The starting kernel is outside the search bounds")
    if not config.minimum_sigma_px <= start.sigma_px <= config.maximum_sigma_px:
        raise ValueError("The starting sigma is outside the search bounds")

    evaluated: dict[GaussianPoint, dict[str, object]] = {}
    path = [start]
    skipped_proposals: list[dict[str, object]] = []
    center = start
    stop_reason = "maximum_iterations_reached"

    for iteration in range(config.maximum_iterations):
        candidates, skipped_kernels = neighborhood(center, config)
        for kernel in skipped_kernels:
            skipped_proposals.append(
                {
                    "iteration": iteration,
                    "center_kernel_px": center.kernel_px,
                    "center_sigma_px": center.sigma_px,
                    "skipped_kernel_px": kernel,
                    "reason": "OpenCV Gaussian kernels must be positive odd integers",
                }
            )
        for point in sorted(candidates):
            if point not in evaluated:
                evaluated[point] = evaluator(point, iteration)

        current_loss = evaluated[center]["provisional_weighted_total_loss"]
        if current_loss is None:
            raise RuntimeError(f"Loss is undefined at search center {center}")
        valid_candidates = [
            point
            for point in candidates
            if evaluated[point]["provisional_weighted_total_loss"] is not None
        ]
        best = min(
            valid_candidates,
            key=lambda point: (
                float(evaluated[point]["provisional_weighted_total_loss"]),
                point != center,
                point.kernel_px,
                point.sigma_px,
            ),
        )
        best_loss = float(evaluated[best]["provisional_weighted_total_loss"])
        improvement = float(current_loss) - best_loss
        if best == center or improvement <= config.minimum_loss_improvement:
            stop_reason = "local_minimum_reached"
            break
        center = best
        path.append(center)

    return evaluated, path, skipped_proposals, stop_reason


def flatten_evaluation(
    point: GaussianPoint,
    first_iteration: int,
    evaluation: dict[str, object],
) -> dict[str, object]:
    raw = evaluation["raw_loss_terms"]
    weighted = evaluation["weighted_loss_terms"]
    tail = evaluation["tail_percentile_values"]
    supporting = evaluation["supporting_statistics"]
    return {
        "gaussian_kernel_px": point.kernel_px,
        "gaussian_sigma_px": point.sigma_px,
        "first_evaluated_iteration": first_iteration,
        "nominal_paired_pass_count": evaluation["nominal_paired_pass_count"],
        "robust_paired_pass_count": evaluation["robust_paired_pass_count"],
        "robust_paired_coverage_fraction": evaluation[
            "robust_paired_coverage_fraction"
        ],
        "nominal_pairs_rejected_by_perturbation_majority_count": evaluation[
            "nominal_pairs_rejected_by_perturbation_majority_count"
        ],
        "coverage_penalty_raw": raw["coverage_penalty"],
        "coverage_contribution": weighted["coverage_penalty"],
        "central_absolute_error_raw_deg": raw["central_absolute_error_deg"],
        "central_absolute_error_contribution": weighted[
            "central_absolute_error_deg"
        ],
        "p90_absolute_error_deg": tail.get("p90_absolute_error_deg"),
        "p95_absolute_error_deg": tail.get("p95_absolute_error_deg"),
        "p99_absolute_error_deg": tail.get("p99_absolute_error_deg"),
        "tail_error_raw_deg": raw["tail_error_deg"],
        "tail_error_contribution": weighted["tail_error_deg"],
        "mean_signed_error_deg": supporting["mean_signed_error_deg"],
        "signed_bias_raw_deg": raw["signed_bias_deg"],
        "signed_bias_contribution": weighted["signed_bias_deg"],
        "mean_stability_error_raw_deg": raw["mean_stability_error_deg"],
        "mean_stability_error_contribution": weighted[
            "mean_stability_error_deg"
        ],
        "low_angle_fraction": supporting["low_angle_fraction"],
        "low_angle_penalty_raw": raw["low_angle_penalty"],
        "low_angle_contribution": weighted["low_angle_penalty"],
        "provisional_weighted_total_loss": evaluation[
            "provisional_weighted_total_loss"
        ],
    }


def write_csv(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def generate_report(
    path: Path,
    records: list[dict[str, object]],
    search_path: list[GaussianPoint],
    fit_config: FitConfig,
    loss_config: LossConfig,
    stop_reason: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(records, key=lambda row: row["provisional_weighted_total_loss"])
    best = ordered[0]
    with PdfPages(path) as pdf:
        fig, ax = plt.subplots(figsize=(11, 8.5))
        sigmas = np.asarray([row["gaussian_sigma_px"] for row in records], dtype=float)
        kernels = np.asarray([row["gaussian_kernel_px"] for row in records], dtype=float)
        losses = np.asarray(
            [row["provisional_weighted_total_loss"] for row in records], dtype=float
        )
        points = ax.scatter(
            sigmas,
            kernels,
            c=losses,
            cmap="viridis_r",
            s=145,
            edgecolor="black",
            linewidth=0.6,
        )
        for sigma, kernel, loss in zip(sigmas, kernels, losses):
            ax.annotate(
                f"{loss:.2f}",
                (sigma, kernel),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
                fontsize=7,
            )
        path_sigmas = [point.sigma_px for point in search_path]
        path_kernels = [point.kernel_px for point in search_path]
        ax.plot(path_sigmas, path_kernels, color="#d62728", linewidth=2, marker="o")
        ax.scatter(
            [best["gaussian_sigma_px"]],
            [best["gaussian_kernel_px"]],
            marker="*",
            s=400,
            color="#ffd700",
            edgecolor="black",
            label="Lowest evaluated loss",
            zorder=5,
        )
        fig.colorbar(points, ax=ax, label="Weighted total loss")
        ax.set_xlabel("Gaussian sigma in pixels")
        ax.set_ylabel("Gaussian kernel size in pixels")
        ax.set_yticks(sorted(set(int(value) for value in kernels)))
        ax.grid(alpha=0.2)
        ax.legend(loc="best")
        ax.set_title("Adaptive Gaussian phase-space search")
        fig.text(
            0.5,
            0.02,
            f"Start: kernel {fit_config.gaussian_kernel_px}, sigma {fit_config.gaussian_sigma_px:g}. "
            f"Stop: {stop_reason.replace('_', ' ')}. Evaluated {len(records)} valid configurations.",
            ha="center",
            fontsize=9,
        )
        fig.tight_layout(rect=(0.03, 0.05, 0.98, 0.96))
        pdf.savefig(fig)
        plt.close(fig)

        top = ordered[: min(15, len(ordered))]
        labels = [
            f"k={row['gaussian_kernel_px']}, s={row['gaussian_sigma_px']:g}"
            for row in top
        ]
        component_keys = [
            ("coverage_contribution", "coverage"),
            ("central_absolute_error_contribution", "central error"),
            ("tail_error_contribution", "tail"),
            ("signed_bias_contribution", "signed bias"),
            ("mean_stability_error_contribution", "stability"),
            ("low_angle_contribution", "low angle"),
        ]
        fig, ax = plt.subplots(figsize=(11, 8.5))
        left = np.zeros(len(top), dtype=float)
        y = np.arange(len(top))
        for key, label in component_keys:
            values = np.asarray([row[key] for row in top], dtype=float)
            ax.barh(y, values, left=left, label=label)
            left += values
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.set_xlabel("Weighted loss contribution")
        ax.set_title("Lowest-loss evaluated configurations")
        ax.legend(loc="lower right")
        ax.grid(axis="x", alpha=0.2)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.06, 0.93, "Gaussian phase-space search summary", fontsize=20, weight="bold")
        lines = [
            f"Lowest evaluated loss: {best['provisional_weighted_total_loss']:.6f}",
            f"Best kernel: {best['gaussian_kernel_px']} pixels",
            f"Best sigma: {best['gaussian_sigma_px']:.3f} pixels",
            f"Robust paired coverage: {100 * best['robust_paired_coverage_fraction']:.3f} percent",
            f"Robust paired passes: {best['robust_paired_pass_count']}",
            f"Mean absolute error: {best['central_absolute_error_raw_deg']:.3f} degrees",
            f"Tail error: {best['tail_error_raw_deg']:.3f} degrees",
            f"Signed bias: {best['signed_bias_raw_deg']:.3f} degrees",
            f"Mean stability error: {best['mean_stability_error_raw_deg']:.3f} degrees",
            f"Low-angle fraction: {100 * best['low_angle_fraction']:.3f} percent",
            "",
            "Search path: "
            + " -> ".join(
                f"({point.kernel_px}, {point.sigma_px:g})" for point in search_path
            ),
            f"Stop reason: {stop_reason.replace('_', ' ')}",
            f"Evaluated valid configurations: {len(records)}",
            "",
            "Loss weights: "
            f"coverage={loss_config.weights.coverage:g}, "
            f"central error={loss_config.weights.central_absolute_error:g}, "
            f"tail={loss_config.weights.tail_error:g}",
            "              "
            f"signed bias={loss_config.weights.signed_bias:g}, "
            f"stability={loss_config.weights.stability_error:g}, "
            f"low angle={loss_config.weights.low_angle:g}",
        ]
        y_pos = 0.86
        for line in lines:
            fig.text(0.07, y_pos, line, fontsize=12)
            y_pos -= 0.047
        pdf.savefig(fig)
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    fit_defaults = FitConfig()
    search_defaults = SearchConfig(maximum_kernel_px=fit_defaults.crop_size_px)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--start-kernel-px", type=int, default=fit_defaults.gaussian_kernel_px)
    parser.add_argument("--start-sigma-px", type=float, default=fit_defaults.gaussian_sigma_px)
    parser.add_argument(
        "--kernel-step-px", type=int, default=search_defaults.requested_kernel_step_px
    )
    parser.add_argument("--sigma-step-px", type=float, default=search_defaults.sigma_step_px)
    parser.add_argument("--minimum-kernel-px", type=int, default=search_defaults.minimum_kernel_px)
    parser.add_argument("--maximum-kernel-px", type=int, default=search_defaults.maximum_kernel_px)
    parser.add_argument("--minimum-sigma-px", type=float, default=search_defaults.minimum_sigma_px)
    parser.add_argument("--maximum-sigma-px", type=float, default=search_defaults.maximum_sigma_px)
    parser.add_argument("--maximum-iterations", type=int, default=search_defaults.maximum_iterations)
    parser.add_argument(
        "--minimum-loss-improvement",
        type=float,
        default=search_defaults.minimum_loss_improvement,
    )
    parser.add_argument("--csv-output", type=Path, default=DEFAULT_CSV_OUTPUT)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_fit_config = FitConfig(
        gaussian_kernel_px=args.start_kernel_px,
        gaussian_sigma_px=args.start_sigma_px,
    )
    loss_config = LossConfig()
    search_config = SearchConfig(
        requested_kernel_step_px=args.kernel_step_px,
        sigma_step_px=args.sigma_step_px,
        minimum_kernel_px=args.minimum_kernel_px,
        maximum_kernel_px=args.maximum_kernel_px,
        minimum_sigma_px=args.minimum_sigma_px,
        maximum_sigma_px=args.maximum_sigma_px,
        maximum_iterations=args.maximum_iterations,
        minimum_loss_improvement=args.minimum_loss_improvement,
    )
    rows = load_manifest(args.input.resolve())
    validate_manifest_truth(rows)
    images = {
        "C1": np.load(C1_IMAGE_PATH, mmap_mode="r"),
        "C2": np.load(C2_IMAGE_PATH, mmap_mode="r"),
    }
    first_iterations: dict[GaussianPoint, int] = {}

    def evaluator(point: GaussianPoint, iteration: int) -> dict[str, object]:
        print(
            f"Evaluating kernel {point.kernel_px}, sigma {point.sigma_px:g} "
            f"at search iteration {iteration}",
            flush=True,
        )
        fit_config = replace(
            base_fit_config,
            gaussian_kernel_px=point.kernel_px,
            gaussian_sigma_px=point.sigma_px,
        )
        nominal_fits = run_fits(rows, images, fit_config)
        loss_inputs = build_loss_inputs(
            rows, images, nominal_fits, fit_config, loss_config
        )
        evaluation, _ = evaluate_loss(loss_inputs, loss_config)
        first_iterations[point] = iteration
        print(
            f"  loss {evaluation['provisional_weighted_total_loss']:.9f}; "
            f"robust pairs {evaluation['robust_paired_pass_count']}",
            flush=True,
        )
        return evaluation

    start = GaussianPoint(
        kernel_px=base_fit_config.gaussian_kernel_px,
        sigma_px=_rounded_sigma(base_fit_config.gaussian_sigma_px),
    )
    evaluated, search_path, skipped, stop_reason = adaptive_search(
        start, search_config, evaluator
    )
    records = [
        flatten_evaluation(point, first_iterations[point], evaluation)
        for point, evaluation in evaluated.items()
    ]
    records.sort(key=lambda row: (row["gaussian_kernel_px"], row["gaussian_sigma_px"]))
    best = min(records, key=lambda row: row["provisional_weighted_total_loss"])

    csv_output = args.csv_output.resolve()
    write_csv(csv_output, records)
    json_output = args.json_output.resolve()
    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(
        json.dumps(
            {
                "method": "adaptive local Gaussian phase-space search",
                "base_fit_config": asdict(base_fit_config),
                "loss_config": asdict(loss_config),
                "search_config": asdict(search_config),
                "opencv_kernel_constraint": (
                    "Gaussian kernel sizes must be positive odd integers. Even "
                    "one-pixel proposals are recorded but not evaluated."
                ),
                "stop_reason": stop_reason,
                "search_path": [asdict(point) for point in search_path],
                "skipped_invalid_kernel_proposals": skipped,
                "evaluated_configuration_count": len(records),
                "best": best,
                "evaluations": records,
            },
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    report_output = args.report_output.resolve()
    generate_report(
        report_output,
        records,
        search_path,
        base_fit_config,
        loss_config,
        stop_reason,
    )

    print(
        f"Best evaluated configuration: kernel {best['gaussian_kernel_px']}, "
        f"sigma {best['gaussian_sigma_px']:g}, "
        f"loss {best['provisional_weighted_total_loss']:.9f}"
    )
    print("Search path: " + " -> ".join(f"({p.kernel_px}, {p.sigma_px:g})" for p in search_path))
    print(f"Stop reason: {stop_reason}")
    print(f"Evaluated valid configurations: {len(records)}")
    print(f"CSV: {csv_output}")
    print(f"JSON: {json_output}")
    print(f"Report: {report_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
