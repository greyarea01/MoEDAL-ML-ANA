"""Configurable loss for selecting an ellipse incident-angle configuration.

This module contains no image I/O.  It accepts nominal and coordinate-jittered
angle results, applies the robustness rule, and returns auditable loss terms.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Sequence

import numpy as np


DEFAULT_PERTURBATION_OFFSETS: tuple[tuple[int, int], ...] = (
    (-1, -1),
    (0, -1),
    (1, -1),
    (-1, 0),
    (1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
)


@dataclass(frozen=True)
class LossWeights:
    """Weights for the six nonnegative loss terms.

    Unit weights are placeholders until the analysis chooses physics-based
    priorities.  Raw and weighted terms are always reported separately.
    """

    coverage: float = 4.0
    central_absolute_error: float = 1.0
    tail_error: float = 0.8
    signed_bias: float = 2.0
    stability_error: float = 2.0
    low_angle: float = 4.0

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"Loss weight {name} must be finite and nonnegative")


@dataclass(frozen=True)
class LossConfig:
    """Configuration for robust-pair selection and loss calculation."""

    weights: LossWeights = field(default_factory=LossWeights)
    perturbation_offsets_xy: tuple[tuple[int, int], ...] = (
        DEFAULT_PERTURBATION_OFFSETS
    )
    minimum_passing_perturbations: int = 5
    stability_failure_penalty_deg: float = 10.0
    tail_percentiles: tuple[float, ...] = (90.0, 95.0, 99.0)
    tail_percentile_weights: tuple[float, ...] = (1.0, 1.0, 0.25)
    low_angle_threshold_deg: float = 5.0
    allowed_low_angle_fraction: float = 0.0
    minimum_robust_paired_coverage_fraction: float | None = None
    weights_are_provisional: bool = True

    def __post_init__(self) -> None:
        offsets = tuple(tuple(offset) for offset in self.perturbation_offsets_xy)
        if not offsets:
            raise ValueError("At least one perturbation offset is required")
        if len(set(offsets)) != len(offsets):
            raise ValueError("Perturbation offsets must be unique")
        if any(len(offset) != 2 for offset in offsets):
            raise ValueError("Each perturbation offset must contain x and y")
        if (0, 0) in offsets:
            raise ValueError("The nominal offset (0, 0) is not a perturbation")
        if not 1 <= self.minimum_passing_perturbations <= len(offsets):
            raise ValueError(
                "minimum_passing_perturbations must be between one and the "
                "number of perturbation offsets"
            )
        if self.minimum_passing_perturbations <= len(offsets) / 2:
            raise ValueError(
                "minimum_passing_perturbations must be a strict majority of "
                "the perturbation offsets"
            )
        if not np.isfinite(self.stability_failure_penalty_deg) or (
            self.stability_failure_penalty_deg < 0
        ):
            raise ValueError(
                "stability_failure_penalty_deg must be finite and nonnegative"
            )
        if len(self.tail_percentiles) != len(self.tail_percentile_weights):
            raise ValueError("Every tail percentile must have one weight")
        if not self.tail_percentiles:
            raise ValueError("At least one tail percentile is required")
        if any(not 0 <= value <= 100 for value in self.tail_percentiles):
            raise ValueError("Tail percentiles must be between zero and 100")
        if any(
            not np.isfinite(value) or value < 0
            for value in self.tail_percentile_weights
        ):
            raise ValueError("Tail percentile weights must be finite and nonnegative")
        if sum(self.tail_percentile_weights) <= 0:
            raise ValueError("At least one tail percentile weight must be positive")
        if not np.isfinite(self.low_angle_threshold_deg) or (
            self.low_angle_threshold_deg < 0
        ):
            raise ValueError("low_angle_threshold_deg must be finite and nonnegative")
        if not 0 <= self.allowed_low_angle_fraction <= 1:
            raise ValueError("allowed_low_angle_fraction must be between zero and one")
        minimum_coverage = self.minimum_robust_paired_coverage_fraction
        if minimum_coverage is not None and not 0 <= minimum_coverage <= 1:
            raise ValueError(
                "minimum_robust_paired_coverage_fraction must be between zero and one"
            )


@dataclass(frozen=True)
class SurfaceLossInput:
    """Nominal result and passing perturbed angles for one pit surface.

    Each perturbation entry is its accepted angle in degrees.  ``None`` means
    that the perturbed fit failed at least one QC cut.
    """

    nominal_qc_passed: bool
    nominal_angle_deg: float | None
    perturbed_angles_deg: tuple[float | None, ...] = ()


@dataclass(frozen=True)
class PitLossInput:
    clean_id: str
    c1: SurfaceLossInput
    c2: SurfaceLossInput


@dataclass(frozen=True)
class SurfaceStabilityResult:
    nominal_qc_passed: bool
    perturbation_pass_count: int
    perturbation_fail_count: int
    robust_qc_passed: bool
    mean_stability_error_deg: float | None


@dataclass(frozen=True)
class PitLossResult:
    clean_id: str
    c1: SurfaceStabilityResult
    c2: SurfaceStabilityResult
    robust_paired_passed: bool
    c1_nominal_angle_deg: float | None
    c2_nominal_angle_deg: float | None
    signed_error_deg: float | None
    absolute_error_deg: float | None
    paired_angle_deg: float | None
    below_low_angle_threshold: bool | None
    mean_stability_error_deg: float | None


def evaluate_surface_stability(
    surface: SurfaceLossInput,
    config: LossConfig,
) -> SurfaceStabilityResult:
    """Apply the perturbation-majority gate and calculate stability cost."""

    if not surface.nominal_qc_passed:
        return SurfaceStabilityResult(
            nominal_qc_passed=False,
            perturbation_pass_count=0,
            perturbation_fail_count=0,
            robust_qc_passed=False,
            mean_stability_error_deg=None,
        )
    if surface.nominal_angle_deg is None or not np.isfinite(surface.nominal_angle_deg):
        raise ValueError("A nominal QC pass must have a finite incident angle")
    if len(surface.perturbed_angles_deg) != len(config.perturbation_offsets_xy):
        raise ValueError(
            "A nominal QC pass must have one result for every perturbation offset"
        )

    passing_angles = [
        float(angle) for angle in surface.perturbed_angles_deg if angle is not None
    ]
    if any(not np.isfinite(angle) for angle in passing_angles):
        raise ValueError("Passing perturbation angles must be finite")
    pass_count = len(passing_angles)
    fail_count = len(surface.perturbed_angles_deg) - pass_count
    robust_passed = pass_count >= config.minimum_passing_perturbations
    if not robust_passed:
        return SurfaceStabilityResult(
            nominal_qc_passed=True,
            perturbation_pass_count=pass_count,
            perturbation_fail_count=fail_count,
            robust_qc_passed=False,
            mean_stability_error_deg=None,
        )

    costs = [
        abs(angle - float(surface.nominal_angle_deg)) for angle in passing_angles
    ]
    costs.extend([config.stability_failure_penalty_deg] * fail_count)
    return SurfaceStabilityResult(
        nominal_qc_passed=True,
        perturbation_pass_count=pass_count,
        perturbation_fail_count=fail_count,
        robust_qc_passed=True,
        mean_stability_error_deg=float(np.mean(costs)),
    )


def _optional_float(value: float | None) -> float | None:
    return None if value is None else float(value)


def evaluate_loss(
    pits: Sequence[PitLossInput],
    config: LossConfig | None = None,
) -> tuple[dict[str, object], list[PitLossResult]]:
    """Calculate all six loss terms and per-pit audit results."""

    config = config or LossConfig()
    if not pits:
        raise ValueError("At least one pit is required")
    clean_ids = [pit.clean_id for pit in pits]
    if len(clean_ids) != len(set(clean_ids)):
        raise ValueError("Pit clean IDs must be unique")

    pit_results: list[PitLossResult] = []
    for pit in pits:
        c1 = evaluate_surface_stability(pit.c1, config)
        c2 = evaluate_surface_stability(pit.c2, config)
        robust_pair = c1.robust_qc_passed and c2.robust_qc_passed

        signed_error: float | None = None
        absolute_error: float | None = None
        paired_angle: float | None = None
        low_angle: bool | None = None
        stability_error: float | None = None
        if robust_pair:
            c1_angle = float(pit.c1.nominal_angle_deg)
            c2_angle = float(pit.c2.nominal_angle_deg)
            signed_error = c2_angle - c1_angle
            absolute_error = abs(signed_error)
            paired_angle = (c1_angle + c2_angle) / 2.0
            low_angle = paired_angle < config.low_angle_threshold_deg
            stability_error = float(
                np.mean(
                    [
                        c1.mean_stability_error_deg,
                        c2.mean_stability_error_deg,
                    ]
                )
            )

        pit_results.append(
            PitLossResult(
                clean_id=pit.clean_id,
                c1=c1,
                c2=c2,
                robust_paired_passed=robust_pair,
                c1_nominal_angle_deg=_optional_float(pit.c1.nominal_angle_deg),
                c2_nominal_angle_deg=_optional_float(pit.c2.nominal_angle_deg),
                signed_error_deg=signed_error,
                absolute_error_deg=absolute_error,
                paired_angle_deg=paired_angle,
                below_low_angle_threshold=low_angle,
                mean_stability_error_deg=stability_error,
            )
        )

    robust = [result for result in pit_results if result.robust_paired_passed]
    robust_count = len(robust)
    total_count = len(pit_results)
    robust_coverage = robust_count / total_count
    coverage_penalty = 1.0 - robust_coverage
    minimum_coverage = config.minimum_robust_paired_coverage_fraction
    meets_minimum_coverage = (
        minimum_coverage is None or robust_coverage >= minimum_coverage
    )

    terms: dict[str, float | None] = {
        "coverage_penalty": coverage_penalty,
        "central_absolute_error_deg": None,
        "tail_error_deg": None,
        "signed_bias_deg": None,
        "mean_stability_error_deg": None,
        "low_angle_penalty": None,
    }
    tail_values: dict[str, float] = {}
    raw_statistics: dict[str, float | None] = {
        "mean_signed_error_deg": None,
        "mean_absolute_error_deg": None,
        "low_angle_fraction": None,
    }

    if robust:
        signed_errors = np.asarray(
            [result.signed_error_deg for result in robust], dtype=float
        )
        absolute_errors = np.abs(signed_errors)
        stability_errors = np.asarray(
            [result.mean_stability_error_deg for result in robust], dtype=float
        )
        low_angle_flags = np.asarray(
            [result.below_low_angle_threshold for result in robust], dtype=bool
        )
        percentile_values = np.percentile(
            absolute_errors, config.tail_percentiles
        )
        tail_values = {
            f"p{percentile:g}_absolute_error_deg": float(value)
            for percentile, value in zip(config.tail_percentiles, percentile_values)
        }
        tail_error = float(
            np.average(percentile_values, weights=config.tail_percentile_weights)
        )
        mean_signed_error = float(np.mean(signed_errors))
        mean_absolute_error = float(np.mean(absolute_errors))
        low_angle_fraction = float(np.mean(low_angle_flags))
        terms.update(
            {
                "central_absolute_error_deg": mean_absolute_error,
                "tail_error_deg": tail_error,
                "signed_bias_deg": abs(mean_signed_error),
                "mean_stability_error_deg": float(np.mean(stability_errors)),
                "low_angle_penalty": max(
                    0.0,
                    low_angle_fraction - config.allowed_low_angle_fraction,
                ),
            }
        )
        raw_statistics.update(
            {
                "mean_signed_error_deg": mean_signed_error,
                "mean_absolute_error_deg": mean_absolute_error,
                "low_angle_fraction": low_angle_fraction,
            }
        )

    weighted_terms: dict[str, float | None] = {}
    weight_mapping = {
        "coverage_penalty": config.weights.coverage,
        "central_absolute_error_deg": config.weights.central_absolute_error,
        "tail_error_deg": config.weights.tail_error,
        "signed_bias_deg": config.weights.signed_bias,
        "mean_stability_error_deg": config.weights.stability_error,
        "low_angle_penalty": config.weights.low_angle,
    }
    for name, value in terms.items():
        weighted_terms[name] = (
            None if value is None else float(value * weight_mapping[name])
        )

    loss_is_valid = bool(robust and meets_minimum_coverage)
    provisional_total = (
        float(sum(value for value in weighted_terms.values() if value is not None))
        if loss_is_valid
        else None
    )
    summary: dict[str, object] = {
        "total_clean_pit_count": total_count,
        "nominal_c1_pass_count": sum(result.c1.nominal_qc_passed for result in pit_results),
        "nominal_c2_pass_count": sum(result.c2.nominal_qc_passed for result in pit_results),
        "nominal_paired_pass_count": sum(
            result.c1.nominal_qc_passed and result.c2.nominal_qc_passed
            for result in pit_results
        ),
        "robust_c1_pass_count": sum(result.c1.robust_qc_passed for result in pit_results),
        "robust_c2_pass_count": sum(result.c2.robust_qc_passed for result in pit_results),
        "robust_paired_pass_count": robust_count,
        "robust_paired_coverage_fraction": robust_coverage,
        "nominal_pairs_rejected_by_perturbation_majority_count": sum(
            result.c1.nominal_qc_passed
            and result.c2.nominal_qc_passed
            and not result.robust_paired_passed
            for result in pit_results
        ),
        "robust_pair_perturbation_pass_count": sum(
            result.c1.perturbation_pass_count + result.c2.perturbation_pass_count
            for result in robust
        ),
        "robust_pair_perturbation_failure_penalty_count": sum(
            result.c1.perturbation_fail_count + result.c2.perturbation_fail_count
            for result in robust
        ),
        "robust_pair_low_angle_count": sum(
            result.below_low_angle_threshold is True for result in robust
        ),
        "minimum_robust_paired_coverage_fraction": minimum_coverage,
        "meets_minimum_coverage": meets_minimum_coverage,
        "loss_is_valid": loss_is_valid,
        "raw_loss_terms": terms,
        "tail_percentile_values": tail_values,
        "supporting_statistics": raw_statistics,
        "weighted_loss_terms": weighted_terms,
        "provisional_weighted_total_loss": provisional_total,
        "weights_are_provisional": config.weights_are_provisional,
    }
    return summary, pit_results
