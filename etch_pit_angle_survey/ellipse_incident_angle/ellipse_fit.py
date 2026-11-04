"""Ellipse fitting and quality control for a single etch-pit image crop."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass(frozen=True)
class FitConfig:
    crop_radius_px: int = 24
    gaussian_kernel_px: int = 5
    gaussian_sigma_px: float = 1.1
    closing_kernel_px: int = 3
    minimum_contour_points: int = 5
    minimum_contour_area_px2: float = 15.0
    minimum_minor_axis_px: float = 3.0
    maximum_major_axis_px: float = 47.0
    maximum_center_offset_px: float = 10.0
    minimum_solidity: float = 0.92
    maximum_residual_p90: float = 0.15
    candidate_score_solidity_reference: float = 0.90
    candidate_score_shape_weight: float = 8.0

    @property
    def crop_size_px(self) -> int:
        return 2 * self.crop_radius_px + 1


@dataclass
class CropData:
    image: np.ndarray
    valid_pixels: np.ndarray
    expected_center_xy: tuple[float, float]
    rounded_source_center_xy: tuple[int, int]


@dataclass
class FitResult:
    candidate_found: bool
    qc_passed: bool
    qc_failure_codes: list[str]
    candidate_failure_detail: str
    crop: CropData
    threshold_value: float | None = None
    segmentation_mask: np.ndarray | None = None
    contour: np.ndarray | None = None
    ellipse: tuple[tuple[float, float], tuple[float, float], float] | None = None
    contour_area_px2: float | None = None
    major_axis_px: float | None = None
    minor_axis_px: float | None = None
    major_axis_orientation_deg: float | None = None
    center_offset_px: float | None = None
    solidity: float | None = None
    ellipse_residual_p90: float | None = None
    touches_crop_boundary: bool | None = None
    touches_source_image_boundary: bool | None = None
    candidate_score: float | None = None
    incident_angle_deg: float | None = None
    candidate_count: int = 0
    contour_count: int = 0
    rejected_candidate_counts: dict[str, int] = field(default_factory=dict)


QC_DEFINITIONS = {
    "A": "A valid ellipse candidate must be available.",
    "B": "Ellipse center offset must be at most 10 pixels.",
    "C": "Contour solidity must be at least 0.92.",
    "D": "The 90th-percentile normalized ellipse residual must be at most 0.15.",
    "E": "The contour must not touch the 49 by 49 analysis-crop boundary.",
    "F": "The contour must not touch a real source-image boundary.",
}


def extract_valid_crop(
    source_image: np.ndarray,
    local_x: float,
    local_y: float,
    config: FitConfig,
) -> CropData:
    """Extract a fixed crop without inventing pixels beyond the source image.

    Invalid positions are filled white for display, accompanied by a validity
    mask, and excluded from all thresholding and contour calculations.
    """

    source = np.asarray(source_image)
    if source.ndim != 2:
        raise ValueError(f"Expected a two-dimensional grayscale image, got {source.shape}")

    radius = config.crop_radius_px
    size = config.crop_size_px
    center_x = int(round(float(local_x)))
    center_y = int(round(float(local_y)))
    requested_x0 = center_x - radius
    requested_y0 = center_y - radius
    requested_x1 = center_x + radius + 1
    requested_y1 = center_y + radius + 1

    source_y0 = max(0, requested_y0)
    source_x0 = max(0, requested_x0)
    source_y1 = min(source.shape[0], requested_y1)
    source_x1 = min(source.shape[1], requested_x1)

    crop = np.full((size, size), 255, dtype=source.dtype)
    valid = np.zeros((size, size), dtype=bool)
    if source_y0 < source_y1 and source_x0 < source_x1:
        target_y0 = source_y0 - requested_y0
        target_x0 = source_x0 - requested_x0
        target_y1 = target_y0 + source_y1 - source_y0
        target_x1 = target_x0 + source_x1 - source_x0
        crop[target_y0:target_y1, target_x0:target_x1] = source[
            source_y0:source_y1, source_x0:source_x1
        ]
        valid[target_y0:target_y1, target_x0:target_x1] = True

    return CropData(
        image=crop,
        valid_pixels=valid,
        expected_center_xy=(float(radius), float(radius)),
        rounded_source_center_xy=(center_x, center_y),
    )


def _valid_normalized_gaussian_blur(
    crop: np.ndarray,
    valid_pixels: np.ndarray,
    config: FitConfig,
) -> np.ndarray:
    """Blur using only observed pixels, with no reflected or replicated padding."""

    valid_float = valid_pixels.astype(np.float32)
    values = crop.astype(np.float32) * valid_float
    kernel = (config.gaussian_kernel_px, config.gaussian_kernel_px)
    weighted_sum = cv2.GaussianBlur(
        values,
        kernel,
        config.gaussian_sigma_px,
        borderType=cv2.BORDER_CONSTANT,
    )
    weight = cv2.GaussianBlur(
        valid_float,
        kernel,
        config.gaussian_sigma_px,
        borderType=cv2.BORDER_CONSTANT,
    )
    blurred = np.full(crop.shape, 255.0, dtype=np.float32)
    observed = valid_pixels & (weight > 1e-8)
    blurred[observed] = weighted_sum[observed] / weight[observed]
    return np.clip(np.rint(blurred), 0, 255).astype(np.uint8)


def _segment_dark_pixels(
    crop: CropData,
    config: FitConfig,
) -> tuple[np.ndarray, float | None]:
    if not np.any(crop.valid_pixels):
        return np.zeros(crop.image.shape, dtype=np.uint8), None

    blurred = _valid_normalized_gaussian_blur(
        crop.image, crop.valid_pixels, config
    )
    observed_values = blurred[crop.valid_pixels].reshape(-1, 1)
    threshold_value, _ = cv2.threshold(
        observed_values,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    mask = np.zeros(crop.image.shape, dtype=np.uint8)
    mask[crop.valid_pixels & (blurred <= threshold_value)] = 255
    return mask, float(threshold_value)


def _close_components_independently(
    threshold_mask: np.ndarray,
    valid_pixels: np.ndarray,
    config: FitConfig,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Close threshold components separately and retain separate contours.

    Closing the union of all threshold components can bridge a real pit to a
    physically disconnected scratch.  Processing each pre-closing component
    independently prevents such an artificial connection, including when two
    independently closed masks overlap.
    """

    kernel = np.ones(
        (config.closing_kernel_px, config.closing_kernel_px), dtype=np.uint8
    )
    component_count, component_labels = cv2.connectedComponents(
        threshold_mask, connectivity=8
    )
    combined_mask = np.zeros(threshold_mask.shape, dtype=np.uint8)
    contours: list[np.ndarray] = []
    for component_label in range(1, component_count):
        component_mask = np.where(
            component_labels == component_label, 255, 0
        ).astype(np.uint8)
        closed_component = cv2.morphologyEx(
            component_mask, cv2.MORPH_CLOSE, kernel
        )
        closed_component[~valid_pixels] = 0
        combined_mask = cv2.bitwise_or(combined_mask, closed_component)
        component_contours, _ = cv2.findContours(
            closed_component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        contours.extend(component_contours)
    return combined_mask, contours


def ellipse_residual_p90(
    contour: np.ndarray,
    ellipse: tuple[tuple[float, float], tuple[float, float], float],
) -> float:
    """Return the 90th percentile radial mismatch from the fitted ellipse."""

    (center_x, center_y), (diameter_a, diameter_b), angle_deg = ellipse
    points = contour[:, 0, :].astype(float)
    angle_rad = math.radians(angle_deg)
    cos_angle = math.cos(angle_rad)
    sin_angle = math.sin(angle_rad)
    delta_x = points[:, 0] - center_x
    delta_y = points[:, 1] - center_y
    ellipse_x = cos_angle * delta_x + sin_angle * delta_y
    ellipse_y = -sin_angle * delta_x + cos_angle * delta_y
    normalized_radius = np.sqrt(
        (ellipse_x / (diameter_a / 2.0)) ** 2
        + (ellipse_y / (diameter_b / 2.0)) ** 2
    )
    return float(np.percentile(np.abs(normalized_radius - 1.0), 90))


def _touches_source_boundary(contour: np.ndarray, valid_pixels: np.ndarray) -> bool:
    if np.all(valid_pixels):
        return False
    invalid = (~valid_pixels).astype(np.uint8)
    adjacent_to_invalid = cv2.dilate(invalid, np.ones((3, 3), np.uint8)) > 0
    points = contour[:, 0, :]
    return bool(np.any(adjacent_to_invalid[points[:, 1], points[:, 0]]))


def _major_axis_orientation(
    ellipse: tuple[tuple[float, float], tuple[float, float], float]
) -> float:
    _, (diameter_a, diameter_b), angle_deg = ellipse
    orientation = angle_deg if diameter_a >= diameter_b else angle_deg + 90.0
    return float(orientation % 180.0)


def fit_etch_pit(
    source_image: np.ndarray,
    local_x: float,
    local_y: float,
    config: FitConfig | None = None,
) -> FitResult:
    """Segment a dark pit, choose one contour, fit an ellipse, and apply QC."""

    config = config or FitConfig()
    crop = extract_valid_crop(source_image, local_x, local_y, config)
    threshold_mask, threshold_value = _segment_dark_pixels(crop, config)
    mask, contours = _close_components_independently(
        threshold_mask, crop.valid_pixels, config
    )
    expected_center = np.asarray(crop.expected_center_xy, dtype=float)
    candidates: list[dict[str, object]] = []
    rejection_counts: Counter[str] = Counter()

    for contour in contours:
        if len(contour) < config.minimum_contour_points:
            rejection_counts["fewer_than_five_points"] += 1
            continue
        contour_area = float(cv2.contourArea(contour))
        if contour_area < config.minimum_contour_area_px2:
            rejection_counts["area_below_minimum"] += 1
            continue

        ellipse = cv2.fitEllipse(contour)
        (center_x, center_y), diameters, _ = ellipse
        minor_axis, major_axis = sorted(float(value) for value in diameters)
        if minor_axis < config.minimum_minor_axis_px:
            rejection_counts["minor_axis_below_minimum"] += 1
            continue
        if major_axis > config.maximum_major_axis_px:
            rejection_counts["major_axis_above_maximum"] += 1
            continue

        center_offset = float(
            np.linalg.norm(np.asarray([center_x, center_y]) - expected_center)
        )
        hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
        solidity = contour_area / hull_area if hull_area > 0 else 0.0
        residual = ellipse_residual_p90(contour, ellipse)
        points = contour[:, 0, :]
        last = config.crop_size_px - 1
        touches_crop = bool(
            np.any(
                (points[:, 0] <= 0)
                | (points[:, 0] >= last)
                | (points[:, 1] <= 0)
                | (points[:, 1] >= last)
            )
        )
        touches_source = _touches_source_boundary(contour, crop.valid_pixels)
        score = (
            center_offset
            + config.candidate_score_shape_weight
            * max(0.0, config.candidate_score_solidity_reference - solidity)
            + config.candidate_score_shape_weight
            * max(0.0, residual - config.maximum_residual_p90)
        )
        candidates.append(
            {
                "contour": contour,
                "ellipse": ellipse,
                "contour_area_px2": contour_area,
                "major_axis_px": major_axis,
                "minor_axis_px": minor_axis,
                "major_axis_orientation_deg": _major_axis_orientation(ellipse),
                "center_offset_px": center_offset,
                "solidity": solidity,
                "ellipse_residual_p90": residual,
                "touches_crop_boundary": touches_crop,
                "touches_source_image_boundary": touches_source,
                "candidate_score": score,
            }
        )

    if not candidates:
        if not np.any(crop.valid_pixels):
            detail = "coordinate_outside_source_image"
        elif not contours:
            detail = "no_external_contour"
        elif rejection_counts:
            detail = ";".join(
                f"{name}:{count}" for name, count in sorted(rejection_counts.items())
            )
        else:
            detail = "no_valid_candidate"
        return FitResult(
            candidate_found=False,
            qc_passed=False,
            qc_failure_codes=["A"],
            candidate_failure_detail=detail,
            crop=crop,
            threshold_value=threshold_value,
            segmentation_mask=mask,
            candidate_count=0,
            contour_count=len(contours),
            rejected_candidate_counts=dict(rejection_counts),
        )

    selected = min(candidates, key=lambda candidate: float(candidate["candidate_score"]))
    failures: list[str] = []
    if float(selected["center_offset_px"]) > config.maximum_center_offset_px:
        failures.append("B")
    if float(selected["solidity"]) < config.minimum_solidity:
        failures.append("C")
    if float(selected["ellipse_residual_p90"]) > config.maximum_residual_p90:
        failures.append("D")
    if bool(selected["touches_crop_boundary"]):
        failures.append("E")
    if bool(selected["touches_source_image_boundary"]):
        failures.append("F")

    major_axis = float(selected["major_axis_px"])
    minor_axis = float(selected["minor_axis_px"])
    incident_angle = math.degrees(
        math.acos(float(np.clip(minor_axis / major_axis, 0.0, 1.0)))
    )
    return FitResult(
        candidate_found=True,
        qc_passed=not failures,
        qc_failure_codes=failures,
        candidate_failure_detail="",
        crop=crop,
        threshold_value=threshold_value,
        segmentation_mask=mask,
        contour=selected["contour"],
        ellipse=selected["ellipse"],
        contour_area_px2=float(selected["contour_area_px2"]),
        major_axis_px=major_axis,
        minor_axis_px=minor_axis,
        major_axis_orientation_deg=float(selected["major_axis_orientation_deg"]),
        center_offset_px=float(selected["center_offset_px"]),
        solidity=float(selected["solidity"]),
        ellipse_residual_p90=float(selected["ellipse_residual_p90"]),
        touches_crop_boundary=bool(selected["touches_crop_boundary"]),
        touches_source_image_boundary=bool(
            selected["touches_source_image_boundary"]
        ),
        candidate_score=float(selected["candidate_score"]),
        incident_angle_deg=incident_angle,
        candidate_count=len(candidates),
        contour_count=len(contours),
        rejected_candidate_counts=dict(rejection_counts),
    )
