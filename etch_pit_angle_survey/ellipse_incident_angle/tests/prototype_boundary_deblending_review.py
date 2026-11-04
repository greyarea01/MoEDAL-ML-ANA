#!/usr/bin/env python3
"""Review component-safe closing and guarded deblending on named problem pits.

This is an experimental diagnostic. It does not alter the production fitter.
"""

from __future__ import annotations

import argparse
import math
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed


TEST_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = TEST_DIR.parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from ellipse_fit import (  # noqa: E402
    FitConfig,
    _touches_source_boundary,
    _valid_normalized_gaussian_blur,
    ellipse_residual_p90,
    extract_valid_crop,
    fit_etch_pit,
)
from run_pipeline import (  # noqa: E402
    C1_IMAGE_PATH,
    C2_IMAGE_PATH,
    DEFAULT_INPUT_PATH,
    FIRST_IMAGE_NUMBER,
    load_manifest,
)


DEFAULT_OUTPUT = (
    Path(tempfile.gettempdir())
    / "boundary_deblending_prototype_C0074_C0075_C0058.pdf"
)
CASE_IDS = ("C0074", "C0075", "C0058")


@dataclass
class Candidate:
    contour: np.ndarray
    ellipse: tuple[tuple[float, float], tuple[float, float], float]
    center_offset_px: float
    solidity: float
    residual_p90: float
    angle_deg: float
    qc_failures: tuple[str, ...]


def raw_otsu_mask(
    source_image: np.ndarray,
    x: float,
    y: float,
    config: FitConfig,
) -> tuple[object, np.ndarray, np.ndarray, float]:
    crop = extract_valid_crop(source_image, x, y, config)
    blurred = _valid_normalized_gaussian_blur(
        crop.image, crop.valid_pixels, config
    )
    threshold, _ = cv2.threshold(
        blurred[crop.valid_pixels].reshape(-1, 1),
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    mask = np.zeros(crop.image.shape, dtype=np.uint8)
    mask[crop.valid_pixels & (blurred <= threshold)] = 255
    return crop, blurred, mask, float(threshold)


def candidate_from_contour(
    contour: np.ndarray,
    crop: object,
    config: FitConfig,
) -> Candidate | None:
    if len(contour) < config.minimum_contour_points:
        return None
    area = float(cv2.contourArea(contour))
    if area < config.minimum_contour_area_px2:
        return None
    ellipse = cv2.fitEllipse(contour)
    minor, major = sorted(float(value) for value in ellipse[1])
    if minor < config.minimum_minor_axis_px or major > config.maximum_major_axis_px:
        return None
    offset = float(
        np.linalg.norm(
            np.asarray(ellipse[0]) - np.asarray(crop.expected_center_xy)
        )
    )
    hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
    solidity = area / hull_area if hull_area else 0.0
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
    failures = tuple(
        code
        for code, failed in (
            ("B", offset > config.maximum_center_offset_px),
            ("C", solidity < config.minimum_solidity),
            ("D", residual > config.maximum_residual_p90),
            ("E", touches_crop),
            ("F", touches_source),
        )
        if failed
    )
    angle = math.degrees(math.acos(float(np.clip(minor / major, 0.0, 1.0))))
    return Candidate(contour, ellipse, offset, solidity, residual, angle, failures)


def componentwise_candidates(
    raw_mask: np.ndarray,
    crop: object,
    config: FitConfig,
) -> list[Candidate]:
    count, labels = cv2.connectedComponents(raw_mask, connectivity=8)
    candidates: list[Candidate] = []
    kernel = np.ones(
        (config.closing_kernel_px, config.closing_kernel_px), dtype=np.uint8
    )
    for label in range(1, count):
        component = np.where(labels == label, 255, 0).astype(np.uint8)
        component = cv2.morphologyEx(component, cv2.MORPH_CLOSE, kernel)
        component[~crop.valid_pixels] = 0
        contours, _ = cv2.findContours(
            component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        for contour in contours:
            candidate = candidate_from_contour(contour, crop, config)
            if candidate is not None:
                candidates.append(candidate)
    return candidates


def watershed_candidates(
    raw_mask: np.ndarray,
    crop: object,
    config: FitConfig,
) -> list[Candidate]:
    component_labels, component_count = ndi.label(
        raw_mask > 0, structure=np.ones((3, 3), dtype=np.uint8)
    )
    candidates: list[Candidate] = []
    for component_number in range(1, component_count + 1):
        component = component_labels == component_number
        if np.sum(component) < config.minimum_contour_area_px2:
            continue
        filled = ndi.binary_fill_holes(component)
        distance = ndi.distance_transform_edt(filled)
        peaks = peak_local_max(
            distance,
            min_distance=1,
            threshold_abs=2.0,
            labels=filled,
            exclude_border=False,
        )
        if not len(peaks):
            peaks = np.asarray(
                [np.unravel_index(np.argmax(distance), distance.shape)]
            )
        markers = np.zeros(component_labels.shape, dtype=np.int32)
        for marker, (peak_y, peak_x) in enumerate(peaks, start=1):
            markers[peak_y, peak_x] = marker
        regions = watershed(
            -distance, markers, mask=filled, watershed_line=True
        )
        for region_number in range(1, int(np.max(regions)) + 1):
            region = np.where(regions == region_number, 255, 0).astype(np.uint8)
            contours, _ = cv2.findContours(
                region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            if not contours:
                continue
            candidate = candidate_from_contour(
                max(contours, key=cv2.contourArea), crop, config
            )
            if candidate is not None:
                candidates.append(candidate)
    return candidates


def hole_seeded_watershed_candidates(
    raw_mask: np.ndarray,
    crop: object,
    config: FitConfig,
) -> list[Candidate]:
    """Split a merged component using its enclosed bright holes as markers."""

    component_count, component_labels = cv2.connectedComponents(
        raw_mask, connectivity=8
    )
    candidates: list[Candidate] = []
    for component_number in range(1, component_count):
        component = component_labels == component_number
        if np.sum(component) < config.minimum_contour_area_px2:
            continue
        component_mask = np.where(component, 255, 0).astype(np.uint8)
        contours, hierarchy = cv2.findContours(
            component_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
        )
        hole_centers: list[tuple[int, int]] = []
        if hierarchy is not None:
            for contour, relation in zip(contours, hierarchy[0]):
                if (
                    int(relation[3]) >= 0
                    and len(contour) >= 5
                    and cv2.contourArea(contour) >= 3
                ):
                    center = cv2.fitEllipse(contour)[0]
                    hole_centers.append((round(center[0]), round(center[1])))
        if len(hole_centers) < 2:
            continue
        filled = ndi.binary_fill_holes(component)
        distance = ndi.distance_transform_edt(filled)
        markers = np.zeros(component_labels.shape, dtype=np.int32)
        for marker, (center_x, center_y) in enumerate(
            hole_centers, start=1
        ):
            markers[center_y, center_x] = marker
        regions = watershed(
            -distance, markers, mask=filled, watershed_line=True
        )
        for region_number in range(1, int(np.max(regions)) + 1):
            region = np.where(regions == region_number, 255, 0).astype(np.uint8)
            region_contours, _ = cv2.findContours(
                region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            if not region_contours:
                continue
            candidate = candidate_from_contour(
                max(region_contours, key=cv2.contourArea), crop, config
            )
            if candidate is not None:
                candidates.append(candidate)
    return candidates


def draw_overlay(
    image: np.ndarray,
    candidates: list[Candidate],
) -> np.ndarray:
    visible = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    colors = [(0, 210, 60), (40, 120, 255), (220, 70, 200)]
    for index, candidate in enumerate(candidates):
        color = colors[index % len(colors)]
        cv2.drawContours(
            visible, [candidate.contour], -1, color, 1, cv2.LINE_AA
        )
        cv2.ellipse(visible, candidate.ellipse, color, 1, cv2.LINE_AA)
    return visible


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--recovery-output",
        type=Path,
        default=None,
        help="Optional full-survey review of guarded fallback recoveries.",
    )
    return parser.parse_args()


def candidate_score(candidate: Candidate, config: FitConfig) -> float:
    return (
        candidate.center_offset_px
        + config.candidate_score_shape_weight
        * max(
            0.0,
            config.candidate_score_solidity_reference - candidate.solidity,
        )
        + config.candidate_score_shape_weight
        * max(0.0, candidate.residual_p90 - config.maximum_residual_p90)
    )


def generate_recovery_review(
    output_path: Path,
    rows: dict[str, dict[str, object]],
    images: dict[str, np.ndarray],
    config: FitConfig,
) -> None:
    cases: list[
        tuple[
            str,
            str,
            object,
            Candidate | None,
            list[Candidate],
            str,
        ]
    ] = []
    for surface in ("C1", "C2"):
        key = surface.lower()
        for clean_id, row in rows.items():
            source = images[surface][
                int(row[f"{key}_image"]) - FIRST_IMAGE_NUMBER
            ]
            x = float(row[f"{key}_local_x"])
            y = float(row[f"{key}_local_y"])
            crop, _, raw_mask, _ = raw_otsu_mask(source, x, y, config)
            ordinary = componentwise_candidates(raw_mask, crop, config)
            selected = (
                min(
                    ordinary,
                    key=lambda candidate: candidate_score(candidate, config),
                )
                if ordinary
                else None
            )
            failures = ("A",) if selected is None else selected.qc_failures
            if not failures or not set(failures).issubset({"C", "D"}):
                continue
            multi_pit = [
                candidate
                for candidate in hole_seeded_watershed_candidates(
                    raw_mask, crop, config
                )
                if not candidate.qc_failures
                and candidate.center_offset_px
                <= config.maximum_center_offset_px
            ]
            if len(multi_pit) >= 2:
                multi_pit.sort(key=lambda candidate: candidate.ellipse[0][0])
                cases.append(
                    (surface, clean_id, crop, selected, multi_pit, "AMBIGUOUS")
                )
                continue
            recovered = [
                candidate
                for candidate in watershed_candidates(raw_mask, crop, config)
                if not candidate.qc_failures
                and candidate.center_offset_px
                <= config.maximum_center_offset_px
            ]
            if len(recovered) == 1:
                cases.append(
                    (surface, clean_id, crop, selected, recovered, "RECOVERED")
                )
            elif len(recovered) > 1:
                recovered.sort(key=lambda candidate: candidate.ellipse[0][0])
                cases.append(
                    (surface, clean_id, crop, selected, recovered, "AMBIGUOUS")
                )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output_path) as pdf:
        recovered_count = sum(case[-1] == "RECOVERED" for case in cases)
        ambiguous_count = sum(case[-1] == "AMBIGUOUS" for case in cases)
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(
            0.06,
            0.92,
            "Guarded C/D-only deblending review",
            fontsize=20,
            weight="bold",
        )
        summary = [
            "Configuration: Gaussian kernel 3, sigma 0.9",
            f"Unique fallback recoveries: {recovered_count}",
            f"Multiple-candidate ambiguities: {ambiguous_count}",
            "",
            "The fallback is considered only after component-wise closing fails C or D.",
            "It cannot override A, B, E, or F.",
            "A fit is recoverable only when exactly one nearby watershed candidate passes all QC.",
            "Multiple passing candidates remain ambiguous and are not accepted.",
        ]
        y_position = 0.82
        for line in summary:
            fig.text(0.07, y_position, line, fontsize=13)
            y_position -= 0.06
        pdf.savefig(fig)
        plt.close(fig)

        for start in range(0, len(cases), 6):
            subset = cases[start : start + 6]
            fig, axes = plt.subplots(
                len(subset), 3, figsize=(11, 2.2 * len(subset) + 0.7),
                squeeze=False,
            )
            fig.suptitle(
                f"Guarded fallback cases {start + 1} through "
                f"{start + len(subset)} of {len(cases)}",
                fontsize=14,
            )
            for row_number, (
                surface,
                clean_id,
                crop,
                selected,
                proposed,
                status,
            ) in enumerate(subset):
                axes[row_number, 0].imshow(
                    crop.image,
                    cmap="gray",
                    vmin=0,
                    vmax=255,
                    interpolation="nearest",
                )
                axes[row_number, 0].set_title(f"{clean_id} {surface} raw")
                current = [] if selected is None else [selected]
                axes[row_number, 1].imshow(
                    draw_overlay(crop.image, current), interpolation="nearest"
                )
                failure_text = (
                    "A" if selected is None
                    else "+".join(selected.qc_failures)
                )
                current_angle = (
                    "n/a" if selected is None
                    else f"{selected.angle_deg:.2f} deg"
                )
                axes[row_number, 1].set_title(
                    f"Ordinary FAIL {failure_text}; {current_angle}"
                )
                axes[row_number, 2].imshow(
                    draw_overlay(crop.image, proposed), interpolation="nearest"
                )
                angles = ", ".join(
                    f"{candidate.angle_deg:.2f}" for candidate in proposed
                )
                axes[row_number, 2].set_title(
                    f"{status}: {angles} deg"
                )
                for axis in axes[row_number]:
                    axis.axis("off")
            fig.tight_layout(rect=(0, 0, 1, 0.96))
            pdf.savefig(fig)
            plt.close(fig)
    print(f"Recovery review: {output_path.resolve()}")


def main() -> int:
    args = parse_args()
    config = FitConfig(gaussian_kernel_px=3, gaussian_sigma_px=0.9)
    rows = {str(row["clean_id"]): row for row in load_manifest(DEFAULT_INPUT_PATH)}
    images = {
        "C1": np.load(C1_IMAGE_PATH, mmap_mode="r"),
        "C2": np.load(C2_IMAGE_PATH, mmap_mode="r"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(args.output) as pdf:
        for clean_id in CASE_IDS:
            fig, axes = plt.subplots(2, 4, figsize=(14, 7.5))
            for row_number, surface in enumerate(("C1", "C2")):
                row = rows[clean_id]
                key = surface.lower()
                source = images[surface][
                    int(row[f"{key}_image"]) - FIRST_IMAGE_NUMBER
                ]
                x = float(row[f"{key}_local_x"])
                y = float(row[f"{key}_local_y"])
                crop, _, raw_mask, threshold = raw_otsu_mask(
                    source, x, y, config
                )
                current = fit_etch_pit(source, x, y, config)
                if clean_id == "C0074":
                    proposed = componentwise_candidates(raw_mask, crop, config)
                    passing = [candidate for candidate in proposed if not candidate.qc_failures]
                    proposed = sorted(
                        passing or proposed,
                        key=lambda candidate: candidate.center_offset_px,
                    )[:1]
                    method = "component-wise closing"
                elif clean_id == "C0075" and surface == "C2":
                    proposed = [
                        candidate
                        for candidate in watershed_candidates(
                            raw_mask, crop, config
                        )
                        if not candidate.qc_failures
                        and candidate.center_offset_px <= config.maximum_center_offset_px
                    ]
                    proposed = sorted(
                        proposed, key=lambda candidate: candidate.center_offset_px
                    )[:1]
                    method = "guarded watershed fallback"
                elif clean_id == "C0058":
                    proposed = [
                        candidate
                        for candidate in hole_seeded_watershed_candidates(
                            raw_mask, crop, config
                        )
                        if not candidate.qc_failures
                        and candidate.center_offset_px <= config.maximum_center_offset_px
                    ]
                    proposed.sort(key=lambda candidate: candidate.ellipse[0][0])
                    method = "multi-pit watershed candidates"
                else:
                    proposed = []
                    if current.contour is not None and current.ellipse is not None:
                        proposed = [
                            Candidate(
                                current.contour,
                                current.ellipse,
                                float(current.center_offset_px),
                                float(current.solidity),
                                float(current.ellipse_residual_p90),
                                float(current.incident_angle_deg),
                                tuple(current.qc_failure_codes),
                            )
                        ]
                    method = "current accepted fit"

                axes[row_number, 0].imshow(
                    crop.image, cmap="gray", vmin=0, vmax=255,
                    interpolation="nearest"
                )
                axes[row_number, 0].set_title(f"{surface} raw crop")
                axes[row_number, 1].imshow(
                    raw_mask, cmap="gray", vmin=0, vmax=255,
                    interpolation="nearest"
                )
                axes[row_number, 1].set_title(
                    f"Otsu mask, threshold {threshold:.0f}"
                )
                current_candidates: list[Candidate] = []
                if current.contour is not None and current.ellipse is not None:
                    current_candidates = [
                        Candidate(
                            current.contour,
                            current.ellipse,
                            float(current.center_offset_px),
                            float(current.solidity),
                            float(current.ellipse_residual_p90),
                            float(current.incident_angle_deg),
                            tuple(current.qc_failure_codes),
                        )
                    ]
                axes[row_number, 2].imshow(
                    draw_overlay(crop.image, current_candidates),
                    interpolation="nearest",
                )
                current_status = (
                    "PASS" if current.qc_passed
                    else "FAIL " + "+".join(current.qc_failure_codes)
                )
                current_angle = (
                    "n/a" if current.incident_angle_deg is None
                    else f"{current.incident_angle_deg:.2f} deg"
                )
                axes[row_number, 2].set_title(
                    f"Current: {current_status}; {current_angle}"
                )
                axes[row_number, 3].imshow(
                    draw_overlay(crop.image, proposed), interpolation="nearest"
                )
                angles = ", ".join(
                    f"{candidate.angle_deg:.2f}" for candidate in proposed
                ) or "none"
                axes[row_number, 3].set_title(
                    f"Prototype: {method}\nangles {angles} deg"
                )
                for axis in axes[row_number]:
                    axis.axis("off")
            fig.suptitle(
                f"{clean_id}: current boundary versus experimental deblending",
                fontsize=15,
            )
            fig.tight_layout(rect=(0, 0, 1, 0.95))
            pdf.savefig(fig)
            plt.close(fig)
    if args.recovery_output is not None:
        generate_recovery_review(
            args.recovery_output.resolve(), rows, images, config
        )
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
