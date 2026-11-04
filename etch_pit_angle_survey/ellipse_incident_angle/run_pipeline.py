#!/usr/bin/env python3
"""Run the image-based C1/C2 etch-pit incident-angle pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.spatial import cKDTree


SCRIPT_DIR = Path(__file__).resolve().parent
SURVEY_ROOT = SCRIPT_DIR.parent
REPO_ROOT = SURVEY_ROOT.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from ellipse_fit import FitConfig, FitResult, QC_DEFINITIONS, fit_etch_pit  # noqa: E402
from reporting import (  # noqa: E402
    generate_detection_report,
    generate_failure_comparison_report,
    generate_main_report,
)


DEFAULT_INPUT_PATH = SCRIPT_DIR / "inputs/clean_surface_locations.csv"
DEFAULT_DATA_DIR = SCRIPT_DIR / "data"
DEFAULT_REPORT_DIR = SCRIPT_DIR / "reports"
C1_IMAGE_PATH = REPO_ROOT / "numpy_data_eval/preprocessed_clean_eval.npy"
C2_IMAGE_PATH = REPO_ROOT / "numpy_data_eval/clean_2_eval.npy"
TRUTH_WITH_IDS_PATH = REPO_ROOT / "predicted_list/ids/Aligned_clean_with_ids.txt"
PREDICTION_PATHS = {
    "fcn": REPO_ROOT / "predicted_list/FCN/aligned/final_pred_list.txt",
    "dcc": REPO_ROOT / "predicted_list/DCC/aligned/final_pred_list.txt",
}
SAVED_CLASSIFICATION_PATHS = {
    "fcn": {
        "detected": REPO_ROOT / "predicted_list/FCN/aligned/final_true_pos.txt",
        "missed": REPO_ROOT / "predicted_list/FCN/aligned/final_false_neg.txt",
    },
    "dcc": {
        "detected": REPO_ROOT / "predicted_list/DCC/aligned/final_true_pos.txt",
        "missed": REPO_ROOT / "predicted_list/DCC/aligned/final_false_neg.txt",
    },
}
DETECTION_THRESHOLD_PX = 15.0
FIRST_IMAGE_NUMBER = 251
RANDOM_SEED = 20261015


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as input_file:
        return json.load(input_file)


def load_manifest(path: Path) -> list[dict[str, object]]:
    required = {
        "clean_id",
        "truth_global_x",
        "truth_global_y",
        "c1_image",
        "c1_local_x",
        "c1_local_y",
        "c2_image",
        "c2_local_x",
        "c2_local_y",
    }
    rows: list[dict[str, object]] = []
    with path.open(newline="", encoding="utf-8") as input_file:
        reader = csv.DictReader(input_file)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            missing = sorted(required - set(reader.fieldnames or []))
            raise ValueError(f"Missing manifest columns in {path}: {missing}")
        for source in reader:
            rows.append(
                {
                    "clean_id": source["clean_id"],
                    "truth_global_x": float(source["truth_global_x"]),
                    "truth_global_y": float(source["truth_global_y"]),
                    "c1_image": int(float(source["c1_image"])),
                    "c1_local_x": float(source["c1_local_x"]),
                    "c1_local_y": float(source["c1_local_y"]),
                    "c2_image": int(float(source["c2_image"])),
                    "c2_local_x": float(source["c2_local_x"]),
                    "c2_local_y": float(source["c2_local_y"]),
                }
            )
    clean_ids = [str(row["clean_id"]) for row in rows]
    if len(clean_ids) != len(set(clean_ids)):
        raise ValueError(f"Duplicate clean IDs in {path}")
    if not rows:
        raise ValueError(f"No rows in {path}")
    return rows


def validate_manifest_truth(rows: Sequence[dict[str, object]]) -> None:
    expected = _load_json(TRUTH_WITH_IDS_PATH)
    expected_rows = [
        (
            str(record["id"]),
            float(record["coordinate"][0]),
            float(record["coordinate"][1]),
        )
        for record in expected
    ]
    actual_rows = [
        (
            str(row["clean_id"]),
            float(row["truth_global_x"]),
            float(row["truth_global_y"]),
        )
        for row in rows
    ]
    if actual_rows != expected_rows:
        raise ValueError(
            "The surface-location manifest does not exactly match the clean truth list"
        )


def _coordinate_counter(points: np.ndarray) -> Counter[tuple[float, float]]:
    return Counter((float(x), float(y)) for x, y in points)


def classify_detections(
    rows: Sequence[dict[str, object]],
) -> dict[str, dict[str, np.ndarray]]:
    truth = np.asarray(
        [[row["truth_global_x"], row["truth_global_y"]] for row in rows],
        dtype=float,
    )
    output: dict[str, dict[str, np.ndarray]] = {}
    for method, path in PREDICTION_PATHS.items():
        predictions = np.asarray(_load_json(path), dtype=float)
        if predictions.ndim != 2 or predictions.shape[1] != 2 or not len(predictions):
            raise ValueError(f"Expected a nonempty two-column prediction list in {path}")
        distances, _ = cKDTree(predictions).query(truth, k=1)
        detected = distances < DETECTION_THRESHOLD_PX
        output[method] = {"distance": distances, "detected": detected}

        expected_memberships = {
            "detected": truth[detected],
            "missed": truth[~detected],
        }
        for status, saved_path in SAVED_CLASSIFICATION_PATHS[method].items():
            saved = np.asarray(_load_json(saved_path), dtype=float)
            if _coordinate_counter(saved) != _coordinate_counter(expected_memberships[status]):
                raise ValueError(
                    f"Recomputed {method.upper()} {status} membership differs from {saved_path}"
                )
    return output


def run_fits(
    rows: Sequence[dict[str, object]],
    images: dict[str, np.ndarray],
    config: FitConfig,
) -> dict[str, list[FitResult]]:
    fits: dict[str, list[FitResult]] = {"C1": [], "C2": []}
    for row in rows:
        for surface in ("C1", "C2"):
            key = surface.lower()
            image_index = int(row[f"{key}_image"]) - FIRST_IMAGE_NUMBER
            if not 0 <= image_index < len(images[surface]):
                raise ValueError(
                    f"{row['clean_id']} {surface} image {row[f'{key}_image']} is outside "
                    f"the loaded image stack"
                )
            fits[surface].append(
                fit_etch_pit(
                    images[surface][image_index],
                    float(row[f"{key}_local_x"]),
                    float(row[f"{key}_local_y"]),
                    config,
                )
            )
    return fits


def _format_float(value: float | None) -> str:
    return "" if value is None else f"{value:.9f}"


def _fit_columns(surface: str, result: FitResult) -> dict[str, object]:
    prefix = surface.lower()
    failure_codes = "+".join(result.qc_failure_codes)
    return {
        f"{prefix}_fit_candidate": result.candidate_found,
        f"{prefix}_fit_qc_passed": result.qc_passed,
        f"{prefix}_failed_qc_codes": failure_codes,
        f"{prefix}_candidate_failure_detail": result.candidate_failure_detail,
        f"{prefix}_valid_crop_pixel_count": int(np.sum(result.crop.valid_pixels)),
        f"{prefix}_crop_crosses_source_boundary": not bool(np.all(result.crop.valid_pixels)),
        f"{prefix}_otsu_threshold": _format_float(result.threshold_value),
        f"{prefix}_external_contour_count": result.contour_count,
        f"{prefix}_ellipse_candidate_count": result.candidate_count,
        f"{prefix}_contour_area_px2": _format_float(result.contour_area_px2),
        f"{prefix}_major_axis_px": _format_float(result.major_axis_px),
        f"{prefix}_minor_axis_px": _format_float(result.minor_axis_px),
        f"{prefix}_major_axis_orientation_deg": _format_float(
            result.major_axis_orientation_deg
        ),
        f"{prefix}_center_offset_px": _format_float(result.center_offset_px),
        f"{prefix}_solidity": _format_float(result.solidity),
        f"{prefix}_ellipse_residual_p90": _format_float(
            result.ellipse_residual_p90
        ),
        f"{prefix}_touches_crop_boundary": (
            "" if result.touches_crop_boundary is None else result.touches_crop_boundary
        ),
        f"{prefix}_touches_source_image_boundary": (
            ""
            if result.touches_source_image_boundary is None
            else result.touches_source_image_boundary
        ),
        f"{prefix}_candidate_score": _format_float(result.candidate_score),
        f"{prefix}_incident_angle_deg": _format_float(result.incident_angle_deg),
    }


def build_output_rows(
    rows: Sequence[dict[str, object]],
    fits: dict[str, Sequence[FitResult]],
    detections: dict[str, dict[str, np.ndarray]],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for index, source in enumerate(rows):
        c1 = fits["C1"][index]
        c2 = fits["C2"][index]
        record = dict(source)
        record.update(_fit_columns("C1", c1))
        record.update(_fit_columns("C2", c2))
        if c1.qc_passed and c2.qc_passed:
            signed_error = float(c2.incident_angle_deg - c1.incident_angle_deg)
            record["signed_incident_angle_error_deg"] = f"{signed_error:.9f}"
            record["absolute_incident_angle_error_deg"] = f"{abs(signed_error):.9f}"
        else:
            record["signed_incident_angle_error_deg"] = ""
            record["absolute_incident_angle_error_deg"] = ""

        for method in ("fcn", "dcc"):
            detected = bool(detections[method]["detected"][index])
            record[f"{method}_nearest_prediction_distance_px"] = (
                f"{float(detections[method]['distance'][index]):.9f}"
            )
            record[f"{method}_detection_status"] = "detected" if detected else "missed"
            if detected and c1.qc_passed:
                category = "detected_angle_qc_passed"
            elif not detected and c1.qc_passed:
                category = "missed_angle_qc_passed"
            elif detected:
                category = "detected_angle_qc_failed"
            else:
                category = "missed_angle_qc_failed"
            record[f"{method}_category"] = category
        output.append(record)
    return output


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_detection_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    fields = [
        "clean_id",
        "c1_fit_candidate",
        "c1_fit_qc_passed",
        "c1_failed_qc_codes",
        "c1_incident_angle_deg",
        "fcn_nearest_prediction_distance_px",
        "fcn_detection_status",
        "fcn_category",
        "dcc_nearest_prediction_distance_px",
        "dcc_detection_status",
        "dcc_category",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def write_qc_failure_combinations(
    path: Path,
    fits: dict[str, Sequence[FitResult]],
) -> None:
    records: list[dict[str, object]] = []
    for surface in ("C1", "C2"):
        failures = [result for result in fits[surface] if not result.qc_passed]
        counts = Counter("+".join(result.qc_failure_codes) for result in failures)
        for codes, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            records.append(
                {
                    "surface": surface,
                    "failed_qc_codes": codes,
                    "count": count,
                    "percent_of_excluded_surface_fits": f"{100.0 * count / len(failures):.6f}",
                    "excluded_surface_fit_count": len(failures),
                    "clean_list_count": len(fits[surface]),
                }
            )
    write_csv(path, records)


def _stats(values: np.ndarray) -> dict[str, float | int]:
    if not len(values):
        return {"count": 0}
    return {
        "count": int(len(values)),
        "minimum": float(np.min(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "standard_deviation": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "maximum": float(np.max(values)),
    }


def build_summary(
    rows: Sequence[dict[str, object]],
    fits: dict[str, Sequence[FitResult]],
    output_rows: Sequence[dict[str, object]],
    config: FitConfig,
    random_seed: int,
) -> dict[str, object]:
    surface_summary: dict[str, object] = {}
    for surface in ("C1", "C2"):
        surface_fits = fits[surface]
        accepted_angles = np.asarray(
            [result.incident_angle_deg for result in surface_fits if result.qc_passed],
            dtype=float,
        )
        failure_counts = Counter(
            "+".join(result.qc_failure_codes)
            for result in surface_fits
            if not result.qc_passed
        )
        surface_summary[surface.lower()] = {
            "candidate_count": sum(result.candidate_found for result in surface_fits),
            "qc_passed_count": sum(result.qc_passed for result in surface_fits),
            "qc_excluded_count": sum(not result.qc_passed for result in surface_fits),
            "crop_crosses_source_boundary_count": sum(
                not bool(np.all(result.crop.valid_pixels)) for result in surface_fits
            ),
            "selected_contour_touches_source_boundary_count": sum(
                result.touches_source_image_boundary is True for result in surface_fits
            ),
            "failure_combinations": dict(sorted(failure_counts.items())),
            "accepted_incident_angle_degrees": _stats(accepted_angles),
        }

    signed = np.asarray(
        [
            float(row["signed_incident_angle_error_deg"])
            for row in output_rows
            if row["signed_incident_angle_error_deg"] != ""
        ],
        dtype=float,
    )
    absolute = np.abs(signed)
    paired = {
        "both_surfaces_qc_passed_count": int(len(signed)),
        "c1_pass_c2_fail_count": sum(
            c1.qc_passed and not c2.qc_passed
            for c1, c2 in zip(fits["C1"], fits["C2"])
        ),
        "c1_fail_c2_pass_count": sum(
            not c1.qc_passed and c2.qc_passed
            for c1, c2 in zip(fits["C1"], fits["C2"])
        ),
        "both_surfaces_fail_count": sum(
            not c1.qc_passed and not c2.qc_passed
            for c1, c2 in zip(fits["C1"], fits["C2"])
        ),
        "signed_error_degrees": _stats(signed),
        "absolute_error_degrees": {
            **_stats(absolute),
            "90th_percentile": float(np.percentile(absolute, 90)),
            "95th_percentile": float(np.percentile(absolute, 95)),
        },
    }

    detection_summary: dict[str, object] = {}
    for method in ("fcn", "dcc"):
        categories: dict[str, object] = {}
        category_names = sorted({str(row[f"{method}_category"]) for row in output_rows})
        for category in category_names:
            subset = [row for row in output_rows if row[f"{method}_category"] == category]
            angles = np.asarray(
                [
                    float(row["c1_incident_angle_deg"])
                    for row in subset
                    if row["c1_incident_angle_deg"] != ""
                ],
                dtype=float,
            )
            categories[category] = {
                "id_count": len(subset),
                "angle_count": len(angles),
                "without_candidate_angle_count": len(subset) - len(angles),
                "angle_degrees": _stats(angles),
            }
        detection_summary[method] = {"categories": categories}

    return {
        "method": "image-based ellipse incident angle",
        "angle_definition": "degrees(acos(minor_axis_pixels / major_axis_pixels))",
        "clean_list_count": len(rows),
        "random_example_seed": random_seed,
        "edge_handling": (
            "Pixels outside a source image are invalid and excluded from Gaussian blur "
            "normalization, Otsu threshold estimation, segmentation, and contours. "
            "No reflected pixels are created. A selected contour touching a source "
            "boundary fails QC F."
        ),
        "segmentation": {
            "crop_size_pixels": config.crop_size_px,
            "gaussian_kernel_pixels": config.gaussian_kernel_px,
            "gaussian_sigma_pixels": config.gaussian_sigma_px,
            "closing_kernel_pixels": config.closing_kernel_px,
            "threshold": "inverse Otsu estimated from valid source pixels only",
        },
        "quality_controls": {
            "definitions": QC_DEFINITIONS,
            "configuration": asdict(config),
        },
        "surfaces": surface_summary,
        "paired_surfaces": paired,
        "detection_threshold_pixels": DETECTION_THRESHOLD_PX,
        "detection_outcomes_using_c1_angle": detection_summary,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--no-reports",
        action="store_true",
        help="Write data and summaries without rendering PDFs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = args.input.resolve()
    data_dir = args.data_dir.resolve()
    report_dir = args.report_dir.resolve()
    rows = load_manifest(input_path)
    validate_manifest_truth(rows)
    images = {
        "C1": np.load(C1_IMAGE_PATH, mmap_mode="r"),
        "C2": np.load(C2_IMAGE_PATH, mmap_mode="r"),
    }
    config = FitConfig()
    fits = run_fits(rows, images, config)
    detections = classify_detections(rows)
    output_rows = build_output_rows(rows, fits, detections)

    main_data_path = data_dir / "ellipse_incident_angles.csv"
    detection_data_path = data_dir / "detection_categories.csv"
    summary_path = data_dir / "summary.json"
    failure_buckets_path = data_dir / "qc_failure_combinations.csv"
    failure_index_path = data_dir / "solidity_residual_failure_index.csv"
    write_csv(main_data_path, output_rows)
    write_detection_csv(detection_data_path, output_rows)
    write_qc_failure_combinations(failure_buckets_path, fits)
    summary = build_summary(rows, fits, output_rows, config, args.random_seed)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    if not args.no_reports:
        generate_main_report(
            report_dir / "incident_angle_report.pdf",
            rows,
            fits,
            config,
            args.random_seed,
        )
        generate_failure_comparison_report(
            report_dir / "solidity_residual_failure_cases.pdf",
            failure_index_path,
            rows,
            fits,
        )
        generate_detection_report(
            report_dir / "detection_outcomes.pdf",
            output_rows,
        )

    print(f"Processed {len(rows)} clean IDs")
    for surface in ("C1", "C2"):
        passed = sum(result.qc_passed for result in fits[surface])
        print(f"{surface}: {passed} QC passed, {len(rows) - passed} excluded")
    paired = summary["paired_surfaces"]
    print(f"Both surfaces passed: {paired['both_surfaces_qc_passed_count']}")
    print(f"Data: {main_data_path}")
    print(f"Summary: {summary_path}")
    if not args.no_reports:
        print(f"Reports: {report_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
