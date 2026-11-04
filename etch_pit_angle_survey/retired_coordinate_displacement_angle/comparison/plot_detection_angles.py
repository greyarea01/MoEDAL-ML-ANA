#!/usr/bin/env python3
"""Plot clean etch-pit angle distributions by FCN and DCC detection status."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
SURVEY_ROOT = SCRIPT_DIR.parent
REPO_ROOT = SCRIPT_DIR.parents[2]

TRUTH_IDS_PATH = REPO_ROOT / "predicted_list/ids/Aligned_clean_with_ids.txt"
ANGLE_PATHS = {
    "lewis": SURVEY_ROOT / "lewis/data/clean_c1_c2_distances.csv",
    "kanik": SURVEY_ROOT / "kanik/data/clean_c1_c2_distances.csv",
}
ANGLE_COMPARISON_PATH = SCRIPT_DIR / "data/lewis_kanik_angle_comparison.csv"
METHOD_PATHS = {
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

DEFAULT_DATA_PATH = SCRIPT_DIR / "data/clean_id_detection_angles.csv"
DEFAULT_PLOT_CANDIDATE_DATA_PATH = (
    SCRIPT_DIR / "data/lowest_error_plot_candidates.csv"
)
DEFAULT_SUMMARY_PATH = SCRIPT_DIR / "data/detection_angle_summary.json"
DEFAULT_PLOT_DIR = SCRIPT_DIR / "plots"

ANGLE_COLUMN = "c1_c2_angle_from_foil_normal_deg"
PLOT_CANDIDATE_LIMIT = 2
DETECTION_THRESHOLD_PX = 15.0
PIXEL_SIZE_UM = 5.6
FOIL_SEPARATION_UM = 200.0
GEOMETRY_MATCH_LIMIT_PX = 12.0
GEOMETRY_ANGLE_LIMIT_DEG = math.degrees(
    math.atan(GEOMETRY_MATCH_LIMIT_PX * PIXEL_SIZE_UM / FOIL_SEPARATION_UM)
)

METHOD_LABELS = {"fcn": "FCN", "dcc": "DCC"}
SOURCE_LABELS = {"lewis": "Lewis", "kanik": "Kanik"}
METHOD_COLORS = {"fcn": "#2468a2", "dcc": "#d97706"}
OUTCOME_COLORS = {
    "detected_by_both": "#16827a",
    "fcn_only": "#2468a2",
    "dcc_only": "#d97706",
    "missed_by_both": "#b23a48",
}
AGGREGATE_MISS_COLORS = {
    "missed_by_fcn": "#7b4f9d",
    "missed_by_dcc": "#526675",
}


def load_json(path: Path) -> object:
    with path.open(encoding="utf-8") as input_file:
        return json.load(input_file)


def load_truth() -> tuple[list[str], np.ndarray]:
    records = load_json(TRUTH_IDS_PATH)
    if not isinstance(records, list):
        raise ValueError(f"Expected a list in {TRUTH_IDS_PATH}")

    clean_ids: list[str] = []
    coordinates: list[list[float]] = []
    for record in records:
        if not isinstance(record, dict) or "id" not in record or "coordinate" not in record:
            raise ValueError(f"Malformed clean truth record in {TRUTH_IDS_PATH}")
        coordinate = record["coordinate"]
        if not isinstance(coordinate, list) or len(coordinate) != 2:
            raise ValueError(f"Malformed clean coordinate in {TRUTH_IDS_PATH}")
        clean_ids.append(str(record["id"]))
        coordinates.append([float(coordinate[0]), float(coordinate[1])])

    if len(clean_ids) != len(set(clean_ids)):
        raise ValueError("Clean IDs are not unique")
    return clean_ids, np.asarray(coordinates, dtype=float)


def load_selected_angles(path: Path, clean_ids: list[str]) -> np.ndarray:
    by_id: dict[str, float] = {}
    with path.open(encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        required = {"clean_id", ANGLE_COLUMN}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Missing required columns in {path}")
        for row in reader:
            clean_id = row["clean_id"]
            if clean_id in by_id:
                raise ValueError(f"Duplicate selected angle for {clean_id} in {path}")
            angle = float(row[ANGLE_COLUMN])
            if not math.isfinite(angle):
                raise ValueError(f"Non-finite selected angle for {clean_id} in {path}")
            by_id[clean_id] = angle

    if set(by_id) != set(clean_ids):
        raise ValueError(f"Selected angle IDs do not exactly match clean IDs in {path}")
    return np.asarray([by_id[clean_id] for clean_id in clean_ids], dtype=float)


def load_lowest_error_candidate_pairs(
    path: Path, clean_ids: list[str]
) -> list[dict[str, str | int]]:
    rows_by_id: dict[str, list[dict[str, str | int]]] = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        required = {
            "clean_id",
            "lewis_angle_deg",
            "kanik_angle_deg",
            "absolute_angle_difference_deg",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Missing required columns in {path}")
        for source_order, row in enumerate(reader):
            clean_id = row["clean_id"]
            lewis_angle = row["lewis_angle_deg"]
            kanik_angle = row["kanik_angle_deg"]
            absolute_error = abs(Decimal(lewis_angle) - Decimal(kanik_angle))
            if absolute_error != Decimal(row["absolute_angle_difference_deg"]):
                raise ValueError(f"Inconsistent angle difference for {clean_id} in {path}")
            rows_by_id[clean_id].append(
                {
                    "clean_id": clean_id,
                    "lewis_angle_deg": lewis_angle,
                    "kanik_angle_deg": kanik_angle,
                    "absolute_angle_error_deg": f"{absolute_error:.6f}",
                    "source_order": source_order,
                }
            )

    if set(rows_by_id) != set(clean_ids):
        raise ValueError(f"Candidate comparison IDs do not exactly match clean IDs in {path}")

    selected: list[dict[str, str | int]] = []
    for clean_id in clean_ids:
        rows = rows_by_id[clean_id]
        lewis_numbers: dict[str, int] = {}
        kanik_numbers: dict[str, int] = {}
        for row in rows:
            lewis_angle = str(row["lewis_angle_deg"])
            kanik_angle = str(row["kanik_angle_deg"])
            if lewis_angle not in lewis_numbers:
                lewis_numbers[lewis_angle] = len(lewis_numbers) + 1
            if kanik_angle not in kanik_numbers:
                kanik_numbers[kanik_angle] = len(kanik_numbers) + 1

        ranked = sorted(
            rows,
            key=lambda row: (
                Decimal(str(row["absolute_angle_error_deg"])),
                int(row["source_order"]),
            ),
        )
        selected_count = min(PLOT_CANDIDATE_LIMIT, len(ranked))
        for selection_rank, row in enumerate(ranked[:selected_count], start=1):
            selected.append(
                {
                    "clean_id": clean_id,
                    "plot_candidate_rank": selection_rank,
                    "plot_candidate_frequency": selected_count,
                    "candidate_pair_count_available": len(rows),
                    "lewis_candidate_number": lewis_numbers[
                        str(row["lewis_angle_deg"])
                    ],
                    "lewis_candidate_count": len(lewis_numbers),
                    "lewis_angle_deg": str(row["lewis_angle_deg"]),
                    "kanik_candidate_number": kanik_numbers[
                        str(row["kanik_angle_deg"])
                    ],
                    "kanik_candidate_count": len(kanik_numbers),
                    "kanik_angle_deg": str(row["kanik_angle_deg"]),
                    "absolute_angle_error_deg": str(row["absolute_angle_error_deg"]),
                }
            )
    return selected


def load_predictions(path: Path) -> np.ndarray:
    records = load_json(path)
    predictions = np.asarray(records, dtype=float)
    if predictions.ndim != 2 or predictions.shape[1] != 2 or len(predictions) == 0:
        raise ValueError(f"Expected a non-empty Nx2 prediction list in {path}")
    return predictions


def classify_clean_ids(
    truth_coordinates: np.ndarray, predictions: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    distances = np.linalg.norm(
        truth_coordinates[:, np.newaxis, :] - predictions[np.newaxis, :, :],
        axis=2,
    )
    nearest_distances = np.min(distances, axis=1)
    return nearest_distances < DETECTION_THRESHOLD_PX, nearest_distances


def coordinate_counter(coordinates: np.ndarray) -> Counter[tuple[float, float]]:
    return Counter((float(x), float(y)) for x, y in coordinates)


def validate_saved_classification(
    method: str,
    truth_coordinates: np.ndarray,
    detected: np.ndarray,
) -> None:
    expected = {
        "detected": coordinate_counter(truth_coordinates[detected]),
        "missed": coordinate_counter(truth_coordinates[~detected]),
    }
    for status, path in SAVED_CLASSIFICATION_PATHS[method].items():
        saved = np.asarray(load_json(path), dtype=float)
        if saved.ndim != 2 or saved.shape[1] != 2:
            raise ValueError(f"Expected an Nx2 coordinate list in {path}")
        if coordinate_counter(saved) != expected[status]:
            raise ValueError(
                f"Recomputed {method.upper()} {status} membership differs from {path}"
            )


def joint_status(fcn_detected: bool, dcc_detected: bool) -> str:
    if fcn_detected and dcc_detected:
        return "detected_by_both"
    if fcn_detected:
        return "fcn_only"
    if dcc_detected:
        return "dcc_only"
    return "missed_by_both"


def write_data(
    output_path: Path,
    clean_ids: list[str],
    truth_coordinates: np.ndarray,
    angles: dict[str, np.ndarray],
    detections: dict[str, np.ndarray],
    nearest_distances: dict[str, np.ndarray],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "clean_id",
        "truth_global_x",
        "truth_global_y",
        "lewis_angle_deg",
        "kanik_angle_deg",
        "fcn_nearest_prediction_distance_px",
        "fcn_detection_status",
        "dcc_nearest_prediction_distance_px",
        "dcc_detection_status",
        "joint_detection_status",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for index, clean_id in enumerate(clean_ids):
            fcn_detected = bool(detections["fcn"][index])
            dcc_detected = bool(detections["dcc"][index])
            writer.writerow(
                {
                    "clean_id": clean_id,
                    "truth_global_x": f"{truth_coordinates[index, 0]:.6f}",
                    "truth_global_y": f"{truth_coordinates[index, 1]:.6f}",
                    "lewis_angle_deg": f"{angles['lewis'][index]:.6f}",
                    "kanik_angle_deg": f"{angles['kanik'][index]:.6f}",
                    "fcn_nearest_prediction_distance_px": (
                        f"{nearest_distances['fcn'][index]:.6f}"
                    ),
                    "fcn_detection_status": "detected" if fcn_detected else "missed",
                    "dcc_nearest_prediction_distance_px": (
                        f"{nearest_distances['dcc'][index]:.6f}"
                    ),
                    "dcc_detection_status": "detected" if dcc_detected else "missed",
                    "joint_detection_status": joint_status(fcn_detected, dcc_detected),
                }
            )


def write_plot_candidate_data(
    output_path: Path,
    candidates: list[dict[str, str | int]],
    clean_id_indices: dict[str, int],
    detections: dict[str, np.ndarray],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "clean_id",
        "plot_candidate_rank",
        "plot_candidate_frequency",
        "candidate_pair_count_available",
        "lewis_candidate_number",
        "lewis_candidate_count",
        "lewis_angle_deg",
        "kanik_candidate_number",
        "kanik_candidate_count",
        "kanik_angle_deg",
        "absolute_angle_error_deg",
        "fcn_detection_status",
        "dcc_detection_status",
        "joint_detection_status",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for candidate in candidates:
            clean_id = str(candidate["clean_id"])
            clean_index = clean_id_indices[clean_id]
            fcn_detected = bool(detections["fcn"][clean_index])
            dcc_detected = bool(detections["dcc"][clean_index])
            writer.writerow(
                {
                    **candidate,
                    "fcn_detection_status": (
                        "detected" if fcn_detected else "missed"
                    ),
                    "dcc_detection_status": (
                        "detected" if dcc_detected else "missed"
                    ),
                    "joint_detection_status": joint_status(
                        fcn_detected, dcc_detected
                    ),
                }
            )


def descriptive_statistics(values: np.ndarray) -> dict[str, float | int]:
    if len(values) == 0:
        return {"count": 0}
    return {
        "count": int(len(values)),
        "minimum_deg": float(np.min(values)),
        "q25_deg": float(np.quantile(values, 0.25)),
        "median_deg": float(np.median(values)),
        "mean_deg": float(np.mean(values)),
        "q75_deg": float(np.quantile(values, 0.75)),
        "maximum_deg": float(np.max(values)),
    }


def create_summary(
    clean_ids: list[str],
    plot_candidate_ids: np.ndarray,
    angles: dict[str, np.ndarray],
    predictions: dict[str, np.ndarray],
    clean_id_detections: dict[str, np.ndarray],
    plot_candidate_detections: dict[str, np.ndarray],
) -> dict[str, object]:
    clean_id_outcomes = np.asarray(
        [
            joint_status(bool(fcn), bool(dcc))
            for fcn, dcc in zip(
                clean_id_detections["fcn"], clean_id_detections["dcc"]
            )
        ]
    )
    plot_candidate_outcomes = np.asarray(
        [
            joint_status(bool(fcn), bool(dcc))
            for fcn, dcc in zip(
                plot_candidate_detections["fcn"],
                plot_candidate_detections["dcc"],
            )
        ]
    )
    angle_statistics: dict[str, object] = {}
    for source, source_angles in angles.items():
        method_statistics: dict[str, object] = {}
        for method, detected in plot_candidate_detections.items():
            method_statistics[method] = {
                "detected": descriptive_statistics(source_angles[detected]),
                "missed": descriptive_statistics(source_angles[~detected]),
            }
        method_statistics["joint_outcomes"] = {
            outcome: descriptive_statistics(
                source_angles[plot_candidate_outcomes == outcome]
            )
            for outcome in OUTCOME_COLORS
        }
        method_statistics["missed_by_model_aggregates"] = {
            "missed_by_fcn": descriptive_statistics(
                source_angles[~plot_candidate_detections["fcn"]]
            ),
            "missed_by_dcc": descriptive_statistics(
                source_angles[~plot_candidate_detections["dcc"]]
            ),
        }
        angle_statistics[source] = method_statistics

    candidate_frequencies = Counter(plot_candidate_ids.tolist())

    return {
        "clean_id_count": len(clean_ids),
        "detection_definition": (
            "A clean ID is detected when its nearest aligned model prediction has "
            "strict Euclidean distance < 15 pixels."
        ),
        "detection_threshold_pixels_strict": DETECTION_THRESHOLD_PX,
        "angle_definition": (
            "atan(C1-to-C2 lateral distance / 200 um), measured from the foil normal"
        ),
        "angle_source": (
            "For each clean ID, retain at most two Lewis/Kanik candidate pairs "
            "having the smallest absolute inter-survey angle errors."
        ),
        "plot_candidate_limit_per_clean_id": PLOT_CANDIDATE_LIMIT,
        "plot_candidate_row_count": len(plot_candidate_ids),
        "plot_candidate_clean_id_count": len(candidate_frequencies),
        "plot_candidate_frequency_counts": {
            str(frequency): sum(
                count == frequency for count in candidate_frequencies.values()
            )
            for frequency in sorted(set(candidate_frequencies.values()))
        },
        "geometry_12_pixel_angle_limit_deg": GEOMETRY_ANGLE_LIMIT_DEG,
        "prediction_counts": {
            method: int(len(method_predictions))
            for method, method_predictions in predictions.items()
        },
        "detection_counts": {
            method: {
                "detected": int(np.sum(detected)),
                "missed": int(np.sum(~detected)),
            }
            for method, detected in clean_id_detections.items()
        },
        "joint_detection_counts": {
            outcome: int(np.sum(clean_id_outcomes == outcome))
            for outcome in OUTCOME_COLORS
        },
        "plot_candidate_detection_counts": {
            method: {
                "detected": int(np.sum(detected)),
                "missed": int(np.sum(~detected)),
            }
            for method, detected in plot_candidate_detections.items()
        },
        "plot_candidate_joint_detection_counts": {
            outcome: int(np.sum(plot_candidate_outcomes == outcome))
            for outcome in OUTCOME_COLORS
        },
        "missed_by_model_counts": {
            method: {
                "clean_ids": int(np.sum(~clean_id_detections[method])),
                "plot_candidates": int(
                    np.sum(~plot_candidate_detections[method])
                ),
            }
            for method in ("fcn", "dcc")
        },
        "angle_statistics": angle_statistics,
    }


def gnuplot_quote(value: str | Path) -> str:
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'


def common_gnuplot_style() -> str:
    return f"""
set encoding utf8
set datafile separator whitespace
set border linewidth 1.0 linecolor rgb "#3f454b"
set grid back linecolor rgb "#d9dde1" linewidth 0.7
set tics textcolor rgb "#25292d"
set xlabel textcolor rgb "#25292d"
set ylabel textcolor rgb "#25292d"
set title textcolor rgb "#25292d"
set key opaque box linecolor rgb "#d9dde1" font ",9"
set xrange [0:19.2]
set xtics 0,3,18
set arrow 99 from {GEOMETRY_ANGLE_LIMIT_DEG:.10f}, graph 0 to {GEOMETRY_ANGLE_LIMIT_DEG:.10f}, graph 1 nohead front dashtype 2 linewidth 1.2 linecolor rgb "#5d6369"
set label 99 "12-pixel limit" at {GEOMETRY_ANGLE_LIMIT_DEG - 0.16:.10f}, graph 0.50 rotate by 90 center front textcolor rgb "#5d6369" font ",8"
"""


def run_gnuplot(body: str, output_path: Path, terminal: str) -> None:
    if shutil.which("gnuplot") is None:
        raise RuntimeError("gnuplot is required to generate the angle plots")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if terminal == "png":
        terminal_command = (
            'set terminal pngcairo size 1800,1200 enhanced font "DejaVu Sans,11"'
        )
    elif terminal == "pdf":
        terminal_command = (
            'set terminal pdfcairo size 12in,8in enhanced font "DejaVu Sans,10"'
        )
    else:
        raise ValueError(f"Unsupported Gnuplot terminal: {terminal}")

    script = (
        f"{terminal_command}\n"
        f"set output {gnuplot_quote(output_path)}\n"
        f"{common_gnuplot_style()}\n"
        f"{body}\n"
        "unset output\n"
    )
    completed = subprocess.run(
        ["gnuplot"],
        input=script,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Gnuplot failed for {output_path}:\n{completed.stderr.strip()}"
        )


def render_plot_pair(body: str, plot_dir: Path, stem: str) -> list[Path]:
    paths = [plot_dir / f"{stem}.png", plot_dir / f"{stem}.pdf"]
    run_gnuplot(body, paths[0], "png")
    run_gnuplot(body, paths[1], "pdf")
    return paths


def write_numeric_table(path: Path, columns: list[np.ndarray]) -> None:
    if not columns or len({len(column) for column in columns}) != 1:
        raise ValueError("Plot columns must have one common non-zero length")
    data = np.column_stack(columns)
    np.savetxt(path, data, fmt="%.10f")


def plot_detected_vs_missed(
    candidate_ids: np.ndarray,
    angles: dict[str, np.ndarray],
    detections: dict[str, np.ndarray],
    plot_dir: Path,
) -> list[Path]:
    bin_edges = np.arange(0.0, 20.0, 1.0)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    columns = [centers]
    column_numbers: dict[tuple[str, str, str], int] = {}
    for method in ("fcn", "dcc"):
        for source in ("lewis", "kanik"):
            for status, mask in (
                ("detected", detections[method]),
                ("missed", ~detections[method]),
            ):
                counts, _ = np.histogram(angles[source][mask], bins=bin_edges)
                percentages = 100.0 * counts / np.sum(counts)
                columns.append(percentages)
                column_numbers[(method, source, status)] = len(columns)

    with tempfile.TemporaryDirectory(prefix="etch_angle_hist_") as temp_dir:
        data_path = Path(temp_dir) / "histograms.dat"
        write_numeric_table(data_path, columns)
        panels: list[str] = []
        for method in ("fcn", "dcc"):
            for source in ("lewis", "kanik"):
                detected_column = column_numbers[(method, source, "detected")]
                missed_column = column_numbers[(method, source, "missed")]
                detected_count = int(np.sum(detections[method]))
                missed_count = int(np.sum(~detections[method]))
                detected_id_count = len(set(candidate_ids[detections[method]]))
                missed_id_count = len(set(candidate_ids[~detections[method]]))
                panels.append(
                    f"""
set title "{METHOD_LABELS[method]} detection | {SOURCE_LABELS[source]} angle"
set xlabel "Angle from foil normal (degrees)"
set ylabel "Share within outcome (%)"
set yrange [0:*]
set key top right
plot {gnuplot_quote(data_path)} using 1:{detected_column} with filledcurves y1=0 fillcolor rgb "#16827a" fillstyle transparent solid 0.18 noborder notitle, \\
     {gnuplot_quote(data_path)} using 1:{missed_column} with filledcurves y1=0 fillcolor rgb "#b23a48" fillstyle transparent solid 0.18 noborder notitle, \\
     {gnuplot_quote(data_path)} using 1:{detected_column} with steps linecolor rgb "#16827a" linewidth 2 title "Detected candidates (n={detected_count}; IDs={detected_id_count})", \\
     {gnuplot_quote(data_path)} using 1:{missed_column} with steps linecolor rgb "#b23a48" linewidth 2 title "Missed candidates (n={missed_count}; IDs={missed_id_count})"
"""
                )
        body = (
            'set multiplot layout 2,2 rowsfirst title "Up to two lowest-error angle candidates per clean ID: detected and missed" font ",15"\n'
            + "\n".join(panels)
            + "\nunset multiplot"
        )
        return render_plot_pair(
            body, plot_dir, "detected_vs_missed_angle_distributions"
        )


def plot_joint_outcomes(
    candidate_ids: np.ndarray,
    angles: dict[str, np.ndarray],
    detections: dict[str, np.ndarray],
    plot_dir: Path,
) -> list[Path]:
    outcomes = np.asarray(
        [
            joint_status(bool(fcn), bool(dcc))
            for fcn, dcc in zip(detections["fcn"], detections["dcc"])
        ]
    )
    categories = [
        (
            "missed_by_dcc",
            "Missed by DCC, aggregate",
            ~detections["dcc"],
            AGGREGATE_MISS_COLORS["missed_by_dcc"],
        ),
        (
            "missed_by_fcn",
            "Missed by FCN, aggregate",
            ~detections["fcn"],
            AGGREGATE_MISS_COLORS["missed_by_fcn"],
        ),
        (
            "missed_by_both",
            "Missed by both",
            outcomes == "missed_by_both",
            OUTCOME_COLORS["missed_by_both"],
        ),
        (
            "dcc_only",
            "DCC only",
            outcomes == "dcc_only",
            OUTCOME_COLORS["dcc_only"],
        ),
        (
            "fcn_only",
            "FCN only",
            outcomes == "fcn_only",
            OUTCOME_COLORS["fcn_only"],
        ),
        (
            "detected_by_both",
            "Detected by both",
            outcomes == "detected_by_both",
            OUTCOME_COLORS["detected_by_both"],
        ),
    ]
    rng = np.random.default_rng(20260804)
    with tempfile.TemporaryDirectory(prefix="etch_angle_joint_") as temp_dir:
        temp_path = Path(temp_dir)
        source_commands: list[str] = []
        for source in ("lewis", "kanik"):
            plot_items: list[str] = []
            statistic_commands: list[str] = []
            for arrow_offset, (position, (outcome, _, mask, color)) in enumerate(
                enumerate(categories), start=1
            ):
                values = angles[source][mask]
                jitter = rng.uniform(-0.16, 0.16, len(values))
                data_path = temp_path / f"{source}_{outcome}.dat"
                write_numeric_table(
                    data_path,
                    [values, np.full(len(values), position, dtype=float) + jitter],
                )
                plot_items.append(
                    f"{gnuplot_quote(data_path)} using 1:2 with points "
                    f"pointtype 7 pointsize 0.35 linecolor rgb \"{color}\" notitle"
                )
                q05, q25, median, q75, q95 = np.quantile(
                    values, [0.05, 0.25, 0.50, 0.75, 0.95]
                )
                base = 100 + arrow_offset * 10
                statistic_commands.extend(
                    [
                        f"set arrow {base} from {q05:.10f},{position} to {q95:.10f},{position} nohead front linewidth 1.5 linecolor rgb \"{color}\"",
                        f"set arrow {base + 1} from {q25:.10f},{position} to {q75:.10f},{position} nohead front linewidth 7.0 linecolor rgb \"{color}\"",
                        f"set arrow {base + 2} from {median:.10f},{position - 0.14:.3f} to {median:.10f},{position + 0.14:.3f} nohead front linewidth 2.0 linecolor rgb \"#25292d\"",
                    ]
                )

            ytics = ", ".join(
                (
                    f'"{label} (cand.={int(np.sum(mask))}; '
                    f'IDs={len(set(candidate_ids[mask]))})" {position}'
                )
                for position, (_, label, mask, _) in enumerate(categories)
            )
            unset_commands = "\n".join(
                f"unset arrow {100 + offset * 10 + detail}"
                for offset in range(1, len(categories) + 1)
                for detail in range(3)
            )
            source_commands.append(
                f"""
{unset_commands}
{chr(10).join(statistic_commands)}
set title "{SOURCE_LABELS[source]} candidate angles"
set xlabel "Angle from foil normal (degrees)"
set ylabel "Detection outcome"
set yrange [-0.5:{len(categories) - 0.5}]
set ytics ({ytics})
unset key
plot {', '.join(plot_items)}
"""
            )

        body = (
            'set multiplot layout 1,2 rowsfirst title "Candidate angles by joint and missed-model outcomes" font ",15"\n'
            + "\n".join(source_commands)
            + "\nunset multiplot"
        )
        return render_plot_pair(body, plot_dir, "joint_detection_angle_distributions")


def binned_detection_rate(
    values: np.ndarray, detected: np.ndarray, bin_edges: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    bin_indices = np.digitize(values, bin_edges, right=False) - 1
    counts = np.asarray(
        [np.sum(bin_indices == index) for index in range(len(bin_edges) - 1)],
        dtype=int,
    )
    rates = np.full(len(counts), np.nan, dtype=float)
    for index, count in enumerate(counts):
        if count:
            rates[index] = 100.0 * np.mean(detected[bin_indices == index])
    return rates, counts


def plot_detection_rate(
    angles: dict[str, np.ndarray],
    detections: dict[str, np.ndarray],
    plot_dir: Path,
) -> list[Path]:
    bin_edges = np.arange(0.0, 20.01, 2.0)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    with tempfile.TemporaryDirectory(prefix="etch_angle_rate_") as temp_dir:
        panels: list[str] = []
        for source in ("lewis", "kanik"):
            source_angles = angles[source]
            fcn_rates, counts = binned_detection_rate(
                source_angles, detections["fcn"], bin_edges
            )
            dcc_rates, dcc_counts = binned_detection_rate(
                source_angles, detections["dcc"], bin_edges
            )
            if not np.array_equal(counts, dcc_counts):
                raise ValueError("Angle-bin counts unexpectedly differ by model")
            data_path = Path(temp_dir) / f"{source}_rates.dat"
            write_numeric_table(data_path, [centers, counts, fcn_rates, dcc_rates])
            y2_max = max(1.0, float(np.max(counts)) * 1.18)
            fcn_overall = 100.0 * np.mean(detections["fcn"])
            dcc_overall = 100.0 * np.mean(detections["dcc"])
            panels.append(
                f"""
set title "{SOURCE_LABELS[source]} candidate angles"
set xlabel "Angle from foil normal (degrees)"
set ylabel "Detected candidate observations (%)"
set y2label "Candidate observations per angle bin" textcolor rgb "#656b70"
set yrange [0:103]
set y2range [0:{y2_max:.10f}]
set ytics nomirror
set y2tics textcolor rgb "#656b70"
set boxwidth 1.72 absolute
set style fill solid 0.22 noborder
set key bottom left
plot {gnuplot_quote(data_path)} using 1:2 axes x1y2 with boxes fillcolor rgb "#aeb5bb" title "Candidates per bin", \\
     {gnuplot_quote(data_path)} using 1:3 axes x1y1 with linespoints pointtype 7 pointsize 0.75 linewidth 2.2 linecolor rgb "{METHOD_COLORS['fcn']}" title "FCN ({fcn_overall:.1f}% overall)", \\
     {gnuplot_quote(data_path)} using 1:4 axes x1y1 with linespoints pointtype 7 pointsize 0.75 linewidth 2.2 linecolor rgb "{METHOD_COLORS['dcc']}" title "DCC ({dcc_overall:.1f}% overall)"
"""
            )
        body = (
            'set multiplot layout 1,2 rowsfirst title "Candidate-weighted detection rate versus etch-pit angle" font ",15"\n'
            + "\n".join(panels)
            + "\nunset multiplot"
        )
        return render_plot_pair(body, plot_dir, "detection_rate_vs_angle")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-output",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help=f"Per-ID output CSV (default: {DEFAULT_DATA_PATH})",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=DEFAULT_SUMMARY_PATH,
        help=f"Summary JSON (default: {DEFAULT_SUMMARY_PATH})",
    )
    parser.add_argument(
        "--plot-candidate-output",
        type=Path,
        default=DEFAULT_PLOT_CANDIDATE_DATA_PATH,
        help=(
            "CSV containing the one or two lowest-error candidates used by the "
            f"plots (default: {DEFAULT_PLOT_CANDIDATE_DATA_PATH})"
        ),
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=DEFAULT_PLOT_DIR,
        help=f"PNG/PDF output directory (default: {DEFAULT_PLOT_DIR})",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    clean_ids, truth_coordinates = load_truth()
    rank_one_angles = {
        source: load_selected_angles(path, clean_ids)
        for source, path in ANGLE_PATHS.items()
    }
    plot_candidates = load_lowest_error_candidate_pairs(
        ANGLE_COMPARISON_PATH, clean_ids
    )
    predictions = {
        method: load_predictions(path) for method, path in METHOD_PATHS.items()
    }
    detections: dict[str, np.ndarray] = {}
    nearest_distances: dict[str, np.ndarray] = {}
    for method, method_predictions in predictions.items():
        detected, nearest = classify_clean_ids(truth_coordinates, method_predictions)
        validate_saved_classification(method, truth_coordinates, detected)
        detections[method] = detected
        nearest_distances[method] = nearest

    clean_id_indices = {
        clean_id: index for index, clean_id in enumerate(clean_ids)
    }
    plot_candidate_ids = np.asarray(
        [str(candidate["clean_id"]) for candidate in plot_candidates]
    )
    plot_candidate_indices = np.asarray(
        [clean_id_indices[clean_id] for clean_id in plot_candidate_ids], dtype=int
    )
    plot_candidate_angles = {
        "lewis": np.asarray(
            [float(candidate["lewis_angle_deg"]) for candidate in plot_candidates]
        ),
        "kanik": np.asarray(
            [float(candidate["kanik_angle_deg"]) for candidate in plot_candidates]
        ),
    }
    plot_candidate_detections = {
        method: detected[plot_candidate_indices]
        for method, detected in detections.items()
    }

    data_path = args.data_output.resolve()
    plot_candidate_data_path = args.plot_candidate_output.resolve()
    summary_path = args.summary_output.resolve()
    plot_dir = args.plot_dir.resolve()
    write_data(
        data_path,
        clean_ids,
        truth_coordinates,
        rank_one_angles,
        detections,
        nearest_distances,
    )
    write_plot_candidate_data(
        plot_candidate_data_path,
        plot_candidates,
        clean_id_indices,
        detections,
    )
    summary = create_summary(
        clean_ids,
        plot_candidate_ids,
        plot_candidate_angles,
        predictions,
        detections,
        plot_candidate_detections,
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, indent=2, sort_keys=True)
        output_file.write("\n")

    plot_paths: list[Path] = []
    plot_paths.extend(
        plot_detected_vs_missed(
            plot_candidate_ids,
            plot_candidate_angles,
            plot_candidate_detections,
            plot_dir,
        )
    )
    plot_paths.extend(
        plot_joint_outcomes(
            plot_candidate_ids,
            plot_candidate_angles,
            plot_candidate_detections,
            plot_dir,
        )
    )
    plot_paths.extend(
        plot_detection_rate(
            plot_candidate_angles, plot_candidate_detections, plot_dir
        )
    )

    joint_counts = summary["joint_detection_counts"]
    print(f"Wrote {len(clean_ids)} per-ID rows to {data_path}", file=sys.stderr)
    print(
        f"Wrote {len(plot_candidates)} lowest-error plot candidates to "
        f"{plot_candidate_data_path}",
        file=sys.stderr,
    )
    print(f"Wrote summary to {summary_path}", file=sys.stderr)
    print(
        "Joint outcomes: "
        f"both={joint_counts['detected_by_both']}, "
        f"FCN-only={joint_counts['fcn_only']}, "
        f"DCC-only={joint_counts['dcc_only']}, "
        f"neither={joint_counts['missed_by_both']}",
        file=sys.stderr,
    )
    for path in plot_paths:
        print(f"Wrote plot to {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
