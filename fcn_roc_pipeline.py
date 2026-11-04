#!/usr/bin/env python3
"""Pascal-only FCN operating-characteristic scan for the V1.1 cuts analysis."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import pickle
import platform
import shutil
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image, ImageChops, ImageDraw
from scipy import ndimage
from scipy.ndimage import gaussian_laplace
from scipy.spatial import cKDTree


FIRST_IMAGE = 251
LAST_IMAGE = 500
IMAGE_COUNT = 250
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
EDGE_MARGIN = 4.0
SIGMA = 8.0
MATCH_TOLERANCE = 12.0
PIXEL_SIZE_MM = 0.000254
WORKERS = 12

COARSE_THRESHOLDS = [
    0.001,
    0.005,
    0.01,
    0.02,
    0.05,
    0.075,
    *[round(0.10 + 0.025 * index, 3) for index in range(17)],
    0.55,
    0.60,
    0.70,
    0.80,
    0.90,
    0.95,
    0.98,
    0.99,
]
COARSE_ALPHAS = [0.0005, 0.00075, 0.001, 0.00125, 0.0015, 0.00175, 0.002, 0.0025, 0.003]
COARSE_BETAS = [value * 1e-6 for value in (2.5, 5, 7.5, 10, 12.5, 15, 20, 30)]

MIN_THRESHOLD_STEP = 0.005
MIN_ALPHA_STEP = 0.000125
MIN_BETA_STEP = 1.25e-6
MAX_REFINEMENT_ROUNDS = 3
MAX_REFINED_CONFIGS = 10_000

RESULT_FIELDS = [
    "config_id",
    "round",
    "front_threshold",
    "back_threshold",
    "sigma",
    "alpha",
    "beta",
    "front_predictions_after_cuts",
    "back_predictions_after_cuts",
    "aligned_predictions",
    "matched_predictions",
    "true_positives",
    "false_positives",
    "false_negatives",
    "truth_total",
    "precision",
    "recall",
    "f1",
    "false_positives_per_mm2",
    "classification_hash",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    parser.add_argument("--workers", type=int, default=WORKERS)
    return parser.parse_args()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def float_token(value: float) -> str:
    return f"{value:.12g}".replace("-", "m").replace(".", "p").replace("+", "")


def cache_path(run_root: Path, side: str, threshold: float, alpha: float, beta: float) -> Path:
    return (
        run_root
        / "cache"
        / "coordinates"
        / side
        / f"t_{float_token(threshold)}"
        / f"a_{float_token(alpha)}_b_{float_token(beta)}.npz"
    )


def extraction_audit_path(run_root: Path, side: str, threshold: float) -> Path:
    return run_root / "cache" / "extraction_audits" / side / f"t_{float_token(threshold)}.json"


def load_coordinate_module(run_root: Path):
    source_root = run_root / "inputs" / "source_snapshot"
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    from modules.dataset_tools import coordinates

    coordinates.img_metadata_location = str(
        source_root / "modules" / "dataset_tools" / "metadata" / "Image_location_metadata.txt"
    )
    return coordinates


def coordinate_system(run_root: Path):
    coordinates = load_coordinate_module(run_root)
    system = coordinates.Coord_system()
    system["d"].flipped = False
    system["d"].reverse = False
    system["dr"].flipped = True
    system["dr"].reverse = True
    return system


def global_from_local(system, side: str, local: tuple[int, int, int]) -> np.ndarray:
    if side == "front":
        c1_local = system.convert(local, "dc")
    elif side == "back":
        c1_local = system.convert(local, "drc")
    else:
        raise ValueError(side)
    return np.asarray(system["c1"].get_gxy(c1_local), dtype=np.float64)


def load_regions(run_root: Path) -> dict:
    return json.loads((run_root / "inputs" / "analysis_v2_regions.json").read_text(encoding="utf-8"))


def fiducial_reasons(global_xy: np.ndarray, local: tuple[int, int, int], regions: dict) -> list[str]:
    _, x_value, y_value = local
    reasons = []
    if (
        x_value < EDGE_MARGIN
        or x_value > IMAGE_WIDTH - EDGE_MARGIN
        or y_value < EDGE_MARGIN
        or y_value > IMAGE_HEIGHT - EDGE_MARGIN
    ):
        reasons.append("image_edge")

    holes = regions["alignment_holes"]
    radius = float(holes["radius_px"])
    for center in holes["centers"]:
        if math.dist(global_xy, center["coordinate"]) <= radius:
            reasons.append("alignment_hole")
            break

    physical = regions["physical_inconsistencies"]
    half_width = float(physical["width_px"]) / 2.0
    half_height = float(physical["height_px"]) / 2.0
    for feature in physical["features"]:
        found = False
        for feature_side in ("front", "back"):
            center = feature[f"{feature_side}_center"]
            if abs(global_xy[0] - center[0]) <= half_width and abs(global_xy[1] - center[1]) <= half_height:
                reasons.append("physical_inconsistency")
                found = True
                break
        if found:
            break
    return reasons


def centers_from_mask(mask: np.ndarray) -> np.ndarray:
    labels, count = ndimage.label(mask)
    if count == 0:
        return np.empty((0, 2), dtype=np.int16)
    centers = ndimage.center_of_mass(mask, labels, range(1, count + 1))
    return np.flip(np.asarray(centers), axis=-1).astype(np.int16)


def save_coordinate_cache(path: Path, global_points: list, local_points: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    global_array = np.asarray(global_points, dtype=np.float64).reshape((-1, 2))
    local_array = np.asarray(local_points, dtype=np.int16).reshape((-1, 3))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, global_points=global_array, local_points=local_array)
    os.replace(temporary, path)


def load_coordinate_cache(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return data["global_points"].astype(np.float64), data["local_points"].astype(np.int16)


def extract_threshold_group(task: dict) -> dict:
    run_root = Path(task["run_root"])
    side = task["side"]
    threshold = float(task["threshold"])
    requests = {float(alpha): [float(beta) for beta in betas] for alpha, betas in task["requests"].items()}
    needed = [
        (alpha, beta)
        for alpha, betas in requests.items()
        for beta in betas
        if not cache_path(run_root, side, threshold, alpha, beta).is_file()
    ]
    if not needed:
        return {"side": side, "threshold": threshold, "status": "cached", "seconds": 0.0}

    started = time.time()
    mask_path = run_root / "inputs" / f"{side}_probability_masks.npy"
    probability_maps = np.load(mask_path, mmap_mode="r", allow_pickle=False)
    if probability_maps.shape != (IMAGE_COUNT, IMAGE_HEIGHT, IMAGE_WIDTH, 1):
        raise ValueError(f"Unexpected {side} probability shape: {probability_maps.shape}")
    system = coordinate_system(run_root)
    regions = load_regions(run_root)

    points = {(alpha, beta): ([], []) for alpha, beta in needed}
    audit = {
        (alpha, beta): {
            "raw_count": 0,
            "retained_count": 0,
            "removed_count": 0,
            "removed_by_reason": defaultdict(int),
        }
        for alpha, beta in needed
    }
    needed_by_alpha = defaultdict(list)
    for alpha, beta in needed:
        needed_by_alpha[alpha].append(beta)

    for array_index in range(IMAGE_COUNT):
        probability = np.asarray(probability_maps[array_index, :, :, 0], dtype=np.float32)
        binary = probability >= threshold
        first_response = gaussian_laplace(binary.astype(np.float32, copy=False), sigma=SIGMA)
        image_number = FIRST_IMAGE + array_index
        for alpha, betas in needed_by_alpha.items():
            positive = first_response * (first_response > alpha)
            second_response = gaussian_laplace(positive, sigma=SIGMA)
            for beta in betas:
                centers = centers_from_mask(second_response < -beta)
                current_audit = audit[(alpha, beta)]
                current_audit["raw_count"] += int(len(centers))
                global_points, local_points = points[(alpha, beta)]
                for x_value, y_value in centers:
                    local = (image_number, int(x_value), int(y_value))
                    global_xy = global_from_local(system, side, local)
                    reasons = fiducial_reasons(global_xy, local, regions)
                    if reasons:
                        current_audit["removed_count"] += 1
                        for reason in set(reasons):
                            current_audit["removed_by_reason"][reason] += 1
                        continue
                    current_audit["retained_count"] += 1
                    global_points.append(global_xy.tolist())
                    local_points.append(local)

    for alpha, beta in needed:
        global_points, local_points = points[(alpha, beta)]
        save_coordinate_cache(cache_path(run_root, side, threshold, alpha, beta), global_points, local_points)

    audit_path = extraction_audit_path(run_root, side, threshold)
    existing = {}
    if audit_path.is_file():
        existing = json.loads(audit_path.read_text(encoding="utf-8")).get("configurations", {})
    for (alpha, beta), value in audit.items():
        key = f"a={alpha:.12g},b={beta:.12g}"
        value["removed_by_reason"] = dict(value["removed_by_reason"])
        existing[key] = value
    write_json(
        audit_path,
        {
            "side": side,
            "threshold": threshold,
            "sigma": SIGMA,
            "configurations": existing,
        },
    )
    return {
        "side": side,
        "threshold": threshold,
        "status": "complete",
        "configurations_written": len(needed),
        "seconds": time.time() - started,
    }


def config_id(parameters: tuple[float, float, float, float]) -> str:
    front_threshold, back_threshold, alpha, beta = parameters
    payload = {
        "front_threshold": float(front_threshold),
        "back_threshold": float(back_threshold),
        "sigma": SIGMA,
        "alpha": float(alpha),
        "beta": float(beta),
        "matching_tolerance_px": MATCH_TOLERANCE,
        "analysis": "Analysis_V1.1_updated_configs_and_cuts",
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:20]


def classify(front: np.ndarray, back: np.ndarray, truth: np.ndarray) -> tuple[dict, dict]:
    if len(front) == 0 or len(back) == 0:
        aligned_indices = np.empty(0, dtype=np.int32)
        aligned = np.empty((0, 2), dtype=np.float64)
    else:
        distances, _ = cKDTree(back).query(front, k=1)
        aligned_indices = np.flatnonzero(distances <= MATCH_TOLERANCE).astype(np.int32)
        aligned = front[aligned_indices]

    if len(aligned) == 0:
        matched_prediction_mask = np.zeros(0, dtype=bool)
        truth_match_mask = np.zeros(len(truth), dtype=bool)
    else:
        pred_distances, _ = cKDTree(truth).query(aligned, k=1)
        truth_distances, _ = cKDTree(aligned).query(truth, k=1)
        matched_prediction_mask = pred_distances <= MATCH_TOLERANCE
        truth_match_mask = truth_distances <= MATCH_TOLERANCE

    matched_aligned_offsets = np.flatnonzero(matched_prediction_mask).astype(np.int32)
    fp_aligned_offsets = np.flatnonzero(~matched_prediction_mask).astype(np.int32)
    tp_truth_indices = np.flatnonzero(truth_match_mask).astype(np.int32)
    fn_truth_indices = np.flatnonzero(~truth_match_mask).astype(np.int32)
    matched_predictions = int(len(matched_aligned_offsets))
    false_positives = int(len(fp_aligned_offsets))
    true_positives = int(len(tp_truth_indices))
    false_negatives = int(len(fn_truth_indices))
    precision = matched_predictions / len(aligned) if len(aligned) else 0.0
    recall = true_positives / len(truth) if len(truth) else 0.0
    f1 = 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)

    digest = hashlib.sha256()
    for array in (aligned_indices, matched_aligned_offsets, fp_aligned_offsets, tp_truth_indices, fn_truth_indices):
        digest.update(np.asarray(array, dtype="<i4").tobytes())
        digest.update(b"|")
    metrics = {
        "aligned_predictions": int(len(aligned)),
        "matched_predictions": matched_predictions,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "classification_hash": digest.hexdigest(),
    }
    classification = {
        "aligned_front_indices": aligned_indices.tolist(),
        "matched_aligned_offsets": matched_aligned_offsets.tolist(),
        "false_positive_aligned_offsets": fp_aligned_offsets.tolist(),
        "true_positive_truth_indices": tp_truth_indices.tolist(),
        "false_negative_truth_indices": fn_truth_indices.tolist(),
    }
    return metrics, classification


def load_truth(run_root: Path) -> tuple[list[str], np.ndarray]:
    records = json.loads((run_root / "inputs" / "Aligned_clean_with_ids.txt").read_text(encoding="utf-8"))
    ids = [record["id"] for record in records]
    coordinates = np.asarray([record["coordinate"] for record in records], dtype=np.float64)
    if len(ids) != 1232 or len(set(ids)) != 1232:
        raise ValueError(f"Expected 1232 unique active truth IDs, found {len(ids)}")
    return ids, coordinates


def result_row(
    parameters: tuple[float, float, float, float],
    round_number: int,
    front_count: int,
    back_count: int,
    metrics: dict,
    truth_count: int,
    area_mm2: float,
) -> dict:
    front_threshold, back_threshold, alpha, beta = parameters
    if metrics["true_positives"] + metrics["false_negatives"] != truth_count:
        raise RuntimeError("TP + FN invariant failed")
    if metrics["matched_predictions"] + metrics["false_positives"] != metrics["aligned_predictions"]:
        raise RuntimeError("matched predictions + FP invariant failed")
    return {
        "config_id": config_id(parameters),
        "round": round_number,
        "front_threshold": front_threshold,
        "back_threshold": back_threshold,
        "sigma": SIGMA,
        "alpha": alpha,
        "beta": beta,
        "front_predictions_after_cuts": front_count,
        "back_predictions_after_cuts": back_count,
        **metrics,
        "truth_total": truth_count,
        "false_positives_per_mm2": metrics["false_positives"] / area_mm2,
    }


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in RESULT_FIELDS})
    os.replace(temporary, path)


def read_rows(path: Path) -> list[dict]:
    integer_fields = {
        "round",
        "front_predictions_after_cuts",
        "back_predictions_after_cuts",
        "aligned_predictions",
        "matched_predictions",
        "true_positives",
        "false_positives",
        "false_negatives",
        "truth_total",
    }
    float_fields = {
        "front_threshold",
        "back_threshold",
        "sigma",
        "alpha",
        "beta",
        "precision",
        "recall",
        "f1",
        "false_positives_per_mm2",
    }
    rows = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            for field in integer_fields:
                row[field] = int(row[field])
            for field in float_fields:
                row[field] = float(row[field])
            rows.append(row)
    return rows


def evaluate_group(task: dict) -> dict:
    run_root = Path(task["run_root"])
    alpha = float(task["alpha"])
    beta = float(task["beta"])
    front_thresholds = [float(value) for value in task["front_thresholds"]]
    back_thresholds = [float(value) for value in task["back_thresholds"]]
    round_number = int(task.get("round", 0))
    area_mm2 = float(task["area_mm2"])
    output_stem = f"a_{float_token(alpha)}_b_{float_token(beta)}"
    output_csv = run_root / "results" / "coarse_parts" / f"{output_stem}.csv"
    output_classes = run_root / "results" / "classification_parts" / f"{output_stem}.jsonl.gz"
    if output_csv.is_file() and output_classes.is_file():
        return {"alpha": alpha, "beta": beta, "status": "cached", "rows": len(front_thresholds) * len(back_thresholds)}

    _, truth = load_truth(run_root)
    front_cache = {
        threshold: load_coordinate_cache(cache_path(run_root, "front", threshold, alpha, beta))[0]
        for threshold in front_thresholds
    }
    back_cache = {
        threshold: load_coordinate_cache(cache_path(run_root, "back", threshold, alpha, beta))[0]
        for threshold in back_thresholds
    }
    rows = []
    classifications = []
    for front_threshold in front_thresholds:
        front = front_cache[front_threshold]
        for back_threshold in back_thresholds:
            back = back_cache[back_threshold]
            parameters = (front_threshold, back_threshold, alpha, beta)
            metrics, classification = classify(front, back, truth)
            row = result_row(parameters, round_number, len(front), len(back), metrics, len(truth), area_mm2)
            rows.append(row)
            classifications.append({"config_id": row["config_id"], **classification})
    write_rows(output_csv, rows)
    output_classes.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_classes.with_suffix(output_classes.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=6) as stream:
        for record in classifications:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")
    os.replace(temporary, output_classes)
    return {"alpha": alpha, "beta": beta, "status": "complete", "rows": len(rows)}


def evaluate_refined_chunk(task: dict) -> dict:
    run_root = Path(task["run_root"])
    configurations = [tuple(map(float, item)) for item in task["configurations"]]
    round_number = int(task["round"])
    part_number = int(task["part"])
    area_mm2 = float(task["area_mm2"])
    output_csv = run_root / "results" / f"refinement_round_{round_number}" / f"part_{part_number:04d}.csv"
    output_classes = run_root / "results" / f"refinement_round_{round_number}" / f"part_{part_number:04d}.jsonl.gz"
    if output_csv.is_file() and output_classes.is_file():
        return {"part": part_number, "status": "cached", "rows": len(configurations)}
    _, truth = load_truth(run_root)
    rows = []
    classes = []
    cache = {}
    for parameters in configurations:
        front_threshold, back_threshold, alpha, beta = parameters
        front_key = ("front", front_threshold, alpha, beta)
        back_key = ("back", back_threshold, alpha, beta)
        if front_key not in cache:
            cache[front_key] = load_coordinate_cache(cache_path(run_root, *front_key))[0]
        if back_key not in cache:
            cache[back_key] = load_coordinate_cache(cache_path(run_root, *back_key))[0]
        front = cache[front_key]
        back = cache[back_key]
        metrics, classification = classify(front, back, truth)
        row = result_row(parameters, round_number, len(front), len(back), metrics, len(truth), area_mm2)
        rows.append(row)
        classes.append({"config_id": row["config_id"], **classification})
    write_rows(output_csv, rows)
    output_classes.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_classes.with_suffix(output_classes.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=6) as stream:
        for record in classes:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")
    os.replace(temporary, output_classes)
    return {"part": part_number, "status": "complete", "rows": len(rows)}


def polygon_corners(system, side: str, image_number: int) -> list[tuple[float, float]]:
    corners = (
        (EDGE_MARGIN, EDGE_MARGIN),
        (IMAGE_WIDTH - EDGE_MARGIN, EDGE_MARGIN),
        (IMAGE_WIDTH - EDGE_MARGIN, IMAGE_HEIGHT - EDGE_MARGIN),
        (EDGE_MARGIN, IMAGE_HEIGHT - EDGE_MARGIN),
    )
    return [tuple(global_from_local(system, side, (image_number, int(x), int(y)))) for x, y in corners]


def calculate_area(run_root: Path) -> dict:
    output = run_root / "results" / "area_audit.json"
    if output.is_file():
        return json.loads(output.read_text(encoding="utf-8"))
    system = coordinate_system(run_root)
    regions = load_regions(run_root)
    front_polygons = [polygon_corners(system, "front", image) for image in range(FIRST_IMAGE, LAST_IMAGE + 1)]
    back_polygons = [polygon_corners(system, "back", image) for image in range(FIRST_IMAGE, LAST_IMAGE + 1)]
    all_points = np.asarray([point for polygon in front_polygons + back_polygons for point in polygon])
    padding = int(regions["alignment_holes"]["radius_px"] + 10)
    min_x = math.floor(float(np.min(all_points[:, 0]))) - padding
    min_y = math.floor(float(np.min(all_points[:, 1]))) - padding
    max_x = math.ceil(float(np.max(all_points[:, 0]))) + padding
    max_y = math.ceil(float(np.max(all_points[:, 1]))) + padding
    size = (max_x - min_x + 1, max_y - min_y + 1)

    def rasterize(polygons):
        image = Image.new("1", size, 0)
        draw = ImageDraw.Draw(image)
        for polygon in polygons:
            draw.polygon([(x - min_x, y - min_y) for x, y in polygon], fill=1)
        return image

    front_mask = rasterize(front_polygons)
    back_mask = rasterize(back_polygons)
    common = ImageChops.logical_and(front_mask, back_mask)
    front_pixels = int(np.count_nonzero(np.asarray(front_mask)))
    back_pixels = int(np.count_nonzero(np.asarray(back_mask)))
    common_before_pixels = int(np.count_nonzero(np.asarray(common)))

    draw = ImageDraw.Draw(common)
    radius = float(regions["alignment_holes"]["radius_px"])
    for center in regions["alignment_holes"]["centers"]:
        x_value, y_value = center["coordinate"]
        draw.ellipse(
            (
                x_value - radius - min_x,
                y_value - radius - min_y,
                x_value + radius - min_x,
                y_value + radius - min_y,
            ),
            fill=0,
        )
    physical = regions["physical_inconsistencies"]
    half_width = float(physical["width_px"]) / 2.0
    half_height = float(physical["height_px"]) / 2.0
    for feature in physical["features"]:
        for side in ("front", "back"):
            x_value, y_value = feature[f"{side}_center"]
            draw.rectangle(
                (
                    x_value - half_width - min_x,
                    y_value - half_height - min_y,
                    x_value + half_width - min_x,
                    y_value + half_height - min_y,
                ),
                fill=0,
            )
    effective_pixels = int(np.count_nonzero(np.asarray(common)))
    pixel_area = PIXEL_SIZE_MM**2
    audit = {
        "method": "one-C1-pixel rasterization of the common front/back 4-pixel-interior footprint, followed by V1.1 alignment-hole and physical-inconsistency exclusions",
        "pixel_size_mm": PIXEL_SIZE_MM,
        "raster_resolution_global_pixels": 1,
        "image_numbers": [FIRST_IMAGE, LAST_IMAGE],
        "image_count_per_side": IMAGE_COUNT,
        "front_interior_union_pixels": front_pixels,
        "back_interior_union_pixels": back_pixels,
        "common_interior_pixels_before_global_regions": common_before_pixels,
        "effective_analyzed_pixels": effective_pixels,
        "front_interior_union_mm2": front_pixels * pixel_area,
        "back_interior_union_mm2": back_pixels * pixel_area,
        "common_interior_before_global_regions_mm2": common_before_pixels * pixel_area,
        "effective_analyzed_area_mm2": effective_pixels * pixel_area,
        "nominal_summed_field_area_mm2": IMAGE_COUNT * IMAGE_WIDTH * IMAGE_HEIGHT * pixel_area,
        "fiducial_region_sha256": sha256_file(run_root / "inputs" / "analysis_v2_regions.json"),
    }
    if audit["effective_analyzed_area_mm2"] <= 0:
        raise RuntimeError("Effective analyzed area is nonpositive")
    write_json(output, audit)
    return audit


def run_parallel(tasks: list[dict], function, workers: int, label: str) -> list[dict]:
    results = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(function, task): task for task in tasks}
        completed = 0
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            completed += 1
            print(f"{label}: {completed}/{len(tasks)} {result}", flush=True)
    return results


def combine_coarse_rows(run_root: Path) -> list[dict]:
    paths = sorted((run_root / "results" / "coarse_parts").glob("*.csv"))
    rows = [row for path in paths for row in read_rows(path)]
    expected = len(COARSE_THRESHOLDS) ** 2 * len(COARSE_ALPHAS) * len(COARSE_BETAS)
    if len(rows) != expected:
        raise RuntimeError(f"Expected {expected} coarse rows, found {len(rows)}")
    rows.sort(key=lambda row: row["config_id"])
    write_rows(run_root / "results" / "coarse_results.csv", rows)
    return rows


def nondominated_froc(rows: list[dict]) -> list[dict]:
    ordered = sorted(rows, key=lambda row: (row["false_positives_per_mm2"], -row["recall"], -row["precision"], row["config_id"]))
    frontier = []
    best_recall = -1.0
    for row in ordered:
        if row["recall"] > best_recall + 1e-15:
            frontier.append(row)
            best_recall = row["recall"]
    return frontier


def nondominated_pr(rows: list[dict]) -> list[dict]:
    ordered = sorted(rows, key=lambda row: (-row["recall"], -row["precision"], row["config_id"]))
    retained = []
    best_precision = -1.0
    for row in ordered:
        if row["precision"] > best_precision + 1e-15:
            retained.append(row)
            best_precision = row["precision"]
    return sorted(retained, key=lambda row: (row["recall"], row["precision"]))


def nearest_neighbors(values: list[float], value: float) -> list[float]:
    ordered = sorted(set(values))
    index = ordered.index(value)
    result = []
    if index > 0:
        result.append(ordered[index - 1])
    if index + 1 < len(ordered):
        result.append(ordered[index + 1])
    return result


def propose_refinements(rows: list[dict], evaluated: set[tuple], remaining_budget: int) -> list[tuple]:
    froc = nondominated_froc(rows)
    pr = nondominated_pr(rows)
    frontier = {row["config_id"]: row for row in froc + pr}
    near = []
    frontier_values = list(frontier.values())
    for row in rows:
        if row["config_id"] in frontier:
            near.append(row)
            continue
        if any(
            abs(row["recall"] - point["recall"]) <= 0.005
            and abs(row["false_positives"] - point["false_positives"]) <= 12
            for point in frontier_values
        ):
            near.append(row)
    near.sort(key=lambda row: (-row["f1"], row["false_positives"], -row["true_positives"], row["config_id"]))
    near = near[: min(1000, len(near))]

    dimension_fields = ["front_threshold", "back_threshold", "alpha", "beta"]
    minimum_steps = [MIN_THRESHOLD_STEP, MIN_THRESHOLD_STEP, MIN_ALPHA_STEP, MIN_BETA_STEP]
    dimension_values = {field: sorted({float(row[field]) for row in rows}) for field in dimension_fields}
    by_parameters = {
        (
            float(row["front_threshold"]),
            float(row["back_threshold"]),
            float(row["alpha"]),
            float(row["beta"]),
        ): row
        for row in rows
    }
    proposals = set()
    for row in near:
        base = [float(row[field]) for field in dimension_fields]
        for dimension, (field, minimum_step) in enumerate(zip(dimension_fields, minimum_steps)):
            for neighbor in nearest_neighbors(dimension_values[field], base[dimension]):
                if abs(neighbor - base[dimension]) <= minimum_step + 1e-15:
                    continue
                endpoint = base.copy()
                endpoint[dimension] = neighbor
                endpoint_row = by_parameters.get(tuple(endpoint))
                if endpoint_row and endpoint_row["classification_hash"] == row["classification_hash"]:
                    continue
                candidate = base.copy()
                candidate[dimension] = (base[dimension] + neighbor) / 2.0
                candidate_tuple = tuple(candidate)
                if candidate_tuple not in evaluated:
                    proposals.add(candidate_tuple)
                if len(proposals) >= remaining_budget:
                    break
            if len(proposals) >= remaining_budget:
                break
        if len(proposals) >= remaining_budget:
            break
    return sorted(proposals)


def frontier_gaps(rows: list[dict], area_mm2: float) -> dict:
    frontier = nondominated_froc(rows)
    recall_gaps = [abs(b["recall"] - a["recall"]) for a, b in zip(frontier, frontier[1:])]
    fp_density_gaps = [
        abs(b["false_positives_per_mm2"] - a["false_positives_per_mm2"])
        for a, b in zip(frontier, frontier[1:])
    ]
    return {
        "frontier_points": len(frontier),
        "maximum_recall_gap": max(recall_gaps, default=0.0),
        "maximum_fp_per_mm2_gap": max(fp_density_gaps, default=0.0),
        "target_recall_gap": 0.005,
        "target_fp_per_mm2_gap": (0.05 * IMAGE_COUNT) / area_mm2,
    }


def ensure_coordinate_requests(run_root: Path, configurations: list[tuple], workers: int) -> None:
    grouped = {"front": defaultdict(lambda: defaultdict(set)), "back": defaultdict(lambda: defaultdict(set))}
    for front_threshold, back_threshold, alpha, beta in configurations:
        grouped["front"][front_threshold][alpha].add(beta)
        grouped["back"][back_threshold][alpha].add(beta)
    tasks = []
    for side in ("front", "back"):
        for threshold, alpha_map in grouped[side].items():
            requests = {alpha: sorted(betas) for alpha, betas in alpha_map.items()}
            tasks.append({"run_root": str(run_root), "side": side, "threshold": threshold, "requests": requests})
    run_parallel(tasks, extract_threshold_group, workers, "refined extraction")


def combine_refinement_rows(run_root: Path, round_number: int) -> list[dict]:
    paths = sorted((run_root / "results" / f"refinement_round_{round_number}").glob("part_*.csv"))
    rows = [row for path in paths for row in read_rows(path)]
    write_rows(run_root / "results" / f"refinement_round_{round_number}.csv", rows)
    return rows


def collapse_frontier(frontier: list[dict], x_field: str, y_field: str) -> list[dict]:
    groups = defaultdict(list)
    for row in frontier:
        groups[(row[x_field], row[y_field])].append(row)
    collapsed = []
    for (x_value, y_value), group in groups.items():
        representative = max(group, key=lambda row: (row["f1"], -row["false_positives"], row["config_id"]))
        collapsed.append({**representative, "contributing_config_ids": ";".join(sorted(row["config_id"] for row in group))})
    return sorted(collapsed, key=lambda row: (row[x_field], row[y_field]))


def write_frontier(path: Path, rows: list[dict]) -> None:
    fields = RESULT_FIELDS + ["contributing_config_ids"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def find_classification_record(run_root: Path, target_id: str) -> dict:
    paths = sorted((run_root / "results").glob("classification_parts/*.jsonl.gz"))
    paths += sorted((run_root / "results").glob("refinement_round_*/*.jsonl.gz"))
    for path in paths:
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                if record["config_id"] == target_id:
                    return record
    raise KeyError(target_id)


def save_optimal_details(run_root: Path, optimal: dict) -> None:
    parameters = (
        optimal["front_threshold"],
        optimal["back_threshold"],
        optimal["alpha"],
        optimal["beta"],
    )
    front, front_local = load_coordinate_cache(
        cache_path(run_root, "front", parameters[0], parameters[2], parameters[3])
    )
    back, back_local = load_coordinate_cache(
        cache_path(run_root, "back", parameters[1], parameters[2], parameters[3])
    )
    truth_ids, truth = load_truth(run_root)
    metrics, classification = classify(front, back, truth)
    if metrics["classification_hash"] != optimal["classification_hash"]:
        raise RuntimeError("Optimal classification hash does not reproduce")
    aligned_front_indices = np.asarray(classification["aligned_front_indices"], dtype=int)
    aligned = front[aligned_front_indices]
    aligned_local = front_local[aligned_front_indices]
    matched_offsets = set(classification["matched_aligned_offsets"])
    fp_offsets = set(classification["false_positive_aligned_offsets"])
    predictions = []
    for offset, (global_xy, local) in enumerate(zip(aligned, aligned_local)):
        prediction_id = f"P{offset:05d}_I{int(local[0]):03d}_X{int(local[1]):03d}_Y{int(local[2]):03d}"
        predictions.append(
            {
                "id": prediction_id,
                "global_coordinate": global_xy.tolist(),
                "front_local_coordinate": local.astype(int).tolist(),
                "classification": "matched" if offset in matched_offsets else "false_positive",
            }
        )
    details = {
        "selection_rule": "maximum F1; ties: fewer FP, then more TP, then config_id",
        "result": optimal,
        "true_positive_truth_ids": [truth_ids[index] for index in classification["true_positive_truth_indices"]],
        "false_negative_truth_ids": [truth_ids[index] for index in classification["false_negative_truth_indices"]],
        "aligned_predictions": predictions,
        "front_predictions_after_cuts": int(len(front)),
        "back_predictions_after_cuts": int(len(back)),
    }
    write_json(run_root / "results" / "optimal_working_point.json", details)


def make_plots(run_root: Path, rows: list[dict]) -> None:
    plot_dir = run_root / "results" / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    froc = collapse_frontier(nondominated_froc(rows), "false_positives_per_mm2", "recall")
    pr = collapse_frontier(nondominated_pr(rows), "recall", "precision")
    write_frontier(run_root / "results" / "froc_frontier.csv", froc)
    write_frontier(run_root / "results" / "precision_recall_frontier.csv", pr)

    legend_handles = [
        Line2D(
            [],
            [],
            linestyle="none",
            marker="o",
            markersize=4.5,
            markerfacecolor="#24577A",
            markeredgecolor="#24577A",
            label="Evaluated configurations",
        ),
        Line2D(
            [],
            [],
            color="black",
            linewidth=1.3,
            marker="o",
            markersize=3.5,
            label="Nondominated frontier",
        ),
    ]

    def apply_paper_style(fig, axis) -> None:
        axis.set_axisbelow(True)
        axis.tick_params(
            axis="both",
            which="major",
            direction="in",
            top=True,
            right=True,
            length=4.0,
            width=0.8,
            labelsize=9,
        )
        axis.tick_params(
            axis="both",
            which="minor",
            direction="in",
            top=True,
            right=True,
            length=2.5,
            width=0.6,
        )
        for spine in axis.spines.values():
            spine.set_linewidth(0.8)
        axis.xaxis.label.set_size(10.5)
        axis.yaxis.label.set_size(10.5)
        axis.set_title(axis.get_title(), fontsize=11.5, pad=7.0)
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.008),
            borderaxespad=0.0,
            ncol=2,
            frameon=False,
            fontsize=9,
            handlelength=2.0,
            handletextpad=0.6,
            columnspacing=1.8,
        )
        fig.subplots_adjust(left=0.105, right=0.985, bottom=0.175, top=0.91)

    positive_froc_rows = [
        row for row in rows if row["false_positives_per_mm2"] > 0.0
    ]
    positive_froc_frontier = [
        row for row in froc if row["false_positives_per_mm2"] > 0.0
    ]
    if not positive_froc_rows or not positive_froc_frontier:
        raise RuntimeError("A logarithmic FROC x-axis requires positive FP/mm^2 values")
    froc_x_min = min(
        row["false_positives_per_mm2"] for row in positive_froc_rows
    )
    froc_x_max = max(
        row["false_positives_per_mm2"] for row in positive_froc_rows
    )

    fig, axis = plt.subplots(figsize=(7.2, 5.2))
    axis.scatter(
        [row["false_positives_per_mm2"] for row in positive_froc_rows],
        [row["recall"] for row in positive_froc_rows],
        s=4,
        alpha=0.08,
        color="steelblue",
        rasterized=True,
        label="evaluated configurations",
    )
    axis.plot(
        [row["false_positives_per_mm2"] for row in positive_froc_frontier],
        [row["recall"] for row in positive_froc_frontier],
        color="black",
        linewidth=1.5,
        marker="o",
        markersize=3,
        label="nondominated frontier",
    )
    axis.set_xlabel(r"False positives per mm$^2$")
    axis.set_ylabel("Truth recall (TP / 1232)")
    axis.set_title("FCN free-response operating characteristic — log FP density")
    axis.set_xscale("log")
    axis.set_xlim(froc_x_min / 1.10, froc_x_max * 1.10)
    axis.set_ylim(0.0, 1.0)
    axis.grid(alpha=0.18, linewidth=0.5, which="both")
    apply_paper_style(fig, axis)
    fig.savefig(
        plot_dir / "froc.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.03,
    )
    fig.savefig(
        plot_dir / "froc.pdf",
        bbox_inches="tight",
        pad_inches=0.03,
    )
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7.2, 5.2))
    axis.scatter(
        [row["recall"] for row in rows],
        [row["precision"] for row in rows],
        s=4,
        alpha=0.08,
        color="steelblue",
        rasterized=True,
        label="evaluated configurations",
    )
    axis.plot(
        [row["recall"] for row in pr],
        [row["precision"] for row in pr],
        color="black",
        linewidth=1.5,
        marker="o",
        markersize=3,
        label="nondominated frontier",
    )
    axis.set_xlabel("Truth recall (TP / 1232)")
    axis.set_ylabel("Prediction-side precision")
    axis.set_title("FCN precision–recall characteristic — cubic precision scale")
    axis.set_xlim(0.0, 1.0)
    precision_power = 3.0
    axis.set_yscale(
        "function",
        functions=(
            lambda values: np.power(values, precision_power),
            lambda values: np.power(values, 1.0 / precision_power),
        ),
    )
    axis.set_ylim(0.0, 1.01)
    axis.set_yticks([0.0, 0.5, 0.7, 0.8, 0.9, 1.0])
    axis.grid(alpha=0.18, linewidth=0.5, which="both")
    apply_paper_style(fig, axis)
    fig.savefig(
        plot_dir / "precision_recall.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.03,
    )
    fig.savefig(
        plot_dir / "precision_recall.pdf",
        bbox_inches="tight",
        pad_inches=0.03,
    )
    plt.close(fig)


def inventory_manifest(run_root: Path, model_name: str) -> dict:
    inputs = []
    for path in sorted((run_root / "inputs").rglob("*")):
        if path.is_file():
            inputs.append(
                {
                    "relative_path": str(path.relative_to(run_root)),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return {
        "schema_version": 1,
        "run_root": str(run_root),
        "model": model_name,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": platform.node(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": __import__("scipy").__version__,
        "inputs": inputs,
        "parameter_grid": {
            "sigma": SIGMA,
            "thresholds": COARSE_THRESHOLDS,
            "alphas": COARSE_ALPHAS,
            "betas": COARSE_BETAS,
            "independent_front_back_thresholds": True,
        },
        "analysis": {
            "truth_count": 1232,
            "matching_tolerance_px": MATCH_TOLERANCE,
            "matching_boundary": "inclusive",
            "fiducials": "Analysis_V1.1_updated_configs_and_cuts",
            "froc_x_axis": "false positives per effective analyzed square millimetre",
            "working_point": "maximum F1; ties fewer FP, more TP, config_id",
        },
    }


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()
    if not run_root.is_dir():
        raise FileNotFoundError(run_root)
    if args.workers < 1 or args.workers > WORKERS:
        raise ValueError(f"workers must be between 1 and {WORKERS}")
    (run_root / "logs").mkdir(exist_ok=True)
    (run_root / "results").mkdir(exist_ok=True)
    write_json(
        run_root / "run_manifest.json",
        inventory_manifest(run_root, args.model_name),
    )
    area = calculate_area(run_root)
    area_mm2 = float(area["effective_analyzed_area_mm2"])
    print(f"Effective analyzed area: {area_mm2:.9f} mm^2", flush=True)

    if args.mode == "smoke":
        thresholds = [0.5]
        alphas = [0.001]
        betas = [1e-5]
    else:
        thresholds = COARSE_THRESHOLDS
        alphas = COARSE_ALPHAS
        betas = COARSE_BETAS

    extraction_tasks = []
    for side in ("front", "back"):
        for threshold in thresholds:
            extraction_tasks.append(
                {
                    "run_root": str(run_root),
                    "side": side,
                    "threshold": threshold,
                    "requests": {alpha: betas for alpha in alphas},
                }
            )
    run_parallel(extraction_tasks, extract_threshold_group, args.workers, "coarse extraction")

    if args.mode == "smoke":
        _, truth = load_truth(run_root)
        front = load_coordinate_cache(cache_path(run_root, "front", 0.5, 0.001, 1e-5))[0]
        back = load_coordinate_cache(cache_path(run_root, "back", 0.5, 0.001, 1e-5))[0]
        metrics, _ = classify(front, back, truth)
        smoke = result_row((0.5, 0.5, 0.001, 1e-5), 0, len(front), len(back), metrics, len(truth), area_mm2)
        write_json(run_root / "results" / "smoke_result.json", smoke)
        print(json.dumps(smoke, indent=2), flush=True)
        return

    evaluation_tasks = [
        {
            "run_root": str(run_root),
            "alpha": alpha,
            "beta": beta,
            "front_thresholds": thresholds,
            "back_thresholds": thresholds,
            "round": 0,
            "area_mm2": area_mm2,
        }
        for alpha in alphas
        for beta in betas
    ]
    run_parallel(evaluation_tasks, evaluate_group, args.workers, "coarse evaluation")
    all_rows = combine_coarse_rows(run_root)
    evaluated = {
        (row["front_threshold"], row["back_threshold"], row["alpha"], row["beta"])
        for row in all_rows
    }
    total_refined = 0
    refinement_history = []
    for round_number in range(1, MAX_REFINEMENT_ROUNDS + 1):
        gaps_before = frontier_gaps(all_rows, area_mm2)
        remaining = MAX_REFINED_CONFIGS - total_refined
        if remaining <= 0:
            break
        proposals = propose_refinements(all_rows, evaluated, remaining)
        if not proposals:
            refinement_history.append(
                {"round": round_number, "proposed": 0, "reason": "no new midpoint proposals", "gaps_before": gaps_before}
            )
            break
        ensure_coordinate_requests(run_root, proposals, args.workers)
        chunk_size = max(1, math.ceil(len(proposals) / args.workers))
        chunks = [proposals[index : index + chunk_size] for index in range(0, len(proposals), chunk_size)]
        tasks = [
            {
                "run_root": str(run_root),
                "configurations": chunk,
                "round": round_number,
                "part": part,
                "area_mm2": area_mm2,
            }
            for part, chunk in enumerate(chunks)
        ]
        run_parallel(tasks, evaluate_refined_chunk, args.workers, f"refinement {round_number} evaluation")
        new_rows = combine_refinement_rows(run_root, round_number)
        all_rows.extend(new_rows)
        all_rows.sort(key=lambda row: row["config_id"])
        evaluated.update(proposals)
        total_refined += len(proposals)
        gaps_after = frontier_gaps(all_rows, area_mm2)
        refinement_history.append(
            {
                "round": round_number,
                "proposed": len(proposals),
                "evaluated_total": len(all_rows),
                "gaps_before": gaps_before,
                "gaps_after": gaps_after,
            }
        )
        target_met = (
            gaps_after["maximum_recall_gap"] <= gaps_after["target_recall_gap"]
            and gaps_after["maximum_fp_per_mm2_gap"] <= gaps_after["target_fp_per_mm2_gap"]
        )
        if target_met:
            break

    # Deduplicate in case a resumed/refined configuration appeared twice.
    unique_rows = {}
    for row in all_rows:
        unique_rows[row["config_id"]] = row
    all_rows = sorted(unique_rows.values(), key=lambda row: row["config_id"])
    write_rows(run_root / "results" / "all_results.csv", all_rows)
    optimal = max(
        all_rows,
        key=lambda row: (
            row["f1"],
            -row["false_positives"],
            row["true_positives"],
            tuple(-ord(char) for char in row["config_id"]),
        ),
    )
    save_optimal_details(run_root, optimal)
    make_plots(run_root, all_rows)
    gaps = frontier_gaps(all_rows, area_mm2)
    summary = {
        "status": "complete",
        "model": args.model_name,
        "truth_count": 1232,
        "matching_tolerance_px": MATCH_TOLERANCE,
        "matching_boundary": "inclusive",
        "effective_analyzed_area_mm2": area_mm2,
        "coarse_configurations": len(COARSE_THRESHOLDS) ** 2 * len(COARSE_ALPHAS) * len(COARSE_BETAS),
        "refined_configurations": total_refined,
        "total_unique_configurations": len(all_rows),
        "optimal_working_point": optimal,
        "frontier_gaps": gaps,
        "refinement_history": refinement_history,
        "outputs": {
            "all_results": "results/all_results.csv",
            "froc_frontier": "results/froc_frontier.csv",
            "precision_recall_frontier": "results/precision_recall_frontier.csv",
            "optimal_details": "results/optimal_working_point.json",
            "area_audit": "results/area_audit.json",
            "plots": "results/plots",
        },
    }
    write_json(run_root / "results" / "summary.json", summary)

    checksums = []
    for path in sorted((run_root / "results").rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            checksums.append(f"{sha256_file(path)}  {path.relative_to(run_root)}")
    (run_root / "results" / "SHA256SUMS").write_text("\n".join(checksums) + "\n", encoding="utf-8")
    print("FINAL SUMMARY", json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

