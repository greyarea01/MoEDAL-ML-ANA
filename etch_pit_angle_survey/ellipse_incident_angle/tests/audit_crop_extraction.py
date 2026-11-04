#!/usr/bin/env python3
"""Audit the crop-extraction stage without running segmentation or fitting."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np


PIPELINE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PIPELINE_DIR.parents[1]
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from ellipse_fit import FitConfig, extract_valid_crop  # noqa: E402


MANIFEST_PATH = PIPELINE_DIR / "inputs/clean_surface_locations.csv"
IMAGE_PATHS = {
    "C1": REPO_ROOT / "numpy_data_eval/preprocessed_clean_eval.npy",
    "C2": REPO_ROOT / "numpy_data_eval/clean_2_eval.npy",
}
FIRST_IMAGE_NUMBER = 251
OUTPUT_PATH = Path(tempfile.gettempdir()) / "etch_pit_crop_extraction_audit.json"


def load_manifest() -> list[dict[str, str]]:
    with MANIFEST_PATH.open(newline="", encoding="utf-8") as input_file:
        return list(csv.DictReader(input_file))


def audit_surface(
    surface: str,
    rows: list[dict[str, str]],
    config: FitConfig,
) -> dict[str, object]:
    images = np.load(IMAGE_PATHS[surface], mmap_mode="r")
    key = surface.lower()
    edge_counts: Counter[str] = Counter()
    corner_counts: Counter[str] = Counter()
    full_crop_count = 0
    partial_crop_count = 0
    coordinate_outside_count = 0
    empty_crop_count = 0
    valid_pixel_counts: list[int] = []
    partial_ids: list[str] = []

    for row in rows:
        image_number = int(float(row[f"{key}_image"]))
        image_index = image_number - FIRST_IMAGE_NUMBER
        source = images[image_index]
        local_x = float(row[f"{key}_local_x"])
        local_y = float(row[f"{key}_local_y"])
        rounded_x = int(round(local_x))
        rounded_y = int(round(local_y))
        height, width = source.shape
        radius = config.crop_radius_px

        crosses_left = rounded_x - radius < 0
        crosses_right = rounded_x + radius >= width
        crosses_top = rounded_y - radius < 0
        crosses_bottom = rounded_y + radius >= height
        crossings = {
            "left": crosses_left,
            "right": crosses_right,
            "top": crosses_top,
            "bottom": crosses_bottom,
        }
        for edge, crossed in crossings.items():
            if crossed:
                edge_counts[edge] += 1

        vertical_edges = [name for name in ("top", "bottom") if crossings[name]]
        horizontal_edges = [name for name in ("left", "right") if crossings[name]]
        for vertical in vertical_edges:
            for horizontal in horizontal_edges:
                corner_counts[f"{vertical}_{horizontal}"] += 1

        coordinate_inside = 0 <= rounded_x < width and 0 <= rounded_y < height
        if not coordinate_inside:
            coordinate_outside_count += 1

        crop = extract_valid_crop(source, local_x, local_y, config)
        if crop.image.shape != (config.crop_size_px, config.crop_size_px):
            raise AssertionError(
                f"{row['clean_id']} {surface} returned crop shape {crop.image.shape}"
            )
        valid_count = int(np.sum(crop.valid_pixels))
        valid_pixel_counts.append(valid_count)
        if valid_count == 0:
            empty_crop_count += 1

        is_full = bool(np.all(crop.valid_pixels))
        crosses_any_edge = any(crossings.values())
        if is_full == crosses_any_edge:
            raise AssertionError(
                f"Boundary calculation disagrees with validity mask for "
                f"{row['clean_id']} {surface}"
            )
        if is_full:
            full_crop_count += 1
        else:
            partial_crop_count += 1
            partial_ids.append(row["clean_id"])

    total = len(rows)
    if full_crop_count + partial_crop_count != total:
        raise AssertionError("Full and partial crop counts do not cover all rows")

    valid_array = np.asarray(valid_pixel_counts, dtype=int)
    return {
        "total_pits": total,
        "full_49_by_49_crop_count": full_crop_count,
        "partial_crop_count": partial_crop_count,
        "full_crop_percent": 100.0 * full_crop_count / total,
        "partial_crop_percent": 100.0 * partial_crop_count / total,
        "stored_center_outside_source_count": coordinate_outside_count,
        "empty_crop_count": empty_crop_count,
        "edge_crossing_counts_not_mutually_exclusive": {
            edge: edge_counts[edge] for edge in ("left", "right", "top", "bottom")
        },
        "corner_crossing_counts": {
            corner: corner_counts[corner]
            for corner in ("top_left", "top_right", "bottom_left", "bottom_right")
        },
        "valid_pixel_count": {
            "full_crop": config.crop_size_px * config.crop_size_px,
            "minimum_observed": int(valid_array.min()),
            "mean": float(valid_array.mean()),
            "maximum_observed": int(valid_array.max()),
        },
        "partial_crop_clean_ids": partial_ids,
    }


def main() -> int:
    rows = load_manifest()
    config = FitConfig()
    result = {
        "definition": (
            "A full crop has all 49 by 49 requested pixels inside the real source "
            "image. A partial crop has at least one requested position outside it."
        ),
        "crop_radius_pixels": config.crop_radius_px,
        "crop_size_pixels": config.crop_size_px,
        "surfaces": {
            surface: audit_surface(surface, rows, config)
            for surface in ("C1", "C2")
        },
    }
    OUTPUT_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(f"Crop radius: {config.crop_radius_px} pixels")
    print(f"Requested crop: {config.crop_size_px} by {config.crop_size_px} pixels")
    for surface in ("C1", "C2"):
        summary = result["surfaces"][surface]
        print(
            f"{surface}: {summary['full_49_by_49_crop_count']} full crops, "
            f"{summary['partial_crop_count']} partial crops, "
            f"{summary['stored_center_outside_source_count']} centers outside source"
        )
        edges = summary["edge_crossing_counts_not_mutually_exclusive"]
        print(
            f"  edges: left {edges['left']}, right {edges['right']}, "
            f"top {edges['top']}, bottom {edges['bottom']}"
        )
    print(f"Audit JSON: {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
