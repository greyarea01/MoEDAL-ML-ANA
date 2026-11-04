#!/usr/bin/env python3
"""Survey the displacement between Kanik C1 and C2 clean etch-pit coordinates.

For every clean truth ID, the survey:

1. Recreates Kanik's historical aligned C1 list by requiring C1 matches on Cr
   and C2 at strict distance less than 67.2 micrometers (12 pixels) and applying
   the existing inverse-image cut.
2. Finds every eligible Kanik C1 candidate within the configured micrometer
   threshold of the Lewis C1 truth coordinate.
3. Finds every transformed C2 candidate within that threshold of each C1
   candidate.
4. Retains every feasible C1/Cr/C2 triplet and selects one deterministically for the
   one-row-per-truth output.

The selected triplet first minimizes the Lewis-truth-to-Kanik-C1 distance. If
that C1 candidate has multiple C2 matches, it then minimizes the C1-to-C2
distance. For the same C1/C2 pair, the nearest Cr point to C1 ranks first. All
alternatives remain available in ``all_candidate_pairs.csv``.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.spatial import cKDTree


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]

# Permit execution from either the repository root or this script's directory.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.dataset_tools import coordinates  # noqa: E402


DEFAULT_TRUTH_PATH = REPO_ROOT / "predicted_list/ids/Aligned_clean_with_ids.txt"
DEFAULT_C1_LOCAL_PATH = REPO_ROOT / "kanik/og_data/CleanTestCentroid.txt"
DEFAULT_C1_PATH = REPO_ROOT / "kanik/cleanTest.txt"
DEFAULT_CR_LOCAL_PATH = REPO_ROOT / "kanik/og_data/CleanReverseTestCentroid.txt"
DEFAULT_CR_PATH = REPO_ROOT / "kanik/cleanReverse.txt"
DEFAULT_C2_LOCAL_PATH = REPO_ROOT / "kanik/og_data/Clean2TestCentroid.txt"
DEFAULT_C2_GLOBAL_PATH = REPO_ROOT / "kanik/clean2.txt"
DEFAULT_ALIGNED_C1_PATH = REPO_ROOT / "kanik/clean_aligned.txt"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "data"
HISTORICAL_ALIGNMENT_THRESHOLD = 12.0
MINIMUM_C1_IMAGE = 251
MICROMETERS_PER_PIXEL = 5.6
C1_C2_SURFACE_SEPARATION_UM = 200.0


@dataclass(frozen=True)
class CandidatePair:
    clean_id: str
    truth_index: int
    c1_index: int
    cr_index: int
    c2_index: int
    cr_candidate_count_for_c1: int
    truth_to_c1_distance_px: float
    c1_to_cr_distance_px: float
    c1_to_c2_distance_px: float
    cr_to_c2_distance_px: float
    truth_to_cr_distance_px: float
    truth_to_c2_distance_px: float

    def sort_key(self) -> tuple[float, float, float, float, float, float, int, int, int]:
        return (
            self.truth_to_c1_distance_px,
            self.c1_to_c2_distance_px,
            self.truth_to_c2_distance_px,
            self.c1_to_cr_distance_px,
            self.truth_to_cr_distance_px,
            self.cr_to_c2_distance_px,
            self.c1_index,
            self.c2_index,
            self.cr_index,
        )


@dataclass(frozen=True)
class AlignmentMembership:
    c1_image_numbers: np.ndarray
    passes_image_cut: np.ndarray
    has_cr_match: np.ndarray
    has_c2_match: np.ndarray
    eligible: np.ndarray


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as input_file:
        return json.load(input_file)


def load_global_coordinates(path: Path) -> np.ndarray:
    points = np.asarray(load_json(path), dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"Expected an N x 2 coordinate list in {path}, got {points.shape}")
    return points


def load_truth(path: Path) -> tuple[list[str], np.ndarray]:
    records = load_json(path)
    clean_ids: list[str] = []
    points: list[list[float]] = []

    for index, record in enumerate(records):
        if not isinstance(record, dict) or "id" not in record or "coordinate" not in record:
            raise ValueError(f"Invalid truth record at index {index} in {path}")
        coordinate = record["coordinate"]
        if not isinstance(coordinate, list) or len(coordinate) != 2:
            raise ValueError(f"Invalid truth coordinate at index {index} in {path}")
        clean_ids.append(str(record["id"]))
        points.append([float(coordinate[0]), float(coordinate[1])])

    if len(clean_ids) != len(set(clean_ids)):
        raise ValueError(f"Clean IDs are not unique in {path}")
    return clean_ids, np.asarray(points, dtype=float)


def load_local_centroids(path: Path) -> np.ndarray:
    rows: list[tuple[int, float, float]] = []
    with path.open(encoding="utf-8", newline="") as input_file:
        reader = csv.reader(input_file, delimiter="\t")
        header = next(reader, None)
        if header is None or header[:3] != ["ImSlNo", "Xcoord", "Ycoord"]:
            raise ValueError(f"Unexpected centroid header in {path}: {header}")

        for line_number, row in enumerate(reader, start=2):
            if not row or all(not value.strip() for value in row):
                continue
            if len(row) < 3:
                raise ValueError(f"Invalid centroid row at {path}:{line_number}: {row}")
            image = int(float(row[0]))
            # Kanik's C1 and Cr files use image 0 as an empty/unassigned row.
            if image == 0:
                continue
            rows.append((image, float(row[1]), float(row[2])))

    return np.asarray(rows, dtype=float)


def create_coordinate_system() -> coordinates.Coord_system:
    # Coord_system loads metadata through a repository-relative default path.
    coordinates.img_metadata_location = str(
        REPO_ROOT / "modules/dataset_tools/metadata/Image_location_metadata.txt"
    )
    with contextlib.redirect_stdout(io.StringIO()):
        return coordinates.Coord_system()


def transform_local_to_global(
    local_points: np.ndarray,
    coordinate_system: coordinates.Coord_system,
    scan_name: str,
) -> np.ndarray:
    transformed = [
        coordinate_system[scan_name].get_gxy((int(image), x, y))
        for image, x, y in local_points
    ]
    return np.asarray(transformed, dtype=float)


def validate_transformed_coordinates(
    label: str,
    transformed: np.ndarray,
    existing: np.ndarray,
) -> float:
    if existing.shape != transformed.shape:
        raise ValueError(
            f"Transformed {label} coordinates and checked-in coordinates have "
            f"different shapes: {transformed.shape} versus {existing.shape}"
        )

    maximum_delta = float(np.max(np.abs(transformed - existing), initial=0.0))
    if maximum_delta > 1e-6:
        raise ValueError(
            f"{label} transformation does not reproduce the checked-in coordinates; "
            "maximum component difference is "
            f"{maximum_delta * MICROMETERS_PER_PIXEL} micrometers"
        )
    return maximum_delta


def reconstruct_alignment_membership(
    c1_points: np.ndarray,
    cr_points: np.ndarray,
    c2_points: np.ndarray,
    coordinate_system: coordinates.Coord_system,
    threshold: float,
) -> AlignmentMembership:
    # This follows the repository's Overlap membership semantics: a C1 point
    # survives when at least one point on each other clean surface is within
    # the strict Euclidean threshold. The inverse-image cut is also present in
    # the historical aligned list.
    has_cr_match = coordinates.Overlap(
        c1_points, cr_points, tolerance=threshold
    ).match_a > 0
    has_c2_match = coordinates.Overlap(
        c1_points, c2_points, tolerance=threshold
    ).match_a > 0
    c1_image_numbers = np.asarray(
        [coordinate_system["c1"].get_ixy(point)[0] for point in c1_points],
        dtype=int,
    )
    passes_image_cut = c1_image_numbers >= MINIMUM_C1_IMAGE
    eligible = passes_image_cut & has_cr_match & has_c2_match
    return AlignmentMembership(
        c1_image_numbers=c1_image_numbers,
        passes_image_cut=passes_image_cut,
        has_cr_match=has_cr_match,
        has_c2_match=has_c2_match,
        eligible=eligible,
    )


def validate_historical_alignment(
    c1_points: np.ndarray,
    membership: AlignmentMembership,
    expected_aligned_points: np.ndarray,
) -> None:
    recreated = c1_points[membership.eligible]
    if recreated.shape != expected_aligned_points.shape:
        raise ValueError(
            "Recreated Kanik aligned list and kanik/clean_aligned.txt have different "
            f"shapes: {recreated.shape} versus {expected_aligned_points.shape}"
        )
    if not np.array_equal(recreated, expected_aligned_points):
        maximum_delta = float(
            np.max(np.abs(recreated - expected_aligned_points), initial=0.0)
        )
        raise ValueError(
            "Recreated Kanik aligned list does not exactly match "
            "kanik/clean_aligned.txt; maximum component difference is "
            f"{maximum_delta * MICROMETERS_PER_PIXEL} micrometers"
        )


def indices_strictly_within(
    tree: cKDTree,
    points: np.ndarray,
    query: np.ndarray,
    threshold: float,
) -> list[int]:
    # query_ball_point includes the radius boundary, while the repository's
    # Overlap class uses a strict distance < tolerance comparison.
    nearby = tree.query_ball_point(query, r=threshold)
    return sorted(
        index
        for index in nearby
        if float(np.linalg.norm(points[index] - query)) < threshold
    )


def build_candidate_pairs(
    clean_ids: list[str],
    truth_points: np.ndarray,
    c1_points: np.ndarray,
    cr_points: np.ndarray,
    c2_points: np.ndarray,
    eligible_c1: np.ndarray,
    threshold: float,
) -> tuple[list[list[int]], list[list[CandidatePair]]]:
    c1_tree = cKDTree(c1_points)
    cr_tree = cKDTree(cr_points)
    c2_tree = cKDTree(c2_points)
    c1_candidates_by_truth: list[list[int]] = []
    pairs_by_truth: list[list[CandidatePair]] = []

    for truth_index, (clean_id, truth_point) in enumerate(zip(clean_ids, truth_points)):
        c1_candidates = [
            index
            for index in indices_strictly_within(
                c1_tree, c1_points, truth_point, threshold
            )
            if eligible_c1[index]
        ]
        pairs: list[CandidatePair] = []

        for c1_index in c1_candidates:
            c1_point = c1_points[c1_index]
            cr_candidates = indices_strictly_within(
                cr_tree, cr_points, c1_point, HISTORICAL_ALIGNMENT_THRESHOLD
            )
            c2_candidates = indices_strictly_within(
                c2_tree, c2_points, c1_point, threshold
            )
            for c2_index in c2_candidates:
                c2_point = c2_points[c2_index]
                for cr_index in cr_candidates:
                    cr_point = cr_points[cr_index]
                    pairs.append(
                        CandidatePair(
                            clean_id=clean_id,
                            truth_index=truth_index,
                            c1_index=c1_index,
                            cr_index=cr_index,
                            c2_index=c2_index,
                            cr_candidate_count_for_c1=len(cr_candidates),
                            truth_to_c1_distance_px=float(
                                np.linalg.norm(truth_point - c1_point)
                            ),
                            c1_to_cr_distance_px=float(
                                np.linalg.norm(c1_point - cr_point)
                            ),
                            c1_to_c2_distance_px=float(
                                np.linalg.norm(c1_point - c2_point)
                            ),
                            cr_to_c2_distance_px=float(
                                np.linalg.norm(cr_point - c2_point)
                            ),
                            truth_to_cr_distance_px=float(
                                np.linalg.norm(truth_point - cr_point)
                            ),
                            truth_to_c2_distance_px=float(
                                np.linalg.norm(truth_point - c2_point)
                            ),
                        )
                    )

        pairs.sort(key=CandidatePair.sort_key)
        c1_candidates_by_truth.append(c1_candidates)
        pairs_by_truth.append(pairs)

    return c1_candidates_by_truth, pairs_by_truth


def format_float(value: float) -> str:
    return f"{value:.6f}"


def format_distance_um(value_pixels: float) -> str:
    return format_float(value_pixels * MICROMETERS_PER_PIXEL)


def c1_c2_angle_from_foil_normal_deg(distance_pixels: float) -> float:
    lateral_distance_um = distance_pixels * MICROMETERS_PER_PIXEL
    return math.degrees(
        math.atan2(lateral_distance_um, C1_C2_SURFACE_SEPARATION_UM)
    )


def write_candidate_pairs(
    path: Path,
    pairs_by_truth: list[list[CandidatePair]],
    truth_points: np.ndarray,
    c1_points: np.ndarray,
    cr_points: np.ndarray,
    c2_local: np.ndarray,
    c2_points: np.ndarray,
) -> None:
    fieldnames = [
        "clean_id",
        "candidate_rank",
        "final_match_status",
        "truth_global_x",
        "truth_global_y",
        "kanik_c1_index",
        "kanik_c1_global_x",
        "kanik_c1_global_y",
        "kanik_cr_index",
        "kanik_cr_candidate_count_for_c1",
        "kanik_cr_global_x",
        "kanik_cr_global_y",
        "kanik_c2_index",
        "kanik_c2_image",
        "kanik_c2_local_x",
        "kanik_c2_local_y",
        "kanik_c2_global_x",
        "kanik_c2_global_y",
        "truth_to_c1_distance_um",
        "c1_to_cr_distance_um",
        "c1_to_c2_distance_um",
        "c1_c2_angle_from_foil_normal_deg",
        "cr_to_c2_distance_um",
        "truth_to_cr_distance_um",
        "truth_to_c2_distance_um",
    ]
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for pairs in pairs_by_truth:
            for rank, pair in enumerate(pairs, start=1):
                truth_point = truth_points[pair.truth_index]
                c1_point = c1_points[pair.c1_index]
                cr_point = cr_points[pair.cr_index]
                c2_source = c2_local[pair.c2_index]
                c2_point = c2_points[pair.c2_index]
                writer.writerow(
                    {
                        "clean_id": pair.clean_id,
                        "candidate_rank": rank,
                        "final_match_status": (
                            "chosen_for_final_output"
                            if rank == 1
                            else "alternative_not_chosen"
                        ),
                        "truth_global_x": format_float(truth_point[0]),
                        "truth_global_y": format_float(truth_point[1]),
                        "kanik_c1_index": pair.c1_index,
                        "kanik_c1_global_x": format_float(c1_point[0]),
                        "kanik_c1_global_y": format_float(c1_point[1]),
                        "kanik_cr_index": pair.cr_index,
                        "kanik_cr_candidate_count_for_c1": (
                            pair.cr_candidate_count_for_c1
                        ),
                        "kanik_cr_global_x": format_float(cr_point[0]),
                        "kanik_cr_global_y": format_float(cr_point[1]),
                        "kanik_c2_index": pair.c2_index,
                        "kanik_c2_image": int(c2_source[0]),
                        "kanik_c2_local_x": format_float(c2_source[1]),
                        "kanik_c2_local_y": format_float(c2_source[2]),
                        "kanik_c2_global_x": format_float(c2_point[0]),
                        "kanik_c2_global_y": format_float(c2_point[1]),
                        "truth_to_c1_distance_um": format_distance_um(
                            pair.truth_to_c1_distance_px
                        ),
                        "c1_to_cr_distance_um": format_distance_um(
                            pair.c1_to_cr_distance_px
                        ),
                        "c1_to_c2_distance_um": format_distance_um(
                            pair.c1_to_c2_distance_px
                        ),
                        "c1_c2_angle_from_foil_normal_deg": format_float(
                            c1_c2_angle_from_foil_normal_deg(
                                pair.c1_to_c2_distance_px
                            )
                        ),
                        "cr_to_c2_distance_um": format_distance_um(
                            pair.cr_to_c2_distance_px
                        ),
                        "truth_to_cr_distance_um": format_distance_um(
                            pair.truth_to_cr_distance_px
                        ),
                        "truth_to_c2_distance_um": format_distance_um(
                            pair.truth_to_c2_distance_px
                        ),
                    }
                )


def selected_status(
    c1_candidate_count: int,
    feasible_candidate_count: int,
    c1_usage_count: int,
    c2_usage_count: int,
) -> str:
    flags: list[str] = []
    if c1_candidate_count > 1 or feasible_candidate_count > 1:
        flags.append("ambiguous_candidates")
    if c1_usage_count > 1:
        flags.append("shared_c1_selection")
    if c2_usage_count > 1:
        flags.append("shared_c2_selection")
    return ";".join(flags) if flags else "unique"


def write_selected_results(
    path: Path,
    clean_ids: list[str],
    truth_points: np.ndarray,
    c1_points: np.ndarray,
    c2_local: np.ndarray,
    c2_points: np.ndarray,
    c1_candidates_by_truth: list[list[int]],
    pairs_by_truth: list[list[CandidatePair]],
) -> list[dict[str, Any]]:
    selected = [pairs[0] if pairs else None for pairs in pairs_by_truth]
    c1_usage = Counter(pair.c1_index for pair in selected if pair is not None)
    c2_usage = Counter(pair.c2_index for pair in selected if pair is not None)

    fieldnames = [
        "clean_id",
        "status",
        "truth_global_x",
        "truth_global_y",
        "kanik_c1_index",
        "kanik_c1_global_x",
        "kanik_c1_global_y",
        "kanik_c2_index",
        "kanik_c2_image",
        "kanik_c2_local_x",
        "kanik_c2_local_y",
        "kanik_c2_global_x",
        "kanik_c2_global_y",
        "truth_to_c1_distance_um",
        "c1_to_c2_distance_um",
        "c1_c2_angle_from_foil_normal_deg",
        "truth_to_c2_distance_um",
        "c1_candidate_count",
        "feasible_candidate_triplet_count",
        "selected_c1_usage_count",
        "selected_c2_usage_count",
    ]
    output_rows: list[dict[str, Any]] = []

    for truth_index, clean_id in enumerate(clean_ids):
        truth_point = truth_points[truth_index]
        pair = selected[truth_index]
        c1_candidate_count = len(c1_candidates_by_truth[truth_index])
        feasible_candidate_count = len(pairs_by_truth[truth_index])

        if pair is None:
            output_rows.append(
                {
                    "clean_id": clean_id,
                    "status": "no_feasible_pair",
                    "truth_global_x": format_float(truth_point[0]),
                    "truth_global_y": format_float(truth_point[1]),
                    "c1_candidate_count": c1_candidate_count,
                    "feasible_candidate_triplet_count": feasible_candidate_count,
                    "selected_c1_usage_count": 0,
                    "selected_c2_usage_count": 0,
                }
            )
            continue

        c1_point = c1_points[pair.c1_index]
        c2_source = c2_local[pair.c2_index]
        c2_point = c2_points[pair.c2_index]
        c1_usage_count = c1_usage[pair.c1_index]
        c2_usage_count = c2_usage[pair.c2_index]
        output_rows.append(
            {
                "clean_id": clean_id,
                "status": selected_status(
                    c1_candidate_count,
                    feasible_candidate_count,
                    c1_usage_count,
                    c2_usage_count,
                ),
                "truth_global_x": format_float(truth_point[0]),
                "truth_global_y": format_float(truth_point[1]),
                "kanik_c1_index": pair.c1_index,
                "kanik_c1_global_x": format_float(c1_point[0]),
                "kanik_c1_global_y": format_float(c1_point[1]),
                "kanik_c2_index": pair.c2_index,
                "kanik_c2_image": int(c2_source[0]),
                "kanik_c2_local_x": format_float(c2_source[1]),
                "kanik_c2_local_y": format_float(c2_source[2]),
                "kanik_c2_global_x": format_float(c2_point[0]),
                "kanik_c2_global_y": format_float(c2_point[1]),
                "truth_to_c1_distance_um": format_distance_um(
                    pair.truth_to_c1_distance_px
                ),
                "c1_to_c2_distance_um": format_distance_um(
                    pair.c1_to_c2_distance_px
                ),
                "c1_c2_angle_from_foil_normal_deg": format_float(
                    c1_c2_angle_from_foil_normal_deg(
                        pair.c1_to_c2_distance_px
                    )
                ),
                "truth_to_c2_distance_um": format_distance_um(
                    pair.truth_to_c2_distance_px
                ),
                "c1_candidate_count": c1_candidate_count,
                "feasible_candidate_triplet_count": feasible_candidate_count,
                "selected_c1_usage_count": c1_usage_count,
                "selected_c2_usage_count": c2_usage_count,
            }
        )

    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    return output_rows


def write_summary(
    path: Path,
    threshold: float,
    clean_ids: list[str],
    c1_points: np.ndarray,
    cr_points: np.ndarray,
    c2_points: np.ndarray,
    transformation_validation_deltas: dict[str, float],
    alignment_membership: AlignmentMembership,
    c1_candidates_by_truth: list[list[int]],
    pairs_by_truth: list[list[CandidatePair]],
    selected_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    selected_distances = np.asarray(
        [
            float(row["c1_to_c2_distance_um"])
            for row in selected_rows
            if row["status"] != "no_feasible_pair"
        ],
        dtype=float,
    )
    selected_angles = np.asarray(
        [
            float(row["c1_c2_angle_from_foil_normal_deg"])
            for row in selected_rows
            if row["status"] != "no_feasible_pair"
        ],
        dtype=float,
    )
    selected_c1_indices = [
        int(row["kanik_c1_index"])
        for row in selected_rows
        if row["status"] != "no_feasible_pair"
    ]
    selected_c2_indices = [
        int(row["kanik_c2_index"])
        for row in selected_rows
        if row["status"] != "no_feasible_pair"
    ]
    c1_usage = Counter(selected_c1_indices)
    c2_usage = Counter(selected_c2_indices)

    summary: dict[str, Any] = {
        "coordinate_conversion_micrometers_per_pixel": MICROMETERS_PER_PIXEL,
        "c1_c2_surface_separation_micrometers": C1_C2_SURFACE_SEPARATION_UM,
        "angle_definition": (
            "degrees from the foil normal: atan(C1-C2 lateral distance / "
            "C1-C2 surface separation)"
        ),
        "threshold_micrometers": round(
            threshold * MICROMETERS_PER_PIXEL, 12
        ),
        "angle_at_historical_12_pixel_boundary_degrees": (
            c1_c2_angle_from_foil_normal_deg(HISTORICAL_ALIGNMENT_THRESHOLD)
        ),
        "accepted_angle_comparison": (
            "strictly less than the angle at the 12-pixel boundary"
        ),
        "threshold_comparison": "strict distance < threshold",
        "matching_rule": [
            "retain Kanik C1 points with inverse C1 image >= 251",
            "retain Kanik C1 points with at least one Kanik Cr point within the historical 67.2-micrometer threshold",
            "retain Kanik C1 points with at least one transformed Kanik C2 point within the historical 67.2-micrometer threshold",
            "enumerate every retained Kanik C1 point within threshold of the Lewis C1 truth",
            "enumerate every transformed Kanik C2 point within threshold of each C1 candidate",
            "enumerate every Kanik Cr point within the historical 67.2-micrometer threshold of each C1 candidate",
            "select minimum truth-to-C1 distance, then minimum C1-to-C2 distance, then minimum C1-to-Cr distance",
            "retain every alternative in all_candidate_pairs.csv",
        ],
        "sources": {
            "clean_truth_with_ids": str(DEFAULT_TRUTH_PATH.relative_to(REPO_ROOT)),
            "kanik_c1_local": str(DEFAULT_C1_LOCAL_PATH.relative_to(REPO_ROOT)),
            "kanik_c1_global": str(DEFAULT_C1_PATH.relative_to(REPO_ROOT)),
            "kanik_cr_local": str(DEFAULT_CR_LOCAL_PATH.relative_to(REPO_ROOT)),
            "kanik_cr_global": str(DEFAULT_CR_PATH.relative_to(REPO_ROOT)),
            "kanik_c2_local": str(DEFAULT_C2_LOCAL_PATH.relative_to(REPO_ROOT)),
            "kanik_c2_existing_global_validation": str(
                DEFAULT_C2_GLOBAL_PATH.relative_to(REPO_ROOT)
            ),
            "kanik_historical_aligned_validation": str(
                DEFAULT_ALIGNED_C1_PATH.relative_to(REPO_ROOT)
            ),
        },
        "clean_truth_count": len(clean_ids),
        "kanik_c1_count": len(c1_points),
        "kanik_cr_count": len(cr_points),
        "kanik_c2_count": len(c2_points),
        "transformation_validation_max_component_delta_micrometers": {
            surface: delta * MICROMETERS_PER_PIXEL
            for surface, delta in transformation_validation_deltas.items()
        },
        "kanik_alignment_reconstruction": {
            "historical_threshold_micrometers": (
                round(
                    HISTORICAL_ALIGNMENT_THRESHOLD * MICROMETERS_PER_PIXEL,
                    12,
                )
            ),
            "minimum_inverse_c1_image": MINIMUM_C1_IMAGE,
            "c1_points_passing_image_cut": int(
                np.count_nonzero(alignment_membership.passes_image_cut)
            ),
            "c1_points_with_cr_match": int(
                np.count_nonzero(alignment_membership.has_cr_match)
            ),
            "c1_points_with_c2_match": int(
                np.count_nonzero(alignment_membership.has_c2_match)
            ),
            "recreated_aligned_c1_count": int(
                np.count_nonzero(alignment_membership.eligible)
            ),
            "exactly_matches_kanik_clean_aligned": True,
        },
        "truths_with_no_c1_candidate": sum(
            not candidates for candidates in c1_candidates_by_truth
        ),
        "truths_with_multiple_c1_candidates": sum(
            len(candidates) > 1 for candidates in c1_candidates_by_truth
        ),
        "truths_with_no_feasible_candidate_triplet": sum(
            not pairs for pairs in pairs_by_truth
        ),
        "truths_with_multiple_feasible_candidate_triplets": sum(
            len(pairs) > 1 for pairs in pairs_by_truth
        ),
        "all_feasible_c1_cr_c2_triplet_count": sum(
            len(pairs) for pairs in pairs_by_truth
        ),
        "all_feasible_c1_c2_pair_count": sum(
            len({(pair.c1_index, pair.c2_index) for pair in pairs})
            for pairs in pairs_by_truth
        ),
        "selected_pair_count": int(selected_distances.size),
        "selected_c1_points_used_by_multiple_truth_ids": sum(
            count > 1 for count in c1_usage.values()
        ),
        "selected_c2_points_used_by_multiple_truth_ids": sum(
            count > 1 for count in c2_usage.values()
        ),
        "selected_c1_to_c2_distance_micrometers": {
            "minimum": float(np.min(selected_distances)),
            "median": float(np.median(selected_distances)),
            "mean": float(np.mean(selected_distances)),
            "maximum": float(np.max(selected_distances)),
        },
        "selected_c1_c2_angle_from_foil_normal_degrees": {
            "minimum": float(np.min(selected_angles)),
            "median": float(np.median(selected_angles)),
            "mean": float(np.mean(selected_angles)),
            "maximum": float(np.max(selected_angles)),
        },
    }

    with path.open("w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, indent=2)
        output_file.write("\n")
    return summary


def print_selected_distances(rows: Iterable[dict[str, Any]]) -> None:
    writer = csv.writer(sys.stdout)
    writer.writerow(
        [
            "clean_id",
            "c1_to_c2_distance_um",
            "c1_c2_angle_from_foil_normal_deg",
            "status",
        ]
    )
    for row in rows:
        writer.writerow(
            [
                row["clean_id"],
                row.get("c1_to_c2_distance_um", ""),
                row.get("c1_c2_angle_from_foil_normal_deg", ""),
                row["status"],
            ]
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--threshold-um",
        type=float,
        default=HISTORICAL_ALIGNMENT_THRESHOLD * MICROMETERS_PER_PIXEL,
        help="Strict matching distance in micrometers (default: 67.2)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Write data files without printing one distance per clean ID",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not np.isfinite(args.threshold_um) or args.threshold_um <= 0:
        raise ValueError("--threshold-um must be a finite positive number")
    threshold_pixels = args.threshold_um / MICROMETERS_PER_PIXEL

    clean_ids, truth_points = load_truth(DEFAULT_TRUTH_PATH)
    c1_local = load_local_centroids(DEFAULT_C1_LOCAL_PATH)
    cr_local = load_local_centroids(DEFAULT_CR_LOCAL_PATH)
    c2_local = load_local_centroids(DEFAULT_C2_LOCAL_PATH)
    coordinate_system = create_coordinate_system()

    # Kanik's C1 and already-flipped Cr centroids both use the C1 coordinate
    # system. C2 uses its own datum-aligned scan before comparison in C1 global
    # coordinates.
    c1_points = transform_local_to_global(c1_local, coordinate_system, "c1")
    cr_points = transform_local_to_global(cr_local, coordinate_system, "c1")
    c2_points = transform_local_to_global(c2_local, coordinate_system, "c2")

    transformation_validation_deltas = {
        "c1": validate_transformed_coordinates(
            "C1", c1_points, load_global_coordinates(DEFAULT_C1_PATH)
        ),
        "cr": validate_transformed_coordinates(
            "Cr", cr_points, load_global_coordinates(DEFAULT_CR_PATH)
        ),
        "c2": validate_transformed_coordinates(
            "C2", c2_points, load_global_coordinates(DEFAULT_C2_GLOBAL_PATH)
        ),
    }

    alignment_membership = reconstruct_alignment_membership(
        c1_points,
        cr_points,
        c2_points,
        coordinate_system,
        HISTORICAL_ALIGNMENT_THRESHOLD,
    )
    validate_historical_alignment(
        c1_points,
        alignment_membership,
        load_global_coordinates(DEFAULT_ALIGNED_C1_PATH),
    )

    c1_candidates_by_truth, pairs_by_truth = build_candidate_pairs(
        clean_ids,
        truth_points,
        c1_points,
        cr_points,
        c2_points,
        alignment_membership.eligible,
        threshold_pixels,
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates_path = output_dir / "all_candidate_pairs.csv"
    selected_path = output_dir / "clean_c1_c2_distances.csv"
    summary_path = output_dir / "summary.json"

    write_candidate_pairs(
        candidates_path,
        pairs_by_truth,
        truth_points,
        c1_points,
        cr_points,
        c2_local,
        c2_points,
    )
    selected_rows = write_selected_results(
        selected_path,
        clean_ids,
        truth_points,
        c1_points,
        c2_local,
        c2_points,
        c1_candidates_by_truth,
        pairs_by_truth,
    )
    summary = write_summary(
        summary_path,
        threshold_pixels,
        clean_ids,
        c1_points,
        cr_points,
        c2_points,
        transformation_validation_deltas,
        alignment_membership,
        c1_candidates_by_truth,
        pairs_by_truth,
        selected_rows,
    )

    if not args.quiet:
        print_selected_distances(selected_rows)

    print(
        f"Wrote {summary['selected_pair_count']} selected distances and "
        f"{summary['all_feasible_c1_cr_c2_triplet_count']} candidate triplets to "
        f"{output_dir}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
