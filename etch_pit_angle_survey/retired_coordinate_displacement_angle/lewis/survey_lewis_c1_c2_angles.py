#!/usr/bin/env python3
"""Survey Lewis C1/Cr/C2 clean etch-pit distances and angles.

The script reconstructs Lewis's historical aligned clean list from the raw
local metadata, validates the complete downstream truth-list construction, and
then associates every clean truth ID with all feasible Lewis C1/Cr/C2 triplets.
Reported distances are in micrometers and angles are measured from the foil
normal.
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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.dataset_tools import coordinates  # noqa: E402


LOCAL_METADATA_PATH = (
    REPO_ROOT / "modules/dataset_tools/metadata/all_matched_local_locations.txt"
)
GLOBAL_METADATA_PATH = (
    REPO_ROOT / "modules/dataset_tools/metadata/all_aligned_Global_locations.txt"
)
KANIK_ALIGNED_PATH = REPO_ROOT / "kanik/clean_aligned.txt"
COMMON_TRUTH_PATH = (
    REPO_ROOT / "predicted_list/universal_true_etch_pit_list/Aligned_common.txt"
)
CLEAN_TRUTH_PATH = REPO_ROOT / "Aligned_clean.txt"
CLEAN_TRUTH_IDS_PATH = (
    REPO_ROOT / "predicted_list/ids/Aligned_clean_with_ids.txt"
)
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "data"

HISTORICAL_THRESHOLD_PIXELS = 12.0
MINIMUM_C1_IMAGE = 251
EDGE_MARGIN_PIXELS = 4
MICROMETERS_PER_PIXEL = 5.6
C1_C2_SURFACE_SEPARATION_UM = 200.0


@dataclass(frozen=True)
class AlignmentMembership:
    has_cr_match: np.ndarray
    has_c2_match_after_cr: np.ndarray
    aligned: np.ndarray


@dataclass(frozen=True)
class CandidateTriplet:
    clean_id: str
    truth_index: int
    c1_index: int
    cr_index: int
    c2_index: int
    cr_candidate_count_for_c1: int
    c1_historical_coordinate_exact_match: bool
    truth_to_c1_distance_px: float
    c1_to_cr_distance_px: float
    c1_to_c2_distance_px: float
    cr_to_c2_distance_px: float
    truth_to_cr_distance_px: float
    truth_to_c2_distance_px: float

    def sort_key(
        self,
    ) -> tuple[int, float, float, float, float, float, float, int, int, int]:
        return (
            0 if self.c1_historical_coordinate_exact_match else 1,
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


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as input_file:
        return json.load(input_file)


def load_coordinate_system() -> coordinates.Coord_system:
    coordinates.img_metadata_location = str(
        REPO_ROOT / "modules/dataset_tools/metadata/Image_location_metadata.txt"
    )
    with contextlib.redirect_stdout(io.StringIO()):
        coordinate_system = coordinates.Coord_system()
    coordinate_system["cr"].flipped = True
    coordinate_system["cr"].reverse = True
    return coordinate_system


def transform_local_points(
    local_points: np.ndarray,
    coordinate_system: coordinates.Coord_system,
    scan_name: str,
) -> np.ndarray:
    return np.asarray(
        [
            coordinate_system[scan_name].get_gxy(
                (int(image), float(x), float(y))
            )
            for image, x, y in local_points
        ],
        dtype=float,
    )


def truncate_global_points(points: np.ndarray) -> np.ndarray:
    # Historical metadata used Python int(), which truncates toward zero.
    return np.asarray([[int(x), int(y)] for x, y in points], dtype=int)


def validate_exact(label: str, actual: np.ndarray, expected: np.ndarray) -> None:
    if actual.shape != expected.shape:
        raise ValueError(
            f"{label} shape differs: reconstructed {actual.shape}, "
            f"checked-in {expected.shape}"
        )
    if not np.array_equal(actual, expected):
        maximum_delta = float(np.max(np.abs(actual - expected), initial=0.0))
        raise ValueError(
            f"{label} is not reproduced exactly; maximum component difference "
            f"is {maximum_delta * MICROMETERS_PER_PIXEL} micrometers"
        )


def reconstruct_alignment(
    c1_points: np.ndarray,
    cr_points: np.ndarray,
    c2_points: np.ndarray,
    threshold_pixels: float,
) -> AlignmentMembership:
    has_cr_match = coordinates.Overlap(
        c1_points, cr_points, tolerance=threshold_pixels
    ).match_a > 0
    c1_after_cr = c1_points[has_cr_match]
    has_c2_after_cr = coordinates.Overlap(
        c1_after_cr, c2_points, tolerance=threshold_pixels
    ).match_a > 0

    aligned = np.zeros(len(c1_points), dtype=bool)
    c1_after_cr_indices = np.flatnonzero(has_cr_match)
    aligned[c1_after_cr_indices[has_c2_after_cr]] = True
    return AlignmentMembership(
        has_cr_match=has_cr_match,
        has_c2_match_after_cr=has_c2_after_cr,
        aligned=aligned,
    )


def validate_alignment_reconstruction(
    local_metadata: dict[str, Any],
    global_metadata: dict[str, Any],
    c1_local: np.ndarray,
    c1_points: np.ndarray,
    cr_points: np.ndarray,
    c2_points: np.ndarray,
    membership: AlignmentMembership,
) -> None:
    validate_exact(
        "Lewis C1 global coordinates",
        truncate_global_points(c1_points),
        np.asarray(global_metadata["c1"], dtype=int),
    )
    validate_exact(
        "Lewis Cr global coordinates",
        truncate_global_points(cr_points),
        np.asarray(global_metadata["cr"], dtype=int),
    )
    validate_exact(
        "Lewis C2 global coordinates",
        truncate_global_points(c2_points),
        np.asarray(global_metadata["c2"], dtype=int),
    )
    validate_exact(
        "Lewis local C1/Cr aligned membership",
        c1_local[membership.has_cr_match],
        np.asarray(local_metadata["foil_1_aligned"], dtype=int),
    )
    validate_exact(
        "Lewis local C1/Cr/C2 aligned membership",
        c1_local[membership.aligned],
        np.asarray(local_metadata["foil_1_2_aligned"], dtype=int),
    )
    validate_exact(
        "Lewis global C1/Cr aligned membership",
        truncate_global_points(c1_points[membership.has_cr_match]),
        np.asarray(global_metadata["foil_1_aligned"], dtype=int),
    )
    validate_exact(
        "Lewis global C1/Cr/C2 aligned membership",
        truncate_global_points(c1_points[membership.aligned]),
        np.asarray(global_metadata["foil_1_2_aligned"], dtype=int),
    )


def validate_truth_construction(
    coordinate_system: coordinates.Coord_system,
    c1_points: np.ndarray,
    membership: AlignmentMembership,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    c1_historical = truncate_global_points(c1_points)
    inverse_images = np.asarray(
        [coordinate_system["c1"].get_ixy(point)[0] for point in c1_historical],
        dtype=int,
    )
    evaluation_mask = membership.aligned & (inverse_images >= MINIMUM_C1_IMAGE)
    lewis_evaluation = c1_historical[evaluation_mask]
    if len(lewis_evaluation) != 1272:
        raise ValueError(
            f"Expected 1272 Lewis evaluation pits, found {len(lewis_evaluation)}"
        )

    kanik_aligned = np.asarray(load_json(KANIK_ALIGNED_PATH), dtype=float)
    common_mask = coordinates.Overlap(
        lewis_evaluation, kanik_aligned, tolerance=HISTORICAL_THRESHOLD_PIXELS
    ).match_a > 0
    common = lewis_evaluation[common_mask]
    validate_exact(
        "Lewis/Kanik common truth list",
        common,
        np.asarray(load_json(COMMON_TRUTH_PATH), dtype=int),
    )

    clean: list[list[int]] = []
    maximum_x = 640 - EDGE_MARGIN_PIXELS
    maximum_y = 480 - EDGE_MARGIN_PIXELS
    for point in common:
        c1_local = coordinate_system["c1"].get_ixy(point)
        exposed_local = coordinate_system.convert(c1_local, "cd")
        image, x, y = exposed_local
        if (
            x < EDGE_MARGIN_PIXELS
            or y < EDGE_MARGIN_PIXELS
            or x > maximum_x
            or y > maximum_y
            or image < MINIMUM_C1_IMAGE
        ):
            continue
        clean.append([int(point[0]), int(point[1])])
    clean_points = np.asarray(clean, dtype=int)
    validate_exact(
        "Final clean truth list",
        clean_points,
        np.asarray(load_json(CLEAN_TRUTH_PATH), dtype=int),
    )

    truth_records = load_json(CLEAN_TRUTH_IDS_PATH)
    clean_ids = [str(record["id"]) for record in truth_records]
    truth_points = np.asarray(
        [record["coordinate"] for record in truth_records], dtype=float
    )
    validate_exact("ID-labelled clean truth list", truth_points, clean_points)
    if len(clean_ids) != len(set(clean_ids)):
        raise ValueError("Clean truth IDs are not unique")
    return clean_ids, truth_points, evaluation_mask


def indices_strictly_within(
    tree: cKDTree,
    points: np.ndarray,
    query: np.ndarray,
    threshold_pixels: float,
) -> list[int]:
    return sorted(
        index
        for index in tree.query_ball_point(query, r=threshold_pixels)
        if float(np.linalg.norm(points[index] - query)) < threshold_pixels
    )


def build_candidate_triplets(
    clean_ids: list[str],
    truth_points: np.ndarray,
    c1_points: np.ndarray,
    cr_points: np.ndarray,
    c2_points: np.ndarray,
    eligible_c1: np.ndarray,
    survey_threshold_pixels: float,
) -> tuple[list[list[int]], list[list[CandidateTriplet]]]:
    c1_tree = cKDTree(c1_points)
    cr_tree = cKDTree(cr_points)
    c2_tree = cKDTree(c2_points)
    historical_c1_points = truncate_global_points(c1_points)
    c1_candidates_by_truth: list[list[int]] = []
    triplets_by_truth: list[list[CandidateTriplet]] = []

    for truth_index, (clean_id, truth_point) in enumerate(
        zip(clean_ids, truth_points)
    ):
        c1_candidates = [
            index
            for index in indices_strictly_within(
                c1_tree, c1_points, truth_point, survey_threshold_pixels
            )
            if eligible_c1[index]
        ]
        triplets: list[CandidateTriplet] = []
        for c1_index in c1_candidates:
            c1_point = c1_points[c1_index]
            cr_candidates = indices_strictly_within(
                cr_tree,
                cr_points,
                c1_point,
                HISTORICAL_THRESHOLD_PIXELS,
            )
            c2_candidates = indices_strictly_within(
                c2_tree,
                c2_points,
                c1_point,
                survey_threshold_pixels,
            )
            for c2_index in c2_candidates:
                c2_point = c2_points[c2_index]
                for cr_index in cr_candidates:
                    cr_point = cr_points[cr_index]
                    triplets.append(
                        CandidateTriplet(
                            clean_id=clean_id,
                            truth_index=truth_index,
                            c1_index=c1_index,
                            cr_index=cr_index,
                            c2_index=c2_index,
                            cr_candidate_count_for_c1=len(cr_candidates),
                            c1_historical_coordinate_exact_match=bool(
                                np.array_equal(
                                    historical_c1_points[c1_index], truth_point
                                )
                            ),
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
        triplets.sort(key=CandidateTriplet.sort_key)
        c1_candidates_by_truth.append(c1_candidates)
        triplets_by_truth.append(triplets)
    return c1_candidates_by_truth, triplets_by_truth


def format_float(value: float) -> str:
    return f"{value:.6f}"


def format_distance_um(value_pixels: float) -> str:
    return format_float(value_pixels * MICROMETERS_PER_PIXEL)


def angle_from_foil_normal_deg(distance_pixels: float) -> float:
    return math.degrees(
        math.atan2(
            distance_pixels * MICROMETERS_PER_PIXEL,
            C1_C2_SURFACE_SEPARATION_UM,
        )
    )


TRIPLET_FIELDS = [
    "truth_global_x",
    "truth_global_y",
    "lewis_c1_index",
    "lewis_c1_image",
    "lewis_c1_local_x",
    "lewis_c1_local_y",
    "lewis_c1_global_x",
    "lewis_c1_global_y",
    "lewis_c1_historical_coordinate_exact_match",
    "lewis_cr_index",
    "lewis_cr_candidate_count_for_c1",
    "lewis_cr_image",
    "lewis_cr_local_x",
    "lewis_cr_local_y",
    "lewis_cr_global_x",
    "lewis_cr_global_y",
    "lewis_c2_index",
    "lewis_c2_image",
    "lewis_c2_local_x",
    "lewis_c2_local_y",
    "lewis_c2_global_x",
    "lewis_c2_global_y",
    "truth_to_c1_distance_um",
    "c1_to_cr_distance_um",
    "c1_to_c2_distance_um",
    "c1_c2_angle_from_foil_normal_deg",
    "cr_to_c2_distance_um",
    "truth_to_cr_distance_um",
    "truth_to_c2_distance_um",
]


def triplet_values(
    triplet: CandidateTriplet,
    truth_points: np.ndarray,
    c1_local: np.ndarray,
    c1_points: np.ndarray,
    cr_local: np.ndarray,
    cr_points: np.ndarray,
    c2_local: np.ndarray,
    c2_points: np.ndarray,
) -> dict[str, Any]:
    truth = truth_points[triplet.truth_index]
    c1_source = c1_local[triplet.c1_index]
    c1 = c1_points[triplet.c1_index]
    cr_source = cr_local[triplet.cr_index]
    cr = cr_points[triplet.cr_index]
    c2_source = c2_local[triplet.c2_index]
    c2 = c2_points[triplet.c2_index]
    return {
        "truth_global_x": format_float(truth[0]),
        "truth_global_y": format_float(truth[1]),
        "lewis_c1_index": triplet.c1_index,
        "lewis_c1_image": int(c1_source[0]),
        "lewis_c1_local_x": format_float(c1_source[1]),
        "lewis_c1_local_y": format_float(c1_source[2]),
        "lewis_c1_global_x": format_float(c1[0]),
        "lewis_c1_global_y": format_float(c1[1]),
        "lewis_c1_historical_coordinate_exact_match": (
            triplet.c1_historical_coordinate_exact_match
        ),
        "lewis_cr_index": triplet.cr_index,
        "lewis_cr_candidate_count_for_c1": triplet.cr_candidate_count_for_c1,
        "lewis_cr_image": int(cr_source[0]),
        "lewis_cr_local_x": format_float(cr_source[1]),
        "lewis_cr_local_y": format_float(cr_source[2]),
        "lewis_cr_global_x": format_float(cr[0]),
        "lewis_cr_global_y": format_float(cr[1]),
        "lewis_c2_index": triplet.c2_index,
        "lewis_c2_image": int(c2_source[0]),
        "lewis_c2_local_x": format_float(c2_source[1]),
        "lewis_c2_local_y": format_float(c2_source[2]),
        "lewis_c2_global_x": format_float(c2[0]),
        "lewis_c2_global_y": format_float(c2[1]),
        "truth_to_c1_distance_um": format_distance_um(
            triplet.truth_to_c1_distance_px
        ),
        "c1_to_cr_distance_um": format_distance_um(
            triplet.c1_to_cr_distance_px
        ),
        "c1_to_c2_distance_um": format_distance_um(
            triplet.c1_to_c2_distance_px
        ),
        "c1_c2_angle_from_foil_normal_deg": format_float(
            angle_from_foil_normal_deg(triplet.c1_to_c2_distance_px)
        ),
        "cr_to_c2_distance_um": format_distance_um(
            triplet.cr_to_c2_distance_px
        ),
        "truth_to_cr_distance_um": format_distance_um(
            triplet.truth_to_cr_distance_px
        ),
        "truth_to_c2_distance_um": format_distance_um(
            triplet.truth_to_c2_distance_px
        ),
    }


def write_all_candidates(
    path: Path,
    triplets_by_truth: list[list[CandidateTriplet]],
    truth_points: np.ndarray,
    c1_local: np.ndarray,
    c1_points: np.ndarray,
    cr_local: np.ndarray,
    cr_points: np.ndarray,
    c2_local: np.ndarray,
    c2_points: np.ndarray,
) -> None:
    fieldnames = [
        "clean_id",
        "candidate_triplet_rank",
        "final_match_status",
        *TRIPLET_FIELDS,
    ]
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for triplets in triplets_by_truth:
            for rank, triplet in enumerate(triplets, start=1):
                writer.writerow(
                    {
                        "clean_id": triplet.clean_id,
                        "candidate_triplet_rank": rank,
                        "final_match_status": (
                            "chosen_for_final_output"
                            if rank == 1
                            else "alternative_not_chosen"
                        ),
                        **triplet_values(
                            triplet,
                            truth_points,
                            c1_local,
                            c1_points,
                            cr_local,
                            cr_points,
                            c2_local,
                            c2_points,
                        ),
                    }
                )


def selected_status(
    c1_candidate_count: int,
    triplet_count: int,
    c1_usage_count: int,
    cr_usage_count: int,
    c2_usage_count: int,
) -> str:
    flags: list[str] = []
    if c1_candidate_count > 1 or triplet_count > 1:
        flags.append("ambiguous_candidates")
    if c1_usage_count > 1:
        flags.append("shared_c1_selection")
    if cr_usage_count > 1:
        flags.append("shared_cr_selection")
    if c2_usage_count > 1:
        flags.append("shared_c2_selection")
    return ";".join(flags) if flags else "unique"


def write_selected_results(
    path: Path,
    clean_ids: list[str],
    triplets_by_truth: list[list[CandidateTriplet]],
    c1_candidates_by_truth: list[list[int]],
    truth_points: np.ndarray,
    c1_local: np.ndarray,
    c1_points: np.ndarray,
    cr_local: np.ndarray,
    cr_points: np.ndarray,
    c2_local: np.ndarray,
    c2_points: np.ndarray,
) -> list[dict[str, Any]]:
    selected = [triplets[0] if triplets else None for triplets in triplets_by_truth]
    c1_usage = Counter(t.c1_index for t in selected if t is not None)
    cr_usage = Counter(t.cr_index for t in selected if t is not None)
    c2_usage = Counter(t.c2_index for t in selected if t is not None)
    fieldnames = [
        "clean_id",
        "status",
        *TRIPLET_FIELDS,
        "c1_candidate_count",
        "feasible_candidate_triplet_count",
        "selected_c1_usage_count",
        "selected_cr_usage_count",
        "selected_c2_usage_count",
    ]
    rows: list[dict[str, Any]] = []
    for truth_index, clean_id in enumerate(clean_ids):
        triplet = selected[truth_index]
        if triplet is None:
            rows.append(
                {
                    "clean_id": clean_id,
                    "status": "no_feasible_triplet",
                    "truth_global_x": format_float(truth_points[truth_index][0]),
                    "truth_global_y": format_float(truth_points[truth_index][1]),
                    "c1_candidate_count": len(c1_candidates_by_truth[truth_index]),
                    "feasible_candidate_triplet_count": 0,
                    "selected_c1_usage_count": 0,
                    "selected_cr_usage_count": 0,
                    "selected_c2_usage_count": 0,
                }
            )
            continue
        rows.append(
            {
                "clean_id": clean_id,
                "status": selected_status(
                    len(c1_candidates_by_truth[truth_index]),
                    len(triplets_by_truth[truth_index]),
                    c1_usage[triplet.c1_index],
                    cr_usage[triplet.cr_index],
                    c2_usage[triplet.c2_index],
                ),
                **triplet_values(
                    triplet,
                    truth_points,
                    c1_local,
                    c1_points,
                    cr_local,
                    cr_points,
                    c2_local,
                    c2_points,
                ),
                "c1_candidate_count": len(c1_candidates_by_truth[truth_index]),
                "feasible_candidate_triplet_count": len(
                    triplets_by_truth[truth_index]
                ),
                "selected_c1_usage_count": c1_usage[triplet.c1_index],
                "selected_cr_usage_count": cr_usage[triplet.cr_index],
                "selected_c2_usage_count": c2_usage[triplet.c2_index],
            }
        )
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def write_summary(
    path: Path,
    threshold_pixels: float,
    membership: AlignmentMembership,
    evaluation_mask: np.ndarray,
    c1_points: np.ndarray,
    cr_points: np.ndarray,
    c2_points: np.ndarray,
    c1_candidates_by_truth: list[list[int]],
    triplets_by_truth: list[list[CandidateTriplet]],
    selected_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    matched_rows = [
        row for row in selected_rows if row["status"] != "no_feasible_triplet"
    ]
    distances = np.asarray(
        [float(row["c1_to_c2_distance_um"]) for row in matched_rows], dtype=float
    )
    angles = np.asarray(
        [
            float(row["c1_c2_angle_from_foil_normal_deg"])
            for row in matched_rows
        ],
        dtype=float,
    )
    summary: dict[str, Any] = {
        "coordinate_conversion_micrometers_per_pixel": MICROMETERS_PER_PIXEL,
        "c1_c2_surface_separation_micrometers": C1_C2_SURFACE_SEPARATION_UM,
        "angle_definition": (
            "degrees from the foil normal: atan(C1-C2 lateral distance / "
            "C1-C2 surface separation)"
        ),
        "survey_threshold_micrometers": round(
            threshold_pixels * MICROMETERS_PER_PIXEL, 12
        ),
        "historical_alignment_threshold_micrometers": round(
            HISTORICAL_THRESHOLD_PIXELS * MICROMETERS_PER_PIXEL, 12
        ),
        "angle_at_historical_12_pixel_boundary_degrees": (
            angle_from_foil_normal_deg(HISTORICAL_THRESHOLD_PIXELS)
        ),
        "threshold_comparison": "strict distance < threshold",
        "sources": {
            "lewis_local_surface_and_alignment_metadata": str(
                LOCAL_METADATA_PATH.relative_to(REPO_ROOT)
            ),
            "lewis_global_surface_and_alignment_metadata": str(
                GLOBAL_METADATA_PATH.relative_to(REPO_ROOT)
            ),
            "clean_truth_with_ids": str(
                CLEAN_TRUTH_IDS_PATH.relative_to(REPO_ROOT)
            ),
        },
        "validation": {
            "raw_surface_transforms_exact": True,
            "historical_c1_cr_membership_exact": True,
            "historical_c1_cr_c2_membership_exact": True,
            "lewis_evaluation_count": int(np.count_nonzero(evaluation_mask)),
            "lewis_kanik_common_truth_count": len(load_json(COMMON_TRUTH_PATH)),
            "final_clean_truth_count": len(load_json(CLEAN_TRUTH_PATH)),
            "id_labelled_truth_exact": True,
        },
        "lewis_c1_count": len(c1_points),
        "lewis_cr_count": len(cr_points),
        "lewis_c2_count": len(c2_points),
        "historical_c1_cr_aligned_count": int(
            np.count_nonzero(membership.has_cr_match)
        ),
        "historical_c1_cr_c2_aligned_count": int(
            np.count_nonzero(membership.aligned)
        ),
        "truths_with_no_c1_candidate": sum(
            not candidates for candidates in c1_candidates_by_truth
        ),
        "truths_with_multiple_c1_candidates": sum(
            len(candidates) > 1 for candidates in c1_candidates_by_truth
        ),
        "truths_with_no_feasible_candidate_triplet": sum(
            not triplets for triplets in triplets_by_truth
        ),
        "truths_with_multiple_feasible_candidate_triplets": sum(
            len(triplets) > 1 for triplets in triplets_by_truth
        ),
        "all_feasible_c1_cr_c2_triplet_count": sum(
            len(triplets) for triplets in triplets_by_truth
        ),
        "all_feasible_c1_c2_pair_count": sum(
            len({(t.c1_index, t.c2_index) for t in triplets})
            for triplets in triplets_by_truth
        ),
        "selected_triplet_count": len(matched_rows),
        "selected_c1_historical_coordinate_exact_match_count": sum(
            bool(row["lewis_c1_historical_coordinate_exact_match"])
            for row in matched_rows
        ),
        "selected_c1_to_c2_distance_micrometers": {
            "minimum": float(np.min(distances)),
            "median": float(np.median(distances)),
            "mean": float(np.mean(distances)),
            "maximum": float(np.max(distances)),
        },
        "selected_c1_c2_angle_from_foil_normal_degrees": {
            "minimum": float(np.min(angles)),
            "median": float(np.median(angles)),
            "mean": float(np.mean(angles)),
            "maximum": float(np.max(angles)),
        },
    }
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, indent=2)
        output_file.write("\n")
    return summary


def print_selected(rows: Iterable[dict[str, Any]]) -> None:
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
        default=HISTORICAL_THRESHOLD_PIXELS * MICROMETERS_PER_PIXEL,
        help="Strict downstream matching distance in micrometers (default: 67.2)",
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
        help="Write data files without printing one selected result per clean ID",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not np.isfinite(args.threshold_um) or args.threshold_um <= 0:
        raise ValueError("--threshold-um must be a finite positive number")
    survey_threshold_pixels = args.threshold_um / MICROMETERS_PER_PIXEL

    local_metadata = load_json(LOCAL_METADATA_PATH)
    global_metadata = load_json(GLOBAL_METADATA_PATH)
    c1_local = np.asarray(local_metadata["c1"], dtype=int)
    cr_local = np.asarray(local_metadata["cr"], dtype=int)
    c2_local = np.asarray(local_metadata["c2"], dtype=int)

    coordinate_system = load_coordinate_system()
    c1_points = transform_local_points(c1_local, coordinate_system, "c1")
    cr_points = transform_local_points(cr_local, coordinate_system, "cr")
    c2_points = transform_local_points(c2_local, coordinate_system, "c2")
    membership = reconstruct_alignment(
        c1_points,
        cr_points,
        c2_points,
        HISTORICAL_THRESHOLD_PIXELS,
    )
    validate_alignment_reconstruction(
        local_metadata,
        global_metadata,
        c1_local,
        c1_points,
        cr_points,
        c2_points,
        membership,
    )
    clean_ids, truth_points, evaluation_mask = validate_truth_construction(
        coordinate_system, c1_points, membership
    )
    c1_candidates_by_truth, triplets_by_truth = build_candidate_triplets(
        clean_ids,
        truth_points,
        c1_points,
        cr_points,
        c2_points,
        evaluation_mask,
        survey_threshold_pixels,
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    all_candidates_path = output_dir / "all_candidate_pairs.csv"
    selected_path = output_dir / "clean_c1_c2_distances.csv"
    summary_path = output_dir / "summary.json"
    write_all_candidates(
        all_candidates_path,
        triplets_by_truth,
        truth_points,
        c1_local,
        c1_points,
        cr_local,
        cr_points,
        c2_local,
        c2_points,
    )
    selected_rows = write_selected_results(
        selected_path,
        clean_ids,
        triplets_by_truth,
        c1_candidates_by_truth,
        truth_points,
        c1_local,
        c1_points,
        cr_local,
        cr_points,
        c2_local,
        c2_points,
    )
    summary = write_summary(
        summary_path,
        survey_threshold_pixels,
        membership,
        evaluation_mask,
        c1_points,
        cr_points,
        c2_points,
        c1_candidates_by_truth,
        triplets_by_truth,
        selected_rows,
    )
    if not args.quiet:
        print_selected(selected_rows)
    print(
        f"Wrote {summary['selected_triplet_count']} selected Lewis triplets and "
        f"{summary['all_feasible_c1_cr_c2_triplet_count']} candidate triplets "
        f"to {output_dir}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
