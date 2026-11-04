"""Assign stable IDs to clean pits and aligned FCN/DCC false positives."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from modules.clean_pit_registry import validate_clean_id_records
from modules.dataset_tools import coordinates, mhio


def generate_pit_ids(tolerance: float = 12) -> dict[str, int]:
    """Generate run-specific false-positive IDs without recalculating clean IDs."""
    common_list_with_ids = validate_clean_id_records(
        mhio.load_json("predicted_list/ids/Aligned_clean_with_ids.txt"),
        context="active clean truth",
    )
    false_pos_fcn = mhio.load_json(
        "predicted_list/FCN/aligned/final_false_pos.txt"
    )
    false_pos_dcc = mhio.load_json(
        "predicted_list/DCC/aligned/final_false_pos.txt"
    )

    overlap = coordinates.Overlap(false_pos_fcn, false_pos_dcc, tolerance)
    overlap_pairs = np.where(overlap.dist <= tolerance)
    common_ids_fcn = overlap_pairs[0]
    common_ids_dcc = overlap_pairs[1]

    graph = defaultdict(set)
    dcc_offset = len(false_pos_fcn)
    for fcn_idx, dcc_idx in zip(common_ids_fcn, common_ids_dcc):
        graph[int(fcn_idx)].add(dcc_offset + int(dcc_idx))
        graph[dcc_offset + int(dcc_idx)].add(int(fcn_idx))

    visited = set()
    clusters = []

    def dfs(node, cluster):
        visited.add(node)
        cluster.append(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs(neighbor, cluster)

    for node in graph:
        if node not in visited:
            cluster = []
            dfs(node, cluster)
            clusters.append(cluster)

    cluster_ids = {}
    for index, cluster in enumerate(clusters):
        cluster_id = f"{index:04d}"
        for node in cluster:
            cluster_ids[node] = cluster_id

    common_fpf = [
        {"id": f"FAD{cluster_ids[index]}", "coordinate": coordinate}
        for index, coordinate in enumerate(false_pos_fcn)
        if index in cluster_ids
    ]
    common_fpd = [
        {
            "id": f"DAF{cluster_ids[dcc_offset + index]}",
            "coordinate": coordinate,
        }
        for index, coordinate in enumerate(false_pos_dcc)
        if dcc_offset + index in cluster_ids
    ]

    # Preserve the legacy set iteration used to assign the numeric ID suffixes.
    unique_fcn_indices = set(range(len(false_pos_fcn))) - set(common_ids_fcn)
    unique_dcc_indices = set(range(len(false_pos_dcc))) - set(common_ids_dcc)
    unique_fcn = [
        {"id": f"FMD{index:04d}", "coordinate": false_pos_fcn[source_index]}
        for index, source_index in enumerate(unique_fcn_indices)
    ]
    unique_dcc = [
        {"id": f"DMF{index:04d}", "coordinate": false_pos_dcc[source_index]}
        for index, source_index in enumerate(unique_dcc_indices)
    ]

    mhio.save_json(common_fpf, "predicted_list/ids/FP_FCN_common_DCC_with_ids.txt")
    mhio.save_json(common_fpd, "predicted_list/ids/FP_DCC_common_FCN_with_ids.txt")
    mhio.save_json(unique_fcn, "predicted_list/ids/FP_unique_FCN_with_ids.txt")
    mhio.save_json(unique_dcc, "predicted_list/ids/FP_unique_DCC_with_ids.txt")
    summary = {
        "clean": len(common_list_with_ids),
        "fcn_common": len(common_fpf),
        "dcc_common": len(common_fpd),
        "fcn_unique": len(unique_fcn),
        "dcc_unique": len(unique_dcc),
    }
    print(summary)
    return summary


if __name__ == "__main__":
    generate_pit_ids()
