"""Generate the selected manuscript example panels for one pipeline run."""

from __future__ import annotations

import numpy as np

from modules.compare.Example_plots import create_example_panel
from modules.dataset_tools import coordinates, mhio


DEFAULT_IDS = [
    "C0001",
    "C0025",
    "C0708",
    "C0743",
    "C0874",
    "C0997",
    "C1222",
    "FMD0008",
    "FMD0012",
    "FAD0003",
]


def _nearby(predictions, primary, tolerance):
    overlap = coordinates.Overlap(
        np.asarray(predictions), np.asarray([primary]), tolerance
    )
    return [
        predictions[index]
        for index, distance in enumerate(overlap.dist[:, 0])
        if distance < tolerance
    ]


def generate_example_panels(
    ids: list[str] | None = None, *, tolerance: float = 15
) -> dict[str, int]:
    """Generate selected panels below ``gallery/examples`` in the active run."""
    ids = DEFAULT_IDS if ids is None else ids
    clean_pits = mhio.load_json("predicted_list/ids/Aligned_clean_with_ids.txt")
    aligned_fcn = mhio.load_json("predicted_list/FCN/aligned/final_pred_list.txt")
    aligned_dcc = mhio.load_json("predicted_list/DCC/aligned/final_pred_list.txt")
    front_fcn = mhio.load_json("predicted_etch_pits/FCN/final_pr.txt")
    front_dcc = mhio.load_json("predicted_etch_pits/DCC/final_pr.txt")
    back_fcn = mhio.load_json("predicted_etch_pits/FCN/final_reverse_pr.txt")
    back_dcc = mhio.load_json("predicted_etch_pits/DCC/final_reverse_pr.txt")
    fad_ids = mhio.load_json("predicted_list/ids/FP_FCN_common_DCC_with_ids.txt")
    daf_ids = mhio.load_json("predicted_list/ids/FP_DCC_common_FCN_with_ids.txt")
    fmd_ids = mhio.load_json("predicted_list/ids/FP_unique_FCN_with_ids.txt")
    dmf_ids = mhio.load_json("predicted_list/ids/FP_unique_DCC_with_ids.txt")

    clean_by_id = {item["id"]: item for item in clean_pits}
    fmd_by_id = {item["id"]: item for item in fmd_ids}
    dmf_by_id = {item["id"]: item for item in dmf_ids}
    clusters = {}
    for item in fad_ids + daf_ids:
        cluster_id = item["id"][-4:]
        cluster = clusters.setdefault(
            cluster_id, {"fcn_coords": [], "dcc_coords": []}
        )
        key = "fcn_coords" if item["id"].startswith("FAD") else "dcc_coords"
        cluster[key].append(item["coordinate"])

    generated = 0
    missing = 0
    processed_clusters = set()
    for case_id in ids:
        if case_id.startswith("C"):
            entry = clean_by_id.get(case_id)
            if entry is None:
                print(f"[WARN] Clean pit ID {case_id} not found")
                missing += 1
                continue
            primary = entry["coordinate"]
            create_example_panel(
                global_coord=tuple(primary),
                extra_fcn_coords_front=_nearby(front_fcn, primary, tolerance),
                extra_dcc_coords_front=_nearby(front_dcc, primary, tolerance),
                extra_fcn_coords_back=_nearby(back_fcn, primary, tolerance),
                extra_dcc_coords_back=_nearby(back_dcc, primary, tolerance),
                save_folder="gallery/examples",
                save_name=case_id,
                title=case_id,
                description=None,
            )
        elif case_id.startswith(("FAD", "DAF")):
            cluster_id = case_id[-4:]
            if cluster_id in processed_clusters:
                continue
            processed_clusters.add(cluster_id)
            cluster = clusters.get(cluster_id)
            if cluster is None:
                print(f"[WARN] False-positive cluster {cluster_id} not found")
                missing += 1
                continue
            create_example_panel(
                global_coord=None,
                extra_fcn_coords=cluster["fcn_coords"],
                extra_dcc_coords=cluster["dcc_coords"],
                save_folder="gallery/examples",
                save_name=f"cluster_{cluster_id}",
                title=f"Cluster {cluster_id}",
                description=None,
            )
        elif case_id.startswith("FMD"):
            entry = fmd_by_id.get(case_id)
            if entry is None:
                print(f"[WARN] FMD ID {case_id} not found")
                missing += 1
                continue
            create_example_panel(
                global_coord=None,
                extra_fcn_coords=[entry["coordinate"]],
                save_folder="gallery/examples",
                save_name=case_id,
                title=case_id,
                description=None,
            )
        elif case_id.startswith("DMF"):
            entry = dmf_by_id.get(case_id)
            if entry is None:
                print(f"[WARN] DMF ID {case_id} not found")
                missing += 1
                continue
            create_example_panel(
                global_coord=None,
                extra_dcc_coords=[entry["coordinate"]],
                save_folder="gallery/examples",
                save_name=case_id,
                title=case_id,
                description=None,
            )
        else:
            print(f"[WARN] Unrecognized ID: {case_id}")
            missing += 1
            continue
        generated += 1

    # Keep these loads purposeful: they verify all aligned products required by
    # the manuscript panel workflow exist in the selected run.
    _ = aligned_fcn, aligned_dcc
    summary = {"generated": generated, "missing": missing}
    print(summary)
    return summary


if __name__ == "__main__":
    generate_example_panels()
