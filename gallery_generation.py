"""Generate the full clean-pit and false-positive panel gallery for one run."""

from __future__ import annotations

import numpy as np

from modules.compare.panels import create_image_panel
from modules.dataset_tools import coordinates, mhio


def _nearby(predictions, coordinate, tolerance):
    overlap = coordinates.Overlap(
        np.asarray(predictions), np.asarray([coordinate]), tolerance
    )
    return [
        predictions[index]
        for index, distance in enumerate(overlap.dist[:, 0])
        if distance < tolerance
    ]


def generate_gallery(tolerance: float = 15, crop_size: int = 56) -> dict[str, int]:
    """Write every gallery panel below ``gallery/56pixel`` in the active run."""
    clean_pits = mhio.load_json("predicted_list/ids/Aligned_clean_with_ids.txt")
    predicted_fcn = mhio.load_json(
        "predicted_list/FCN/aligned/final_pred_list.txt"
    )
    predicted_dcc = mhio.load_json(
        "predicted_list/DCC/aligned/final_pred_list.txt"
    )
    fad_ids = mhio.load_json("predicted_list/ids/FP_FCN_common_DCC_with_ids.txt")
    daf_ids = mhio.load_json("predicted_list/ids/FP_DCC_common_FCN_with_ids.txt")
    fmd_ids = mhio.load_json("predicted_list/ids/FP_unique_FCN_with_ids.txt")
    dmf_ids = mhio.load_json("predicted_list/ids/FP_unique_DCC_with_ids.txt")

    clean_dir = "gallery/56pixel/clean_pits"
    intersection_dir = "gallery/56pixel/false_positive/intersection"
    fmd_dir = "gallery/56pixel/false_positive/FMD"
    dmf_dir = "gallery/56pixel/false_positive/DMF"

    for clean_pit in clean_pits:
        primary_id = clean_pit["id"]
        primary = clean_pit["coordinate"]
        extra_fcn = _nearby(predicted_fcn, primary, tolerance)
        extra_dcc = _nearby(predicted_dcc, primary, tolerance)
        description = "Extra coordinates:\n"
        if extra_fcn:
            description += f"FCN ({len(extra_fcn)} points): {extra_fcn}\n"
        if extra_dcc:
            description += f"DCC ({len(extra_dcc)} points): {extra_dcc}\n"
        create_image_panel(
            global_coord=primary,
            save_folder=clean_dir,
            title=f"{primary_id} {primary}",
            description=description,
            extra_fcn_coords=extra_fcn,
            extra_dcc_coords=extra_dcc,
            crop_size=crop_size,
        )

    clusters = {}
    for item in fad_ids + daf_ids:
        cluster_id = item["id"][-4:]
        cluster = clusters.setdefault(
            cluster_id, {"fcn_coords": [], "dcc_coords": []}
        )
        key = "fcn_coords" if item["id"].startswith("FAD") else "dcc_coords"
        cluster[key].append(item["coordinate"])
    for cluster_id, cluster in clusters.items():
        description = "Cluster Details:\n"
        if cluster["fcn_coords"]:
            description += (
                f"FAD Points ({len(cluster['fcn_coords'])}): "
                f"{cluster['fcn_coords']}\n"
            )
        if cluster["dcc_coords"]:
            description += (
                f"DAF Points ({len(cluster['dcc_coords'])}): "
                f"{cluster['dcc_coords']}\n"
            )
        create_image_panel(
            global_coord=None,
            save_folder=intersection_dir,
            title=f"false_positive_cluster_{cluster_id}",
            description=description,
            extra_fcn_coords=cluster["fcn_coords"],
            extra_dcc_coords=cluster["dcc_coords"],
            crop_size=crop_size,
        )

    for item in fmd_ids:
        create_image_panel(
            global_coord=None,
            save_folder=fmd_dir,
            title=f"{item['id']} {item['coordinate']}",
            extra_fcn_coords=[item["coordinate"]],
            crop_size=crop_size,
        )
    for item in dmf_ids:
        create_image_panel(
            global_coord=None,
            save_folder=dmf_dir,
            title=f"{item['id']} {item['coordinate']}",
            extra_dcc_coords=[item["coordinate"]],
            crop_size=crop_size,
        )

    summary = {
        "clean_panels": len(clean_pits),
        "intersection_panels": len(clusters),
        "fmd_panels": len(fmd_ids),
        "dmf_panels": len(dmf_ids),
    }
    print(summary)
    return summary


if __name__ == "__main__":
    generate_gallery()
