from __future__ import annotations

import argparse
from pathlib import Path

from modules.clean_pit_registry import (
    save_clean_truth_selection,
    select_clean_truth,
)
from modules.dataset_tools import mhio, coordinates

CS = coordinates.Coord_system()


def clean_common_aligned(
    threshold: int = 4,
    *,
    clean_id_registry: str | Path,
    fiducial_cut_files: tuple[str | Path, ...] = (),
    spatial_fiducial_file: str | Path | None = None,
):
    """
    Filters the common etch pit list based on a threshold, removing pits
    that are outside of the defined x and y coordinate limits.

    Parameters:
    - threshold: The margin threshold for filtering out pits close to the edges (default is 4).

    Returns:
    - None
    """

    # Set thresholds for valid coordinates
    x_threshold = 640 - threshold
    y_threshold = 480 - threshold

    # Load the pit lists
    list_1 = mhio.load_json("predicted_list/universal_true_etch_pit_list/Aligned_K_Not_L.txt")  # Kanik's only
    list_2 = mhio.load_json("predicted_list/universal_true_etch_pit_list/Aligned_L_Not_K.txt")  # Lewis's only
    list_3 = mhio.load_json("predicted_list/universal_true_etch_pit_list/Aligned_common.txt")  # Common pits

    common = list_3
    mask = []

    # Loop through the common pits and apply filtering based on coordinate thresholds
    for i in range(len(common)):
        tmp = CS['c1'].get_ixy(common[i])  # Convert to ixy coordinates
        tmp = CS.convert(tmp, 'cd')  # Convert to cd coordinates
        # Filter out pits that are too close to the edges or below the minimum depth threshold
        if tmp[1] < threshold or tmp[2] < threshold or tmp[1] > x_threshold or tmp[2] > y_threshold or tmp[0] < 251:
            continue
        mask.append(common[i][:])  # Append valid pits to the mask

    common = mask  # Update the common pit list with filtered results
    selection = select_clean_truth(
        common,
        registry_path=clean_id_registry,
        fiducial_cut_files=fiducial_cut_files,
        spatial_fiducial_file=spatial_fiducial_file,
    )
    save_clean_truth_selection(selection)
    print(
        "Clean truth: "
        f"{selection['active_count']} active, "
        f"{selection['excluded_count']} excluded, "
        f"{selection['registry_count']} registered"
    )
    return selection


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-id-registry", required=True)
    parser.add_argument("--fiducial-cut-file", action="append", default=[])
    parser.add_argument("--spatial-fiducial-file")
    parser.add_argument("--threshold", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    clean_common_aligned(
        threshold=args.threshold,
        clean_id_registry=args.clean_id_registry,
        fiducial_cut_files=tuple(args.fiducial_cut_file),
        spatial_fiducial_file=args.spatial_fiducial_file,
    )
