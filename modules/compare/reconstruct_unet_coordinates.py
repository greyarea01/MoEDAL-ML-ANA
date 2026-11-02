"""Convert U-Net probability masks to front/back global etch-pit coordinates."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from modules.dataset_tools import coordinates, image_tools, mhio
from modules.pipeline_paths import (
    RUN_MANIFEST,
    active_run_root,
    artifact_path,
    sha256_file,
)


FIRST_IMAGE_NUMBER = 251


def _extract_side(
    probability_maps: np.ndarray,
    *,
    threshold: float,
    sigma: float,
    alpha: float,
    beta: float,
    side: str,
) -> tuple[list[list[float]], list[list[int]], list[int]]:
    if probability_maps.ndim != 4 or probability_maps.shape[-1] != 1:
        raise ValueError(
            f"Expected probability maps shaped (N,H,W,1); found {probability_maps.shape}"
        )
    if side not in {"front", "back"}:
        raise ValueError(f"Unsupported side: {side}")

    coordinate_system = coordinates.Coord_system()
    if side == "front":
        coordinate_system["d"].flipped = False
        coordinate_system["d"].reverse = False
    else:
        coordinate_system["dr"].flipped = True
        coordinate_system["dr"].reverse = True

    global_points: list[list[float]] = []
    local_points: list[list[int]] = []
    per_image_counts: list[int] = []
    for array_index in range(probability_maps.shape[0]):
        probability = np.asarray(
            probability_maps[array_index, :, :, 0], dtype=np.float32
        )
        binary = probability >= threshold
        filtered = image_tools.laplace_filter(
            binary.astype(np.float32, copy=False),
            alpha=alpha,
            beta=beta,
            sigma=sigma,
        )
        centers = image_tools.coord_list(filtered)
        image_number = FIRST_IMAGE_NUMBER + array_index
        per_image_counts.append(int(len(centers)))
        for x_value, y_value in centers:
            local = [image_number, int(x_value), int(y_value)]
            if side == "front":
                c1_coordinate = coordinate_system.convert(local, "dc")
            else:
                c1_coordinate = coordinate_system.convert(local, "drc")
            global_xy = coordinate_system["c1"].get_gxy(c1_coordinate)
            local_points.append(local)
            global_points.append([float(global_xy[0]), float(global_xy[1])])
    return global_points, local_points, per_image_counts


def reconstruct_unet_coordinates(
    *,
    front_threshold: float,
    back_threshold: float,
    sigma: float,
    alpha: float,
    beta: float,
) -> dict:
    """Reconstruct both sides using shared Laplace parameters.

    The front and back probability thresholds are explicit and may differ;
    sigma, alpha, and beta are deliberately shared between both sides.
    """
    for name, value in {
        "front_threshold": front_threshold,
        "back_threshold": back_threshold,
    }.items():
        if not 0.0 < value < 1.0:
            raise ValueError(f"{name} must be strictly between 0 and 1")
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    if alpha < 0.0 or beta < 0.0:
        raise ValueError("alpha and beta must be nonnegative")

    front_path = artifact_path(
        "model_outputs/FCN/front_probability_masks.npy", must_exist=True
    )
    back_path = artifact_path(
        "model_outputs/FCN/back_probability_masks.npy", must_exist=True
    )
    front_maps = np.load(front_path, mmap_mode="r", allow_pickle=False)
    back_maps = np.load(back_path, mmap_mode="r", allow_pickle=False)
    if front_maps.shape != back_maps.shape:
        raise ValueError(
            f"Front/back probability shapes differ: {front_maps.shape} vs {back_maps.shape}"
        )

    started = time.time()
    front_global, front_local, front_counts = _extract_side(
        front_maps,
        threshold=front_threshold,
        sigma=sigma,
        alpha=alpha,
        beta=beta,
        side="front",
    )
    back_global, back_local, back_counts = _extract_side(
        back_maps,
        threshold=back_threshold,
        sigma=sigma,
        alpha=alpha,
        beta=beta,
        side="back",
    )

    mhio.save_json(front_global, "predicted_etch_pits/FCN/final_pr.txt")
    mhio.save_json(back_global, "predicted_etch_pits/FCN/final_reverse_pr.txt")
    mhio.save_json(
        front_local, "predicted_etch_pits/FCN/front_local_coordinates.txt"
    )
    mhio.save_json(
        back_local, "predicted_etch_pits/FCN/back_local_coordinates.txt"
    )

    report = {
        "parameters": {
            "front_threshold": front_threshold,
            "back_threshold": back_threshold,
            "sigma": sigma,
            "alpha": alpha,
            "beta": beta,
        },
        "coordinate_mapping": {
            "front": {"system": "d_to_c1", "flipped": False, "reverse": False},
            "back": {
                "system": "dr_to_c1",
                "flipped": True,
                "reverse": True,
            },
        },
        "first_image_number": FIRST_IMAGE_NUMBER,
        "inputs": {
            "front": {"path": str(front_path), "sha256": sha256_file(front_path)},
            "back": {"path": str(back_path), "sha256": sha256_file(back_path)},
        },
        "outputs": {
            "front_count": len(front_global),
            "back_count": len(back_global),
            "front_per_image_counts": front_counts,
            "back_per_image_counts": back_counts,
        },
        "elapsed_seconds": time.time() - started,
    }
    report_path = artifact_path(
        "predicted_etch_pits/FCN/reconstruction_metadata.json", for_write=True
    )
    with report_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")

    manifest_path = active_run_root() / RUN_MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["coordinate_reconstruction"] = report
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, manifest_path)
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--front-threshold", type=float, required=True)
    parser.add_argument("--back-threshold", type=float, required=True)
    parser.add_argument("--sigma", type=float, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--beta", type=float, required=True)
    return parser.parse_args()


def main() -> None:
    from modules.pipeline_paths import activate_pipeline_run

    args = _parse_args()
    activate_pipeline_run(args.run_dir)
    report = reconstruct_unet_coordinates(
        front_threshold=args.front_threshold,
        back_threshold=args.back_threshold,
        sigma=args.sigma,
        alpha=args.alpha,
        beta=args.beta,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
