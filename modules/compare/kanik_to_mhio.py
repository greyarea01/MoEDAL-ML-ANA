"""Convert DCC centroid text outputs to the pipeline's global-coordinate format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from modules.dataset_tools import coordinates, mhio
from modules.pipeline_paths import (
    activate_pipeline_run,
    resolve_pipeline_path,
    sha256_file,
)


def _load_local_centroids(path: Path, edge_threshold: int) -> list[list[float]]:
    x_max = 640 - edge_threshold
    y_max = 480 - edge_threshold
    points = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            fields = line.strip().split("\t")
            try:
                image_number, x_value, y_value = map(float, fields)
            except (TypeError, ValueError):
                continue
            if (
                image_number == 0
                or x_value < edge_threshold
                or y_value < edge_threshold
                or x_value > x_max
                or y_value > y_max
            ):
                continue
            points.append([image_number, x_value, y_value])
    return points


def _to_global(
    points: list[list[float]], *, side: str
) -> list[list[float]]:
    system = coordinates.Coord_system()
    if side == "front":
        system["d"].flipped = False
        system["d"].reverse = False
        conversion = "dc"
    elif side == "back":
        system["dr"].flipped = True
        system["dr"].reverse = True
        conversion = "drc"
    else:
        raise ValueError(f"Unsupported side: {side}")

    output = []
    for point in points:
        c1_coordinate = system.convert(point, conversion)
        output.append(system["c1"].get_gxy(c1_coordinate).tolist())
    return output


def convert_dcc_predictions(
    *,
    front_source: str = "model_outputs/DCC/DirtyTestCentroid.txt",
    back_source: str = "model_outputs/DCC/DirtyTestReverseCentroid.txt",
    edge_threshold: int = 4,
) -> dict:
    """Convert both DCC sides and save them under the active run."""
    if not 0 <= edge_threshold < 240:
        raise ValueError("edge_threshold must be between 0 and 239 pixels")
    front_path = resolve_pipeline_path(front_source, must_exist=True)
    back_path = resolve_pipeline_path(back_source, must_exist=True)
    front_local = _load_local_centroids(front_path, edge_threshold)
    back_local = _load_local_centroids(back_path, edge_threshold)
    front_global = _to_global(front_local, side="front")
    back_global = _to_global(back_local, side="back")
    mhio.save_json(front_global, "predicted_etch_pits/DCC/final_pr.txt")
    mhio.save_json(back_global, "predicted_etch_pits/DCC/final_reverse_pr.txt")
    mhio.save_json(
        front_local, "predicted_etch_pits/DCC/front_local_coordinates.txt"
    )
    mhio.save_json(
        back_local, "predicted_etch_pits/DCC/back_local_coordinates.txt"
    )
    report = {
        "edge_threshold": edge_threshold,
        "front": {
            "source": str(front_path),
            "source_sha256": sha256_file(front_path),
            "count": len(front_global),
        },
        "back": {
            "source": str(back_path),
            "source_sha256": sha256_file(back_path),
            "count": len(back_global),
        },
        "mapping": {
            "front": {
                "system": "d_to_c1",
                "preset": "dc",
                "offset": list(coordinates.Coord_system.offset["dc"]),
                "flipped": False,
                "reverse": False,
            },
            "back": {
                "system": "dr_to_c1",
                "preset": "drc",
                "offset": list(coordinates.Coord_system.offset["drc"]),
                "flipped": True,
                "reverse": True,
            },
        },
    }
    mhio.save_json(report, "predicted_etch_pits/DCC/conversion_metadata.json")
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--front-source", default="model_outputs/DCC/DirtyTestCentroid.txt"
    )
    parser.add_argument(
        "--back-source", default="model_outputs/DCC/DirtyTestReverseCentroid.txt"
    )
    parser.add_argument("--edge-threshold", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    activate_pipeline_run(args.run_dir)
    print(
        json.dumps(
            convert_dcc_predictions(
                front_source=args.front_source,
                back_source=args.back_source,
                edge_threshold=args.edge_threshold,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
