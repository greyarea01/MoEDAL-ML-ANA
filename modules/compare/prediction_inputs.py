"""Resolve raw or fiducial-filtered prediction inputs for comparison stages."""

from __future__ import annotations

from modules.dataset_tools import mhio
from modules.pipeline_paths import resolve_pipeline_path


def prediction_relative_path(method: str, side: str) -> str:
    if method not in {"CNN", "FCN", "DCC"}:
        raise ValueError(f"Unsupported prediction method: {method}")
    if side not in {"front", "back"}:
        raise ValueError(f"Unsupported prediction side: {side}")
    suffix = "pr" if side == "front" else "reverse_pr"
    filtered = f"predicted_etch_pits/{method}/fiducial_{suffix}.txt"
    if resolve_pipeline_path(filtered, must_exist=False).is_file():
        return filtered
    return f"predicted_etch_pits/{method}/final_{suffix}.txt"


def load_prediction_coordinates(method: str, side: str) -> list:
    return mhio.load_json(prediction_relative_path(method, side))

