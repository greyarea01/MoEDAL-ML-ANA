"""Validate and stage the three supported pipeline starting points."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from modules.pipeline_paths import (
    copy_source_artifact,
    seed_artifacts,
    source_path,
)


STARTING_POINTS = (
    "model",
    "model_output",
    "preprocessed_predicted_coordinates",
)

PREDICTION_COORDINATE_ARTIFACTS = [
    f"predicted_etch_pits/{method}/{side}.txt"
    for method in ("CNN", "FCN", "DCC")
    for side in ("final_pr", "final_reverse_pr")
]
CNN_COORDINATE_ARTIFACTS = [
    "predicted_etch_pits/CNN/final_pr.txt",
    "predicted_etch_pits/CNN/final_reverse_pr.txt",
]
CNN_AUXILIARY_ARTIFACTS = [
    "predicted_etch_pits/CNN/GXY_allchan_008thresh_eval.txt",
    "predicted_etch_pits/CNN/IXY_allchan_008thresh.txt",
    "predicted_etch_pits/CNN/IXY_allchan_008thresh_eval.txt",
]
CNN_STARTING_ARTIFACTS = [
    *CNN_COORDINATE_ARTIFACTS,
    *CNN_AUXILIARY_ARTIFACTS,
]
OPTIONAL_COORDINATE_PROVENANCE_ARTIFACTS = [
    "predicted_etch_pits/FCN/front_local_coordinates.txt",
    "predicted_etch_pits/FCN/back_local_coordinates.txt",
    "predicted_etch_pits/FCN/reconstruction_metadata.json",
    "predicted_etch_pits/DCC/conversion_metadata.json",
]

FCN_FRONT_MODEL_OUTPUT = "model_outputs/FCN/front_probability_masks.npy"
FCN_BACK_MODEL_OUTPUT = "model_outputs/FCN/back_probability_masks.npy"
FCN_MODEL_OUTPUT_METADATA = "model_outputs/FCN/generation_metadata.json"
DCC_FRONT_MODEL_OUTPUT = "model_outputs/DCC/DirtyTestCentroid.txt"
DCC_BACK_MODEL_OUTPUT = "model_outputs/DCC/DirtyTestReverseCentroid.txt"
DCC_FRONT_KANIK_OUTPUT = "kanik/og_data/DirtyTestCentroid.txt"
DCC_BACK_KANIK_OUTPUT = "kanik/og_data/DirtyTestReverseCentroid.txt"
KANIK_CLEAN_TRUTH = "kanik/clean_aligned.txt"


def resolve_input_source(input_source: str | Path) -> Path:
    """Resolve and validate the common source root used by starting artifacts."""
    root = source_path(input_source)
    if not root.is_dir():
        raise NotADirectoryError(f"Pipeline input source is not a directory: {root}")
    return root


def source_file_under(
    input_source: str | Path, relative_path: str | Path, *, required: bool = True
) -> Path:
    """Resolve a safe file below an explicitly selected source root."""
    root = resolve_input_source(input_source)
    relative = Path(relative_path)
    if relative.is_absolute() or relative == Path() or ".." in relative.parts:
        raise ValueError(f"Expected a safe source-relative path, found: {relative_path}")
    path = (root / relative).resolve()
    if root not in path.parents:
        raise ValueError(f"Starting artifact escaped its source root: {path}")
    if required and not path.is_file():
        raise FileNotFoundError(f"Required starting artifact does not exist: {path}")
    return path


def _require_source_artifacts(
    input_source: str | Path, relative_paths: Iterable[str]
) -> None:
    for relative_path in relative_paths:
        source_file_under(input_source, relative_path)


def _kanik_source_artifacts(input_source: str | Path) -> tuple[Path, list[str]]:
    """Return every regular file in the archived Kanik tree."""
    root = resolve_input_source(input_source)
    directory = source_file_under(root, "kanik", required=False)
    if not directory.is_dir():
        raise NotADirectoryError(f"Archived Kanik directory does not exist: {directory}")
    paths = []
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Archived Kanik tree may not contain symlinks: {path}")
        if path.is_file():
            paths.append(path.relative_to(root).as_posix())
    if not paths:
        raise FileNotFoundError(f"Archived Kanik directory contains no files: {directory}")
    return root, paths


def validate_starting_point_inputs(
    *,
    start_from: str,
    input_source: str | Path,
    fcn_front_model_output: str = FCN_FRONT_MODEL_OUTPUT,
    fcn_back_model_output: str = FCN_BACK_MODEL_OUTPUT,
    model_path: str | Path = "models/FCN/unet_ROC.keras",
    front_images: str | Path = "numpy_data_eval/preprocessed_dirty_eval.npy",
    back_images: str | Path = "numpy_data_eval/preprocessed_dirty_reverse_eval.npy",
    dcc_front_model_output: str = DCC_FRONT_KANIK_OUTPUT,
    dcc_back_model_output: str = DCC_BACK_KANIK_OUTPUT,
) -> None:
    """Fail before run creation when a selected starting input is unavailable."""
    if start_from not in STARTING_POINTS:
        raise ValueError(f"Unsupported pipeline starting point: {start_from}")
    resolve_input_source(input_source)
    _kanik_source_artifacts(input_source)
    source_file_under(input_source, KANIK_CLEAN_TRUTH)

    if start_from == "preprocessed_predicted_coordinates":
        _require_source_artifacts(
            input_source,
            [*PREDICTION_COORDINATE_ARTIFACTS, *CNN_AUXILIARY_ARTIFACTS],
        )
        return

    _require_source_artifacts(input_source, CNN_STARTING_ARTIFACTS)
    source_file_under(input_source, dcc_front_model_output)
    source_file_under(input_source, dcc_back_model_output)
    if start_from == "model_output":
        source_file_under(input_source, fcn_front_model_output)
        source_file_under(input_source, fcn_back_model_output)
    else:
        source_path(model_path)
        source_path(front_images)
        source_path(back_images)


def seed_preprocessed_prediction_coordinates(input_source: str | Path) -> dict:
    """Copy all six final coordinates and three auxiliary CNN results."""
    root = resolve_input_source(input_source)
    destinations = seed_artifacts(PREDICTION_COORDINATE_ARTIFACTS, source_root=root)
    cnn_auxiliary = seed_artifacts(CNN_AUXILIARY_ARTIFACTS, source_root=root)
    optional = [
        relative
        for relative in OPTIONAL_COORDINATE_PROVENANCE_ARTIFACTS
        if source_file_under(root, relative, required=False).is_file()
    ]
    if optional:
        seed_artifacts(optional, source_root=root)
    return {
        "source_root": str(root),
        "coordinate_artifacts": [str(path) for path in destinations],
        "cnn_auxiliary_artifacts": [str(path) for path in cnn_auxiliary],
        "provenance_artifacts": optional,
    }


def seed_kanik_archive(input_source: str | Path) -> dict:
    """Copy the complete archived Kanik tree into the active run."""
    root, relative_paths = _kanik_source_artifacts(input_source)
    destinations = seed_artifacts(relative_paths, source_root=root)
    return {
        "source_root": str(root),
        "artifacts": [str(path) for path in destinations],
        "file_count": len(destinations),
    }


def seed_cnn_prediction_coordinates(input_source: str | Path) -> dict:
    """Copy CNN front/back coordinates and its three auxiliary result files."""
    root = resolve_input_source(input_source)
    destinations = seed_artifacts(CNN_STARTING_ARTIFACTS, source_root=root)
    return {
        "source_root": str(root),
        "coordinate_artifacts": [str(path) for path in destinations[:2]],
        "auxiliary_artifacts": [str(path) for path in destinations[2:]],
    }


def seed_fcn_model_outputs(
    input_source: str | Path,
    *,
    front_relative_path: str = FCN_FRONT_MODEL_OUTPUT,
    back_relative_path: str = FCN_BACK_MODEL_OUTPUT,
) -> dict:
    """Copy externally generated FCN probability maps into canonical run paths."""
    root = resolve_input_source(input_source)
    front_source = source_file_under(root, front_relative_path)
    back_source = source_file_under(root, back_relative_path)
    front = copy_source_artifact(front_source, FCN_FRONT_MODEL_OUTPUT)
    back = copy_source_artifact(back_source, FCN_BACK_MODEL_OUTPUT)

    metadata_source = source_file_under(
        root, FCN_MODEL_OUTPUT_METADATA, required=False
    )
    metadata = None
    if metadata_source.is_file():
        metadata = copy_source_artifact(metadata_source, FCN_MODEL_OUTPUT_METADATA)
    return {
        "source_root": str(root),
        "front": str(front),
        "back": str(back),
        "source_metadata": str(metadata) if metadata else None,
    }


def seed_dcc_model_outputs(
    input_source: str | Path,
    *,
    front_relative_path: str = DCC_FRONT_KANIK_OUTPUT,
    back_relative_path: str = DCC_BACK_KANIK_OUTPUT,
) -> dict:
    """Copy Kanik's raw DCC centroid outputs into the active run."""
    root = resolve_input_source(input_source)
    front_source = source_file_under(root, front_relative_path)
    back_source = source_file_under(root, back_relative_path)
    front = copy_source_artifact(front_source, DCC_FRONT_MODEL_OUTPUT)
    back = copy_source_artifact(back_source, DCC_BACK_MODEL_OUTPUT)
    return {"source_root": str(root), "front": str(front), "back": str(back)}
