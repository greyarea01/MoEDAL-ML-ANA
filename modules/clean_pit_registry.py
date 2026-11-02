"""Load immutable clean-pit IDs and select a run's active truth records."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from numbers import Real
from pathlib import Path
from typing import Iterable, Sequence

from modules.dataset_tools import mhio
from modules.pipeline_paths import PROJECT_ROOT, source_path


CLEAN_ID_PATTERN = re.compile(r"^C\d{4,}$")
REGISTRY_ROOT = (PROJECT_ROOT / "clean_pit_id_registries").resolve()
CUT_ROOT = (PROJECT_ROOT / "fiducial_cuts").resolve()


def _require_rooted_file(path: Path, root: Path, *, kind: str) -> None:
    if root not in path.parents:
        raise ValueError(f"{kind} must be stored below {root}: {path}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_coordinate(coordinate: object, *, context: str) -> list[Real]:
    if not isinstance(coordinate, list) or len(coordinate) != 2:
        raise ValueError(f"{context} must contain a two-value coordinate list")
    if any(
        isinstance(value, bool) or not isinstance(value, Real)
        for value in coordinate
    ):
        raise ValueError(f"{context} coordinate values must be numeric")
    return coordinate


def validate_clean_id_records(
    records: object, *, context: str = "clean-pit registry"
) -> list[dict]:
    """Validate ID records without changing their order or stored values."""
    if not isinstance(records, list):
        raise ValueError(f"{context} must be a JSON list")

    clean_records = []
    seen_ids = set()
    for index, record in enumerate(records):
        record_context = f"{context} record {index}"
        if not isinstance(record, dict):
            raise ValueError(f"{record_context} must be an object")
        clean_id = record.get("id")
        if not isinstance(clean_id, str) or not CLEAN_ID_PATTERN.fullmatch(clean_id):
            raise ValueError(f"{record_context} has an invalid clean ID: {clean_id!r}")
        if clean_id in seen_ids:
            raise ValueError(f"{context} contains duplicate ID {clean_id}")
        _validate_coordinate(record.get("coordinate"), context=record_context)
        seen_ids.add(clean_id)
        clean_records.append(record)
    return clean_records


def load_clean_id_registry(
    registry_path: str | Path,
) -> tuple[Path, list[dict], str]:
    """Load one explicitly selected, repository-external-to-runs registry."""
    path = source_path(registry_path)
    _require_rooted_file(path, REGISTRY_ROOT, kind="Clean-pit ID registry")
    if not path.is_file():
        raise FileNotFoundError(f"Clean-pit ID registry is not a file: {path}")
    with path.open(encoding="utf-8") as stream:
        records = validate_clean_id_records(
            json.load(stream), context=f"clean-pit registry {path}"
        )
    return path, records, _sha256(path)


def _load_cut_exclusions(
    cut_paths: Iterable[str | Path],
    *,
    registry_ids: set[str],
    registry_sha256: str,
) -> tuple[list[Path], dict[str, list[dict]]]:
    resolved_paths = []
    exclusions: dict[str, list[dict]] = defaultdict(list)

    for cut_path in cut_paths:
        path = source_path(cut_path)
        _require_rooted_file(path, CUT_ROOT, kind="Fiducial cut file")
        resolved_paths.append(path)
        with path.open(encoding="utf-8") as stream:
            payload = json.load(stream)
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            raise ValueError(f"Fiducial cut file must use schema_version 1: {path}")
        expected_hash = payload.get("clean_id_registry_sha256")
        if expected_hash != registry_sha256:
            raise ValueError(
                f"Fiducial cut file {path} targets registry SHA-256 "
                f"{expected_hash!r}, not selected registry {registry_sha256}"
            )
        entries = payload.get("exclude_clean_ids", [])
        if not isinstance(entries, list):
            raise ValueError(f"exclude_clean_ids must be a list in {path}")
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise ValueError(
                    f"Fiducial cut entry {index} in {path} must be an object"
                )
            clean_id = entry.get("id")
            reason = entry.get("reason")
            if clean_id not in registry_ids:
                raise ValueError(
                    f"Fiducial cut file {path} references unknown ID {clean_id!r}"
                )
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError(
                    f"Fiducial cut entry {clean_id} in {path} needs a reason"
                )
            exclusions[clean_id].append(
                {"cut_file": str(path), "reason": reason.strip()}
            )

    return resolved_paths, exclusions


def validate_clean_truth_configuration(
    registry_path: str | Path,
    fiducial_cut_files: Iterable[str | Path] = (),
) -> dict:
    """Validate selected immutable inputs before a pipeline run is created."""
    path, registry, registry_hash = load_clean_id_registry(registry_path)
    cut_paths, exclusions = _load_cut_exclusions(
        fiducial_cut_files,
        registry_ids={record["id"] for record in registry},
        registry_sha256=registry_hash,
    )
    return {
        "registry_path": str(path),
        "registry_sha256": registry_hash,
        "registry_count": len(registry),
        "fiducial_cut_files": [str(item) for item in cut_paths],
        "excluded_ids": list(exclusions),
    }


def select_clean_truth(
    regenerated_coordinates: Sequence[Sequence[Real]],
    *,
    registry_path: str | Path,
    fiducial_cut_files: Iterable[str | Path] = (),
) -> dict:
    """Validate the regenerated truth, then filter registry records by ID.

    IDs are never inferred from row positions here.  The row-by-row coordinate
    check is an integrity assertion against the selected immutable registry.
    Any intentional base-truth change therefore requires an explicit registry
    migration instead of silently renumbering clean pits.
    """
    path, registry, registry_hash = load_clean_id_registry(registry_path)
    regenerated = [list(coordinate) for coordinate in regenerated_coordinates]
    registry_coordinates = [record["coordinate"] for record in registry]

    if len(regenerated) != len(registry_coordinates):
        raise ValueError(
            "Regenerated clean truth does not match the selected ID registry: "
            f"{len(regenerated)} coordinates versus {len(registry_coordinates)} records"
        )
    for index, (coordinate, expected) in enumerate(
        zip(regenerated, registry_coordinates)
    ):
        _validate_coordinate(coordinate, context=f"regenerated clean truth row {index}")
        if coordinate != expected:
            raise ValueError(
                "Regenerated clean truth does not match the selected ID registry at "
                f"row {index} ({registry[index]['id']}): {coordinate} != {expected}"
            )

    cut_paths, exclusions = _load_cut_exclusions(
        fiducial_cut_files,
        registry_ids={record["id"] for record in registry},
        registry_sha256=registry_hash,
    )
    active = [record for record in registry if record["id"] not in exclusions]
    excluded = []
    for record in registry:
        clean_id = record["id"]
        if clean_id in exclusions:
            excluded_record = dict(record)
            excluded_record["exclusions"] = exclusions[clean_id]
            excluded.append(excluded_record)

    return {
        "registry_path": str(path),
        "registry_sha256": registry_hash,
        "registry_count": len(registry),
        "fiducial_cut_files": [str(item) for item in cut_paths],
        "active_count": len(active),
        "excluded_count": len(excluded),
        "active_records": active,
        "excluded_records": excluded,
    }


def save_clean_truth_selection(selection: dict) -> None:
    """Save active coordinates/IDs and the complete exclusion audit."""
    active_records = selection["active_records"]
    excluded_records = selection["excluded_records"]
    mhio.save_json(
        [record["coordinate"] for record in active_records], "Aligned_clean.txt"
    )
    mhio.save_json(
        active_records, "predicted_list/ids/Aligned_clean_with_ids.txt"
    )
    mhio.save_json(
        excluded_records, "predicted_list/ids/Excluded_clean_with_ids.txt"
    )
    report = {
        key: value
        for key, value in selection.items()
        if key not in {"active_records", "excluded_records"}
    }
    report["active_ids"] = [record["id"] for record in active_records]
    report["excluded_ids"] = [record["id"] for record in excluded_records]
    mhio.save_json(report, "predicted_list/ids/clean_truth_selection.json")
