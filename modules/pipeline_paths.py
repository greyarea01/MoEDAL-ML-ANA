"""Safe, configurable paths for MoEDAL analysis pipeline runs.

Historical results live at the repository root.  New pipeline runs must be
created explicitly and write the same relative artifact layout below one run
directory.  This keeps old results immutable and makes a complete run movable.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_DIR_ENV = "MOEDAL_PIPELINE_RUN_DIR"
RUN_MANIFEST = "pipeline_run.json"

# Relative paths beginning with these directories are generated artifacts.
ARTIFACT_DIRECTORIES = frozenset(
    {
        "gallery",
        "kanik",
        "MLout",
        "model_outputs",
        "my_panels",
        "plots",
        "predicted_etch_pits",
        "predicted_list",
        "probability_masks",
        "stitched_images",
    }
)

# Files that were historically written at repository root or mixed with inputs.
ARTIFACT_FILES = frozenset({"Aligned_clean.txt"})


def _absolute(path: str | os.PathLike[str], base: Path = PROJECT_ROOT) -> Path:
    value = Path(path).expanduser()
    if not value.is_absolute():
        value = base / value
    return value.resolve()


def _validate_relative(path: str | os.PathLike[str]) -> Path:
    value = Path(path)
    if value.is_absolute() or value == Path() or ".." in value.parts:
        raise ValueError(f"Expected a safe relative path, found: {path}")
    return value


def _is_pipeline_artifact(path: Path) -> bool:
    if not path.parts:
        return False
    if path.parts[0] in ARTIFACT_DIRECTORIES or path.as_posix() in ARTIFACT_FILES:
        return True
    if path.parts[0] == "numpy_data_eval" and path.name.startswith("final_"):
        return True
    return False


def sha256_file(path: str | os.PathLike[str], chunk_size: int = 8 * 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest without loading a large file in memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state() -> dict:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=PROJECT_ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"revision": revision, "tracked_files_dirty": dirty}
    except (OSError, subprocess.CalledProcessError) as error:
        return {"error": str(error)}


def create_pipeline_run(
    run_dir: str | os.PathLike[str],
    *,
    metadata: dict | None = None,
) -> Path:
    """Create and activate a new, empty pipeline run.

    Existing paths are rejected even when empty.  A new run name is therefore
    required for every rerun, preventing accidental replacement of old output.
    """
    root = _absolute(run_dir)
    forbidden = {PROJECT_ROOT, PROJECT_ROOT.parent, Path("/")}
    if root in forbidden:
        raise ValueError(f"Unsafe pipeline run directory: {root}")
    legacy_roots = [PROJECT_ROOT / name for name in ARTIFACT_DIRECTORIES]
    legacy_roots.append(PROJECT_ROOT / "numpy_data_eval")
    if any(legacy == root or legacy in root.parents for legacy in legacy_roots):
        raise ValueError(
            f"A run cannot be nested inside a legacy data/result directory: {root}"
        )
    if root.exists():
        raise FileExistsError(
            f"Refusing to reuse pipeline run directory: {root}. "
            "Choose a new run name so no prior result can be overwritten."
        )

    root.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(PROJECT_ROOT),
        "run_root": str(root),
        "git": _git_state(),
        "metadata": metadata or {},
        "seeded_artifacts": [],
        "source_inputs": [],
    }
    (root / RUN_MANIFEST).write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    os.environ[RUN_DIR_ENV] = str(root)
    return root


def activate_pipeline_run(run_dir: str | os.PathLike[str]) -> Path:
    """Activate an existing run created by :func:`create_pipeline_run`."""
    root = _absolute(run_dir)
    manifest = root / RUN_MANIFEST
    if not manifest.is_file():
        raise FileNotFoundError(
            f"Not a pipeline run (missing {RUN_MANIFEST}): {root}"
        )
    os.environ[RUN_DIR_ENV] = str(root)
    return root


def active_run_root() -> Path:
    """Return the active run root or fail before legacy data can be overwritten."""
    configured = os.environ.get(RUN_DIR_ENV)
    if not configured:
        raise RuntimeError(
            f"No pipeline run is active. Call create_pipeline_run(...) first or set "
            f"{RUN_DIR_ENV} to an existing marked run directory."
        )
    return activate_pipeline_run(configured)


def source_path(
    relative_path: str | os.PathLike[str], *, must_exist: bool = True
) -> Path:
    """Resolve an immutable repository input independently of the working directory."""
    value = Path(relative_path).expanduser()
    if value.is_absolute():
        path = value.resolve()
    else:
        relative = _validate_relative(value)
        path = (PROJECT_ROOT / relative).resolve()
    if must_exist and not path.exists():
        raise FileNotFoundError(f"Pipeline source input does not exist: {path}")
    return path


def artifact_path(
    relative_path: str | os.PathLike[str],
    *,
    for_write: bool = False,
    must_exist: bool = False,
) -> Path:
    """Resolve a path beneath the active run, optionally protecting a new write."""
    relative = _validate_relative(relative_path)
    path = (active_run_root() / relative).resolve()
    if active_run_root() not in path.parents:
        raise ValueError(f"Artifact escaped the active run directory: {path}")
    if for_write:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite pipeline artifact: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    if must_exist and not path.exists():
        raise FileNotFoundError(f"Required pipeline artifact does not exist: {path}")
    return path


def artifact_directory(relative_path: str | os.PathLike[str]) -> Path:
    """Create or return a directory beneath the active run."""
    path = artifact_path(relative_path)
    if path.exists() and not path.is_dir():
        raise FileExistsError(f"Artifact directory path is already a file: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_output_path(
    path: str | os.PathLike[str], *, for_write: bool = True
) -> Path:
    """Resolve a relative or absolute output while enforcing the active run boundary."""
    value = Path(path)
    if not value.is_absolute():
        return artifact_path(value, for_write=for_write)
    resolved = value.expanduser().resolve()
    root = active_run_root()
    if root not in resolved.parents:
        raise ValueError(f"Output is outside the active pipeline run: {resolved}")
    if for_write:
        if resolved.exists():
            raise FileExistsError(f"Refusing to overwrite pipeline artifact: {resolved}")
        resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def resolve_pipeline_path(
    path: str | os.PathLike[str],
    *,
    for_write: bool = False,
    must_exist: bool = False,
) -> Path:
    """Resolve legacy relative names to either a run artifact or source input.

    Absolute paths remain explicit and are never silently redirected.
    """
    value = Path(path)
    if value.is_absolute():
        resolved = value.expanduser().resolve()
        if for_write:
            if resolved.exists():
                raise FileExistsError(f"Refusing to overwrite explicit output: {resolved}")
            resolved.parent.mkdir(parents=True, exist_ok=True)
        if must_exist and not resolved.exists():
            raise FileNotFoundError(resolved)
        return resolved

    relative = _validate_relative(value)
    if _is_pipeline_artifact(relative):
        if not os.environ.get(RUN_DIR_ENV) and not for_write:
            return source_path(relative, must_exist=must_exist)
        return artifact_path(relative, for_write=for_write, must_exist=must_exist)
    if for_write:
        path = source_path(relative, must_exist=False)
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing file: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    return source_path(relative, must_exist=must_exist)


def seed_artifacts(
    relative_paths: Iterable[str | os.PathLike[str]],
    *,
    source_root: str | os.PathLike[str] = PROJECT_ROOT,
) -> list[Path]:
    """Copy starting artifacts into the active run and record hashes in its manifest."""
    source_base = _absolute(source_root)
    records = []
    destinations = []
    for item in relative_paths:
        relative = _validate_relative(item)
        source = (source_base / relative).resolve()
        if source_base != source and source_base not in source.parents:
            raise ValueError(f"Seed source escaped its root: {source}")
        if not source.is_file():
            raise FileNotFoundError(f"Seed artifact does not exist: {source}")
        destination = artifact_path(relative, for_write=True)
        shutil.copy2(source, destination)
        digest = sha256_file(destination)
        records.append(
            {
                "relative_path": relative.as_posix(),
                "source": str(source),
                "destination": str(destination),
                "sha256": digest,
                "size_bytes": destination.stat().st_size,
            }
        )
        destinations.append(destination)

    manifest_path = active_run_root() / RUN_MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["seeded_artifacts"].extend(records)
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, manifest_path)
    return destinations


def copy_source_artifact(
    source_pathname: str | os.PathLike[str],
    destination_relative_path: str | os.PathLike[str],
) -> Path:
    """Copy one immutable source to a differently named run artifact."""
    source = source_path(source_pathname)
    if not source.is_file():
        raise FileNotFoundError(f"Seed source is not a file: {source}")
    relative = _validate_relative(destination_relative_path)
    destination = artifact_path(relative, for_write=True)
    shutil.copy2(source, destination)
    digest = sha256_file(destination)

    manifest_path = active_run_root() / RUN_MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["seeded_artifacts"].append(
        {
            "relative_path": relative.as_posix(),
            "source": str(source),
            "destination": str(destination),
            "sha256": digest,
            "size_bytes": destination.stat().st_size,
        }
    )
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, manifest_path)
    return destination


def record_source_inputs(
    relative_paths: Iterable[str | os.PathLike[str]],
) -> list[dict]:
    """Record immutable input locations, sizes, and hashes in the run manifest."""
    records = []
    for item in relative_paths:
        relative = _validate_relative(item)
        path = source_path(relative)
        if not path.is_file():
            raise FileNotFoundError(f"Source input is not a file: {path}")
        records.append(
            {
                "relative_path": relative.as_posix(),
                "path": str(path),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    manifest_path = active_run_root() / RUN_MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_inputs"].extend(records)
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, manifest_path)
    return records

