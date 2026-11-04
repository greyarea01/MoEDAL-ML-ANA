"""Standalone JSON I/O used by the frozen coordinate-system snapshot."""

from __future__ import annotations

import json
from pathlib import Path


def load_json(filename):
    with Path(filename).open(encoding="utf-8") as stream:
        return json.load(stream)


def save_json(data, filename):
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(data, stream)
