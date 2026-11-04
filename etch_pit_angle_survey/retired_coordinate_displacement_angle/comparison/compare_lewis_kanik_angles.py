#!/usr/bin/env python3
"""Compare every unique Lewis and Kanik candidate angle for each clean ID."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from decimal import Decimal
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SURVEY_ROOT = SCRIPT_DIR.parent
REPO_ROOT = SCRIPT_DIR.parents[2]
LEWIS_CANDIDATES_PATH = SURVEY_ROOT / "lewis/data/all_candidate_pairs.csv"
KANIK_CANDIDATES_PATH = SURVEY_ROOT / "kanik/data/all_candidate_pairs.csv"
TRUTH_IDS_PATH = REPO_ROOT / "predicted_list/ids/Aligned_clean_with_ids.txt"
DEFAULT_OUTPUT_PATH = SCRIPT_DIR / "data/lewis_kanik_angle_comparison.csv"
ANGLE_COLUMN = "c1_c2_angle_from_foil_normal_deg"


def load_unique_angles(path: Path) -> dict[str, list[str]]:
    angles: dict[str, list[str]] = defaultdict(list)
    seen: dict[str, set[str]] = defaultdict(set)
    with path.open(encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        required = {"clean_id", ANGLE_COLUMN}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Missing required columns in {path}")
        for row in reader:
            clean_id = row["clean_id"]
            angle = row[ANGLE_COLUMN]
            if angle not in seen[clean_id]:
                seen[clean_id].add(angle)
                angles[clean_id].append(angle)
    return dict(angles)


def load_clean_ids() -> list[str]:
    with TRUTH_IDS_PATH.open(encoding="utf-8") as input_file:
        records = json.load(input_file)
    clean_ids = [str(record["id"]) for record in records]
    if len(clean_ids) != len(set(clean_ids)):
        raise ValueError("Clean IDs are not unique")
    return clean_ids


def write_comparison(output_path: Path) -> int:
    clean_ids = load_clean_ids()
    lewis_angles = load_unique_angles(LEWIS_CANDIDATES_PATH)
    kanik_angles = load_unique_angles(KANIK_CANDIDATES_PATH)
    expected_ids = set(clean_ids)
    if set(lewis_angles) != expected_ids:
        raise ValueError("Lewis candidate IDs do not exactly match the clean IDs")
    if set(kanik_angles) != expected_ids:
        raise ValueError("Kanik candidate IDs do not exactly match the clean IDs")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "clean_id",
        "lewis_angle_deg",
        "kanik_angle_deg",
        "absolute_angle_difference_deg",
    ]
    row_count = 0
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for clean_id in clean_ids:
            for lewis_angle in lewis_angles[clean_id]:
                for kanik_angle in kanik_angles[clean_id]:
                    difference = abs(
                        Decimal(lewis_angle) - Decimal(kanik_angle)
                    )
                    writer.writerow(
                        {
                            "clean_id": clean_id,
                            "lewis_angle_deg": lewis_angle,
                            "kanik_angle_deg": kanik_angle,
                            "absolute_angle_difference_deg": f"{difference:.6f}",
                        }
                    )
                    row_count += 1
    return row_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT_PATH})",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = args.output.resolve()
    row_count = write_comparison(output_path)
    print(f"Wrote {row_count} angle comparisons to {output_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
