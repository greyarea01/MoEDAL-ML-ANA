#!/usr/bin/env python3
"""Write Lewis/Kanik angle errors only for IDs with multiple candidates."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from decimal import Decimal
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_PATH = SCRIPT_DIR / "data/lewis_kanik_angle_comparison.csv"
DEFAULT_OUTPUT_PATH = SCRIPT_DIR / "data/multiple_candidate_angle_errors.csv"


def append_unique(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)


def extract_multiple_candidate_errors(input_path: Path, output_path: Path) -> tuple[int, int]:
    rows_by_id: dict[str, list[dict[str, str]]] = defaultdict(list)
    lewis_angles: dict[str, list[str]] = defaultdict(list)
    kanik_angles: dict[str, list[str]] = defaultdict(list)

    with input_path.open(encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        required = {
            "clean_id",
            "lewis_angle_deg",
            "kanik_angle_deg",
            "absolute_angle_difference_deg",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Missing required columns in {input_path}")
        for row in reader:
            clean_id = row["clean_id"]
            rows_by_id[clean_id].append(row)
            append_unique(lewis_angles[clean_id], row["lewis_angle_deg"])
            append_unique(kanik_angles[clean_id], row["kanik_angle_deg"])

    multiple_candidate_ids = {
        clean_id
        for clean_id in rows_by_id
        if len(lewis_angles[clean_id]) > 1 or len(kanik_angles[clean_id]) > 1
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "clean_id",
        "clean_id_frequency",
        "candidate_pair_number",
        "candidate_pair_count",
        "lewis_candidate_number",
        "lewis_candidate_count",
        "lewis_angle_deg",
        "kanik_candidate_number",
        "kanik_candidate_count",
        "kanik_angle_deg",
        "absolute_angle_error_deg",
    ]
    output_row_count = 0
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for clean_id, rows in rows_by_id.items():
            if clean_id not in multiple_candidate_ids:
                continue
            lewis_numbers = {
                angle: index for index, angle in enumerate(lewis_angles[clean_id], start=1)
            }
            kanik_numbers = {
                angle: index for index, angle in enumerate(kanik_angles[clean_id], start=1)
            }
            pair_count = len(rows)
            for pair_number, row in enumerate(rows, start=1):
                lewis_angle = row["lewis_angle_deg"]
                kanik_angle = row["kanik_angle_deg"]
                absolute_error = abs(Decimal(lewis_angle) - Decimal(kanik_angle))
                source_error = Decimal(row["absolute_angle_difference_deg"])
                if absolute_error != source_error:
                    raise ValueError(
                        f"Stored angle difference is inconsistent for {clean_id}"
                    )
                writer.writerow(
                    {
                        "clean_id": clean_id,
                        "clean_id_frequency": pair_count,
                        "candidate_pair_number": pair_number,
                        "candidate_pair_count": pair_count,
                        "lewis_candidate_number": lewis_numbers[lewis_angle],
                        "lewis_candidate_count": len(lewis_numbers),
                        "lewis_angle_deg": lewis_angle,
                        "kanik_candidate_number": kanik_numbers[kanik_angle],
                        "kanik_candidate_count": len(kanik_numbers),
                        "kanik_angle_deg": kanik_angle,
                        "absolute_angle_error_deg": f"{absolute_error:.6f}",
                    }
                )
                output_row_count += 1

    return len(multiple_candidate_ids), output_row_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Input angle comparison CSV (default: {DEFAULT_INPUT_PATH})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT_PATH})",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    id_count, row_count = extract_multiple_candidate_errors(
        args.input.resolve(), args.output.resolve()
    )
    print(
        f"Wrote {row_count} candidate-pair errors for {id_count} clean IDs "
        f"to {args.output.resolve()}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
