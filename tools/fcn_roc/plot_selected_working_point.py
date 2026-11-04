#!/usr/bin/env python3
"""Render full-range publication curves from an existing scan; no inference."""
import argparse
from pathlib import Path
from fcn_roc_pipeline import make_plots, read_rows, write_json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-csv", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--config-id", default="7dc92f2228c54ad55267")
    args = parser.parse_args()
    rows = read_rows(args.results_csv)
    selected = next(row for row in rows if row["config_id"] == args.config_id)
    make_plots(args.output_root, rows, working_point=selected)
    write_json(args.output_root / "results" / "selected_working_point.json", {
        "source_csv": str(args.results_csv), "configurations": len(rows),
        "working_point": selected,
        "note": "ROC evaluator counts, before local analysis image-validity filtering.",
    })
    print(selected)
