#!/usr/bin/env python3
"""Targeted follow-up sampling for sparse FCN ROC/FROC frontier regions."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--target-run", type=Path, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args()


def rounded(value: float) -> float:
    return float(f"{value:.12g}")


def selected_pairs() -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    low_pairs: set[tuple[float, float]] = set()
    for alpha in (0.000375, 0.0005, 0.000625, 0.00075):
        for beta_micro in (1.25, 2.5, 5.0):
            low_pairs.add((alpha, beta_micro * 1e-6))
    for alpha in (0.001, 0.00125, 0.0015, 0.00175):
        for beta_micro in (5.0, 15.0, 30.0, 35.0):
            low_pairs.add((alpha, beta_micro * 1e-6))
    for alpha in (0.002, 0.0025, 0.003, 0.00325, 0.0035):
        for beta_micro in (20.0, 30.0, 35.0, 40.0):
            low_pairs.add((alpha, beta_micro * 1e-6))

    high_pairs: set[tuple[float, float]] = set()
    for alpha in (0.002, 0.0025, 0.00275):
        for beta_micro in (20.0, 25.0, 30.0, 35.0, 40.0, 50.0):
            high_pairs.add((alpha, beta_micro * 1e-6))
    for alpha in (0.003, 0.003125, 0.00325, 0.0035, 0.004):
        for beta_micro in (25.0, 30.0, 32.5, 35.0, 40.0, 50.0):
            high_pairs.add((alpha, beta_micro * 1e-6))

    normalize = lambda values: sorted((rounded(a), rounded(b)) for a, b in values)
    return normalize(low_pairs), normalize(high_pairs)


def planned_configurations() -> tuple[list[tuple[float, float, float, float]], dict]:
    low_thresholds = [
        0.0005,
        0.001,
        0.0015,
        0.002,
        0.0025,
        0.003,
        0.004,
        0.005,
        0.00625,
        0.0075,
        0.00875,
        0.01,
        0.0125,
        0.015,
        0.0175,
        0.02,
        0.025,
        0.03,
        0.04,
        0.05,
        0.0625,
        0.075,
        0.1,
        0.125,
        0.15,
        0.175,
        0.2,
    ]
    high_front_thresholds = [
        0.02,
        0.025,
        0.03,
        0.035,
        0.04,
        0.045,
        0.05,
        0.06,
        0.075,
        0.1,
        0.125,
        0.15,
        0.175,
        0.2,
        0.225,
        0.25,
        0.275,
        0.3,
        0.325,
        0.35,
        0.375,
        0.4,
        0.425,
        0.45,
        0.475,
        0.5,
        0.525,
    ]
    high_back_thresholds = [
        0.03,
        0.04,
        0.05,
        0.06,
        0.075,
        0.1,
        0.125,
        0.15,
        0.175,
        0.2,
        0.225,
        0.25,
        0.275,
        0.3,
        0.35,
        0.4,
        0.45,
        0.5,
        0.55,
        0.575,
        0.6,
        0.625,
        0.65,
    ]
    low_pairs, high_pairs = selected_pairs()
    low_family = {
        (front, back, alpha, beta)
        for front in low_thresholds
        for back in low_thresholds
        for alpha, beta in low_pairs
    }
    high_family = {
        (front, back, alpha, beta)
        for front in high_front_thresholds
        for back in high_back_thresholds
        for alpha, beta in high_pairs
    }
    configurations = sorted(low_family | high_family)
    plan = {
        "purpose": "Targeted densification of all frontier regions identified in the 2026-11-04 audit",
        "families": {
            "low_fp_and_mid_recall": {
                "plot_regions": {
                    "froc_fp_per_mm2": [0.0, 3.58],
                    "pr_recall": [0.1688, 0.929],
                },
                "front_thresholds": low_thresholds,
                "back_thresholds": low_thresholds,
                "alpha_beta_pairs": [list(pair) for pair in low_pairs],
                "raw_configurations": len(low_family),
            },
            "high_recall_cliff": {
                "plot_regions": {
                    "froc_fp_per_mm2": [7.82, 138.36],
                    "pr_recall": [0.95049, 0.95292],
                },
                "front_thresholds": high_front_thresholds,
                "back_thresholds": high_back_thresholds,
                "alpha_beta_pairs": [list(pair) for pair in high_pairs],
                "raw_configurations": len(high_family),
                "extends_alpha_to": 0.004,
                "extends_beta_to": 50e-6,
            },
        },
        "raw_unique_configurations": len(configurations),
    }
    return configurations, plan


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    base_run = args.base_run.resolve()
    target_run = args.target_run.resolve()
    if base_run == target_run:
        raise ValueError("The targeted run must be isolated from the base run")

    sys.path.insert(0, str(target_run / "src"))
    import fcn_roc_pipeline as pipeline

    target_run.mkdir(parents=True, exist_ok=True)
    results_dir = target_run / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    configurations, plan = planned_configurations()
    base_rows = pipeline.read_rows(base_run / "results" / "all_results.csv")
    existing_signatures = {
        tuple(
            rounded(row[field])
            for field in ("front_threshold", "back_threshold", "alpha", "beta")
        )
        for row in base_rows
    }
    configurations = [
        values
        for values in configurations
        if tuple(rounded(value) for value in values) not in existing_signatures
    ]
    plan.update(
        {
            "base_run": str(base_run),
            "target_run": str(target_run),
            "workers": args.workers,
            "logical_cpus_on_host": os.cpu_count(),
            "maximum_cpu_fraction": args.workers / os.cpu_count(),
            "gpu_required": False,
            "existing_configurations_excluded": plan["raw_unique_configurations"] - len(configurations),
            "new_configurations_planned": len(configurations),
        }
    )
    pipeline.write_json(target_run / "targeted_scan_plan.json", plan)
    with (target_run / "planned_configurations.csv").open("w", encoding="utf-8") as stream:
        stream.write("front_threshold,back_threshold,alpha,beta,config_id\n")
        for values in configurations:
            stream.write(",".join([*(f"{value:.12g}" for value in values), pipeline.config_id(values)]) + "\n")
    print(json.dumps(plan, indent=2), flush=True)
    if args.plan_only:
        return

    started = time.time()
    area_mm2 = float(
        json.loads((base_run / "results" / "summary.json").read_text(encoding="utf-8"))[
            "effective_analyzed_area_mm2"
        ]
    )
    pipeline.ensure_coordinate_requests(target_run, configurations, args.workers)

    chunk_size = max(1, math.ceil(len(configurations) / args.workers))
    chunks = [
        configurations[index : index + chunk_size]
        for index in range(0, len(configurations), chunk_size)
    ]
    tasks = [
        {
            "run_root": str(target_run),
            "configurations": chunk,
            "round": 100,
            "part": part,
            "area_mm2": area_mm2,
        }
        for part, chunk in enumerate(chunks)
    ]
    pipeline.run_parallel(tasks, pipeline.evaluate_refined_chunk, args.workers, "targeted evaluation")
    targeted_rows = pipeline.combine_refinement_rows(target_run, 100)
    by_id = {row["config_id"]: row for row in base_rows}
    by_id.update({row["config_id"]: row for row in targeted_rows})
    combined_rows = sorted(by_id.values(), key=lambda row: row["config_id"])
    pipeline.write_rows(results_dir / "targeted_results.csv", targeted_rows)
    pipeline.write_rows(results_dir / "combined_results.csv", combined_rows)
    pipeline.make_plots(target_run, combined_rows)
    optimal = max(
        combined_rows,
        key=lambda row: (
            row["f1"],
            -row["false_positives"],
            row["true_positives"],
            tuple(-ord(char) for char in row["config_id"]),
        ),
    )
    pipeline.save_optimal_details(target_run, optimal)
    summary = {
        "status": "complete",
        "base_run": str(base_run),
        "targeted_configurations": len(targeted_rows),
        "combined_configurations": len(combined_rows),
        "workers": args.workers,
        "maximum_cpu_fraction": args.workers / os.cpu_count(),
        "elapsed_seconds": time.time() - started,
        "optimal_working_point": optimal,
        "frontier_gaps": pipeline.frontier_gaps(combined_rows, area_mm2),
    }
    pipeline.write_json(results_dir / "summary.json", summary)
    checksums = []
    for path in sorted(results_dir.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            checksums.append(f"{pipeline.sha256_file(path)}  {path.relative_to(target_run)}")
    (results_dir / "SHA256SUMS").write_text("\n".join(checksums) + "\n", encoding="utf-8")
    print("TARGETED REFINEMENT COMPLETE", json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
