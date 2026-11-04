#!/usr/bin/env python3
"""Generate a complete nominal-QC diagnostic PDF for one fit configuration."""

from __future__ import annotations

import argparse
import csv
import math
import random
import sys
import tempfile
import textwrap
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image, ImageDraw


TEST_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = TEST_DIR.parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from ellipse_fit import FitConfig, FitResult, QC_DEFINITIONS  # noqa: E402
from reporting import (  # noqa: E402
    GREEN,
    GRID,
    MUTED,
    PAGE_DPI,
    PAGE_HEIGHT,
    PAGE_WIDTH,
    RED,
    TEXT,
    _draw_error_hist_page,
    _draw_failure_bucket_page,
    _draw_qc_page,
    _draw_surface_hist_page,
    _draw_invalid_hatch,
    _font,
    _new_page,
)
from run_pipeline import (  # noqa: E402
    C1_IMAGE_PATH,
    C2_IMAGE_PATH,
    DEFAULT_INPUT_PATH,
    load_manifest,
    run_fits,
    validate_manifest_truth,
)


DEFAULT_OUTPUT = (
    Path(tempfile.gettempdir())
    / "ellipse_incident_angle_k3_sigma0p9_complete_qc_report.pdf"
)
DEFAULT_INDEX_OUTPUT = (
    Path(tempfile.gettempdir())
    / "ellipse_incident_angle_k3_sigma0p9_failure_index.csv"
)
RANDOM_SEED = 20261022


class StreamingPdf:
    """Write PIL pages one at a time without retaining the full report in memory."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.pdf = PdfPages(
            path,
            metadata={
                "Title": "Ellipse incident-angle QC report: kernel 3, sigma 0.9",
                "Author": "MoEDAL ellipse incident-angle survey",
            },
        )
        self.page_number = 0

    def add(self, page: Image.Image) -> int:
        self.page_number += 1
        draw = ImageDraw.Draw(page)
        label = f"Page {self.page_number}"
        box = draw.textbbox((0, 0), label, font=_font(12))
        width = box[2] - box[0]
        draw.text(
            (PAGE_WIDTH - width - 24, PAGE_HEIGHT - 27),
            label,
            font=_font(12),
            fill=MUTED,
        )
        figure = plt.figure(
            figsize=(PAGE_WIDTH / PAGE_DPI, PAGE_HEIGHT / PAGE_DPI),
            dpi=PAGE_DPI,
            frameon=False,
        )
        axis = figure.add_axes((0, 0, 1, 1))
        axis.imshow(page)
        axis.axis("off")
        self.pdf.savefig(figure, dpi=PAGE_DPI, bbox_inches=None, pad_inches=0)
        plt.close(figure)
        return self.page_number

    def close(self) -> None:
        self.pdf.close()


def _failure_label(fit: FitResult) -> str:
    return "+".join(fit.qc_failure_codes)


def _optional_number(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _render_boundary_and_ellipse(
    fit: FitResult,
    output_size: int,
    overlay: bool,
) -> Image.Image:
    """Render raw data or only the selected contour and fitted ellipse."""

    visible = _draw_invalid_hatch(fit.crop.image, fit.crop.valid_pixels)
    if overlay:
        if fit.contour is not None:
            cv2.drawContours(visible, [fit.contour], -1, (255, 220, 0), 1, cv2.LINE_AA)
        if fit.ellipse is not None:
            color = (0, 210, 70) if fit.qc_passed else (255, 135, 0)
            cv2.ellipse(visible, fit.ellipse, color, 1, cv2.LINE_AA)
    return Image.fromarray(visible).resize(
        (output_size, output_size), Image.Resampling.NEAREST
    )


def _title_page(
    config: FitConfig,
    fits: dict[str, Sequence[FitResult]],
    paired_nominal_count: int,
    robust_paired_count: int,
) -> Image.Image:
    page, draw = _new_page(
        "Ellipse incident-angle QC review",
        "Complete nominal-QC audit for Gaussian kernel 3 pixels and sigma 0.9 pixels.",
    )
    draw.rounded_rectangle((55, 125, 1595, 480), radius=18, fill="#f7f9fb", outline=GRID)
    lines = [
        f"Clean IDs: {len(fits['C1'])}",
        f"C1 nominal QC: {sum(f.qc_passed for f in fits['C1'])} pass, {sum(not f.qc_passed for f in fits['C1'])} fail",
        f"C2 nominal QC: {sum(f.qc_passed for f in fits['C2'])} pass, {sum(not f.qc_passed for f in fits['C2'])} fail",
        f"Both surfaces pass nominal QC: {paired_nominal_count}",
        f"Both surfaces pass nominal QC and perturbation-majority gate: {robust_paired_count}",
    ]
    y = 160
    for line in lines:
        draw.text((90, y), line, font=_font(24, True), fill=TEXT)
        y += 58

    draw.rounded_rectangle((55, 530, 1595, 1135), radius=18, fill="#fbfcfd", outline=GRID)
    body = [
        "Report contents",
        "1. A reproducible sample of 20 clean IDs that pass nominal QC on both surfaces, with raw and overlaid crops.",
        "2. C1 and C2 incident-angle distributions using every nominally accepted surface fit.",
        "3. Absolute and signed C2-minus-C1 error distributions using every nominal paired pass.",
        "4. Exhaustive, mutually exclusive failure-bucket tables for C1 and C2.",
        "5. Raw and overlaid crops for every failed surface fit, grouped by its complete A-through-F failure combination.",
        "",
        "Nominal QC means cuts A through F on the selected contour. The perturbation-majority gate belongs to the loss evaluator and is reported above for reference; it does not create an A-through-F failure code.",
    ]
    y = 570
    for index, paragraph in enumerate(body):
        font = _font(23, True) if index == 0 else _font(18)
        for line in textwrap.wrap(paragraph, width=132) if paragraph else [""]:
            draw.text((90, y), line, font=font, fill=TEXT if index == 0 else MUTED)
            y += 33
        y += 10
    return page


def _accepted_gallery_page(
    rows: Sequence[dict[str, object]],
    fits: dict[str, Sequence[FitResult]],
    indices: Sequence[int],
    local_page: int,
    total_pages: int,
) -> Image.Image:
    page, draw = _new_page(
        f"Paired nominal-QC passes — sample page {local_page} of {total_pages}",
        "Each row is one clean ID. Within each surface, left is raw and right adds the selected boundary and ellipse.",
    )
    headers = [(115, "C1 RAW"), (300, "C1 OVERLAY"), (690, "C2 RAW"), (875, "C2 OVERLAY")]
    for x, label in headers:
        draw.text((x, 95), label, font=_font(14, True), fill=MUTED)
    display = 170
    for slot, row_index in enumerate(indices):
        y0 = 125 + slot * 222
        draw.rounded_rectangle((40, y0 - 8, 1605, y0 + 202), radius=11, fill="#fafbfc", outline=GRID)
        positions = [(85, "C1", False), (270, "C1", True), (660, "C2", False), (845, "C2", True)]
        for x0, surface, overlay in positions:
            page.paste(
                _render_boundary_and_ellipse(
                    fits[surface][row_index], display, overlay
                ),
                (x0, y0 + 8),
            )
        row = rows[row_index]
        c1 = fits["C1"][row_index]
        c2 = fits["C2"][row_index]
        draw.text((1050, y0 + 18), str(row["clean_id"]), font=_font(22, True), fill=GREEN)
        draw.text((1050, y0 + 58), f"C1 angle: {c1.incident_angle_deg:.3f} degrees", font=_font(17), fill=TEXT)
        draw.text((1050, y0 + 92), f"C2 angle: {c2.incident_angle_deg:.3f} degrees", font=_font(17), fill=TEXT)
        signed = float(c2.incident_angle_deg - c1.incident_angle_deg)
        draw.text((1050, y0 + 126), f"Signed error: {signed:+.3f} degrees", font=_font(17), fill=TEXT)
        draw.text((1050, y0 + 160), f"Absolute error: {abs(signed):.3f} degrees", font=_font(17), fill=TEXT)
    return page


def _failed_gallery_page(
    surface: str,
    bucket: str,
    cases: Sequence[tuple[int, FitResult]],
    rows: Sequence[dict[str, object]],
    local_page: int,
    total_pages: int,
) -> Image.Image:
    page, draw = _new_page(
        f"{surface} QC failures — bucket {bucket} — page {local_page} of {total_pages}",
        f"Exact failed-cut combination: {bucket}. Each failed {surface} fit appears in exactly one bucket. Left is raw; right adds boundary and ellipse.",
    )
    for slot, (row_index, fit) in enumerate(cases):
        column = slot % 2
        row_number = slot // 2
        x0 = 28 + column * 807
        y0 = 102 + row_number * 291
        draw.rounded_rectangle(
            (x0, y0, x0 + 780, y0 + 274),
            radius=12,
            fill="#fafbfc",
            outline=GRID,
            width=2,
        )
        display = 190
        page.paste(
            _render_boundary_and_ellipse(fit, display, overlay=False),
            (x0 + 15, y0 + 45),
        )
        page.paste(
            _render_boundary_and_ellipse(fit, display, overlay=True),
            (x0 + 220, y0 + 45),
        )
        draw.text((x0 + 82, y0 + 15), "RAW", font=_font(14, True), fill=MUTED)
        draw.text((x0 + 270, y0 + 15), "OVERLAY", font=_font(14, True), fill=MUTED)
        text_x = x0 + 425
        draw.text((text_x, y0 + 20), f"{rows[row_index]['clean_id']} | {surface}", font=_font(19, True), fill=TEXT)
        draw.text((text_x, y0 + 54), f"FAIL {bucket}", font=_font(17, True), fill=RED)
        if fit.candidate_found:
            details = [
                f"Angle: {_optional_number(fit.incident_angle_deg)} degrees",
                f"Center offset: {_optional_number(fit.center_offset_px)} px",
                f"Solidity: {_optional_number(fit.solidity)}",
                f"Residual P90: {_optional_number(fit.ellipse_residual_p90)}",
                f"Axes: {_optional_number(fit.major_axis_px, 1)}, {_optional_number(fit.minor_axis_px, 1)} px",
            ]
        else:
            details = ["No valid ellipse candidate"]
            details.extend(textwrap.wrap(fit.candidate_failure_detail, width=42)[:4])
        for line_number, line in enumerate(details):
            draw.text((text_x, y0 + 90 + 29 * line_number), line, font=_font(14), fill=TEXT)
    return page


def _write_failure_index(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--failure-index-output", type=Path, default=DEFAULT_INDEX_OUTPUT)
    parser.add_argument("--gaussian-kernel-px", type=int, default=3)
    parser.add_argument("--gaussian-sigma-px", type=float, default=0.9)
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--accepted-sample-size", type=int, default=20)
    parser.add_argument(
        "--robust-paired-count",
        type=int,
        default=883,
        help="Reference count from the matching loss evaluation.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = FitConfig(
        gaussian_kernel_px=args.gaussian_kernel_px,
        gaussian_sigma_px=args.gaussian_sigma_px,
    )
    rows = load_manifest(args.input.resolve())
    validate_manifest_truth(rows)
    images = {
        "C1": np.load(C1_IMAGE_PATH, mmap_mode="r"),
        "C2": np.load(C2_IMAGE_PATH, mmap_mode="r"),
    }
    fits = run_fits(rows, images, config)
    paired_indices = [
        index
        for index, (c1, c2) in enumerate(zip(fits["C1"], fits["C2"]))
        if c1.qc_passed and c2.qc_passed
    ]
    sample_size = min(args.accepted_sample_size, len(paired_indices))
    accepted_sample = random.Random(args.random_seed).sample(paired_indices, sample_size)
    paired_errors = np.asarray(
        [
            fits["C2"][index].incident_angle_deg - fits["C1"][index].incident_angle_deg
            for index in paired_indices
        ],
        dtype=float,
    )

    grouped: dict[str, dict[str, list[tuple[int, FitResult]]]] = {
        "C1": defaultdict(list),
        "C2": defaultdict(list),
    }
    for surface in ("C1", "C2"):
        for index, fit in enumerate(fits[surface]):
            if not fit.qc_passed:
                grouped[surface][_failure_label(fit)].append((index, fit))
        for cases in grouped[surface].values():
            cases.sort(key=lambda item: str(rows[item[0]]["clean_id"]))

        failed_indices = [index for bucket in grouped[surface].values() for index, _ in bucket]
        expected_indices = [index for index, fit in enumerate(fits[surface]) if not fit.qc_passed]
        if len(failed_indices) != len(set(failed_indices)) or sorted(failed_indices) != expected_indices:
            raise RuntimeError(f"{surface} failure buckets are not exhaustive and mutually exclusive")

    writer = StreamingPdf(args.output.resolve())
    failure_index: list[dict[str, object]] = []
    try:
        writer.add(
            _title_page(
                config,
                fits,
                paired_nominal_count=len(paired_indices),
                robust_paired_count=args.robust_paired_count,
            )
        )
        writer.add(_draw_qc_page(config))

        accepted_pages = math.ceil(len(accepted_sample) / 5)
        for page_index in range(accepted_pages):
            subset = accepted_sample[page_index * 5 : (page_index + 1) * 5]
            writer.add(
                _accepted_gallery_page(
                    rows, fits, subset, page_index + 1, accepted_pages
                )
            )

        writer.add(_draw_surface_hist_page("C1", fits["C1"]))
        writer.add(_draw_surface_hist_page("C2", fits["C2"]))
        writer.add(_draw_error_hist_page(paired_errors, absolute=True))
        writer.add(_draw_error_hist_page(paired_errors, absolute=False))
        writer.add(_draw_failure_bucket_page("C1", fits["C1"]))
        writer.add(_draw_failure_bucket_page("C2", fits["C2"]))

        for surface in ("C1", "C2"):
            ordered_buckets = sorted(
                grouped[surface],
                key=lambda bucket: (-len(grouped[surface][bucket]), bucket),
            )
            for bucket in ordered_buckets:
                cases = grouped[surface][bucket]
                bucket_pages = math.ceil(len(cases) / 8)
                for page_index in range(bucket_pages):
                    subset = cases[page_index * 8 : (page_index + 1) * 8]
                    pdf_page = writer.add(
                        _failed_gallery_page(
                            surface,
                            bucket,
                            subset,
                            rows,
                            page_index + 1,
                            bucket_pages,
                        )
                    )
                    for slot, (row_index, fit) in enumerate(subset, start=1):
                        key = surface.lower()
                        failure_index.append(
                            {
                                "pdf_page": pdf_page,
                                "slot_on_page": slot,
                                "clean_id": rows[row_index]["clean_id"],
                                "surface": surface,
                                "failure_bucket": bucket,
                                "image_number": rows[row_index][f"{key}_image"],
                                "local_x": rows[row_index][f"{key}_local_x"],
                                "local_y": rows[row_index][f"{key}_local_y"],
                                "candidate_found": fit.candidate_found,
                                "candidate_failure_detail": fit.candidate_failure_detail,
                            }
                        )
    finally:
        writer.close()

    if len(failure_index) != sum(not fit.qc_passed for surface in fits.values() for fit in surface):
        raise RuntimeError("Failure index does not contain every failed surface fit")
    _write_failure_index(args.failure_index_output.resolve(), failure_index)
    print(f"PDF pages: {writer.page_number}")
    print(f"Paired nominal passes: {len(paired_indices)}")
    print(f"Failed surface fits plotted: {len(failure_index)}")
    print(f"PDF: {args.output.resolve()}")
    print(f"Failure index: {args.failure_index_output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
