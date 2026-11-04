"""PDF report generation for the ellipse incident-angle pipeline."""

from __future__ import annotations

import csv
import math
import random
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ellipse_fit import FitConfig, FitResult, QC_DEFINITIONS


PAGE_WIDTH = 1650
PAGE_HEIGHT = 1275
PAGE_DPI = 150.0
BACKGROUND = "#ffffff"
TEXT = "#17212b"
MUTED = "#52606b"
GRID = "#d8dee5"
BLUE = "#2468a2"
PURPLE = "#7952a8"
ORANGE = "#d97706"
GREEN = "#16827a"
RED = "#bd3f32"


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / name
    return ImageFont.truetype(str(path), size)


def _new_page(title: str, subtitle: str = "") -> tuple[Image.Image, ImageDraw.ImageDraw]:
    page = Image.new("RGB", (PAGE_WIDTH, PAGE_HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(page)
    draw.text((38, 25), title, font=_font(29, True), fill=TEXT)
    if subtitle:
        draw.text((38, 69), subtitle, font=_font(16), fill=MUTED)
    return page, draw


def _save_pages(pages: Sequence[Image.Image], path: Path) -> None:
    if not pages:
        raise ValueError("Cannot write an empty PDF")
    path.parent.mkdir(parents=True, exist_ok=True)
    pages[0].save(
        path,
        "PDF",
        save_all=True,
        append_images=list(pages[1:]),
        resolution=PAGE_DPI,
        quality=92,
    )


def _draw_invalid_hatch(image: np.ndarray, valid: np.ndarray) -> np.ndarray:
    visible = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    invalid_y, invalid_x = np.nonzero(~valid)
    visible[invalid_y, invalid_x] = (235, 220, 229)
    for y, x in zip(invalid_y, invalid_x):
        if (x + y) % 5 == 0:
            visible[y, x] = (190, 125, 155)
    return visible


def _draw_axis(
    image: np.ndarray,
    center: tuple[float, float],
    length: float,
    angle_deg: float,
    color: tuple[int, int, int],
    width: int = 1,
) -> None:
    angle = math.radians(angle_deg)
    delta_x = length * math.cos(angle) / 2.0
    delta_y = length * math.sin(angle) / 2.0
    start = (round(center[0] - delta_x), round(center[1] - delta_y))
    end = (round(center[0] + delta_x), round(center[1] + delta_y))
    cv2.line(image, start, end, color, width, cv2.LINE_AA)


def render_fit_crop(
    result: FitResult,
    output_size: int,
    overlay: bool,
) -> Image.Image:
    visible = _draw_invalid_hatch(result.crop.image, result.crop.valid_pixels)
    if overlay:
        center = tuple(round(value) for value in result.crop.expected_center_xy)
        cv2.drawMarker(
            visible,
            center,
            (255, 40, 40),
            cv2.MARKER_CROSS,
            7,
            1,
            cv2.LINE_AA,
        )
        if result.contour is not None:
            cv2.drawContours(visible, [result.contour], -1, (255, 220, 0), 1, cv2.LINE_AA)
        if result.ellipse is not None:
            color = (0, 210, 70) if result.qc_passed else (255, 135, 0)
            cv2.ellipse(visible, result.ellipse, color, 1, cv2.LINE_AA)
            ellipse_center = result.ellipse[0]
            _draw_axis(
                visible,
                ellipse_center,
                float(result.major_axis_px),
                float(result.major_axis_orientation_deg),
                (255, 235, 0),
            )
            _draw_axis(
                visible,
                ellipse_center,
                float(result.minor_axis_px),
                (float(result.major_axis_orientation_deg) + 90.0) % 180.0,
                (0, 235, 255),
            )
    return Image.fromarray(visible).resize(
        (output_size, output_size), Image.Resampling.NEAREST
    )


def _failure_label(result: FitResult) -> str:
    return "+".join(result.qc_failure_codes) if result.qc_failure_codes else "PASS"


def _failure_counts(fits: Iterable[FitResult]) -> Counter[str]:
    return Counter(_failure_label(result) for result in fits if not result.qc_passed)


def _draw_histogram(
    draw: ImageDraw.ImageDraw,
    values: np.ndarray,
    bins: np.ndarray,
    box: tuple[int, int, int, int],
    color: str,
    x_label: str,
    y_label: str = "Percent of accepted fits",
    zero_line: bool = False,
) -> None:
    left, top, right, bottom = box
    counts, _ = np.histogram(values, bins=bins)
    percentages = 100.0 * counts / len(values) if len(values) else counts.astype(float)
    maximum = max(1.0, float(percentages.max(initial=0.0)) * 1.12)

    for step in range(6):
        fraction = step / 5.0
        y = bottom - fraction * (bottom - top)
        draw.line((left, y, right, y), fill=GRID, width=1)
        draw.text(
            (left - 62, y - 9),
            f"{maximum * fraction:.1f}",
            font=_font(13),
            fill=MUTED,
        )
    width = (right - left) / len(counts)
    for index, percent in enumerate(percentages):
        x0 = left + index * width + 1
        x1 = left + (index + 1) * width - 1
        y = bottom - percent / maximum * (bottom - top)
        draw.rectangle((x0, y, x1, bottom), fill=color)

    tick_every = max(1, math.ceil(len(counts) / 10))
    for index in range(0, len(counts) + 1, tick_every):
        x = left + index * width
        value = bins[min(index, len(bins) - 1)]
        draw.line((x, bottom, x, bottom + 8), fill=TEXT, width=2)
        draw.text((x - 18, bottom + 13), f"{value:g}", font=_font(13), fill=MUTED)
    draw.line((left, top, left, bottom), fill=TEXT, width=2)
    draw.line((left, bottom, right, bottom), fill=TEXT, width=2)
    if zero_line and bins[0] < 0 < bins[-1]:
        zero_x = left + (0.0 - bins[0]) / (bins[-1] - bins[0]) * (right - left)
        draw.line((zero_x, top, zero_x, bottom), fill=RED, width=3)
    draw.text(((left + right) // 2 - 155, bottom + 55), x_label, font=_font(17), fill=TEXT)
    draw.text((35, top - 42), y_label, font=_font(16), fill=TEXT)


def _draw_failure_bucket_page(
    surface: str,
    fits: Sequence[FitResult],
) -> Image.Image:
    excluded = [fit for fit in fits if not fit.qc_passed]
    counts = _failure_counts(fits)
    page, draw = _new_page(
        f"{surface} quality-control exclusions",
        "Each row is one exhaustive, mutually exclusive combination of failed cuts.",
    )
    draw.rounded_rectangle((45, 118, 1605, 1160), radius=18, fill="#f7f9fb", outline=GRID)
    draw.text((80, 148), "Failed cuts", font=_font(20, True), fill=TEXT)
    draw.text((650, 148), "Etch pits", font=_font(20, True), fill=TEXT)
    draw.text((920, 148), "Fraction of excluded pits", font=_font(20, True), fill=TEXT)
    y = 205
    for label, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        fraction = 100.0 * count / len(excluded) if excluded else 0.0
        draw.line((75, y - 14, 1570, y - 14), fill=GRID, width=1)
        draw.text((85, y), label, font=_font(20, True), fill=RED)
        draw.text((680, y), str(count), font=_font(20), fill=TEXT)
        draw.text((995, y), f"{fraction:.1f}%", font=_font(20), fill=TEXT)
        y += 68
    draw.text(
        (80, 1090),
        f"Accepted: {len(fits) - len(excluded)}    Excluded: {len(excluded)}    Total clean list: {len(fits)}",
        font=_font(19, True),
        fill=TEXT,
    )
    return page


def _draw_qc_page(config: FitConfig) -> Image.Image:
    page, draw = _new_page(
        "Incident-angle ellipse quality controls",
        "All cuts are applied to the one selected contour on each surface.",
    )
    details = {
        "A": (
            "Candidate available",
            f"At least {config.minimum_contour_points} contour points; area at least "
            f"{config.minimum_contour_area_px2:g} square pixels; minor axis at least "
            f"{config.minimum_minor_axis_px:g} pixels; major axis at most "
            f"{config.maximum_major_axis_px:g} pixels.",
        ),
        "B": (
            "Center offset",
            f"Distance from stored pit location is at most {config.maximum_center_offset_px:g} pixels.",
        ),
        "C": ("Solidity", f"Contour solidity is at least {config.minimum_solidity:.2f}."),
        "D": (
            "Ellipse residual",
            f"90th-percentile normalized residual is at most {config.maximum_residual_p90:.2f}.",
        ),
        "E": ("Analysis crop", "Contour does not touch the 49 by 49 crop boundary."),
        "F": (
            "Source image",
            "Contour does not touch a real source-image boundary. Pixels beyond the source image are masked, never reflected.",
        ),
    }
    y = 125
    for code in QC_DEFINITIONS:
        title, body = details[code]
        draw.rounded_rectangle((55, y, 1595, y + 145), radius=15, fill="#f7f9fb", outline=GRID)
        draw.ellipse((82, y + 37, 146, y + 101), fill=BLUE)
        draw.text((103, y + 49), code, font=_font(25, True), fill="white")
        draw.text((180, y + 25), title, font=_font(21, True), fill=TEXT)
        draw.text((180, y + 68), body, font=_font(17), fill=MUTED)
        y += 162
    return page


def _draw_random_fit_page(
    rows: Sequence[dict[str, object]],
    c1_fits: Sequence[FitResult],
    random_seed: int,
) -> Image.Image:
    page, draw = _new_page(
        "20 randomly selected clean C1 etch pits: ellipse fits",
        f"Seed {random_seed}. Green ellipse: accepted. Orange ellipse: rejected. Pink hatch: no source pixel.",
    )
    selected = random.Random(random_seed).sample(range(len(rows)), 20)
    columns = 5
    display = 218
    panel_width = 320
    panel_height = 286
    top = 108
    for number, index in enumerate(selected):
        panel_x = 30 + (number % columns) * panel_width
        panel_y = top + (number // columns) * panel_height
        fit = c1_fits[index]
        page.paste(render_fit_crop(fit, display, overlay=True), (panel_x, panel_y))
        draw.rectangle(
            (panel_x, panel_y, panel_x + display - 1, panel_y + display - 1),
            outline="#56616b",
        )
        status = "PASS" if fit.qc_passed else f"FAIL {_failure_label(fit)}"
        status_color = GREEN if fit.qc_passed else RED
        draw.text(
            (panel_x, panel_y + display + 5),
            f"{rows[index]['clean_id']}  {status}",
            font=_font(15, True),
            fill=status_color,
        )
        if fit.candidate_found:
            detail = (
                f"axes {fit.major_axis_px:.1f}, {fit.minor_axis_px:.1f} px; "
                f"angle {fit.incident_angle_deg:.1f} deg"
            )
        else:
            detail = "no valid ellipse candidate"
        draw.text((panel_x, panel_y + display + 28), detail, font=_font(12), fill=TEXT)
    return page


def _draw_surface_hist_page(
    surface: str,
    fits: Sequence[FitResult],
) -> Image.Image:
    accepted = np.asarray(
        [fit.incident_angle_deg for fit in fits if fit.qc_passed], dtype=float
    )
    excluded = len(fits) - len(accepted)
    page, draw = _new_page(
        f"{surface} incident-angle distribution",
        f"Clean list: {len(fits)}    Accepted: {len(accepted)}    Excluded by QC: {excluded}",
    )
    _draw_histogram(
        draw,
        accepted,
        np.arange(0.0, 76.0, 2.0),
        (120, 185, 1570, 1035),
        BLUE if surface == "C1" else GREEN,
        "Incident angle in degrees",
    )
    draw.text(
        (125, 1120),
        f"Mean: {accepted.mean():.3f} degrees    Median: {np.median(accepted):.3f} degrees",
        font=_font(20, True),
        fill=TEXT,
    )
    return page


def _draw_error_hist_page(
    errors: np.ndarray,
    absolute: bool,
) -> Image.Image:
    if absolute:
        title = "Absolute C1-to-C2 incident-angle error"
        values = np.abs(errors)
        bins = np.arange(0.0, 54.0, 2.0)
        x_label = "Absolute incident-angle error in degrees"
        color = ORANGE
    else:
        title = "Signed C1-to-C2 incident-angle error"
        values = errors
        bins = np.arange(-52.0, 54.0, 2.0)
        x_label = "C2 incident angle minus C1 incident angle, in degrees"
        color = PURPLE
    page, draw = _new_page(
        title,
        f"Includes {len(values)} clean IDs that passed all ellipse QC on both surfaces.",
    )
    _draw_histogram(
        draw,
        values,
        bins,
        (120, 185, 1570, 1035),
        color,
        x_label,
        y_label="Percent of paired accepted fits",
        zero_line=not absolute,
    )
    if absolute:
        stats = (
            f"Mean: {values.mean():.3f} degrees    Median: {np.median(values):.3f} degrees    "
            f"95th percentile: {np.percentile(values, 95):.3f} degrees"
        )
    else:
        stats = (
            f"Mean: {values.mean():.3f} degrees    Median: {np.median(values):.3f} degrees    "
            f"Standard deviation: {values.std(ddof=1):.3f} degrees"
        )
    draw.text((125, 1120), stats, font=_font(19, True), fill=TEXT)
    return page


def generate_main_report(
    output_path: Path,
    rows: Sequence[dict[str, object]],
    fits: dict[str, Sequence[FitResult]],
    config: FitConfig,
    random_seed: int,
) -> None:
    paired_errors = np.asarray(
        [
            c2.incident_angle_deg - c1.incident_angle_deg
            for c1, c2 in zip(fits["C1"], fits["C2"])
            if c1.qc_passed and c2.qc_passed
        ],
        dtype=float,
    )
    pages = [
        _draw_random_fit_page(rows, fits["C1"], random_seed),
        _draw_qc_page(config),
        _draw_failure_bucket_page("C1", fits["C1"]),
        _draw_surface_hist_page("C1", fits["C1"]),
        _draw_failure_bucket_page("C2", fits["C2"]),
        _draw_surface_hist_page("C2", fits["C2"]),
        _draw_error_hist_page(paired_errors, absolute=True),
        _draw_error_hist_page(paired_errors, absolute=False),
    ]
    _save_pages(pages, output_path)


def generate_failure_comparison_report(
    output_path: Path,
    index_path: Path,
    rows: Sequence[dict[str, object]],
    fits: dict[str, Sequence[FitResult]],
) -> None:
    cases: dict[str, list[tuple[int, FitResult]]] = {"C1": [], "C2": []}
    for surface in ("C1", "C2"):
        for index, fit in enumerate(fits[surface]):
            if "C" in fit.qc_failure_codes or "D" in fit.qc_failure_codes:
                cases[surface].append((index, fit))
        cases[surface].sort(
            key=lambda item: (
                0 if {"C", "D"}.issubset(item[1].qc_failure_codes) else 1,
                str(rows[item[0]]["clean_id"]),
            )
        )

    pages: list[Image.Image] = []
    index_rows: list[dict[str, object]] = []
    pdf_page = 0
    for surface in ("C1", "C2"):
        total_pages = math.ceil(len(cases[surface]) / 8)
        solidity_only = sum(
            "C" in fit.qc_failure_codes and "D" not in fit.qc_failure_codes
            for _, fit in cases[surface]
        )
        residual_only = sum(
            "D" in fit.qc_failure_codes and "C" not in fit.qc_failure_codes
            for _, fit in cases[surface]
        )
        both = sum(
            {"C", "D"}.issubset(fit.qc_failure_codes) for _, fit in cases[surface]
        )
        for local_page in range(total_pages):
            pdf_page += 1
            page, draw = _new_page(
                f"{surface} solidity or ellipse-residual failures — {local_page + 1} of {total_pages}",
                f"{len(cases[surface])} cases: {solidity_only} solidity only, {residual_only} residual only, {both} both. Left is raw; right adds boundary and fit.",
            )
            subset = cases[surface][local_page * 8 : (local_page + 1) * 8]
            for slot, (row_index, fit) in enumerate(subset):
                column = slot % 2
                row_number = slot // 2
                x0 = 28 + column * 807
                y0 = 102 + row_number * 291
                draw.rounded_rectangle(
                    (x0, y0, x0 + 780, y0 + 274),
                    radius=12,
                    fill="#fafbfc",
                    outline="#c8d0d8",
                    width=2,
                )
                display = 190
                raw = render_fit_crop(fit, display, overlay=False)
                overlay = render_fit_crop(fit, display, overlay=True)
                page.paste(raw, (x0 + 15, y0 + 45))
                page.paste(overlay, (x0 + 220, y0 + 45))
                draw.text((x0 + 82, y0 + 15), "RAW", font=_font(14, True), fill=MUTED)
                draw.text((x0 + 270, y0 + 15), "OVERLAY", font=_font(14, True), fill=MUTED)
                label = (
                    "BOTH" if {"C", "D"}.issubset(fit.qc_failure_codes)
                    else "SOLIDITY ONLY" if "C" in fit.qc_failure_codes
                    else "RESIDUAL ONLY"
                )
                text_x = x0 + 425
                draw.text(
                    (text_x, y0 + 24),
                    f"{rows[row_index]['clean_id']} | {surface}",
                    font=_font(19, True),
                    fill=TEXT,
                )
                draw.text((text_x, y0 + 61), label, font=_font(17, True), fill=RED)
                draw.text(
                    (text_x, y0 + 96),
                    f"Solidity: {fit.solidity:.3f}",
                    font=_font(15),
                    fill=TEXT,
                )
                draw.text(
                    (text_x, y0 + 124),
                    f"Residual: {fit.ellipse_residual_p90:.3f}",
                    font=_font(15),
                    fill=TEXT,
                )
                draw.text(
                    (text_x, y0 + 152),
                    f"Axes: {fit.major_axis_px:.1f}, {fit.minor_axis_px:.1f} px",
                    font=_font(14),
                    fill=TEXT,
                )
                other = [code for code in fit.qc_failure_codes if code not in {"C", "D"}]
                if other:
                    draw.text(
                        (text_x, y0 + 184),
                        f"Also failed: {'+'.join(other)}",
                        font=_font(14, True),
                        fill=RED,
                    )
                index_rows.append(
                    {
                        "pdf_page": pdf_page,
                        "slot_on_page": slot + 1,
                        "clean_id": rows[row_index]["clean_id"],
                        "surface": surface,
                        "failure_class": label,
                        "all_failed_qc_codes": _failure_label(fit),
                        "solidity": f"{fit.solidity:.9f}",
                        "ellipse_residual_p90": f"{fit.ellipse_residual_p90:.9f}",
                        "major_axis_px": f"{fit.major_axis_px:.6f}",
                        "minor_axis_px": f"{fit.minor_axis_px:.6f}",
                        "image_number": rows[row_index][f"{surface.lower()}_image"],
                        "local_x": rows[row_index][f"{surface.lower()}_local_x"],
                        "local_y": rows[row_index][f"{surface.lower()}_local_y"],
                    }
                )
            pages.append(page)
    _save_pages(pages, output_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(index_rows[0]))
        writer.writeheader()
        writer.writerows(index_rows)


def _draw_grouped_detection_histogram(
    draw: ImageDraw.ImageDraw,
    category_values: dict[str, np.ndarray],
    box: tuple[int, int, int, int],
) -> None:
    bins = np.arange(0.0, 76.0, 2.0)
    labels = [
        ("detected_angle_qc_passed", "Detected, QC passed", BLUE),
        ("missed_angle_qc_passed", "Missed, QC passed", PURPLE),
        ("detected_angle_qc_failed", "Detected, QC failed", ORANGE),
    ]
    histograms: list[np.ndarray] = []
    for key, _, _ in labels:
        counts, _ = np.histogram(category_values[key], bins=bins)
        values = category_values[key]
        histograms.append(100.0 * counts / len(values) if len(values) else counts.astype(float))
    maximum = max(1.0, max(float(hist.max(initial=0.0)) for hist in histograms) * 1.12)
    left, top, right, bottom = box
    for step in range(6):
        fraction = step / 5.0
        y = bottom - fraction * (bottom - top)
        draw.line((left, y, right, y), fill=GRID, width=1)
        draw.text((left - 62, y - 9), f"{maximum * fraction:.1f}", font=_font(13), fill=MUTED)
    group_width = (right - left) / len(histograms[0])
    bar_width = max(2.0, (group_width - 2.0) / len(labels))
    for category_index, ((_, _, color), histogram) in enumerate(zip(labels, histograms)):
        for bin_index, percent in enumerate(histogram):
            x0 = left + bin_index * group_width + 1 + category_index * bar_width
            x1 = x0 + bar_width
            y = bottom - percent / maximum * (bottom - top)
            draw.rectangle((x0, y, x1, bottom), fill=color)
    for index in range(0, len(histograms[0]) + 1, 4):
        x = left + index * group_width
        draw.line((x, bottom, x, bottom + 8), fill=TEXT, width=2)
        draw.text((x - 15, bottom + 13), f"{bins[index]:g}", font=_font(13), fill=MUTED)
    draw.line((left, top, left, bottom), fill=TEXT, width=2)
    draw.line((left, bottom, right, bottom), fill=TEXT, width=2)
    draw.text((620, bottom + 55), "C1 incident angle in degrees", font=_font(17), fill=TEXT)
    draw.text((35, top - 42), "Percent within each category", font=_font(16), fill=TEXT)
    legend_x = 915
    for offset, (_, label, color) in enumerate(labels):
        x = legend_x + (offset % 2) * 320
        y = 120 + (offset // 2) * 35
        draw.rectangle((x, y, x + 22, y + 17), fill=color)
        draw.text((x + 31, y - 3), label, font=_font(14), fill=TEXT)


def generate_detection_report(
    output_path: Path,
    pipeline_rows: Sequence[dict[str, object]],
) -> None:
    pages: list[Image.Image] = []
    for method in ("fcn", "dcc"):
        categories: dict[str, list[float]] = {
            "detected_angle_qc_passed": [],
            "missed_angle_qc_passed": [],
            "detected_angle_qc_failed": [],
            "missed_angle_qc_failed": [],
        }
        id_counts: Counter[str] = Counter()
        without_angle: Counter[str] = Counter()
        for row in pipeline_rows:
            category = str(row[f"{method}_category"])
            id_counts[category] += 1
            angle = row["c1_incident_angle_deg"]
            if angle in (None, ""):
                without_angle[category] += 1
            else:
                categories[category].append(float(angle))
        arrays = {key: np.asarray(value, dtype=float) for key, value in categories.items()}
        page, draw = _new_page(
            f"{method.upper()} detection outcomes by C1 incident angle",
            "Failed-QC angles are shown only when an ellipse candidate exists; they are diagnostic values, not accepted measurements.",
        )
        _draw_grouped_detection_histogram(draw, arrays, (120, 205, 1570, 965))
        y = 1050
        descriptions = [
            ("detected_angle_qc_passed", "Detected and QC passed"),
            ("missed_angle_qc_passed", "Missed and QC passed"),
            ("detected_angle_qc_failed", "Detected but QC failed"),
            ("missed_angle_qc_failed", "Missed and QC failed"),
        ]
        for number, (key, label) in enumerate(descriptions):
            values = arrays[key]
            mean = f"{values.mean():.2f} deg" if len(values) else "n/a"
            text = (
                f"{label}: {id_counts[key]} IDs; {len(values)} plotted; "
                f"{without_angle[key]} without candidate angle; mean {mean}"
            )
            draw.text((125 + (number % 2) * 760, y + (number // 2) * 58), text, font=_font(15), fill=TEXT)
        pages.append(page)
    _save_pages(pages, output_path)
