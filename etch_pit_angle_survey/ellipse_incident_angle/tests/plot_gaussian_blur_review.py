#!/usr/bin/env python3
"""Render raw crops beside the valid-pixel normalized Gaussian output."""

from __future__ import annotations

import csv
import math
import random
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


PIPELINE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PIPELINE_DIR.parents[1]
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from ellipse_fit import (  # noqa: E402
    CropData,
    FitConfig,
    FitResult,
    _valid_normalized_gaussian_blur,
    extract_valid_crop,
    fit_etch_pit,
)


MANIFEST_PATH = PIPELINE_DIR / "inputs/clean_surface_locations.csv"
IMAGE_PATHS = {
    "C1": REPO_ROOT / "numpy_data_eval/preprocessed_clean_eval.npy",
    "C2": REPO_ROOT / "numpy_data_eval/clean_2_eval.npy",
}
FIRST_IMAGE_NUMBER = 251
RANDOM_SEED = 20261022
OUTPUT_PATH = Path(tempfile.gettempdir()) / "etch_pit_gaussian_blur_review.pdf"
PAGE_WIDTH = 1650
PAGE_HEIGHT = 1275
PAGE_DPI = 150.0


@dataclass
class GaussianCase:
    clean_id: str
    surface: str
    image_number: int
    local_x: float
    local_y: float
    crop: CropData
    blurred: np.ndarray
    crossed_edges: tuple[str, ...]
    fit: FitResult

    @property
    def is_full(self) -> bool:
        return bool(np.all(self.crop.valid_pixels))


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(
        str(Path("/usr/share/fonts/truetype/dejavu") / name), size
    )


def load_manifest() -> list[dict[str, str]]:
    with MANIFEST_PATH.open(newline="", encoding="utf-8") as input_file:
        return list(csv.DictReader(input_file))


def crossed_edges(
    center_x: int,
    center_y: int,
    width: int,
    height: int,
    radius: int,
) -> tuple[str, ...]:
    result: list[str] = []
    if center_x - radius < 0:
        result.append("left")
    if center_x + radius >= width:
        result.append("right")
    if center_y - radius < 0:
        result.append("top")
    if center_y + radius >= height:
        result.append("bottom")
    return tuple(result)


def make_case(
    row: dict[str, str],
    surface: str,
    images: np.ndarray,
    config: FitConfig,
) -> GaussianCase:
    key = surface.lower()
    image_number = int(float(row[f"{key}_image"]))
    source = images[image_number - FIRST_IMAGE_NUMBER]
    local_x = float(row[f"{key}_local_x"])
    local_y = float(row[f"{key}_local_y"])
    crop = extract_valid_crop(source, local_x, local_y, config)
    blurred = _valid_normalized_gaussian_blur(
        crop.image, crop.valid_pixels, config
    )
    edges = crossed_edges(
        int(round(local_x)),
        int(round(local_y)),
        source.shape[1],
        source.shape[0],
        config.crop_radius_px,
    )
    return GaussianCase(
        clean_id=row["clean_id"],
        surface=surface,
        image_number=image_number,
        local_x=local_x,
        local_y=local_y,
        crop=crop,
        blurred=blurred,
        crossed_edges=edges,
        fit=fit_etch_pit(source, local_x, local_y, config),
    )


def render_image(
    image: np.ndarray,
    valid_pixels: np.ndarray,
    expected_center_xy: tuple[float, float],
    output_size: int,
) -> Image.Image:
    visible = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    invalid_y, invalid_x = np.nonzero(~valid_pixels)
    visible[invalid_y, invalid_x] = (235, 220, 229)
    for y, x in zip(invalid_y, invalid_x):
        if (x + y) % 5 == 0:
            visible[y, x] = (190, 125, 155)
    center = tuple(round(value) for value in expected_center_xy)
    cv2.drawMarker(
        visible,
        center,
        (255, 30, 30),
        cv2.MARKER_CROSS,
        7,
        1,
        cv2.LINE_AA,
    )
    return Image.fromarray(visible).resize(
        (output_size, output_size), Image.Resampling.NEAREST
    )


def render_raw(case: GaussianCase, output_size: int) -> Image.Image:
    return render_image(
        case.crop.image,
        case.crop.valid_pixels,
        case.crop.expected_center_xy,
        output_size,
    )


def render_blurred(case: GaussianCase, output_size: int) -> Image.Image:
    return render_image(
        case.blurred,
        case.crop.valid_pixels,
        case.crop.expected_center_xy,
        output_size,
    )


def new_page(title: str, subtitle: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    page = Image.new("RGB", (PAGE_WIDTH, PAGE_HEIGHT), "white")
    draw = ImageDraw.Draw(page)
    draw.text((35, 24), title, font=font(29, True), fill="#17212b")
    draw.text((35, 68), subtitle, font=font(15), fill="#52606b")
    return page, draw


def qc_label(case: GaussianCase) -> tuple[str, str]:
    if case.fit.qc_passed:
        return "QC PASS", "#16827a"
    return f"QC FAIL {'+'.join(case.fit.qc_failure_codes)}", "#bd3f32"


def full_surface_page(surface: str, cases: list[GaussianCase]) -> Image.Image:
    page, draw = new_page(
        f"Gaussian blur: 10 random full {surface} crops",
        (
            f"Left: raw crop. Right: 5 by 5 normalized Gaussian blur with sigma 1.1 pixels. "
            f"Seed {RANDOM_SEED}. Red cross: stored location."
        ),
    )
    panel_width = 320
    panel_height = 545
    display = 138
    for index, case in enumerate(cases):
        x0 = 25 + (index % 5) * panel_width
        y0 = 118 + (index // 5) * panel_height
        draw.rounded_rectangle(
            (x0, y0, x0 + 302, y0 + 505),
            radius=9,
            fill="#fafbfc",
            outline="#c8d0d8",
        )
        draw.text((x0 + 10, y0 + 10), case.clean_id, font=font(17, True), fill="#17212b")
        draw.text((x0 + 10, y0 + 39), "RAW", font=font(12, True), fill="#52606b")
        draw.text((x0 + 157, y0 + 39), "GAUSSIAN", font=font(12, True), fill="#52606b")
        page.paste(render_raw(case, display), (x0 + 10, y0 + 60))
        page.paste(render_blurred(case, display), (x0 + 157, y0 + 60))
        label, color = qc_label(case)
        draw.text((x0 + 10, y0 + 210), label, font=font(13, True), fill=color)
        draw.text(
            (x0 + 10, y0 + 237),
            f"image {case.image_number}; local ({case.local_x:g}, {case.local_y:g})",
            font=font(11),
            fill="#52606b",
        )
    return page


def partner_status(case: GaussianCase, selected_for_partial: bool) -> str:
    if selected_for_partial:
        return f"PARTIAL: {','.join(case.crossed_edges)}"
    return "FULL PARTNER"


def partial_combo_pages(
    selected_ids: list[str],
    cases_by_id: dict[str, dict[str, GaussianCase]],
    partial_surfaces_by_id: dict[str, set[str]],
) -> list[Image.Image]:
    cases_per_page = 6
    page_count = math.ceil(len(selected_ids) / cases_per_page)
    pages: list[Image.Image] = []
    for page_index in range(page_count):
        page, draw = new_page(
            f"Gaussian blur: 30 random IDs containing partial crops — page {page_index + 1} of {page_count}",
            (
                f"Raw and Gaussian output are shown for both partners. Seed {RANDOM_SEED}. "
                "Pink hatch: unavailable pixels. FULL PARTNER means that surface has a complete 49 by 49 crop."
            ),
        )
        subset = selected_ids[
            page_index * cases_per_page : (page_index + 1) * cases_per_page
        ]
        for slot, clean_id in enumerate(subset):
            column = slot % 2
            row_number = slot // 2
            x0 = 25 + column * 805
            y0 = 108 + row_number * 382
            draw.rounded_rectangle(
                (x0, y0, x0 + 780, y0 + 360),
                radius=10,
                fill="#fafbfc",
                outline="#c8d0d8",
                width=2,
            )
            draw.text((x0 + 14, y0 + 10), clean_id, font=font(18, True), fill="#17212b")
            for surface_index, surface in enumerate(("C1", "C2")):
                case = cases_by_id[clean_id][surface]
                surface_x = x0 + 14 + surface_index * 382
                image_y = y0 + 66
                display = 164
                draw.text((surface_x, y0 + 37), surface, font=font(14, True), fill="#17212b")
                status = partner_status(
                    case, surface in partial_surfaces_by_id[clean_id]
                )
                status_color = "#bd3f32" if status.startswith("PARTIAL") else "#2468a2"
                draw.text((surface_x + 32, y0 + 37), status, font=font(12, True), fill=status_color)
                draw.text((surface_x, y0 + 55), "RAW", font=font(10, True), fill="#52606b")
                draw.text((surface_x + 174, y0 + 55), "GAUSSIAN", font=font(10, True), fill="#52606b")
                page.paste(render_raw(case, display), (surface_x, image_y))
                page.paste(render_blurred(case, display), (surface_x + 174, image_y))
                label, color = qc_label(case)
                draw.text((surface_x, image_y + 172), label, font=font(12, True), fill=color)
                draw.text(
                    (surface_x, image_y + 194),
                    f"img {case.image_number}; ({case.local_x:g},{case.local_y:g})",
                    font=font(10),
                    fill="#52606b",
                )
        pages.append(page)
    return pages


def main() -> int:
    rows = load_manifest()
    rows_by_id = {row["clean_id"]: row for row in rows}
    config = FitConfig()
    images = {
        surface: np.load(path, mmap_mode="r")
        for surface, path in IMAGE_PATHS.items()
    }

    partial_surfaces_by_id: dict[str, set[str]] = {}
    full_rows: dict[str, list[dict[str, str]]] = {"C1": [], "C2": []}
    for row in rows:
        for surface in ("C1", "C2"):
            key = surface.lower()
            image_number = int(float(row[f"{key}_image"]))
            source = images[surface][image_number - FIRST_IMAGE_NUMBER]
            edges = crossed_edges(
                int(round(float(row[f"{key}_local_x"]))),
                int(round(float(row[f"{key}_local_y"]))),
                source.shape[1],
                source.shape[0],
                config.crop_radius_px,
            )
            if edges:
                partial_surfaces_by_id.setdefault(row["clean_id"], set()).add(surface)
            else:
                full_rows[surface].append(row)

    generator = random.Random(RANDOM_SEED)
    selected_full_rows = {
        "C1": generator.sample(full_rows["C1"], 10),
        "C2": generator.sample(full_rows["C2"], 10),
    }
    partial_ids = sorted(partial_surfaces_by_id)
    selected_partial_ids = generator.sample(partial_ids, 30)

    selected_full_cases = {
        surface: [
            make_case(row, surface, images[surface], config)
            for row in selected_full_rows[surface]
        ]
        for surface in ("C1", "C2")
    }
    cases_by_id: dict[str, dict[str, GaussianCase]] = {}
    for clean_id in selected_partial_ids:
        row = rows_by_id[clean_id]
        cases_by_id[clean_id] = {
            surface: make_case(row, surface, images[surface], config)
            for surface in ("C1", "C2")
        }

    pages = [
        full_surface_page("C1", selected_full_cases["C1"]),
        full_surface_page("C2", selected_full_cases["C2"]),
    ]
    pages.extend(
        partial_combo_pages(
            selected_partial_ids,
            cases_by_id,
            partial_surfaces_by_id,
        )
    )
    pages[0].save(
        OUTPUT_PATH,
        "PDF",
        save_all=True,
        append_images=pages[1:],
        resolution=PAGE_DPI,
        quality=92,
    )

    selected_with_full_partner = sum(
        len(partial_surfaces_by_id[clean_id]) == 1
        for clean_id in selected_partial_ids
    )
    print("Full C1 examples: 10")
    print("Full C2 examples: 10")
    print("Random partial-ID combinations: 30")
    print(f"Selected combinations with one full partner: {selected_with_full_partner}")
    print(f"PDF pages: {len(pages)}")
    print(f"PDF: {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
