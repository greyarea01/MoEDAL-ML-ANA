#!/usr/bin/env python3
"""Render full and partial crop examples for the crop-extraction audit."""

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

from ellipse_fit import CropData, FitConfig, FitResult, extract_valid_crop, fit_etch_pit  # noqa: E402


MANIFEST_PATH = PIPELINE_DIR / "inputs/clean_surface_locations.csv"
IMAGE_PATHS = {
    "C1": REPO_ROOT / "numpy_data_eval/preprocessed_clean_eval.npy",
    "C2": REPO_ROOT / "numpy_data_eval/clean_2_eval.npy",
}
FIRST_IMAGE_NUMBER = 251
RANDOM_SEED = 20261022
OUTPUT_PATH = Path(tempfile.gettempdir()) / "etch_pit_crop_stage_review.pdf"
PAGE_WIDTH = 1650
PAGE_HEIGHT = 1275
PAGE_DPI = 150.0


@dataclass
class CropCase:
    clean_id: str
    surface: str
    image_number: int
    local_x: float
    local_y: float
    crop: CropData
    crossed_edges: tuple[str, ...]
    fit: FitResult | None = None

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


def edge_names(
    center_x: int,
    center_y: int,
    width: int,
    height: int,
    radius: int,
) -> tuple[str, ...]:
    edges: list[str] = []
    if center_x - radius < 0:
        edges.append("left")
    if center_x + radius >= width:
        edges.append("right")
    if center_y - radius < 0:
        edges.append("top")
    if center_y + radius >= height:
        edges.append("bottom")
    return tuple(edges)


def collect_cases(
    rows: list[dict[str, str]],
    config: FitConfig,
) -> dict[str, list[CropCase]]:
    output: dict[str, list[CropCase]] = {"C1": [], "C2": []}
    for surface in ("C1", "C2"):
        images = np.load(IMAGE_PATHS[surface], mmap_mode="r")
        key = surface.lower()
        for row in rows:
            image_number = int(float(row[f"{key}_image"]))
            source = images[image_number - FIRST_IMAGE_NUMBER]
            local_x = float(row[f"{key}_local_x"])
            local_y = float(row[f"{key}_local_y"])
            rounded_x = int(round(local_x))
            rounded_y = int(round(local_y))
            edges = edge_names(
                rounded_x,
                rounded_y,
                source.shape[1],
                source.shape[0],
                config.crop_radius_px,
            )
            crop = extract_valid_crop(source, local_x, local_y, config)
            fit = None
            if edges:
                fit = fit_etch_pit(source, local_x, local_y, config)
            output[surface].append(
                CropCase(
                    clean_id=row["clean_id"],
                    surface=surface,
                    image_number=image_number,
                    local_x=local_x,
                    local_y=local_y,
                    crop=crop,
                    crossed_edges=edges,
                    fit=fit,
                )
            )
    return output


def render_crop(crop: CropData, output_size: int) -> Image.Image:
    visible = cv2.cvtColor(crop.image, cv2.COLOR_GRAY2RGB)
    invalid_y, invalid_x = np.nonzero(~crop.valid_pixels)
    visible[invalid_y, invalid_x] = (235, 220, 229)
    for y, x in zip(invalid_y, invalid_x):
        if (x + y) % 5 == 0:
            visible[y, x] = (190, 125, 155)
    center = tuple(round(value) for value in crop.expected_center_xy)
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


def new_page(title: str, subtitle: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    page = Image.new("RGB", (PAGE_WIDTH, PAGE_HEIGHT), "white")
    draw = ImageDraw.Draw(page)
    draw.text((35, 24), title, font=font(29, True), fill="#17212b")
    draw.text((35, 68), subtitle, font=font(15), fill="#52606b")
    return page, draw


def full_crop_page(surface: str, cases: list[CropCase]) -> Image.Image:
    page, draw = new_page(
        f"10 randomly selected full {surface} crops",
        f"Crop radius 24 pixels; 49 by 49 real source pixels. Seed {RANDOM_SEED}. Red cross: stored pit location.",
    )
    display = 224
    panel_width = 320
    panel_height = 540
    for index, case in enumerate(cases):
        x = 30 + (index % 5) * panel_width
        y = 130 + (index // 5) * panel_height
        page.paste(render_crop(case.crop, display), (x, y))
        draw.rectangle((x, y, x + display - 1, y + display - 1), outline="#52606b")
        draw.text((x, y + display + 9), case.clean_id, font=font(17, True), fill="#17212b")
        draw.text(
            (x, y + display + 37),
            f"image {case.image_number}; local ({case.local_x:g}, {case.local_y:g})",
            font=font(13),
            fill="#52606b",
        )
    return page


def qc_label(case: CropCase) -> tuple[str, str]:
    if case.fit is None:
        return "", "#52606b"
    if case.fit.qc_passed:
        return "QC PASS", "#16827a"
    failures = "+".join(case.fit.qc_failure_codes)
    return f"QC FAIL {failures}", "#bd3f32"


def partial_crop_pages(
    partial_by_id: dict[str, dict[str, CropCase]],
    c1_passed: int,
    c2_passed: int,
    shared_count: int,
) -> list[Image.Image]:
    clean_ids = sorted(partial_by_id)
    cases_per_page = 12
    page_count = math.ceil(len(clean_ids) / cases_per_page)
    pages: list[Image.Image] = []
    for page_index in range(page_count):
        page, draw = new_page(
            f"All partial source-image crops — page {page_index + 1} of {page_count}",
            (
                f"C1: 224 partial, {c1_passed} pass all QC. C2: 226 partial, {c2_passed} pass all QC. "
                f"{shared_count} IDs are partial on both. Pink hatch: unavailable source pixels."
            ),
        )
        subset = clean_ids[
            page_index * cases_per_page : (page_index + 1) * cases_per_page
        ]
        for slot, clean_id in enumerate(subset):
            column = slot % 4
            row_number = slot // 4
            x0 = 25 + column * 405
            y0 = 112 + row_number * 382
            draw.rounded_rectangle(
                (x0, y0, x0 + 385, y0 + 360),
                radius=10,
                fill="#fafbfc",
                outline="#c8d0d8",
                width=2,
            )
            draw.text((x0 + 14, y0 + 12), clean_id, font=font(18, True), fill="#17212b")
            for surface_index, surface in enumerate(("C1", "C2")):
                image_x = x0 + 14 + surface_index * 185
                image_y = y0 + 48
                case = partial_by_id[clean_id].get(surface)
                draw.text((image_x, y0 + 33), surface, font=font(13, True), fill="#52606b")
                if case is None:
                    draw.rounded_rectangle(
                        (image_x, image_y, image_x + 165, image_y + 165),
                        radius=6,
                        fill="#eef1f4",
                        outline="#d8dee5",
                    )
                    draw.text(
                        (image_x + 37, image_y + 72),
                        "full crop",
                        font=font(14),
                        fill="#7a8791",
                    )
                    continue
                page.paste(render_crop(case.crop, 165), (image_x, image_y))
                draw.rectangle(
                    (image_x, image_y, image_x + 164, image_y + 164),
                    outline="#52606b",
                )
                label, color = qc_label(case)
                draw.text((image_x, image_y + 173), label, font=font(12, True), fill=color)
                draw.text(
                    (image_x, image_y + 196),
                    f"edge: {','.join(case.crossed_edges)}",
                    font=font(11),
                    fill="#52606b",
                )
                draw.text(
                    (image_x, image_y + 216),
                    f"img {case.image_number}; ({case.local_x:g},{case.local_y:g})",
                    font=font(10),
                    fill="#52606b",
                )
        pages.append(page)
    return pages


def main() -> int:
    config = FitConfig()
    rows = load_manifest()
    cases = collect_cases(rows, config)
    full = {
        surface: [case for case in cases[surface] if case.is_full]
        for surface in ("C1", "C2")
    }
    partial = {
        surface: [case for case in cases[surface] if not case.is_full]
        for surface in ("C1", "C2")
    }
    if len(partial["C1"]) != 224 or len(partial["C2"]) != 226:
        raise AssertionError(
            f"Unexpected partial counts: C1 {len(partial['C1'])}, C2 {len(partial['C2'])}"
        )

    generator = random.Random(RANDOM_SEED)
    selected_full = {
        "C1": generator.sample(full["C1"], 10),
        "C2": generator.sample(full["C2"], 10),
    }
    partial_by_id: dict[str, dict[str, CropCase]] = {}
    for surface in ("C1", "C2"):
        for case in partial[surface]:
            partial_by_id.setdefault(case.clean_id, {})[surface] = case
    shared_count = sum(len(surface_cases) == 2 for surface_cases in partial_by_id.values())
    c1_passed = sum(bool(case.fit and case.fit.qc_passed) for case in partial["C1"])
    c2_passed = sum(bool(case.fit and case.fit.qc_passed) for case in partial["C2"])
    shared_both_passed = sum(
        len(surface_cases) == 2
        and all(
            case.fit is not None and case.fit.qc_passed
            for case in surface_cases.values()
        )
        for surface_cases in partial_by_id.values()
    )
    ids_with_at_least_one_pass = sum(
        any(
            case.fit is not None and case.fit.qc_passed
            for case in surface_cases.values()
        )
        for surface_cases in partial_by_id.values()
    )

    pages = [
        full_crop_page("C1", selected_full["C1"]),
        full_crop_page("C2", selected_full["C2"]),
    ]
    pages.extend(
        partial_crop_pages(
            partial_by_id,
            c1_passed,
            c2_passed,
            shared_count,
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

    print(f"C1 full crops: {len(full['C1'])}; random examples: 10")
    print(f"C2 full crops: {len(full['C2'])}; random examples: 10")
    print(f"C1 partial crops: {len(partial['C1'])}; all-QC passes: {c1_passed}")
    print(f"C2 partial crops: {len(partial['C2'])}; all-QC passes: {c2_passed}")
    print(f"Unique partial IDs: {len(partial_by_id)}; partial on both: {shared_count}")
    print(f"Shared partial IDs passing QC on both surfaces: {shared_both_passed}")
    print(
        f"Partial IDs with at least one partial surface passing QC: "
        f"{ids_with_at_least_one_pass}"
    )
    print(f"PDF pages: {len(pages)}")
    print(f"PDF: {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
