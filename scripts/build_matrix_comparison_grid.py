#!/usr/bin/env python3
"""Build 9 side-by-side comparison plots (one per frame in original 299-307),
each with 12 matrix-run overlays + 1 CV-keyframe-segmentation benchmark.

Layout: 4 rows x 4 cols.
  Row 0: benchmark in col 0, three blank cells.
  Rows 1-3: matrix cells.
    Cols within rows 1-3: bp_no, bp_yes, l_no, l_yes  (left to right).
    Row 1: min_area=428.   Row 2: min_area=68.   Row 3: min_area=27.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path("/home/aaron/Downloads/bubble_tracking")
MATRIX_OVERLAYS = ROOT / "outputs/matrix_overlays"
BENCHMARK_DIR = ROOT / "outputs/keyframe_segmentation_sweep_299_307_offset6/original_299-307/minarea_27"
OUTPUT_DIR = ROOT / "outputs/matrix_comparison_grid"
FONT_PATH = Path("/usr/share/fonts/opentype/urw-base35/NimbusRoman-Regular.otf")


COLS = ["bp_no", "bp_yes", "l_no", "l_yes"]
COL_LABELS = ["b+ / no extras", "b+ / +extras 299-307", "large / no extras", "large / +extras 299-307"]
ROWS_MIN = [428, 68, 27]


def fname_for(orig: int, rel: int) -> str:
    return f"frame_original_{orig:04d}_relabelled_{rel:04d}.jpg"


def load_resized(path: Path, target_w: int) -> Image.Image:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    h, w = img.shape[:2]
    scale = target_w / w
    new = cv2.resize(img, (target_w, int(round(h * scale))), interpolation=cv2.INTER_AREA)
    return Image.fromarray(cv2.cvtColor(new, cv2.COLOR_BGR2RGB))


def build_one(orig: int, rel: int, panel_w: int = 760, label_h: int = 36, gap: int = 12,
              title_h: int = 70) -> Image.Image:
    """Compose a 4x4 grid (with one slot for benchmark; remainder blank)."""
    # Load all panels first to determine panel height.
    bench_path = BENCHMARK_DIR / fname_for(orig, rel)
    matrix_paths: dict[tuple[int, str], Path] = {}
    for ma in ROWS_MIN:
        for col_key in COLS:
            run_name = f"matrix_{col_key}_min{ma}_off6"
            matrix_paths[(ma, col_key)] = MATRIX_OVERLAYS / run_name / fname_for(orig, rel)

    # Load the bench image to get panel height.
    bench_img = load_resized(bench_path, panel_w)
    panel_h = bench_img.height
    # Sanity: matrix images should have same dimensions as source frame, so same panel_h.
    sample = load_resized(next(iter(matrix_paths.values())), panel_w)
    if sample.height != panel_h:
        panel_h = max(panel_h, sample.height)

    n_cols, n_rows = 4, 4
    grid_w = n_cols * panel_w + (n_cols - 1) * gap
    grid_h = title_h + n_rows * (panel_h + label_h) + (n_rows - 1) * gap

    canvas = Image.new("RGB", (grid_w, grid_h), color=(245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    title_font = ImageFont.truetype(str(FONT_PATH), 36)
    label_font = ImageFont.truetype(str(FONT_PATH), 22)

    # Title
    title = f"Original frame {orig:04d}  (relabelled {rel:04d})    benchmark = CV keyframe seg, min_area=27, threshold_offset=6"
    draw.text((20, 18), title, fill=(0, 0, 0), font=title_font)

    def paste(img: Image.Image, col: int, row: int, label: str) -> None:
        x = col * (panel_w + gap)
        y = title_h + row * (panel_h + label_h + gap)
        # Label
        draw.text((x + 6, y + 4), label, fill=(0, 0, 0), font=label_font)
        canvas.paste(img, (x, y + label_h))

    # Row 0, col 0: benchmark
    paste(bench_img, col=0, row=0, label="benchmark (CV keyframe min_area=27)")

    # Rows 1-3: matrix cells
    for r_idx, ma in enumerate(ROWS_MIN, start=1):
        for c_idx, (col_key, col_label) in enumerate(zip(COLS, COL_LABELS)):
            p = matrix_paths[(ma, col_key)]
            img = load_resized(p, panel_w)
            paste(img, col=c_idx, row=r_idx, label=f"{col_label}  |  min_area={ma}")

    return canvas


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for frame_idx in range(228, 237):
        rel = frame_idx + 1
        orig = rel + 70
        composite = build_one(orig, rel)
        out_path = OUTPUT_DIR / f"comparison_original_{orig:04d}_relabelled_{rel:04d}.jpg"
        composite.save(out_path, quality=88)
        print(f"wrote {out_path}  ({composite.size[0]}x{composite.size[1]})")


if __name__ == "__main__":
    main()
