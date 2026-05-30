"""Sum contour lengths (outer vs inner/holes) across all binary masks in a directory.

Outputs a per-mask CSV and a totals-summary JSON.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

DEFAULT_MM_PER_PIXEL = 0.09107  # zero_degree: 100 mm = 2196 px (downscaled scale)


@dataclass
class MaskStats:
    filename: str
    num_outer: int
    num_inner: int
    outer_len_px: float
    inner_len_px: float

    @property
    def total_len_px(self) -> float:
        return self.outer_len_px + self.inner_len_px


def measure_mask(mask_path: Path) -> MaskStats:
    img = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read mask: {mask_path}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = (img > 0).astype(np.uint8) * 255

    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
    )

    outer_len = 0.0
    inner_len = 0.0
    n_outer = 0
    n_inner = 0

    if hierarchy is not None:
        # hierarchy shape: (1, N, 4) -> [next, prev, first_child, parent]
        h = hierarchy[0]
        for i, contour in enumerate(contours):
            length = float(cv2.arcLength(contour, closed=True))
            parent = int(h[i][3])
            if parent == -1:
                outer_len += length
                n_outer += 1
            else:
                inner_len += length
                n_inner += 1

    return MaskStats(
        filename=mask_path.name,
        num_outer=n_outer,
        num_inner=n_inner,
        outer_len_px=outer_len,
        inner_len_px=inner_len,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("outputs/zero_degree_publish_debug_no_morph_median_contrast_lightmap_normalize"),
        help="Directory containing mask PNGs.",
    )
    parser.add_argument(
        "--pattern",
        default="*_mask.png",
        help="Glob pattern for mask files (default: %(default)s).",
    )
    parser.add_argument(
        "--mm-per-pixel",
        type=float,
        default=DEFAULT_MM_PER_PIXEL,
        help=f"Pixel-to-mm scale (default: {DEFAULT_MM_PER_PIXEL} for zero_degree).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Per-mask CSV output path. Defaults to <input-dir>/contour_lengths_per_mask.csv",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Totals JSON output path. Defaults to <input-dir>/contour_lengths_summary.json",
    )
    args = parser.parse_args()

    input_dir: Path = args.input_dir
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")

    output_csv: Path = args.output_csv or (input_dir / "contour_lengths_per_mask.csv")
    output_json: Path = args.output_json or (input_dir / "contour_lengths_summary.json")
    mm_per_px: float = args.mm_per_pixel

    mask_paths = sorted(input_dir.glob(args.pattern))
    if not mask_paths:
        raise SystemExit(f"No masks matched {args.pattern} in {input_dir}")

    stats_list: list[MaskStats] = []
    for p in mask_paths:
        s = measure_mask(p)
        stats_list.append(s)
        print(
            f"{s.filename}: outer={s.num_outer} ({s.outer_len_px:.1f}px / "
            f"{s.outer_len_px * mm_per_px:.3f}mm), "
            f"inner={s.num_inner} ({s.inner_len_px:.1f}px / "
            f"{s.inner_len_px * mm_per_px:.3f}mm)"
        )

    total_outer_px = sum(s.outer_len_px for s in stats_list)
    total_inner_px = sum(s.inner_len_px for s in stats_list)
    total_px = total_outer_px + total_inner_px
    total_outer_mm = total_outer_px * mm_per_px
    total_inner_mm = total_inner_px * mm_per_px
    total_mm = total_px * mm_per_px

    with output_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "filename",
            "num_outer",
            "num_inner",
            "outer_len_px",
            "inner_len_px",
            "total_len_px",
            "outer_len_mm",
            "inner_len_mm",
            "total_len_mm",
        ])
        for s in stats_list:
            writer.writerow([
                s.filename,
                s.num_outer,
                s.num_inner,
                f"{s.outer_len_px:.4f}",
                f"{s.inner_len_px:.4f}",
                f"{s.total_len_px:.4f}",
                f"{s.outer_len_px * mm_per_px:.6f}",
                f"{s.inner_len_px * mm_per_px:.6f}",
                f"{s.total_len_px * mm_per_px:.6f}",
            ])

    summary = {
        "input_dir": str(input_dir),
        "pattern": args.pattern,
        "mm_per_pixel": mm_per_px,
        "num_masks": len(stats_list),
        "contour_extraction": {
            "retrieval_mode": "RETR_CCOMP",
            "approx_method": "CHAIN_APPROX_NONE",
            "outer_definition": "top-level contours (hierarchy parent == -1)",
            "inner_definition": "holes nested inside outer contours (parent != -1)",
        },
        "totals": {
            "outer_len_px": total_outer_px,
            "inner_len_px": total_inner_px,
            "total_len_px": total_px,
            "outer_len_mm": total_outer_mm,
            "inner_len_mm": total_inner_mm,
            "total_len_mm": total_mm,
        },
    }
    with output_json.open("w") as f:
        json.dump(summary, f, indent=2)

    print()
    print(f"Masks processed: {len(stats_list)}")
    print(
        f"Outer total: {total_outer_px:.1f} px / {total_outer_mm:.3f} mm"
    )
    print(
        f"Inner total: {total_inner_px:.1f} px / {total_inner_mm:.3f} mm"
    )
    print(f"Combined  : {total_px:.1f} px / {total_mm:.3f} mm")
    print(f"Per-mask CSV: {output_csv}")
    print(f"Summary JSON: {output_json}")


if __name__ == "__main__":
    main()
