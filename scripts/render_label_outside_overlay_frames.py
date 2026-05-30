#!/usr/bin/env python3
"""Render label-outside bubble overlays for a frame range, mirroring the
case1_air1_publish_original_0304 enlarge style: black text placed below the
bbox using a serif font, with mask overlay + centroid + bbox rectangle.

Bubble IDs are read directly from each run's per-frame ID mask (pixel value
== object_id), so per-run colors and IDs are preserved.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bubble_tracking.frame_io import sorted_frame_paths
from bubble_tracking.visualization import draw_mask_overlay, generate_colors


DEFAULT_FONT = Path("/usr/share/fonts/opentype/urw-base35/NimbusRoman-Regular.otf")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render label-outside overlays for a frame range using run masks"
    )
    parser.add_argument("--run-dir", required=True, help="Run directory containing masks/")
    parser.add_argument("--frames", required=True, help="Source frames directory (relabeled)")
    parser.add_argument("--start", type=int, required=True, help="Start frame_idx (inclusive, 0-indexed)")
    parser.add_argument("--end", type=int, required=True, help="End frame_idx (inclusive, 0-indexed)")
    parser.add_argument("--output-dir", required=True, help="Output directory for annotated jpegs")
    parser.add_argument("--first-original-frame", type=int, default=71,
                        help="Original frame number of relabeled #1 (default 71 for case1_air1_publish)")
    parser.add_argument("--font", default=str(DEFAULT_FONT), help="Font file path")
    parser.add_argument("--font-size", type=int, default=60, help="Text font size")
    parser.add_argument("--x-offset", type=int, default=-70, help="Label x offset from bbox x_max")
    parser.add_argument("--y-gap", type=int, default=4, help="Vertical gap below bbox y_max")
    parser.add_argument("--alpha", type=float, default=0.4, help="Mask overlay alpha")
    parser.add_argument("--mask-only", action="store_true",
                        help="Draw only the mask overlay (no centroid, bbox, or ID label)")
    parser.add_argument("--no-labels", action="store_true", help="Skip ID text labels")
    parser.add_argument("--no-centroids", action="store_true", help="Skip centroid circles")
    parser.add_argument("--no-bboxes", action="store_true", help="Skip bounding box rectangles")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    return parser.parse_args()


def _render_frame(
    frame_bgr: np.ndarray,
    mask: np.ndarray,
    *,
    font: ImageFont.FreeTypeFont,
    x_offset: int,
    y_gap: int,
    alpha: float,
    mask_only: bool = False,
    draw_labels: bool = True,
    draw_centroids: bool = True,
    draw_bboxes: bool = True,
) -> np.ndarray:
    id_values = [int(v) for v in np.unique(mask) if int(v) > 0]
    colors = {obj_id: generate_colors(1, seed=obj_id * 7)[0] for obj_id in id_values}

    out = draw_mask_overlay(frame_bgr, mask.astype(np.int32), colors=colors, alpha=alpha)
    if mask_only:
        return out

    h, w = out.shape[:2]

    labels: list[tuple[int, int, int]] = []
    for obj_id in sorted(id_values):
        ys, xs = np.where(mask == obj_id)
        if xs.size == 0:
            continue

        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())
        cx, cy = int(round(float(xs.mean()))), int(round(float(ys.mean())))

        color = colors.get(obj_id, (255, 255, 255))
        if draw_centroids:
            cv2.circle(out, (cx, cy), 5, color, -1)
            cv2.circle(out, (cx, cy), 7, (255, 255, 255), 1)
        if draw_bboxes:
            cv2.rectangle(out, (x_min, y_min), (x_max, y_max), color, 2)
        labels.append((obj_id, x_max, y_max))

    if not draw_labels:
        return out

    pil_img = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    for obj_id, x_max, y_max in labels:
        label = f"#{obj_id}"
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        text_w = int(right - left)
        text_h = int(bottom - top)
        x_raw = int(x_max + x_offset)
        y_raw = int(y_max + y_gap)
        x = max(0, min(x_raw, w - text_w - 1))
        y = max(0, min(y_raw, h - text_h - 1))
        draw.text((x, y), label, font=font, fill=(0, 0, 0))

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def main() -> None:
    args = _parse_args()

    run_dir = Path(args.run_dir)
    masks_dir = run_dir / "masks"
    if not masks_dir.exists():
        raise FileNotFoundError(f"masks directory not found: {masks_dir}")

    frame_paths = sorted_frame_paths(str(Path(args.frames)))
    if not frame_paths:
        raise ValueError(f"No frames found in {args.frames}")
    if args.start < 0 or args.end < args.start or args.end >= len(frame_paths):
        raise ValueError(
            f"Invalid range [{args.start}, {args.end}] for {len(frame_paths)} frames"
        )

    font_path = Path(args.font)
    if not font_path.exists():
        raise FileNotFoundError(f"Font not found: {font_path}")
    font = ImageFont.truetype(str(font_path), int(args.font_size))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for frame_idx in range(args.start, args.end + 1):
        frame_path = frame_paths[frame_idx]
        frame_bgr = cv2.imread(frame_path, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            raise RuntimeError(f"Failed to read frame: {frame_path}")

        mask_path = masks_dir / f"{frame_idx:06d}.png"
        if not mask_path.exists():
            raise FileNotFoundError(f"mask not found: {mask_path}")
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None or mask.ndim != 2:
            raise RuntimeError(f"Bad mask: {mask_path}")

        out_bgr = _render_frame(
            frame_bgr,
            mask,
            font=font,
            x_offset=int(args.x_offset),
            y_gap=int(args.y_gap),
            alpha=float(args.alpha),
            mask_only=bool(args.mask_only),
            draw_labels=not bool(args.no_labels),
            draw_centroids=not bool(args.no_centroids),
            draw_bboxes=not bool(args.no_bboxes),
        )

        relabeled_num = frame_idx + 1
        original_num = relabeled_num + int(args.first_original_frame) - 1
        out_name = f"frame_original_{original_num:04d}_relabelled_{relabeled_num:04d}.jpg"
        out_path = output_dir / out_name
        if not cv2.imwrite(str(out_path), out_bgr,
                           [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)]):
            raise RuntimeError(f"Failed to write {out_path}")
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
