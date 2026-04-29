#!/usr/bin/env python3
"""
Render a single annotated frame for case1 relabeled frame 0234 from a manually edited ID mask.

This preserves bubble numbering by using mask pixel values as object IDs.
Centroids and bounding boxes are recomputed from the edited mask, so changes to #49
or any other ID are reflected directly in the output.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bubble_tracking.visualization import draw_mask_overlay, generate_colors


DEFAULT_FRAME = Path("data/bubbles/case1_air1_publish/frames_0p5/000234.jpg")
DEFAULT_MASK = Path(
    "outputs/case1_air1_publish_original_0304_relabeled_0234_manual_change/000233_manual_mask.png"
)
DEFAULT_OUTPUT = Path(
    "outputs/case1_air1_publish_original_0304_relabeled_0234_manual_change/"
    "case1_air1_publish_original_0304_relabeled_0234_label_outside_black_text_scale_60_over_36.jpg"
)
DEFAULT_FONT = Path("/usr/share/fonts/opentype/urw-base35/NimbusRoman-Regular.otf")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render case1 relabeled frame 0234 from a manually edited ID mask"
    )
    parser.add_argument("--frame", default=str(DEFAULT_FRAME), help="Source frame image path")
    parser.add_argument("--mask", default=str(DEFAULT_MASK), help="Edited ID mask path (uint16 png)")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output annotated image path")
    parser.add_argument("--font", default=str(DEFAULT_FONT), help="Font file path for numbering text")
    parser.add_argument("--font-size", type=int, default=48, help="Text font size")
    parser.add_argument("--x-offset", type=int, default=-70, help="Label x offset from bbox x_max")
    parser.add_argument("--y-gap", type=int, default=4, help="Vertical gap below bbox y_max")
    parser.add_argument("--alpha", type=float, default=0.4, help="Mask overlay alpha")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    frame_path = Path(args.frame)
    mask_path = Path(args.mask)
    output_path = Path(args.output)
    font_path = Path(args.font)

    if not frame_path.exists():
        raise FileNotFoundError(f"Frame not found: {frame_path}")
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask not found: {mask_path}")
    if not font_path.exists():
        raise FileNotFoundError(f"Font not found: {font_path}")

    frame_bgr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    if frame_bgr is None:
        raise RuntimeError(f"Failed to read frame: {frame_path}")

    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise RuntimeError(f"Failed to read mask: {mask_path}")
    if mask.ndim != 2:
        raise ValueError(f"Mask must be single-channel, got shape={mask.shape}")

    id_values = [int(v) for v in np.unique(mask) if int(v) > 0]
    colors = {obj_id: generate_colors(1, seed=obj_id * 7)[0] for obj_id in id_values}

    # Draw instance mask overlay from edited mask.
    frame_bgr = draw_mask_overlay(frame_bgr, mask.astype(np.int32), colors=colors, alpha=float(args.alpha))
    h, w = frame_bgr.shape[:2]

    # Recompute centroid and bbox directly from edited mask for each ID.
    labels_for_text: list[tuple[int, int, int]] = []
    for obj_id in sorted(id_values):
        ys, xs = np.where(mask == obj_id)
        if xs.size == 0:
            continue

        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())
        cx, cy = int(round(float(xs.mean()))), int(round(float(ys.mean())))

        color = colors.get(obj_id, (255, 255, 255))
        cv2.circle(frame_bgr, (cx, cy), 5, color, -1)
        cv2.circle(frame_bgr, (cx, cy), 7, (255, 255, 255), 1)
        cv2.rectangle(frame_bgr, (x_min, y_min), (x_max, y_max), color, 2)
        labels_for_text.append((obj_id, x_max, y_max))

    # Draw numbering with the selected TTF/OTF font.
    pil_img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font = ImageFont.truetype(str(font_path), int(args.font_size))

    clamped = 0
    for obj_id, x_max, y_max in labels_for_text:
        label = f"#{obj_id}"
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        text_w = int(right - left)
        text_h = int(bottom - top)

        x_raw = int(x_max + int(args.x_offset))
        y_raw = int(y_max + int(args.y_gap))
        x = max(0, min(x_raw, w - text_w - 1))
        y = max(0, min(y_raw, h - text_h - 1))
        if x != x_raw or y != y_raw:
            clamped += 1

        draw.text((x, y), label, font=font, fill=(0, 0, 0))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(output_path), out_bgr):
        raise RuntimeError(f"Failed to write output image: {output_path}")

    print(f"Wrote: {output_path}")
    print(f"IDs rendered: {len(labels_for_text)}")
    print(f"clamped_labels={clamped}")


if __name__ == "__main__":
    main()
