#!/usr/bin/env python3
"""
Generate prompts from SAM2 automatic mask generator on a single frame (within ROI).

Example:
  /home/aaron/Downloads/bubble_tracking/venv311/bin/python scripts/auto_mask_prompts.py \
      --frames data/interim/bubbles_air1_072_271/frames \
      --frame-idx 0 \
      --roi-file data/interim/bubbles_air1_072_271/roi.json \
      --checkpoint checkpoints/sam2_hiera_tiny.pt \
      --model-cfg sam2_hiera_t.yaml \
      --output data/interim/bubbles_air1_072_271/prompts_auto_sam2.yaml
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def load_frame(frames_dir: str, frame_idx: int):
    from bubble_tracking.frame_io import sorted_frame_paths

    frame_paths = sorted_frame_paths(frames_dir)
    if not frame_paths:
        raise ValueError(f"No frames found in {frames_dir}")
    if frame_idx < 0 or frame_idx >= len(frame_paths):
        raise IndexError(f"frame_idx {frame_idx} out of range (0..{len(frame_paths)-1})")
    frame_path = frame_paths[frame_idx]
    frame = cv2.imread(frame_path)
    if frame is None:
        raise ValueError(f"Cannot read frame: {frame_path}")
    return frame, frame_path


def main():
    parser = argparse.ArgumentParser(description="Auto-mask prompts using SAM2 on a frame")
    parser.add_argument("--frames", required=True, help="Directory of frames (numeric names)")
    parser.add_argument("--frame-idx", type=int, default=0, help="Frame index in sorted order")
    parser.add_argument("--roi-file", type=str, default=None, help="ROI JSON with x1,y1,x2,y2")
    parser.add_argument("--roi-mask", type=str, default=None, help="ROI mask PNG (nonzero=keep)")
    parser.add_argument("--checkpoint", required=True, help="SAM2 checkpoint path")
    parser.add_argument("--model-cfg", default="sam2_hiera_t.yaml", help="SAM2 model cfg name")
    parser.add_argument("--output", required=True, help="Output YAML path for prompts")
    parser.add_argument("--stats-out", default=None, help="Optional JSON path for mask stats")
    parser.add_argument("--debug-dir", default=None, help="Optional directory for debug images")
    parser.add_argument("--points-per-side", type=int, default=32, help="Point grid density")
    parser.add_argument("--min-area", type=int, default=20, help="Min mask area to keep")
    parser.add_argument("--max-area", type=int, default=None, help="Max mask area to keep")
    parser.add_argument(
        "--max-area-ratio",
        type=float,
        default=None,
        help="Max mask area as fraction of ROI area (0-1)",
    )
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    args = parser.parse_args()

    frame, frame_path = load_frame(args.frames, args.frame_idx)
    h, w = frame.shape[:2]

    roi_mask = None
    if args.roi_mask:
        roi_mask = cv2.imread(args.roi_mask, cv2.IMREAD_GRAYSCALE)
        if roi_mask is None:
            raise ValueError(f"Failed to read roi mask: {args.roi_mask}")
        if roi_mask.shape[:2] != frame.shape[:2]:
            raise ValueError("ROI mask size does not match frame size")
        ys, xs = np.where(roi_mask > 0)
        if ys.size == 0:
            raise ValueError("ROI mask is empty")
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        x1, x2 = int(xs.min()), int(xs.max()) + 1
    elif args.roi_file:
        roi = json.loads(Path(args.roi_file).read_text())
        x1, y1, x2, y2 = int(roi["x1"]), int(roi["y1"]), int(roi["x2"]), int(roi["y2"])
    else:
        x1, y1, x2, y2 = 0, 0, w, h

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        raise ValueError("ROI crop is empty; check roi.json coordinates.")

    if roi_mask is not None:
        crop_mask = roi_mask[y1:y2, x1:x2]
        crop = crop.copy()
        crop[crop_mask == 0] = 255

    # Convert BGR->RGB for SAM2
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    import torch
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    model = build_sam2(args.model_cfg, args.checkpoint, device=args.device)
    mask_generator = SAM2AutomaticMaskGenerator(
        model,
        points_per_side=args.points_per_side,
        output_mode="binary_mask",
    )

    with torch.inference_mode(), torch.autocast(args.device, dtype=torch.bfloat16):
        masks = mask_generator.generate(crop_rgb)

    # Build prompts
    from bubble_tracking.schemas import PromptSpec, ObjectPrompt, PointPrompt

    prompts = []
    stats = []
    obj_id = 1
    if roi_mask is not None:
        roi_area = int((roi_mask > 0).sum())
    else:
        roi_area = (x2 - x1) * (y2 - y1)
    for m in masks:
        mask = m["segmentation"]
        area = int(mask.sum())
        if area < args.min_area:
            continue
        if args.max_area is not None and area > args.max_area:
            continue
        if args.max_area_ratio is not None and roi_area > 0:
            if area / roi_area > args.max_area_ratio:
                continue
        ys, xs = np.where(mask)
        if ys.size == 0:
            continue
        cy = int(np.mean(ys)) + y1
        cx = int(np.mean(xs)) + x1
        bbox = m.get("bbox")
        if bbox is not None:
            bx, by, bw, bh = bbox
            bbox_abs = [int(bx + x1), int(by + y1), int(bx + bw + x1), int(by + bh + y1)]
        else:
            bbox_abs = None
        prompts.append(ObjectPrompt(
            object_id=obj_id,
            frame_idx=args.frame_idx,
            points=[PointPrompt(x=cx, y=cy, label=1)],
        ))
        stats.append({
            "object_id": obj_id,
            "area": area,
            "centroid": [cx, cy],
            "bbox_xyxy": bbox_abs,
            "pred_iou": m.get("predicted_iou"),
            "stability": m.get("stability_score"),
        })
        obj_id += 1

    spec = PromptSpec(prompts=prompts)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        import yaml
        yaml.safe_dump(spec.to_dict(), f, default_flow_style=False)

    if args.stats_out:
        os.makedirs(os.path.dirname(args.stats_out), exist_ok=True)
        with open(args.stats_out, "w") as f:
            json.dump(stats, f, indent=2)

    if args.debug_dir:
        os.makedirs(args.debug_dir, exist_ok=True)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(os.path.join(args.debug_dir, "01_gray.jpg"), gray)
        dbg_roi = frame.copy()
        if roi_mask is not None:
            overlay = dbg_roi.copy()
            overlay[roi_mask == 0] = 255
            dbg_roi = cv2.addWeighted(overlay, 0.7, dbg_roi, 0.3, 0)
        cv2.rectangle(dbg_roi, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.imwrite(os.path.join(args.debug_dir, "02_roi.jpg"), dbg_roi)

        dbg_masks = frame.copy()
        for s in stats:
            bbox = s.get("bbox_xyxy")
            if bbox:
                bx1, by1, bx2, by2 = bbox
                cv2.rectangle(dbg_masks, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
            cx, cy = s["centroid"]
            cv2.circle(dbg_masks, (cx, cy), 4, (0, 0, 255), -1)
        cv2.imwrite(os.path.join(args.debug_dir, "03_masks_boxes.jpg"), dbg_masks)

    print(f"Frame: {frame_path}")
    print(f"ROI: x=[{x1},{x2}] y=[{y1},{y2}]")
    print(f"Masks generated: {len(masks)}")
    print(f"Prompts saved: {args.output} (objects={len(prompts)})")


if __name__ == "__main__":
    main()
