#!/usr/bin/env python3
"""
Auto-prompt with SAM2 on the first frame that yields masks, then track with SAM2.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bubble_tracking.frame_io import sorted_frame_paths
from bubble_tracking.schemas import PromptSpec, ObjectPrompt, PointPrompt
from bubble_tracking.sam2_adapter import create_runner
from bubble_tracking.track_builder import save_tracks, compute_track_statistics
from bubble_tracking.mask_events import analyze_masks_by_frame, save_area_timeseries_csv
from bubble_tracking.visualization import create_tracking_video

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


def load_roi_mask(mask_path: str, shape: tuple) -> np.ndarray:
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Failed to read ROI mask: {mask_path}")
    if mask.shape[:2] != shape[:2]:
        raise ValueError("ROI mask size does not match frame size")
    return mask


def roi_bbox(mask: np.ndarray):
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        raise ValueError("ROI mask is empty")
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def automask_candidates(
    frame_bgr: np.ndarray,
    roi_mask: np.ndarray | None,
    mask_gen: SAM2AutomaticMaskGenerator,
    device: str,
    min_area: int,
    max_area_ratio: float | None,
    mask_scale: float,
):
    h, w = frame_bgr.shape[:2]
    if roi_mask is not None:
        x1, y1, x2, y2 = roi_bbox(roi_mask)
        crop = frame_bgr[y1:y2, x1:x2].copy()
        crop_mask = roi_mask[y1:y2, x1:x2]
        crop[crop_mask == 0] = 255
        roi_area = int((roi_mask > 0).sum())
    else:
        x1, y1, x2, y2 = 0, 0, w, h
        crop = frame_bgr
        roi_area = h * w

    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    scale = mask_scale
    if scale < 1.0:
        rgb = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
        masks = mask_gen.generate(rgb)

    candidates = []
    for m in masks:
        mask = m["segmentation"]
        if scale < 1.0:
            mask = cv2.resize(mask.astype(np.uint8), (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
        area = int(mask.sum())
        if area < min_area:
            continue
        if max_area_ratio is not None and roi_area > 0:
            if area / roi_area > max_area_ratio:
                continue
        ys, xs = np.where(mask)
        if ys.size == 0:
            continue
        cx = int(xs.mean()) + x1
        cy = int(ys.mean()) + y1
        candidates.append({"area": area, "centroid": (cx, cy)})

    return candidates


def main():
    parser = argparse.ArgumentParser(description="Auto-prompt then track with SAM2")
    parser.add_argument("--frames", required=True, help="Directory of frames")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--checkpoint", required=True, help="SAM2 checkpoint path")
    parser.add_argument("--model-cfg", default="sam2_hiera_t.yaml", help="SAM2 model cfg")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--roi-mask", type=str, default=None, help="ROI mask PNG (nonzero=keep)")
    parser.add_argument("--start-frame", type=int, default=0, help="Start index to search")
    parser.add_argument("--max-search-frames", type=int, default=200, help="Max frames to search")
    parser.add_argument("--points-per-side", type=int, default=16, help="Automask density")
    parser.add_argument("--min-area", type=int, default=20, help="Min mask area")
    parser.add_argument("--max-area-ratio", type=float, default=0.5, help="Max mask area / ROI")
    parser.add_argument("--mask-scale", type=float, default=1.0, help="Scale for automask input")
    parser.add_argument("--max-prompts", type=int, default=1, help="Max prompts to keep")
    parser.add_argument(
        "--fill-hole-area",
        type=int,
        default=8,
        help="SAM2 background-hole fill area in low-res masks (upstream default 8). Set to 0 to preserve genuine holes.",
    )
    parser.add_argument("--prompts-out", type=str, default=None, help="Optional YAML path to save prompts")
    parser.add_argument("--event-min-area", type=int, default=25, help="Minimum connected-component area for split/merge analysis")
    parser.add_argument("--event-min-overlap-px", type=int, default=10, help="Minimum overlap pixels required to link components across frames")
    parser.add_argument("--event-min-link-iou", type=float, default=0.05, help="Minimum IoU for parent-child component linkage")
    parser.add_argument("--event-min-link-overlap-ratio", type=float, default=0.25, help="Minimum overlap/min(area) ratio for parent-child linkage")
    parser.add_argument("--fps", type=float, default=10.0, help="FPS for overlay video")
    parser.add_argument("--no-video", action="store_true", help="Skip overlay video")
    args = parser.parse_args()

    frame_paths = sorted_frame_paths(args.frames)
    if not frame_paths:
        raise ValueError(f"No frames found in {args.frames}")

    first = cv2.imread(frame_paths[0])
    if first is None:
        raise ValueError("Failed to read first frame")
    roi_mask = load_roi_mask(args.roi_mask, first.shape) if args.roi_mask else None

    # Build automask model once
    automask_model = build_sam2(args.model_cfg, args.checkpoint, device=args.device)
    mask_gen = SAM2AutomaticMaskGenerator(
        automask_model,
        points_per_side=args.points_per_side,
        output_mode="binary_mask",
    )

    found = None
    candidates = []
    end = min(len(frame_paths), args.start_frame + args.max_search_frames)
    for idx in range(args.start_frame, end):
        frame = cv2.imread(frame_paths[idx])
        if frame is None:
            continue
        try:
            candidates = automask_candidates(
                frame, roi_mask, mask_gen, args.device, args.min_area, args.max_area_ratio, args.mask_scale
            )
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            mask_gen = SAM2AutomaticMaskGenerator(
                automask_model,
                points_per_side=max(8, args.points_per_side // 2),
                output_mode="binary_mask",
            )
            candidates = automask_candidates(
                frame, roi_mask, mask_gen, args.device, args.min_area, args.max_area_ratio, args.mask_scale
            )
        if candidates:
            found = idx
            break

    if found is None:
        raise RuntimeError("No prompts found in search window")

    # Select prompts by area
    candidates.sort(key=lambda c: c["area"], reverse=True)
    candidates = candidates[: max(1, args.max_prompts)]

    prompts = []
    for obj_id, c in enumerate(candidates, start=1):
        cx, cy = c["centroid"]
        prompts.append(ObjectPrompt(
            object_id=obj_id,
            frame_idx=found,
            points=[PointPrompt(x=cx, y=cy, label=1)],
        ))

    spec = PromptSpec(prompts=prompts)
    if args.prompts_out:
        Path(args.prompts_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.prompts_out, "w") as f:
            yaml.safe_dump(spec.to_dict(), f, default_flow_style=False)

    # Run SAM2 tracking
    os.makedirs(args.output, exist_ok=True)
    masks_dir = os.path.join(args.output, "masks")
    os.makedirs(masks_dir, exist_ok=True)

    run_config = {
        "frames_dir": os.path.abspath(args.frames),
        "checkpoint": args.checkpoint,
        "model_cfg": args.model_cfg,
        "device": args.device,
        "created_at": datetime.now().isoformat(),
        "start_frame": args.start_frame,
        "found_frame": found,
        "num_prompts": len(prompts),
        "roi_mask": os.path.abspath(args.roi_mask) if args.roi_mask else None,
        "event_analysis": {
            "min_component_area": args.event_min_area,
            "min_overlap_pixels": args.event_min_overlap_px,
            "min_link_iou": args.event_min_link_iou,
            "min_link_overlap_ratio": args.event_min_link_overlap_ratio,
        },
    }
    with open(os.path.join(args.output, "run_config.json"), "w") as f:
        json.dump(run_config, f, indent=2)

    runner = create_runner(
        checkpoint_path=args.checkpoint,
        model_cfg=args.model_cfg,
        device=args.device,
        fill_hole_area=args.fill_hole_area,
    )
    result, masks = runner.run(
        frames_dir=args.frames,
        prompts=spec,
        output_dir=args.output,
        roi_mask=roi_mask,
    )

    save_tracks(result, os.path.join(args.output, "tracks.json"))
    stats = compute_track_statistics(result)
    mask_analysis = analyze_masks_by_frame(
        masks_by_frame=masks,
        min_component_area=args.event_min_area,
        min_overlap_pixels=args.event_min_overlap_px,
        min_link_iou=args.event_min_link_iou,
        min_link_overlap_ratio=args.event_min_link_overlap_ratio,
    )
    area_json_path = os.path.join(args.output, "area_timeseries.json")
    with open(area_json_path, "w") as f:
        json.dump(mask_analysis["area_timeseries"], f, indent=2)
    area_csv_path = os.path.join(args.output, "area_timeseries.csv")
    save_area_timeseries_csv(mask_analysis["area_timeseries"], area_csv_path)
    events_path = os.path.join(args.output, "events.json")
    with open(events_path, "w") as f:
        json.dump(
            {
                "metadata": mask_analysis["analysis_metadata"],
                "summary": mask_analysis["event_summary"],
                "events": mask_analysis["events"],
            },
            f,
            indent=2,
        )
    stats["bubble_area"] = mask_analysis["area_summary"]
    stats["mask_event_summary"] = mask_analysis["event_summary"]
    stats["mask_event_config"] = mask_analysis["analysis_metadata"]
    with open(os.path.join(args.output, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    if not args.no_video:
        create_tracking_video(
            frames_dir=args.frames,
            masks_dir=masks_dir,
            result=result,
            output_path=os.path.join(args.output, "overlay.mp4"),
            fps=args.fps,
        )

    print(f"Found first prompts at frame {found}")
    print(f"Prompts: {len(prompts)} (top by area)")
    print(
        f"Events: splits={mask_analysis['event_summary']['num_splits']}, "
        f"merges={mask_analysis['event_summary']['num_merges']}"
    )
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
