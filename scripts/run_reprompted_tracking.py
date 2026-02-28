#!/usr/bin/env python3
"""
Run SAM2 tracking with keyframe re-prompting (automask every N frames).
"""

import argparse
import sys
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bubble_tracking.frame_io import sorted_frame_paths, save_mask
from bubble_tracking.schemas import PromptSpec, ObjectPrompt, PointPrompt
from bubble_tracking.track_builder import build_tracks_from_masks, compute_track_statistics, save_tracks
from bubble_tracking.mask_events import analyze_masks_dir, save_area_timeseries_csv
from bubble_tracking.visualization import create_tracking_video
from bubble_tracking.reprompt_matcher import match_keyframes

from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


def load_roi_mask(roi_mask_path: str, shape: tuple) -> np.ndarray:
    mask = cv2.imread(roi_mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Failed to read roi mask: {roi_mask_path}")
    if mask.shape[:2] != shape[:2]:
        raise ValueError("ROI mask size does not match frame size")
    return mask


def automask_keyframe(frame_bgr, roi_mask, mask_gen, device, min_area, max_area_ratio, mask_scale):
    h, w = frame_bgr.shape[:2]
    if roi_mask is not None:
        frame_bgr = frame_bgr.copy()
        frame_bgr[roi_mask == 0] = 255
        roi_area = int((roi_mask > 0).sum())
    else:
        roi_area = h * w

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    scale = mask_scale
    if scale < 1.0:
        frame_rgb = cv2.resize(frame_rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
        masks = mask_gen.generate(frame_rgb)

    keep_masks = []
    for m in masks:
        mask = m["segmentation"]
        if scale < 1.0:
            mask = cv2.resize(mask.astype(np.uint8), (frame_bgr.shape[1], frame_bgr.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
        area = int(mask.sum())
        if area < min_area:
            continue
        if max_area_ratio is not None and roi_area > 0:
            if area / roi_area > max_area_ratio:
                continue
        keep_masks.append(mask)
    return keep_masks


def build_prompts(masks, frame_idx, id_map):
    prompts = []
    for j, mask in enumerate(masks):
        ys, xs = np.where(mask)
        if ys.size == 0:
            continue
        cx = int(xs.mean())
        cy = int(ys.mean())
        obj_id = id_map[j]
        prompts.append(ObjectPrompt(
            object_id=obj_id,
            frame_idx=frame_idx,
            points=[PointPrompt(x=cx, y=cy, label=1)],
        ))
    return PromptSpec(prompts=prompts)


def main():
    parser = argparse.ArgumentParser(description="Re-prompted SAM2 tracking")
    parser.add_argument("--frames", required=True, help="Frames directory")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--checkpoint", required=True, help="SAM2 checkpoint path")
    parser.add_argument("--model-cfg", default="sam2_hiera_t.yaml", help="SAM2 model cfg")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--keyframe-interval", type=int, default=5, help="N for re-prompting")
    parser.add_argument("--points-per-side", type=int, default=32, help="Automask density")
    parser.add_argument("--min-points-per-side", type=int, default=8, help="Min automask density on fallback")
    parser.add_argument("--min-area", type=int, default=20, help="Min mask area")
    parser.add_argument("--mask-scale", type=float, default=1.0, help="Scale for automask input (e.g., 0.5)")
    parser.add_argument("--max-area-ratio", type=float, default=0.5, help="Max mask area / ROI")
    parser.add_argument("--roi-mask", type=str, default=None, help="ROI mask PNG")
    parser.add_argument("--iou-threshold", type=float, default=0.2, help="IoU threshold")
    parser.add_argument("--dist-threshold", type=float, default=50.0, help="Centroid distance threshold")
    parser.add_argument("--event-min-area", type=int, default=25, help="Minimum connected-component area for split/merge analysis")
    parser.add_argument("--event-min-overlap-px", type=int, default=10, help="Minimum overlap pixels required to link components across frames")
    parser.add_argument("--event-min-link-iou", type=float, default=0.05, help="Minimum IoU for parent-child component linkage")
    parser.add_argument("--event-min-link-overlap-ratio", type=float, default=0.25, help="Minimum overlap/min(area) ratio for parent-child linkage")
    parser.add_argument("--no-video", action="store_true", help="Skip overlay video")
    parser.add_argument("--fps", type=float, default=10.0, help="FPS for overlay video")
    args = parser.parse_args()

    frame_paths = sorted_frame_paths(args.frames)
    if not frame_paths:
        raise ValueError(f"No frames found in {args.frames}")
    first = cv2.imread(frame_paths[0])
    if first is None:
        raise ValueError("Failed to read first frame")
    roi_mask = load_roi_mask(args.roi_mask, first.shape) if args.roi_mask else None

    os.makedirs(args.output, exist_ok=True)
    masks_dir = os.path.join(args.output, "masks")
    os.makedirs(masks_dir, exist_ok=True)

    keyframe_events = []
    prev_ids = []
    prev_masks = []
    next_id = 1

    # Build SAM2 automask once to avoid repeated GPU allocations
    automask_model = build_sam2(args.model_cfg, args.checkpoint, device=args.device)
    mask_gen = SAM2AutomaticMaskGenerator(
        automask_model,
        points_per_side=args.points_per_side,
        output_mode="binary_mask",
    )

    # Build video predictor once and init state once to avoid reloading frames
    predictor = build_sam2_video_predictor(args.model_cfg, args.checkpoint, device=args.device)
    state = predictor.init_state(video_path=args.frames)

    for t in range(0, len(frame_paths), args.keyframe_interval):
        frame = cv2.imread(frame_paths[t])
        if frame is None:
            continue

        try:
            curr_masks = automask_keyframe(
                frame,
                roi_mask,
                mask_gen,
                args.device,
                args.min_area,
                args.max_area_ratio,
                args.mask_scale,
            )
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            fallback_pps = max(args.min_points_per_side, args.points_per_side // 2)
            mask_gen = SAM2AutomaticMaskGenerator(
                automask_model,
                points_per_side=fallback_pps,
                output_mode="binary_mask",
            )
            curr_masks = automask_keyframe(
                frame,
                roi_mask,
                mask_gen,
                args.device,
                args.min_area,
                args.max_area_ratio,
                args.mask_scale,
            )

        match = match_keyframes(
            prev_ids,
            prev_masks,
            curr_masks,
            iou_threshold=args.iou_threshold,
            dist_threshold=args.dist_threshold,
        )

        id_map = {}
        for j, prev_id in match.matches.items():
            id_map[j] = prev_id
        for j in match.births:
            id_map[j] = next_id
            next_id += 1
        for prev_id, split_js in match.splits.items():
            for j in split_js:
                id_map[j] = next_id
                next_id += 1

        keyframe_events.append({
            "frame_idx": t,
            "births": [id_map[j] for j in match.births],
            "deaths": match.deaths,
            "splits": {str(k): [id_map[j] for j in v] for k, v in match.splits.items()},
        })

        # Build prompts for this keyframe
        if not curr_masks:
            keyframe_events.append({
                "frame_idx": t,
                "births": [],
                "deaths": match.deaths,
                "splits": {},
                "reason": "no_masks",
            })
            continue

        prompts = build_prompts(curr_masks, t, id_map)

        # Reset state for this keyframe and add prompts
        predictor.reset_state(state)
        for prompt in prompts.prompts:
            points = np.array([[p.x, p.y] for p in prompt.points])
            labels = np.array([p.label for p in prompt.points])
            predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=prompt.frame_idx,
                obj_id=prompt.object_id,
                points=points,
                labels=labels,
            )

        with torch.inference_mode(), torch.autocast(args.device, dtype=torch.bfloat16):
            for frame_idx, object_ids_out, masks_out in predictor.propagate_in_video(state):
                if frame_idx < t:
                    continue
                if frame_idx >= min(t + args.keyframe_interval, len(frame_paths)):
                    break
                if torch.is_tensor(object_ids_out):
                    object_ids_out = object_ids_out.tolist()
                if torch.is_tensor(masks_out):
                    masks_out = masks_out.detach().cpu().numpy()
                combined = np.zeros(masks_out.shape[-2:], dtype=np.int32)
                for i, obj_id in enumerate(object_ids_out):
                    obj_mask = masks_out[i, 0] > 0.0
                    combined[obj_mask] = obj_id
                if roi_mask is not None:
                    combined[roi_mask == 0] = 0
                save_mask(combined, os.path.join(masks_dir, f"{frame_idx:06d}.png"))

        # Update previous keyframe state
        prev_ids = [id_map[j] for j in range(len(curr_masks)) if j in id_map]
        prev_masks = curr_masks

    # Build tracks + stats
    result = build_tracks_from_masks(masks_dir)
    save_tracks(result, os.path.join(args.output, "tracks.json"))
    stats = compute_track_statistics(result)

    mask_analysis = analyze_masks_dir(
        masks_dir=masks_dir,
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
    with open(os.path.join(args.output, "keyframes.json"), "w") as f:
        json.dump(keyframe_events, f, indent=2)

    if not args.no_video:
        create_tracking_video(
            frames_dir=args.frames,
            masks_dir=masks_dir,
            result=result,
            output_path=os.path.join(args.output, "overlay.mp4"),
            fps=args.fps,
        )


if __name__ == "__main__":
    main()
