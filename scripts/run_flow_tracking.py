#!/usr/bin/env python3
"""
Run flow-guided bubble tracking using a zero-bubble reference image.
"""

import argparse
import json
import os
import re
import sys
from contextlib import nullcontext
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bubble_tracking.flow_tracking import (
    DebounceConfig,
    FlowConfig,
    MatchConfig,
    PreprocessConfig,
    PersistenceConfig,
    SegmentationConfig,
    TrackingConfig,
    load_roi_json,
    track_frames,
)
from bubble_tracking.sam2_adapter import create_runner
from bubble_tracking.schemas import ObjectPrompt, PointPrompt, PromptSpec
from bubble_tracking.visualization import create_tracking_video
from bubble_tracking.frame_io import save_mask, sorted_frame_paths


def _frame_index_from_path(path: str) -> int:
    match = re.search(r"(\d+)\.(?:jpg|jpeg|png)$", Path(path).name, re.IGNORECASE)
    if not match:
        raise ValueError(f"Failed to parse frame index from {path}")
    return int(match.group(1))


def _load_roi_mask(mask_path: str, shape: tuple) -> np.ndarray:
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Failed to read ROI mask: {mask_path}")
    if mask.shape[:2] != shape[:2]:
        raise ValueError("ROI mask size does not match frame size")
    return mask


def _roi_mask_from_json(roi: dict, shape: tuple) -> np.ndarray:
    mask = np.zeros(shape[:2], dtype=np.uint8)
    mask[roi["y1"]:roi["y2"], roi["x1"]:roi["x2"]] = 255
    return mask


def _should_log(idx: int, total: int, log_every: int) -> bool:
    if log_every <= 0:
        return False
    if idx == total - 1:
        return True
    return (idx % log_every) == 0


def _build_sam2_masks(
    frame_paths: list[str],
    output_dir: str,
    roi_mask: np.ndarray | None,
    device: str,
    checkpoint: str,
    model_cfg: str,
    points_per_side: int,
    min_area: int,
    max_area_ratio: float | None,
    mask_scale: float,
    log_every: int,
) -> str:
    try:
        import torch
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    except Exception as exc:
        raise ImportError(
            "SAM2 is required for --sam2-auto. Install segment-anything-2 first."
        ) from exc

    os.makedirs(output_dir, exist_ok=True)
    automask_model = build_sam2(model_cfg, checkpoint, device=device)
    mask_gen = SAM2AutomaticMaskGenerator(
        automask_model,
        points_per_side=points_per_side,
        output_mode="binary_mask",
    )

    total = len(frame_paths)

    for idx, frame_path in enumerate(frame_paths):
        frame = cv2.imread(frame_path)
        if frame is None:
            raise ValueError(f"Failed to read frame: {frame_path}")

        frame_idx = _frame_index_from_path(frame_path)
        frame_bgr = frame
        roi_area = int((roi_mask > 0).sum()) if roi_mask is not None else frame.shape[0] * frame.shape[1]
        if roi_mask is not None:
            frame_bgr = frame.copy()
            frame_bgr[roi_mask == 0] = 255
            if roi_area == 0:
                raise ValueError("ROI mask is empty")

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        scale = float(mask_scale)
        if scale < 1.0:
            frame_rgb = cv2.resize(
                frame_rgb,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_AREA,
            )

        if device.startswith("cuda"):
            autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        else:
            autocast = nullcontext()

        with torch.inference_mode(), autocast:
            masks = mask_gen.generate(frame_rgb)

        combined = np.zeros(frame.shape[:2], dtype=np.uint8)
        for m in masks:
            mask = m["segmentation"]
            if scale < 1.0:
                mask = cv2.resize(
                    mask.astype(np.uint8),
                    (frame.shape[1], frame.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            area = int(mask.sum())
            if area < min_area:
                continue
            if max_area_ratio is not None and roi_area > 0:
                if area / roi_area > max_area_ratio:
                    continue
            combined[mask] = 1

        save_mask(combined, os.path.join(output_dir, f"{frame_idx:06d}.png"))

        if _should_log(idx, total, log_every):
            print(f"[progress] stage=sam2_auto frame {idx + 1}/{total}")

    return output_dir


def _sam2_candidates(
    frame_bgr: np.ndarray,
    roi_mask: np.ndarray | None,
    mask_gen,
    device: str,
    min_area: int,
    max_area_ratio: float | None,
    mask_scale: float,
) -> list[dict]:
    import torch

    roi_area = int((roi_mask > 0).sum()) if roi_mask is not None else frame_bgr.shape[0] * frame_bgr.shape[1]
    if roi_mask is not None:
        frame_bgr = frame_bgr.copy()
        frame_bgr[roi_mask == 0] = 255
        if roi_area == 0:
            raise ValueError("ROI mask is empty")

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    scale = float(mask_scale)
    if scale < 1.0:
        frame_rgb = cv2.resize(
            frame_rgb,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )

    if device.startswith("cuda"):
        autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        autocast = nullcontext()

    with torch.inference_mode(), autocast:
        masks = mask_gen.generate(frame_rgb)

    candidates = []
    for m in masks:
        mask = m["segmentation"]
        if scale < 1.0:
            mask = cv2.resize(
                mask.astype(np.uint8),
                (frame_bgr.shape[1], frame_bgr.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        area = int(mask.sum())
        if area < min_area:
            continue
        if max_area_ratio is not None and roi_area > 0:
            if area / roi_area > max_area_ratio:
                continue
        ys, xs = np.where(mask)
        if ys.size == 0:
            continue
        cx = int(xs.mean())
        cy = int(ys.mean())
        candidates.append({"area": area, "centroid": (cx, cy)})
    return candidates


def _find_sam2_prompts(
    frame_paths: list[str],
    roi_mask: np.ndarray | None,
    device: str,
    checkpoint: str,
    model_cfg: str,
    points_per_side: int,
    min_points_per_side: int,
    min_area: int,
    max_area_ratio: float | None,
    mask_scale: float,
    min_mask_scale: float,
    start_frame: int,
    max_search_frames: int,
    max_prompts: int,
    log_every: int,
) -> tuple[PromptSpec, int, dict, list[dict]]:
    try:
        import torch
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    except Exception as exc:
        raise ImportError(
            "SAM2 is required for --sam2-propagate. Install segment-anything-2 first."
        ) from exc

    automask_model = build_sam2(model_cfg, checkpoint, device=device)
    current_points = int(points_per_side)
    current_scale = float(mask_scale)
    mask_gen = SAM2AutomaticMaskGenerator(
        automask_model,
        points_per_side=current_points,
        output_mode="binary_mask",
    )

    oom_events: list[dict] = []
    end = min(len(frame_paths), start_frame + max_search_frames)
    total = max(0, end - start_frame)
    for idx in range(start_frame, end):
        frame = cv2.imread(frame_paths[idx])
        if frame is None:
            raise ValueError(f"Failed to read frame: {frame_paths[idx]}")

        while True:
            try:
                candidates = _sam2_candidates(
                    frame_bgr=frame,
                    roi_mask=roi_mask,
                    mask_gen=mask_gen,
                    device=device,
                    min_area=min_area,
                    max_area_ratio=max_area_ratio,
                    mask_scale=current_scale,
                )
                break
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                old_points, old_scale = current_points, current_scale
                if current_points > min_points_per_side:
                    current_points = max(min_points_per_side, current_points // 2)
                elif current_scale > min_mask_scale:
                    current_scale = max(min_mask_scale, current_scale * 0.5)
                else:
                    raise
                oom_events.append({
                    "frame_idx": idx,
                    "points_per_side_from": old_points,
                    "points_per_side_to": current_points,
                    "mask_scale_from": old_scale,
                    "mask_scale_to": current_scale,
                })
                print(
                    "[sam2] OOM during prompt search; "
                    f"points_per_side {old_points}->{current_points}, "
                    f"mask_scale {old_scale}->{current_scale}",
                    file=sys.stderr,
                )
                mask_gen = SAM2AutomaticMaskGenerator(
                    automask_model,
                    points_per_side=current_points,
                    output_mode="binary_mask",
                )
                continue

        if candidates:
            candidates.sort(key=lambda c: c["area"], reverse=True)
            candidates = candidates[: max(1, max_prompts)]
            prompts = []
            for obj_id, c in enumerate(candidates, start=1):
                cx, cy = c["centroid"]
                prompts.append(ObjectPrompt(
                    object_id=obj_id,
                    frame_idx=idx,
                    points=[PointPrompt(x=cx, y=cy, label=1)],
                ))
            spec = PromptSpec(prompts=prompts)
            return spec, idx, {
                "points_per_side": current_points,
                "mask_scale": current_scale,
            }, oom_events

        if _should_log(idx - start_frame, total, log_every):
            print(f"[progress] stage=sam2_search frame {idx + 1}/{len(frame_paths)}")

    raise RuntimeError("No prompts found in search window")


def _build_sam2_masks_propagate(
    frame_paths: list[str],
    output_dir: str,
    roi_mask: np.ndarray | None,
    device: str,
    checkpoint: str,
    model_cfg: str,
    prompts: PromptSpec,
) -> str:
    try:
        import torch
    except Exception as exc:
        raise ImportError("SAM2 is required for --sam2-propagate.") from exc

    runner = create_runner(
        checkpoint_path=checkpoint,
        model_cfg=model_cfg,
        device=device,
    )
    try:
        _, masks_by_frame = runner.run(
            frames_dir=str(Path(frame_paths[0]).parent),
            prompts=prompts,
            output_dir=None,
            roi_mask=roi_mask,
        )
    except torch.OutOfMemoryError as exc:
        print("[sam2] OOM during propagation; consider a smaller model or lower resolution.", file=sys.stderr)
        raise exc

    os.makedirs(output_dir, exist_ok=True)
    frame_indices = [_frame_index_from_path(p) for p in frame_paths]
    first_frame = cv2.imread(frame_paths[0])
    if first_frame is None:
        raise ValueError(f"Failed to read frame: {frame_paths[0]}")
    empty = np.zeros(first_frame.shape[:2], dtype=np.uint8)

    for idx, frame_idx in enumerate(frame_indices):
        mask = masks_by_frame.get(idx, empty)
        save_mask(mask, os.path.join(output_dir, f"{frame_idx:06d}.png"))

    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Flow-guided bubble tracking")
    parser.add_argument("--frames", required=True, help="Frames directory")
    parser.add_argument("--reference", required=True, help="Zero-bubble reference image")
    parser.add_argument("--masks-dir", default=None, help="Optional SAM2 masks directory for segmentation")
    parser.add_argument("--sam2-auto", action="store_true", help="Generate SAM2 auto-masks for every frame")
    parser.add_argument("--sam2-propagate", action="store_true", help="Auto-search prompt then propagate SAM2 masks")
    parser.add_argument("--sam2-checkpoint", default="checkpoints/sam2_hiera_tiny.pt", help="SAM2 checkpoint path")
    parser.add_argument("--sam2-model-cfg", default="sam2_hiera_t.yaml", help="SAM2 model cfg name")
    parser.add_argument("--sam2-device", default="cuda", help="SAM2 device (cuda or cpu)")
    parser.add_argument("--sam2-roi-mask", default=None, help="ROI mask PNG for SAM2 auto-masks")
    parser.add_argument("--sam2-points-per-side", type=int, default=32, help="SAM2 automask density")
    parser.add_argument("--sam2-min-points-per-side", type=int, default=8, help="Minimum points per side after OOM")
    parser.add_argument("--sam2-min-area", type=int, default=20, help="Minimum SAM2 mask area")
    parser.add_argument(
        "--sam2-max-area-ratio",
        type=float,
        default=0.5,
        help="Maximum SAM2 mask area / ROI (0 disables)",
    )
    parser.add_argument("--sam2-mask-scale", type=float, default=1.0, help="Scale for SAM2 automask input")
    parser.add_argument("--sam2-min-mask-scale", type=float, default=0.25, help="Minimum mask scale after OOM")
    parser.add_argument("--sam2-start-frame", type=int, default=0, help="Start index for prompt search")
    parser.add_argument("--sam2-max-search-frames", type=int, default=200, help="Max frames to search for prompt")
    parser.add_argument("--sam2-max-prompts", type=int, default=1, help="Max prompts to keep from search")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--roi", default=None, help="ROI JSON path (from scripts/select_roi.py)")
    parser.add_argument("--fps", type=float, default=4.0, help="Frames per second")
    parser.add_argument("--mm-per-pixel", type=float, default=None, help="Scale in mm per pixel")
    parser.add_argument("--diff-sigma", type=float, default=0.5, help="Gaussian sigma for diff smoothing")
    parser.add_argument("--grid-rows", type=int, default=8, help="Light matrix grid rows")
    parser.add_argument("--grid-cols", type=int, default=5, help="Light matrix grid cols")
    parser.add_argument(
        "--preprocess-mode",
        choices=["raw", "default", "automatrix"],
        default="default",
        help="Preprocess mode (automatrix matches picture_automatrix.py)",
    )
    parser.add_argument(
        "--preprocess-raw-sigma",
        type=float,
        default=1.0,
        help="Gaussian sigma used in raw preprocess mode (0 disables blur)",
    )
    parser.add_argument(
        "--use-flow",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable optical flow for association and velocity estimation",
    )
    parser.add_argument(
        "--velocity-mode",
        choices=["flow_median", "centroid_delta"],
        default="flow_median",
        help="Per-object velocity source",
    )
    parser.add_argument(
        "--debounce-birth-frames",
        type=int,
        default=1,
        help="Require N consecutive detections before confirming a birth (1 disables)",
    )
    parser.add_argument(
        "--debounce-death-frames",
        type=int,
        default=1,
        help="Require N consecutive misses before confirming a death (1 disables)",
    )
    parser.add_argument(
        "--debounce-relink-max-centroid-dist-px",
        type=float,
        default=20.0,
        help="Max centroid distance (px) to relink a reappeared ID to a recently missing ID",
    )
    parser.add_argument(
        "--debounce-relink-min-area-ratio",
        type=float,
        default=0.35,
        help="Min area ratio (new/old) allowed for ID relink",
    )
    parser.add_argument(
        "--debounce-relink-max-area-ratio",
        type=float,
        default=2.8,
        help="Max area ratio (new/old) allowed for ID relink",
    )
    parser.add_argument("--use-clahe", action="store_true", help="Enable CLAHE on normalized diff")
    parser.add_argument("--clahe-clip-limit", type=float, default=2.0, help="CLAHE clip limit")
    parser.add_argument(
        "--clahe-tile-grid",
        type=int,
        nargs=2,
        default=(8, 8),
        metavar=("TILE_W", "TILE_H"),
        help="CLAHE tile grid size (cols rows)",
    )
    parser.add_argument("--min-area", type=int, default=500, help="Minimum component area")
    parser.add_argument("--compact-max-area", type=int, default=0, help="Max area for compact filter")
    parser.add_argument(
        "--compact-min-dark-delta",
        type=float,
        default=0.0,
        help="Min darkness delta below threshold for compact filter",
    )
    parser.add_argument(
        "--edge-min-strength",
        type=float,
        default=0.0,
        help="Min boundary gradient strength to keep a component (0 disables)",
    )
    parser.add_argument(
        "--edge-boundary-radius",
        type=int,
        default=1,
        help="Boundary radius in pixels for edge-strength gate",
    )
    parser.add_argument(
        "--edge-min-fraction",
        type=float,
        default=0.0,
        help="Min fraction of boundary pixels above threshold (0 disables)",
    )
    parser.add_argument("--threshold-offset", type=int, default=7, help="Histogram peak offset")
    parser.add_argument("--flow-winsize", type=int, default=15, help="Farneback window size")
    parser.add_argument("--flow-levels", type=int, default=3, help="Farneback pyramid levels")
    parser.add_argument("--flow-iterations", type=int, default=3, help="Farneback iterations")
    parser.add_argument("--flow-backend", choices=["auto", "cpu"], default="auto", help="Flow backend selection")
    parser.add_argument("--cuda-device", type=int, default=0, help="CUDA device index for flow")
    parser.add_argument("--log-every", type=int, default=10, help="Log progress every N frames (0 disables)")
    parser.add_argument(
        "--persistence-min-overlap-px",
        type=int,
        default=0,
        help="Min overlap pixels to keep a component (0 disables temporal gating)",
    )
    parser.add_argument(
        "--persistence-max-area",
        type=int,
        default=0,
        help="Max area for persistence gating (0 applies to all components)",
    )
    parser.add_argument("--event-min-overlap-px", type=int, default=10, help="Minimum overlap pixels to link")
    parser.add_argument("--event-min-link-iou", type=float, default=0.05, help="Minimum IoU to link")
    parser.add_argument("--event-min-link-overlap-ratio", type=float, default=0.25, help="Minimum overlap ratio to link")
    parser.add_argument("--no-video", action="store_true", help="Skip overlay video")
    parser.add_argument("--video-fps", type=float, default=10.0, help="FPS for overlay video")
    parser.add_argument("--overlay-dir", default=None, help="Optional directory to save overlay frames")
    args = parser.parse_args()

    roi = load_roi_json(args.roi) if args.roi else None
    if args.sam2_auto and args.sam2_propagate:
        parser.error("--sam2-auto and --sam2-propagate are mutually exclusive")
    if args.sam2_auto and args.masks_dir:
        parser.error("--sam2-auto and --masks-dir are mutually exclusive")
    if args.sam2_propagate and args.masks_dir:
        parser.error("--sam2-propagate and --masks-dir are mutually exclusive")

    masks_dir = args.masks_dir
    max_area_ratio = args.sam2_max_area_ratio
    if max_area_ratio is not None and max_area_ratio <= 0:
        max_area_ratio = None

    frame_paths = None
    sam2_roi_mask = None
    if args.sam2_auto or args.sam2_propagate:
        frame_paths = sorted_frame_paths(args.frames)
        if not frame_paths:
            raise ValueError(f"No frames found in {args.frames}")
        first_frame = cv2.imread(frame_paths[0])
        if first_frame is None:
            raise ValueError("Failed to read first frame for SAM2 masks")

        if args.sam2_roi_mask:
            sam2_roi_mask = _load_roi_mask(args.sam2_roi_mask, first_frame.shape)
        elif roi is not None:
            sam2_roi_mask = _roi_mask_from_json(roi, first_frame.shape)

    if args.sam2_auto:
        sam2_masks_dir = os.path.join(args.output, "sam2_masks")
        masks_dir = _build_sam2_masks(
            frame_paths=frame_paths,
            output_dir=sam2_masks_dir,
            roi_mask=sam2_roi_mask,
            device=args.sam2_device,
            checkpoint=args.sam2_checkpoint,
            model_cfg=args.sam2_model_cfg,
            points_per_side=args.sam2_points_per_side,
            min_area=args.sam2_min_area,
            max_area_ratio=max_area_ratio,
            mask_scale=args.sam2_mask_scale,
            log_every=args.log_every,
        )

        os.makedirs(args.output, exist_ok=True)
        sam2_config = {
            "frames_dir": os.path.abspath(args.frames),
            "checkpoint": args.sam2_checkpoint,
            "model_cfg": args.sam2_model_cfg,
            "device": args.sam2_device,
            "roi_mask": os.path.abspath(args.sam2_roi_mask) if args.sam2_roi_mask else None,
            "roi_json": os.path.abspath(args.roi) if args.roi else None,
            "mode": "auto",
            "points_per_side": args.sam2_points_per_side,
            "min_area": args.sam2_min_area,
            "max_area_ratio": max_area_ratio,
            "mask_scale": args.sam2_mask_scale,
        }
        with open(os.path.join(args.output, "sam2_auto_config.json"), "w") as f:
            json.dump(sam2_config, f, indent=2)

    if args.sam2_propagate:
        if frame_paths is None:
            raise ValueError("Frame paths not initialized for SAM2 propagation")
        prompt_spec, prompt_frame_idx, used_params, oom_events = _find_sam2_prompts(
            frame_paths=frame_paths,
            roi_mask=sam2_roi_mask,
            device=args.sam2_device,
            checkpoint=args.sam2_checkpoint,
            model_cfg=args.sam2_model_cfg,
            points_per_side=args.sam2_points_per_side,
            min_points_per_side=args.sam2_min_points_per_side,
            min_area=args.sam2_min_area,
            max_area_ratio=max_area_ratio,
            mask_scale=args.sam2_mask_scale,
            min_mask_scale=args.sam2_min_mask_scale,
            start_frame=args.sam2_start_frame,
            max_search_frames=args.sam2_max_search_frames,
            max_prompts=args.sam2_max_prompts,
            log_every=args.log_every,
        )

        sam2_masks_dir = os.path.join(args.output, "sam2_masks")
        masks_dir = _build_sam2_masks_propagate(
            frame_paths=frame_paths,
            output_dir=sam2_masks_dir,
            roi_mask=sam2_roi_mask,
            device=args.sam2_device,
            checkpoint=args.sam2_checkpoint,
            model_cfg=args.sam2_model_cfg,
            prompts=prompt_spec,
        )

        frame_indices = [_frame_index_from_path(p) for p in frame_paths]
        prompt_frame_number = frame_indices[prompt_frame_idx] if prompt_frame_idx < len(frame_indices) else prompt_frame_idx
        os.makedirs(args.output, exist_ok=True)
        sam2_config = {
            "frames_dir": os.path.abspath(args.frames),
            "checkpoint": args.sam2_checkpoint,
            "model_cfg": args.sam2_model_cfg,
            "device": args.sam2_device,
            "roi_mask": os.path.abspath(args.sam2_roi_mask) if args.sam2_roi_mask else None,
            "roi_json": os.path.abspath(args.roi) if args.roi else None,
            "mode": "propagate",
            "prompt_frame_idx": int(prompt_frame_idx),
            "prompt_frame_number": int(prompt_frame_number),
            "points_per_side": used_params["points_per_side"],
            "mask_scale": used_params["mask_scale"],
            "min_area": args.sam2_min_area,
            "max_area_ratio": max_area_ratio,
            "start_frame": args.sam2_start_frame,
            "max_search_frames": args.sam2_max_search_frames,
            "max_prompts": args.sam2_max_prompts,
            "oom_events": oom_events,
        }
        with open(os.path.join(args.output, "sam2_auto_config.json"), "w") as f:
            json.dump(sam2_config, f, indent=2)

    config = TrackingConfig(
        fps=args.fps,
        mm_per_pixel=args.mm_per_pixel,
        flow_backend=args.flow_backend,
        cuda_device=args.cuda_device,
        log_every=args.log_every,
        use_flow=args.use_flow,
        velocity_mode=args.velocity_mode,
        debounce=DebounceConfig(
            birth_confirm_frames=max(1, int(args.debounce_birth_frames)),
            death_confirm_frames=max(1, int(args.debounce_death_frames)),
            relink_max_centroid_dist_px=float(args.debounce_relink_max_centroid_dist_px),
            relink_min_area_ratio=float(args.debounce_relink_min_area_ratio),
            relink_max_area_ratio=float(args.debounce_relink_max_area_ratio),
        ),
        preprocess=PreprocessConfig(
            diff_sigma=args.diff_sigma,
            grid_rows=args.grid_rows,
            grid_cols=args.grid_cols,
            mode=args.preprocess_mode,
            raw_sigma=args.preprocess_raw_sigma,
            use_clahe=args.use_clahe,
            clahe_clip_limit=args.clahe_clip_limit,
            clahe_tile_grid=tuple(args.clahe_tile_grid),
        ),
        segment=SegmentationConfig(
            min_area=args.min_area,
            threshold_offset=args.threshold_offset,
            compact_max_area=args.compact_max_area,
            compact_min_dark_delta=args.compact_min_dark_delta,
            edge_min_strength=args.edge_min_strength,
            edge_boundary_radius=args.edge_boundary_radius,
            edge_min_fraction=args.edge_min_fraction,
        ),
        flow=FlowConfig(
            winsize=args.flow_winsize,
            levels=args.flow_levels,
            iterations=args.flow_iterations,
        ),
        persistence=PersistenceConfig(
            min_overlap_pixels=args.persistence_min_overlap_px,
            max_area=args.persistence_max_area,
        ),
        match=MatchConfig(
            min_overlap_pixels=args.event_min_overlap_px,
            min_link_iou=args.event_min_link_iou,
            min_link_overlap_ratio=args.event_min_link_overlap_ratio,
        ),
    )

    outputs = track_frames(
        frames_dir=args.frames,
        reference_path=args.reference,
        output_dir=args.output,
        masks_dir=masks_dir,
        roi=roi,
        config=config,
    )

    if not args.no_video:
        output_path = Path(args.output) / "tracking_overlay.mp4"
        create_tracking_video(
            frames_dir=args.frames,
            masks_dir=outputs["masks_dir"],
            result=outputs["result"],
            output_path=str(output_path),
            fps=args.video_fps,
            alpha=0.4,
            overlay_dir=args.overlay_dir,
        )

    print("Tracking complete.")
    print(f"Flow backend: {outputs['flow_backend']}")
    print(f"Flow enabled: {args.use_flow}")
    print(f"Velocity mode: {args.velocity_mode}")
    print(f"Objects: {outputs['result'].num_objects}")
    print(f"Events: {outputs['event_summary']}")
    print(f"Masks: {outputs['masks_dir']}")


if __name__ == "__main__":
    main()
