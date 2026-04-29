#!/usr/bin/env python3
"""
Run flow tracking using SAM2 chunked self-reprompting.

Workflow:
1. Seed SAM2 with one initial mask prompt at --seed-frame-idx.
2. Propagate in chunks of --chunk-interval frames.
3. At each boundary, use SAM2's predicted boundary mask as the next mask prompt.
4. Save SAM2 masks and run downstream flow/debounce tracking.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bubble_tracking.flow_tracking import (  # noqa: E402
    DebounceConfig,
    PreprocessConfig,
    SegmentationConfig,
    TrackingConfig,
    load_roi_json,
    track_frames,
)
from bubble_tracking.frame_io import save_mask, sorted_frame_paths  # noqa: E402
from bubble_tracking.visualization import create_tracking_video  # noqa: E402


def _frame_index_from_path(path: str) -> int:
    match = re.search(r"(\d+)\.(?:jpg|jpeg|png)$", Path(path).name, re.IGNORECASE)
    if not match:
        raise ValueError(f"Failed to parse frame index from {path}")
    return int(match.group(1))


def _load_roi_mask(mask_path: str, shape: Tuple[int, ...]) -> np.ndarray:
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Failed to read ROI mask: {mask_path}")
    if mask.shape[:2] != shape[:2]:
        raise ValueError("ROI mask size does not match frame size")
    return mask


def _roi_mask_from_json(roi: dict, shape: Tuple[int, ...]) -> np.ndarray:
    mask = np.zeros(shape[:2], dtype=np.uint8)
    mask[roi["y1"]:roi["y2"], roi["x1"]:roi["x2"]] = 255
    return mask


def _load_seed_mask(mask_path: str, shape: Tuple[int, ...], roi_mask: Optional[np.ndarray]) -> np.ndarray:
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Failed to read seed mask: {mask_path}")
    if mask.shape[:2] != shape[:2]:
        raise ValueError("Seed mask size does not match frame size")
    seed = (mask > 0)
    if roi_mask is not None:
        seed = seed & (roi_mask > 0)
    if not np.any(seed):
        raise ValueError("Seed mask is empty after ROI masking")
    return seed.astype(bool)


def _save_masks_for_frames(
    frame_paths: List[str],
    masks_by_frame: Dict[int, np.ndarray],
    output_dir: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    frame_indices = [_frame_index_from_path(p) for p in frame_paths]
    first_frame = cv2.imread(frame_paths[0])
    if first_frame is None:
        raise ValueError(f"Failed to read frame: {frame_paths[0]}")
    empty = np.zeros(first_frame.shape[:2], dtype=np.uint8)

    for idx, frame_idx in enumerate(frame_indices):
        mask = masks_by_frame.get(idx, empty)
        save_mask(mask, os.path.join(output_dir, f"{frame_idx:06d}.png"))


def _should_log(idx: int, total: int, log_every: int) -> bool:
    if log_every <= 0:
        return False
    if idx == total - 1:
        return True
    return (idx % log_every) == 0


def _chunked_self_prompt_propagation(
    frames_dir: str,
    frame_paths: List[str],
    seed_frame_idx: int,
    seed_mask: np.ndarray,
    chunk_interval: int,
    roi_mask: Optional[np.ndarray],
    checkpoint: str,
    model_cfg: str,
    device: str,
    log_every: int,
) -> Tuple[Dict[int, np.ndarray], List[dict]]:
    import torch
    from sam2.build_sam import build_sam2_video_predictor

    if chunk_interval <= 0:
        raise ValueError("--chunk-interval must be > 0")
    if seed_frame_idx < 0:
        raise ValueError("--seed-frame-idx must be >= 0")

    total = len(frame_paths)
    if seed_frame_idx >= total:
        raise ValueError(f"--seed-frame-idx {seed_frame_idx} out of range for {total} frames")

    predictor = build_sam2_video_predictor(model_cfg, checkpoint, device=device)
    state = predictor.init_state(video_path=frames_dir)

    masks_by_frame: Dict[int, np.ndarray] = {}
    chunk_logs: List[dict] = []

    current_prompt_idx = int(seed_frame_idx)
    current_prompt_mask = seed_mask.astype(bool)
    chunk_idx = 0

    while current_prompt_idx < total:
        save_until = min(current_prompt_idx + chunk_interval, total) - 1
        next_prompt_idx = current_prompt_idx + chunk_interval
        need_next_prompt = next_prompt_idx < total
        read_until = next_prompt_idx if need_next_prompt else save_until

        predictor.reset_state(state)
        predictor.add_new_mask(
            inference_state=state,
            frame_idx=int(current_prompt_idx),
            obj_id=1,
            mask=current_prompt_mask.astype(np.uint8),
        )

        next_prompt_mask: Optional[np.ndarray] = None
        last_nonempty_saved: Optional[np.ndarray] = None

        device_l = (device or "").lower()
        use_cuda_autocast = device_l.startswith("cuda")
        autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_cuda_autocast else nullcontext()

        with torch.inference_mode():
            with autocast_ctx:
                for frame_idx, object_ids_out, masks_out in predictor.propagate_in_video(state):
                    if frame_idx < current_prompt_idx:
                        continue
                    if frame_idx > read_until:
                        break

                    if torch.is_tensor(object_ids_out):
                        object_ids_out = object_ids_out.tolist()
                    if torch.is_tensor(masks_out):
                        masks_out = masks_out.detach().cpu().numpy()

                    combined = np.zeros(masks_out.shape[-2:], dtype=np.int32)
                    for i, obj_id in enumerate(object_ids_out):
                        obj_mask = masks_out[i, 0] > 0.0
                        combined[obj_mask] = int(obj_id)

                    if roi_mask is not None:
                        combined[roi_mask == 0] = 0

                    area = int((combined > 0).sum())
                    if frame_idx <= save_until:
                        masks_by_frame[int(frame_idx)] = combined
                        if area > 0:
                            last_nonempty_saved = (combined > 0)
                        if _should_log(frame_idx, total, log_every):
                            print(
                                f"[progress] stage=self_chunk_propagate "
                                f"chunk={chunk_idx} frame {frame_idx + 1}/{total}"
                            )

                    if need_next_prompt and frame_idx == next_prompt_idx:
                        if area > 0:
                            next_prompt_mask = (combined > 0)
                        elif last_nonempty_saved is not None:
                            next_prompt_mask = last_nonempty_saved.copy()
                        else:
                            next_prompt_mask = np.zeros_like(current_prompt_mask, dtype=bool)
                        break

        log_entry = {
            "chunk_idx": int(chunk_idx),
            "prompt_frame_idx": int(current_prompt_idx),
            "prompt_area": int(current_prompt_mask.sum()),
            "saved_frame_start_idx": int(current_prompt_idx),
            "saved_frame_end_idx": int(save_until),
            "next_prompt_frame_idx": int(next_prompt_idx) if need_next_prompt else None,
            "next_prompt_area": int(next_prompt_mask.sum()) if next_prompt_mask is not None else None,
            "status": "ok",
        }

        if not need_next_prompt:
            chunk_logs.append(log_entry)
            break

        if next_prompt_mask is None or not np.any(next_prompt_mask):
            log_entry["status"] = "empty_next_prompt"
            chunk_logs.append(log_entry)
            break

        chunk_logs.append(log_entry)
        current_prompt_idx = int(next_prompt_idx)
        current_prompt_mask = next_prompt_mask.astype(bool)
        chunk_idx += 1

    return masks_by_frame, chunk_logs


def main() -> None:
    parser = argparse.ArgumentParser(description="SAM2 self-prompted chunked flow tracking")
    parser.add_argument("--frames", required=True, help="Frames directory")
    parser.add_argument("--reference", required=True, help="Zero-bubble reference image")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--roi", default=None, help="ROI JSON path (x1,y1,x2,y2)")
    parser.add_argument("--roi-mask", default=None, help="ROI mask PNG (optional polygon mask)")
    parser.add_argument("--seed-mask", required=True, help="Initial binary mask PNG used as first SAM2 mask prompt")
    parser.add_argument("--seed-frame-idx", type=int, default=2, help="Frame index for initial mask prompt")
    parser.add_argument("--chunk-interval", type=int, default=160, help="Chunk interval in frames")
    parser.add_argument("--sam2-checkpoint", default="checkpoints/sam2_hiera_tiny.pt", help="SAM2 checkpoint path")
    parser.add_argument("--sam2-model-cfg", default="sam2_hiera_t.yaml", help="SAM2 model cfg name")
    parser.add_argument("--sam2-device", default="cuda", help="SAM2 device (cuda or cpu)")
    parser.add_argument("--min-area", type=int, default=500, help="Min component area for downstream tracking")
    parser.add_argument("--fps", type=float, default=4.0, help="Frames per second")
    parser.add_argument("--mm-per-pixel", type=float, default=None, help="Scale in mm per pixel")
    parser.add_argument(
        "--preprocess-mode",
        choices=["raw", "default", "automatrix"],
        default="raw",
        help="Preprocess mode for optical flow/tracking stage",
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
    parser.add_argument("--log-every", type=int, default=10, help="Log progress every N frames (0 disables)")
    parser.add_argument("--no-video", action="store_true", help="Skip overlay video")
    parser.add_argument("--video-fps", type=float, default=10.0, help="FPS for overlay video")
    parser.add_argument("--overlay-dir", default=None, help="Optional directory to save overlay frames")
    args = parser.parse_args()

    roi = load_roi_json(args.roi) if args.roi else None
    frame_paths = sorted_frame_paths(args.frames)
    if not frame_paths:
        raise ValueError(f"No frames found in {args.frames}")

    first_frame = cv2.imread(frame_paths[0])
    if first_frame is None:
        raise ValueError("Failed to read first frame")

    roi_mask = None
    if args.roi_mask:
        roi_mask = _load_roi_mask(args.roi_mask, first_frame.shape)
    elif roi is not None:
        roi_mask = _roi_mask_from_json(roi, first_frame.shape)

    seed_mask = _load_seed_mask(args.seed_mask, first_frame.shape, roi_mask)

    sam2_masks_dir = os.path.join(args.output, "sam2_masks")
    os.makedirs(args.output, exist_ok=True)

    masks_by_frame, chunk_logs = _chunked_self_prompt_propagation(
        frames_dir=args.frames,
        frame_paths=frame_paths,
        seed_frame_idx=args.seed_frame_idx,
        seed_mask=seed_mask,
        chunk_interval=args.chunk_interval,
        roi_mask=roi_mask,
        checkpoint=args.sam2_checkpoint,
        model_cfg=args.sam2_model_cfg,
        device=args.sam2_device,
        log_every=args.log_every,
    )
    _save_masks_for_frames(frame_paths, masks_by_frame, sam2_masks_dir)

    config_out = {
        "frames_dir": os.path.abspath(args.frames),
        "reference_path": os.path.abspath(args.reference),
        "output_dir": os.path.abspath(args.output),
        "roi_json": os.path.abspath(args.roi) if args.roi else None,
        "roi_mask": os.path.abspath(args.roi_mask) if args.roi_mask else None,
        "mode": "sam2_self_prompted_chunks",
        "seed_mask": os.path.abspath(args.seed_mask),
        "seed_frame_idx": int(args.seed_frame_idx),
        "chunk_interval": int(args.chunk_interval),
        "sam2_checkpoint": args.sam2_checkpoint,
        "sam2_model_cfg": args.sam2_model_cfg,
        "sam2_device": args.sam2_device,
        "chunk_logs": chunk_logs,
        "tracking": {
            "use_flow": bool(args.use_flow),
            "velocity_mode": args.velocity_mode,
            "debounce_birth_frames": int(args.debounce_birth_frames),
            "debounce_death_frames": int(args.debounce_death_frames),
            "debounce_relink_max_centroid_dist_px": float(args.debounce_relink_max_centroid_dist_px),
            "debounce_relink_min_area_ratio": float(args.debounce_relink_min_area_ratio),
            "debounce_relink_max_area_ratio": float(args.debounce_relink_max_area_ratio),
            "fps": float(args.fps),
            "mm_per_pixel": args.mm_per_pixel,
            "preprocess_mode": args.preprocess_mode,
            "preprocess_raw_sigma": float(args.preprocess_raw_sigma),
            "min_area": int(args.min_area),
        },
    }
    with open(os.path.join(args.output, "sam2_self_prompted_flow_config.json"), "w") as f:
        json.dump(config_out, f, indent=2)

    tracking_cfg = TrackingConfig(
        fps=args.fps,
        mm_per_pixel=args.mm_per_pixel,
        use_flow=args.use_flow,
        velocity_mode=args.velocity_mode,
        log_every=args.log_every,
        debounce=DebounceConfig(
            birth_confirm_frames=max(1, int(args.debounce_birth_frames)),
            death_confirm_frames=max(1, int(args.debounce_death_frames)),
            relink_max_centroid_dist_px=float(args.debounce_relink_max_centroid_dist_px),
            relink_min_area_ratio=float(args.debounce_relink_min_area_ratio),
            relink_max_area_ratio=float(args.debounce_relink_max_area_ratio),
        ),
        preprocess=PreprocessConfig(
            mode=args.preprocess_mode,
            raw_sigma=args.preprocess_raw_sigma,
        ),
        segment=SegmentationConfig(min_area=args.min_area),
    )
    outputs = track_frames(
        frames_dir=args.frames,
        reference_path=args.reference,
        output_dir=args.output,
        masks_dir=sam2_masks_dir,
        roi=roi,
        config=tracking_cfg,
    )

    if not args.no_video:
        output_path = Path(args.output) / "tracking_overlay.mp4"
        create_tracking_video(
            frames_dir=args.frames,
            masks_dir=outputs["masks_dir"],
            result=outputs["result"],
            output_path=str(output_path),
            fps=args.video_fps,
            overlay_dir=args.overlay_dir,
        )

    print(f"Done. Output: {os.path.abspath(args.output)}")
    print(f"Chunk logs: {len(chunk_logs)}")
    print(f"Masks saved: {len(frame_paths)}")


if __name__ == "__main__":
    main()
