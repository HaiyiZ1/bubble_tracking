#!/usr/bin/env python3
"""
Run flow tracking using SAM2 propagation seeded by a classic or SAM2 prompt.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bubble_tracking.flow_tracking import (
    DebounceConfig,
    PreprocessConfig,
    SegmentationConfig,
    TrackingConfig,
    load_roi_json,
    preprocess_frame,
    preprocess_frame_with_debug,
    segment_frame,
    segment_frame_automatrix,
    track_frames,
)
from bubble_tracking.frame_io import save_mask, sorted_frame_paths
from bubble_tracking.sam2_adapter import create_runner
from bubble_tracking.schemas import ObjectPrompt, PointPrompt, PromptSpec
from bubble_tracking.visualization import create_tracking_video, draw_mask_overlay


def _frame_index_from_path(path: str) -> int:
    match = re.search(r"(\d+)\.(?:jpg|jpeg|png)$", Path(path).name, re.IGNORECASE)
    if not match:
        raise ValueError(f"Failed to parse frame index from {path}")
    return int(match.group(1))


def _apply_roi(frame: np.ndarray, roi: Optional[dict]) -> Tuple[np.ndarray, Tuple[int, int]]:
    if roi is None:
        return frame, (0, 0)
    x1, y1, x2, y2 = roi["x1"], roi["y1"], roi["x2"], roi["y2"]
    return frame[y1:y2, x1:x2], (x1, y1)


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


def _classic_prompt(
    frame_paths: List[str],
    reference_full: np.ndarray,
    roi: Optional[dict],
    roi_mask: Optional[np.ndarray],
    min_area: int,
    log_every: int,
) -> Tuple[int, np.ndarray, Dict[str, float]]:
    preprocess_cfg = PreprocessConfig()
    segment_cfg = SegmentationConfig(min_area=min_area)
    total = len(frame_paths)

    if roi is not None:
        target_x = 0.5 * (roi["x1"] + roi["x2"])
        target_y = float(roi["y2"] - 1)
    else:
        h, w = reference_full.shape[:2]
        target_x = 0.5 * w
        target_y = float(h - 1)

    for idx, path in enumerate(frame_paths):
        frame_full = cv2.imread(path)
        if frame_full is None:
            raise ValueError(f"Failed to read frame: {path}")
        if frame_full.shape[:2] != reference_full.shape[:2]:
            raise ValueError("Frame and reference image sizes do not match")

        frame_crop, (x_off, y_off) = _apply_roi(frame_full, roi)
        ref_crop, _ = _apply_roi(reference_full, roi)

        gray = preprocess_frame(frame_crop, ref_crop, preprocess_cfg)
        labels, components = segment_frame(gray, segment_cfg)
        if components:
            target_x_crop = target_x - x_off
            target_y_crop = target_y - y_off
            best = None
            best_dist = None
            for comp in components.values():
                dx = comp.centroid[0] - target_x_crop
                dy = comp.centroid[1] - target_y_crop
                dist = math.hypot(dx, dy)
                if best is None or dist < best_dist:
                    best = comp
                    best_dist = dist

            if best is None:
                raise RuntimeError("Failed to select a classic prompt component")

            mask_crop = labels == best.label
            if roi is None:
                mask_full = mask_crop.astype(bool)
            else:
                mask_full = np.zeros(frame_full.shape[:2], dtype=bool)
                x1, y1, x2, y2 = roi["x1"], roi["y1"], roi["x2"], roi["y2"]
                mask_full[y1:y2, x1:x2] = mask_crop

            if roi_mask is not None:
                mask_full = mask_full & (roi_mask > 0)

            prompt_info = {
                "centroid_x": float(best.centroid[0] + x_off),
                "centroid_y": float(best.centroid[1] + y_off),
                "area_px": float(best.area),
                "target_x": float(target_x),
                "target_y": float(target_y),
                "distance_px": float(best_dist if best_dist is not None else 0.0),
            }
            return idx, mask_full, prompt_info

        if _should_log(idx, total, log_every):
            print(f"[progress] stage=classic_search frame {idx + 1}/{total}")

    raise RuntimeError("No classic prompt found in frames")


def _sam2_candidates(
    frame_bgr: np.ndarray,
    roi_mask: Optional[np.ndarray],
    mask_gen,
    device: str,
    min_area: int,
    max_area_ratio: Optional[float],
    mask_scale: float,
) -> List[dict]:
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


def _find_sam2_prompt(
    frame_paths: List[str],
    roi_mask: Optional[np.ndarray],
    device: str,
    checkpoint: str,
    model_cfg: str,
    points_per_side: int,
    min_points_per_side: int,
    min_area: int,
    max_area_ratio: Optional[float],
    mask_scale: float,
    min_mask_scale: float,
    start_frame: int,
    max_search_frames: int,
    max_prompts: int,
    log_every: int,
) -> Tuple[PromptSpec, int, Dict[str, float], List[dict]]:
    try:
        import torch
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    except Exception as exc:
        raise ImportError("SAM2 is required for prompt search.") from exc

    automask_model = build_sam2(model_cfg, checkpoint, device=device)
    current_points = int(points_per_side)
    current_scale = float(mask_scale)
    mask_gen = SAM2AutomaticMaskGenerator(
        automask_model,
        points_per_side=current_points,
        output_mode="binary_mask",
    )

    oom_events: List[dict] = []
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
                "points_per_side": float(current_points),
                "mask_scale": float(current_scale),
            }, oom_events

        if _should_log(idx - start_frame, total, log_every):
            print(f"[progress] stage=sam2_search frame {idx + 1}/{len(frame_paths)}")

    raise RuntimeError("No prompts found in search window")


def _run_sam2_propagation(
    frames_dir: str,
    prompts: PromptSpec,
    roi_mask: Optional[np.ndarray],
    checkpoint: str,
    model_cfg: str,
    device: str,
    fill_hole_area: int = 8,
    offload_video_to_cpu: bool = False,
    offload_state_to_cpu: bool = False,
) -> Dict[int, np.ndarray]:
    import torch

    runner = create_runner(
        checkpoint_path=checkpoint,
        model_cfg=model_cfg,
        device=device,
        fill_hole_area=fill_hole_area,
        offload_video_to_cpu=offload_video_to_cpu,
        offload_state_to_cpu=offload_state_to_cpu,
    )
    try:
        _, masks_by_frame = runner.run(
            frames_dir=frames_dir,
            prompts=prompts,
            output_dir=None,
            roi_mask=roi_mask,
        )
    except torch.OutOfMemoryError as exc:
        print("[sam2] OOM during propagation; consider lower resolution or smaller model.", file=sys.stderr)
        raise exc
    return masks_by_frame


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


def _save_cv_keyframe_debug_artifacts(
    debug_dir: str,
    frame_full: np.ndarray,
    preprocessed_gray: np.ndarray,
    mask_full: np.ndarray,
    base_idx: int,
    prompt_idx: int,
) -> Dict[str, str]:
    os.makedirs(debug_dir, exist_ok=True)
    prefix = f"base_{base_idx:06d}_prompt_{prompt_idx:06d}"

    mask_u8 = (mask_full.astype(np.uint8) * 255)
    inst_mask = np.zeros(mask_full.shape[:2], dtype=np.int32)
    inst_mask[mask_full > 0] = 1
    overlay = draw_mask_overlay(
        frame_full,
        inst_mask,
        colors={1: (0, 255, 0)},
        alpha=0.45,
    )

    overlay_path = os.path.join(debug_dir, f"{prefix}_overlay.jpg")
    mask_path = os.path.join(debug_dir, f"{prefix}_mask.png")
    preprocessed_path = os.path.join(debug_dir, f"{prefix}_preprocessed.png")
    cv2.imwrite(overlay_path, overlay)
    cv2.imwrite(mask_path, mask_u8)
    cv2.imwrite(preprocessed_path, preprocessed_gray)
    return {
        "overlay": os.path.abspath(overlay_path),
        "mask": os.path.abspath(mask_path),
        "preprocessed": os.path.abspath(preprocessed_path),
    }


def _negative_points_from_holes(
    mask_full: np.ndarray,
    min_hole_area: int,
    max_neg_points: int,
) -> List[PointPrompt]:
    """Find interior holes in a binary mask and return their representative
    points as negative (label=0) prompts. Holes that touch the image border
    are skipped — those are outside background, not enclosed holes."""
    if max_neg_points <= 0:
        return []

    binary = mask_full.astype(bool)
    if not binary.any():
        return []

    H, W = binary.shape
    bg = (~binary).astype(np.uint8)
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bg, connectivity=8)

    holes: List[Tuple[int, int, int]] = []
    for lbl in range(1, n_labels):
        x, y, w, h, area = stats[lbl]
        if x == 0 or y == 0 or x + w >= W or y + h >= H:
            continue
        if area < min_hole_area:
            continue
        cx, cy = centroids[lbl]
        cx_i, cy_i = int(round(cx)), int(round(cy))
        # Centroid can fall outside non-convex holes; fall back to any pixel inside.
        if not (0 <= cy_i < H and 0 <= cx_i < W and labels[cy_i, cx_i] == lbl):
            ys, xs = np.where(labels == lbl)
            if len(ys) == 0:
                continue
            cx_i, cy_i = int(xs[0]), int(ys[0])
        holes.append((int(area), cx_i, cy_i))

    # Largest holes first; they are the most reliable negatives.
    holes.sort(reverse=True)
    holes = holes[:max_neg_points]
    return [PointPrompt(x=cx, y=cy, label=0) for _, cx, cy in holes]


def _build_cv_keyframe_mask_prompts(
    frame_paths: List[str],
    reference_full: np.ndarray,
    roi: Optional[dict],
    roi_mask: Optional[np.ndarray],
    keyframe_interval: int,
    keyframe_start_idx: int,
    preprocess_mode: str,
    diff_sigma: float,
    use_clahe: bool,
    disable_light_map: bool,
    no_normalize: bool,
    min_area: int,
    threshold_offset: int,
    opening_radius: int,
    median_ksize: int,
    debug_dir: Optional[str],
    log_every: int,
    neg_points_from_holes: bool = False,
    min_hole_area: int = 4,
    max_neg_points: int = 16,
) -> Tuple[PromptSpec, Optional[int], List[dict]]:
    if keyframe_interval <= 0:
        raise ValueError("--cv-keyframe-interval must be > 0")
    if keyframe_start_idx < 0:
        raise ValueError("--cv-keyframe-start-idx must be >= 0")

    preprocess_cfg = PreprocessConfig(
        mode=preprocess_mode,
        diff_sigma=diff_sigma,
        use_clahe=use_clahe,
    )
    segment_cfg = SegmentationConfig(
        min_area=min_area,
        threshold_offset=threshold_offset,
        opening_radius=opening_radius,
        median_ksize=median_ksize,
    )

    prompts: List[ObjectPrompt] = []
    logs: List[dict] = []
    total = len(frame_paths)
    debug_root = os.path.abspath(debug_dir) if debug_dir else None
    if debug_root:
        os.makedirs(debug_root, exist_ok=True)

    for base_idx in range(keyframe_start_idx, total, keyframe_interval):
        segment_end = min(base_idx + keyframe_interval, total)
        selected = None

        # Search forward within this keyframe window; first non-empty CV mask wins.
        for search_idx in range(base_idx, segment_end):
            frame_full = cv2.imread(frame_paths[search_idx])
            if frame_full is None:
                continue
            if frame_full.shape[:2] != reference_full.shape[:2]:
                raise ValueError("Frame and reference image sizes do not match")

            frame_crop, _ = _apply_roi(frame_full, roi)
            ref_crop, _ = _apply_roi(reference_full, roi)

            if no_normalize and (preprocess_mode or "default").lower() in {"default", "automatrix"}:
                # No normalization at all: segment on raw diff gray with fixed 128 shift.
                diff = frame_crop.astype(np.float32) - ref_crop.astype(np.float32)
                if diff_sigma > 0:
                    diff = cv2.GaussianBlur(diff, (0, 0), sigmaX=diff_sigma, sigmaY=diff_sigma)
                diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
                gray = np.clip(diff_gray + 128.0, 0, 255).astype(np.uint8)
                if use_clahe:
                    clahe = cv2.createCLAHE(
                        clipLimit=preprocess_cfg.clahe_clip_limit,
                        tileGridSize=preprocess_cfg.clahe_tile_grid,
                    )
                    gray = clahe.apply(gray)
            elif disable_light_map and (preprocess_mode or "default").lower() in {"default", "automatrix"}:
                # Keep reference-diff contrast adjustment, but skip light-map division and second min-max stretch.
                debug = preprocess_frame_with_debug(frame_crop, ref_crop, preprocess_cfg)
                adjusted = debug["adjusted"]
                if adjusted.ndim == 3:
                    gray = cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY)
                else:
                    gray = adjusted.astype(np.uint8)
                if use_clahe:
                    clahe = cv2.createCLAHE(
                        clipLimit=preprocess_cfg.clahe_clip_limit,
                        tileGridSize=preprocess_cfg.clahe_tile_grid,
                    )
                    gray = clahe.apply(gray)
            else:
                gray = preprocess_frame(frame_crop, ref_crop, preprocess_cfg)
            if (preprocess_mode or "default").lower() == "automatrix":
                labels, components = segment_frame_automatrix(gray, segment_cfg)
            else:
                labels, components = segment_frame(gray, segment_cfg)
            if not components:
                continue

            mask_crop = labels > 0
            if roi is None:
                mask_full = mask_crop.astype(bool)
            else:
                mask_full = np.zeros(frame_full.shape[:2], dtype=bool)
                x1, y1, x2, y2 = roi["x1"], roi["y1"], roi["x2"], roi["y2"]
                mask_full[y1:y2, x1:x2] = mask_crop

            if roi_mask is not None:
                mask_full &= (roi_mask > 0)

            area = int(mask_full.sum())
            if area <= 0:
                continue

            debug_paths = {}
            if debug_root:
                debug_paths = _save_cv_keyframe_debug_artifacts(
                    debug_dir=debug_root,
                    frame_full=frame_full,
                    preprocessed_gray=gray,
                    mask_full=mask_full,
                    base_idx=base_idx,
                    prompt_idx=search_idx,
                )

            selected = {
                "base_idx": int(base_idx),
                "prompt_idx": int(search_idx),
                "area": int(area),
                "segment_end": int(segment_end),
                "num_components": int(len(components)),
                "mask": mask_full.astype(np.uint8),
                "debug_paths": debug_paths,
            }
            break

        if selected is None:
            logs.append({
                "base_idx": int(base_idx),
                "segment_end": int(segment_end),
                "status": "no_cv_mask",
            })
        else:
            prompt_idx = int(selected["prompt_idx"])
            neg_points: List[PointPrompt] = []
            if neg_points_from_holes:
                neg_points = _negative_points_from_holes(
                    selected["mask"],
                    min_hole_area=min_hole_area,
                    max_neg_points=max_neg_points,
                )
            prompts.append(
                ObjectPrompt(
                    object_id=1,
                    frame_idx=prompt_idx,
                    mask=selected["mask"],
                    points=neg_points,
                )
            )
            log_entry = {
                "base_idx": int(selected["base_idx"]),
                "prompt_idx": int(prompt_idx),
                "segment_end": int(selected["segment_end"]),
                "area": int(selected["area"]),
                "num_components": int(selected["num_components"]),
                "num_neg_points": int(len(neg_points)),
                "status": "ok",
            }
            if selected.get("debug_paths"):
                log_entry["debug"] = selected["debug_paths"]
            logs.append(log_entry)

        if _should_log(base_idx, total, log_every):
            print(f"[progress] stage=cv_keyframe_prompt frame {base_idx + 1}/{total}")

    if not prompts:
        raise RuntimeError("No CV keyframe prompts found. Try lower --cv-keyframe-min-area.")

    prompts.sort(key=lambda p: int(p.frame_idx))
    first_idx = int(prompts[0].frame_idx) if prompts else None
    return PromptSpec(prompts=prompts), first_idx, logs


def _build_cv_extra_keyframe_prompts(
    frame_paths: List[str],
    reference_full: np.ndarray,
    roi: Optional[dict],
    roi_mask: Optional[np.ndarray],
    extra_indices: List[int],
    preprocess_mode: str,
    diff_sigma: float,
    use_clahe: bool,
    disable_light_map: bool,
    no_normalize: bool,
    min_area: int,
    threshold_offset: int,
    opening_radius: int,
    median_ksize: int,
    debug_dir: Optional[str],
    neg_points_from_holes: bool = False,
    min_hole_area: int = 4,
    max_neg_points: int = 16,
) -> Tuple[List[ObjectPrompt], List[dict]]:
    """Build CV keyframe prompts at explicit frame indices (no within-window search)."""
    preprocess_cfg = PreprocessConfig(
        mode=preprocess_mode,
        diff_sigma=diff_sigma,
        use_clahe=use_clahe,
    )
    segment_cfg = SegmentationConfig(
        min_area=min_area,
        threshold_offset=threshold_offset,
        opening_radius=opening_radius,
        median_ksize=median_ksize,
    )
    debug_root = os.path.abspath(debug_dir) if debug_dir else None
    if debug_root:
        os.makedirs(debug_root, exist_ok=True)

    out_prompts: List[ObjectPrompt] = []
    out_logs: List[dict] = []
    total = len(frame_paths)
    for idx in extra_indices:
        if idx < 0 or idx >= total:
            out_logs.append({"extra_idx": int(idx), "status": "out_of_range"})
            continue
        frame_full = cv2.imread(frame_paths[idx])
        if frame_full is None:
            out_logs.append({"extra_idx": int(idx), "status": "frame_read_failed"})
            continue
        if frame_full.shape[:2] != reference_full.shape[:2]:
            raise ValueError("Frame and reference image sizes do not match")
        frame_crop, _ = _apply_roi(frame_full, roi)
        ref_crop, _ = _apply_roi(reference_full, roi)

        if no_normalize and (preprocess_mode or "default").lower() in {"default", "automatrix"}:
            diff = frame_crop.astype(np.float32) - ref_crop.astype(np.float32)
            if diff_sigma > 0:
                diff = cv2.GaussianBlur(diff, (0, 0), sigmaX=diff_sigma, sigmaY=diff_sigma)
            diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            gray = np.clip(diff_gray + 128.0, 0, 255).astype(np.uint8)
            if use_clahe:
                clahe = cv2.createCLAHE(
                    clipLimit=preprocess_cfg.clahe_clip_limit,
                    tileGridSize=preprocess_cfg.clahe_tile_grid,
                )
                gray = clahe.apply(gray)
        elif disable_light_map and (preprocess_mode or "default").lower() in {"default", "automatrix"}:
            debug = preprocess_frame_with_debug(frame_crop, ref_crop, preprocess_cfg)
            adjusted = debug["adjusted"]
            gray = cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY) if adjusted.ndim == 3 else adjusted.astype(np.uint8)
            if use_clahe:
                clahe = cv2.createCLAHE(
                    clipLimit=preprocess_cfg.clahe_clip_limit,
                    tileGridSize=preprocess_cfg.clahe_tile_grid,
                )
                gray = clahe.apply(gray)
        else:
            gray = preprocess_frame(frame_crop, ref_crop, preprocess_cfg)

        if (preprocess_mode or "default").lower() == "automatrix":
            labels, components = segment_frame_automatrix(gray, segment_cfg)
        else:
            labels, components = segment_frame(gray, segment_cfg)
        if not components:
            out_logs.append({"extra_idx": int(idx), "status": "no_cv_mask"})
            continue

        mask_crop = labels > 0
        if roi is None:
            mask_full = mask_crop.astype(bool)
        else:
            mask_full = np.zeros(frame_full.shape[:2], dtype=bool)
            x1, y1, x2, y2 = roi["x1"], roi["y1"], roi["x2"], roi["y2"]
            mask_full[y1:y2, x1:x2] = mask_crop
        if roi_mask is not None:
            mask_full &= (roi_mask > 0)

        area = int(mask_full.sum())
        if area <= 0:
            out_logs.append({"extra_idx": int(idx), "status": "empty_after_roi"})
            continue

        debug_paths = {}
        if debug_root:
            debug_paths = _save_cv_keyframe_debug_artifacts(
                debug_dir=debug_root,
                frame_full=frame_full,
                preprocessed_gray=gray,
                mask_full=mask_full,
                base_idx=idx,
                prompt_idx=idx,
            )

        neg_points: List[PointPrompt] = []
        if neg_points_from_holes:
            neg_points = _negative_points_from_holes(
                mask_full,
                min_hole_area=min_hole_area,
                max_neg_points=max_neg_points,
            )
        out_prompts.append(
            ObjectPrompt(
                object_id=1,
                frame_idx=int(idx),
                mask=mask_full,
                points=neg_points,
            )
        )
        log_entry = {
            "extra_idx": int(idx),
            "prompt_idx": int(idx),
            "area": int(area),
            "num_components": int(len(components)),
            "num_neg_points": int(len(neg_points)),
            "status": "ok",
        }
        if debug_paths:
            log_entry["debug"] = debug_paths
        out_logs.append(log_entry)

    return out_prompts, out_logs


def main() -> None:
    parser = argparse.ArgumentParser(description="SAM2 prompted flow tracking")
    parser.add_argument("--frames", required=True, help="Frames directory")
    parser.add_argument("--reference", required=True, help="Zero-bubble reference image")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--roi", default=None, help="ROI JSON path (x1,y1,x2,y2)")
    parser.add_argument("--roi-mask", default=None, help="ROI mask PNG (optional polygon mask)")
    parser.add_argument("--prompt-mode", choices=["classic", "sam2"], default="sam2")
    parser.add_argument(
        "--sam2-tracking-min-area",
        type=int,
        default=None,
        help="Min component area (px) applied to SAM2-propagated masks before tracking; "
             "also used as the classic-prompt min component area. "
             "Defaults to --cv-keyframe-min-area so the prompt-stage and tracking-stage "
             "floors stay aligned unless explicitly overridden.",
    )
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
    parser.add_argument(
        "--sam2-only",
        action="store_true",
        help="Run only SAM2 prompt+propagation and export sam2_masks; skip tracking/events/video",
    )
    parser.add_argument("--no-video", action="store_true", help="Skip overlay video")
    parser.add_argument("--video-fps", type=float, default=10.0, help="FPS for overlay video")
    parser.add_argument("--overlay-dir", default=None, help="Optional directory to save overlay frames")
    parser.add_argument("--sam2-checkpoint", default="checkpoints/sam2_hiera_tiny.pt", help="SAM2 checkpoint path")
    parser.add_argument("--sam2-model-cfg", default="sam2_hiera_t.yaml", help="SAM2 model cfg name")
    parser.add_argument("--sam2-device", default="cuda", help="SAM2 device (cuda or cpu)")
    parser.add_argument("--sam2-start-frame", type=int, default=0, help="Start index for SAM2 prompt search")
    parser.add_argument("--sam2-max-search-frames", type=int, default=200, help="Max frames to search for prompt")
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
    parser.add_argument("--sam2-max-prompts", type=int, default=1, help="Max prompts to keep from search")
    parser.add_argument(
        "--sam2-fill-hole-area",
        type=int,
        default=8,
        help="SAM2 background-hole fill area in low-res masks (upstream default 8). Set to 0 to preserve genuine holes.",
    )
    parser.add_argument(
        "--cv-keyframe-prompts",
        action="store_true",
        help="Generate CV masks at keyframes and use them as SAM2 mask prompts in one run",
    )
    parser.add_argument("--cv-keyframe-interval", type=int, default=100, help="Keyframe interval for CV prompts")
    parser.add_argument("--cv-keyframe-start-idx", type=int, default=0, help="Start frame index for CV prompts")
    parser.add_argument(
        "--cv-keyframe-preprocess-mode",
        choices=["raw", "default", "automatrix"],
        default="default",
        help="Preprocess mode used only for CV keyframe prompt generation",
    )
    parser.add_argument(
        "--cv-keyframe-diff-sigma",
        type=float,
        default=0.5,
        help="Diff smoothing sigma used for CV keyframe prompt generation",
    )
    parser.add_argument(
        "--cv-keyframe-use-clahe",
        action="store_true",
        help="Enable CLAHE for CV keyframe prompt generation",
    )
    parser.add_argument(
        "--cv-keyframe-disable-light-map",
        action="store_true",
        help="Disable light-map division during CV keyframe prompt preprocessing",
    )
    parser.add_argument(
        "--cv-keyframe-no-normalize",
        action="store_true",
        help="No normalization at all for CV keyframe prompts (raw diff gray with 128 shift)",
    )
    parser.add_argument(
        "--cv-keyframe-min-area",
        type=int,
        default=100,
        help="Minimum component area for CV keyframe prompt masks",
    )
    parser.add_argument(
        "--cv-keyframe-threshold-offset",
        type=int,
        default=7,
        help="Threshold offset for CV keyframe prompt segmentation",
    )
    parser.add_argument(
        "--cv-keyframe-opening-radius",
        type=int,
        default=0,
        help="Morphological opening radius (pixels) for CV keyframe prompt segmentation. 0 disables opening.",
    )
    parser.add_argument(
        "--cv-keyframe-median-ksize",
        type=int,
        default=10,
        help="Median smoothing kernel size for CV keyframe prompt segmentation (rounded up to odd). <=1 disables.",
    )
    parser.add_argument(
        "--cv-keyframe-debug-dir",
        default=None,
        help="Optional directory to save CV keyframe debug overlays/masks/preprocessed images",
    )
    parser.add_argument(
        "--cv-keyframe-neg-points-from-holes",
        action="store_true",
        help="Auto-place negative point prompts inside enclosed holes of each keyframe CV mask. "
             "SAM2 uses the mask as seed (via prev_sam_mask_logits) and the points refine it, "
             "so negatives suppress the propagator's tendency to fill holes.",
    )
    parser.add_argument(
        "--cv-keyframe-min-hole-area",
        type=int,
        default=4,
        help="Minimum enclosed-hole area (pixels, original resolution) to receive a negative point.",
    )
    parser.add_argument(
        "--cv-keyframe-max-neg-points",
        type=int,
        default=16,
        help="Cap on negative points per keyframe. Largest holes are kept first.",
    )
    parser.add_argument(
        "--cv-keyframe-extra-indices",
        type=int,
        nargs="+",
        default=None,
        help="Extra frame indices to use as CV keyframe prompts in addition to the interval schedule.",
    )
    parser.add_argument(
        "--sam2-offload-video-to-cpu",
        action="store_true",
        help="Keep frame tensors on CPU; swap to GPU per access. Saves GPU memory at small latency cost.",
    )
    parser.add_argument(
        "--sam2-offload-state-to-cpu",
        action="store_true",
        help="Offload SAM2 inference state to CPU between frames. Saves more GPU memory; bigger latency cost.",
    )
    args = parser.parse_args()

    if args.sam2_tracking_min_area is None:
        args.sam2_tracking_min_area = args.cv_keyframe_min_area

    if args.cv_keyframe_prompts and args.prompt_mode != "sam2":
        parser.error("--cv-keyframe-prompts requires --prompt-mode sam2")
    if args.cv_keyframe_disable_light_map and args.cv_keyframe_no_normalize:
        parser.error("--cv-keyframe-disable-light-map and --cv-keyframe-no-normalize are mutually exclusive")
    if args.cv_keyframe_no_normalize and args.cv_keyframe_preprocess_mode == "raw":
        parser.error("--cv-keyframe-no-normalize is only valid with default/automatrix preprocess modes")

    roi = load_roi_json(args.roi) if args.roi else None
    frame_paths = sorted_frame_paths(args.frames)
    if not frame_paths:
        raise ValueError(f"No frames found in {args.frames}")

    reference_full = cv2.imread(args.reference)
    if reference_full is None:
        raise ValueError(f"Failed to read reference image: {args.reference}")

    first_frame = cv2.imread(frame_paths[0])
    if first_frame is None:
        raise ValueError("Failed to read first frame")

    roi_mask = None
    if args.roi_mask:
        roi_mask = _load_roi_mask(args.roi_mask, first_frame.shape)
    elif roi is not None:
        roi_mask = _roi_mask_from_json(roi, first_frame.shape)

    max_area_ratio = args.sam2_max_area_ratio
    if max_area_ratio is not None and max_area_ratio <= 0:
        max_area_ratio = None

    prompt_info = {}
    prompt_frame_idx: Optional[int] = None
    prompt_spec = PromptSpec(prompts=[])
    oom_events: List[dict] = []
    used_params = {}
    cv_keyframe_logs: List[dict] = []
    sam2_masks_dir = os.path.join(args.output, "sam2_masks")

    if args.cv_keyframe_prompts:
        prompt_spec, prompt_frame_idx, cv_keyframe_logs = _build_cv_keyframe_mask_prompts(
            frame_paths=frame_paths,
            reference_full=reference_full,
            roi=roi,
            roi_mask=roi_mask,
            keyframe_interval=args.cv_keyframe_interval,
            keyframe_start_idx=args.cv_keyframe_start_idx,
            preprocess_mode=args.cv_keyframe_preprocess_mode,
            diff_sigma=args.cv_keyframe_diff_sigma,
            use_clahe=args.cv_keyframe_use_clahe,
            disable_light_map=args.cv_keyframe_disable_light_map,
            no_normalize=args.cv_keyframe_no_normalize,
            min_area=args.cv_keyframe_min_area,
            threshold_offset=args.cv_keyframe_threshold_offset,
            opening_radius=args.cv_keyframe_opening_radius,
            median_ksize=args.cv_keyframe_median_ksize,
            debug_dir=args.cv_keyframe_debug_dir,
            log_every=args.log_every,
            neg_points_from_holes=args.cv_keyframe_neg_points_from_holes,
            min_hole_area=args.cv_keyframe_min_hole_area,
            max_neg_points=args.cv_keyframe_max_neg_points,
        )
        if args.cv_keyframe_extra_indices:
            existing_idx = {int(p.frame_idx) for p in prompt_spec.prompts}
            extras = [i for i in args.cv_keyframe_extra_indices if int(i) not in existing_idx]
            extra_prompts, extra_logs = _build_cv_extra_keyframe_prompts(
                frame_paths=frame_paths,
                reference_full=reference_full,
                roi=roi,
                roi_mask=roi_mask,
                extra_indices=extras,
                preprocess_mode=args.cv_keyframe_preprocess_mode,
                diff_sigma=args.cv_keyframe_diff_sigma,
                use_clahe=args.cv_keyframe_use_clahe,
                disable_light_map=args.cv_keyframe_disable_light_map,
                no_normalize=args.cv_keyframe_no_normalize,
                min_area=args.cv_keyframe_min_area,
                threshold_offset=args.cv_keyframe_threshold_offset,
                opening_radius=args.cv_keyframe_opening_radius,
                median_ksize=args.cv_keyframe_median_ksize,
                debug_dir=args.cv_keyframe_debug_dir,
                neg_points_from_holes=args.cv_keyframe_neg_points_from_holes,
                min_hole_area=args.cv_keyframe_min_hole_area,
                max_neg_points=args.cv_keyframe_max_neg_points,
            )
            cv_keyframe_logs = list(cv_keyframe_logs) + extra_logs
            merged = list(prompt_spec.prompts) + extra_prompts
            merged.sort(key=lambda p: int(p.frame_idx))
            prompt_spec = PromptSpec(prompts=merged)
            if merged:
                prompt_frame_idx = int(merged[0].frame_idx)
        masks_by_frame = _run_sam2_propagation(
            frames_dir=args.frames,
            prompts=prompt_spec,
            roi_mask=roi_mask,
            checkpoint=args.sam2_checkpoint,
            model_cfg=args.sam2_model_cfg,
            device=args.sam2_device,
            fill_hole_area=args.sam2_fill_hole_area,
            offload_video_to_cpu=args.sam2_offload_video_to_cpu,
            offload_state_to_cpu=args.sam2_offload_state_to_cpu,
        )
        _save_masks_for_frames(frame_paths, masks_by_frame, sam2_masks_dir)
        prompt_info = {
            "mode": "cv_keyframe_prompts",
            "num_prompts": int(len(prompt_spec.prompts)),
        }
        used_params = {
            "interval": int(args.cv_keyframe_interval),
            "start_idx": int(args.cv_keyframe_start_idx),
            "preprocess_mode": args.cv_keyframe_preprocess_mode,
            "diff_sigma": float(args.cv_keyframe_diff_sigma),
            "use_clahe": bool(args.cv_keyframe_use_clahe),
            "disable_light_map": bool(args.cv_keyframe_disable_light_map),
            "no_normalize": bool(args.cv_keyframe_no_normalize),
            "min_area": int(args.cv_keyframe_min_area),
            "threshold_offset": int(args.cv_keyframe_threshold_offset),
            "opening_radius": int(args.cv_keyframe_opening_radius),
            "median_ksize": int(args.cv_keyframe_median_ksize),
            "debug_dir": os.path.abspath(args.cv_keyframe_debug_dir) if args.cv_keyframe_debug_dir else None,
            "extra_indices": [int(i) for i in (args.cv_keyframe_extra_indices or [])],
        }
    else:
        if args.prompt_mode == "classic":
            prompt_frame_idx, prompt_mask, prompt_info = _classic_prompt(
                frame_paths=frame_paths,
                reference_full=reference_full,
                roi=roi,
                roi_mask=roi_mask,
                min_area=args.sam2_tracking_min_area,
                log_every=args.log_every,
            )
            prompt_spec = PromptSpec(prompts=[
                ObjectPrompt(
                    object_id=1,
                    frame_idx=prompt_frame_idx,
                    mask=prompt_mask,
                )
            ])
            used_params = {"mode": "classic"}
        else:
            prompt_spec, prompt_frame_idx, used_params, oom_events = _find_sam2_prompt(
                frame_paths=frame_paths,
                roi_mask=roi_mask,
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

        masks_by_frame = _run_sam2_propagation(
            frames_dir=args.frames,
            prompts=prompt_spec,
            roi_mask=roi_mask,
            checkpoint=args.sam2_checkpoint,
            model_cfg=args.sam2_model_cfg,
            device=args.sam2_device,
            fill_hole_area=args.sam2_fill_hole_area,
            offload_video_to_cpu=args.sam2_offload_video_to_cpu,
            offload_state_to_cpu=args.sam2_offload_state_to_cpu,
        )
        _save_masks_for_frames(frame_paths, masks_by_frame, sam2_masks_dir)

    frame_indices = [_frame_index_from_path(p) for p in frame_paths]
    prompt_frame_number: Optional[int] = None
    if prompt_frame_idx is not None:
        if prompt_frame_idx < len(frame_indices):
            prompt_frame_number = int(frame_indices[prompt_frame_idx])
        else:
            prompt_frame_number = int(prompt_frame_idx)
    os.makedirs(args.output, exist_ok=True)
    config_out = {
        "frames_dir": os.path.abspath(args.frames),
        "reference": os.path.abspath(args.reference),
        "roi_json": os.path.abspath(args.roi) if args.roi else None,
        "roi_mask": os.path.abspath(args.roi_mask) if args.roi_mask else None,
        "prompt_mode": "sam2_cv_keyframe_prompts" if args.cv_keyframe_prompts else args.prompt_mode,
        "prompt_frame_idx": int(prompt_frame_idx) if prompt_frame_idx is not None else None,
        "prompt_frame_number": int(prompt_frame_number) if prompt_frame_number is not None else None,
        "prompt_info": prompt_info,
        "sam2_checkpoint": args.sam2_checkpoint,
        "sam2_model_cfg": args.sam2_model_cfg,
        "sam2_device": args.sam2_device,
        "sam2_search": {
            "enabled": bool(not args.cv_keyframe_prompts),
            "start_frame": args.sam2_start_frame,
            "max_search_frames": args.sam2_max_search_frames,
            "points_per_side": args.sam2_points_per_side,
            "min_points_per_side": args.sam2_min_points_per_side,
            "min_area": args.sam2_min_area,
            "max_area_ratio": max_area_ratio,
            "mask_scale": args.sam2_mask_scale,
            "min_mask_scale": args.sam2_min_mask_scale,
            "max_prompts": args.sam2_max_prompts,
            "used_params": used_params,
            "oom_events": oom_events,
        },
        "sam2_cv_keyframe_prompts": {
            "enabled": bool(args.cv_keyframe_prompts),
            "interval": int(args.cv_keyframe_interval),
            "start_idx": int(args.cv_keyframe_start_idx),
            "preprocess_mode": args.cv_keyframe_preprocess_mode,
            "diff_sigma": float(args.cv_keyframe_diff_sigma),
            "use_clahe": bool(args.cv_keyframe_use_clahe),
            "disable_light_map": bool(args.cv_keyframe_disable_light_map),
            "no_normalize": bool(args.cv_keyframe_no_normalize),
            "min_area": int(args.cv_keyframe_min_area),
            "threshold_offset": int(args.cv_keyframe_threshold_offset),
            "opening_radius": int(args.cv_keyframe_opening_radius),
            "median_ksize": int(args.cv_keyframe_median_ksize),
            "debug_dir": os.path.abspath(args.cv_keyframe_debug_dir) if args.cv_keyframe_debug_dir else None,
            "num_prompts": int(len(prompt_spec.prompts)) if args.cv_keyframe_prompts else 0,
            "keyframe_logs": cv_keyframe_logs if args.cv_keyframe_prompts else [],
        },
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
        },
    }
    with open(os.path.join(args.output, "sam2_prompted_flow_config.json"), "w") as f:
        json.dump(config_out, f, indent=2)

    outputs = None
    if not args.sam2_only:
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
            segment=SegmentationConfig(min_area=args.sam2_tracking_min_area),
        )
        outputs = track_frames(
            frames_dir=args.frames,
            reference_path=args.reference,
            output_dir=args.output,
            masks_dir=sam2_masks_dir,
            roi=roi,
            config=tracking_cfg,
        )

    if args.cv_keyframe_prompts:
        with open(os.path.join(args.output, "sam2_cv_keyframe_prompt_summary.json"), "w") as f:
            json.dump({
                "num_prompts": int(len(prompt_spec.prompts)),
                "used_params": used_params,
                "keyframe_logs": cv_keyframe_logs,
            }, f, indent=2)

    if (not args.sam2_only) and (not args.no_video):
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

    print(f"Prompt mode: {'sam2_cv_keyframe_prompts' if args.cv_keyframe_prompts else args.prompt_mode}")
    print(f"CV keyframe prompts: {args.cv_keyframe_prompts}")
    print(f"Prompt frame: {prompt_frame_number if prompt_frame_number is not None else 'cv-keyframes'}")
    if args.sam2_only:
        print("SAM2-only run complete.")
        print(f"Masks: {sam2_masks_dir}")
    else:
        print("Tracking complete.")
        print(f"Flow enabled: {args.use_flow}")
        print(f"Velocity mode: {args.velocity_mode}")
        print(f"Objects: {outputs['result'].num_objects}")
        print(f"Events: {outputs['event_summary']}")
        print(f"Masks: {outputs['masks_dir']}")


if __name__ == "__main__":
    main()
