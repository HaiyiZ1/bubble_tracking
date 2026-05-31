"""CV-keyframe prompt builder for Phase 1.

Builds CV-keyframe prompts and serializes them to disk so Phase 2 can
read them without needing to re-run the CV pipeline.

Translated from phase2_track/sam2.py:
  _build_cv_keyframe_mask_prompts
  _build_cv_extra_keyframe_prompts
  _save_cv_keyframe_debug_artifacts
  _negative_points_from_holes
  _apply_roi
  _should_log
  _draw_mask_overlay

Notable changes vs the phase2 origin:
- Function bodies preserved verbatim.
- build_cv_keyframes() added as the public orchestrator that writes to disk.
- Disk format: uint16 instance-ID masks + prompts.json metadata.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from bubble_common.schemas import ObjectPrompt, PointPrompt, PromptSpec

# ---------------------------------------------------------------------------
# v1 frozen constants (moved from phase2_track/schemas.py)
# ---------------------------------------------------------------------------

CV_KEYFRAME_INTERVAL: int = 160
CV_KEYFRAME_PREPROCESS_MODE: str = "default"
CV_KEYFRAME_DIFF_SIGMA: float = 0.5
CV_KEYFRAME_NORMALIZE_CONTRAST: bool = False  # matches --cv-keyframe-no-normalize
# Historical default from original scripts/run_sam2_prompted_flow.py (was 25 before the
# median_ksize default was lowered to 10 in flow_tracking.py). Matching the publishing
# oracle runs requires 25 here.
CV_KEYFRAME_MEDIAN_KSIZE: int = 25


# ---------------------------------------------------------------------------
# Private helpers (verbatim from phase2_track/sam2.py)
# ---------------------------------------------------------------------------


def _apply_roi(
    frame: np.ndarray, roi: Optional[dict]
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Crop frame to ROI rectangle. Returns (crop, (x1, y1)) offset."""
    if roi is None:
        return frame, (0, 0)
    x1, y1, x2, y2 = roi["x1"], roi["y1"], roi["x2"], roi["y2"]
    return frame[y1:y2, x1:x2], (x1, y1)


def _should_log(idx: int, total: int, log_every: int) -> bool:
    if log_every <= 0:
        return False
    if idx == total - 1:
        return True
    return (idx % log_every) == 0


def _draw_mask_overlay(
    frame: np.ndarray,
    mask: np.ndarray,
    colors: Optional[Dict[int, Tuple[int, int, int]]] = None,
    alpha: float = 0.5,
) -> np.ndarray:
    """Draw colored mask overlay on frame (inlined from visualization.py)."""
    result = frame.copy()
    unique_ids = np.unique(mask)
    unique_ids = unique_ids[unique_ids > 0]

    if colors is None:
        # Simple colour cycle when no palette provided
        palette = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]
        colors = {int(uid): palette[i % len(palette)] for i, uid in enumerate(unique_ids)}

    for obj_id in unique_ids:
        obj_mask = mask == obj_id
        color = colors.get(int(obj_id), (255, 255, 255))
        overlay = result.copy()
        overlay[obj_mask] = color
        result = cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0)
        contours, _ = cv2.findContours(
            obj_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(result, contours, -1, color, 2)

    return result


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
    overlay = _draw_mask_overlay(
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
    normalize_lightmap: bool,
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
    from bubble_common.preprocess import preprocess_frame_with_debug
    from bubble_common.schemas import PreprocessConfig, SegmentationConfig

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

            if no_normalize and (preprocess_mode or "default").lower() in {"default"}:
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
            elif (not normalize_lightmap) and (preprocess_mode or "default").lower() in {"default"}:
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
                from bubble_common.preprocess import preprocess_frame
                gray = preprocess_frame(frame_crop, ref_crop, preprocess_cfg)

            from bubble_common.segment import segment_frame
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
    normalize_lightmap: bool,
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
    from bubble_common.preprocess import preprocess_frame_with_debug
    from bubble_common.schemas import PreprocessConfig, SegmentationConfig

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

        if no_normalize and (preprocess_mode or "default").lower() in {"default"}:
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
        elif (not normalize_lightmap) and (preprocess_mode or "default").lower() in {"default"}:
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
            from bubble_common.preprocess import preprocess_frame
            gray = preprocess_frame(frame_crop, ref_crop, preprocess_cfg)

        from bubble_common.segment import segment_frame
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


# ---------------------------------------------------------------------------
# Public orchestrator
# ---------------------------------------------------------------------------


def build_cv_keyframes(
    *,
    frame_paths: List[str],
    reference_full: np.ndarray,
    roi: Optional[dict],
    roi_mask: Optional[np.ndarray],
    out_prompts_dir: Path,
    out_debug_dir: Path,
    start_idx: int,
    threshold_offset: int,
    min_area: int,
    normalize_lightmap: bool,
    extra_indices: Optional[List[int]] = None,
) -> dict:
    """Build CV-keyframe prompts and persist them to disk.

    Calls _build_cv_keyframe_mask_prompts (and optionally
    _build_cv_extra_keyframe_prompts), then writes:
      out_prompts_dir/<frame_idx:06d>.png  — uint16 instance-ID mask
                                             (pixel = object_id, 0 = bg)
      out_prompts_dir/prompts.json         — metadata (config used,
                                             per-object negative points)

    Debug artifacts (mask/overlay/preprocessed PNGs per keyframe) are
    written by _save_cv_keyframe_debug_artifacts to out_debug_dir.

    Returns a summary dict for inclusion in phase1_summary.json:
      {"n_keyframes": ..., "n_objects": ..., "frame_indices": [...]}
    """
    out_prompts_dir = Path(out_prompts_dir)
    out_debug_dir = Path(out_debug_dir)
    out_prompts_dir.mkdir(parents=True, exist_ok=True)
    out_debug_dir.mkdir(parents=True, exist_ok=True)

    prompt_spec, _first_idx, cv_logs = _build_cv_keyframe_mask_prompts(
        frame_paths=frame_paths,
        reference_full=reference_full,
        roi=roi,
        roi_mask=roi_mask,
        keyframe_interval=CV_KEYFRAME_INTERVAL,
        keyframe_start_idx=start_idx,
        preprocess_mode=CV_KEYFRAME_PREPROCESS_MODE,
        diff_sigma=CV_KEYFRAME_DIFF_SIGMA,
        use_clahe=False,
        normalize_lightmap=normalize_lightmap,
        no_normalize=not CV_KEYFRAME_NORMALIZE_CONTRAST,
        min_area=min_area,
        threshold_offset=threshold_offset,
        opening_radius=0,
        median_ksize=CV_KEYFRAME_MEDIAN_KSIZE,
        debug_dir=str(out_debug_dir),
        log_every=10,
    )

    # Merge any extra keyframe indices.
    if extra_indices:
        existing_idx = {int(p.frame_idx) for p in prompt_spec.prompts}
        extras = [i for i in extra_indices if int(i) not in existing_idx]
        if extras:
            extra_prompts, extra_logs = _build_cv_extra_keyframe_prompts(
                frame_paths=frame_paths,
                reference_full=reference_full,
                roi=roi,
                roi_mask=roi_mask,
                extra_indices=extras,
                preprocess_mode=CV_KEYFRAME_PREPROCESS_MODE,
                diff_sigma=CV_KEYFRAME_DIFF_SIGMA,
                use_clahe=False,
                normalize_lightmap=normalize_lightmap,
                no_normalize=not CV_KEYFRAME_NORMALIZE_CONTRAST,
                min_area=min_area,
                threshold_offset=threshold_offset,
                opening_radius=0,
                median_ksize=CV_KEYFRAME_MEDIAN_KSIZE,
                debug_dir=str(out_debug_dir),
            )
            cv_logs = list(cv_logs) + extra_logs
            merged = list(prompt_spec.prompts) + extra_prompts
            merged.sort(key=lambda p: int(p.frame_idx))
            prompt_spec = PromptSpec(prompts=merged)

    print(f"[phase1] {len(prompt_spec.prompts)} CV-keyframe prompts built.")

    # -----------------------------------------------------------------------
    # Serialize: per-keyframe instance-ID masks (uint16 PNG) + prompts.json
    # -----------------------------------------------------------------------

    # Group prompts by frame_idx.  There can be multiple object_ids per frame
    # (in v1 all have object_id=1, but we write the actual value for future-proofing).
    from collections import defaultdict
    prompts_by_frame: dict = defaultdict(list)
    for p in prompt_spec.prompts:
        prompts_by_frame[int(p.frame_idx)].append(p)

    keyframes_meta = []
    for frame_idx in sorted(prompts_by_frame.keys()):
        frame_prompts = prompts_by_frame[frame_idx]

        # Determine mask shape from the first prompt that has a mask.
        mask_shape = None
        for p in frame_prompts:
            if p.mask is not None:
                mask_shape = np.asarray(p.mask).shape[:2]
                break
        if mask_shape is None:
            # Fallback: write a zero mask sized to the first frame.
            first_frame = cv2.imread(frame_paths[0])
            mask_shape = first_frame.shape[:2] if first_frame is not None else (1, 1)

        instance_mask = np.zeros(mask_shape, dtype=np.uint16)
        objects_meta = []
        for p in frame_prompts:
            obj_id = int(p.object_id)
            if p.mask is not None:
                obj_mask = np.asarray(p.mask).astype(bool)
                instance_mask[obj_mask] = obj_id
            neg_pts = []
            if p.points:
                neg_pts = [[int(pt.x), int(pt.y)] for pt in p.points if pt.label == 0]
            objects_meta.append({
                "object_id": obj_id,
                "negative_points": neg_pts,
            })

        png_path = out_prompts_dir / f"{frame_idx:06d}.png"
        cv2.imwrite(str(png_path), instance_mask)

        keyframes_meta.append({
            "frame_idx": frame_idx,
            "n_objects": len(frame_prompts),
            "objects": objects_meta,
        })

    config_dict = {
        "start_idx": start_idx,
        "interval": CV_KEYFRAME_INTERVAL,
        "threshold_offset": threshold_offset,
        "min_area": min_area,
        "normalize_lightmap": normalize_lightmap,
        "extra_indices": [int(i) for i in (extra_indices or [])],
        "diff_sigma": CV_KEYFRAME_DIFF_SIGMA,
        "preprocess_mode": CV_KEYFRAME_PREPROCESS_MODE,
        "median_ksize": CV_KEYFRAME_MEDIAN_KSIZE,
        "no_normalize": not CV_KEYFRAME_NORMALIZE_CONTRAST,
        "opening_radius": 0,
    }

    prompts_json = {
        "schema_version": "1",
        "config": config_dict,
        "keyframes": keyframes_meta,
    }
    (out_prompts_dir / "prompts.json").write_text(
        json.dumps(prompts_json, indent=2) + "\n"
    )

    frame_indices = [kf["frame_idx"] for kf in keyframes_meta]
    n_objects_total = sum(kf["n_objects"] for kf in keyframes_meta)
    summary = {
        "n_keyframes": len(keyframes_meta),
        "n_objects": n_objects_total,
        "frame_indices": frame_indices,
    }
    print(f"[phase1] CV-keyframe prompts written to {out_prompts_dir}")
    return summary
