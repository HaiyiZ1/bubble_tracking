"""SAM2 wrapper + CV-keyframe prompt builder for Phase 2.

Translated from src/bubble_tracking/sam2_adapter.py and the
_build_cv_keyframe_mask_prompts / _build_cv_extra_keyframe_prompts portions
of scripts/run_sam2_prompted_flow.py.

Self-contained: pulls frame_io helpers inline (sorted_frame_paths,
save_mask) so Phase 2 has no cross-package imports.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from .schemas import (
    BoundingBox,
    FrameData,
    ObjectPrompt,
    PointPrompt,
    PromptSpec,
    TrackedObject,
    TrackingResult,
    SAM2_FILL_HOLE_AREA,
)

_FRAME_EXTS = (".jpg", ".jpeg", ".png")


def sorted_frame_paths(directory: str | Path) -> List[str]:
    """Inlined from src/bubble_tracking/frame_io.py."""
    d = Path(directory)
    # Note: SAM2 requires .jpg format, but we support both for loading
    pattern = re.compile(r"(\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)
    paths = []
    for fname in os.listdir(str(d)):
        match = pattern.search(fname)
        if match:
            paths.append((int(match.group(1)), str(d / fname)))
    paths.sort(key=lambda x: x[0])
    return [p[1] for p in paths]


def save_mask(mask: np.ndarray, path: str | Path) -> None:
    """Inlined from src/bubble_tracking/frame_io.py."""
    output_path = str(path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Save as 16-bit PNG to support many object IDs
    cv2.imwrite(output_path, mask.astype(np.uint16))


def compute_mask_properties(mask: np.ndarray, object_id: int) -> Optional[FrameData]:
    """
    Compute centroid, area, and bounding box for an object in a mask.

    Args:
        mask: 2D array with instance IDs
        object_id: The object ID to analyze

    Returns:
        FrameData with computed properties, or None if object not in mask
    """
    binary = (mask == object_id)
    coords = np.where(binary)

    if len(coords[0]) == 0:
        return None

    area = int(np.sum(binary))
    centroid_y = float(np.mean(coords[0]))
    centroid_x = float(np.mean(coords[1]))

    y_min, y_max = int(np.min(coords[0])), int(np.max(coords[0]))
    x_min, x_max = int(np.min(coords[1])), int(np.max(coords[1]))

    return FrameData(
        centroid=(centroid_x, centroid_y),
        area=area,
        bbox=BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max),
    )


@dataclass
class SAM2Config:
    """Configuration for SAM2 runner."""
    checkpoint_path: str
    model_cfg: str = "sam2_hiera_t.yaml"
    device: str = "cpu"
    # SAM2's video predictor fills small background holes (area <= fill_hole_area)
    # in low-res masks. Default 8 matches upstream SAM2; set to 0 to preserve
    # genuine holes (e.g. porous bubbles whose interior holes are real signal).
    fill_hole_area: int = 8
    # Push frame tensors to CPU between accesses (saves GPU memory at ~5-15% latency cost).
    # v1 default: True (offload enabled by default).
    offload_video_to_cpu: bool = True


class SAM2Runner:
    """
    Runner for SAM2 video segmentation and tracking.

    Usage:
        runner = SAM2Runner(config)
        result = runner.run(frames, prompts)
    """

    def __init__(self, config: SAM2Config):
        self.config = config
        self.predictor = None
        self._load_model()

    def _load_model(self) -> None:
        """Load SAM2 model and predictor."""
        try:
            from sam2.build_sam import build_sam2_video_predictor
        except ImportError:
            raise ImportError(
                "SAM2 not installed. Please install with:\n"
                "  git clone https://github.com/facebookresearch/segment-anything-2.git\n"
                "  cd segment-anything-2 && pip install -e ."
            )

        self.predictor = build_sam2_video_predictor(
            self.config.model_cfg,
            self.config.checkpoint_path,
            device=self.config.device,
            hydra_overrides_extra=[
                f"++model.fill_hole_area={int(self.config.fill_hole_area)}",
            ],
        )

    def run(
        self,
        frames_dir: str,
        prompts: PromptSpec,
        output_dir: Optional[str] = None,
        roi_mask: Optional[np.ndarray] = None,
    ) -> Tuple[TrackingResult, Dict[int, np.ndarray]]:
        """
        Run SAM2 video tracking on a sequence of frames.

        Args:
            frames_dir: Directory containing frame images
            prompts: Prompt specification for object initialization
            output_dir: Optional directory to save masks

        Returns:
            Tuple of (TrackingResult, dict of frame_idx -> mask)
        """
        frame_paths = sorted_frame_paths(frames_dir)
        num_frames = len(frame_paths)

        if num_frames == 0:
            raise ValueError(f"No frames found in {frames_dir}")

        # Initialize predictor state with video
        inference_state = self.predictor.init_state(
            video_path=frames_dir,
            offload_video_to_cpu=bool(self.config.offload_video_to_cpu),
        )

        # Add prompts for each object
        object_ids = set()
        for prompt in prompts.prompts:
            object_ids.add(prompt.object_id)

            has_mask = getattr(prompt, "mask", None) is not None
            has_points = bool(prompt.points)
            has_box = prompt.box is not None

            # When both mask and point/box prompts are provided on the same frame,
            # call add_new_mask first so its prediction logits seed the subsequent
            # add_new_points_or_box call via SAM2's prev_sam_mask_logits path.
            # This lets negative points (e.g., placed inside holes) refine a
            # mask seed instead of replacing it.
            if has_mask:
                self.predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=prompt.frame_idx,
                    obj_id=prompt.object_id,
                    mask=prompt.mask,
                )

            if has_points or has_box:
                points = None
                labels = None
                box = None

                if has_points:
                    points = np.array([[p.x, p.y] for p in prompt.points])
                    labels = np.array([p.label for p in prompt.points])

                if has_box:
                    box = np.array(prompt.box.bbox.to_list())

                self.predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=prompt.frame_idx,
                    obj_id=prompt.object_id,
                    points=points,
                    labels=labels,
                    box=box,
                )

        # Propagate through video
        masks_by_frame: Dict[int, np.ndarray] = {}
        objects: Dict[int, TrackedObject] = {
            obj_id: TrackedObject(object_id=obj_id) for obj_id in object_ids
        }

        # Forward propagation
        for frame_idx, object_ids_out, masks_out in self.predictor.propagate_in_video(
            inference_state
        ):
            if torch.is_tensor(object_ids_out):
                object_ids_out = object_ids_out.tolist()
            if torch.is_tensor(masks_out):
                masks_out = masks_out.detach().cpu().numpy()
            # Combine masks into single instance mask
            combined_mask = np.zeros(masks_out.shape[-2:], dtype=np.int32)

            for i, obj_id in enumerate(object_ids_out):
                # masks_out shape: (num_objects, 1, H, W)
                obj_mask = masks_out[i, 0] > 0.0
                combined_mask[obj_mask] = obj_id

            if roi_mask is not None:
                if roi_mask.shape != combined_mask.shape:
                    raise ValueError("ROI mask size does not match mask output size")
                combined_mask[roi_mask == 0] = 0

            # Compute per-object properties from the final per-frame mask.
            for obj_id in object_ids_out:
                obj_id = int(obj_id)
                frame_data = compute_mask_properties(combined_mask, obj_id)
                if frame_data and obj_id in objects:
                    objects[obj_id].frames[frame_idx] = frame_data

            masks_by_frame[frame_idx] = combined_mask

        # Save masks if output directory specified
        if output_dir:
            masks_dir = os.path.join(output_dir, "masks")
            os.makedirs(masks_dir, exist_ok=True)
            for frame_idx, mask in masks_by_frame.items():
                save_mask(mask, os.path.join(masks_dir, f"{frame_idx:06d}.png"))

        # Build result
        result = TrackingResult(
            num_frames=num_frames,
            num_objects=len(objects),
            created_at=datetime.now(),
            objects=objects,
        )

        return result, masks_by_frame

    def reset(self) -> None:
        """Reset predictor state for a new video."""
        # SAM2 predictor state is managed per-video, no explicit reset needed
        pass


def create_runner(
    checkpoint_path: str,
    model_cfg: str = "sam2_hiera_t.yaml",
    device: Optional[str] = None,
    fill_hole_area: int = 8,
    offload_video_to_cpu: bool = True,
) -> SAM2Runner:
    """
    Factory function to create a SAM2 runner.

    Args:
        checkpoint_path: Path to SAM2 checkpoint file
        model_cfg: SAM2 model configuration name
        device: Device to use ("cpu", "cuda", or None for auto-detect)

    Returns:
        Configured SAM2Runner instance
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    config = SAM2Config(
        checkpoint_path=checkpoint_path,
        model_cfg=model_cfg,
        device=device,
        fill_hole_area=fill_hole_area,
        offload_video_to_cpu=offload_video_to_cpu,
    )
    return SAM2Runner(config)


# ---------------------------------------------------------------------------
# CV-keyframe prompt builder.
# Translated from scripts/run_sam2_prompted_flow.py:
#   _build_cv_keyframe_mask_prompts
#   _build_cv_extra_keyframe_prompts
#   _save_cv_keyframe_debug_artifacts
#   _negative_points_from_holes
# Notable changes vs original:
# - `disable_light_map` parameter replaced with `normalize_lightmap` (default
#   False per v1 spec). Conditionals are flipped accordingly.
# - automatrix preprocess branch dropped (v1 supports default + raw only).
# - preprocess_frame_with_debug imported lazily from .flow (built in C.5).
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
    from .flow import preprocess_frame_with_debug  # lazy: phase2_track.flow built in C.5

    from .schemas import PreprocessConfig, SegmentationConfig
    # These are also importable from the top-level .schemas but we re-import
    # here for clarity in case the caller already did so elsewhere.

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
                from .flow import preprocess_frame  # lazy: phase2_track.flow built in C.5
                gray = preprocess_frame(frame_crop, ref_crop, preprocess_cfg)

            from .flow import segment_frame  # lazy: phase2_track.flow built in C.5
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
    from .flow import preprocess_frame_with_debug  # lazy: phase2_track.flow built in C.5

    from .schemas import PreprocessConfig, SegmentationConfig

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
            from .flow import preprocess_frame  # lazy: phase2_track.flow built in C.5
            gray = preprocess_frame(frame_crop, ref_crop, preprocess_cfg)

        from .flow import segment_frame  # lazy: phase2_track.flow built in C.5
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
