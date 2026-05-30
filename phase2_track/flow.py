"""Frame-level preprocessing, segmentation, and optical-flow tracking.

Translated from src/bubble_tracking/flow_tracking.py. Drops:
- automatrix preprocessing mode (segment_frame_automatrix, the automatrix
  branch in preprocess_frame_with_debug)
- the unused `compact_max_area` / edge-gate / persistence paths if they
  are not exercised by the v1 publishing pipeline (audit during
  translation; preserve any path reachable from segment_frame with
  default SegmentationConfig).

C.5 covers preprocessing + segmentation. C.6 appends the optical-flow +
tracking loop to the bottom of this file.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .schemas import (
    BoundingBox,
    FrameData,
    TrackedObject,
    TrackingResult,
    PreprocessConfig,
    SegmentationConfig,
    FlowConfig,
    MatchConfig,
    PersistenceConfig,
    DebounceConfig,
    TrackingConfig,
)


@dataclass
class Component:
    label: int
    area: int
    centroid: Tuple[float, float]
    bbox: BoundingBox


# ---------------------------------------------------------------------------
# ROI + index helpers
# ---------------------------------------------------------------------------


def load_roi_json(path: Optional[str]) -> Optional[dict]:
    if not path:
        return None
    roi_path = Path(path)
    data = json.loads(roi_path.read_text())
    return {
        "x1": int(data["x1"]),
        "y1": int(data["y1"]),
        "x2": int(data["x2"]),
        "y2": int(data["y2"]),
    }


def _ensure_odd(value: int, minimum: int = 1) -> int:
    value = max(value, minimum)
    return value if value % 2 == 1 else value + 1


def _apply_roi(frame: np.ndarray, roi: Optional[dict]) -> Tuple[np.ndarray, Tuple[int, int]]:
    if roi is None:
        return frame, (0, 0)
    x1, y1, x2, y2 = roi["x1"], roi["y1"], roi["x2"], roi["y2"]
    return frame[y1:y2, x1:x2], (x1, y1)


def _frame_index_from_path(path: str) -> Optional[int]:
    match = re.search(r"(\d+)\.(?:jpg|jpeg|png)$", Path(path).name, re.IGNORECASE)
    if not match:
        return None
    return int(match.group(1))


def _build_index_map(paths: List[str], label: str) -> Dict[int, str]:
    index_map: Dict[int, str] = {}
    for path in paths:
        idx = _frame_index_from_path(path)
        if idx is None:
            raise ValueError(f"Failed to parse {label} index from {path}")
        if idx in index_map:
            raise ValueError(f"Duplicate {label} index {idx} for {path}")
        index_map[idx] = path
    return index_map


def _prepare_mask_labels(
    mask: np.ndarray,
    frame_shape: Tuple[int, int],
    roi: Optional[dict],
    min_area: int,
) -> Tuple[np.ndarray, Dict[int, Component]]:
    if mask.ndim == 3:
        mask = mask[:, :, 0]

    if roi:
        roi_h = roi["y2"] - roi["y1"]
        roi_w = roi["x2"] - roi["x1"]
        if mask.shape[:2] == frame_shape:
            mask_crop, _ = _apply_roi(mask, roi)
        elif mask.shape[:2] == (roi_h, roi_w):
            mask_crop = mask
        else:
            raise ValueError(
                f"Mask shape {mask.shape[:2]} does not match frame {frame_shape} or ROI {(roi_h, roi_w)}"
            )
    else:
        if mask.shape[:2] != frame_shape:
            raise ValueError(f"Mask shape {mask.shape[:2]} does not match frame {frame_shape}")
        mask_crop = mask

    binary = (mask_crop > 0).astype(np.uint8)
    return _label_components(binary, min_area)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


def _estimate_light_map(gray: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    h, w = gray.shape
    if cfg.grid_rows <= 0 or cfg.grid_cols <= 0:
        return np.ones((h, w), dtype=np.float32)
    cell_h = h // cfg.grid_rows
    cell_w = w // cfg.grid_cols
    if cell_h == 0 or cell_w == 0:
        return np.ones((h, w), dtype=np.float32)

    light_values = np.full((cfg.grid_rows, cfg.grid_cols), np.nan, dtype=np.float32)
    valid = np.zeros((cfg.grid_rows, cfg.grid_cols), dtype=bool)

    for r in range(cfg.grid_rows):
        for c in range(cfg.grid_cols):
            y0, y1 = r * cell_h, (r + 1) * cell_h
            x0, x1 = c * cell_w, (c + 1) * cell_w
            cell = gray[y0:y1, x0:x1].reshape(-1)
            if cell.size == 0:
                continue
            thresh = np.percentile(cell, cfg.light_percentile)
            bright = cell[cell >= thresh]
            if bright.size >= cfg.min_bright_pixels:
                light_values[r, c] = float(np.mean(bright))
                valid[r, c] = True

    if np.all(np.isnan(light_values)):
        return np.ones((h, w), dtype=np.float32)

    global_avg = float(np.nanmean(light_values))
    for r in range(cfg.grid_rows):
        for c in range(cfg.grid_cols):
            if valid[r, c] and abs(light_values[r, c] - global_avg) >= cfg.light_tolerance:
                light_values[r, c] = np.nan
                valid[r, c] = False

    def _neighbor_avg(rr: int, cc: int) -> float:
        neighbors = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                r2, c2 = rr + dr, cc + dc
                if 0 <= r2 < cfg.grid_rows and 0 <= c2 < cfg.grid_cols:
                    val = light_values[r2, c2]
                    if not np.isnan(val):
                        neighbors.append(float(val))
        return float(np.mean(neighbors)) if neighbors else global_avg

    for r in range(cfg.grid_rows):
        for c in range(cfg.grid_cols):
            if np.isnan(light_values[r, c]):
                light_values[r, c] = _neighbor_avg(r, c)

    light_values = np.nan_to_num(light_values, nan=global_avg)
    light_full = cv2.resize(
        light_values,
        (w, h),
        interpolation=cv2.INTER_CUBIC,
    ).astype(np.float32)
    sigma = max(1.0, min(cell_h, cell_w) / 2.0)
    light_smooth = cv2.GaussianBlur(light_full, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return light_smooth


def preprocess_frame(
    frame_bgr: np.ndarray,
    reference_bgr: np.ndarray,
    cfg: PreprocessConfig,
) -> np.ndarray:
    return preprocess_frame_with_debug(frame_bgr, reference_bgr, cfg)["normalized"]


def preprocess_frame_with_debug(
    frame_bgr: np.ndarray,
    reference_bgr: np.ndarray,
    cfg: PreprocessConfig,
) -> dict:
    mode = (cfg.mode or "default").lower()
    if mode == "raw":
        return _preprocess_frame_raw_with_debug(frame_bgr, reference_bgr, cfg)
    if mode == "automatrix":
        raise ValueError(
            "automatrix preprocessing mode has been removed in phase2_track. "
            "Use 'default' or 'raw'."
        )
    if mode != "default":
        raise ValueError(f"Unknown preprocess mode: {cfg.mode}")
    return _preprocess_frame_default_with_debug(frame_bgr, reference_bgr, cfg)


def _preprocess_frame_default_with_debug(
    frame_bgr: np.ndarray,
    reference_bgr: np.ndarray,
    cfg: PreprocessConfig,
) -> dict:
    diff = frame_bgr.astype(np.float32) - reference_bgr.astype(np.float32)
    if cfg.diff_sigma > 0:
        diff = cv2.GaussianBlur(diff, (0, 0), sigmaX=cfg.diff_sigma, sigmaY=cfg.diff_sigma)

    clip_min, clip_max = np.percentile(diff, cfg.clip_percentiles)
    if clip_max <= clip_min:
        clip_max = clip_min + 1.0
    diff = np.clip(diff, clip_min, clip_max)
    diff = (diff - clip_min) / (clip_max - clip_min)
    adjusted = (diff * 255).astype(np.uint8)
    gray = cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY)

    if cfg.use_clahe:
        clahe = cv2.createCLAHE(
            clipLimit=cfg.clahe_clip_limit,
            tileGridSize=cfg.clahe_tile_grid,
        )
        gray = clahe.apply(gray)

    light_map = _estimate_light_map(gray, cfg)
    normalized = gray.astype(np.float32) / (light_map + 1e-5)
    norm_min, norm_max = float(np.min(normalized)), float(np.max(normalized))
    if norm_max <= norm_min:
        norm_max = norm_min + 1.0
    normalized_scaled = (255 * (normalized - norm_min) / (norm_max - norm_min)).astype(np.uint8)
    return {
        "normalized": normalized_scaled,
        "diff": diff,
        "adjusted": adjusted,
        "light_map": light_map,
    }


def _preprocess_frame_raw_with_debug(
    frame_bgr: np.ndarray,
    reference_bgr: np.ndarray,
    cfg: PreprocessConfig,
) -> dict:
    # Minimal preprocessing for flow: raw frame-reference difference without light-map normalization.
    diff = frame_bgr.astype(np.float32) - reference_bgr.astype(np.float32)
    if cfg.raw_sigma > 0:
        diff = cv2.GaussianBlur(diff, (0, 0), sigmaX=cfg.raw_sigma, sigmaY=cfg.raw_sigma)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    normalized = np.clip(diff_gray + 128.0, 0, 255).astype(np.uint8)
    return {
        "normalized": normalized,
        "diff": diff,
        "diff_gray": diff_gray,
    }


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------


def _label_components(binary: np.ndarray, min_area: int) -> Tuple[np.ndarray, Dict[int, Component]]:
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return np.zeros_like(labels, dtype=np.int32), {}

    filtered = np.zeros_like(labels, dtype=np.int32)
    components: Dict[int, Component] = {}
    new_id = 1

    for old_id in range(1, num_labels):
        area = int(stats[old_id, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x = int(stats[old_id, cv2.CC_STAT_LEFT])
        y = int(stats[old_id, cv2.CC_STAT_TOP])
        w = int(stats[old_id, cv2.CC_STAT_WIDTH])
        h = int(stats[old_id, cv2.CC_STAT_HEIGHT])
        bbox = BoundingBox(x_min=x, y_min=y, x_max=x + w - 1, y_max=y + h - 1)
        centroid = (float(centroids[old_id][0]), float(centroids[old_id][1]))
        filtered[labels == old_id] = new_id
        components[new_id] = Component(label=new_id, area=area, centroid=centroid, bbox=bbox)
        new_id += 1

    return filtered, components


def segment_frame(
    gray: np.ndarray,
    cfg: SegmentationConfig,
) -> Tuple[np.ndarray, Dict[int, Component]]:
    blur_ksize = _ensure_odd(cfg.blur_ksize, 3)
    blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    hist_bins = max(1, int(cfg.hist_bins))
    hist = cv2.calcHist([blurred], [0], None, [hist_bins], [0, 256])
    hist_vals = hist.reshape(-1)
    bin_edges = np.linspace(0, 256, hist_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    peak_val = int(bin_centers[int(np.argmax(hist_vals))])
    thresh = min(max(0, peak_val - cfg.threshold_offset), cfg.max_threshold)

    binary = (blurred < thresh).astype(np.uint8)
    if cfg.opening_radius > 0:
        k = cfg.opening_radius * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    labels, components = _label_components(binary, cfg.min_area)

    if cfg.median_ksize > 1:
        median_ksize = _ensure_odd(cfg.median_ksize, 3)
        smoothed = cv2.medianBlur((labels > 0).astype(np.uint8) * 255, median_ksize)
        labels, components = _label_components((smoothed > 127).astype(np.uint8), cfg.min_area)

    labels, components = _apply_compact_filter(labels, components, blurred, thresh, cfg)
    return labels, components


def _apply_edge_gate(
    labels: np.ndarray,
    components: Dict[int, Component],
    gray: np.ndarray,
    cfg: SegmentationConfig,
) -> Tuple[np.ndarray, Dict[int, Component]]:
    if cfg.edge_min_strength <= 0:
        return labels, components
    if not components:
        return labels, components

    radius = int(cfg.edge_boundary_radius)
    if radius < 0:
        radius = 0
    kernel = None
    if radius > 0:
        ksize = radius * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))

    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(grad_x, grad_y)

    keep_labels: List[int] = []
    for label_id, comp in components.items():
        mask = (labels == label_id).astype(np.uint8)
        if radius > 0 and kernel is not None:
            dilated = cv2.dilate(mask, kernel)
            eroded = cv2.erode(mask, kernel)
            ring = (dilated > 0) & (eroded == 0)
        else:
            ring = mask > 0
        if not np.any(ring):
            keep_labels.append(label_id)
            continue

        values = grad_mag[ring]
        strength = float(np.median(values))
        if strength < cfg.edge_min_strength:
            continue
        if cfg.edge_min_fraction > 0:
            fraction = float(np.mean(values >= cfg.edge_min_strength))
            if fraction < cfg.edge_min_fraction:
                continue
        keep_labels.append(label_id)

    if not keep_labels:
        return np.zeros_like(labels, dtype=np.int32), {}

    keep_mask = np.isin(labels, keep_labels).astype(np.uint8)
    return _label_components(keep_mask, cfg.min_area)


def _apply_compact_filter(
    labels: np.ndarray,
    components: Dict[int, Component],
    intensity: np.ndarray,
    thresh: int,
    cfg: SegmentationConfig,
) -> Tuple[np.ndarray, Dict[int, Component]]:
    if cfg.compact_max_area <= 0 or cfg.compact_min_dark_delta <= 0:
        return labels, components

    dark_cutoff = max(0.0, float(thresh) - float(cfg.compact_min_dark_delta))
    to_drop: List[int] = []

    for label_id, comp in components.items():
        if comp.area > cfg.compact_max_area:
            continue
        mask = labels == label_id
        if not np.any(mask):
            continue
        median_val = float(np.median(intensity[mask]))
        if median_val >= dark_cutoff:
            to_drop.append(label_id)

    if not to_drop:
        return labels, components

    filtered = labels.copy()
    filtered[np.isin(filtered, to_drop)] = 0
    return _label_components((filtered > 0).astype(np.uint8), cfg.min_area)


def _confirm_persistent_components(
    prev_labels: np.ndarray,
    prev_components: Dict[int, Component],
    prev_confirmed_labels: Optional[np.ndarray],
    flow_prev_confirmed_to_prev: Optional[np.ndarray],
    next_labels: Optional[np.ndarray],
    flow_prev_to_next: Optional[np.ndarray],
    cfg: PersistenceConfig,
    min_area: int,
) -> Tuple[np.ndarray, Dict[int, Component]]:
    if cfg.min_overlap_pixels <= 0:
        return prev_labels, prev_components

    if prev_confirmed_labels is None and next_labels is None:
        return prev_labels, prev_components

    num_prev = int(prev_labels.max())
    overlap_prev_total = np.zeros(num_prev, dtype=np.int64)
    overlap_next_total = np.zeros(num_prev, dtype=np.int64)

    if (
        prev_confirmed_labels is not None
        and flow_prev_confirmed_to_prev is not None
        and num_prev > 0
        and prev_confirmed_labels.size > 0
    ):
        warped_prev_confirmed = _warp_labels(prev_confirmed_labels, flow_prev_confirmed_to_prev)
        overlap_prev = _overlap_counts(warped_prev_confirmed, prev_labels)
        if overlap_prev.size > 0:
            overlap_prev_total = overlap_prev.sum(axis=0)

    if (
        next_labels is not None
        and flow_prev_to_next is not None
        and num_prev > 0
        and next_labels.size > 0
    ):
        warped_prev = _warp_labels(prev_labels, flow_prev_to_next)
        overlap_next = _overlap_counts(warped_prev, next_labels)
        if overlap_next.size > 0:
            overlap_next_total = overlap_next.sum(axis=1)

    keep_labels: List[int] = []
    for label_id, comp in prev_components.items():
        if cfg.max_area > 0 and comp.area > cfg.max_area:
            keep_labels.append(label_id)
            continue
        idx = label_id - 1
        prev_overlap = int(overlap_prev_total[idx]) if idx < len(overlap_prev_total) else 0
        next_overlap = int(overlap_next_total[idx]) if idx < len(overlap_next_total) else 0
        if prev_overlap >= cfg.min_overlap_pixels or next_overlap >= cfg.min_overlap_pixels:
            keep_labels.append(label_id)

    if not keep_labels:
        return np.zeros_like(prev_labels, dtype=np.int32), {}

    keep_mask = np.isin(prev_labels, keep_labels).astype(np.uint8)
    return _label_components(keep_mask, min_area)


# ---------------------------------------------------------------------------
# Flow helpers (needed by _confirm_persistent_components above)
# ---------------------------------------------------------------------------


def _warp_labels(labels: np.ndarray, flow: np.ndarray) -> np.ndarray:
    h, w = labels.shape
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (grid_x + flow[..., 0]).astype(np.float32)
    map_y = (grid_y + flow[..., 1]).astype(np.float32)
    warped = cv2.remap(
        labels.astype(np.float32),
        map_x,
        map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return warped.astype(np.int32)


def _overlap_counts(prev_labels: np.ndarray, curr_labels: np.ndarray) -> np.ndarray:
    if prev_labels.shape != curr_labels.shape:
        raise ValueError("Previous and current labels must share shape for overlap analysis")

    num_prev = int(prev_labels.max())
    num_curr = int(curr_labels.max())
    if num_prev == 0 or num_curr == 0:
        return np.zeros((num_prev, num_curr), dtype=np.int64)

    prev_flat = prev_labels.astype(np.int64).ravel()
    curr_flat = curr_labels.astype(np.int64).ravel()
    stride = num_curr + 1
    counts = np.bincount(
        prev_flat * stride + curr_flat,
        minlength=(num_prev + 1) * (num_curr + 1),
    )
    table = counts.reshape(num_prev + 1, num_curr + 1)
    return table[1:, 1:]
