"""Threshold + connected-component segmentation shared across phases.

Extracted from phase2_track/flow.py (segmentation section).

SegmentationConfig and Component are imported from .schemas (canonical home).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .schemas import BoundingBox, Component, SegmentationConfig


def _ensure_odd(value: int, minimum: int = 1) -> int:
    value = max(value, minimum)
    return value if value % 2 == 1 else value + 1


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
    cfg,  # PersistenceConfig — not imported here to avoid circular deps
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
