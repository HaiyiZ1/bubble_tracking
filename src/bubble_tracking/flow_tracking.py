"""
Flow-guided tracking for bubble masks derived from a zero-bubble reference.
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

from .frame_io import load_mask, save_mask, sorted_frame_paths
from .mask_events import save_area_timeseries_csv
from .schemas import BoundingBox, FrameData, TrackedObject, TrackingResult
from .track_builder import save_tracks


@dataclass
class PreprocessConfig:
    diff_sigma: float = 0.5
    clip_percentiles: Tuple[float, float] = (1.0, 99.0)
    grid_rows: int = 8
    grid_cols: int = 5
    light_percentile: float = 95.0
    min_bright_pixels: int = 10
    light_tolerance: float = 10.0
    use_clahe: bool = False
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: Tuple[int, int] = (8, 8)
    mode: str = "default"


@dataclass
class SegmentationConfig:
    blur_ksize: int = 3
    hist_bins: int = 35
    threshold_offset: int = 7
    max_threshold: int = 248
    opening_radius: int = 4
    min_area: int = 500
    median_ksize: int = 25
    compact_max_area: int = 0
    compact_min_dark_delta: float = 0.0
    automatrix_median_radius: int = 12
    edge_min_strength: float = 0.0
    edge_boundary_radius: int = 1
    edge_min_fraction: float = 0.0


@dataclass
class FlowConfig:
    pyr_scale: float = 0.5
    levels: int = 3
    winsize: int = 15
    iterations: int = 3
    poly_n: int = 5
    poly_sigma: float = 1.2
    fast_pyramids: bool = False
    flags: int = 0


@dataclass
class MatchConfig:
    min_overlap_pixels: int = 10
    min_link_iou: float = 0.05
    min_link_overlap_ratio: float = 0.25


@dataclass
class PersistenceConfig:
    min_overlap_pixels: int = 0
    max_area: int = 0


@dataclass
class TrackingConfig:
    fps: float = 4.0
    mm_per_pixel: Optional[float] = None
    flow_backend: str = "auto"
    cuda_device: int = 0
    log_every: int = 10
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    segment: SegmentationConfig = field(default_factory=SegmentationConfig)
    flow: FlowConfig = field(default_factory=FlowConfig)
    match: MatchConfig = field(default_factory=MatchConfig)
    persistence: PersistenceConfig = field(default_factory=PersistenceConfig)


@dataclass
class Component:
    label: int
    area: int
    centroid: Tuple[float, float]
    bbox: BoundingBox


class FlowEngine:
    def __init__(self, cfg: TrackingConfig):
        self.cfg = cfg
        self.backend = "cpu"
        self._warned = False
        self._cuda_flow = None
        self._use_cuda = False
        self._init_backend()

    def _warn_once(self, message: str) -> None:
        if not self._warned:
            print(f"[flow] {message}", file=sys.stderr)
            self._warned = True

    def _init_backend(self) -> None:
        requested = (self.cfg.flow_backend or "auto").lower()
        if requested not in ("auto", "cpu"):
            raise ValueError("flow_backend must be 'auto' or 'cpu'")
        if requested == "cpu":
            self.backend = "cpu"
            return

        if not hasattr(cv2, "cuda") or not hasattr(cv2, "cuda_FarnebackOpticalFlow"):
            self._warn_once("OpenCV CUDA Farneback not available; falling back to CPU.")
            return
        if not hasattr(cv2.cuda, "getCudaEnabledDeviceCount"):
            self._warn_once("OpenCV CUDA support not detected; falling back to CPU.")
            return

        try:
            count = int(cv2.cuda.getCudaEnabledDeviceCount())
        except cv2.error:
            count = 0
        if count <= 0:
            self._warn_once("No CUDA devices found; falling back to CPU.")
            return

        try:
            cv2.cuda.setDevice(int(self.cfg.cuda_device))
            create_fn = getattr(cv2.cuda_FarnebackOpticalFlow, "create", None)
            if create_fn is None:
                self._warn_once("CUDA Farneback factory not available; falling back to CPU.")
                return
            try:
                self._cuda_flow = create_fn(
                    self.cfg.flow.levels,
                    self.cfg.flow.pyr_scale,
                    self.cfg.flow.fast_pyramids,
                    self.cfg.flow.winsize,
                    self.cfg.flow.iterations,
                    self.cfg.flow.poly_n,
                    self.cfg.flow.poly_sigma,
                    self.cfg.flow.flags,
                )
            except cv2.error:
                self._cuda_flow = create_fn(
                    self.cfg.flow.levels,
                    self.cfg.flow.pyr_scale,
                    self.cfg.flow.winsize,
                    self.cfg.flow.iterations,
                    self.cfg.flow.poly_n,
                    self.cfg.flow.poly_sigma,
                    self.cfg.flow.flags,
                )
            self.backend = "cuda"
            self._use_cuda = True
        except cv2.error:
            self._warn_once("Failed to initialize CUDA Farneback; falling back to CPU.")
            self._cuda_flow = None
            self._use_cuda = False
            self.backend = "cpu"

    def compute(self, prev_gray: np.ndarray, curr_gray: np.ndarray) -> np.ndarray:
        if self._use_cuda and self._cuda_flow is not None:
            try:
                prev_gpu = cv2.cuda_GpuMat()
                curr_gpu = cv2.cuda_GpuMat()
                prev_gpu.upload(prev_gray)
                curr_gpu.upload(curr_gray)
                flow_gpu = self._cuda_flow.calc(prev_gpu, curr_gpu, None)
                return flow_gpu.download()
            except cv2.error:
                self._warn_once("CUDA flow failed; falling back to CPU.")
                self._use_cuda = False
                self.backend = "cpu"

        return cv2.calcOpticalFlowFarneback(
            prev_gray,
            curr_gray,
            None,
            self.cfg.flow.pyr_scale,
            self.cfg.flow.levels,
            self.cfg.flow.winsize,
            self.cfg.flow.iterations,
            self.cfg.flow.poly_n,
            self.cfg.flow.poly_sigma,
            self.cfg.flow.flags,
        )


def _should_log_progress(frame_idx: int, total_frames: int, log_every: int) -> bool:
    if log_every <= 0:
        return False
    if frame_idx == total_frames - 1:
        return True
    return (frame_idx % log_every) == 0


def _log_progress(frame_idx: int, total_frames: int, stage: str, log_every: int) -> None:
    if _should_log_progress(frame_idx, total_frames, log_every):
        print(f"[progress] stage={stage} frame {frame_idx + 1}/{total_frames}")


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
    if mode == "automatrix":
        return _preprocess_frame_automatrix_with_debug(frame_bgr, reference_bgr, cfg)
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


def _preprocess_frame_automatrix_with_debug(
    frame_bgr: np.ndarray,
    reference_bgr: np.ndarray,
    cfg: PreprocessConfig,
) -> dict:
    try:
        from scipy.interpolate import griddata
        from scipy.ndimage import gaussian_filter, zoom
    except Exception as exc:  # pragma: no cover - dependency check
        raise ImportError(
            "Automatrix mode requires scipy (gaussian_filter, griddata, zoom)."
        ) from exc

    diff = frame_bgr.astype(np.float32) - reference_bgr.astype(np.float32)
    if cfg.diff_sigma > 0:
        diff = gaussian_filter(diff, sigma=cfg.diff_sigma)

    clip_min, clip_max = np.percentile(diff, cfg.clip_percentiles)
    if clip_max <= clip_min:
        clip_max = clip_min + 1.0
    diff = np.clip(diff, clip_min, clip_max)
    diff = (diff - clip_min) / (clip_max - clip_min)
    adjusted = (diff * 255).astype(np.uint8)
    gray = cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY).astype(np.float32)

    grid_rows = cfg.grid_rows
    grid_cols = cfg.grid_cols
    percentile = cfg.light_percentile
    min_bright_pixel_count = cfg.min_bright_pixels
    tolerance = cfg.light_tolerance
    h, w = gray.shape
    cell_h = h // grid_rows if grid_rows > 0 else 0
    cell_w = w // grid_cols if grid_cols > 0 else 0
    light_values = np.full((grid_rows, grid_cols), np.nan)
    valid_mask = np.zeros((grid_rows, grid_cols), dtype=bool)

    for r in range(grid_rows):
        for c in range(grid_cols):
            y0, y1 = r * cell_h, (r + 1) * cell_h
            x0, x1 = c * cell_w, (c + 1) * cell_w
            cell_gray = gray[y0:y1, x0:x1].flatten()
            if cell_gray.size == 0:
                continue
            thresh = np.percentile(cell_gray, percentile)
            bright_pixels = cell_gray[cell_gray >= thresh]
            if len(bright_pixels) >= min_bright_pixel_count:
                light_values[r, c] = float(np.mean(bright_pixels))
                valid_mask[r, c] = True

    global_avg = float(np.nanmean(light_values))
    for r in range(grid_rows):
        for c in range(grid_cols):
            if valid_mask[r, c] and abs(light_values[r, c] - global_avg) >= tolerance:
                valid_mask[r, c] = False
                light_values[r, c] = np.nan

    def _average_neighbors(rr: int, cc: int) -> float:
        neighbors = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                r2, c2 = rr + dr, cc + dc
                if 0 <= r2 < grid_rows and 0 <= c2 < grid_cols:
                    val = light_values[r2, c2]
                    if not np.isnan(val):
                        neighbors.append(float(val))
        return float(np.mean(neighbors)) if neighbors else float("nan")

    for r in range(grid_rows):
        for c in range(grid_cols):
            if not valid_mask[r, c]:
                avg_val = _average_neighbors(r, c)
                light_values[r, c] = avg_val

    yy, xx = np.mgrid[0:grid_rows, 0:grid_cols]
    points = np.array(
        [
            (r, c)
            for r in range(grid_rows)
            for c in range(grid_cols)
            if not np.isnan(light_values[r, c])
        ]
    )
    values = np.array([light_values[r, c] for r, c in points])
    filled = griddata(points, values, (yy, xx), method="cubic")
    filled = np.nan_to_num(filled, nan=np.nanmean(filled))

    zoom_factors = (h / grid_rows, w / grid_cols)
    light_full = zoom(filled, zoom_factors, order=3)
    light_full = light_full[:h, :w]
    light_smooth = gaussian_filter(light_full, sigma=cell_h / 2)

    normalized = gray / (light_smooth + 1e-5)
    norm_min, norm_max = float(normalized.min()), float(normalized.max())
    if norm_max <= norm_min:
        norm_max = norm_min + 1.0
    normalized_scaled = (255 * (normalized - norm_min) / (norm_max - norm_min)).astype(np.uint8)
    return {
        "normalized": normalized_scaled,
        "diff": diff,
        "adjusted": adjusted,
        "light_map": light_smooth,
    }


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


def segment_frame_automatrix(
    gray: np.ndarray,
    cfg: SegmentationConfig,
) -> Tuple[np.ndarray, Dict[int, Component]]:
    try:
        from skimage.filters import median as sk_median
        from skimage.measure import label as sk_label
        from skimage.morphology import disk, opening, remove_small_objects
    except Exception as exc:  # pragma: no cover - dependency check
        raise ImportError(
            "Automatrix mode requires scikit-image (median, label, remove_small_objects)."
        ) from exc

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    hist_vals, bin_edges = np.histogram(blurred, bins=cfg.hist_bins, range=(0, 255))
    bin_centers = bin_edges[:-1]
    peak_val = bin_centers[int(np.argmax(hist_vals))]
    thresh = min(peak_val - cfg.threshold_offset, cfg.max_threshold)

    binary_mask = blurred < thresh
    _ = opening(binary_mask, disk(cfg.opening_radius))
    cleaned = remove_small_objects(sk_label(binary_mask), min_size=cfg.min_area) > 0
    median_filtered = sk_median(cleaned, disk(cfg.automatrix_median_radius))
    binary_final = (median_filtered > 0).astype(np.uint8)
    return _label_components(binary_final, 1)


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


def _build_links(
    overlap: np.ndarray,
    prev_areas: Dict[int, int],
    curr_areas: Dict[int, int],
    cfg: MatchConfig,
) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    parents_to_children = {pid: [] for pid in prev_areas}
    children_to_parents = {cid: [] for cid in curr_areas}

    if overlap.size == 0:
        return parents_to_children, children_to_parents

    rows, cols = np.nonzero(overlap >= cfg.min_overlap_pixels)
    for row, col in zip(rows.tolist(), cols.tolist()):
        parent_id = row + 1
        child_id = col + 1
        inter = int(overlap[row, col])
        parent_area = prev_areas.get(parent_id, 0)
        child_area = curr_areas.get(child_id, 0)
        if parent_area <= 0 or child_area <= 0:
            continue

        union = parent_area + child_area - inter
        iou = float(inter / union) if union > 0 else 0.0
        overlap_ratio = float(inter / min(parent_area, child_area))

        if iou >= cfg.min_link_iou or overlap_ratio >= cfg.min_link_overlap_ratio:
            parents_to_children[parent_id].append(child_id)
            children_to_parents[child_id].append(parent_id)

    return parents_to_children, children_to_parents


def _update_prev_velocities(
    prev_labels: np.ndarray,
    prev_id_map: Dict[int, int],
    flow: np.ndarray,
    result: TrackingResult,
    frame_idx: int,
    fps: float,
) -> None:
    for label_id, obj_id in prev_id_map.items():
        mask = prev_labels == label_id
        if not np.any(mask):
            continue
        values = flow[mask]
        median_dx = float(np.median(values[:, 0]))
        median_dy = float(np.median(values[:, 1]))
        vx = median_dx * fps
        vy = median_dy * fps
        frame_data = result.objects[obj_id].frames.get(frame_idx)
        if frame_data:
            frame_data.velocity = (vx, vy)


def save_tracks_csv(
    result: TrackingResult,
    output_path: str,
    fps: float,
    mm_per_pixel: Optional[float] = None,
) -> None:
    fields = [
        "object_id",
        "frame_idx",
        "time_s",
        "centroid_x",
        "centroid_y",
        "area_px",
        "area_mm2",
        "velocity_x_px_s",
        "velocity_y_px_s",
        "velocity_mag_px_s",
        "velocity_x_mm_s",
        "velocity_y_mm_s",
        "velocity_mag_mm_s",
    ]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    def _fmt(value: Optional[float]) -> str:
        if value is None:
            return ""
        return f"{value:.6f}"

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for obj_id, obj in result.objects.items():
            for frame_idx, frame in sorted(obj.frames.items()):
                time_s = frame_idx / fps if fps > 0 else 0.0
                area_mm2 = None
                if mm_per_pixel is not None:
                    area_mm2 = frame.area * (mm_per_pixel ** 2)
                vx_px, vy_px = (frame.velocity if frame.velocity is not None else (None, None))
                vel_mag_px = None
                if vx_px is not None and vy_px is not None:
                    vel_mag_px = float(np.hypot(vx_px, vy_px))
                vx_mm = vy_mm = vel_mag_mm = None
                if mm_per_pixel is not None and vx_px is not None and vy_px is not None:
                    vx_mm = vx_px * mm_per_pixel
                    vy_mm = vy_px * mm_per_pixel
                    vel_mag_mm = float(np.hypot(vx_mm, vy_mm))
                writer.writerow({
                    "object_id": obj_id,
                    "frame_idx": frame_idx,
                    "time_s": _fmt(time_s),
                    "centroid_x": _fmt(frame.centroid[0]),
                    "centroid_y": _fmt(frame.centroid[1]),
                    "area_px": frame.area,
                    "area_mm2": _fmt(area_mm2),
                    "velocity_x_px_s": _fmt(vx_px),
                    "velocity_y_px_s": _fmt(vy_px),
                    "velocity_mag_px_s": _fmt(vel_mag_px),
                    "velocity_x_mm_s": _fmt(vx_mm),
                    "velocity_y_mm_s": _fmt(vy_mm),
                    "velocity_mag_mm_s": _fmt(vel_mag_mm),
                })


def track_frames(
    frames_dir: str,
    reference_path: str,
    output_dir: str,
    masks_dir: Optional[str] = None,
    roi: Optional[dict] = None,
    config: Optional[TrackingConfig] = None,
) -> dict:
    cfg = config or TrackingConfig()
    frame_paths = sorted_frame_paths(frames_dir)
    if not frame_paths:
        raise ValueError(f"No frames found in {frames_dir}")
    total_frames = len(frame_paths)
    frame_indices: List[int] = []
    for path in frame_paths:
        idx = _frame_index_from_path(path)
        if idx is None:
            raise ValueError(f"Failed to parse frame index from {path}")
        frame_indices.append(idx)

    mask_index_map: Optional[Dict[int, str]] = None
    if masks_dir:
        mask_paths = sorted_frame_paths(masks_dir)
        if not mask_paths:
            raise ValueError(f"No masks found in {masks_dir}")
        mask_index_map = _build_index_map(mask_paths, "mask")
        missing = [idx for idx in frame_indices if idx not in mask_index_map]
        extra = [idx for idx in mask_index_map if idx not in frame_indices]
        if missing or extra:
            raise ValueError(
                "Mask/frame index mismatch. "
                f"missing_masks={missing[:5]} extra_masks={extra[:5]}"
            )

    reference_full = cv2.imread(reference_path)
    if reference_full is None:
        raise ValueError(f"Failed to read reference image: {reference_path}")

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    masks_dir = output_root / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    result = TrackingResult(
        num_frames=total_frames,
        num_objects=0,
        created_at=datetime.now(),
        objects={},
    )
    events: List[dict] = []
    area_timeseries: List[dict] = []

    frame_shape = reference_full.shape[:2]
    roi_offset = (roi["x1"], roi["y1"]) if roi else (0, 0)

    prev_raw_gray = None
    prev_raw_labels = None
    prev_raw_components: Dict[int, Component] = {}
    prev_raw_frame_idx = None

    prev_confirmed_labels = None
    prev_confirmed_components: Dict[int, Component] = {}
    prev_confirmed_frame_idx = None
    prev_id_map: Dict[int, int] = {}
    prev_flow = None
    next_id = 1
    flow_engine = FlowEngine(cfg)

    for frame_idx, frame_path in enumerate(frame_paths):
        frame_full = cv2.imread(frame_path)
        if frame_full is None:
            raise ValueError(f"Failed to read frame: {frame_path}")

        if frame_full.shape[:2] != reference_full.shape[:2]:
            raise ValueError("Frame and reference image sizes do not match")

        frame_crop, _ = _apply_roi(frame_full, roi)
        ref_crop, _ = _apply_roi(reference_full, roi)

        gray = preprocess_frame(frame_crop, ref_crop, cfg.preprocess)
        if mask_index_map is not None:
            mask_path = mask_index_map[frame_indices[frame_idx]]
            raw_mask = load_mask(mask_path)
            raw_labels, raw_components = _prepare_mask_labels(
                raw_mask,
                frame_shape,
                roi,
                cfg.segment.min_area,
            )
        else:
            if (cfg.preprocess.mode or "default").lower() == "automatrix":
                raw_labels, raw_components = segment_frame_automatrix(gray, cfg.segment)
            else:
                raw_labels, raw_components = segment_frame(gray, cfg.segment)
        raw_labels, raw_components = _apply_edge_gate(raw_labels, raw_components, gray, cfg.segment)

        if prev_raw_labels is None:
            prev_raw_gray = gray
            prev_raw_labels = raw_labels
            prev_raw_components = raw_components
            prev_raw_frame_idx = frame_idx
            continue

        flow_prev_raw_to_curr = flow_engine.compute(prev_raw_gray, gray)

        confirmed_labels, confirmed_components = _confirm_persistent_components(
            prev_labels=prev_raw_labels,
            prev_components=prev_raw_components,
            prev_confirmed_labels=prev_confirmed_labels,
            flow_prev_confirmed_to_prev=prev_flow,
            next_labels=raw_labels,
            flow_prev_to_next=flow_prev_raw_to_curr,
            cfg=cfg.persistence,
            min_area=cfg.segment.min_area,
        )

        confirm_frame_idx = int(prev_raw_frame_idx)
        total_area = int((confirmed_labels > 0).sum())
        largest_component = max((c.area for c in confirmed_components.values()), default=0)
        area_timeseries.append({
            "frame_idx": confirm_frame_idx,
            "total_area_px": total_area,
            "num_components": int(len(confirmed_components)),
            "largest_component_area_px": int(largest_component),
        })

        stage = "segment"
        if (
            prev_confirmed_labels is not None
            and prev_confirmed_frame_idx is not None
            and prev_flow is not None
        ):
            _update_prev_velocities(
                prev_confirmed_labels,
                prev_id_map,
                prev_flow,
                result,
                prev_confirmed_frame_idx,
                cfg.fps,
            )
            warped_prev = _warp_labels(prev_confirmed_labels, prev_flow)
            overlap = _overlap_counts(warped_prev, confirmed_labels)
            prev_areas = {pid: comp.area for pid, comp in prev_confirmed_components.items()}
            curr_areas = {cid: comp.area for cid, comp in confirmed_components.items()}
            parents_to_children, children_to_parents = _build_links(
                overlap=overlap,
                prev_areas=prev_areas,
                curr_areas=curr_areas,
                cfg=cfg.match,
            )
            stage = "flow+match"
        else:
            parents_to_children = {}
            children_to_parents = {cid: [] for cid in confirmed_components}
            curr_areas = {cid: comp.area for cid, comp in confirmed_components.items()}
            prev_areas = {}

        curr_id_map: Dict[int, int] = {}
        for child_id, parent_ids in children_to_parents.items():
            if len(parent_ids) == 1:
                parent_id = parent_ids[0]
                if len(parents_to_children.get(parent_id, [])) == 1:
                    curr_id_map[child_id] = prev_id_map[parent_id]
                    continue
            curr_id_map[child_id] = next_id
            next_id += 1

        if prev_confirmed_labels is not None and prev_confirmed_frame_idx is not None:
            for child_id, parent_ids in children_to_parents.items():
                if not parent_ids:
                    events.append({
                        "type": "birth",
                        "prev_frame_idx": int(prev_confirmed_frame_idx),
                        "frame_idx": confirm_frame_idx,
                        "object_id": int(curr_id_map[child_id]),
                        "area_px": int(curr_areas.get(child_id, 0)),
                    })
                elif len(parent_ids) >= 2:
                    events.append({
                        "type": "merge",
                        "prev_frame_idx": int(prev_confirmed_frame_idx),
                        "frame_idx": confirm_frame_idx,
                        "parent_ids": [int(prev_id_map[p]) for p in parent_ids],
                        "child_id": int(curr_id_map[child_id]),
                    })

            for parent_id, child_ids in parents_to_children.items():
                if not child_ids:
                    events.append({
                        "type": "death",
                        "prev_frame_idx": int(prev_confirmed_frame_idx),
                        "frame_idx": confirm_frame_idx,
                        "object_id": int(prev_id_map[parent_id]),
                        "area_px": int(prev_areas.get(parent_id, 0)),
                    })
                elif len(child_ids) >= 2:
                    events.append({
                        "type": "split",
                        "prev_frame_idx": int(prev_confirmed_frame_idx),
                        "frame_idx": confirm_frame_idx,
                        "parent_id": int(prev_id_map[parent_id]),
                        "child_ids": [int(curr_id_map[c]) for c in child_ids],
                    })

        id_mask_crop = np.zeros_like(confirmed_labels, dtype=np.int32)
        for label_id, comp in confirmed_components.items():
            obj_id = curr_id_map.get(label_id)
            if obj_id is None:
                continue
            id_mask_crop[confirmed_labels == label_id] = obj_id

            x_offset, y_offset = roi_offset
            centroid = (comp.centroid[0] + x_offset, comp.centroid[1] + y_offset)
            bbox = BoundingBox(
                x_min=comp.bbox.x_min + x_offset,
                y_min=comp.bbox.y_min + y_offset,
                x_max=comp.bbox.x_max + x_offset,
                y_max=comp.bbox.y_max + y_offset,
            )
            frame_data = FrameData(centroid=centroid, area=comp.area, bbox=bbox)
            if obj_id not in result.objects:
                result.objects[obj_id] = TrackedObject(object_id=obj_id)
            result.objects[obj_id].frames[confirm_frame_idx] = frame_data

        if roi:
            full_mask = np.zeros(frame_shape, dtype=np.int32)
            x1, y1 = roi["x1"], roi["y1"]
            x2, y2 = roi["x2"], roi["y2"]
            full_mask[y1:y2, x1:x2] = id_mask_crop
        else:
            full_mask = id_mask_crop
        save_mask(full_mask, str(masks_dir / f"{confirm_frame_idx:06d}.png"))

        prev_confirmed_labels = confirmed_labels
        prev_confirmed_components = confirmed_components
        prev_confirmed_frame_idx = confirm_frame_idx
        prev_id_map = curr_id_map
        prev_flow = flow_prev_raw_to_curr

        prev_raw_gray = gray
        prev_raw_labels = raw_labels
        prev_raw_components = raw_components
        prev_raw_frame_idx = frame_idx

        _log_progress(confirm_frame_idx, total_frames, stage, cfg.log_every)

    if prev_raw_labels is not None and prev_raw_frame_idx is not None:
        confirmed_labels, confirmed_components = _confirm_persistent_components(
            prev_labels=prev_raw_labels,
            prev_components=prev_raw_components,
            prev_confirmed_labels=prev_confirmed_labels,
            flow_prev_confirmed_to_prev=prev_flow,
            next_labels=None,
            flow_prev_to_next=None,
            cfg=cfg.persistence,
            min_area=cfg.segment.min_area,
        )

        confirm_frame_idx = int(prev_raw_frame_idx)
        total_area = int((confirmed_labels > 0).sum())
        largest_component = max((c.area for c in confirmed_components.values()), default=0)
        area_timeseries.append({
            "frame_idx": confirm_frame_idx,
            "total_area_px": total_area,
            "num_components": int(len(confirmed_components)),
            "largest_component_area_px": int(largest_component),
        })

        stage = "segment"
        if (
            prev_confirmed_labels is not None
            and prev_confirmed_frame_idx is not None
            and prev_flow is not None
        ):
            _update_prev_velocities(
                prev_confirmed_labels,
                prev_id_map,
                prev_flow,
                result,
                prev_confirmed_frame_idx,
                cfg.fps,
            )
            warped_prev = _warp_labels(prev_confirmed_labels, prev_flow)
            overlap = _overlap_counts(warped_prev, confirmed_labels)
            prev_areas = {pid: comp.area for pid, comp in prev_confirmed_components.items()}
            curr_areas = {cid: comp.area for cid, comp in confirmed_components.items()}
            parents_to_children, children_to_parents = _build_links(
                overlap=overlap,
                prev_areas=prev_areas,
                curr_areas=curr_areas,
                cfg=cfg.match,
            )
            stage = "flow+match"
        else:
            parents_to_children = {}
            children_to_parents = {cid: [] for cid in confirmed_components}
            curr_areas = {cid: comp.area for cid, comp in confirmed_components.items()}
            prev_areas = {}

        curr_id_map: Dict[int, int] = {}
        for child_id, parent_ids in children_to_parents.items():
            if len(parent_ids) == 1:
                parent_id = parent_ids[0]
                if len(parents_to_children.get(parent_id, [])) == 1:
                    curr_id_map[child_id] = prev_id_map[parent_id]
                    continue
            curr_id_map[child_id] = next_id
            next_id += 1

        if prev_confirmed_labels is not None and prev_confirmed_frame_idx is not None:
            for child_id, parent_ids in children_to_parents.items():
                if not parent_ids:
                    events.append({
                        "type": "birth",
                        "prev_frame_idx": int(prev_confirmed_frame_idx),
                        "frame_idx": confirm_frame_idx,
                        "object_id": int(curr_id_map[child_id]),
                        "area_px": int(curr_areas.get(child_id, 0)),
                    })
                elif len(parent_ids) >= 2:
                    events.append({
                        "type": "merge",
                        "prev_frame_idx": int(prev_confirmed_frame_idx),
                        "frame_idx": confirm_frame_idx,
                        "parent_ids": [int(prev_id_map[p]) for p in parent_ids],
                        "child_id": int(curr_id_map[child_id]),
                    })

            for parent_id, child_ids in parents_to_children.items():
                if not child_ids:
                    events.append({
                        "type": "death",
                        "prev_frame_idx": int(prev_confirmed_frame_idx),
                        "frame_idx": confirm_frame_idx,
                        "object_id": int(prev_id_map[parent_id]),
                        "area_px": int(prev_areas.get(parent_id, 0)),
                    })
                elif len(child_ids) >= 2:
                    events.append({
                        "type": "split",
                        "prev_frame_idx": int(prev_confirmed_frame_idx),
                        "frame_idx": confirm_frame_idx,
                        "parent_id": int(prev_id_map[parent_id]),
                        "child_ids": [int(curr_id_map[c]) for c in child_ids],
                    })

        id_mask_crop = np.zeros_like(confirmed_labels, dtype=np.int32)
        for label_id, comp in confirmed_components.items():
            obj_id = curr_id_map.get(label_id)
            if obj_id is None:
                continue
            id_mask_crop[confirmed_labels == label_id] = obj_id

            x_offset, y_offset = roi_offset
            centroid = (comp.centroid[0] + x_offset, comp.centroid[1] + y_offset)
            bbox = BoundingBox(
                x_min=comp.bbox.x_min + x_offset,
                y_min=comp.bbox.y_min + y_offset,
                x_max=comp.bbox.x_max + x_offset,
                y_max=comp.bbox.y_max + y_offset,
            )
            frame_data = FrameData(centroid=centroid, area=comp.area, bbox=bbox)
            if obj_id not in result.objects:
                result.objects[obj_id] = TrackedObject(object_id=obj_id)
            result.objects[obj_id].frames[confirm_frame_idx] = frame_data

        if roi:
            full_mask = np.zeros(frame_shape, dtype=np.int32)
            x1, y1 = roi["x1"], roi["y1"]
            x2, y2 = roi["x2"], roi["y2"]
            full_mask[y1:y2, x1:x2] = id_mask_crop
        else:
            full_mask = id_mask_crop
        save_mask(full_mask, str(masks_dir / f"{confirm_frame_idx:06d}.png"))

        _log_progress(confirm_frame_idx, total_frames, stage, cfg.log_every)

    result.num_objects = len(result.objects)

    tracks_path = output_root / "tracks.json"
    save_tracks(result, str(tracks_path))
    save_tracks_csv(result, str(output_root / "tracks.csv"), cfg.fps, cfg.mm_per_pixel)

    area_timeseries_path = output_root / "area_timeseries.json"
    area_timeseries_path.write_text(json.dumps(area_timeseries, indent=2))
    save_area_timeseries_csv(area_timeseries, str(output_root / "area_timeseries.csv"))

    event_summary = {
        "num_splits": sum(1 for e in events if e["type"] == "split"),
        "num_merges": sum(1 for e in events if e["type"] == "merge"),
        "num_births": sum(1 for e in events if e["type"] == "birth"),
        "num_deaths": sum(1 for e in events if e["type"] == "death"),
    }
    events_path = output_root / "events.json"
    events_path.write_text(json.dumps({
        "summary": event_summary,
        "config": asdict(cfg),
        "runtime": {
            "flow_backend": flow_engine.backend,
            "flow_backend_requested": cfg.flow_backend,
            "cuda_device": int(cfg.cuda_device),
            "mask_source_dir": str(masks_dir) if masks_dir else None,
        },
        "events": events,
    }, indent=2))

    return {
        "result": result,
        "events": events,
        "event_summary": event_summary,
        "area_timeseries": area_timeseries,
        "masks_dir": str(masks_dir),
        "flow_backend": flow_engine.backend,
    }
