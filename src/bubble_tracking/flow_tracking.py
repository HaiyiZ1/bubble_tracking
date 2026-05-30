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
    raw_sigma: float = 1.0


@dataclass
class SegmentationConfig:
    blur_ksize: int = 3
    hist_bins: int = 35
    threshold_offset: int = 7
    max_threshold: int = 248
    opening_radius: int = 0
    min_area: int = 500
    median_ksize: int = 10
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
class DebounceConfig:
    # Require this many consecutive detections before a new ID is kept.
    birth_confirm_frames: int = 1
    # Require this many consecutive missing frames before an ID is removed.
    death_confirm_frames: int = 1
    # Max centroid distance (pixels) when re-linking a new ID to a recently missing ID.
    relink_max_centroid_dist_px: float = 20.0
    # Area ratio bounds (new/old) for ID re-linking.
    relink_min_area_ratio: float = 0.35
    relink_max_area_ratio: float = 2.8


@dataclass
class TrackingConfig:
    fps: float = 4.0
    mm_per_pixel: Optional[float] = None
    flow_backend: str = "auto"
    cuda_device: int = 0
    log_every: int = 10
    use_flow: bool = True
    velocity_mode: str = "flow_median"
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    segment: SegmentationConfig = field(default_factory=SegmentationConfig)
    flow: FlowConfig = field(default_factory=FlowConfig)
    match: MatchConfig = field(default_factory=MatchConfig)
    persistence: PersistenceConfig = field(default_factory=PersistenceConfig)
    debounce: DebounceConfig = field(default_factory=DebounceConfig)


@dataclass
class Component:
    label: int
    area: int
    centroid: Tuple[float, float]
    bbox: BoundingBox


@dataclass
class TrackRuntime:
    object_id: int
    confirmed: bool
    first_seen_frame_idx: int
    seen_streak: int = 1
    miss_count: int = 0
    first_missing_frame_idx: Optional[int] = None
    missing_frame_indices: List[int] = field(default_factory=list)
    pending_frame_indices: List[int] = field(default_factory=list)
    emit_birth_event_on_confirm: bool = False
    birth_event_emitted: bool = False


@dataclass
class FrameBufferEntry:
    frame_idx: int
    mask: np.ndarray


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
    if mode == "raw":
        return _preprocess_frame_raw_with_debug(frame_bgr, reference_bgr, cfg)
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


def _update_prev_velocities_from_centroids(
    prev_components: Dict[int, Component],
    curr_components: Dict[int, Component],
    parents_to_children: Dict[int, List[int]],
    children_to_parents: Dict[int, List[int]],
    prev_id_map: Dict[int, int],
    curr_id_map: Dict[int, int],
    result: TrackingResult,
    frame_idx: int,
    fps: float,
) -> None:
    # Only assign centroid-delta velocity on one-to-one links (no split/merge ambiguity).
    for child_id, parent_ids in children_to_parents.items():
        if len(parent_ids) != 1:
            continue
        parent_id = parent_ids[0]
        if len(parents_to_children.get(parent_id, [])) != 1:
            continue
        prev_obj_id = prev_id_map.get(parent_id)
        curr_obj_id = curr_id_map.get(child_id)
        if prev_obj_id is None or curr_obj_id is None or prev_obj_id != curr_obj_id:
            continue
        prev_comp = prev_components.get(parent_id)
        curr_comp = curr_components.get(child_id)
        if prev_comp is None or curr_comp is None:
            continue
        vx = (float(curr_comp.centroid[0]) - float(prev_comp.centroid[0])) * fps
        vy = (float(curr_comp.centroid[1]) - float(prev_comp.centroid[1])) * fps
        obj = result.objects.get(int(curr_obj_id))
        if obj is None:
            continue
        frame_data = obj.frames.get(frame_idx)
        if frame_data is not None:
            frame_data.velocity = (vx, vy)


def _update_prev_velocities_from_state_ids(
    prev_components: Dict[int, Component],
    prev_id_map: Dict[int, int],
    curr_components: Dict[int, Component],
    curr_id_map: Dict[int, int],
    result: TrackingResult,
    frame_idx: int,
    fps: float,
) -> None:
    prev_by_obj = {
        int(obj_id): prev_components[label_id]
        for label_id, obj_id in prev_id_map.items()
        if label_id in prev_components
    }
    curr_by_obj = {
        int(obj_id): curr_components[label_id]
        for label_id, obj_id in curr_id_map.items()
        if label_id in curr_components
    }
    for obj_id in sorted(set(prev_by_obj) & set(curr_by_obj)):
        prev_comp = prev_by_obj[obj_id]
        curr_comp = curr_by_obj[obj_id]
        vx = (float(curr_comp.centroid[0]) - float(prev_comp.centroid[0])) * fps
        vy = (float(curr_comp.centroid[1]) - float(prev_comp.centroid[1])) * fps
        obj = result.objects.get(int(obj_id))
        if obj is None:
            continue
        frame_data = obj.frames.get(frame_idx)
        if frame_data is not None:
            frame_data.velocity = (vx, vy)


def _id_mask_from_labels(labels: np.ndarray, id_map: Dict[int, int]) -> np.ndarray:
    out = np.zeros_like(labels, dtype=np.int32)
    for label_id, obj_id in id_map.items():
        out[labels == int(label_id)] = int(obj_id)
    return out


def _build_full_id_mask(
    id_mask_crop: np.ndarray,
    frame_shape: Tuple[int, int],
    roi: Optional[dict],
) -> np.ndarray:
    if roi is None:
        return id_mask_crop.copy()
    full_mask = np.zeros(frame_shape, dtype=np.int32)
    x1, y1 = roi["x1"], roi["y1"]
    x2, y2 = roi["x2"], roi["y2"]
    full_mask[y1:y2, x1:x2] = id_mask_crop
    return full_mask


def _component_from_binary_mask(binary: np.ndarray, label: int) -> Optional[Component]:
    yy, xx = np.where(binary)
    if xx.size == 0:
        return None
    bbox = BoundingBox(
        x_min=int(np.min(xx)),
        y_min=int(np.min(yy)),
        x_max=int(np.max(xx)),
        y_max=int(np.max(yy)),
    )
    return Component(
        label=int(label),
        area=int(xx.size),
        centroid=(float(np.mean(xx)), float(np.mean(yy))),
        bbox=bbox,
    )


def _state_from_id_mask(id_mask: np.ndarray) -> Tuple[np.ndarray, Dict[int, Component], Dict[int, int]]:
    labels = np.zeros_like(id_mask, dtype=np.int32)
    components: Dict[int, Component] = {}
    id_map: Dict[int, int] = {}
    next_label = 1
    for obj_id in sorted(int(v) for v in np.unique(id_mask) if int(v) > 0):
        binary = id_mask == int(obj_id)
        comp = _component_from_binary_mask(binary, next_label)
        if comp is None:
            continue
        labels[binary] = int(next_label)
        components[int(next_label)] = comp
        id_map[int(next_label)] = int(obj_id)
        next_label += 1
    return labels, components, id_map


def _update_result_from_id_mask(
    result: TrackingResult,
    id_mask_crop: np.ndarray,
    frame_idx: int,
    roi_offset: Tuple[int, int],
) -> None:
    x_offset, y_offset = roi_offset
    for obj_id in sorted(int(v) for v in np.unique(id_mask_crop) if int(v) > 0):
        binary = id_mask_crop == int(obj_id)
        yy, xx = np.where(binary)
        if xx.size == 0:
            continue
        bbox = BoundingBox(
            x_min=int(np.min(xx)) + x_offset,
            y_min=int(np.min(yy)) + y_offset,
            x_max=int(np.max(xx)) + x_offset,
            y_max=int(np.max(yy)) + y_offset,
        )
        frame_data = FrameData(
            centroid=(float(np.mean(xx)) + x_offset, float(np.mean(yy)) + y_offset),
            area=int(xx.size),
            bbox=bbox,
        )
        if int(obj_id) not in result.objects:
            result.objects[int(obj_id)] = TrackedObject(object_id=int(obj_id))
        result.objects[int(obj_id)].frames[int(frame_idx)] = frame_data


def _remove_result_frame(
    result: TrackingResult,
    obj_id: int,
    frame_idx: int,
) -> None:
    obj = result.objects.get(int(obj_id))
    if obj is None:
        return
    obj.frames.pop(int(frame_idx), None)
    if not obj.frames:
        result.objects.pop(int(obj_id), None)


def _discard_tentative_track(
    result: TrackingResult,
    frame_buffer: List[FrameBufferEntry],
    runtime: TrackRuntime,
) -> None:
    for frame_idx in runtime.pending_frame_indices:
        _remove_result_frame(result, runtime.object_id, frame_idx)
        for entry in frame_buffer:
            if int(entry.frame_idx) == int(frame_idx):
                entry.mask[entry.mask == int(runtime.object_id)] = 0


def _discard_missing_grace_frames(
    result: TrackingResult,
    frame_buffer: List[FrameBufferEntry],
    runtime: TrackRuntime,
    masks_dir: Path,
    area_timeseries: List[dict],
) -> None:
    for frame_idx in runtime.missing_frame_indices:
        _remove_result_frame(result, runtime.object_id, frame_idx)
        for entry in frame_buffer:
            if int(entry.frame_idx) == int(frame_idx):
                entry.mask[entry.mask == int(runtime.object_id)] = 0
                break
        else:
            mask_path = masks_dir / f"{int(frame_idx):06d}.png"
            if not mask_path.exists():
                continue
            mask = load_mask(str(mask_path))
            if not np.any(mask == int(runtime.object_id)):
                continue
            mask = mask.copy()
            mask[mask == int(runtime.object_id)] = 0
            save_mask(mask, str(mask_path))
            _update_area_timeseries_record(area_timeseries, int(frame_idx), mask)


def _confirm_injection_track(
    runtime: TrackRuntime,
    result: TrackingResult,
    events: List[dict],
) -> None:
    if runtime.birth_event_emitted or not runtime.emit_birth_event_on_confirm:
        runtime.confirmed = True
        runtime.pending_frame_indices.clear()
        return
    obj = result.objects.get(int(runtime.object_id))
    if obj is None:
        return
    frame_data = obj.frames.get(int(runtime.first_seen_frame_idx))
    if frame_data is None:
        return
    events.append({
        "type": "birth",
        "prev_frame_idx": int(max(0, runtime.first_seen_frame_idx - 1)),
        "frame_idx": int(runtime.first_seen_frame_idx),
        "object_id": int(runtime.object_id),
        "area_px": int(frame_data.area),
    })
    runtime.confirmed = True
    runtime.birth_event_emitted = True
    runtime.pending_frame_indices.clear()


def _make_area_timeseries_record(
    frame_idx: int,
    mask: np.ndarray,
) -> dict:
    fg = (mask > 0).astype(np.uint8)
    total_area = int(fg.sum())
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    largest = int(stats[1:, cv2.CC_STAT_AREA].max()) if n_labels > 1 else 0
    return {
        "frame_idx": int(frame_idx),
        "total_area_px": total_area,
        "num_components": int(max(0, n_labels - 1)),
        "largest_component_area_px": largest,
    }


def _append_area_timeseries_record(
    area_timeseries: List[dict],
    frame_idx: int,
    mask: np.ndarray,
) -> None:
    area_timeseries.append(_make_area_timeseries_record(frame_idx, mask))


def _update_area_timeseries_record(
    area_timeseries: List[dict],
    frame_idx: int,
    mask: np.ndarray,
) -> None:
    updated = _make_area_timeseries_record(frame_idx, mask)
    for idx, record in enumerate(area_timeseries):
        if int(record.get("frame_idx", -1)) == int(frame_idx):
            area_timeseries[idx] = updated
            return
    area_timeseries.append(updated)


def _flush_frame_buffer_entries(
    frame_buffer: List[FrameBufferEntry],
    flush_before_frame_idx: int,
    masks_dir: Path,
    area_timeseries: List[dict],
    force: bool = False,
) -> None:
    while frame_buffer:
        if not force and int(frame_buffer[0].frame_idx) > int(flush_before_frame_idx):
            break
        entry = frame_buffer.pop(0)
        save_mask(entry.mask, str(masks_dir / f"{int(entry.frame_idx):06d}.png"))
        _append_area_timeseries_record(area_timeseries, int(entry.frame_idx), entry.mask)


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


def _normalize_related_ids(values: Optional[List[int]]) -> List[int]:
    if not values:
        return []
    return sorted({int(v) for v in values})


def _set_track_birth(
    track: TrackedObject,
    reason: str,
    frame_idx: int,
    parent_ids: Optional[List[int]] = None,
) -> None:
    new_frame_idx = int(frame_idx)
    if track.birth_frame_idx is None or new_frame_idx < int(track.birth_frame_idx):
        track.birth_reason = reason
        track.birth_frame_idx = new_frame_idx
        track.birth_parent_ids = _normalize_related_ids(parent_ids)
        return
    if new_frame_idx == int(track.birth_frame_idx):
        track.birth_reason = reason
        track.birth_parent_ids = _normalize_related_ids(parent_ids)


def _set_track_death(
    track: TrackedObject,
    reason: str,
    frame_idx: int,
    child_ids: Optional[List[int]] = None,
) -> None:
    new_frame_idx = int(frame_idx)
    if track.death_frame_idx is None or new_frame_idx > int(track.death_frame_idx):
        track.death_reason = reason
        track.death_frame_idx = new_frame_idx
        track.death_child_ids = _normalize_related_ids(child_ids)
        return
    if new_frame_idx == int(track.death_frame_idx):
        track.death_reason = reason
        track.death_child_ids = _normalize_related_ids(child_ids)


def _annotate_track_lifecycle(result: TrackingResult, events: List[dict]) -> None:
    for obj in result.objects.values():
        obj.birth_reason = None
        obj.birth_frame_idx = None
        obj.birth_parent_ids = []
        obj.death_reason = None
        obj.death_frame_idx = None
        obj.death_child_ids = []

    sorted_events = sorted(
        events,
        key=lambda e: (int(e.get("frame_idx", -1)), str(e.get("type", ""))),
    )
    for event in sorted_events:
        event_type = str(event.get("type", ""))
        frame_idx = int(event.get("frame_idx", -1))
        if event_type == "birth":
            track = result.objects.get(int(event["object_id"]))
            if track is not None:
                _set_track_birth(track, reason="injection", frame_idx=frame_idx)
            continue
        if event_type == "death":
            track = result.objects.get(int(event["object_id"]))
            if track is not None:
                _set_track_death(track, reason="death", frame_idx=frame_idx)
            continue
        if event_type == "merge":
            child_id = int(event["child_id"])
            parent_ids = [int(v) for v in event.get("parent_ids", [])]
            child = result.objects.get(child_id)
            if child is not None:
                _set_track_birth(child, reason="merge", frame_idx=frame_idx, parent_ids=parent_ids)
            for parent_id in parent_ids:
                parent = result.objects.get(parent_id)
                if parent is not None:
                    _set_track_death(parent, reason="merge", frame_idx=frame_idx, child_ids=[child_id])
            continue
        if event_type == "split":
            parent_id = int(event["parent_id"])
            child_ids = [int(v) for v in event.get("child_ids", [])]
            parent = result.objects.get(parent_id)
            if parent is not None:
                _set_track_death(parent, reason="split", frame_idx=frame_idx, child_ids=child_ids)
            for child_id in child_ids:
                child = result.objects.get(child_id)
                if child is not None:
                    _set_track_birth(child, reason="split", frame_idx=frame_idx, parent_ids=[parent_id])

    for obj in result.objects.values():
        if not obj.frames:
            continue
        first_frame_idx = int(min(obj.frames))
        if obj.birth_reason is None:
            obj.birth_reason = "injection"
            obj.birth_frame_idx = first_frame_idx
            obj.birth_parent_ids = []
        elif obj.birth_frame_idx is None:
            obj.birth_frame_idx = first_frame_idx


def _iter_runs(binary: np.ndarray, value: bool) -> List[Tuple[int, int]]:
    runs: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for idx, v in enumerate(binary.tolist()):
        if bool(v) == value:
            if start is None:
                start = idx
        elif start is not None:
            runs.append((start, idx - 1))
            start = None
    if start is not None:
        runs.append((start, len(binary) - 1))
    return runs


def _smooth_presence_1d(
    presence: np.ndarray,
    birth_confirm_frames: int,
    death_confirm_frames: int,
) -> np.ndarray:
    smoothed = presence.copy()

    on_min = max(1, int(birth_confirm_frames))
    if on_min > 1:
        for s, e in _iter_runs(smoothed, True):
            if (e - s + 1) < on_min:
                smoothed[s:e + 1] = False

    # If death requires N consecutive misses, fill gaps up to N-1.
    off_gap_max = max(0, int(death_confirm_frames) - 1)
    if off_gap_max > 0:
        for s, e in _iter_runs(smoothed, False):
            run_len = e - s + 1
            has_left = s > 0 and bool(smoothed[s - 1])
            has_right = e < (len(smoothed) - 1) and bool(smoothed[e + 1])
            if has_left and has_right and run_len <= off_gap_max:
                smoothed[s:e + 1] = True

    return smoothed


def _build_gap_relink_aliases(
    result: TrackingResult,
    cfg: DebounceConfig,
) -> Dict[int, int]:
    max_gap = max(0, int(cfg.death_confirm_frames) - 1)
    if max_gap <= 0:
        return {}

    tracks: Dict[int, dict] = {}
    for obj_id, obj in result.objects.items():
        if not obj.frames:
            continue
        frame_ids = sorted(obj.frames.keys())
        first_idx = int(frame_ids[0])
        last_idx = int(frame_ids[-1])
        first = obj.frames[first_idx]
        last = obj.frames[last_idx]
        tracks[int(obj_id)] = {
            "first_idx": first_idx,
            "last_idx": last_idx,
            "first_centroid": (float(first.centroid[0]), float(first.centroid[1])),
            "last_centroid": (float(last.centroid[0]), float(last.centroid[1])),
            "first_area": float(first.area),
            "last_area": float(last.area),
        }

    candidates: List[Tuple[float, int, int]] = []
    obj_ids = sorted(tracks.keys())
    for old_id in obj_ids:
        old = tracks[old_id]
        for new_id in obj_ids:
            if new_id == old_id:
                continue
            new = tracks[new_id]
            if new["first_idx"] <= old["last_idx"]:
                continue
            gap = int(new["first_idx"] - old["last_idx"] - 1)
            if gap < 0 or gap > max_gap:
                continue
            dx = float(new["first_centroid"][0] - old["last_centroid"][0])
            dy = float(new["first_centroid"][1] - old["last_centroid"][1])
            dist = float(np.hypot(dx, dy))
            if dist > float(cfg.relink_max_centroid_dist_px):
                continue
            area_old = max(1.0, float(old["last_area"]))
            area_ratio = float(new["first_area"] / area_old)
            if area_ratio < float(cfg.relink_min_area_ratio) or area_ratio > float(cfg.relink_max_area_ratio):
                continue
            score = dist + 5.0 * float(gap)
            candidates.append((score, int(old_id), int(new_id)))

    candidates.sort(key=lambda x: x[0])
    alias: Dict[int, int] = {}
    used_old: set[int] = set()
    used_new: set[int] = set()
    for _, old_id, new_id in candidates:
        if old_id in used_old or new_id in used_new:
            continue
        alias[new_id] = old_id
        used_old.add(old_id)
        used_new.add(new_id)
    return alias


def _collapse_alias_map(alias_map: Dict[int, int]) -> Dict[int, int]:
    if not alias_map:
        return {}

    def _root(x: int) -> int:
        seen: set[int] = set()
        cur = int(x)
        while cur in alias_map and cur not in seen:
            seen.add(cur)
            nxt = int(alias_map[cur])
            if nxt == cur:
                break
            cur = nxt
        return cur

    collapsed: Dict[int, int] = {}
    for src in list(alias_map.keys()):
        dst = _root(src)
        if dst != src:
            collapsed[int(src)] = int(dst)
    return collapsed


def _apply_aliases_to_mask(mask: np.ndarray, alias_map: Dict[int, int]) -> np.ndarray:
    if not alias_map:
        return mask
    out = mask.copy()
    for src, dst in alias_map.items():
        src_i = int(src)
        dst_i = int(dst)
        if src_i == dst_i:
            continue
        out[mask == src_i] = dst_i
    return out


def _remap_event_ids(event: dict, alias_map: Dict[int, int]) -> dict:
    if not alias_map:
        return event

    def _map_id(v: int) -> int:
        cur = int(v)
        seen: set[int] = set()
        while cur in alias_map and cur not in seen:
            seen.add(cur)
            nxt = int(alias_map[cur])
            if nxt == cur:
                break
            cur = nxt
        return cur

    out = dict(event)
    if "object_id" in out:
        out["object_id"] = int(_map_id(int(out["object_id"])))
    if "parent_id" in out:
        out["parent_id"] = int(_map_id(int(out["parent_id"])))
    if "child_id" in out:
        out["child_id"] = int(_map_id(int(out["child_id"])))
    if "parent_ids" in out and isinstance(out["parent_ids"], list):
        out["parent_ids"] = [int(_map_id(int(v))) for v in out["parent_ids"]]
    if "child_ids" in out and isinstance(out["child_ids"], list):
        out["child_ids"] = [int(_map_id(int(v))) for v in out["child_ids"]]
    return out


def _debounce_saved_masks(
    masks_dir: Path,
    total_frames: int,
    cfg: DebounceConfig,
    log_every: int,
    id_alias_map: Optional[Dict[int, int]] = None,
) -> Tuple[List[dict], List[dict]]:
    if total_frames <= 0:
        return [], []
    birth_confirm = max(1, int(cfg.birth_confirm_frames))
    death_confirm = max(1, int(cfg.death_confirm_frames))
    if birth_confirm <= 1 and death_confirm <= 1:
        return [], []
    alias_map = _collapse_alias_map(id_alias_map or {})

    frame_present_ids: List[np.ndarray] = []
    all_ids: set[int] = set()

    for frame_idx in range(total_frames):
        mask_path = masks_dir / f"{frame_idx:06d}.png"
        mask = load_mask(str(mask_path))
        if alias_map:
            mask = _apply_aliases_to_mask(mask, alias_map)
        ids = np.unique(mask)
        ids = ids[ids > 0].astype(np.int32)
        frame_present_ids.append(ids)
        for oid in ids.tolist():
            all_ids.add(int(oid))

    if not all_ids:
        empty_series = [
            {
                "frame_idx": int(frame_idx),
                "total_area_px": 0,
                "num_components": 0,
                "largest_component_area_px": 0,
            }
            for frame_idx in range(total_frames)
        ]
        return empty_series, []

    presence_by_id: Dict[int, np.ndarray] = {
        oid: np.zeros(total_frames, dtype=bool) for oid in sorted(all_ids)
    }
    for frame_idx, ids in enumerate(frame_present_ids):
        for oid in ids.tolist():
            presence_by_id[int(oid)][frame_idx] = True

    smoothed_by_id: Dict[int, np.ndarray] = {}
    for oid, presence in presence_by_id.items():
        smoothed_by_id[oid] = _smooth_presence_1d(
            presence=presence,
            birth_confirm_frames=birth_confirm,
            death_confirm_frames=death_confirm,
        )

    area_timeseries: List[dict] = []
    bd_events: List[dict] = []
    prev_out: Optional[np.ndarray] = None
    prev_present: set[int] = set()
    prev_area_by_id: Dict[int, int] = {}

    for frame_idx in range(total_frames):
        mask_path = masks_dir / f"{frame_idx:06d}.png"
        raw = load_mask(str(mask_path))
        orig = _apply_aliases_to_mask(raw, alias_map) if alias_map else raw
        out = orig.copy()

        orig_ids = frame_present_ids[frame_idx].tolist()
        for oid in orig_ids:
            if not bool(smoothed_by_id[int(oid)][frame_idx]):
                out[out == int(oid)] = 0

        curr_ids = np.unique(out)
        curr_ids = curr_ids[curr_ids > 0].astype(np.int32)
        curr_present: set[int] = set(int(v) for v in curr_ids.tolist())

        if prev_out is not None and prev_present:
            fill_ids = [
                oid
                for oid in prev_present
                if oid not in curr_present and bool(smoothed_by_id.get(oid, np.zeros(total_frames, dtype=bool))[frame_idx])
            ]
            if fill_ids:
                if len(fill_ids) == 1:
                    fill_id = int(fill_ids[0])
                    fill_mask = (prev_out == fill_id) & (out == 0)
                    out[fill_mask] = fill_id
                else:
                    fill_src = np.isin(prev_out, np.asarray(fill_ids, dtype=prev_out.dtype))
                    fill_mask = fill_src & (out == 0)
                    out[fill_mask] = prev_out[fill_mask]
                curr_ids = np.unique(out)
                curr_ids = curr_ids[curr_ids > 0].astype(np.int32)
                curr_present = set(int(v) for v in curr_ids.tolist())

        if not np.array_equal(out, raw):
            save_mask(out, str(mask_path))

        fg = (out > 0).astype(np.uint8)
        total_area = int(fg.sum())
        n_labels, _, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
        largest = int(stats[1:, cv2.CC_STAT_AREA].max()) if n_labels > 1 else 0
        area_timeseries.append({
            "frame_idx": int(frame_idx),
            "total_area_px": total_area,
            "num_components": int(max(0, n_labels - 1)),
            "largest_component_area_px": largest,
        })

        ids_now, counts_now = np.unique(out, return_counts=True)
        curr_area_by_id: Dict[int, int] = {
            int(oid): int(cnt)
            for oid, cnt in zip(ids_now.tolist(), counts_now.tolist())
            if int(oid) > 0
        }

        births = curr_present - prev_present
        deaths = prev_present - curr_present
        prev_idx = int(max(0, frame_idx - 1))
        for oid in sorted(births):
            bd_events.append({
                "type": "birth",
                "prev_frame_idx": prev_idx,
                "frame_idx": int(frame_idx),
                "object_id": int(oid),
                "area_px": int(curr_area_by_id.get(int(oid), 0)),
            })
        for oid in sorted(deaths):
            bd_events.append({
                "type": "death",
                "prev_frame_idx": prev_idx,
                "frame_idx": int(frame_idx),
                "object_id": int(oid),
                "area_px": int(prev_area_by_id.get(int(oid), 0)),
            })

        prev_out = out
        prev_present = curr_present
        prev_area_by_id = curr_area_by_id

        _log_progress(frame_idx, total_frames, "debounce", log_every)

    return area_timeseries, bd_events


def _rebuild_result_from_saved_masks(
    result: TrackingResult,
    masks_dir: Path,
    total_frames: int,
    roi_offset: Tuple[int, int],
    fps: float,
) -> None:
    objects: Dict[int, TrackedObject] = {}
    x_offset, y_offset = roi_offset

    for frame_idx in range(total_frames):
        mask_path = masks_dir / f"{frame_idx:06d}.png"
        mask = load_mask(str(mask_path))
        ids = np.unique(mask)
        ids = ids[ids > 0].astype(np.int32)
        for oid in ids.tolist():
            yy, xx = np.where(mask == int(oid))
            if xx.size == 0:
                continue
            centroid = (float(np.mean(xx)) + x_offset, float(np.mean(yy)) + y_offset)
            bbox = BoundingBox(
                x_min=int(np.min(xx)) + x_offset,
                y_min=int(np.min(yy)) + y_offset,
                x_max=int(np.max(xx)) + x_offset,
                y_max=int(np.max(yy)) + y_offset,
            )
            frame_data = FrameData(
                centroid=centroid,
                area=int(xx.size),
                bbox=bbox,
            )
            obj = objects.get(int(oid))
            if obj is None:
                obj = TrackedObject(object_id=int(oid))
                objects[int(oid)] = obj
            obj.frames[int(frame_idx)] = frame_data

    # Recompute per-frame velocity via centroid delta for consecutive frames.
    for obj in objects.values():
        frame_ids = sorted(obj.frames.keys())
        for i in range(len(frame_ids) - 1):
            f0 = frame_ids[i]
            f1 = frame_ids[i + 1]
            if f1 != f0 + 1:
                continue
            p0 = obj.frames[f0].centroid
            p1 = obj.frames[f1].centroid
            vx = (float(p1[0]) - float(p0[0])) * fps
            vy = (float(p1[1]) - float(p0[1])) * fps
            obj.frames[f0].velocity = (vx, vy)

    result.objects = objects
    result.num_objects = len(objects)


def track_frames(
    frames_dir: str,
    reference_path: str,
    output_dir: str,
    masks_dir: Optional[str] = None,
    roi: Optional[dict] = None,
    config: Optional[TrackingConfig] = None,
) -> dict:
    cfg = config or TrackingConfig()
    valid_velocity_modes = {"flow_median", "centroid_delta"}
    if cfg.velocity_mode not in valid_velocity_modes:
        raise ValueError(f"velocity_mode must be one of {sorted(valid_velocity_modes)}")
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
    birth_confirm = max(1, int(cfg.debounce.birth_confirm_frames))
    death_confirm = max(1, int(cfg.debounce.death_confirm_frames))
    buffer_lag = max(0, birth_confirm - 1)

    prev_raw_gray = None
    prev_raw_labels = None
    prev_raw_components: Dict[int, Component] = {}
    prev_raw_frame_idx = None

    prev_state_labels = None
    prev_state_components: Dict[int, Component] = {}
    prev_state_frame_idx = None
    prev_state_id_map: Dict[int, int] = {}
    prev_flow = None
    next_id = 1
    flow_engine = FlowEngine(cfg) if cfg.use_flow else None
    debounce_alias_count = 0
    tracks_runtime: Dict[int, TrackRuntime] = {}
    frame_buffer: List[FrameBufferEntry] = []

    def _process_confirmed_frame(
        confirm_frame_idx: int,
        confirmed_labels: np.ndarray,
        confirmed_components: Dict[int, Component],
    ) -> str:
        nonlocal prev_state_labels
        nonlocal prev_state_components
        nonlocal prev_state_frame_idx
        nonlocal prev_state_id_map
        nonlocal next_id

        stage = "segment"
        if prev_state_labels is not None and prev_state_frame_idx is not None:
            if cfg.use_flow and prev_flow is not None:
                if cfg.velocity_mode == "flow_median":
                    _update_prev_velocities(
                        prev_state_labels,
                        prev_state_id_map,
                        prev_flow,
                        result,
                        prev_state_frame_idx,
                        cfg.fps,
                    )
                warped_prev_labels = _warp_labels(prev_state_labels, prev_flow)
                stage = "flow+match"
            else:
                warped_prev_labels = prev_state_labels
                stage = "match"
            overlap = _overlap_counts(warped_prev_labels, confirmed_labels)
            prev_areas = {pid: comp.area for pid, comp in prev_state_components.items()}
            curr_areas = {cid: comp.area for cid, comp in confirmed_components.items()}
            parents_to_children, children_to_parents = _build_links(
                overlap=overlap,
                prev_areas=prev_areas,
                curr_areas=curr_areas,
                cfg=cfg.match,
            )
            warped_prev_id_mask = _id_mask_from_labels(warped_prev_labels, prev_state_id_map)
        else:
            parents_to_children = {}
            children_to_parents = {cid: [] for cid in confirmed_components}
            warped_prev_id_mask = np.zeros_like(confirmed_labels, dtype=np.int32)

        parent_obj_to_children: Dict[int, List[int]] = {}
        child_to_parent_objs: Dict[int, List[int]] = {}
        for parent_label in prev_state_components:
            obj_id = prev_state_id_map.get(int(parent_label))
            if obj_id is not None:
                parent_obj_to_children[int(obj_id)] = []
        for child_id, parent_labels in children_to_parents.items():
            parent_obj_ids = sorted({
                int(prev_state_id_map[parent_label])
                for parent_label in parent_labels
                if parent_label in prev_state_id_map
            })
            child_to_parent_objs[int(child_id)] = parent_obj_ids
            for obj_id in parent_obj_ids:
                parent_obj_to_children.setdefault(int(obj_id), []).append(int(child_id))

        current_id_mask_crop = np.zeros_like(confirmed_labels, dtype=np.int32)
        current_child_obj_ids: Dict[int, int] = {}
        tracks_to_confirm: set[int] = set()

        for child_id in sorted(confirmed_components):
            parent_obj_ids = child_to_parent_objs.get(int(child_id), [])
            is_one_to_one = (
                len(parent_obj_ids) == 1
                and len(parent_obj_to_children.get(int(parent_obj_ids[0]), [])) == 1
            )
            if is_one_to_one:
                obj_id = int(parent_obj_ids[0])
                runtime = tracks_runtime.get(obj_id)
                if runtime is None:
                    runtime = TrackRuntime(
                        object_id=obj_id,
                        confirmed=True,
                        first_seen_frame_idx=confirm_frame_idx,
                        emit_birth_event_on_confirm=False,
                        birth_event_emitted=True,
                    )
                    tracks_runtime[obj_id] = runtime
                runtime.miss_count = 0
                runtime.first_missing_frame_idx = None
                runtime.missing_frame_indices.clear()
                if not runtime.confirmed:
                    runtime.seen_streak += 1
                    if int(confirm_frame_idx) not in runtime.pending_frame_indices:
                        runtime.pending_frame_indices.append(int(confirm_frame_idx))
                    if runtime.seen_streak >= birth_confirm:
                        tracks_to_confirm.add(obj_id)
                current_child_obj_ids[int(child_id)] = obj_id
            elif len(parent_obj_ids) >= 2 or (
                len(parent_obj_ids) == 1
                and len(parent_obj_to_children.get(int(parent_obj_ids[0]), [])) >= 2
            ):
                obj_id = int(next_id)
                next_id += 1
                tracks_runtime[obj_id] = TrackRuntime(
                    object_id=obj_id,
                    confirmed=True,
                    first_seen_frame_idx=int(confirm_frame_idx),
                    seen_streak=birth_confirm,
                    emit_birth_event_on_confirm=False,
                    birth_event_emitted=True,
                )
                current_child_obj_ids[int(child_id)] = obj_id
            else:
                obj_id = int(next_id)
                next_id += 1
                confirmed_now = birth_confirm <= 1
                tracks_runtime[obj_id] = TrackRuntime(
                    object_id=obj_id,
                    confirmed=bool(confirmed_now),
                    first_seen_frame_idx=int(confirm_frame_idx),
                    seen_streak=1,
                    pending_frame_indices=[] if confirmed_now else [int(confirm_frame_idx)],
                    emit_birth_event_on_confirm=True,
                    birth_event_emitted=False,
                )
                if confirmed_now:
                    tracks_to_confirm.add(obj_id)
                current_child_obj_ids[int(child_id)] = obj_id

            current_id_mask_crop[confirmed_labels == int(child_id)] = int(current_child_obj_ids[int(child_id)])

        topology_parents_to_remove: set[int] = set()
        for child_id in sorted(current_child_obj_ids):
            parent_obj_ids = child_to_parent_objs.get(int(child_id), [])
            if len(parent_obj_ids) < 2:
                continue
            for parent_id in parent_obj_ids:
                runtime = tracks_runtime.get(int(parent_id))
                if runtime is not None and not runtime.confirmed:
                    _confirm_injection_track(runtime, result, events)
                topology_parents_to_remove.add(int(parent_id))
            events.append({
                "type": "merge",
                "prev_frame_idx": int(prev_state_frame_idx),
                "frame_idx": int(confirm_frame_idx),
                "parent_ids": [int(v) for v in parent_obj_ids],
                "child_id": int(current_child_obj_ids[int(child_id)]),
            })

        for parent_obj_id, child_ids in parent_obj_to_children.items():
            if len(child_ids) < 2:
                continue
            runtime = tracks_runtime.get(int(parent_obj_id))
            if runtime is not None and not runtime.confirmed:
                _confirm_injection_track(runtime, result, events)
            topology_parents_to_remove.add(int(parent_obj_id))
            events.append({
                "type": "split",
                "prev_frame_idx": int(prev_state_frame_idx),
                "frame_idx": int(confirm_frame_idx),
                "parent_id": int(parent_obj_id),
                "child_ids": [int(current_child_obj_ids[int(child_id)]) for child_id in child_ids],
            })

        for parent_obj_id, child_ids in parent_obj_to_children.items():
            if child_ids or int(parent_obj_id) in topology_parents_to_remove:
                continue
            runtime = tracks_runtime.get(int(parent_obj_id))
            if runtime is None:
                continue
            if runtime.confirmed:
                if runtime.miss_count == 0:
                    runtime.first_missing_frame_idx = int(confirm_frame_idx)
                runtime.miss_count += 1
                if runtime.miss_count < death_confirm:
                    persist_binary = (
                        (warped_prev_id_mask == int(parent_obj_id))
                        & (current_id_mask_crop == 0)
                    )
                    current_id_mask_crop[persist_binary] = int(parent_obj_id)
                    if np.any(persist_binary):
                        runtime.missing_frame_indices.append(int(confirm_frame_idx))
                else:
                    death_frame_idx = int(
                        runtime.first_missing_frame_idx
                        if runtime.first_missing_frame_idx is not None
                        else confirm_frame_idx
                    )
                    last_seen_frame_idx = max(0, death_frame_idx - 1)
                    prev_area = 0
                    obj = result.objects.get(int(parent_obj_id))
                    if obj is not None:
                        frame_data = obj.frames.get(int(last_seen_frame_idx))
                        if frame_data is not None:
                            prev_area = int(frame_data.area)
                    _discard_missing_grace_frames(
                        result=result,
                        frame_buffer=frame_buffer,
                        runtime=runtime,
                        masks_dir=masks_dir,
                        area_timeseries=area_timeseries,
                    )
                    events.append({
                        "type": "death",
                        "prev_frame_idx": int(last_seen_frame_idx),
                        "frame_idx": int(death_frame_idx),
                        "object_id": int(parent_obj_id),
                        "area_px": int(prev_area),
                    })
                    tracks_runtime.pop(int(parent_obj_id), None)
            else:
                _discard_tentative_track(result, frame_buffer, runtime)
                tracks_runtime.pop(int(parent_obj_id), None)

        for parent_obj_id in topology_parents_to_remove:
            tracks_runtime.pop(int(parent_obj_id), None)

        current_state_labels, current_state_components, current_state_id_map = _state_from_id_mask(
            current_id_mask_crop
        )
        _update_result_from_id_mask(
            result=result,
            id_mask_crop=current_id_mask_crop,
            frame_idx=int(confirm_frame_idx),
            roi_offset=roi_offset,
        )
        for obj_id in sorted(tracks_to_confirm):
            runtime = tracks_runtime.get(int(obj_id))
            if runtime is not None:
                _confirm_injection_track(runtime, result, events)
        full_mask = _build_full_id_mask(
            current_id_mask_crop,
            frame_shape=frame_shape,
            roi=roi,
        )
        frame_buffer.append(FrameBufferEntry(frame_idx=int(confirm_frame_idx), mask=full_mask))

        if (
            cfg.velocity_mode == "centroid_delta"
            and prev_state_frame_idx is not None
        ):
            _update_prev_velocities_from_state_ids(
                prev_components=prev_state_components,
                prev_id_map=prev_state_id_map,
                curr_components=current_state_components,
                curr_id_map=current_state_id_map,
                result=result,
                frame_idx=int(prev_state_frame_idx),
                fps=cfg.fps,
            )

        _flush_frame_buffer_entries(
            frame_buffer=frame_buffer,
            flush_before_frame_idx=int(confirm_frame_idx) - int(buffer_lag),
            masks_dir=masks_dir,
            area_timeseries=area_timeseries,
        )

        prev_state_labels = current_state_labels
        prev_state_components = current_state_components
        prev_state_frame_idx = int(confirm_frame_idx)
        prev_state_id_map = current_state_id_map
        return stage

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

        if cfg.use_flow:
            if flow_engine is None:
                raise RuntimeError("Flow engine is not initialized")
            flow_prev_raw_to_curr = flow_engine.compute(prev_raw_gray, gray)
        else:
            # Identity warp for overlap-only matching/persistence when flow is disabled.
            flow_prev_raw_to_curr = np.zeros((gray.shape[0], gray.shape[1], 2), dtype=np.float32)

        confirmed_labels, confirmed_components = _confirm_persistent_components(
            prev_labels=prev_raw_labels,
            prev_components=prev_raw_components,
            prev_confirmed_labels=prev_state_labels,
            flow_prev_confirmed_to_prev=prev_flow,
            next_labels=raw_labels,
            flow_prev_to_next=flow_prev_raw_to_curr,
            cfg=cfg.persistence,
            min_area=cfg.segment.min_area,
        )
        stage = _process_confirmed_frame(
            confirm_frame_idx=int(prev_raw_frame_idx),
            confirmed_labels=confirmed_labels,
            confirmed_components=confirmed_components,
        )
        prev_flow = flow_prev_raw_to_curr

        prev_raw_gray = gray
        prev_raw_labels = raw_labels
        prev_raw_components = raw_components
        prev_raw_frame_idx = frame_idx

        _log_progress(int(prev_state_frame_idx), total_frames, stage, cfg.log_every)

    if prev_raw_labels is not None and prev_raw_frame_idx is not None:
        confirmed_labels, confirmed_components = _confirm_persistent_components(
            prev_labels=prev_raw_labels,
            prev_components=prev_raw_components,
            prev_confirmed_labels=prev_state_labels,
            flow_prev_confirmed_to_prev=prev_flow,
            next_labels=None,
            flow_prev_to_next=None,
            cfg=cfg.persistence,
            min_area=cfg.segment.min_area,
        )
        stage = _process_confirmed_frame(
            confirm_frame_idx=int(prev_raw_frame_idx),
            confirmed_labels=confirmed_labels,
            confirmed_components=confirmed_components,
        )
        _log_progress(int(prev_state_frame_idx), total_frames, stage, cfg.log_every)

    for obj_id, runtime in list(tracks_runtime.items()):
        if runtime.confirmed:
            continue
        _discard_tentative_track(result, frame_buffer, runtime)
        tracks_runtime.pop(int(obj_id), None)

    _flush_frame_buffer_entries(
        frame_buffer=frame_buffer,
        flush_before_frame_idx=total_frames,
        masks_dir=masks_dir,
        area_timeseries=area_timeseries,
        force=True,
    )

    events = sorted(
        events,
        key=lambda e: (int(e.get("frame_idx", -1)), str(e.get("type", ""))),
    )
    area_timeseries = sorted(
        area_timeseries,
        key=lambda record: int(record.get("frame_idx", -1)),
    )

    _annotate_track_lifecycle(result, events)
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
            "flow_backend": flow_engine.backend if flow_engine is not None else "disabled",
            "flow_backend_requested": cfg.flow_backend,
            "cuda_device": int(cfg.cuda_device),
            "mask_source_dir": str(masks_dir) if masks_dir else None,
            "flow_enabled": bool(cfg.use_flow),
            "velocity_mode": cfg.velocity_mode,
            "debounce_birth_confirm_frames": int(cfg.debounce.birth_confirm_frames),
            "debounce_death_confirm_frames": int(cfg.debounce.death_confirm_frames),
            "debounce_relink_max_centroid_dist_px": float(cfg.debounce.relink_max_centroid_dist_px),
            "debounce_relink_min_area_ratio": float(cfg.debounce.relink_min_area_ratio),
            "debounce_relink_max_area_ratio": float(cfg.debounce.relink_max_area_ratio),
            "debounce_relinked_ids": int(debounce_alias_count),
        },
        "events": events,
    }, indent=2))

    return {
        "result": result,
        "events": events,
        "event_summary": event_summary,
        "area_timeseries": area_timeseries,
        "masks_dir": str(masks_dir),
        "flow_backend": flow_engine.backend if flow_engine is not None else "disabled",
    }
