"""Preprocessing shared across Phase 1 and Phase 2.

Translated from phase1_preprocess/preprocess.py. Supports modes "default"
(reference-diff + Gaussian smoothing + light-map normalization + morphology
+ median + contrast stretch) and "raw" (minimal reference-diff + Gaussian only).

Dead code from the original (`automatrix` mode) is dropped.

PreprocessConfig is imported from .schemas (canonical home).
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from .schemas import PreprocessConfig


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
