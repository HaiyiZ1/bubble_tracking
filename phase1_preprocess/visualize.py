"""Render keyframe preview PNGs for human review.

For each keyframe index, emit PNGs to <out_dir>/keyframe_previews/:
- <idx>_input.png        : raw frame cropped to rectangular ROI (grayscale)
- <idx>_diff.png         : reference-diff (float, normalised to uint8 for display)
- <idx>_adjusted.png     : contrast-stretched diff before light-map (default mode)
- <idx>_lightmap.png     : light-map (intensity gain field) — default mode only
- <idx>_normalized.png   : final preprocessed output fed to segmentation

Note: the spec described stages named `threshold`, `morph`, and `median`, but
those keys do not exist in phase1_preprocess.preprocess — they belong to the
downstream segmentation step (Phase 2).  The actual debug dict keys produced by
preprocess_frame_with_debug are:
  default mode: diff, adjusted, light_map, normalized
  raw mode:     diff, diff_gray, normalized

Keyframe indices: at every CV_KEYFRAME_INTERVAL starting from
`start_keyframe_idx` (defaults to 1 in Phase 1 — Phase 2 has its own
start-idx flag).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

import cv2
import numpy as np

from .preprocess import PreprocessConfig, preprocess_frame_with_debug

DEFAULT_KEYFRAME_INTERVAL = 160


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    """Normalise any dtype to uint8 for PNG export."""
    if arr.dtype == np.uint8:
        return arr
    a = arr.astype(np.float32)
    lo, hi = float(np.min(a)), float(np.max(a))
    if hi <= lo:
        hi = lo + 1.0
    return ((a - lo) / (hi - lo) * 255).astype(np.uint8)


def render_previews(
    *,
    frames_dir: Path,
    reference_path: Path,
    roi: dict,
    roi_mask: np.ndarray,
    out_dir: Path,
    cfg: PreprocessConfig,
    keyframe_indices: Iterable[int],
) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ref_bgr = cv2.imread(str(reference_path))
    if ref_bgr is None:
        raise ValueError(f"Could not read {reference_path}")

    x1, y1, x2, y2 = roi["x1"], roi["y1"], roi["x2"], roi["y2"]

    def _crop(arr: np.ndarray) -> np.ndarray:
        return arr[y1:y2, x1:x2]

    def _save(p: Path, arr: np.ndarray) -> None:
        cv2.imwrite(str(p), _crop(_to_uint8(arr)))

    written: List[Path] = []
    indices_list = list(keyframe_indices)

    for idx in indices_list:
        frame_path = frames_dir / f"{idx:06d}.jpg"
        if not frame_path.exists():
            continue
        frame_bgr = cv2.imread(str(frame_path))
        if frame_bgr is None:
            continue

        debug = preprocess_frame_with_debug(frame_bgr, ref_bgr, cfg)

        # input: raw frame as grayscale
        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        p_input = out_dir / f"{idx:06d}_input.png"
        _save(p_input, frame_gray)
        written.append(p_input)

        # diff: reference-diff (may be float or multi-channel — reduce to 2-D for display)
        if "diff" in debug:
            d = debug["diff"]
            if d.ndim == 3:
                d = cv2.cvtColor(_to_uint8(d), cv2.COLOR_BGR2GRAY).astype(np.float32)
            p_diff = out_dir / f"{idx:06d}_diff.png"
            _save(p_diff, d)
            written.append(p_diff)

        # adjusted: contrast-stretched diff before light-map (default mode only)
        if "adjusted" in debug:
            p_adj = out_dir / f"{idx:06d}_adjusted.png"
            adj = debug["adjusted"]
            if adj.ndim == 3:
                adj = cv2.cvtColor(adj, cv2.COLOR_BGR2GRAY)
            _save(p_adj, adj)
            written.append(p_adj)

        # lightmap: intensity gain field (default mode only)
        if "light_map" in debug:
            p_lm = out_dir / f"{idx:06d}_lightmap.png"
            _save(p_lm, debug["light_map"])
            written.append(p_lm)

        # normalized: final preprocessed output fed to segmentation
        if "normalized" in debug:
            p_norm = out_dir / f"{idx:06d}_normalized.png"
            _save(p_norm, debug["normalized"])
            written.append(p_norm)

    # Summary record
    (out_dir / "previews.json").write_text(json.dumps({
        "keyframes": indices_list,
        "stages": ["input", "diff", "adjusted", "lightmap", "normalized"],
        "preprocess_mode": cfg.mode,
    }, indent=2) + "\n")
    return written


def keyframe_indices_for(
    n_frames: int, *, interval: int, start: int, extras: Iterable[int] | None = None
) -> List[int]:
    if interval <= 0:
        raise ValueError(f"interval must be positive, got {interval}")
    base = list(range(start, n_frames + 1, interval))
    if extras:
        for e in extras:
            if 1 <= int(e) <= n_frames and int(e) not in base:
                base.append(int(e))
        base.sort()
    return base
