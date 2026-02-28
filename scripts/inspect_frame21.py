#!/usr/bin/env python3
"""
Generate a diagnostic panel for a single frame: baseline, reference, diff, mask, mask with CLAHE.
"""

import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bubble_tracking.flow_tracking import (
    PreprocessConfig,
    SegmentationConfig,
    _apply_edge_gate,
    load_roi_json,
    preprocess_frame_with_debug,
    segment_frame,
    segment_frame_automatrix,
)
from bubble_tracking.frame_io import load_mask


def _load_config(events_path: str):
    data = json.loads(Path(events_path).read_text())
    cfg = data.get("config", {})
    preprocess_cfg = cfg.get("preprocess", {})
    segment_cfg = cfg.get("segment", {})
    return PreprocessConfig(**preprocess_cfg), SegmentationConfig(**segment_cfg)


def _apply_roi(frame: np.ndarray, roi):
    if roi is None:
        return frame
    return frame[roi["y1"]:roi["y2"], roi["x1"]:roi["x2"]]


def _mask_from_labels(labels: np.ndarray) -> np.ndarray:
    mask = np.zeros_like(labels, dtype=np.uint8)
    mask[labels > 0] = 255
    return mask


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a single frame for noise diagnostics")
    parser.add_argument("--frame-idx", type=int, default=21, help="Frame index (0-based)")
    parser.add_argument("--frames", required=True, help="Frames directory")
    parser.add_argument("--reference", required=True, help="Reference image path")
    parser.add_argument("--roi", default=None, help="ROI JSON path")
    parser.add_argument("--events", required=True, help="events.json from a run")
    parser.add_argument("--masks", required=True, help="Masks directory from a run")
    parser.add_argument("--out", required=True, help="Output PNG path")
    args = parser.parse_args()

    roi = load_roi_json(args.roi) if args.roi else None
    preprocess_cfg, segment_cfg = _load_config(args.events)

    frame_path = Path(args.frames) / f"{args.frame_idx:06d}.jpg"
    mask_path = Path(args.masks) / f"{args.frame_idx:06d}.png"
    if not frame_path.exists():
        raise FileNotFoundError(f"Frame not found: {frame_path}")
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask not found: {mask_path}")

    frame = cv2.imread(str(frame_path))
    reference = cv2.imread(str(args.reference))
    if frame is None or reference is None:
        raise ValueError("Failed to load frame or reference image")

    frame_roi = _apply_roi(frame, roi)
    reference_roi = _apply_roi(reference, roi)

    debug = preprocess_frame_with_debug(frame_roi, reference_roi, preprocess_cfg)
    diff_norm = debug["normalized"]

    current_mask = load_mask(str(mask_path))
    if roi:
        current_mask = _apply_roi(current_mask, roi)
    current_mask = _mask_from_labels(current_mask)

    clahe_cfg = PreprocessConfig(**{**preprocess_cfg.__dict__, "use_clahe": True})
    debug_clahe = preprocess_frame_with_debug(frame_roi, reference_roi, clahe_cfg)
    if (preprocess_cfg.mode or "default").lower() == "automatrix":
        labels_clahe, comps_clahe = segment_frame_automatrix(debug_clahe["normalized"], segment_cfg)
    else:
        labels_clahe, comps_clahe = segment_frame(debug_clahe["normalized"], segment_cfg)
    labels_clahe, _ = _apply_edge_gate(labels_clahe, comps_clahe, debug_clahe["normalized"], segment_cfg)
    clahe_mask = _mask_from_labels(labels_clahe)

    fig, axes = plt.subplots(1, 5, figsize=(20, 4), dpi=200)
    axes[0].imshow(cv2.cvtColor(frame_roi, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Baseline (frame)")
    axes[1].imshow(cv2.cvtColor(reference_roi, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Reference")
    axes[2].imshow(diff_norm, cmap="gray")
    axes[2].set_title("Diff normalized")
    axes[3].imshow(current_mask, cmap="gray")
    axes[3].set_title("Current mask")
    axes[4].imshow(clahe_mask, cmap="gray")
    axes[4].set_title("Mask w/ CLAHE")

    for ax in axes:
        ax.axis("off")

    fig.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved diagnostic to {out_path}")


if __name__ == "__main__":
    main()
