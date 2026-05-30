#!/usr/bin/env python3
"""Apply the CV keyframe segmentation pipeline (ref diff -> Gaussian blur ->
threshold -> connected components with min_area filter) to a specific frame
range, sweeping multiple min_area thresholds.

Mirrors the no_normalize+default branch of
scripts/run_sam2_prompted_flow.py::_build_cv_keyframe_mask_prompts so the
result is directly comparable to base_NNN_prompt_NNN_overlay.jpg debug files.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bubble_tracking.flow_tracking import (
    SegmentationConfig,
    load_roi_json,
    segment_frame,
)
from bubble_tracking.frame_io import sorted_frame_paths
from bubble_tracking.visualization import draw_mask_overlay


def _apply_roi(frame: np.ndarray, roi):
    if roi is None:
        return frame, (0, 0)
    x1, y1, x2, y2 = roi["x1"], roi["y1"], roi["x2"], roi["y2"]
    return frame[y1:y2, x1:x2], (x1, y1)


def _load_roi_mask(path, shape):
    if not path:
        return None
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(path)
    if m.shape[:2] != shape[:2]:
        raise ValueError(f"roi_mask shape {m.shape} != {shape}")
    return m


def _no_normalize_gray(frame_crop, ref_crop, diff_sigma):
    diff = frame_crop.astype(np.float32) - ref_crop.astype(np.float32)
    if diff_sigma > 0:
        diff = cv2.GaussianBlur(diff, (0, 0), sigmaX=diff_sigma, sigmaY=diff_sigma)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    return np.clip(diff_gray + 128.0, 0, 255).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep CV keyframe segmentation on a frame range")
    parser.add_argument("--frames", required=True, help="Source frames dir (relabeled)")
    parser.add_argument("--reference", required=True, help="Reference frame path")
    parser.add_argument("--roi-json", default=None)
    parser.add_argument("--roi-mask", default=None)
    parser.add_argument("--start", type=int, required=True, help="Inclusive frame_idx start")
    parser.add_argument("--end", type=int, required=True, help="Inclusive frame_idx end")
    parser.add_argument("--first-original-frame", type=int, default=71)
    parser.add_argument("--diff-sigma", type=float, default=0.5)
    parser.add_argument("--threshold-offset", type=int, default=7)
    parser.add_argument("--min-areas", type=int, nargs="+", required=True,
                        help="Min-area thresholds to sweep (e.g. 428 68 27)")
    parser.add_argument("--output-dir", required=True, help="Output root dir (created)")
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    args = parser.parse_args()

    frame_paths = sorted_frame_paths(args.frames)
    if not frame_paths:
        raise ValueError(f"No frames in {args.frames}")
    if args.start < 0 or args.end < args.start or args.end >= len(frame_paths):
        raise ValueError(f"Bad range [{args.start},{args.end}] for {len(frame_paths)} frames")

    reference_full = cv2.imread(args.reference)
    if reference_full is None:
        raise FileNotFoundError(args.reference)

    roi = load_roi_json(args.roi_json) if args.roi_json else None
    roi_mask = _load_roi_mask(args.roi_mask, reference_full.shape) if args.roi_mask else None

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    per_min_dirs = {}
    for ma in args.min_areas:
        d = out_root / f"minarea_{int(ma)}"
        d.mkdir(parents=True, exist_ok=True)
        per_min_dirs[int(ma)] = d

    H, W = reference_full.shape[:2]
    for frame_idx in range(args.start, args.end + 1):
        frame_full = cv2.imread(frame_paths[frame_idx])
        if frame_full is None:
            raise RuntimeError(f"Could not read {frame_paths[frame_idx]}")
        frame_crop, _ = _apply_roi(frame_full, roi)
        ref_crop, _ = _apply_roi(reference_full, roi)
        gray = _no_normalize_gray(frame_crop, ref_crop, float(args.diff_sigma))

        relabeled_num = frame_idx + 1
        original_num = relabeled_num + int(args.first_original_frame) - 1
        fname = f"frame_original_{original_num:04d}_relabelled_{relabeled_num:04d}.jpg"

        for min_area in args.min_areas:
            cfg = SegmentationConfig(
                min_area=int(min_area),
                threshold_offset=int(args.threshold_offset),
            )
            labels, components = segment_frame(gray, cfg)
            mask_crop = labels > 0
            if roi is None:
                mask_full = mask_crop.astype(bool)
            else:
                mask_full = np.zeros((H, W), dtype=bool)
                x1, y1, x2, y2 = roi["x1"], roi["y1"], roi["x2"], roi["y2"]
                mask_full[y1:y2, x1:x2] = mask_crop
            if roi_mask is not None:
                mask_full &= (roi_mask > 0)

            inst = np.zeros((H, W), dtype=np.int32)
            inst[mask_full] = 1
            overlay = draw_mask_overlay(frame_full, inst, colors={1: (0, 255, 0)}, alpha=float(args.alpha))

            out_path = per_min_dirs[int(min_area)] / fname
            cv2.imwrite(str(out_path), overlay, [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)])
            print(f"min_area={int(min_area):>4} frame_idx={frame_idx} comps={len(components)} area={int(mask_full.sum())} -> {out_path}")


if __name__ == "__main__":
    main()
