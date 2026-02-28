#!/usr/bin/env python3
"""
Export preprocessed frames (reference diff + optional CLAHE) to a video.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bubble_tracking.flow_tracking import PreprocessConfig, load_roi_json, preprocess_frame_with_debug
from bubble_tracking.frame_io import sorted_frame_paths


def _apply_roi(frame, roi):
    if roi is None:
        return frame
    x1, y1, x2, y2 = roi["x1"], roi["y1"], roi["x2"], roi["y2"]
    return frame[y1:y2, x1:x2]


def _resize_to_max_width(frame, max_width):
    if max_width <= 0:
        return frame
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale = max_width / float(w)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    if new_h % 2 == 1:
        new_h -= 1
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export preprocessed frames to video")
    parser.add_argument("--frames", required=True, help="Frames directory (numeric names)")
    parser.add_argument("--reference", required=True, help="Reference image path")
    parser.add_argument("--output", required=True, help="Output video path (mp4)")
    parser.add_argument("--roi", default=None, help="ROI JSON path (optional)")
    parser.add_argument("--fps", type=float, default=10.0, help="Output video FPS")
    parser.add_argument("--max-width", type=int, default=0, help="Downscale to this width (0 disables)")
    parser.add_argument("--max-frames", type=int, default=0, help="Limit frames (0 disables)")
    parser.add_argument("--diff-sigma", type=float, default=0.5, help="Gaussian sigma for diff smoothing")
    parser.add_argument("--use-clahe", action="store_true", help="Enable CLAHE on normalized diff")
    parser.add_argument("--clahe-clip-limit", type=float, default=2.0, help="CLAHE clip limit")
    parser.add_argument(
        "--clahe-tile-grid",
        type=int,
        nargs=2,
        default=(8, 8),
        metavar=("TILE_W", "TILE_H"),
        help="CLAHE tile grid size (cols rows)",
    )
    parser.add_argument("--grid-rows", type=int, default=8, help="Light matrix grid rows")
    parser.add_argument("--grid-cols", type=int, default=5, help="Light matrix grid cols")
    parser.add_argument("--log-every", type=int, default=200, help="Log every N frames")
    args = parser.parse_args()

    roi = load_roi_json(args.roi) if args.roi else None
    frame_paths = sorted_frame_paths(args.frames)
    if not frame_paths:
        raise ValueError(f"No frames found in {args.frames}")

    reference_full = cv2.imread(args.reference)
    if reference_full is None:
        raise ValueError(f"Failed to read reference image: {args.reference}")
    reference_crop = _apply_roi(reference_full, roi)

    cfg = PreprocessConfig(
        diff_sigma=args.diff_sigma,
        grid_rows=args.grid_rows,
        grid_cols=args.grid_cols,
        use_clahe=args.use_clahe,
        clahe_clip_limit=args.clahe_clip_limit,
        clahe_tile_grid=tuple(args.clahe_tile_grid),
        mode="default",
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = None
    total = len(frame_paths)
    max_frames = args.max_frames if args.max_frames > 0 else total

    for idx, frame_path in enumerate(frame_paths):
        if idx >= max_frames:
            break
        frame_full = cv2.imread(frame_path)
        if frame_full is None:
            raise ValueError(f"Failed to read frame: {frame_path}")
        if frame_full.shape[:2] != reference_full.shape[:2]:
            raise ValueError("Frame and reference image sizes do not match")
        frame_crop = _apply_roi(frame_full, roi)

        normalized = preprocess_frame_with_debug(frame_crop, reference_crop, cfg)["normalized"]
        preview = cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)
        preview = _resize_to_max_width(preview, args.max_width)

        if writer is None:
            h, w = preview.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_path), fourcc, args.fps, (w, h))
            if not writer.isOpened():
                raise RuntimeError("Failed to open video writer")

        writer.write(preview)
        if args.log_every > 0 and (idx + 1) % args.log_every == 0:
            print(f"[progress] wrote {idx + 1}/{max_frames} frames")

    if writer is not None:
        writer.release()
    print(f"Wrote video to {output_path}")


if __name__ == "__main__":
    main()
