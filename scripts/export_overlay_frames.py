#!/usr/bin/env python3
"""Render per-frame PNG overlays (mask + bubble IDs) for a frame range."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import cv2

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bubble_tracking.frame_io import load_mask, sorted_frame_paths
from bubble_tracking.track_builder import load_tracks
from bubble_tracking.visualization import draw_mask_overlay, draw_tracking_info, generate_colors


def main() -> None:
    parser = argparse.ArgumentParser(description="Export overlay PNGs for a frame range")
    parser.add_argument("--run-dir", required=True, help="Run directory (must contain tracks.json and masks/)")
    parser.add_argument("--frames", required=True, help="Source frames directory")
    parser.add_argument("--start", type=int, required=True, help="Start frame_idx (inclusive, 0-indexed)")
    parser.add_argument("--end", type=int, required=True, help="End frame_idx (inclusive, 0-indexed)")
    parser.add_argument("--output-dir", default=None, help="Output dir (default: <run-dir>/overlay_frames)")
    parser.add_argument("--alpha", type=float, default=0.4, help="Mask overlay alpha")
    parser.add_argument("--font-scale", type=float, default=1.2)
    parser.add_argument("--font-thickness", type=int, default=3)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    tracks_path = run_dir / "tracks.json"
    masks_dir = run_dir / "masks"
    if not tracks_path.exists():
        raise FileNotFoundError(f"tracks.json not found: {tracks_path}")
    if not masks_dir.exists():
        raise FileNotFoundError(f"masks directory not found: {masks_dir}")

    frame_paths = sorted_frame_paths(str(Path(args.frames)))
    if not frame_paths:
        raise ValueError(f"No frames found in {args.frames}")

    if args.start < 0 or args.end < args.start or args.end >= len(frame_paths):
        raise ValueError(
            f"Invalid range [{args.start}, {args.end}] for {len(frame_paths)} frames"
        )

    result = load_tracks(str(tracks_path))
    colors = {obj_id: generate_colors(1, seed=obj_id * 7)[0] for obj_id in result.objects}

    output_dir = Path(args.output_dir) if args.output_dir else (run_dir / "overlay_frames")
    output_dir.mkdir(parents=True, exist_ok=True)

    for frame_idx in range(args.start, args.end + 1):
        frame_path = frame_paths[frame_idx]
        frame = cv2.imread(frame_path)
        if frame is None:
            raise ValueError(f"Failed to read frame: {frame_path}")

        mask_path = masks_dir / f"{frame_idx:06d}.png"
        if mask_path.exists():
            mask = load_mask(str(mask_path))
            frame = draw_mask_overlay(frame, mask, colors=colors, alpha=float(args.alpha))

        frame = draw_tracking_info(
            frame,
            result,
            frame_idx,
            colors=colors,
            font_scale=float(args.font_scale),
            font_thickness=int(args.font_thickness),
        )

        out_path = output_dir / f"overlay_{frame_idx:06d}.png"
        cv2.imwrite(str(out_path), frame)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
