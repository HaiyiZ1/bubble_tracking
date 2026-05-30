#!/usr/bin/env python3
"""Export a raw-frames publishing video with an elapsed-time banner."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import cv2

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bubble_tracking.frame_io import sorted_frame_paths


def _draw_text_with_box(
    frame,
    text: str,
    origin: tuple[int, int],
    font_scale: float,
    thickness: int,
    text_color: tuple[int, int, int] = (255, 255, 255),
    box_color: tuple[int, int, int] = (0, 0, 0),
    padding: int = 8,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = origin
    x1 = max(0, x - padding)
    y1 = max(0, y - text_h - padding)
    x2 = min(frame.shape[1] - 1, x + text_w + padding)
    y2 = min(frame.shape[0] - 1, y + baseline + padding)
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, -1)
    cv2.putText(frame, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)


def _format_elapsed(elapsed_seconds: float, resolution: str) -> str:
    if resolution == "sec":
        total = int(round(elapsed_seconds))
        minutes, seconds = divmod(total, 60)
        return f"Time elapsed: {minutes} min {seconds} sec."
    return f"Time elapsed: {elapsed_seconds / 60.0:.2f} min."


def main() -> None:
    parser = argparse.ArgumentParser(description="Export raw publishing video with elapsed-time banner")
    parser.add_argument("--frames", required=True, help="Frames directory")
    parser.add_argument("--output", required=True, help="Output video path")
    parser.add_argument("--fps", type=float, default=4.0, help="Acquisition FPS used for elapsed time")
    parser.add_argument("--speedup", type=float, default=10.0, help="Speed multiplier for playback")
    parser.add_argument("--max-minutes", type=float, default=5.0, help="Maximum real-time minutes to render")
    parser.add_argument(
        "--skip-frames",
        type=int,
        default=0,
        help="Number of leading frames to drop before rendering (matches publishing relabel offset).",
    )
    parser.add_argument(
        "--time-resolution",
        choices=("min", "sec"),
        default="sec",
        help="Elapsed-time format: 'min' (x.xx min) or 'sec' (x min y sec)",
    )
    args = parser.parse_args()

    frame_paths = sorted_frame_paths(str(Path(args.frames)))
    if not frame_paths:
        raise ValueError(f"No frames found in {args.frames}")

    if args.skip_frames < 0:
        raise ValueError("--skip-frames must be >= 0")
    if args.skip_frames >= len(frame_paths):
        raise ValueError(
            f"--skip-frames {args.skip_frames} >= total frames {len(frame_paths)}"
        )
    frame_paths = frame_paths[args.skip_frames:]

    max_frames = int(round(float(args.fps) * float(args.max_minutes) * 60.0))
    if max_frames <= 0:
        raise ValueError("--max-minutes and --fps must yield at least one frame")
    frame_paths = frame_paths[:max_frames]

    first = cv2.imread(frame_paths[0])
    if first is None:
        raise ValueError(f"Failed to read {frame_paths[0]}")
    height, width = first.shape[:2]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(args.fps) * float(args.speedup),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open writer: {output_path}")

    info_scale = max(0.65, min(1.0, width / 2200.0))
    info_thickness = max(2, int(round(info_scale * 2)))
    font = cv2.FONT_HERSHEY_SIMPLEX
    (_, text_h), _ = cv2.getTextSize("Mg", font, info_scale, info_thickness)
    text_y = text_h + 16

    try:
        for frame_idx, frame_path in enumerate(frame_paths):
            frame = cv2.imread(frame_path)
            if frame is None:
                raise ValueError(f"Failed to read {frame_path}")
            elapsed = float(frame_idx) / float(args.fps)
            _draw_text_with_box(
                frame,
                _format_elapsed(elapsed, args.time_resolution),
                (18, text_y),
                font_scale=info_scale,
                thickness=info_thickness,
                padding=8,
            )
            writer.write(frame)
    finally:
        writer.release()

    print(f"Wrote raw publishing video: {output_path}")


if __name__ == "__main__":
    main()
