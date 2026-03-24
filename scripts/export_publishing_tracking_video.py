#!/usr/bin/env python3
"""
Export publishing-ready tracking overlay videos from a completed run.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import cv2

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bubble_tracking.frame_io import load_mask, sorted_frame_paths
from bubble_tracking.publishing import PUBLISHING_CASES, get_publishing_case
from bubble_tracking.track_builder import load_tracks
from bubble_tracking.visualization import draw_mask_overlay, draw_tracking_info, generate_colors


def _resolve_title(case_id: str | None, title: str | None) -> str:
    if title:
        return title
    if case_id:
        return get_publishing_case(case_id).title
    raise ValueError("Provide either --case or --title")


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


def _add_publishing_banner(frame, title: str, elapsed_minutes: float) -> None:
    height, width = frame.shape[:2]
    overlay = frame.copy()
    banner_height = max(72, int(round(height * 0.1)))
    cv2.rectangle(overlay, (0, 0), (width - 1, banner_height), (0, 0, 0), -1)
    frame[:] = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)

    title_scale = max(0.8, min(1.4, width / 1800.0))
    info_scale = max(0.65, min(1.0, width / 2200.0))
    title_thickness = max(2, int(round(title_scale * 2)))
    info_thickness = max(2, int(round(info_scale * 2)))

    font = cv2.FONT_HERSHEY_SIMPLEX
    (title_w, title_h), _ = cv2.getTextSize(title, font, title_scale, title_thickness)
    title_x = max(12, (width - title_w) // 2)
    title_y = max(32, int(round(banner_height * 0.45)))
    time_y = min(height - 12, int(round(banner_height * 0.84)))

    _draw_text_with_box(
        frame,
        title,
        (title_x, title_y),
        font_scale=title_scale,
        thickness=title_thickness,
        padding=10,
    )
    _draw_text_with_box(
        frame,
        f"Time elapsed: {elapsed_minutes:.2f} min.",
        (18, time_y),
        font_scale=info_scale,
        thickness=info_thickness,
        padding=8,
    )


def _render_video(
    frame_paths: list[str],
    masks_dir: Path,
    result,
    title: str,
    output_path: Path,
    output_fps: float,
    time_fps: float,
    alpha: float,
) -> None:
    first_frame = cv2.imread(frame_paths[0])
    if first_frame is None:
        raise ValueError(f"Failed to read frame: {frame_paths[0]}")
    height, width = first_frame.shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(output_fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open writer: {output_path}")

    colors = {
        obj_id: generate_colors(1, seed=obj_id * 7)[0]
        for obj_id in result.objects
    }

    try:
        for frame_idx, frame_path in enumerate(frame_paths):
            frame = cv2.imread(frame_path)
            if frame is None:
                raise ValueError(f"Failed to read frame: {frame_path}")

            mask_path = masks_dir / f"{frame_idx:06d}.png"
            if mask_path.exists():
                mask = load_mask(str(mask_path))
                frame = draw_mask_overlay(frame, mask, colors=colors, alpha=alpha)

            frame = draw_tracking_info(
                frame,
                result,
                frame_idx,
                colors=colors,
                font_scale=0.9,
                font_thickness=2,
            )
            _add_publishing_banner(
                frame,
                title=title,
                elapsed_minutes=(float(frame_idx) / float(time_fps)) / 60.0,
            )
            writer.write(frame)
    finally:
        writer.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="Export publishing-ready tracking overlay videos")
    parser.add_argument("--run-dir", required=True, help="Tracking run directory with tracks.json and masks/")
    parser.add_argument("--frames", required=True, help="Relabeled frames directory used for publishing")
    parser.add_argument(
        "--case",
        choices=sorted(PUBLISHING_CASES),
        default=None,
        help="Publishing case preset for default title",
    )
    parser.add_argument("--title", default=None, help="Override title text")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for publishing videos (default: <run-dir>/publishing_overlay)",
    )
    parser.add_argument("--fps", type=float, default=4.0, help="Acquisition FPS used for elapsed time and 1x output")
    parser.add_argument("--speedup", type=float, default=10.0, help="Speed multiplier for the fast export")
    parser.add_argument("--max-minutes", type=float, default=5.0, help="Maximum duration to render from the start")
    parser.add_argument("--alpha", type=float, default=0.4, help="Mask overlay alpha")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    tracks_path = run_dir / "tracks.json"
    masks_dir = run_dir / "masks"
    if not tracks_path.exists():
        raise FileNotFoundError(f"tracks.json not found: {tracks_path}")
    if not masks_dir.exists():
        raise FileNotFoundError(f"masks directory not found: {masks_dir}")

    frames_dir = Path(args.frames)
    frame_paths = sorted_frame_paths(str(frames_dir))
    if not frame_paths:
        raise ValueError(f"No frames found in {frames_dir}")

    max_frames = int(round(float(args.fps) * float(args.max_minutes) * 60.0))
    if max_frames <= 0:
        raise ValueError("--max-minutes and --fps must yield at least one frame")
    frame_paths = frame_paths[:max_frames]

    title = _resolve_title(case_id=args.case, title=args.title)
    result = load_tracks(str(tracks_path))
    output_dir = Path(args.output_dir) if args.output_dir else (run_dir / "publishing_overlay")
    output_dir.mkdir(parents=True, exist_ok=True)

    exports = [
        (1.0, float(args.fps), output_dir / "tracking_overlay_publish_1x.mp4"),
        (
            float(args.speedup),
            float(args.fps) * float(args.speedup),
            output_dir / f"tracking_overlay_publish_{int(round(float(args.speedup)))}x.mp4",
        ),
    ]

    for _, output_fps, output_path in exports:
        _render_video(
            frame_paths=frame_paths,
            masks_dir=masks_dir,
            result=result,
            title=title,
            output_path=output_path,
            output_fps=output_fps,
            time_fps=float(args.fps),
            alpha=float(args.alpha),
        )
        print(f"Wrote publishing overlay video: {output_path}")


if __name__ == "__main__":
    main()
