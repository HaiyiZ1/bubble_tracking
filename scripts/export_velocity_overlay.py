#!/usr/bin/env python3
"""
Export velocity-vector overlays on original frames from a completed tracking run.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import Dict, List

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bubble_tracking.frame_io import load_mask, sorted_frame_paths
from bubble_tracking.visualization import draw_mask_overlay, generate_colors


def _read_tracks(tracks_csv: Path) -> tuple[Dict[int, List[dict]], List[int]]:
    by_frame: Dict[int, List[dict]] = {}
    object_ids = set()
    with tracks_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                frame_idx = int(row["frame_idx"])
                obj_id = int(row["object_id"])
            except Exception:
                continue

            def _f(key: str):
                val = row.get(key, "")
                if val is None or val == "":
                    return None
                try:
                    return float(val)
                except ValueError:
                    return None

            entry = {
                "object_id": obj_id,
                "centroid_x": _f("centroid_x"),
                "centroid_y": _f("centroid_y"),
                "vx_px_s": _f("velocity_x_px_s"),
                "vy_px_s": _f("velocity_y_px_s"),
                "vmag_px_s": _f("velocity_mag_px_s"),
                "vx_mm_s": _f("velocity_x_mm_s"),
                "vy_mm_s": _f("velocity_y_mm_s"),
                "vmag_mm_s": _f("velocity_mag_mm_s"),
            }
            by_frame.setdefault(frame_idx, []).append(entry)
            object_ids.add(obj_id)
    return by_frame, sorted(object_ids)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export velocity overlay video from a tracking run")
    parser.add_argument("--run-dir", required=True, help="Run output directory (contains masks/ and tracks.csv)")
    parser.add_argument("--frames", required=True, help="Original frames directory")
    parser.add_argument("--output", default=None, help="Output video path (default: <run-dir>/velocity_overlay.mp4)")
    parser.add_argument("--fps", type=float, default=10.0, help="Output video FPS")
    parser.add_argument("--alpha", type=float, default=0.35, help="Mask overlay alpha")
    parser.add_argument("--arrow-seconds", type=float, default=0.25, help="Arrow length time horizon in seconds")
    parser.add_argument("--line-thickness", type=int, default=2, help="Arrow line thickness")
    parser.add_argument("--tip-length", type=float, default=0.3, help="Arrow tip length")
    parser.add_argument("--label-units", choices=["mm_s", "px_s"], default="mm_s", help="Velocity units in labels")
    parser.add_argument("--overlay-dir", default=None, help="Optional directory to save overlay frames")
    parser.add_argument("--start-frame", type=int, default=0, help="Start frame index (0-based)")
    parser.add_argument("--max-frames", type=int, default=0, help="Max frames to export (0 = all)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    frames_dir = Path(args.frames)
    tracks_csv = run_dir / "tracks.csv"
    masks_dir = run_dir / "masks"

    if not tracks_csv.exists():
        raise FileNotFoundError(f"tracks.csv not found: {tracks_csv}")
    if not masks_dir.exists():
        raise FileNotFoundError(f"masks dir not found: {masks_dir}")

    frame_paths = sorted_frame_paths(str(frames_dir))
    if not frame_paths:
        raise ValueError(f"No frames found in {frames_dir}")

    by_frame, object_ids = _read_tracks(tracks_csv)
    color_list = generate_colors(max(1, len(object_ids)))
    colors = {obj_id: color_list[i % len(color_list)] for i, obj_id in enumerate(object_ids)}

    out_path = Path(args.output) if args.output else (run_dir / "velocity_overlay.mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.overlay_dir:
        Path(args.overlay_dir).mkdir(parents=True, exist_ok=True)

    first = cv2.imread(frame_paths[0])
    if first is None:
        raise ValueError(f"Failed to read frame: {frame_paths[0]}")
    h, w = first.shape[:2]
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (w, h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open writer: {out_path}")

    start = max(0, int(args.start_frame))
    end = len(frame_paths)
    if args.max_frames > 0:
        end = min(end, start + int(args.max_frames))

    for frame_idx in range(start, end):
        frame = cv2.imread(frame_paths[frame_idx])
        if frame is None:
            continue

        mask_path = masks_dir / f"{frame_idx:06d}.png"
        if mask_path.exists():
            mask = load_mask(str(mask_path))
            frame = draw_mask_overlay(frame, mask, colors=colors, alpha=args.alpha)

        for item in by_frame.get(frame_idx, []):
            cx = item["centroid_x"]
            cy = item["centroid_y"]
            vx_px_s = item["vx_px_s"]
            vy_px_s = item["vy_px_s"]
            if cx is None or cy is None or vx_px_s is None or vy_px_s is None:
                continue

            obj_id = int(item["object_id"])
            color = colors.get(obj_id, (255, 255, 255))
            p0 = (int(round(cx)), int(round(cy)))
            dx = vx_px_s * float(args.arrow_seconds)
            dy = vy_px_s * float(args.arrow_seconds)
            p1 = (int(round(cx + dx)), int(round(cy + dy)))

            cv2.arrowedLine(
                frame,
                p0,
                p1,
                color=color,
                thickness=max(1, int(args.line_thickness)),
                tipLength=float(args.tip_length),
            )
            cv2.circle(frame, p0, 3, color, -1)

            if args.label_units == "mm_s":
                vmag = item["vmag_mm_s"]
                units = "mm/s"
            else:
                vmag = item["vmag_px_s"]
                units = "px/s"
            if vmag is not None:
                label = f"#{obj_id} {vmag:.2f} {units}"
                cv2.putText(
                    frame,
                    label,
                    (p0[0] + 6, p0[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    cv2.LINE_AA,
                )

        cv2.putText(
            frame,
            f"Frame {frame_idx}",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        writer.write(frame)
        if args.overlay_dir:
            cv2.imwrite(str(Path(args.overlay_dir) / f"{frame_idx:06d}.jpg"), frame)

    writer.release()
    print(f"Wrote velocity overlay video: {out_path}")


if __name__ == "__main__":
    main()
