#!/usr/bin/env python3
"""
Plot moving-window vanished-bubble area from true death events.

For each integer second t, compute:
  total vanished area in [t - half_window, t + half_window]

Outputs four plots:
  - 2s window, x-axis in seconds
  - 2s window, x-axis in frames
  - 5s window, x-axis in seconds
  - 5s window, x-axis in frames
"""

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np


WINDOWS_SECONDS = [2.0, 5.0]


def _load_events(events_path: Path) -> dict:
    return json.loads(events_path.read_text())


def _load_total_frames(area_path: Path, fallback_max_frame: int) -> int:
    if area_path.exists():
        area_rows = json.loads(area_path.read_text())
        if area_rows:
            max_frame = max(int(row.get("frame_idx", -1)) for row in area_rows)
            return max(0, max_frame + 1)
    return max(0, fallback_max_frame + 1)


def _extract_death_events(events_data: dict, max_frame_exclusive: int) -> List[Tuple[int, int]]:
    rows: List[Tuple[int, int]] = []
    for ev in events_data.get("events", []):
        if str(ev.get("type", "")) != "death":
            continue
        frame_idx = int(ev.get("frame_idx", -1))
        if frame_idx < 0 or frame_idx >= int(max_frame_exclusive):
            continue
        area_px = int(ev.get("area_px", 0))
        if area_px <= 0:
            continue
        rows.append((frame_idx, area_px))
    rows.sort(key=lambda x: x[0])
    return rows


def _moving_window_totals_per_second(
    deaths: List[Tuple[int, int]],
    fps: float,
    total_frames: int,
    window_seconds: float,
) -> Tuple[List[int], List[float], List[float]]:
    if fps <= 0:
        raise ValueError("FPS must be > 0")
    if total_frames <= 0:
        return [], [], []

    max_second = int(math.floor((total_frames - 1) / fps))
    seconds = list(range(max_second + 1))
    center_frames = [sec * fps for sec in seconds]
    half_window_frames = (window_seconds / 2.0) * fps

    totals: List[float] = []
    for center_frame in center_frames:
        total = 0.0
        for death_frame, death_area in deaths:
            if abs(float(death_frame) - float(center_frame)) <= half_window_frames:
                total += float(death_area)
        totals.append(total)
    return seconds, center_frames, totals


def _plot_series(
    x: List[float],
    y: List[float],
    xlabel: str,
    title: str,
    output_path: Path,
    log_y: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=160)
    if log_y:
        # Log axis cannot show non-positive values; keep those as NaN so gaps are explicit.
        y_plot = np.asarray([float(v) if float(v) > 0.0 else np.nan for v in y], dtype=np.float64)
    else:
        y_plot = np.asarray(y, dtype=np.float64)
    ax.plot(x, y_plot, color="#1f77b4", linewidth=1.8)
    ax.set_xlabel(xlabel)
    if log_y:
        ax.set_yscale("log")
        ax.set_ylabel("Vanished Bubble Area (px, log scale)")
    else:
        ax.set_ylabel("Vanished Bubble Area (px)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _write_csv(
    output_path: Path,
    seconds: List[int],
    center_frames: List[float],
    totals_by_window: Dict[float, List[float]],
) -> None:
    fieldnames = [
        "second",
        "center_frame",
        "vanished_area_window_2s_px",
        "vanished_area_window_5s_px",
    ]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, second in enumerate(seconds):
            row = {
                "second": int(second),
                "center_frame": float(center_frames[idx]),
                "vanished_area_window_2s_px": float(totals_by_window.get(2.0, [0.0] * len(seconds))[idx]),
                "vanished_area_window_5s_px": float(totals_by_window.get(5.0, [0.0] * len(seconds))[idx]),
            }
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot moving-window vanished-bubble area from true death events"
    )
    parser.add_argument("--run-dir", required=True, help="Run output directory")
    parser.add_argument("--events-json", default=None, help="Path to events.json")
    parser.add_argument("--area-json", default=None, help="Path to area_timeseries.json")
    parser.add_argument("--output", default=None, help="Output directory (default: run-dir)")
    parser.add_argument("--fps", type=float, default=None, help="FPS override")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap: only include events and timeline before this frame index",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise ValueError(f"Run dir not found: {run_dir}")

    events_path = Path(args.events_json) if args.events_json else run_dir / "events.json"
    area_path = Path(args.area_json) if args.area_json else run_dir / "area_timeseries.json"
    if not events_path.exists():
        raise ValueError(f"Missing events.json: {events_path}")

    events_data = _load_events(events_path)
    fps = args.fps or events_data.get("config", {}).get("fps")
    if fps is None:
        raise ValueError("FPS not found in events.json; provide --fps")
    fps = float(fps)
    if fps <= 0:
        raise ValueError("FPS must be > 0")

    fallback_max_frame = max(
        [int(ev.get("frame_idx", -1)) for ev in events_data.get("events", [])] + [-1]
    )
    total_frames = _load_total_frames(area_path, fallback_max_frame)
    if args.max_frames is not None:
        total_frames = min(total_frames, max(0, int(args.max_frames)))

    deaths = _extract_death_events(events_data, max_frame_exclusive=total_frames)

    output_dir = Path(args.output) if args.output else run_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    totals_by_window: Dict[float, List[float]] = {}
    seconds: List[int] = []
    center_frames: List[float] = []
    for window_seconds in WINDOWS_SECONDS:
        sec, center_fr, totals = _moving_window_totals_per_second(
            deaths=deaths,
            fps=fps,
            total_frames=total_frames,
            window_seconds=window_seconds,
        )
        totals_by_window[window_seconds] = totals
        if not seconds:
            seconds = sec
            center_frames = center_fr

    csv_path = output_dir / "vanished_area_moving_window.csv"
    _write_csv(
        output_path=csv_path,
        seconds=seconds,
        center_frames=center_frames,
        totals_by_window=totals_by_window,
    )

    for window_seconds in WINDOWS_SECONDS:
        half = window_seconds / 2.0
        totals = totals_by_window[window_seconds]
        window_tag = f"{int(window_seconds)}s"

        sec_plot_path = output_dir / f"vanished_area_window_{window_tag}_x_seconds.png"
        _plot_series(
            x=[float(s) for s in seconds],
            y=totals,
            xlabel="Time (s)",
            title=f"Vanished Bubble Area ({window_tag} window, +/-{half:.1f}s)",
            output_path=sec_plot_path,
        )
        sec_plot_path_logy = output_dir / f"vanished_area_window_{window_tag}_x_seconds_logy.png"
        _plot_series(
            x=[float(s) for s in seconds],
            y=totals,
            xlabel="Time (s)",
            title=f"Vanished Bubble Area ({window_tag} window, +/-{half:.1f}s, log y)",
            output_path=sec_plot_path_logy,
            log_y=True,
        )

        frame_plot_path = output_dir / f"vanished_area_window_{window_tag}_x_frames.png"
        _plot_series(
            x=center_frames,
            y=totals,
            xlabel="Frame Index",
            title=f"Vanished Bubble Area ({window_tag} window, +/-{half:.1f}s)",
            output_path=frame_plot_path,
        )
        frame_plot_path_logy = output_dir / f"vanished_area_window_{window_tag}_x_frames_logy.png"
        _plot_series(
            x=center_frames,
            y=totals,
            xlabel="Frame Index",
            title=f"Vanished Bubble Area ({window_tag} window, +/-{half:.1f}s, log y)",
            output_path=frame_plot_path_logy,
            log_y=True,
        )

    print(f"Wrote: {csv_path}")
    for window_seconds in WINDOWS_SECONDS:
        window_tag = f"{int(window_seconds)}s"
        print(f"Wrote: {output_dir / f'vanished_area_window_{window_tag}_x_seconds.png'}")
        print(f"Wrote: {output_dir / f'vanished_area_window_{window_tag}_x_seconds_logy.png'}")
        print(f"Wrote: {output_dir / f'vanished_area_window_{window_tag}_x_frames.png'}")
        print(f"Wrote: {output_dir / f'vanished_area_window_{window_tag}_x_frames_logy.png'}")
    print(f"Deaths counted: {len(deaths)}")
    print(f"FPS used: {fps}")


if __name__ == "__main__":
    main()
