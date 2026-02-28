#!/usr/bin/env python3
"""
Summarize flow-tracking outputs into per-second stats and plots.

Reads area_timeseries.json + events.json from a run directory, then writes:
  - per_second_area.csv
  - per_second_events.csv
  - per_second_summary.png
"""

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt


def _bucket_second(frame_idx: int, fps: float) -> int:
    return int(math.floor(frame_idx / fps))


def _load_area(area_path: Path, fps: float, max_frames: int) -> dict:
    data = json.loads(area_path.read_text())
    buckets = {}
    for row in data:
        frame_idx = int(row["frame_idx"])
        if frame_idx >= max_frames:
            continue
        sec = _bucket_second(frame_idx, fps)
        buckets.setdefault(sec, []).append(int(row["total_area_px"]))
    return buckets


def _load_events(events_path: Path, fps: float, max_frames: int) -> dict:
    data = json.loads(events_path.read_text())
    buckets = {}
    for ev in data.get("events", []):
        frame_idx = int(ev.get("frame_idx", 0))
        if frame_idx >= max_frames:
            continue
        sec = _bucket_second(frame_idx, fps)
        buckets.setdefault(sec, []).append(ev.get("type", ""))
    return buckets, data


def _write_area_csv(buckets: dict, output_path: Path) -> None:
    rows = []
    for sec, areas in buckets.items():
        if not areas:
            continue
        rows.append({
            "second": sec,
            "mean_area_px": sum(areas) / len(areas),
            "min_area_px": min(areas),
            "max_area_px": max(areas),
            "num_frames": len(areas),
        })
    rows.sort(key=lambda r: r["second"])
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["second", "mean_area_px", "min_area_px", "max_area_px", "num_frames"],
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_events_csv(buckets: dict, output_path: Path) -> None:
    rows = []
    for sec, events in buckets.items():
        rows.append({
            "second": sec,
            "merge": events.count("merge"),
            "split": events.count("split"),
            "birth": events.count("birth"),
            "death": events.count("death"),
        })
    rows.sort(key=lambda r: r["second"])
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["second", "merge", "split", "birth", "death"],
        )
        writer.writeheader()
        writer.writerows(rows)


def _plot_summary(area_buckets: dict, event_buckets: dict, output_path: Path) -> None:
    seconds = sorted(set(area_buckets.keys()) | set(event_buckets.keys()))
    if not seconds:
        raise ValueError("No data to plot")

    mean_area = []
    merges = []
    splits = []
    for sec in seconds:
        areas = area_buckets.get(sec, [])
        mean_area.append(sum(areas) / len(areas) if areas else 0.0)
        evs = event_buckets.get(sec, [])
        merges.append(evs.count("merge"))
        splits.append(evs.count("split"))

    fig, (ax_area, ax_events) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, dpi=150)

    ax_area.plot(seconds, mean_area, color="black", linewidth=1.5)
    ax_area.set_ylabel("Mean Area (px)")
    ax_area.set_title("Total Bubble Area (per second)")
    ax_area.grid(True, alpha=0.3)

    ax_events.plot(seconds, merges, label="merge/s", color="#1f77b4", linewidth=1.5)
    ax_events.plot(seconds, splits, label="split/s", color="#d62728", linewidth=1.5)
    ax_events.set_xlabel("Second")
    ax_events.set_ylabel("Events / s")
    ax_events.set_title("Merge/Split Frequency (per second)")
    ax_events.grid(True, alpha=0.3)
    ax_events.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize run stats per second")
    parser.add_argument("--run-dir", required=True, help="Run output directory")
    parser.add_argument("--area-json", default=None, help="Path to area_timeseries.json")
    parser.add_argument("--events-json", default=None, help="Path to events.json")
    parser.add_argument("--output", default=None, help="Output directory (default: run-dir)")
    parser.add_argument("--fps", type=float, default=None, help="FPS override")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=900,
        help="Only analyze first N frames (default: 900)",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise ValueError(f"Run dir not found: {run_dir}")

    area_path = Path(args.area_json) if args.area_json else run_dir / "area_timeseries.json"
    events_path = Path(args.events_json) if args.events_json else run_dir / "events.json"
    if not area_path.exists():
        raise ValueError(f"Missing area_timeseries.json: {area_path}")
    if not events_path.exists():
        raise ValueError(f"Missing events.json: {events_path}")

    events_data = json.loads(events_path.read_text())
    fps = args.fps or events_data.get("config", {}).get("fps")
    if fps is None:
        raise ValueError("FPS not found in events.json; provide --fps")

    area_buckets = _load_area(area_path, fps, args.max_frames)
    event_buckets, _ = _load_events(events_path, fps, args.max_frames)

    output_dir = Path(args.output) if args.output else run_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    area_csv = output_dir / "per_second_area.csv"
    events_csv = output_dir / "per_second_events.csv"
    plot_png = output_dir / "per_second_summary.png"

    _write_area_csv(area_buckets, area_csv)
    _write_events_csv(event_buckets, events_csv)
    _plot_summary(area_buckets, event_buckets, plot_png)

    print(f"Wrote: {area_csv}")
    print(f"Wrote: {events_csv}")
    print(f"Wrote: {plot_png}")


if __name__ == "__main__":
    main()
