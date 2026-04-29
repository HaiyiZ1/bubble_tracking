#!/usr/bin/env python3
"""
Export framewise injection-birth and true-vanish-death area totals.

For each frame, writes:
1) frame index
2) time in seconds
3) total injection-birth area in px^2
4) total injection-birth area in mm^2
5) total vanished-death area in px^2
6) total vanished-death area in mm^2
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


DEFAULT_RUN_DIRS = [
    Path("outputs/case1_air1_publish"),
    Path("outputs/case2_air1_publish"),
    Path("outputs/case3_air1_publish"),
]


def _load_json(path: Path) -> dict | list:
    return json.loads(path.read_text())


def _frame_count_from_area(area_rows: List[dict]) -> int:
    if not area_rows:
        return 0
    max_frame = max(int(row.get("frame_idx", -1)) for row in area_rows)
    return max(0, max_frame + 1)


def _aggregate_event_areas(
    events: List[dict],
    frame_count: int,
) -> tuple[List[float], List[float]]:
    birth_px = [0.0] * frame_count
    death_px = [0.0] * frame_count
    for ev in events:
        ev_type = str(ev.get("type", ""))
        frame_idx = int(ev.get("frame_idx", -1))
        if frame_idx < 0 or frame_idx >= frame_count:
            continue
        area_px = float(ev.get("area_px", 0.0) or 0.0)
        if area_px <= 0:
            continue
        # In this pipeline, "birth" means injection birth.
        if ev_type == "birth":
            birth_px[frame_idx] += area_px
        # In this pipeline, "death" means true vanish death.
        elif ev_type == "death":
            death_px[frame_idx] += area_px
    return birth_px, death_px


def _export_run(run_dir: Path, output_name: str) -> Path:
    events_path = run_dir / "events.json"
    area_path = run_dir / "area_timeseries.json"
    if not events_path.exists():
        raise FileNotFoundError(f"Missing events.json: {events_path}")
    if not area_path.exists():
        raise FileNotFoundError(f"Missing area_timeseries.json: {area_path}")

    events_data = _load_json(events_path)
    area_rows = _load_json(area_path)
    if not isinstance(events_data, dict):
        raise ValueError(f"Invalid events.json format: {events_path}")
    if not isinstance(area_rows, list):
        raise ValueError(f"Invalid area_timeseries.json format: {area_path}")

    fps = float(events_data.get("config", {}).get("fps", 0.0))
    mm_per_pixel = float(events_data.get("config", {}).get("mm_per_pixel", 0.0))
    if fps <= 0:
        raise ValueError(f"Invalid fps in {events_path}: {fps}")
    if mm_per_pixel <= 0:
        raise ValueError(f"Invalid mm_per_pixel in {events_path}: {mm_per_pixel}")

    frame_count = _frame_count_from_area(area_rows)
    events = events_data.get("events", [])
    if not isinstance(events, list):
        raise ValueError(f"Invalid events list in {events_path}")
    birth_px, death_px = _aggregate_event_areas(events, frame_count)

    mm2_scale = mm_per_pixel ** 2
    output_csv = run_dir / output_name
    with output_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame_idx",
            "time_s",
            "total_injection_birth_area_px2",
            "total_injection_birth_area_mm2",
            "total_vanished_death_area_px2",
            "total_vanished_death_area_mm2",
        ])
        for frame_idx in range(frame_count):
            bpx = float(birth_px[frame_idx])
            dpx = float(death_px[frame_idx])
            writer.writerow([
                frame_idx,
                f"{frame_idx / fps:.6f}",
                f"{bpx:.6f}",
                f"{bpx * mm2_scale:.6f}",
                f"{dpx:.6f}",
                f"{dpx * mm2_scale:.6f}",
            ])
    return output_csv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export per-frame injection-birth and vanished-death area totals for publish runs."
    )
    parser.add_argument(
        "--run-dir",
        action="append",
        default=[],
        help="Run directory (repeatable). Defaults to case1/2/3 air1 publish runs.",
    )
    parser.add_argument(
        "--output-name",
        default="injection_vanished_area_timeseries.csv",
        help="Output CSV filename written into each run-dir.",
    )
    args = parser.parse_args()

    run_dirs = [Path(p) for p in args.run_dir] if args.run_dir else DEFAULT_RUN_DIRS
    for run_dir in run_dirs:
        if not run_dir.exists():
            raise FileNotFoundError(f"Run dir not found: {run_dir}")
        out_csv = _export_run(run_dir=run_dir, output_name=args.output_name)
        print(f"Wrote: {out_csv}")


if __name__ == "__main__":
    main()
