#!/usr/bin/env python3
"""Export per-bubble area, maximum-velocity, and lineage summaries from tracks.csv/tracks.json."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


def _to_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = value.strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _load_run_config(run_dir: Path) -> dict:
    config_path = run_dir / "sam2_prompted_flow_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Run config not found: {config_path}")
    return json.loads(config_path.read_text())


def _load_tracks(tracks_csv: Path) -> Dict[int, List[dict]]:
    by_object: Dict[int, List[dict]] = {}
    with tracks_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            obj_id = int(row["object_id"])
            by_object.setdefault(obj_id, []).append(row)
    return by_object


def _load_lineage(tracks_json: Path) -> Dict[int, dict]:
    if not tracks_json.exists():
        return {}
    data = json.loads(tracks_json.read_text())
    objects = data.get("objects", {})
    out: Dict[int, dict] = {}
    for obj_id_text, payload in objects.items():
        try:
            obj_id = int(obj_id_text)
        except (TypeError, ValueError):
            continue
        out[obj_id] = {
            "birth_reason": payload.get("birth_reason", ""),
            "birth_frame_idx": payload.get("birth_frame_idx", ""),
            "birth_parent_ids": [int(v) for v in payload.get("birth_parent_ids", [])],
            "death_reason": payload.get("death_reason", ""),
            "death_frame_idx": payload.get("death_frame_idx", ""),
            "death_child_ids": [int(v) for v in payload.get("death_child_ids", [])],
        }
    return out


def _join_ids(values: List[int]) -> str:
    if not values:
        return ""
    return "|".join(str(int(v)) for v in values)


def _summarize_object(rows: List[dict], fps: float, mm_per_pixel: float, lineage: dict | None) -> dict:
    frame_indices = sorted(int(row["frame_idx"]) for row in rows)
    areas_px = [float(row["area_px"]) for row in rows]

    vx_px_s = [abs(v) for row in rows if (v := _to_float(row.get("velocity_x_px_s"))) is not None]
    vy_px_s = [abs(v) for row in rows if (v := _to_float(row.get("velocity_y_px_s"))) is not None]
    vmag_px_s = [float(v) for row in rows if (v := _to_float(row.get("velocity_mag_px_s"))) is not None]

    mean_area_px = sum(areas_px) / len(areas_px)
    max_area_px = max(areas_px)
    max_vx_px_frame = (max(vx_px_s) / fps) if vx_px_s else 0.0
    max_vy_px_frame = (max(vy_px_s) / fps) if vy_px_s else 0.0
    max_vmag_px_frame = (max(vmag_px_s) / fps) if vmag_px_s else 0.0

    area_scale = mm_per_pixel ** 2
    return {
        "bubble_ID": int(rows[0]["object_id"]),
        "A_bubble_avg in pixel^2": mean_area_px,
        "A_bubble_avg in mm^2": mean_area_px * area_scale,
        "A_bubble_max in pixel^2": max_area_px,
        "A_bubble_max in mm^2": max_area_px * area_scale,
        "max_velocity_x in pixel/frame": max_vx_px_frame,
        "max_velocity_x in mm/s": max_vx_px_frame * fps * mm_per_pixel,
        "max_velocity_y in pixel/frame": max_vy_px_frame,
        "max_velocity_y in mm/s": max_vy_px_frame * fps * mm_per_pixel,
        "max_velocity in pixel/frame": max_vmag_px_frame,
        "max_velocity in mm/s": max_vmag_px_frame * fps * mm_per_pixel,
        "NO_start_frame": frame_indices[0],
        "NO_end_frame": frame_indices[-1],
        "frame_amount_of_appearance": len(rows),
        "time_of_appearance in s": len(rows) / fps,
        "birth_reason": str((lineage or {}).get("birth_reason", "")),
        "birth_frame_idx": (lineage or {}).get("birth_frame_idx", ""),
        "birth_parent_ids": _join_ids((lineage or {}).get("birth_parent_ids", [])),
        "death_reason": str((lineage or {}).get("death_reason", "")),
        "death_frame_idx": (lineage or {}).get("death_frame_idx", ""),
        "death_child_ids": _join_ids((lineage or {}).get("death_child_ids", [])),
    }


def _write_summary(output_csv: Path, summaries: List[dict]) -> None:
    fieldnames = [
        "bubble_ID",
        "A_bubble_avg in pixel^2",
        "A_bubble_avg in mm^2",
        "A_bubble_max in pixel^2",
        "A_bubble_max in mm^2",
        "max_velocity_x in pixel/frame",
        "max_velocity_x in mm/s",
        "max_velocity_y in pixel/frame",
        "max_velocity_y in mm/s",
        "max_velocity in pixel/frame",
        "max_velocity in mm/s",
        "NO_start_frame",
        "NO_end_frame",
        "frame_amount_of_appearance",
        "time_of_appearance in s",
        "birth_reason",
        "birth_frame_idx",
        "birth_parent_ids",
        "death_reason",
        "death_frame_idx",
        "death_child_ids",
    ]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summaries:
            writer.writerow({
                "bubble_ID": row["bubble_ID"],
                "A_bubble_avg in pixel^2": f"{row['A_bubble_avg in pixel^2']:.6f}",
                "A_bubble_avg in mm^2": f"{row['A_bubble_avg in mm^2']:.6f}",
                "A_bubble_max in pixel^2": f"{row['A_bubble_max in pixel^2']:.6f}",
                "A_bubble_max in mm^2": f"{row['A_bubble_max in mm^2']:.6f}",
                "max_velocity_x in pixel/frame": f"{row['max_velocity_x in pixel/frame']:.6f}",
                "max_velocity_x in mm/s": f"{row['max_velocity_x in mm/s']:.6f}",
                "max_velocity_y in pixel/frame": f"{row['max_velocity_y in pixel/frame']:.6f}",
                "max_velocity_y in mm/s": f"{row['max_velocity_y in mm/s']:.6f}",
                "max_velocity in pixel/frame": f"{row['max_velocity in pixel/frame']:.6f}",
                "max_velocity in mm/s": f"{row['max_velocity in mm/s']:.6f}",
                "NO_start_frame": row["NO_start_frame"],
                "NO_end_frame": row["NO_end_frame"],
                "frame_amount_of_appearance": row["frame_amount_of_appearance"],
                "time_of_appearance in s": f"{row['time_of_appearance in s']:.6f}",
                "birth_reason": row["birth_reason"],
                "birth_frame_idx": row["birth_frame_idx"],
                "birth_parent_ids": row["birth_parent_ids"],
                "death_reason": row["death_reason"],
                "death_frame_idx": row["death_frame_idx"],
                "death_child_ids": row["death_child_ids"],
            })


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export case-level per-bubble area and max-velocity summary CSVs."
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory where summary CSVs are written (default: outputs)",
    )
    args = parser.parse_args()

    runs = {
        "case1_air1_publish_A_bubble_max_velocity.csv": Path("outputs/case1_air1_publish"),
        "case2_air1_publish_A_bubble_max_velocity.csv": Path("outputs/case2_air1_publish"),
        "case3_air1_publish_A_bubble_max_velocity.csv": Path("outputs/case3_air1_publish"),
    }

    output_dir = Path(args.output_dir)
    for output_name, run_dir in runs.items():
        config = _load_run_config(run_dir)
        tracking_cfg = config.get("tracking", {})
        fps = float(tracking_cfg.get("fps", 4.0))
        mm_per_pixel = float(tracking_cfg["mm_per_pixel"])

        tracks_csv = run_dir / "tracks.csv"
        tracks_json = run_dir / "tracks.json"
        by_object = _load_tracks(tracks_csv)
        lineage_by_object = _load_lineage(tracks_json)
        summaries = [
            _summarize_object(
                rows=rows,
                fps=fps,
                mm_per_pixel=mm_per_pixel,
                lineage=lineage_by_object.get(int(obj_id), {}),
            )
            for obj_id, rows in sorted(by_object.items())
        ]
        _write_summary(output_dir / output_name, summaries)
        print(f"Wrote: {output_dir / output_name}")


if __name__ == "__main__":
    main()
