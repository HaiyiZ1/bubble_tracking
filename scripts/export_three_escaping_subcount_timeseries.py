#!/usr/bin/env python3
"""
Export framewise subcount time series for three escaping situations:
1) vanished/death
2) split escape
3) merge escape

Output files:
  outputs/case1_air1_time_series_subcount_three_escaping_situation.csv
  outputs/case2_air1_time_series_subcount_three_escaping_situation.csv
  outputs/case3_air1_time_series_subcount_three_escaping_situation.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple


# Downscaled (0.5x) standard from pixel_mm_conversion_standard.md:
#   case1: 200 mm = 2335 px
#   case2: 200 mm = 2300 px
#   case3: 200 mm = 2420 px
MM_PER_PIXEL_BY_CASE: Dict[str, float] = {
    "case1": 200.0 / 2335.0,
    "case2": 200.0 / 2300.0,
    "case3": 200.0 / 2420.0,
}

DEFAULT_RATIO_THRESHOLD = 0.2
DEFAULT_MM2_THRESHOLD = 3.14

DEFAULT_CASE_RUNS = {
    "case1": Path("outputs/case1_air1_publish"),
    "case2": Path("outputs/case2_air1_publish"),
    "case3": Path("outputs/case3_air1_publish"),
}


def _load_json(path: Path) -> dict:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return data


def _get_area_from_frame(tracks_obj: dict, object_id: int, frame_idx: int) -> float:
    obj = tracks_obj.get(str(int(object_id)))
    if obj is None:
        return 0.0
    frame_data = obj.get("frames", {}).get(str(int(frame_idx)))
    if frame_data is None:
        return 0.0
    return float(frame_data.get("area", 0.0) or 0.0)


def _join_ids(values: List[int], sep: str = "|") -> str:
    if not values:
        return ""
    return sep.join(str(int(v)) for v in sorted(int(v) for v in values))


def _build_existing_frame_maps(tracks_obj: dict, frame_count: int) -> Tuple[List[float], List[str]]:
    areas_px = [0.0] * frame_count
    ids_per_frame: List[List[int]] = [[] for _ in range(frame_count)]

    for obj_id_text, payload in tracks_obj.items():
        obj_id = int(obj_id_text)
        frames = payload.get("frames", {})
        for frame_idx_text, frame_payload in frames.items():
            frame_idx = int(frame_idx_text)
            if frame_idx < 0 or frame_idx >= frame_count:
                continue
            area = float(frame_payload.get("area", 0.0) or 0.0)
            if area > 0.0:
                areas_px[frame_idx] += area
            ids_per_frame[frame_idx].append(obj_id)

    ids_joined = [_join_ids(ids) for ids in ids_per_frame]
    return areas_px, ids_joined


def _build_vanished_maps(events: List[dict], frame_count: int) -> Tuple[List[float], List[str]]:
    areas_px = [0.0] * frame_count
    ids_per_frame: List[List[int]] = [[] for _ in range(frame_count)]
    for ev in events:
        if str(ev.get("type", "")) != "death":
            continue
        frame_idx = int(ev.get("frame_idx", -1))
        if frame_idx < 0 or frame_idx >= frame_count:
            continue
        obj_id = int(ev.get("object_id", -1))
        area_px = float(ev.get("area_px", 0.0) or 0.0)
        if obj_id >= 0:
            ids_per_frame[frame_idx].append(obj_id)
        if area_px > 0.0:
            areas_px[frame_idx] += area_px
    ids_joined = [_join_ids(ids) for ids in ids_per_frame]
    return areas_px, ids_joined


def _build_split_maps(
    events: List[dict],
    tracks_obj: dict,
    frame_count: int,
    ratio_threshold: float,
    px2_threshold_mm2: float,
) -> Tuple[List[float], List[str]]:
    areas_px = [0.0] * frame_count
    items_per_frame: List[List[Tuple[int, List[int]]]] = [[] for _ in range(frame_count)]

    for ev in events:
        if str(ev.get("type", "")) != "split":
            continue
        curr_frame_idx = int(ev.get("frame_idx", -1))
        prev_frame_idx = int(ev.get("prev_frame_idx", -1))
        if curr_frame_idx < 0 or curr_frame_idx >= frame_count:
            continue
        parent_id = int(ev.get("parent_id", -1))
        child_ids = [int(v) for v in ev.get("child_ids", [])]
        if parent_id < 0 or not child_ids or prev_frame_idx < 0:
            continue

        parent_prev_area = _get_area_from_frame(tracks_obj, parent_id, prev_frame_idx)
        children_curr_area = sum(_get_area_from_frame(tracks_obj, cid, curr_frame_idx) for cid in child_ids)
        diff = parent_prev_area - children_curr_area
        threshold = max(float(ratio_threshold) * parent_prev_area, px2_threshold_mm2)
        if diff >= threshold:
            areas_px[curr_frame_idx] += diff
            items_per_frame[curr_frame_idx].append((parent_id, sorted(child_ids)))

    text_per_frame: List[str] = []
    for items in items_per_frame:
        items_sorted = sorted(items, key=lambda x: int(x[0]))
        parts: List[str] = []
        for parent_id, child_ids in items_sorted:
            child_text = ",".join(str(int(v)) for v in child_ids)
            parts.append(f"{int(parent_id)}({child_text})")
        text_per_frame.append("|".join(parts))

    return areas_px, text_per_frame


def _build_merge_maps(
    events: List[dict],
    tracks_obj: dict,
    frame_count: int,
    ratio_threshold: float,
    px2_threshold_mm2: float,
) -> Tuple[List[float], List[str]]:
    areas_px = [0.0] * frame_count
    items_per_frame: List[List[Tuple[List[int], int]]] = [[] for _ in range(frame_count)]

    for ev in events:
        if str(ev.get("type", "")) != "merge":
            continue
        curr_frame_idx = int(ev.get("frame_idx", -1))
        prev_frame_idx = int(ev.get("prev_frame_idx", -1))
        if curr_frame_idx < 0 or curr_frame_idx >= frame_count:
            continue
        parent_ids = sorted(int(v) for v in ev.get("parent_ids", []))
        child_id = int(ev.get("child_id", -1))
        if len(parent_ids) < 2 or child_id < 0 or prev_frame_idx < 0:
            continue

        parents_prev_sum = sum(_get_area_from_frame(tracks_obj, pid, prev_frame_idx) for pid in parent_ids)
        child_curr_area = _get_area_from_frame(tracks_obj, child_id, curr_frame_idx)
        diff = parents_prev_sum - child_curr_area
        threshold = max(float(ratio_threshold) * parents_prev_sum, px2_threshold_mm2)
        if diff >= threshold:
            areas_px[curr_frame_idx] += diff
            items_per_frame[curr_frame_idx].append((parent_ids, child_id))

    text_per_frame: List[str] = []
    for items in items_per_frame:
        items_sorted = sorted(items, key=lambda x: (min(x[0]), x[1]))
        parts: List[str] = []
        for parent_ids, child_id in items_sorted:
            parent_text = ",".join(str(int(v)) for v in parent_ids)
            parts.append(f"({parent_text}){int(child_id)}")
        text_per_frame.append("|".join(parts))
    return areas_px, text_per_frame


def _export_case(
    case_key: str,
    run_dir: Path,
    output_csv: Path,
    ratio_threshold: float,
    mm2_threshold: float,
) -> None:
    tracks = _load_json(run_dir / "tracks.json")
    events = _load_json(run_dir / "events.json")

    frame_count = int(tracks.get("metadata", {}).get("num_frames", 0))
    if frame_count <= 0:
        raise ValueError(f"Invalid frame count in {run_dir / 'tracks.json'}")

    fps = float(events.get("config", {}).get("fps", 4.0))
    if fps <= 0:
        raise ValueError(f"Invalid fps in {run_dir / 'events.json'}")

    mm_per_pixel = float(MM_PER_PIXEL_BY_CASE[case_key])
    mm2_per_px2 = mm_per_pixel ** 2
    px2_threshold_mm2 = float(mm2_threshold) / mm2_per_px2

    tracks_obj = tracks.get("objects", {})
    events_list = events.get("events", [])

    existing_px, existing_ids = _build_existing_frame_maps(tracks_obj, frame_count)
    vanished_px, vanished_ids = _build_vanished_maps(events_list, frame_count)
    split_px, split_ids = _build_split_maps(
        events_list,
        tracks_obj,
        frame_count,
        ratio_threshold=ratio_threshold,
        px2_threshold_mm2=px2_threshold_mm2,
    )
    merge_px, merge_ids = _build_merge_maps(
        events_list,
        tracks_obj,
        frame_count,
        ratio_threshold=ratio_threshold,
        px2_threshold_mm2=px2_threshold_mm2,
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame_idx",
            "time_s",
            "total_existing_bubble_area_px2",
            "total_existing_bubble_area_mm2",
            "numberings_of_existing_bubbles",
            "total_vanished_escape_area_px2",
            "total_vanished_escape_area_mm2",
            "numberings_of_vanished_escape_bubbles",
            "total_split_escape_area_px2",
            "total_split_escape_area_mm2",
            "numberings_of_split_escape_bubbles",
            "total_merge_escape_area_px2",
            "total_merge_escape_area_mm2",
            "numberings_of_merge_escape_bubbles",
            "total_escape_area_across_3_cases_px2",
            "total_escape_area_across_3_cases_mm2",
        ])

        for frame_idx in range(frame_count):
            epx = float(existing_px[frame_idx])
            vpx = float(vanished_px[frame_idx])
            spx = float(split_px[frame_idx])
            mpx = float(merge_px[frame_idx])
            total_escape_px = vpx + spx + mpx

            writer.writerow([
                frame_idx,
                f"{frame_idx / fps:.6f}",
                f"{epx:.6f}",
                f"{epx * mm2_per_px2:.6f}",
                existing_ids[frame_idx],
                f"{vpx:.6f}",
                f"{vpx * mm2_per_px2:.6f}",
                vanished_ids[frame_idx],
                f"{spx:.6f}",
                f"{spx * mm2_per_px2:.6f}",
                split_ids[frame_idx],
                f"{mpx:.6f}",
                f"{mpx * mm2_per_px2:.6f}",
                merge_ids[frame_idx],
                f"{total_escape_px:.6f}",
                f"{total_escape_px * mm2_per_px2:.6f}",
            ])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export case1/2/3 Air1 subcount time-series CSVs for vanished/split/merge escapes."
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory where output CSVs are written (default: outputs)",
    )
    parser.add_argument(
        "--ratio-threshold",
        type=float,
        default=DEFAULT_RATIO_THRESHOLD,
        help="Fraction threshold for split/merge escape candidate check (default: 0.2)",
    )
    parser.add_argument(
        "--mm2-threshold",
        type=float,
        default=DEFAULT_MM2_THRESHOLD,
        help="Absolute area threshold in mm^2 for split/merge escape candidate check (default: 3.14)",
    )
    parser.add_argument(
        "--output-template",
        default="{case_key}_air1_time_series_subcount_three_escaping_situation.csv",
        help="Output filename template, supports {case_key}",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    for case_key, run_dir in DEFAULT_CASE_RUNS.items():
        out_name = str(args.output_template).format(case_key=case_key)
        out_csv = output_dir / out_name
        _export_case(
            case_key=case_key,
            run_dir=run_dir,
            output_csv=out_csv,
            ratio_threshold=float(args.ratio_threshold),
            mm2_threshold=float(args.mm2_threshold),
        )
        print(f"Wrote: {out_csv}")


if __name__ == "__main__":
    main()
