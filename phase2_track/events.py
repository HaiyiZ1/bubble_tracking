"""Mask-level events, area timeseries, and track writer.

Merges:
- src/bubble_tracking/mask_events.py (event detection, area timeseries)
- src/bubble_tracking/track_builder.py (tracks.csv/json writer)

Both source files were small (~330 + ~130 lines) and tightly related, so
phase2_track keeps them in a single events module rather than separate
files.
"""

from __future__ import annotations

import csv
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

from .schemas import (
    BoundingBox,
    FrameData,
    TrackedObject,
    TrackingResult,
)


# ---------------------------------------------------------------------------
# Internal helpers (translated from src/bubble_tracking/mask_events.py)
# ---------------------------------------------------------------------------

_FRAME_RE = re.compile(r"(\d+)\.(?:png|jpg|jpeg)$", re.IGNORECASE)


def _connected_components(
    binary_mask: np.ndarray,
    min_component_area: int,
) -> Tuple[np.ndarray, Dict[int, int]]:
    """Label foreground components and filter tiny regions."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_mask.astype(np.uint8),
        connectivity=8,
    )

    if num_labels <= 1:
        return np.zeros_like(labels, dtype=np.int32), {}

    filtered = np.zeros_like(labels, dtype=np.int32)
    areas: Dict[int, int] = {}
    new_id = 1

    for old_id in range(1, num_labels):
        area = int(stats[old_id, cv2.CC_STAT_AREA])
        if area < min_component_area:
            continue
        filtered[labels == old_id] = new_id
        areas[new_id] = area
        new_id += 1

    return filtered, areas


def _overlap_counts(prev_labels: np.ndarray, curr_labels: np.ndarray) -> np.ndarray:
    """
    Return foreground overlap matrix with shape (num_prev_components, num_curr_components).
    """
    if prev_labels.shape != curr_labels.shape:
        raise ValueError("Previous and current labels must share shape for overlap analysis")

    num_prev = int(prev_labels.max())
    num_curr = int(curr_labels.max())
    if num_prev == 0 or num_curr == 0:
        return np.zeros((num_prev, num_curr), dtype=np.int64)

    prev_flat = prev_labels.astype(np.int64).ravel()
    curr_flat = curr_labels.astype(np.int64).ravel()
    stride = num_curr + 1

    counts = np.bincount(
        prev_flat * stride + curr_flat,
        minlength=(num_prev + 1) * (num_curr + 1),
    )
    table = counts.reshape(num_prev + 1, num_curr + 1)
    return table[1:, 1:]


def _build_links(
    overlap: np.ndarray,
    prev_areas: Dict[int, int],
    curr_areas: Dict[int, int],
    min_overlap_pixels: int,
    min_link_iou: float,
    min_link_overlap_ratio: float,
) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    """
    Build parent->child and child->parent links from overlap statistics.
    """
    parents_to_children = {pid: [] for pid in prev_areas}
    children_to_parents = {cid: [] for cid in curr_areas}

    if overlap.size == 0:
        return parents_to_children, children_to_parents

    rows, cols = np.nonzero(overlap >= min_overlap_pixels)
    for row, col in zip(rows.tolist(), cols.tolist()):
        parent_id = row + 1
        child_id = col + 1
        inter = int(overlap[row, col])
        parent_area = prev_areas.get(parent_id, 0)
        child_area = curr_areas.get(child_id, 0)
        if parent_area <= 0 or child_area <= 0:
            continue

        union = parent_area + child_area - inter
        iou = float(inter / union) if union > 0 else 0.0
        overlap_ratio = float(inter / min(parent_area, child_area))

        if iou >= min_link_iou or overlap_ratio >= min_link_overlap_ratio:
            parents_to_children[parent_id].append(child_id)
            children_to_parents[child_id].append(parent_id)

    return parents_to_children, children_to_parents


def _pair_events_from_components(
    prev_labels: np.ndarray,
    prev_areas: Dict[int, int],
    curr_labels: np.ndarray,
    curr_areas: Dict[int, int],
    prev_frame_idx: int,
    frame_idx: int,
    min_overlap_pixels: int,
    min_link_iou: float,
    min_link_overlap_ratio: float,
) -> List[dict]:
    overlap = _overlap_counts(prev_labels, curr_labels)
    parents_to_children, children_to_parents = _build_links(
        overlap=overlap,
        prev_areas=prev_areas,
        curr_areas=curr_areas,
        min_overlap_pixels=min_overlap_pixels,
        min_link_iou=min_link_iou,
        min_link_overlap_ratio=min_link_overlap_ratio,
    )

    events: List[dict] = []

    for parent_id, child_ids in parents_to_children.items():
        if len(child_ids) >= 2:
            events.append(
                {
                    "type": "split",
                    "prev_frame_idx": int(prev_frame_idx),
                    "frame_idx": int(frame_idx),
                    "parent_component_id": int(parent_id),
                    "child_component_ids": [int(cid) for cid in sorted(child_ids)],
                    "parent_area_px": int(prev_areas[parent_id]),
                    "child_areas_px": [int(curr_areas[cid]) for cid in sorted(child_ids)],
                }
            )

    for child_id, parent_ids in children_to_parents.items():
        if len(parent_ids) >= 2:
            events.append(
                {
                    "type": "merge",
                    "prev_frame_idx": int(prev_frame_idx),
                    "frame_idx": int(frame_idx),
                    "parent_component_ids": [int(pid) for pid in sorted(parent_ids)],
                    "child_component_id": int(child_id),
                    "parent_areas_px": [int(prev_areas[pid]) for pid in sorted(parent_ids)],
                    "child_area_px": int(curr_areas[child_id]),
                }
            )

    for child_id, parent_ids in children_to_parents.items():
        if not parent_ids:
            events.append(
                {
                    "type": "birth",
                    "prev_frame_idx": int(prev_frame_idx),
                    "frame_idx": int(frame_idx),
                    "component_id": int(child_id),
                    "area_px": int(curr_areas[child_id]),
                }
            )

    for parent_id, child_ids in parents_to_children.items():
        if not child_ids:
            events.append(
                {
                    "type": "death",
                    "prev_frame_idx": int(prev_frame_idx),
                    "frame_idx": int(frame_idx),
                    "component_id": int(parent_id),
                    "area_px": int(prev_areas[parent_id]),
                }
            )

    return events


def _analyze_sorted_masks(
    mask_items: Iterable[Tuple[int, np.ndarray]],
    min_component_area: int,
    min_overlap_pixels: int,
    min_link_iou: float,
    min_link_overlap_ratio: float,
) -> dict:
    area_timeseries: List[dict] = []
    events: List[dict] = []

    prev_frame_idx = None
    prev_labels = None
    prev_areas: Dict[int, int] = {}
    frame_count = 0

    for frame_idx, mask in mask_items:
        binary = (mask > 0).astype(np.uint8)
        labels, areas = _connected_components(binary, min_component_area=min_component_area)
        total_area_px = int(binary.sum())
        largest_component = int(max(areas.values())) if areas else 0

        area_timeseries.append(
            {
                "frame_idx": int(frame_idx),
                "total_area_px": total_area_px,
                "num_components": int(len(areas)),
                "largest_component_area_px": largest_component,
            }
        )

        if prev_frame_idx is not None and prev_labels is not None:
            events.extend(
                _pair_events_from_components(
                    prev_labels=prev_labels,
                    prev_areas=prev_areas,
                    curr_labels=labels,
                    curr_areas=areas,
                    prev_frame_idx=prev_frame_idx,
                    frame_idx=frame_idx,
                    min_overlap_pixels=min_overlap_pixels,
                    min_link_iou=min_link_iou,
                    min_link_overlap_ratio=min_link_overlap_ratio,
                )
            )

        prev_frame_idx = frame_idx
        prev_labels = labels
        prev_areas = areas
        frame_count += 1

    totals = [x["total_area_px"] for x in area_timeseries]
    event_summary = {
        "num_splits": sum(1 for e in events if e["type"] == "split"),
        "num_merges": sum(1 for e in events if e["type"] == "merge"),
        "num_births": sum(1 for e in events if e["type"] == "birth"),
        "num_deaths": sum(1 for e in events if e["type"] == "death"),
    }

    return {
        "analysis_metadata": {
            "num_frames": frame_count,
            "min_component_area": int(min_component_area),
            "min_overlap_pixels": int(min_overlap_pixels),
            "min_link_iou": float(min_link_iou),
            "min_link_overlap_ratio": float(min_link_overlap_ratio),
        },
        "area_summary": {
            "mean_total_area_px": float(np.mean(totals)) if totals else 0.0,
            "min_total_area_px": int(min(totals)) if totals else 0,
            "max_total_area_px": int(max(totals)) if totals else 0,
        },
        "area_timeseries": area_timeseries,
        "events": events,
        "event_summary": event_summary,
    }


# ---------------------------------------------------------------------------
# Public API — event analysis (translated from mask_events.py)
# ---------------------------------------------------------------------------

def analyze_masks_by_frame(
    masks_by_frame: Dict[int, np.ndarray],
    min_component_area: int = 25,
    min_overlap_pixels: int = 10,
    min_link_iou: float = 0.05,
    min_link_overlap_ratio: float = 0.25,
) -> dict:
    """Analyze in-memory masks keyed by frame index."""
    sorted_items = sorted(masks_by_frame.items(), key=lambda x: int(x[0]))
    return _analyze_sorted_masks(
        mask_items=((int(frame_idx), mask) for frame_idx, mask in sorted_items),
        min_component_area=min_component_area,
        min_overlap_pixels=min_overlap_pixels,
        min_link_iou=min_link_iou,
        min_link_overlap_ratio=min_link_overlap_ratio,
    )


def _frame_idx_from_path(path: str, fallback: int) -> int:
    name = os.path.basename(path)
    match = _FRAME_RE.search(name)
    if match:
        return int(match.group(1))
    return fallback


def analyze_masks_dir(
    masks_dir: str,
    min_component_area: int = 25,
    min_overlap_pixels: int = 10,
    min_link_iou: float = 0.05,
    min_link_overlap_ratio: float = 0.25,
) -> dict:
    """Analyze masks stored on disk in a directory."""
    from .sam2 import sorted_frame_paths

    def _load_mask(path: str) -> np.ndarray:
        mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise ValueError(f"Cannot read mask: {path}")
        return mask

    mask_paths = sorted_frame_paths(masks_dir)
    if not mask_paths:
        raise ValueError(f"No masks found in {masks_dir}")

    def _iter_masks() -> Iterable[Tuple[int, np.ndarray]]:
        for fallback_idx, path in enumerate(mask_paths):
            frame_idx = _frame_idx_from_path(path, fallback_idx)
            yield frame_idx, _load_mask(path)

    return _analyze_sorted_masks(
        mask_items=_iter_masks(),
        min_component_area=min_component_area,
        min_overlap_pixels=min_overlap_pixels,
        min_link_iou=min_link_iou,
        min_link_overlap_ratio=min_link_overlap_ratio,
    )


def save_area_timeseries_csv(area_timeseries: List[dict], output_path: str) -> None:
    """Save area timeseries records as CSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fieldnames = [
        "frame_idx",
        "total_area_px",
        "num_components",
        "largest_component_area_px",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in area_timeseries:
            writer.writerow({name: row.get(name) for name in fieldnames})


# ---------------------------------------------------------------------------
# Public API — track building (translated from track_builder.py)
# ---------------------------------------------------------------------------

def build_tracks_from_masks(
    masks_dir: str,
    object_ids: Optional[List[int]] = None,
) -> TrackingResult:
    """
    Build tracking result from a directory of mask files.

    Args:
        masks_dir: Directory containing mask PNG files
        object_ids: Optional list of object IDs to track (auto-detect if None)

    Returns:
        TrackingResult with computed properties for each object
    """
    from .sam2 import sorted_frame_paths, compute_mask_properties

    def _load_mask(path: str) -> np.ndarray:
        mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise ValueError(f"Cannot read mask: {path}")
        return mask

    mask_paths = sorted_frame_paths(masks_dir)
    if not mask_paths:
        raise ValueError(f"No masks found in {masks_dir}")

    # Auto-detect object IDs from first mask
    if object_ids is None:
        first_mask = _load_mask(mask_paths[0])
        object_ids = [int(x) for x in np.unique(first_mask) if x > 0]

    objects: Dict[int, TrackedObject] = {
        obj_id: TrackedObject(object_id=obj_id) for obj_id in object_ids
    }

    # Process each mask
    for frame_idx, mask_path in enumerate(mask_paths):
        mask = _load_mask(mask_path)

        for obj_id in object_ids:
            frame_data = compute_mask_properties(mask, obj_id)
            if frame_data:
                objects[obj_id].frames[frame_idx] = frame_data

    return TrackingResult(
        num_frames=len(mask_paths),
        num_objects=len(objects),
        created_at=datetime.now(),
        objects=objects,
    )


def save_tracks(result: TrackingResult, output_path: str) -> None:
    """Save tracking result to JSON file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)


def load_tracks(input_path: str) -> TrackingResult:
    """Load tracking result from JSON file."""
    with open(input_path, "r") as f:
        data = json.load(f)
    return TrackingResult.from_dict(data)


def compute_track_statistics(result: TrackingResult) -> Dict:
    """
    Compute summary statistics for tracking results.

    Returns dictionary with:
        - per_object: stats for each object (lifetime, avg_area, displacement, etc.)
        - overall: aggregate statistics
    """
    stats: Dict = {"per_object": {}, "overall": {}}

    total_detections = 0
    all_areas: List[int] = []

    for obj_id, obj in result.objects.items():
        if not obj.frames:
            continue

        frame_indices = sorted(obj.frames.keys())
        areas = [obj.frames[f].area for f in frame_indices]
        centroids = [obj.frames[f].centroid for f in frame_indices]

        # Compute displacement (first to last frame)
        if len(centroids) >= 2:
            dx = centroids[-1][0] - centroids[0][0]
            dy = centroids[-1][1] - centroids[0][1]
            displacement = float(np.sqrt(dx**2 + dy**2))
        else:
            displacement = 0.0

        obj_stats = {
            "first_frame": frame_indices[0],
            "last_frame": frame_indices[-1],
            "lifetime_frames": len(frame_indices),
            "avg_area": float(np.mean(areas)),
            "min_area": int(min(areas)),
            "max_area": int(max(areas)),
            "total_displacement": float(displacement),
        }
        stats["per_object"][str(obj_id)] = obj_stats

        total_detections += len(frame_indices)
        all_areas.extend(areas)

    stats["overall"] = {
        "num_objects": len(result.objects),
        "num_frames": result.num_frames,
        "total_detections": total_detections,
        "avg_detections_per_frame": total_detections / max(1, result.num_frames),
        "global_avg_area": float(np.mean(all_areas)) if all_areas else 0.0,
    }

    return stats
