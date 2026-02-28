"""
Mask-level analytics for bubble area dynamics and topology events.

This module is intentionally ID-agnostic: it treats each frame as binary
foreground (`mask > 0`) and derives connected components per frame.
"""

from __future__ import annotations

import csv
import os
import re
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np

from .frame_io import load_mask, sorted_frame_paths


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
    mask_paths = sorted_frame_paths(masks_dir)
    if not mask_paths:
        raise ValueError(f"No masks found in {masks_dir}")

    def _iter_masks() -> Iterable[Tuple[int, np.ndarray]]:
        for fallback_idx, path in enumerate(mask_paths):
            frame_idx = _frame_idx_from_path(path, fallback_idx)
            yield frame_idx, load_mask(path)

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
