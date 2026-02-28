"""
Track building utilities for post-processing SAM2 outputs.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .schemas import (
    BoundingBox,
    FrameData,
    TrackedObject,
    TrackingResult,
)
from .sam2_adapter import compute_mask_properties


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
    from .frame_io import load_mask, sorted_frame_paths

    mask_paths = sorted_frame_paths(masks_dir)
    if not mask_paths:
        raise ValueError(f"No masks found in {masks_dir}")

    # Auto-detect object IDs from first mask
    if object_ids is None:
        first_mask = load_mask(mask_paths[0])
        object_ids = [int(x) for x in np.unique(first_mask) if x > 0]

    objects: Dict[int, TrackedObject] = {
        obj_id: TrackedObject(object_id=obj_id) for obj_id in object_ids
    }

    # Process each mask
    for frame_idx, mask_path in enumerate(mask_paths):
        mask = load_mask(mask_path)

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
    stats = {"per_object": {}, "overall": {}}

    total_detections = 0
    all_areas = []

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
            displacement = np.sqrt(dx**2 + dy**2)
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

