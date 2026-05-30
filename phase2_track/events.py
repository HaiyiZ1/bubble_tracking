"""Minimal stub for phase2_track event + track persistence helpers.

C.7 will replace this file entirely. For C.6 verification, we need only
`save_tracks` and `save_area_timeseries_csv` so that `track_frames` can
lazy-import them at runtime.

Functions copied verbatim from:
  src/bubble_tracking/track_builder.py  -> save_tracks
  src/bubble_tracking/mask_events.py    -> save_area_timeseries_csv
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import List

from .schemas import TrackingResult


def save_tracks(result: TrackingResult, output_path: str) -> None:
    """Save tracking result to JSON file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)


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
