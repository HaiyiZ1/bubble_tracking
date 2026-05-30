"""Phase 3 publishing module.

Provides:
  write_publishing_overlay_videos  — 1x / 10x publishing overlay MP4s
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from matplotlib.colors import hsv_to_rgb

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


# ---------------------------------------------------------------------------
# Per-case display titles (replaces PUBLISHING_CASES registry)
# ---------------------------------------------------------------------------

CASE_TITLES: Dict[str, str] = {
    "case1.air1": "High Porosity-Air1",
    "case2.air1": "Low Porosity-Air1",
    "case3.air1": "Medium Porosity-Air1",
}


# ---------------------------------------------------------------------------
# Inlined schema dataclasses (from src/bubble_tracking/schemas.py)
# ---------------------------------------------------------------------------


@dataclass
class BoundingBox:
    x_min: int
    y_min: int
    x_max: int
    y_max: int

    @classmethod
    def from_list(cls, bbox: List[int]) -> "BoundingBox":
        return cls(x_min=bbox[0], y_min=bbox[1], x_max=bbox[2], y_max=bbox[3])


@dataclass
class FrameData:
    centroid: Tuple[float, float]
    area: int
    bbox: Optional[BoundingBox] = None
    velocity: Optional[Tuple[float, float]] = None

    @classmethod
    def from_dict(cls, data: dict) -> "FrameData":
        bbox = None
        if "bbox" in data:
            bbox = BoundingBox.from_list(data["bbox"])
        velocity = None
        if "velocity" in data:
            velocity = tuple(data["velocity"])
        return cls(
            centroid=tuple(data["centroid"]),
            area=data["area"],
            bbox=bbox,
            velocity=velocity,
        )


@dataclass
class TrackedObject:
    object_id: int
    frames: Dict[int, FrameData] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, object_id: int, data: dict) -> "TrackedObject":
        frames = {
            int(frame_idx): FrameData.from_dict(frame_data)
            for frame_idx, frame_data in data["frames"].items()
        }
        return cls(object_id=object_id, frames=frames)


@dataclass
class TrackingResult:
    num_frames: int
    num_objects: int
    created_at: datetime
    objects: Dict[int, TrackedObject] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> "TrackingResult":
        metadata = data["metadata"]
        objects = {
            int(obj_id): TrackedObject.from_dict(int(obj_id), obj_data)
            for obj_id, obj_data in data["objects"].items()
        }
        return cls(
            num_frames=metadata["num_frames"],
            num_objects=metadata["num_objects"],
            created_at=datetime.fromisoformat(metadata["created_at"]),
            objects=objects,
        )


# ---------------------------------------------------------------------------
# Inlined layout + frame-IO helpers
# ---------------------------------------------------------------------------


def _resolve_tracks_dir(run_dir: Path) -> Path:
    """Return the dir containing tracks.json."""
    if (run_dir / "phase2" / "tracks.json").exists():
        return run_dir / "phase2"
    if (run_dir / "tracks.json").exists():
        return run_dir
    raise FileNotFoundError(
        f"No tracks.json under {run_dir} (tried phase2/ and root)"
    )


def _resolve_masks_dir(run_dir: Path) -> Path:
    """Return the dir containing the masks/ subdir."""
    for subpath in (run_dir / "phase2" / "masks", run_dir / "masks"):
        if subpath.exists():
            return subpath
    raise FileNotFoundError(
        f"No masks/ directory under {run_dir} (tried phase2/masks and masks)"
    )


def _resolve_frames_dir(run_dir: Path) -> Path:
    """Return the frames directory for the publishing overlay.

    Priority:
      1. <run_dir>/phase1/frames_0p5  (3-phase fresh runs)
      2. run_config.json → frames_dir
    """
    candidate = run_dir / "phase1" / "frames_0p5"
    if candidate.exists():
        return candidate
    config_path = run_dir / "run_config.json"
    if config_path.exists():
        cfg = json.loads(config_path.read_text())
        frames_dir = cfg.get("frames_dir")
        if frames_dir and Path(frames_dir).exists():
            return Path(frames_dir)
    raise FileNotFoundError(
        f"Cannot find frames directory for {run_dir}. "
        "Expected phase1/frames_0p5 or a run_config.json with a valid 'frames_dir'."
    )


def _sorted_frame_paths(frames_dir: Path) -> List[str]:
    pattern = re.compile(r"(\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)
    paths: List[Tuple[int, str]] = []
    for fname in os.listdir(str(frames_dir)):
        match = pattern.search(fname)
        if match:
            paths.append((int(match.group(1)), str(frames_dir / fname)))
    paths.sort(key=lambda x: x[0])
    return [p[1] for p in paths]


def _load_tracks(tracks_json: Path) -> TrackingResult:
    with tracks_json.open("r") as f:
        data = json.load(f)
    return TrackingResult.from_dict(data)


def _load_mask(mask_path: str) -> np.ndarray:
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise ValueError(f"Cannot read mask: {mask_path}")
    return mask


# ---------------------------------------------------------------------------
# Inlined visualization helpers (from src/bubble_tracking/visualization.py)
# ---------------------------------------------------------------------------


def _generate_colors(n: int, seed: int = 42) -> List[Tuple[int, int, int]]:
    np.random.seed(seed)
    colors = []
    for i in range(n):
        h = i / max(n, 1)
        s = 0.7 + np.random.random() * 0.3
        v = 0.7 + np.random.random() * 0.3
        rgb = hsv_to_rgb([h, s, v])
        colors.append(tuple(int(c * 255) for c in rgb))
    return colors


def _draw_mask_overlay(
    frame: np.ndarray,
    mask: np.ndarray,
    colors: Optional[Dict[int, Tuple[int, int, int]]],
    alpha: float,
) -> np.ndarray:
    result = frame.copy()
    unique_ids = np.unique(mask)
    unique_ids = unique_ids[unique_ids > 0]
    if colors is None:
        color_list = _generate_colors(len(unique_ids))
        colors = {int(uid): color_list[i] for i, uid in enumerate(unique_ids)}
    for obj_id in unique_ids:
        obj_mask = mask == obj_id
        color = colors.get(int(obj_id), (255, 255, 255))
        overlay = result.copy()
        overlay[obj_mask] = color
        result = cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0)
        contours, _ = cv2.findContours(
            obj_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(result, contours, -1, color, 2)
    return result


def _draw_tracking_info(
    frame: np.ndarray,
    result: TrackingResult,
    frame_idx: int,
    colors: Optional[Dict[int, Tuple[int, int, int]]],
    font_scale: float = 0.9,
    font_thickness: int = 2,
) -> np.ndarray:
    output = frame.copy()
    if colors is None:
        color_list = _generate_colors(len(result.objects))
        colors = {obj_id: color_list[i] for i, obj_id in enumerate(result.objects)}
    for obj_id, obj in result.objects.items():
        if frame_idx not in obj.frames:
            continue
        frame_data = obj.frames[frame_idx]
        color = colors.get(obj_id, (255, 255, 255))
        cx, cy = int(frame_data.centroid[0]), int(frame_data.centroid[1])
        cv2.circle(output, (cx, cy), 5, color, -1)
        cv2.circle(output, (cx, cy), 7, (255, 255, 255), 1)
        label = f"#{obj_id}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
        tx, ty = cx + 10, cy - 10
        cv2.rectangle(output, (tx - 4, ty - th - 4), (tx + tw + 4, ty + baseline + 4), (0, 0, 0), -1)
        cv2.putText(output, label, (tx, ty), font, font_scale, color, font_thickness)
        if frame_data.bbox:
            bbox = frame_data.bbox
            cv2.rectangle(output, (bbox.x_min, bbox.y_min), (bbox.x_max, bbox.y_max), color, 2)
    return output


# ---------------------------------------------------------------------------
# Publishing banner helpers (translated from export_publishing_tracking_video.py)
# ---------------------------------------------------------------------------


def _draw_text_with_box(
    frame: np.ndarray,
    text: str,
    origin: Tuple[int, int],
    font_scale: float,
    thickness: int,
    text_color: Tuple[int, int, int] = (255, 255, 255),
    box_color: Tuple[int, int, int] = (0, 0, 0),
    padding: int = 8,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = origin
    x1 = max(0, x - padding)
    y1 = max(0, y - text_h - padding)
    x2 = min(frame.shape[1] - 1, x + text_w + padding)
    y2 = min(frame.shape[0] - 1, y + baseline + padding)
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, -1)
    cv2.putText(frame, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)


def _format_elapsed(elapsed_seconds: float, resolution: str) -> str:
    if resolution == "sec":
        total = int(round(elapsed_seconds))
        minutes, seconds = divmod(total, 60)
        return f"Time elapsed: {minutes} min {seconds} sec."
    return f"Time elapsed: {elapsed_seconds / 60.0:.2f} min."


def _add_publishing_banner(
    frame: np.ndarray,
    title: str,
    elapsed_seconds: float,
    time_resolution: str,
) -> None:
    height, width = frame.shape[:2]
    overlay = frame.copy()
    banner_height = max(72, int(round(height * 0.1)))
    cv2.rectangle(overlay, (0, 0), (width - 1, banner_height), (0, 0, 0), -1)
    frame[:] = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)

    title_scale = max(0.8, min(1.4, width / 1800.0))
    info_scale = max(0.65, min(1.0, width / 2200.0))
    title_thickness = max(2, int(round(title_scale * 2)))
    info_thickness = max(2, int(round(info_scale * 2)))

    font = cv2.FONT_HERSHEY_SIMPLEX
    (title_w, title_h), _ = cv2.getTextSize(title, font, title_scale, title_thickness)
    title_x = max(12, (width - title_w) // 2)
    title_y = max(32, int(round(banner_height * 0.45)))
    time_y = min(height - 12, int(round(banner_height * 0.84)))

    _draw_text_with_box(
        frame, title, (title_x, title_y),
        font_scale=title_scale, thickness=title_thickness, padding=10,
    )
    _draw_text_with_box(
        frame, _format_elapsed(elapsed_seconds, time_resolution), (18, time_y),
        font_scale=info_scale, thickness=info_thickness, padding=8,
    )


def _render_video(
    frame_paths: List[str],
    masks_dir: Path,
    result: TrackingResult,
    title: str,
    output_path: Path,
    output_fps: float,
    time_fps: float,
    alpha: float,
    time_resolution: str,
) -> None:
    first_frame = cv2.imread(frame_paths[0])
    if first_frame is None:
        raise ValueError(f"Failed to read frame: {frame_paths[0]}")
    height, width = first_frame.shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(output_fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open writer: {output_path}")

    colors = {
        obj_id: _generate_colors(1, seed=obj_id * 7)[0]
        for obj_id in result.objects
    }

    try:
        for frame_idx, frame_path in enumerate(frame_paths):
            frame = cv2.imread(frame_path)
            if frame is None:
                raise ValueError(f"Failed to read frame: {frame_path}")

            mask_path = masks_dir / f"{frame_idx:06d}.png"
            if mask_path.exists():
                mask = _load_mask(str(mask_path))
                frame = _draw_mask_overlay(frame, mask, colors=colors, alpha=alpha)

            frame = _draw_tracking_info(
                frame, result, frame_idx, colors=colors,
                font_scale=0.9, font_thickness=2,
            )
            _add_publishing_banner(
                frame, title=title,
                elapsed_seconds=float(frame_idx) / float(time_fps),
                time_resolution=time_resolution,
            )
            writer.write(frame)
    finally:
        writer.release()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def write_publishing_overlay_videos(
    *,
    run_dir: Path,
    case: str,
    fps: float,
    speedup: float = 10.0,
    max_minutes: float = 5.0,
    alpha: float = 0.4,
    time_resolution: str = "min",
) -> None:
    """Write phase3/publishing_overlay/tracking_overlay_publish_{1x,10x}.mp4.

    ``case`` is in dotted form (e.g. 'case1.air1'); maps to a CASE_TITLES key.
    Falls back to the case string as the title if the key is not found.

    Reads frames from ``<run_dir>/phase1/frames_0p5`` (fresh runs) or
    from ``run_config.json`` (legacy). Reads tracks.json and masks/ from
    ``<run_dir>/phase2/`` (fresh) or ``<run_dir>/`` (oracle).

    Parameters
    ----------
    run_dir:          Per-case run directory.
    case:             Dotted case key, e.g. 'case1.air1'.
    fps:              Acquisition FPS (used for elapsed-time labels and 1x output).
    speedup:          Speed multiplier for the fast export (default: 10.0).
    max_minutes:      Maximum duration to render from the start (default: 5.0 min).
    alpha:            Mask overlay alpha (default: 0.4).
    time_resolution:  Elapsed-time format: 'min' or 'sec' (default: 'min').
    """
    title = CASE_TITLES.get(case, case)

    tracks_dir = _resolve_tracks_dir(run_dir)
    tracks_path = tracks_dir / "tracks.json"
    masks_dir = _resolve_masks_dir(run_dir)
    frames_dir = _resolve_frames_dir(run_dir)

    frame_paths = _sorted_frame_paths(frames_dir)
    if not frame_paths:
        raise ValueError(f"No frames found in {frames_dir}")

    max_frames = int(round(float(fps) * float(max_minutes) * 60.0))
    if max_frames <= 0:
        raise ValueError("max_minutes and fps must yield at least one frame")
    frame_paths = frame_paths[:max_frames]

    result = _load_tracks(tracks_path)

    output_dir = run_dir / "phase3" / "publishing_overlay"
    output_dir.mkdir(parents=True, exist_ok=True)

    speedup_int = int(round(speedup))
    exports = [
        (1.0, float(fps), output_dir / "tracking_overlay_publish_1x.mp4"),
        (speedup, float(fps) * speedup, output_dir / f"tracking_overlay_publish_{speedup_int}x.mp4"),
    ]

    for _, output_fps, output_path in exports:
        _render_video(
            frame_paths=frame_paths,
            masks_dir=masks_dir,
            result=result,
            title=title,
            output_path=output_path,
            output_fps=output_fps,
            time_fps=float(fps),
            alpha=float(alpha),
            time_resolution=time_resolution,
        )
