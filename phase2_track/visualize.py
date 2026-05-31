"""Overlay video writers for Phase 2.

Two videos per run:
- <run-dir>/phase2/sam2_overlay.mp4 — color-by-object_id overlay of
  SAM2 masks on raw frames (replaces the legacy in-bash overlay loop
  at commands.sh:940-1010).
- <run-dir>/phase2/tracking_overlay.mp4 — tracked-id overlay (from
  src/bubble_tracking/visualization.py:create_tracking_video).

Translated from src/bubble_tracking/visualization.py (`create_tracking_video`,
`draw_mask_overlay`) plus a small new helper `create_sam2_overlay_video`
that reproduces the bash inline overlay loop's color-by-id behaviour.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from bubble_common.schemas import TrackingResult


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------

def _color_for_id(obj_id: int) -> np.ndarray:
    """Deterministic per-object BGR color matching commands.sh:970-972.

    Uses the same seed formula (obj_id * 7) and range (50-255) as the
    bash inline overlay loop so that sam2_overlay.mp4 colours are
    bit-for-bit identical to those produced by the legacy script.
    """
    np.random.seed(obj_id * 7)
    return np.array(np.random.randint(50, 255, 3), dtype=np.uint8)


def generate_colors(n: int, seed: int = 42) -> list:
    """Generate n distinct HSV-derived BGR colors for visualization."""
    from matplotlib.colors import hsv_to_rgb as _hsv_to_rgb
    np.random.seed(seed)
    colors = []
    for i in range(n):
        h = i / max(n, 1)
        s = 0.7 + np.random.random() * 0.3
        v = 0.7 + np.random.random() * 0.3
        rgb = _hsv_to_rgb([h, s, v])
        colors.append(tuple(int(c * 255) for c in rgb))
    return colors


# ---------------------------------------------------------------------------
# SAM2 overlay video (new — replaces commands.sh bash loop)
# ---------------------------------------------------------------------------

def create_sam2_overlay_video(
    frames_dir,
    masks_dir,
    out_path,
    *,
    fps: float,
    alpha: float = 0.40,
) -> None:
    """Write an mp4 with per-object-id color blended over raw frames.

    Reproduces the in-bash overlay loop at commands.sh lines 950-1003.
    Colour formula: ``np.random.seed(obj_id * 7); np.random.randint(50, 255, 3)``
    Blend formula:  ``overlay[mask] = overlay[mask] * (1-alpha) + c * alpha``

    Args:
        frames_dir: Directory of source frames (jpg/png, sorted by stem number).
        masks_dir:  Directory of SAM2 mask PNGs (named ``<stem>.png``).
        out_path:   Destination mp4 path (parent dirs created automatically).
        fps:        Output video frame rate.
        alpha:      Overlay opacity (default 0.40 matches commands.sh).
    """
    from .sam2 import sorted_frame_paths

    frames_dir = Path(frames_dir)
    masks_dir = Path(masks_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    frame_paths = sorted_frame_paths(str(frames_dir))
    if not frame_paths:
        raise ValueError(f"No frames found in {frames_dir}")

    first = cv2.imread(frame_paths[0])
    if first is None:
        raise ValueError(f"Failed to read first frame: {frame_paths[0]}")
    h, w = first.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    if not writer.isOpened():
        raise OSError(f"cv2.VideoWriter failed to open: {out_path}")

    try:
        for fp in frame_paths:
            frame = cv2.imread(fp)
            if frame is None:
                continue

            stem = Path(fp).stem
            mp = masks_dir / f"{stem}.png"
            if mp.exists():
                m = cv2.imread(str(mp), cv2.IMREAD_UNCHANGED)
                if m is not None:
                    if m.ndim == 3:
                        m = m[:, :, 0]
                    overlay = frame.copy()
                    for obj_id in np.unique(m):
                        if obj_id == 0:
                            continue
                        mask = m == obj_id
                        c = _color_for_id(int(obj_id))
                        overlay[mask] = (
                            overlay[mask] * (1.0 - alpha) + c * alpha
                        ).astype(np.uint8)
                    frame = overlay

            cv2.putText(
                frame,
                stem,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            writer.write(frame)
    finally:
        writer.release()


# ---------------------------------------------------------------------------
# Mask overlay helper (translated from src/bubble_tracking/visualization.py)
# ---------------------------------------------------------------------------

def draw_mask_overlay(
    frame: np.ndarray,
    mask: np.ndarray,
    colors: Optional[Dict[int, Tuple[int, int, int]]] = None,
    alpha: float = 0.5,
) -> np.ndarray:
    """Draw colored mask overlay on frame.

    Args:
        frame:  BGR image.
        mask:   Instance segmentation mask (0 = background).
        colors: Optional dict mapping object_id -> BGR color tuple.
                Auto-generated from ``generate_colors`` if None.
        alpha:  Overlay opacity.

    Returns:
        Frame with per-object color blend + contour drawn.
    """
    result = frame.copy()
    unique_ids = np.unique(mask)
    unique_ids = unique_ids[unique_ids > 0]

    if colors is None:
        color_list = generate_colors(len(unique_ids))
        colors = {int(uid): color_list[i] for i, uid in enumerate(unique_ids)}

    for obj_id in unique_ids:
        obj_mask = mask == obj_id
        color = colors.get(int(obj_id), (255, 255, 255))

        # Blend color on masked region
        overlay = result.copy()
        overlay[obj_mask] = color
        result = cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0)

        # Draw contour
        contours, _ = cv2.findContours(
            obj_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(result, contours, -1, color, 2)

    return result


def _draw_tracking_info(
    frame: np.ndarray,
    result: TrackingResult,
    frame_idx: int,
    colors: Optional[Dict[int, Tuple[int, int, int]]] = None,
    font_scale: float = 0.9,
    font_thickness: int = 2,
) -> np.ndarray:
    """Draw tracking information (centroids, IDs, bboxes) on frame."""
    output = frame.copy()

    if colors is None:
        color_list = generate_colors(len(result.objects))
        colors = {obj_id: color_list[i] for i, obj_id in enumerate(result.objects)}

    for obj_id, obj in result.objects.items():
        if frame_idx not in obj.frames:
            continue

        frame_data = obj.frames[frame_idx]
        color = colors.get(obj_id, (255, 255, 255))

        # Draw centroid
        cx, cy = int(frame_data.centroid[0]), int(frame_data.centroid[1])
        cv2.circle(output, (cx, cy), 5, color, -1)
        cv2.circle(output, (cx, cy), 7, (255, 255, 255), 1)

        # Draw ID label with background for readability
        label = f"#{obj_id}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
        tx, ty = cx + 10, cy - 10
        box_x1 = tx - 4
        box_y1 = ty - th - 4
        box_x2 = tx + tw + 4
        box_y2 = ty + baseline + 4
        cv2.rectangle(output, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), -1)
        cv2.putText(output, label, (tx, ty), font, font_scale, color, font_thickness)

        # Draw bounding box
        if frame_data.bbox:
            bbox = frame_data.bbox
            cv2.rectangle(
                output,
                (bbox.x_min, bbox.y_min),
                (bbox.x_max, bbox.y_max),
                color,
                2,
            )

    return output


# ---------------------------------------------------------------------------
# Tracking overlay video (translated from src/bubble_tracking/visualization.py)
# ---------------------------------------------------------------------------

def create_tracking_video(
    frames_dir: str,
    masks_dir: str,
    result: TrackingResult,
    output_path: str,
    fps: float = 10.0,
    alpha: float = 0.4,
    show_info: bool = True,
    overlay_dir: Optional[str] = None,
    label_scale: float = 0.9,
    label_thickness: int = 2,
) -> None:
    """Create a video with mask overlays and tracking annotations.

    Args:
        frames_dir:  Directory with original frames.
        masks_dir:   Directory with mask PNGs (named ``<frame_idx:06d>.png``).
        result:      TrackingResult for per-frame annotation.
        output_path: Destination mp4 path.
        fps:         Frames per second.
        alpha:       Mask overlay opacity.
        show_info:   Whether to draw ID labels and bounding boxes.
        overlay_dir: Optional directory to also write per-frame JPEGs.
        label_scale: Font scale for ID labels.
        label_thickness: Font thickness for ID labels.
    """
    from .sam2 import sorted_frame_paths

    frame_paths = sorted_frame_paths(frames_dir)
    if not frame_paths:
        raise ValueError(f"No frames found in {frames_dir}")

    # Read first frame for dimensions
    first_frame = cv2.imread(frame_paths[0])
    if first_frame is None:
        raise ValueError(f"Cannot read first frame: {frame_paths[0]}")
    height, width = first_frame.shape[:2]

    # Generate consistent colors (same seed formula as visualization.py)
    colors = {
        obj_id: generate_colors(1, seed=obj_id * 7)[0]
        for obj_id in result.objects
    }

    # Create video writer
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    if not out.isOpened():
        raise OSError(f"cv2.VideoWriter failed to open: {output_path}")

    try:
        if overlay_dir:
            os.makedirs(overlay_dir, exist_ok=True)

        for frame_idx, frame_path in enumerate(frame_paths):
            frame = cv2.imread(frame_path)
            if frame is None:
                continue

            # Load and overlay mask
            mask_path = os.path.join(masks_dir, f"{frame_idx:06d}.png")
            if os.path.exists(mask_path):
                m = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
                if m is not None:
                    frame = draw_mask_overlay(frame, m, colors, alpha)

            # Add tracking info
            if show_info:
                frame = _draw_tracking_info(
                    frame,
                    result,
                    frame_idx,
                    colors,
                    font_scale=label_scale,
                    font_thickness=label_thickness,
                )

            # Add frame counter
            cv2.putText(
                frame,
                f"Frame {frame_idx}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            out.write(frame)
            if overlay_dir:
                cv2.imwrite(os.path.join(overlay_dir, f"{frame_idx:06d}.jpg"), frame)
    finally:
        out.release()
