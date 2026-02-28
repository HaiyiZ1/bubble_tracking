"""
Visualization utilities for bubble tracking.
"""

import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

from .schemas import TrackingResult


def generate_colors(n: int, seed: int = 42) -> List[Tuple[int, int, int]]:
    """Generate n distinct colors for visualization."""
    np.random.seed(seed)
    colors = []
    for i in range(n):
        # Use HSV for better color distribution
        h = i / n
        s = 0.7 + np.random.random() * 0.3
        v = 0.7 + np.random.random() * 0.3
        rgb = hsv_to_rgb([h, s, v])
        colors.append(tuple(int(c * 255) for c in rgb))
    return colors


def draw_mask_overlay(
    frame: np.ndarray,
    mask: np.ndarray,
    colors: Optional[Dict[int, Tuple[int, int, int]]] = None,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Draw colored mask overlay on frame.

    Args:
        frame: BGR image
        mask: Instance segmentation mask
        colors: Optional dict mapping object ID to BGR color
        alpha: Overlay opacity

    Returns:
        Frame with overlay
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


def draw_tracking_info(
    frame: np.ndarray,
    result: TrackingResult,
    frame_idx: int,
    colors: Optional[Dict[int, Tuple[int, int, int]]] = None,
    font_scale: float = 0.9,
    font_thickness: int = 2,
) -> np.ndarray:
    """
    Draw tracking information (centroids, IDs, bboxes) on frame.

    Args:
        frame: BGR image
        result: Tracking result
        frame_idx: Current frame index
        colors: Optional dict mapping object ID to BGR color

    Returns:
        Frame with annotations
    """
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
                color, 2
            )

    return output


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
    """
    Create a video with mask overlays and tracking annotations.

    Args:
        frames_dir: Directory with original frames
        masks_dir: Directory with mask PNGs
        result: Tracking result for annotations
        output_path: Output video path
        fps: Frames per second
        alpha: Mask overlay opacity
        show_info: Whether to show tracking info (IDs, bboxes)
    """
    from .frame_io import sorted_frame_paths, load_mask

    frame_paths = sorted_frame_paths(frames_dir)
    if not frame_paths:
        raise ValueError(f"No frames found in {frames_dir}")

    # Read first frame for dimensions
    first_frame = cv2.imread(frame_paths[0])
    height, width = first_frame.shape[:2]

    # Generate consistent colors
    colors = {
        obj_id: generate_colors(1, seed=obj_id * 7)[0]
        for obj_id in result.objects
    }

    # Create video writer
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    try:
        if overlay_dir:
            os.makedirs(overlay_dir, exist_ok=True)

        for frame_idx, frame_path in enumerate(frame_paths):
            frame = cv2.imread(frame_path)

            # Load and overlay mask
            mask_path = os.path.join(masks_dir, f"{frame_idx:06d}.png")
            if os.path.exists(mask_path):
                mask = load_mask(mask_path)
                frame = draw_mask_overlay(frame, mask, colors, alpha)

            # Add tracking info
            if show_info:
                frame = draw_tracking_info(
                    frame,
                    result,
                    frame_idx,
                    colors,
                    font_scale=label_scale,
                    font_thickness=label_thickness,
                )

            # Add frame counter
            cv2.putText(
                frame, f"Frame {frame_idx}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
            )

            out.write(frame)
            if overlay_dir:
                cv2.imwrite(os.path.join(overlay_dir, f"{frame_idx:06d}.jpg"), frame)
    finally:
        out.release()


def plot_trajectories(
    result: TrackingResult,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
) -> None:
    """
    Plot object trajectories over time.

    Args:
        result: Tracking result
        output_path: Optional path to save figure
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)

    colors = generate_colors(len(result.objects))

    for i, (obj_id, obj) in enumerate(result.objects.items()):
        if not obj.frames:
            continue

        frame_indices = sorted(obj.frames.keys())
        xs = [obj.frames[f].centroid[0] for f in frame_indices]
        ys = [obj.frames[f].centroid[1] for f in frame_indices]

        color = np.array(colors[i]) / 255.0
        ax.plot(xs, ys, "-", color=color, linewidth=2, label=f"Object {obj_id}")
        ax.scatter(xs[0], ys[0], c=[color], s=100, marker="o", edgecolors="black")
        ax.scatter(xs[-1], ys[-1], c=[color], s=100, marker="s", edgecolors="black")

    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
    ax.set_title("Object Trajectories")
    ax.legend(loc="best")
    ax.invert_yaxis()  # Image coordinates have Y increasing downward
    ax.set_aspect("equal")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        plt.close()
    else:
        plt.show()


def plot_area_over_time(
    result: TrackingResult,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6),
) -> None:
    """
    Plot object areas over time.

    Args:
        result: Tracking result
        output_path: Optional path to save figure
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)

    colors = generate_colors(len(result.objects))

    for i, (obj_id, obj) in enumerate(result.objects.items()):
        if not obj.frames:
            continue

        frame_indices = sorted(obj.frames.keys())
        areas = [obj.frames[f].area for f in frame_indices]

        color = np.array(colors[i]) / 255.0
        ax.plot(frame_indices, areas, "-o", color=color, linewidth=2,
                markersize=4, label=f"Object {obj_id}")

    ax.set_xlabel("Frame")
    ax.set_ylabel("Area (pixels)")
    ax.set_title("Object Area Over Time")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        plt.close()
    else:
        plt.show()
