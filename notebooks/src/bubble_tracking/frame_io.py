"""
Frame I/O utilities for reading and writing video frames.
"""

import os
import re
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import cv2
import numpy as np


def extract_frames_from_video(
    video_path: str,
    output_dir: str,
    max_frames: Optional[int] = None,
    target_size: Optional[Tuple[int, int]] = None,
) -> int:
    """
    Extract frames from a video file to a directory.

    Args:
        video_path: Path to input video file
        output_dir: Directory to save frames (created if doesn't exist)
        max_frames: Maximum number of frames to extract (None = all)
        target_size: Optional (width, height) to resize frames

    Returns:
        Number of frames extracted
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    frame_count = 0
    try:
        while True:
            if max_frames and frame_count >= max_frames:
                break

            ret, frame = cap.read()
            if not ret:
                break

            if target_size:
                frame = cv2.resize(frame, target_size)

            # Save as JPEG (SAM2 requires .jpg format)
            frame_path = os.path.join(output_dir, f"{frame_count:06d}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_count += 1
    finally:
        cap.release()

    return frame_count


def load_frames(
    frames_dir: str,
    max_frames: Optional[int] = None,
) -> List[np.ndarray]:
    """
    Load frames from a directory.

    Args:
        frames_dir: Directory containing frame images
        max_frames: Maximum number of frames to load

    Returns:
        List of frames as numpy arrays (BGR format)
    """
    frame_paths = sorted_frame_paths(frames_dir)
    if max_frames:
        frame_paths = frame_paths[:max_frames]

    frames = []
    for path in frame_paths:
        frame = cv2.imread(path)
        if frame is None:
            raise ValueError(f"Cannot read frame: {path}")
        frames.append(frame)

    return frames


def sorted_frame_paths(frames_dir: str) -> List[str]:
    """Get sorted list of frame image paths from a directory."""
    # Note: SAM2 requires .jpg format, but we support both for loading
    pattern = re.compile(r"(\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)
    paths = []

    for fname in os.listdir(frames_dir):
        match = pattern.search(fname)
        if match:
            paths.append((int(match.group(1)), os.path.join(frames_dir, fname)))

    paths.sort(key=lambda x: x[0])
    return [p[1] for p in paths]


def iter_frames(frames_dir: str) -> Iterator[Tuple[int, np.ndarray]]:
    """
    Iterate over frames in a directory.

    Yields:
        Tuples of (frame_index, frame_array)
    """
    for idx, path in enumerate(sorted_frame_paths(frames_dir)):
        frame = cv2.imread(path)
        if frame is None:
            raise ValueError(f"Cannot read frame: {path}")
        yield idx, frame


def save_mask(mask: np.ndarray, output_path: str) -> None:
    """
    Save a segmentation mask to file.

    Args:
        mask: 2D numpy array with instance IDs (0 = background)
        output_path: Path to save the mask PNG
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Save as 16-bit PNG to support many object IDs
    cv2.imwrite(output_path, mask.astype(np.uint16))


def load_mask(mask_path: str) -> np.ndarray:
    """Load a segmentation mask from file."""
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise ValueError(f"Cannot read mask: {mask_path}")
    return mask


def create_overlay_video(
    frames_dir: str,
    masks_dir: str,
    output_path: str,
    fps: float = 10.0,
    alpha: float = 0.5,
) -> None:
    """
    Create an overlay video showing masks on top of frames.

    Args:
        frames_dir: Directory with original frames
        masks_dir: Directory with mask PNGs
        output_path: Output video path
        fps: Frames per second
        alpha: Mask overlay opacity
    """
    frame_paths = sorted_frame_paths(frames_dir)
    if not frame_paths:
        raise ValueError(f"No frames found in {frames_dir}")

    # Read first frame to get dimensions
    first_frame = cv2.imread(frame_paths[0])
    height, width = first_frame.shape[:2]

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Generate distinct colors for objects
    np.random.seed(42)
    colors = np.random.randint(0, 255, (256, 3), dtype=np.uint8)
    colors[0] = [0, 0, 0]  # Background is black

    try:
        for idx, frame_path in enumerate(frame_paths):
            frame = cv2.imread(frame_path)

            mask_path = os.path.join(masks_dir, f"{idx:06d}.png")
            if os.path.exists(mask_path):
                mask = load_mask(mask_path)

                # Create colored overlay
                overlay = colors[mask % 256]

                # Blend where mask is non-zero
                mask_bool = mask > 0
                frame[mask_bool] = (
                    (1 - alpha) * frame[mask_bool] + alpha * overlay[mask_bool]
                ).astype(np.uint8)

            out.write(frame)
    finally:
        out.release()

