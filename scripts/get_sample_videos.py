#!/usr/bin/env python3
"""
Download or copy sample videos for testing the SAM2 pipeline.

Usage:
    python scripts/get_sample_videos.py
    python scripts/get_sample_videos.py --source /path/to/local/video.mp4
    python scripts/get_sample_videos.py --use-davis  # Download DAVIS sample

This script provides sample videos for testing. By default, it creates a
synthetic test video with moving circles (useful for CPU-only testing).
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np


def create_synthetic_video(output_dir: str, num_frames: int = 50) -> str:
    """
    Create a synthetic video with moving circles for testing.

    This provides a simple, deterministic test case that doesn't require
    downloading external data.
    """
    frames_dir = os.path.join(output_dir, "synthetic_frames")
    os.makedirs(frames_dir, exist_ok=True)

    width, height = 480, 360
    np.random.seed(42)

    # Define 3 circles with initial positions and velocities
    circles = [
        {"pos": [100, 150], "vel": [3, 2], "radius": 30, "id": 1},
        {"pos": [250, 100], "vel": [-2, 3], "radius": 25, "id": 2},
        {"pos": [350, 250], "vel": [2, -2], "radius": 35, "id": 3},
    ]

    for frame_idx in range(num_frames):
        # Create frame with gradient background
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :, 0] = 40  # Blue channel
        frame[:, :, 1] = 30  # Green channel
        frame[:, :, 2] = 20  # Red channel

        # Draw and update circles
        for circle in circles:
            # Update position
            circle["pos"][0] += circle["vel"][0]
            circle["pos"][1] += circle["vel"][1]

            # Bounce off walls
            r = circle["radius"]
            if circle["pos"][0] < r or circle["pos"][0] > width - r:
                circle["vel"][0] *= -1
            if circle["pos"][1] < r or circle["pos"][1] > height - r:
                circle["vel"][1] *= -1

            # Clamp position
            circle["pos"][0] = np.clip(circle["pos"][0], r, width - r)
            circle["pos"][1] = np.clip(circle["pos"][1], r, height - r)

            # Draw filled circle with distinct color per ID
            colors = {1: (200, 100, 50), 2: (50, 200, 100), 3: (100, 50, 200)}
            color = colors.get(circle["id"], (200, 200, 200))
            cv2.circle(
                frame,
                (int(circle["pos"][0]), int(circle["pos"][1])),
                circle["radius"],
                color,
                -1,
            )

        # Save frame (as JPEG for SAM2 compatibility)
        frame_path = os.path.join(frames_dir, f"{frame_idx:06d}.jpg")
        cv2.imwrite(frame_path, frame)

    print(f"Created synthetic video: {num_frames} frames in {frames_dir}")
    return frames_dir


def create_default_prompts(output_dir: str) -> str:
    """Create a default prompts file for the synthetic video."""
    import yaml

    prompts_dir = os.path.join(output_dir, "prompts")
    os.makedirs(prompts_dir, exist_ok=True)

    # Prompts for the 3 circles at their initial positions (frame 0)
    prompts = {
        "prompts": [
            {"object_id": 1, "frame_idx": 0, "points": [{"x": 100, "y": 150, "label": 1}]},
            {"object_id": 2, "frame_idx": 0, "points": [{"x": 250, "y": 100, "label": 1}]},
            {"object_id": 3, "frame_idx": 0, "points": [{"x": 350, "y": 250, "label": 1}]},
        ]
    }

    prompts_path = os.path.join(prompts_dir, "synthetic_prompts.yaml")
    with open(prompts_path, "w") as f:
        yaml.dump(prompts, f, default_flow_style=False)

    print(f"Created prompts file: {prompts_path}")
    return prompts_path


def copy_local_video(source_path: str, output_dir: str) -> str:
    """Copy a local video file to the data directory."""
    os.makedirs(output_dir, exist_ok=True)
    dest_path = os.path.join(output_dir, os.path.basename(source_path))
    shutil.copy2(source_path, dest_path)
    print(f"Copied video to: {dest_path}")
    return dest_path


def download_davis_sample(output_dir: str) -> str:
    """
    Download a sample from DAVIS 2017 dataset (small subset).

    Note: This requires internet access. For offline testing,
    use --synthetic instead.
    """
    import urllib.request
    import zipfile

    # DAVIS 2017 trainval (480p) - we'll just get one sequence
    davis_url = "https://data.vision.ee.ethz.ch/cserber/share/davis/DAVIS-2017-trainval-480p.zip"

    print("Note: Full DAVIS download is ~1GB. For quick testing, use synthetic mode.")
    print("Downloading DAVIS sample...")

    zip_path = os.path.join(output_dir, "davis_sample.zip")
    os.makedirs(output_dir, exist_ok=True)

    try:
        urllib.request.urlretrieve(davis_url, zip_path)

        # Extract just one sequence
        with zipfile.ZipFile(zip_path, "r") as zf:
            # Extract blackswan sequence as example
            members = [m for m in zf.namelist() if "blackswan" in m]
            for member in members:
                zf.extract(member, output_dir)

        os.remove(zip_path)
        frames_dir = os.path.join(output_dir, "DAVIS", "JPEGImages", "480p", "blackswan")
        print(f"Downloaded DAVIS sample to: {frames_dir}")
        return frames_dir
    except Exception as e:
        print(f"Failed to download DAVIS: {e}")
        print("Falling back to synthetic video...")
        return create_synthetic_video(output_dir)


def main():
    parser = argparse.ArgumentParser(description="Get sample videos for testing")
    parser.add_argument(
        "--source",
        type=str,
        help="Path to local video file to copy",
    )
    parser.add_argument(
        "--use-davis",
        action="store_true",
        help="Download DAVIS 2017 sample (requires internet, ~1GB)",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        default=True,
        help="Create synthetic test video (default, no download needed)",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=50,
        help="Number of frames for synthetic video",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw",
        help="Output directory for sample data",
    )

    args = parser.parse_args()

    # Resolve output directory
    script_dir = Path(__file__).parent.parent
    output_dir = script_dir / args.output_dir

    if args.source:
        copy_local_video(args.source, str(output_dir))
    elif args.use_davis:
        download_davis_sample(str(output_dir))
    else:
        frames_dir = create_synthetic_video(str(output_dir), args.num_frames)
        create_default_prompts(str(output_dir))

    print("\nSample data ready! Next steps:")
    print("  1. Run: python scripts/run_sam2_video.py --help")
    print("  2. Or for synthetic: python scripts/run_sam2_video.py \\")
    print("       --frames data/raw/synthetic_frames \\")
    print("       --prompts data/raw/prompts/synthetic_prompts.yaml")


if __name__ == "__main__":
    main()

