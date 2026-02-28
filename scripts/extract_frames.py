#!/usr/bin/env python3
"""
Extract frames from a video file.

Usage:
    python scripts/extract_frames.py --input video.mp4 --output frames/
    python scripts/extract_frames.py --input video.mp4 --output frames/ --max-frames 100
    python scripts/extract_frames.py --input video.mp4 --output frames/ --resize 480x360
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bubble_tracking.frame_io import extract_frames_from_video


DEFAULT_FORMAT = "jpg"  # SAM2 requires JPEG format


def parse_resolution(res_str: str) -> tuple:
    """Parse resolution string like '480x360' to (width, height) tuple."""
    parts = res_str.lower().split("x")
    if len(parts) != 2:
        raise ValueError(f"Invalid resolution format: {res_str}. Use WIDTHxHEIGHT")
    return int(parts[0]), int(parts[1])


def main():
    parser = argparse.ArgumentParser(description="Extract frames from video")
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input video file path",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output directory for frames",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to extract (default: all)",
    )
    parser.add_argument(
        "--resize",
        type=str,
        default=None,
        help="Target resolution as WIDTHxHEIGHT (e.g., 480x360)",
    )

    args = parser.parse_args()

    # Validate input
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    # Parse resize option
    target_size = None
    if args.resize:
        target_size = parse_resolution(args.resize)

    print(f"Extracting frames from: {args.input}")
    print(f"Output directory: {args.output}")
    if args.max_frames:
        print(f"Max frames: {args.max_frames}")
    if target_size:
        print(f"Resize to: {target_size[0]}x{target_size[1]}")

    # Extract frames
    num_frames = extract_frames_from_video(
        video_path=args.input,
        output_dir=args.output,
        max_frames=args.max_frames,
        target_size=target_size,
    )

    print(f"\nExtracted {num_frames} frames to {args.output}")


if __name__ == "__main__":
    main()

