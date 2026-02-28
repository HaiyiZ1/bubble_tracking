#!/usr/bin/env python3
"""
Compute cumulative grayscale intensity histogram from the first N frames.

Example:
  /home/aaron/Downloads/bubble_tracking/venv/bin/python scripts/inspect_intensity_hist.py \
      --frames data/bubbles/Air1 \
      --count 1000 \
      --out data/interim/bubbles_air1_hist
"""

import argparse
import os
import re
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt


def collect_frames(frames_dir: Path, max_count: int):
    pattern = re.compile(r"\((\d+)\)\.(jpg|jpeg|png)$", re.IGNORECASE)
    idx_to_file = {}
    for f in os.listdir(frames_dir):
        m = pattern.search(f)
        if m:
            idx_to_file[int(m.group(1))] = f
    indices = sorted(idx_to_file)[:max_count]
    return [frames_dir / idx_to_file[i] for i in indices]


def main():
    parser = argparse.ArgumentParser(description="Inspect grayscale intensity histogram")
    parser.add_argument("--frames", required=True, help="Directory with frames (e.g., data/bubbles/Air1)")
    parser.add_argument("--count", type=int, default=1000, help="Number of frames to sample")
    parser.add_argument("--out", required=True, help="Output directory to save results")
    args = parser.parse_args()

    frames_dir = Path(args.frames)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = collect_frames(frames_dir, args.count)
    hist = np.zeros(256, dtype=np.int64)
    used = 0

    for path in frame_paths:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        h, _ = np.histogram(img, bins=256, range=(0, 256))
        hist += h
        used += 1

    np.save(out_dir / "hist.npy", hist)
    (out_dir / "hist.csv").write_text(
        "intensity,count\n" + "\n".join(f"{i},{int(c)}" for i, c in enumerate(hist))
    )

    # Simple smoothing + peak reporting
    kernel = np.ones(9) / 9.0
    smooth = np.convolve(hist, kernel, mode="same")
    peaks = np.argsort(smooth)[-5:][::-1]

    summary = [
        f"frames_used: {used}",
        "top_peaks: " + ", ".join(f"{int(p)}({int(smooth[p])})" for p in peaks),
    ]
    (out_dir / "summary.txt").write_text("\n".join(summary))

    # Plot histogram
    plt.figure(figsize=(10, 4))
    plt.plot(np.arange(256), hist, color="black", linewidth=1)
    plt.title("Grayscale Intensity Histogram (Cumulative)")
    plt.xlabel("Intensity")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_dir / "hist.png", dpi=150)
    plt.close()

    print("\n".join(summary))


if __name__ == "__main__":
    main()
