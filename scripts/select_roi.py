#!/usr/bin/env python3
"""
Interactive ROI selector using OpenCV.

Example:
  /home/aaron/Downloads/bubble_tracking/venv/bin/python scripts/select_roi.py \
      --frames data/interim/bubbles_air1_072_271/frames \
      --frame-idx 0 \
      --out data/interim/bubbles_air1_072_271/roi.json
"""

import argparse
import json
from pathlib import Path

import cv2


def fit_to_screen(img, max_w=1600, max_h=900):
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale == 1.0:
        return img, 1.0
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def main():
    parser = argparse.ArgumentParser(description="Select ROI on a frame")
    parser.add_argument("--frames", required=True, help="Directory with frames (numeric filenames)")
    parser.add_argument("--frame-idx", type=int, default=0, help="Frame index to display")
    parser.add_argument("--out", required=True, help="Output JSON path for ROI")
    args = parser.parse_args()

    frames_dir = Path(args.frames)
    frame_path = frames_dir / f"{args.frame_idx:06d}.jpg"
    if not frame_path.exists():
        raise FileNotFoundError(f"Frame not found: {frame_path}")

    img = cv2.imread(str(frame_path))
    if img is None:
        raise ValueError(f"Failed to read: {frame_path}")

    display, scale = fit_to_screen(img)
    r = cv2.selectROI("Select ROI (press Enter to confirm)", display, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()

    x, y, w, h = map(int, r)
    x1 = int(x / scale)
    y1 = int(y / scale)
    x2 = int((x + w) / scale)
    y2 = int((y + h) / scale)
    roi = {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "frame": str(frame_path)}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(roi, indent=2))

    print(f"Saved ROI to {out_path}")
    print(roi)


if __name__ == "__main__":
    main()
