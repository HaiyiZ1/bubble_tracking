#!/usr/bin/env python3
"""
Interactive polygon ROI selector using OpenCV.

Usage:
  /home/aaron/Downloads/bubble_tracking/venv/bin/python scripts/select_polygon_roi.py \
      --frames data/interim/bubbles_air1_072_271/frames \
      --frame-idx 0 \
      --out-json data/interim/bubbles_air1_072_271/roi_poly.json \
      --out-mask data/interim/bubbles_air1_072_271/roi_mask.png
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def fit_to_screen(img, max_w=1600, max_h=900):
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale == 1.0:
        return img, 1.0
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def main():
    parser = argparse.ArgumentParser(description="Select polygon ROI on a frame")
    parser.add_argument("--frames", required=True, help="Directory with frames (numeric filenames)")
    parser.add_argument("--frame-idx", type=int, default=0, help="Frame index to display")
    parser.add_argument("--out-json", required=True, help="Output JSON path for polygon")
    parser.add_argument("--out-mask", required=True, help="Output PNG mask path")
    args = parser.parse_args()

    frame_path = Path(args.frames) / f"{args.frame_idx:06d}.jpg"
    if not frame_path.exists():
        raise FileNotFoundError(f"Frame not found: {frame_path}")

    img = cv2.imread(str(frame_path))
    if img is None:
        raise ValueError(f"Failed to read: {frame_path}")

    display, scale = fit_to_screen(img)
    points = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))

    win = "Polygon ROI (click to add points, Enter=finish, C=clear)"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)

    while True:
        canvas = display.copy()
        if points:
            for p in points:
                cv2.circle(canvas, p, 4, (0, 0, 255), -1)
            for i in range(1, len(points)):
                cv2.line(canvas, points[i - 1], points[i], (0, 255, 0), 2)
        cv2.imshow(win, canvas)
        key = cv2.waitKey(20) & 0xFF
        if key == 13:  # Enter
            break
        if key in (ord("c"), ord("C")):
            points = []

    cv2.destroyAllWindows()

    if len(points) < 3:
        raise ValueError("Need at least 3 points for a polygon ROI.")

    # Scale points back to original image coords
    pts = [(int(x / scale), int(y / scale)) for x, y in points]

    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], 255)

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({
        "frame": str(frame_path),
        "points": pts,
    }, indent=2))

    out_mask = Path(args.out_mask)
    out_mask.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_mask), mask)

    print(f"Saved polygon to {out_json}")
    print(f"Saved mask to {out_mask}")


if __name__ == "__main__":
    main()
