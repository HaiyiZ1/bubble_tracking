"""ROI handling for Phase 1.

Two paths:
- `import_roi(roi_json_in, roi_mask_in, out_dir)` — copies an existing
  rectangular ROI JSON and polygon ROI mask PNG into the run dir's phase1/.
- `interactive_select(frames_dir, out_dir)` — launches cv2 GUI selectors,
  identical to the historical scripts/select_roi.py and
  scripts/select_polygon_roi.py.

In both cases, the output files are:
- <out_dir>/roi_0p5.json     (rectangular ROI; keys x1,y1,x2,y2)
- <out_dir>/roi_mask_0p5.png (polygon ROI mask, 0/255)
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


def import_roi(roi_json_in: Path, roi_mask_in: Path, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    dst_json = out_dir / "roi_0p5.json"
    dst_mask = out_dir / "roi_mask_0p5.png"
    if not roi_json_in.exists():
        raise FileNotFoundError(roi_json_in)
    if not roi_mask_in.exists():
        raise FileNotFoundError(roi_mask_in)
    shutil.copyfile(roi_json_in, dst_json)
    shutil.copyfile(roi_mask_in, dst_mask)
    # Validate
    data = json.loads(dst_json.read_text())
    for key in ("x1", "y1", "x2", "y2"):
        if key not in data:
            raise ValueError(f"{dst_json}: missing key {key}")
    mask = cv2.imread(str(dst_mask), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not read {dst_mask} as grayscale image")
    return dst_json, dst_mask


def interactive_select(frames_dir: Path, out_dir: Path, frame_idx: int = 1) -> tuple[Path, Path]:
    """Open cv2 windows to capture rectangular + polygon ROI.

    Translates scripts/select_roi.py + scripts/select_polygon_roi.py.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_path = frames_dir / f"{frame_idx:06d}.jpg"
    if not frame_path.exists():
        raise FileNotFoundError(frame_path)
    img = cv2.imread(str(frame_path))
    if img is None:
        raise ValueError(f"Could not read {frame_path}")

    # Rectangular ROI (translation of scripts/select_roi.py).
    rect = cv2.selectROI("Select rectangular ROI (press Enter to confirm)", img, False, False)
    cv2.destroyAllWindows()
    x, y, w, h = (int(v) for v in rect)
    if w <= 0 or h <= 0:
        raise RuntimeError("Empty ROI selection")
    roi = {"x1": x, "y1": y, "x2": x + w, "y2": y + h}
    out_json = out_dir / "roi_0p5.json"
    out_json.write_text(json.dumps(roi, indent=2) + "\n")

    # Polygon ROI (translation of scripts/select_polygon_roi.py).
    # Click vertices; ENTER to close polygon; ESC to cancel.
    polygon: list[tuple[int, int]] = []
    disp = img.copy()

    def on_mouse(event, mx, my, flags, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            polygon.append((mx, my))
            cv2.circle(disp, (mx, my), 3, (0, 255, 0), -1)
            if len(polygon) > 1:
                cv2.line(disp, polygon[-2], polygon[-1], (0, 255, 0), 1)
            cv2.imshow("polygon ROI (Enter=close, Esc=abort)", disp)

    cv2.imshow("polygon ROI (Enter=close, Esc=abort)", disp)
    cv2.setMouseCallback("polygon ROI (Enter=close, Esc=abort)", on_mouse)
    while True:
        k = cv2.waitKey(20) & 0xFF
        if k == 13 and len(polygon) >= 3:
            break
        if k == 27:
            cv2.destroyAllWindows()
            raise RuntimeError("Polygon selection cancelled")
    cv2.destroyAllWindows()

    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(polygon, dtype=np.int32)], 255)
    out_mask = out_dir / "roi_mask_0p5.png"
    if not cv2.imwrite(str(out_mask), mask):
        raise RuntimeError(f"Failed to write {out_mask}")
    return out_json, out_mask
