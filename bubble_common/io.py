"""Frame I/O helpers shared across phases.

Extracted from phase2_track/sam2.py (sorted_frame_paths, save_mask).
load_mask translated from src/bubble_tracking/frame_io.py reference.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List

import cv2
import numpy as np

_FRAME_EXTS = (".jpg", ".jpeg", ".png")


def sorted_frame_paths(directory: str | Path) -> List[str]:
    """Return frame paths sorted numerically by frame index."""
    d = Path(directory)
    pattern = re.compile(r"(\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)
    paths = []
    for fname in os.listdir(str(d)):
        match = pattern.search(fname)
        if match:
            paths.append((int(match.group(1)), str(d / fname)))
    paths.sort(key=lambda x: x[0])
    return [p[1] for p in paths]


def save_mask(mask: np.ndarray, path: str | Path) -> None:
    """Save a label mask as a 16-bit PNG (supports many object IDs)."""
    output_path = str(path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, mask.astype(np.uint16))


def load_mask(path: str | Path) -> np.ndarray:
    """Read a mask PNG back as uint16 (or whatever bit depth was written)."""
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise ValueError(f"Failed to read mask: {path}")
    return arr
