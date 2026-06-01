"""Downscale and optionally relabel raw frames into <run_dir>/phase1/frames_0p5/.

Auto-detect already-relabeled input:
- If the input directory's first sorted file is `000001.jpg` (or .png),
  treat as already-relabeled and skip relabel.
- Otherwise, relabel using start_frame: drop frames < start_frame, then
  renumber so frame start_frame -> 000001.

Always downscales by the given factor (default 0.5) using cv2.INTER_AREA,
matching the historical scripts/copy_and_compress_frames.py behavior.

JPEG quality 95 matches the historical pipeline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2

_FRAME_EXTS = (".jpg", ".jpeg", ".png")
_JPEG_QUALITY = 95


@dataclass(frozen=True)
class PreparedFrames:
    out_dir: Path
    n_frames: int
    relabel_skipped: bool


def sorted_frame_paths(directory: Path) -> List[Path]:
    """Return frame files in numeric order (by trailing integer in filename)."""
    files = [
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in _FRAME_EXTS
    ]
    def key(p: Path) -> Tuple[int, str]:
        m = re.search(r"(\d+)\.(?:jpg|jpeg|png)$", p.name, re.IGNORECASE)
        return (int(m.group(1)) if m else 10**12, p.name.lower())
    return sorted(files, key=key)


def _is_already_relabeled(first_frame_path: Path) -> bool:
    m = re.search(r"^0*1\.(jpg|jpeg|png)$", first_frame_path.name, re.IGNORECASE)
    return m is not None


def prepare(
    raw_frames: Path,
    out_dir: Path,
    *,
    downscale: float,
    start_frame: int,
) -> PreparedFrames:
    """Downscale all frames in `raw_frames` into `out_dir`.

    If raw frames are not already relabeled, relabel-then-downscale.
    """
    if downscale <= 0 or downscale > 1:
        raise ValueError(f"downscale must be in (0, 1], got {downscale}")
    if not raw_frames.exists():
        raise FileNotFoundError(f"--raw-frames not found: {raw_frames}")
    out_dir.mkdir(parents=True, exist_ok=True)
    if any(out_dir.iterdir()):
        raise FileExistsError(
            f"{out_dir} is not empty. Remove it or pick a fresh --run-dir."
        )

    paths = sorted_frame_paths(raw_frames)
    if not paths:
        raise ValueError(f"No frames (.jpg/.jpeg/.png) found in {raw_frames}")

    relabel_skipped = _is_already_relabeled(paths[0])
    if relabel_skipped:
        in_paths = paths
    else:
        kept: List[Path] = []
        for p in paths:
            m = re.search(r"(\d+)\.(?:jpg|jpeg|png)$", p.name, re.IGNORECASE)
            if m and int(m.group(1)) >= start_frame:
                kept.append(p)
        if not kept:
            raise ValueError(
                f"No frames at or after start_frame={start_frame} in {raw_frames}"
            )
        in_paths = kept

    n = 0
    for new_idx, src in enumerate(in_paths, start=1):
        img = cv2.imread(str(src))
        if img is None:
            raise ValueError(f"Failed to read {src}")
        if downscale != 1.0:
            h, w = img.shape[:2]
            img = cv2.resize(
                img,
                (max(1, int(round(w * downscale))), max(1, int(round(h * downscale)))),
                interpolation=cv2.INTER_AREA,
            )
        dst = out_dir / f"{new_idx:06d}.jpg"
        if not cv2.imwrite(str(dst), img, [int(cv2.IMWRITE_JPEG_QUALITY), _JPEG_QUALITY]):
            raise RuntimeError(f"Failed to write {dst}")
        n += 1

    return PreparedFrames(out_dir=out_dir, n_frames=n, relabel_skipped=relabel_skipped)
