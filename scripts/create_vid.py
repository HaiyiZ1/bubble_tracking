import os
import re
from pathlib import Path

import cv2

frames_dir = Path("data/bubbles/frames")
out_path = Path("outputs/air1_preview_lowres.mp4")

target_w = 1280   # set lower (e.g., 960) if you want smaller
fps = 10
stride = 1        # set 2 or 3 to skip frames

pattern = re.compile(r"(\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)
paths = []
for name in os.listdir(frames_dir):
    m = pattern.search(name)
    if m:
        paths.append((int(m.group(1)), frames_dir / name))
paths.sort(key=lambda x: x[0])

if not paths:
    raise SystemExit(f"No frames found in {frames_dir}")

first = cv2.imread(str(paths[0][1]))
if first is None:
    raise SystemExit(f"Failed to read {paths[0][1]}")

h, w = first.shape[:2]
if w <= target_w:
    target_w = w
scale = target_w / float(w)
target_h = int(h * scale)
if target_h % 2 == 1:
    target_h -= 1

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(str(out_path), fourcc, fps, (target_w, target_h))
if not writer.isOpened():
    raise SystemExit("Failed to open VideoWriter. Try a different codec or install ffmpeg.")

count = 0
for _, path in paths[::stride]:
    frame = cv2.imread(str(path))
    if frame is None:
        raise SystemExit(f"Failed to read {path}")
    if target_w != w:
        frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
    writer.write(frame)
    count += 1
    if count % 200 == 0:
        print(f"wrote {count} frames...")

writer.release()
print(f"Wrote {count} frames to {out_path}")