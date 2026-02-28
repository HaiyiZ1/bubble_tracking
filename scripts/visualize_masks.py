import os, cv2, numpy as np
from pathlib import Path

masks_dir = Path("outputs/bubbles_air1_072_271_auto_search/masks")
out_path = "outputs/bubbles_air1_072_271_auto_search/masks_only.mp4"
fps = 10

mask_paths = sorted(masks_dir.glob("*.png"))
if not mask_paths:
    raise SystemExit("No masks found")

# read size from first mask
first = cv2.imread(str(mask_paths[0]), cv2.IMREAD_UNCHANGED)
h, w = first.shape[:2]

def color_for_id(i):
    # simple deterministic color
    np.random.seed(i * 7)
    return tuple(int(c) for c in np.random.randint(50, 255, 3))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

for p in mask_paths:
    m = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if m.ndim == 3:
        m = m[:, :, 0]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    for obj_id in np.unique(m):
        if obj_id == 0:
            continue
        color = color_for_id(int(obj_id))
        canvas[m == obj_id] = color
    out.write(canvas)

out.release()
print("saved:", out_path)
