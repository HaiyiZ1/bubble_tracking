import os, re, json
from pathlib import Path
import cv2

src = Path("data/bubbles/frames")
dst = Path("data/bubbles/frames_0p5")
scale = 0.5
quality = 90

roi_json_in = Path("data/interim/bubbles_air1_072_271/roi.json")
roi_json_out = Path("data/interim/bubbles_air1_072_271/roi_0p5.json")
roi_mask_in = Path("data/interim/bubbles_air1_072_271/roi_mask.png")
roi_mask_out = Path("data/interim/bubbles_air1_072_271/roi_mask_0p5.png")

dst.mkdir(parents=True, exist_ok=True)

pattern = re.compile(r"(\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)
paths = []
for name in os.listdir(src):
    m = pattern.search(name)
    if m:
        paths.append((int(m.group(1)), name))
paths.sort()

if not paths:
    raise SystemExit(f"No frames found in {src}")

first = cv2.imread(str(src / paths[0][1]))
if first is None:
    raise SystemExit("Failed to read first frame")

h, w = first.shape[:2]
new_w = int(round(w * scale))
new_h = int(round(h * scale))

for i, (idx, name) in enumerate(paths):
    out_path = dst / f"{idx:06d}.jpg"
    if out_path.exists():
        continue
    frame = cv2.imread(str(src / name))
    if frame is None:
        raise SystemExit(f"Failed to read {name}")
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out_path), resized, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if (i + 1) % 200 == 0:
        print(f"resized {i+1}/{len(paths)}")

if roi_json_in.exists():
    roi = json.loads(roi_json_in.read_text())
    scaled = {
        "x1": int(round(roi["x1"] * scale)),
        "y1": int(round(roi["y1"] * scale)),
        "x2": int(round(roi["x2"] * scale)),
        "y2": int(round(roi["y2"] * scale)),
    }
    scaled["x2"] = min(scaled["x2"], new_w)
    scaled["y2"] = min(scaled["y2"], new_h)
    roi_json_out.write_text(json.dumps(scaled, indent=2))

if roi_mask_in.exists():
    mask = cv2.imread(str(roi_mask_in), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise SystemExit("Failed to read ROI mask")
    mask_small = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    mask_small = (mask_small > 0).astype("uint8") * 255
    cv2.imwrite(str(roi_mask_out), mask_small)

print("Done.")