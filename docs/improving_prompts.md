To reliably prompt *only bubbles*, we want two upgrades:

1. **A detector that proposes bubble candidates** (centers + tight boxes) while rejecting rails/edges.
2. **Stronger SAM prompts**: positive point(s) + **negative point(s)** + **box prompt** per bubble (or at least negative points if we can’t do boxes).

Below is a concrete, CV-centric approach that works well for “small dark-ish circles on a bright background with strong edges”.

---

## 1) our current pipeline’s key issues

### A. `min_area=200` is too high for small bubbles

In our sample, bubble radii are on the order of a few pixels to ~10 px. Area ≈ πr² → e.g. r=4 → 50 px, r=8 → 200 px. So `min_area=200` removes many bubbles entirely.

**Fix:** start with something like:

* `--min-area 20`
* `--max-area-ratio` much smaller than 0.2 (0.2 is enormous; it allows big hardware chunks)

### B. Otsu thresholding on the full frame prefers hardware edges

Otsu splits foreground/background by global bimodality. our frame is dominated by:

* big dark columns (rails)
* bright interior
* strong seam line

So our connected components are mostly “not bubbles”.

**Fix:** constrain the search region and suppress edges before thresholding, or use a blob/circle detector rather than CC on Otsu.

---

## 2) Make a “fluid interior mask” first (strong ROI, not just a rectangle)

Even if we keep our ROI logic, don’t rely on `(roi_x_center, roi_x_width)` alone. we want to eliminate rails/clamps aggressively.

A robust trick for this kind of image:

1. Compute a **column intensity profile** (mean per column).
2. Find the large bright plateau (the interior window).
3. Use that to build a binary mask: only process pixels inside the window.

Conceptually:

* Rails are dark → low column mean
* Interior is bright → high column mean

This gives we a data-driven vessel ROI that adapts per frame.

---

## 3) Remove the meniscus/bottom seam region

The bottom seam/meniscus produces tons of false positives.

A simple and effective heuristic:

1. Compute Sobel vertical gradient (`dy`) or Canny edges.
2. Sum edge magnitude **per row**.
3. The meniscus/seam region will produce a strong horizontal edge peak.
4. Mask out a band around that peak (e.g. ±20 px) or ignore detections below it.

---

## 4) Use blob/circle detection tuned to bubbles

For bubbles, `cv2.SimpleBlobDetector` (with circularity/inertia filters) is often better than connected components on Otsu.

Recommended features to filter by:

* **Area** (small)
* **Circularity** (close to 1)
* **Inertia ratio** (penalize elongated shapes)
* Optionally **color** (dark blobs vs bright background)

If bubbles are “rings” (bright edges + darker center) we can also detect using LoG/DoG and then filter circularity.

---

## 5) Generate stronger SAM2 prompts per bubble

### Minimum improvement (easy): add negative points

For each candidate bubble centroid `(cx, cy)`:

* Positive point: `(cx, cy, label=1)`
* Negative points: 4–8 points around it (a small ring outside expected bubble radius), label=0

This prevents SAM2 from leaking to the meniscus line or rails.

Example ring negatives at radius `r_out = max(8, int(1.5 * est_radius))`:

* (cx + r_out, cy)
* (cx - r_out, cy)
* (cx, cy + r_out)
* (cx, cy - r_out)

### Best improvement: also supply a tight box

If SAM2 supports box prompts in our wrapper/schema, include:

* box = bbox of blob expanded by ~2–5 px

Box prompts are extremely effective at preventing “mask runs away to the nearest big edge”.

If our current `ObjectPrompt` schema doesn’t include a box, it is worth adding; otherwise we can emulate it by adding **4 negative points at bbox corners outside** and **1–2 positive points inside**, but true box is better.

---

## 6) Concrete code changes (drop-in direction)

Below is an outline we can integrate into our script while keeping our YAML output logic.

### A. Detect interior window (mask rails)

```python
col_mean = gray.mean(axis=0)  # shape (w,)
# smooth it
col_smooth = cv2.GaussianBlur(col_mean.astype(np.float32), (0,0), 15).ravel()

# pick threshold for "bright interior"
thr = np.percentile(col_smooth, 70)  # tune: 60-80
inside = col_smooth > thr

# find largest contiguous True segment = interior window
# (simple scan)
best = (0, 0, 0)  # (len, start, end)
start = None
for i, v in enumerate(inside):
    if v and start is None: start = i
    if (not v or i == w-1) and start is not None:
        end = i if not v else i+1
        L = end - start
        if L > best[0]: best = (L, start, end)
        start = None
_, x1, x2 = best

mask = np.zeros_like(gray, dtype=np.uint8)
mask[:, x1:x2] = 255
```

### B. Detect meniscus/seam y and mask a band

```python
dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
row_energy = np.abs(dy).mean(axis=1)  # shape (h,)
row_smooth = cv2.GaussianBlur(row_energy.astype(np.float32), (0,0), 7).ravel()
y_seam = int(np.argmax(row_smooth[int(h*0.6):]) + int(h*0.6))  # search lower half

# mask out seam band
band = 25  # tune 15-40
mask[max(0, y_seam-band):min(h, y_seam+band), :] = 0
```

### C. Blob detection on masked, contrast-normalized image

```python
roi = cv2.bitwise_and(gray, gray, mask=mask)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16,16))
roi_eq = clahe.apply(roi)

params = cv2.SimpleBlobDetector_Params()
params.filterByArea = True
params.minArea = 20          # start here
params.maxArea = 800         # start here (tune)

params.filterByCircularity = True
params.minCircularity = 0.6  # tune 0.5-0.85

params.filterByInertia = True
params.minInertiaRatio = 0.3 # tune

params.filterByConvexity = False

# If bubbles are darker than background:
params.filterByColor = True
params.blobColor = 0

det = cv2.SimpleBlobDetector_create(params)
keypoints = det.detect(roi_eq)
```

Each `keypoint` gives:

* `kp.pt = (x, y)`
* `kp.size` (diameter estimate)

From that we can build:

* `cx, cy = int(kp.pt[0]), int(kp.pt[1])`
* `r = int(kp.size / 2)`

### D. Prompts: positive + ring negatives (+ optional box)

```python
def ring_points(cx, cy, r_out):
    return [
        (cx + r_out, cy),
        (cx - r_out, cy),
        (cx, cy + r_out),
        (cx, cy - r_out),
        (cx + r_out, cy + r_out),
        (cx - r_out, cy + r_out),
        (cx + r_out, cy - r_out),
        (cx - r_out, cy - r_out),
    ]

for obj_id, kp in enumerate(keypoints, start=1):
    cx, cy = map(int, kp.pt)
    r = max(3, int(kp.size / 2))
    r_out = int(1.8 * r) + 3

    pts = [PointPrompt(x=cx, y=cy, label=1)]
    for nx, ny in ring_points(cx, cy, r_out):
        if 0 <= nx < w and 0 <= ny < h:
            pts.append(PointPrompt(x=nx, y=ny, label=0))

    prompts.append(ObjectPrompt(
        object_id=obj_id,
        frame_idx=args.frame_idx,
        points=pts,
        # If supported by our schema, add:
        # box=(cx-r-3, cy-r-3, cx+r+3, cy+r+3)
    ))
```

---

## 7) Parameter recommendations to try first

1. Tighten area thresholds:

* `--min-area 20`
* and introduce a real `--max-area` (absolute), or set `--max-area-ratio 0.002` (instead of 0.2)

2. Stronger ROI:

* Derive ROI from column means (interior window), not just a center fraction.

3. Mask seam band:

* Detect seam `y` and zero out ±25 px.

4. Add negative prompts:

* This alone often fixes “SAM grabs the meniscus / clamp edge”.

---

## 8) Practical debugging outputs we should add

we already save a lot of debug images. Add these two (they will immediately show we why candidates are wrong):

* `mask_visualization.jpg`: draw interior window bounds + seam band.
* `blob_candidates.jpg`: draw detected keypoints as circles (using `kp.size`) so we can see if we’re actually finding bubbles.

---

## 9) If we want the highest tracking stability

Once we have *reasonable* prompts on frame 0, tracking usually improves dramatically if we **propagate prompts temporally**:

* For frame t>0, don’t rediscover everything from scratch.
* Use the previous frame’s SAM2 masks to predict bubble centroids (connected components on masks), then use those as prompts for the next frame.
* Only run blob detection as a “recovery mode” when tracks are lost.

This reduces false positives a lot because our prompt generator becomes “track-following” rather than “scene parsing”.

---

If we tell me whether our SAM2 wrapper/schema supports **box prompts**, I can give we the exact YAML shape and the minimal schema extension needed (so we can generate (point + box + negatives) cleanly).
