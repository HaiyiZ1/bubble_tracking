# Bubble Tracking Pipeline Design

## Goals
- Track bubble instances over time and report per-bubble area and velocity.
- Log merge/split events with new IDs (no lineage graph).
- Produce consistent outputs across sparse and dense phases.

## Assumptions
- A zero-bubble reference image is available for each run.
- Frame rate is constant at 4 FPS (0.25 s per frame).
- Camera is stationary (no global motion).
- Scale is 100 mm = 2335 px.

## Pipeline Overview
1) Preprocess each frame with reference subtraction and light normalization.
2) Compute dense optical flow between consecutive frames.
3) Run instance segmentation to obtain bubble masks per frame.
4) Match masks across frames to form track IDs and event logs.
5) Compute per-bubble area and velocity from masks and flow.

## Preprocessing
- Crop a fixed ROI (manual once, then reuse).
- Subtract the zero-bubble reference image.
- Normalize lighting with a grid-based light matrix (like `picture_automatrix.py`).
- Optional: CLAHE or mild Gaussian blur to stabilize contrast and reduce noise.

Output: normalized grayscale frames used for both flow and segmentation.

## Optical Flow
- Compute dense flow between frame t and t+1.
- Convert displacement to velocity using FPS (vx, vy in px/s).
- Optionally convert to mm/s using the scale.

Flow is also used for tracking by predicting where each bubble will move.

## Segmentation
- Run SAM2 automatic mask generation per frame.
- For each mask, store: binary mask, bbox, centroid, pixel area.
- The mask list is per-frame (no IDs yet).

## Tracking and Events
Match masks from t to t+1 using a flow-guided prediction:
- Predict each centroid position with the flow field.
- Build candidate matches using gates:
  - Centroid distance <= k * sqrt(area)
  - IoU >= threshold
  - Area ratio within bounds
- Solve one-to-one matches with Hungarian or greedy matching on a cost:
  - cost = w1*(1 - IoU) + w2*distance + w3*area_change

Event handling:
- Merge: multiple prior masks map to one current mask. Log merge and assign a new ID.
- Split: one prior mask maps to multiple current masks. Log split and assign new IDs.
- Unmatched current masks start new IDs.
- Unmatched prior masks terminate.

Store a match confidence score for post-filtering in dense scenes.

## Per-Bubble Metrics
- Area: count mask pixels and convert to mm^2.
- Velocity: median flow vector inside the mask (robust to outliers).
- Speed: magnitude of velocity.

## Outputs
- Per-frame table: ID, time, area, vx, vy, speed, centroid.
- Event log: merge/split with source IDs and new ID.
- Optional QA plots: total bubble area vs time, velocity histograms.

## Error Handling and Fallbacks
- If flow is unreliable (NaN-heavy or low-texture), fall back to centroid-only matching.
- If segmentation returns no masks, emit an empty frame record.
- Track per-frame brightness stats to flag reference-subtraction issues.

## Validation Plan
- Hand-annotate a short clip (10-20 frames) to verify ID continuity.
- Spot-check velocities against manual displacement.
- Inspect total bubble area trends to catch drift or segmentation bias.

## Open Questions
- Flow method choice (e.g., Farneback vs RAFT) depends on runtime constraints.
- Tracking cost weights and gates need tuning on a representative dataset.
