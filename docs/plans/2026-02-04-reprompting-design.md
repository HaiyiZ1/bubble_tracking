# Re-prompting SAM2 with Automask Keyframes (N=5)

## Goal
Use SAM2 automatic mask generation every 5 frames to re-prompt the video tracker, maintaining stable IDs and capturing new bubbles as they appear.

## Overview
We will run SAM2’s **automatic mask generator** on keyframes (t = 0, 5, 10, …), filter masks by ROI and area, then match those masks to existing tracks using a **hybrid matcher** (IoU → centroid fallback). The matched IDs are used to re-initialize SAM2’s video predictor for the next 5 frames. This reduces drift and quickly captures newly born bubbles without running automask on every frame.

## Data Flow
1. **Keyframe automask** (frame t):
   - Generate masks via `SAM2AutomaticMaskGenerator` on the ROI.
   - Filter masks by min area and max area ratio (≤50% of ROI).
2. **Matching**:
   - Compute IoU between previous keyframe masks and current keyframe masks.
   - Hungarian assignment using cost = 1 - IoU; accept if IoU ≥ 0.2.
   - Unmatched masks: use centroid distance matching (≤ 50 px).
   - Remaining unmatched current masks → births; unmatched previous masks → deaths.
   - If multiple current masks match one previous above threshold → split (new IDs with parent metadata).
3. **Re-prompt & propagate**:
   - Build prompt spec for frame t using matched IDs (centroid points; optional boxes later).
   - Run SAM2 video propagation for frames t..t+N-1.
4. **Outputs**:
   - `masks/` per frame
   - `tracks.json` with stable IDs
   - `stats.json`
   - `keyframes.json` with births/splits/deaths and re-prompt events

## Key Parameters (initial)
- Keyframe interval: **N = 5**
- IoU threshold: **0.2**
- Centroid distance threshold: **50 px**
- Max area ratio: **0.5** of ROI

## Components
- **Keyframe automask**: reuse `auto_mask_prompts.py` logic (but return raw masks + stats).
- **Matcher**: new module `reprompt_matcher.py` for IoU + centroid hybrid matching.
- **Orchestrator**: new script `run_reprompted_tracking.py` to loop keyframes, re-prompt, and propagate.

## Error Handling
- **No masks on keyframe**: skip re-prompting, continue propagation; log event.
- **GPU OOM**: fallback to lower `points_per_side` on that keyframe.
- **Matcher conflicts**: resolve splits by keeping best IoU as original ID and spawning new IDs for the rest.

## Testing
- Unit tests for matcher with synthetic masks (overlap, split, merge).
- Short run (10–20 frames) to verify ID stability.
- Visual overlay to confirm no ID flips and reasonable split/birth behavior.

## Notes
- Automask is image-only; it does not track. It helps detection, but association is still required.
- ROI masking is required to avoid electrodes and background artifacts.
