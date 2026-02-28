# Phase 2 Pipeline (Bubble Tracking)

This document captures the Phase 2 pipeline described in `docs/next_steps_26-02-21.md` and how it is implemented in the current codebase.

## Goal
- Per-frame time index (`frame_idx`, `time_s`)
- Per-bubble area (`area_px`, optional `area_mm2`)
- Per-bubble velocity vector (`vx`, `vy`, `|v|`)
- Aggregate dynamics (total area, merge/split/birth/death events)

## Pipeline and Implementation
1. **Preprocess frames**
   - Reference-diff normalization with optional CLAHE and light normalization.
   - Implementation: `src/bubble_tracking/flow_tracking.py` (`preprocess_frame`, `PreprocessConfig`).
   - Notes: this stage prepares grayscale frames for both flow and segmentation logic.

2. **Dense optical flow**
   - Compute flow for each pair `t -> t+1`.
   - Implementation: `FlowEngine` in `src/bubble_tracking/flow_tracking.py`.
   - Backends: CPU Farneback; CUDA Farneback when available (`--flow-backend`, `--cuda-device`).

3. **Instance segmentation**
   - Two paths:
   - Classical segmentation from preprocessed frame (`segment_frame`).
   - SAM2 masks (`--masks-dir`, `--sam2-auto`, or `scripts/run_sam2_prompted_flow.py`).
   - Output is per-frame component bubble labels/masks.

4. **Tracking / ID association**
   - Flow-guided matching between previous confirmed labels and current labels.
   - Handles birth/death/split/merge and persistent IDs.
   - Implementation: matching + event logic in `track_frames()` (`src/bubble_tracking/flow_tracking.py`).

5. **Per-bubble velocity**
   - For each bubble at frame `t`, sample flow vectors inside its mask and use median vector.
   - Implementation: `_update_prev_velocities()` in `src/bubble_tracking/flow_tracking.py`.

6. **Area measurement**
   - Per-bubble area from mask pixel count.
   - Total area per frame from union of foreground.
   - Physical conversion via `--mm-per-pixel`.

## Main Entrypoints
- `scripts/run_flow_tracking.py`: configurable full tracking runner.
- `scripts/run_sam2_prompted_flow.py`: streamlined SAM2-seeded tracking runner.

## Output Artifacts
- `tracks.csv` / `tracks.json`: per-object per-frame centroid, area, velocity, time.
- `events.json`: merge/split/birth/death events + run config/runtime.
- `area_timeseries.csv` / `area_timeseries.json`: total bubble area over time.
- `masks/`, `tracking_overlay.mp4`: visual QA artifacts.

## Current Practical Note
- Long sequences can lose SAM2 continuity or miss newly appearing bubbles with single-seed propagation.
- Next step is robust re-discovery/re-prompting (e.g., combine classic segmentation with periodic SAM2 re-prompt).
