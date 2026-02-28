# Design: SAM2 Mask Integration with Flow Tracking (2026-02-22)

## Goal
Integrate SAM2 masks directly into the flow-tracking pipeline so every frame has a mask (including early empty frames), eliminating mask/index mismatches and keeping the run reproducible in one output directory.

## Scope
- Add `--sam2-auto` (per-frame automask) and `--sam2-propagate` (auto-search prompt + video propagation) to `scripts/run_flow_tracking.py`.
- Save SAM2 masks to `output/sam2_masks/` and feed them into the existing flow tracker (flow + matching + events + velocity + area).
- Keep preprocessing + optical flow unchanged; segmentation comes from SAM2 masks.

## Approach
- **Auto mode (`--sam2-auto`)**:
  - Build the SAM2 automask model once (`build_sam2`) and a `SAM2AutomaticMaskGenerator`.
  - For each frame:
    - Optionally apply ROI mask (use `--sam2-roi-mask`; if absent, build rectangular mask from `--roi` JSON).
    - Convert to RGB and downscale by `--sam2-mask-scale` (if < 1.0).
    - Generate SAM2 masks, filter by `--sam2-min-area` and optional `--sam2-max-area-ratio`.
    - Combine into a binary mask (foreground union) and save as `sam2_masks/000000.png`.
- **Propagate mode (`--sam2-propagate`)**:
  - Auto-search a prompt using SAM2 automask on successive frames (start + max search window).
  - On OOM, reduce `points_per_side` or `mask_scale` and log the downgrade.
  - Run SAM2 video propagation from the prompt frame, then backfill empty masks for earlier frames.
- Run `track_frames()` with `masks_dir=output/sam2_masks` (components are re-labeled per frame).
- Save `sam2_auto_config.json` with SAM2 parameters, prompt frame, and any OOM downgrades.

## CLI Additions
- `--sam2-auto` (per-frame automask)
- `--sam2-propagate` (auto-search prompt + propagate)
- `--sam2-checkpoint`, `--sam2-model-cfg`, `--sam2-device`
- `--sam2-roi-mask`, `--sam2-points-per-side`, `--sam2-min-area`,
  `--sam2-max-area-ratio`, `--sam2-mask-scale`
- `--sam2-start-frame`, `--sam2-max-search-frames`, `--sam2-max-prompts`
- `--sam2-min-points-per-side`, `--sam2-min-mask-scale`

## Error Handling
- Fail fast if SAM2 is not installed.
- Validate ROI mask size against frame size.
- Always write a mask per frame (empty mask if no SAM2 masks pass filters).
- Report OOM downgrades (points-per-side or mask-scale) during prompt search.

## Outputs
- `output/sam2_masks/*.png`: per-frame auto-masks
- `output/sam2_auto_config.json`: SAM2 run parameters
- Standard flow-tracking artifacts: `tracks.json`, `events.json`, `area_timeseries.*`, overlay video/frames

## Testing
- Run on the 200-frame subset (`data/interim/bubbles_air1_072_271/frames`) with ROI mask enabled.
- Verify no mask/frame index mismatch and ensure `events.json` is produced.
