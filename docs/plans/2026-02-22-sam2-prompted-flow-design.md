# Design: SAM2 Prompted Flow Tracking (2026-02-22)

## Goal
Provide a clean end-to-end script that seeds SAM2 propagation using either classic segmentation or SAM2 auto-search, then runs flow-guided tracking on the resulting masks without exposing edge/persistence tuning flags.

## Scope
- Add a minimal script `scripts/run_sam2_prompted_flow.py`.
- Support two prompt modes: `classic` (reference-diff segmentation) and `sam2` (auto-search).
- Use SAM2 mask prompts for propagation and output `sam2_masks/`.
- Run `track_frames()` on those masks and emit standard outputs.

## Prompt Modes
### Classic Prompt
- Iterate frames in order.
- Apply the same reference-diff preprocessing and threshold segmentation used in flow tracking.
- Find the first frame with a component >= `--min-area`.
- Choose the component closest to ROI bounding-box middle-bottom:
  - `x = (x1 + x2) / 2`, `y = y2 - 1`.
- Convert that component to a full-frame mask prompt (apply ROI mask if provided).

### SAM2 Auto-Search Prompt
- Use `SAM2AutomaticMaskGenerator` over a search window.
- Filter by `--sam2-min-area` and optional `--sam2-max-area-ratio`.
- On OOM, reduce `points_per_side` or `mask_scale` and log the downgrade.
- Use up to `--sam2-max-prompts` largest masks as prompts.

## Propagation + Flow Tracking
- Run SAM2 video propagation using `add_new_mask` (classic) or point prompts (sam2).
- Save `output/sam2_masks/*.png` for every frame (empty masks before prompt).
- Call `track_frames(..., masks_dir=output/sam2_masks)` with default preprocessing and segmentation.

## Outputs
- `output/sam2_masks/*.png`
- `output/sam2_prompted_flow_config.json` (prompt frame, OOM downgrades, params)
- Standard flow tracking outputs: `tracks.json`, `events.json`, `area_timeseries.*`, overlay video

## Testing
- Run on `data/interim/bubbles_air1_072_271/frames` with `--prompt-mode classic`.
- Verify prompt frame is earlier than SAM2 auto-search and that no mask indices are missing.
