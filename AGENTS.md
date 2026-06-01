# Agent Guidance

This repository is organized for **reproduction of publishing runs** for bubble-tracking results.

## Primary Rule

Use `README.md` as the source of truth for runnable commands.
- Do not invent alternate commands when a README command exists.
- If a command fails, diagnose prerequisites first (env, checkpoints, data paths), then retry.

## What To Reproduce

1. Environment setup and validation (venv + SAM2 + checkpoints).
2. Phase 1 — Preprocessing: frame downscale, ROI capture/import, and CV-keyframe prompt building (writes `cv_keyframe_prompts/` + `cv_keyframe_debug/` to disk).
3. Phase 2 — Tracking: reads CV-keyframe prompts from disk, runs SAM2 propagation + optical-flow + linkage tracking, debounce.
4. Phase 3 — Postprocessing: velocity overlays, vanished-area plots, per-bubble metrics, publishing 1x/10x videos.

## Key Concepts

- `*_publish` datasets are relabeled so frame `000001` is the first usable bubble frame.
- `meta.yaml` is the source of truth for per-case calibration (`scale: {mm, pixel}`) and per-air relabel start frame, plus the global `fps`.
- Each per-case run uses one run-dir (`outputs/<case>_air1_publish/`) containing `phase1/`, `phase2/`, `phase3/` and a top-level `run_config.json` that the phases share.
- Phase 2 reproduces deterministically across runs when invoked with `PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8` env vars. Without those, GPU cuBLAS/cuDNN heuristics drift between runs.
- The historical pre-refactor oracle outputs (`outputs/case<N>_air1_publish/sam2_masks/`, `tracks.csv` at the run-dir root) were generated nondeterministically. The new pipeline produces byte-identical results between consecutive runs of its own code, not against the historical oracle.

## Reproduction Targets

For each publishing case run (`outputs/case{1,2,3}_air1_publish/`), verify:
- `run_config.json` at the root with `phase1_done`, `phase2_done`, `phase3_done` all true.
- `phase1/`: `frames_0p5/`, `roi_0p5.json`, `roi_mask_0p5.png`, `cv_keyframe_prompts/`, `cv_keyframe_debug/`, `phase1_summary.json`.
- `phase2/`: `sam2_masks/`, `masks/`, `tracks.csv`, `tracks.json`, `events.json`, `area_timeseries.csv/json`, `sam2_overlay.mp4`, `tracking_overlay.mp4`.
- `phase3/`: `velocity_overlay.mp4`, `bubble_velocity_area_metrics.csv`, `bubble_max_velocity.csv`, `vanished_area_moving_window.csv`, the 8 vanished-area PNG variants, the velocity-area scatter PNGs, and `publishing_overlay/tracking_overlay_publish_{1x,10x}.mp4`.

## Scope Discipline

- Keep changes minimal and reproducibility-focused.
- Do not commit local scratch / planning / legacy artifacts.
- If asked to change workflow behavior, update `README.md` command blocks in the same change.
- `src/` and the consumed `scripts/*.py` files were removed in the v1 refactor. Don't add code to `src/`; add it to the appropriate `phaseN_*` package or to `scripts/check_env.py` (the only remaining standalone tool).
