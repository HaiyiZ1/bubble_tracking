# Agent Guidance

This repository is organized for **reproduction of publishing runs** for bubble-tracking results.

## Primary Rule

Use `README.md` as the source of truth for runnable commands.
- Do not invent alternate commands when a README command exists.
- If a command fails, diagnose prerequisites first (env, checkpoints, data paths), then retry.

## What To Reproduce

1. Environment setup and validation.
2. Publishing relabel step for case1/case2/case3 Air1.
3. Tracking runs with CV keyframe prompts and debug output directories.
4. Publishing overlay exports (1x and 10x).
5. Optional visualization scripts for reviewer figures/tables.

## Key Concepts

- `*_publish` datasets are relabeled so frame `000001` is the first usable bubble frame.
- Tracking uses SAM2 propagation with CV keyframe prompt masks.
- `--cv-keyframe-debug-dir` is required for keyframe-level debug artifacts.
- Tracking configuration includes:
  - optical-flow-assisted association (`--use-flow`)
  - centroid-delta velocity mode (`--velocity-mode centroid_delta`)
  - raw preprocess mode (`--preprocess-mode raw`)
  - birth/death debounce to reduce flicker noise.

## Reproduction Targets

For each publishing case run (`outputs/case{1,2,3}_air1_publish`), verify:
- `tracks.csv` and `tracks.json`
- `events.json`
- `masks/` and `sam2_masks/`
- `tracking_overlay.mp4`
- `sam2_prompted_flow_config.json`

For publishing exports, verify:
- `publishing_overlay/tracking_overlay_publish_1x.mp4`
- `publishing_overlay/tracking_overlay_publish_10x.mp4`

## Scope Discipline

- Keep changes minimal and reproducibility-focused.
- Do not commit local scratch/planning/legacy artifacts.
- If asked to change workflow behavior, update `README.md` command blocks in the same change.
