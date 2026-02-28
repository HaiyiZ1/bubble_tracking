# Edge Strength Diagnostics Export

## Goal
Provide an empirical view of edge-strength values across selected frames so we can pick a
threshold for edge gating based on real data.

## Outputs
- Per-frame overlay images with boundary heatmaps and per-object labels.
- `edge_strength.csv` and `edge_strength.json` with per-object statistics.
- `metadata.json` capturing frame ranges and config context.

## Inputs
- Frames directory and reference image.
- Masks directory from a run (object IDs).
- Optional `events.json` to reuse preprocessing/segmentation config.
- Optional ROI JSON to match the tracking run.
- Frame ranges (default: 21–41 and 180–200).

## Computation
1) Preprocess frames using the same mode as the run (from `events.json`).
2) Compute Sobel gradient magnitude on the normalized grayscale.
3) For each object mask, build a boundary ring and compute:
   - mean, median, p10, p90, min, max, boundary pixel count.
4) Render overlays by blending a gradient heatmap on boundary pixels and labeling each object
   with `#id median_strength`.

## CLI
`scripts/export_edge_strength_diagnostics.py`:
- `--frames`, `--reference`, `--masks`, `--out-dir`
- `--events` (optional), `--roi` (optional)
- `--ranges` (default `21-41,180-200`)
- `--edge-boundary-radius` override (optional)
