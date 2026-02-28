# Automatrix Preprocess Mode (Strict)

## Goal
Match `picture_automatrix.py` preprocessing and segmentation as closely as possible
while keeping the flow-tracking pipeline unchanged. This isolates whether the
observed noise comes from small algorithmic differences.

## Scope
- Add a preprocess mode selector (`default` vs `automatrix`) to the flow-tracking
  runner and config.
- Implement strict automatrix preprocessing and segmentation using SciPy and
  scikit-image, applied to all frames with the existing ROI.

## Preprocessing (Automatrix)
1) Diff current frame against reference (float32).
2) SciPy `gaussian_filter` with sigma `diff_sigma` (default 0.5).
3) Percentile clip (1, 99), normalize to 0-255, cast to uint8.
4) Convert to grayscale (float32).
5) Build light matrix on grayscale using grid (8x5), percentile 95, min bright
   pixel count 10, tolerance 10.
6) Fill missing grid cells by neighbor averaging.
7) Interpolate grid using `griddata(..., method="cubic")`.
8) Upsample with `zoom(..., order=3)` and smooth via `gaussian_filter` with
   sigma = cell_h / 2.
9) Normalize grayscale by light map and min-max to uint8.

## Segmentation (Automatrix)
1) Gaussian blur (3x3).
2) Histogram peak using left bin edges from `np.histogram`.
3) Threshold = min(peak - 7, 248).
4) `opening(binary_mask, disk(4))` (kept for parity).
5) `remove_small_objects(label(binary_mask), min_size=500) > 0`.
6) `median(cleaned, disk(12))`.
7) Label connected components without a second min-area filter.

## Config / CLI
- `--preprocess-mode {default,automatrix}` selects the path.
- Default remains `default`; automatrix does not use CLAHE or compact filters.
- Dependencies are optional and only imported when automatrix is enabled.

## Outputs
Tracks, masks, and events remain identical in format. The automatrix mode
affects only the preprocessing/segmentation stage.
