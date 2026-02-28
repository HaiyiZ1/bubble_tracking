# Edge-Strength Gate (Boundary Gradient)

## Goal
Suppress vague, low-contrast blobs while preserving bubbles that show strong, well-defined edges.

## Approach
Compute a gradient magnitude image on the normalized grayscale (post-preprocessing). For each
connected component, evaluate the gradient strength along a thin boundary ring and drop the
component if the edge response is too weak.

## Algorithm
1) Compute Sobel gradients (x,y) and magnitude on normalized grayscale.
2) For each component mask, build a boundary ring using `dilate(mask, r) - erode(mask, r)`.
3) Compute the median gradient magnitude on the ring.
4) Keep the component if median >= `edge_min_strength`.
5) Optionally, require `edge_min_fraction` of boundary pixels to be above the same threshold.

## Config / CLI
- `edge_min_strength` (float): minimum boundary strength (0 disables gate).
- `edge_boundary_radius` (int): ring radius in pixels (default 1).
- `edge_min_fraction` (float): fraction of boundary pixels above threshold (0 disables).

CLI flags:
- `--edge-min-strength`
- `--edge-boundary-radius`
- `--edge-min-fraction`

## Notes
- The gate runs after segmentation and before temporal persistence gating.
- Works for both `default` and `automatrix` preprocessing modes.
- Threshold scale depends on the normalized grayscale (0-255) and Sobel magnitude.
