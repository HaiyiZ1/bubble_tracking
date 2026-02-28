# Temporal Persistence Gate (Flicker Noise)

## Goal
Suppress flickering/moving false positives by requiring temporal persistence, while keeping
original-frame labels via 1-frame backfill.

## Overview
The gate operates after segmentation and before matching. A component is confirmed if it overlaps
either (a) the previous confirmed frame or (b) the next frame. Overlaps are computed on flow-warped
labels to tolerate motion. Components that do not meet the overlap requirement are dropped for that
frame. This filters single-frame noise while preserving bubbles that persist across frames.

## Data Flow (1-Frame Backfill)
1) Segment current frame (raw labels).
2) Confirm the previous frame using overlaps with the previous confirmed frame and the current raw
   frame (flow-warped).
3) Run flow+match on confirmed frames t-2 -> t-1; assign IDs/events for frame t-1.
4) Write masks and track data for frame t-1 (backfill).

The last frame is confirmed using only overlap with the previous confirmed frame.

## Config
- `persistence.min_overlap_pixels` (default 0): minimum overlap to keep a component (0 disables).
- `persistence.max_area` (default 0): if set, components larger than this bypass gating.

CLI flags:
- `--persistence-min-overlap-px`
- `--persistence-max-area`

## Edge Cases
- First frame is confirmed using overlap with frame 1 (if available).
- Single-frame runs keep all components even if persistence is enabled.
- Last-frame singletons without overlap to the previous confirmed frame are dropped.

## Output Implications
- Birth events are delayed in processing time, but recorded on their original frame indices.
- Tracks and masks are consistent with original frames; only logging is delayed.
