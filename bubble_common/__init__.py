"""Shared building blocks for the bubble-tracking 3-phase pipeline.

Contains code that both Phase 1 and Phase 2 need:
- schemas:  dataclasses (BoundingBox/FrameData/.../PromptSpec) + config dataclasses
- preprocess:  reference-diff + light-map + smoothing (PreprocessConfig + preprocess_frame)
- segment:    threshold + CC analysis + filters (SegmentationConfig + segment_frame)
- run_config: <run-dir>/run_config.json contract (RunConfig + load + mark_phase_done)
- io:         sorted_frame_paths + save_mask + load_mask
"""

__all__: list[str] = []
