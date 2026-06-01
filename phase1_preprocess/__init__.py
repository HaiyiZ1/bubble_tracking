"""Phase 1 of the bubble-tracking pipeline: frame prep + ROI + preprocessing.

Reads case metadata from meta.yaml, downscales (optionally relabels) raw
frames, captures ROI, and writes preprocessed keyframe previews and a
run_config.json that Phase 2 consumes.
"""

__all__: list[str] = []
