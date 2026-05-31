"""Re-exports from bubble_common.preprocess for back-compat.

This module previously held the preprocessing functions. They moved to
bubble_common/ in the phase-boundary refactor. Imports here still work.
"""
from bubble_common.preprocess import (
    preprocess_frame,
    preprocess_frame_with_debug,
    _preprocess_frame_default_with_debug,
    _preprocess_frame_raw_with_debug,
    _estimate_light_map,
)
from bubble_common.schemas import PreprocessConfig

__all__ = [
    "preprocess_frame",
    "preprocess_frame_with_debug",
    "PreprocessConfig",
    # private helpers re-exported for compat with old debug scripts
    "_preprocess_frame_default_with_debug",
    "_preprocess_frame_raw_with_debug",
    "_estimate_light_map",
]
