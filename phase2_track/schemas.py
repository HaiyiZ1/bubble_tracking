"""Dataclasses + v1 frozen constants for Phase 2.

Translated from src/bubble_tracking/schemas.py and the config dataclasses
at the top of src/bubble_tracking/flow_tracking.py.

Stage 3 of phase-boundary refactor: dataclass definitions moved to
bubble_common/schemas.py; re-exported here for back-compat.
"""

from __future__ import annotations

from typing import Dict, Tuple


# ---------------------------------------------------------------------------
# v1 frozen constants (see docs/superpowers/specs/2026-05-10-three-phase-...md).
# Values that are identical across case1/2/3 publishing runs and not exposed
# as CLI flags in v1. v2 may promote a subset back to flags.
# ---------------------------------------------------------------------------

CV_KEYFRAME_INTERVAL: int = 160
CV_KEYFRAME_PREPROCESS_MODE: str = "default"
CV_KEYFRAME_DIFF_SIGMA: float = 0.5
CV_KEYFRAME_NORMALIZE_CONTRAST: bool = False  # matches --cv-keyframe-no-normalize
# Historical default from original scripts/run_sam2_prompted_flow.py (was 25 before the
# median_ksize default was lowered to 10 in flow_tracking.py). Matching the publishing
# oracle runs requires 25 here.
CV_KEYFRAME_MEDIAN_KSIZE: int = 25

PREPROCESS_MODE: str = "raw"
PREPROCESS_RAW_SIGMA: float = 1.0
VELOCITY_MODE: str = "centroid_delta"
USE_FLOW: bool = True
FLOW_BACKEND: str = "auto"
DEBOUNCE_BIRTH_FRAMES: int = 2
DEBOUNCE_DEATH_FRAMES: int = 3
SAM2_FILL_HOLE_AREA: int = 8

# ---------------------------------------------------------------------------
# SAM2 model registry (--sam2-model shorthand)
# ---------------------------------------------------------------------------

SAM2_MODELS: Dict[str, Tuple[str, str]] = {
    "tiny": ("checkpoints/sam2_hiera_tiny.pt", "sam2_hiera_t.yaml"),
    "small": ("checkpoints/sam2_hiera_small.pt", "sam2_hiera_s.yaml"),
    "base_plus": ("checkpoints/sam2_hiera_base_plus.pt", "sam2_hiera_b+.yaml"),
    "large": ("checkpoints/sam2_hiera_large.pt", "sam2_hiera_l.yaml"),
}

# ---------------------------------------------------------------------------
# Dataclass re-exports from bubble_common.schemas (for back-compat with
# phase2_track internal imports). Stage 9 will remove these.
# ---------------------------------------------------------------------------

from bubble_common.schemas import (
    BoundingBox, FrameData, TrackedObject, TrackingResult,
    PointPrompt, BoxPrompt, ObjectPrompt, PromptSpec, Component,
    PreprocessConfig, SegmentationConfig, FlowConfig, MatchConfig,
    PersistenceConfig, DebounceConfig, TrackingConfig,
)

__all__ = [
    # Constants
    "CV_KEYFRAME_INTERVAL", "CV_KEYFRAME_PREPROCESS_MODE",
    "CV_KEYFRAME_DIFF_SIGMA", "CV_KEYFRAME_NORMALIZE_CONTRAST",
    "CV_KEYFRAME_MEDIAN_KSIZE", "PREPROCESS_MODE", "PREPROCESS_RAW_SIGMA",
    "VELOCITY_MODE", "USE_FLOW", "FLOW_BACKEND",
    "DEBOUNCE_BIRTH_FRAMES", "DEBOUNCE_DEATH_FRAMES",
    "SAM2_FILL_HOLE_AREA", "SAM2_MODELS",
    # Dataclasses (re-exported)
    "BoundingBox", "FrameData", "TrackedObject", "TrackingResult",
    "PointPrompt", "BoxPrompt", "ObjectPrompt", "PromptSpec", "Component",
    "PreprocessConfig", "SegmentationConfig", "FlowConfig", "MatchConfig",
    "PersistenceConfig", "DebounceConfig", "TrackingConfig",
]
