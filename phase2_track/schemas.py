"""Dataclasses + v1 frozen constants for Phase 2.

Translated from src/bubble_tracking/schemas.py and the config dataclasses
at the top of src/bubble_tracking/flow_tracking.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# v1 frozen constants (see docs/superpowers/specs/2026-05-10-three-phase-...md).
# Values that are identical across case1/2/3 publishing runs and not exposed
# as CLI flags in v1. v2 may promote a subset back to flags.
# ---------------------------------------------------------------------------

CV_KEYFRAME_INTERVAL: int = 160
CV_KEYFRAME_PREPROCESS_MODE: str = "default"
CV_KEYFRAME_DIFF_SIGMA: float = 0.5
CV_KEYFRAME_NORMALIZE_CONTRAST: bool = False  # matches --cv-keyframe-no-normalize

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
# Per-frame / per-object schemas
# Translated from src/bubble_tracking/schemas.py
# ---------------------------------------------------------------------------


@dataclass
class BoundingBox:
    """Axis-aligned bounding box [x_min, y_min, x_max, y_max]."""
    x_min: int
    y_min: int
    x_max: int
    y_max: int

    def to_list(self) -> List[int]:
        return [self.x_min, self.y_min, self.x_max, self.y_max]

    @classmethod
    def from_list(cls, bbox: List[int]) -> "BoundingBox":
        return cls(x_min=bbox[0], y_min=bbox[1], x_max=bbox[2], y_max=bbox[3])


@dataclass
class FrameData:
    """Per-frame data for a single tracked object."""
    centroid: Tuple[float, float]  # (x, y) in pixels
    area: int  # number of pixels
    bbox: Optional[BoundingBox] = None
    velocity: Optional[Tuple[float, float]] = None  # (vx, vy) in pixels/sec

    def to_dict(self) -> dict:
        result = {
            "centroid": list(self.centroid),
            "area": self.area,
        }
        if self.bbox:
            result["bbox"] = self.bbox.to_list()
        if self.velocity is not None:
            result["velocity"] = list(self.velocity)
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "FrameData":
        bbox = None
        if "bbox" in data:
            bbox = BoundingBox.from_list(data["bbox"])
        velocity = None
        if "velocity" in data:
            velocity = tuple(data["velocity"])
        return cls(
            centroid=tuple(data["centroid"]),
            area=data["area"],
            bbox=bbox,
            velocity=velocity,
        )


@dataclass
class TrackedObject:
    """A single tracked object across frames."""
    object_id: int
    frames: Dict[int, FrameData] = field(default_factory=dict)
    birth_reason: Optional[str] = None
    birth_frame_idx: Optional[int] = None
    birth_parent_ids: List[int] = field(default_factory=list)
    death_reason: Optional[str] = None
    death_frame_idx: Optional[int] = None
    death_child_ids: List[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        result = {
            "frames": {
                str(frame_idx): frame_data.to_dict()
                for frame_idx, frame_data in self.frames.items()
            }
        }
        if self.birth_reason is not None:
            result["birth_reason"] = self.birth_reason
            if self.birth_frame_idx is not None:
                result["birth_frame_idx"] = self.birth_frame_idx
            result["birth_parent_ids"] = [int(v) for v in self.birth_parent_ids]
        if self.death_reason is not None:
            result["death_reason"] = self.death_reason
            if self.death_frame_idx is not None:
                result["death_frame_idx"] = self.death_frame_idx
            result["death_child_ids"] = [int(v) for v in self.death_child_ids]
        return result

    @classmethod
    def from_dict(cls, object_id: int, data: dict) -> "TrackedObject":
        frames = {
            int(frame_idx): FrameData.from_dict(frame_data)
            for frame_idx, frame_data in data["frames"].items()
        }
        return cls(
            object_id=object_id,
            frames=frames,
            birth_reason=data.get("birth_reason"),
            birth_frame_idx=data.get("birth_frame_idx"),
            birth_parent_ids=[int(v) for v in data.get("birth_parent_ids", [])],
            death_reason=data.get("death_reason"),
            death_frame_idx=data.get("death_frame_idx"),
            death_child_ids=[int(v) for v in data.get("death_child_ids", [])],
        )


@dataclass
class TrackingResult:
    """Complete tracking result for a video sequence."""
    num_frames: int
    num_objects: int
    created_at: datetime
    objects: Dict[int, TrackedObject] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "metadata": {
                "num_frames": self.num_frames,
                "num_objects": self.num_objects,
                "created_at": self.created_at.isoformat(),
            },
            "objects": {
                str(obj_id): obj.to_dict()
                for obj_id, obj in self.objects.items()
            }
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TrackingResult":
        metadata = data["metadata"]
        objects = {
            int(obj_id): TrackedObject.from_dict(int(obj_id), obj_data)
            for obj_id, obj_data in data["objects"].items()
        }
        return cls(
            num_frames=metadata["num_frames"],
            num_objects=metadata["num_objects"],
            created_at=datetime.fromisoformat(metadata["created_at"]),
            objects=objects,
        )


@dataclass
class PointPrompt:
    """A point prompt for SAM2."""
    x: int
    y: int
    label: int = 1  # 1 = foreground, 0 = background


@dataclass
class BoxPrompt:
    """A box prompt for SAM2."""
    bbox: BoundingBox


@dataclass
class ObjectPrompt:
    """Prompts for initializing a single object on a specific frame."""
    object_id: int
    frame_idx: int
    points: List[PointPrompt] = field(default_factory=list)
    box: Optional[BoxPrompt] = None
    # Optional in-memory mask prompt (2D numpy-like bool/int array). Not serialized to YAML/JSON.
    mask: Optional[Any] = None


@dataclass
class PromptSpec:
    """Complete prompt specification for a video."""
    prompts: List[ObjectPrompt] = field(default_factory=list)

    def to_dict(self) -> dict:
        result = {"prompts": []}
        for prompt in self.prompts:
            p = {
                "object_id": prompt.object_id,
                "frame_idx": prompt.frame_idx,
            }
            if prompt.points:
                p["points"] = [
                    {"x": pt.x, "y": pt.y, "label": pt.label}
                    for pt in prompt.points
                ]
            if prompt.box:
                p["box"] = prompt.box.bbox.to_list()
            result["prompts"].append(p)
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "PromptSpec":
        prompts = []
        for p in data.get("prompts", []):
            points = [
                PointPrompt(x=pt["x"], y=pt["y"], label=pt.get("label", 1))
                for pt in p.get("points", [])
            ]
            box = None
            if "box" in p:
                box = BoxPrompt(bbox=BoundingBox.from_list(p["box"]))
            prompts.append(ObjectPrompt(
                object_id=p["object_id"],
                frame_idx=p["frame_idx"],
                points=points,
                box=box,
            ))
        return cls(prompts=prompts)


# ---------------------------------------------------------------------------
# Pipeline config dataclasses
# Translated from src/bubble_tracking/flow_tracking.py lines 26-113
# ---------------------------------------------------------------------------


@dataclass
class PreprocessConfig:
    diff_sigma: float = 0.5
    clip_percentiles: Tuple[float, float] = (1.0, 99.0)
    grid_rows: int = 8
    grid_cols: int = 5
    light_percentile: float = 95.0
    min_bright_pixels: int = 10
    light_tolerance: float = 10.0
    use_clahe: bool = False
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: Tuple[int, int] = (8, 8)
    mode: str = "default"
    raw_sigma: float = 1.0


@dataclass
class SegmentationConfig:
    blur_ksize: int = 3
    hist_bins: int = 35
    threshold_offset: int = 7
    max_threshold: int = 248
    opening_radius: int = 0
    min_area: int = 500
    median_ksize: int = 10
    compact_max_area: int = 0
    compact_min_dark_delta: float = 0.0
    automatrix_median_radius: int = 12
    edge_min_strength: float = 0.0
    edge_boundary_radius: int = 1
    edge_min_fraction: float = 0.0


@dataclass
class FlowConfig:
    pyr_scale: float = 0.5
    levels: int = 3
    winsize: int = 15
    iterations: int = 3
    poly_n: int = 5
    poly_sigma: float = 1.2
    fast_pyramids: bool = False
    flags: int = 0


@dataclass
class MatchConfig:
    min_overlap_pixels: int = 10
    min_link_iou: float = 0.05
    min_link_overlap_ratio: float = 0.25


@dataclass
class PersistenceConfig:
    min_overlap_pixels: int = 0
    max_area: int = 0


@dataclass
class DebounceConfig:
    # Require this many consecutive detections before a new ID is kept.
    birth_confirm_frames: int = 1
    # Require this many consecutive missing frames before an ID is removed.
    death_confirm_frames: int = 1
    # Max centroid distance (pixels) when re-linking a new ID to a recently missing ID.
    relink_max_centroid_dist_px: float = 20.0
    # Area ratio bounds (new/old) for ID re-linking.
    relink_min_area_ratio: float = 0.35
    relink_max_area_ratio: float = 2.8


@dataclass
class TrackingConfig:
    fps: float = 4.0
    mm_per_pixel: Optional[float] = None
    flow_backend: str = "auto"
    cuda_device: int = 0
    log_every: int = 10
    use_flow: bool = True
    velocity_mode: str = "flow_median"
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    segment: SegmentationConfig = field(default_factory=SegmentationConfig)
    flow: FlowConfig = field(default_factory=FlowConfig)
    match: MatchConfig = field(default_factory=MatchConfig)
    persistence: PersistenceConfig = field(default_factory=PersistenceConfig)
    debounce: DebounceConfig = field(default_factory=DebounceConfig)
