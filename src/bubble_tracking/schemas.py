"""
Data schemas for bubble tracking pipeline.

Defines the canonical formats for tracks, masks, and prompts.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime


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

    def to_dict(self) -> dict:
        return {
            "frames": {
                str(frame_idx): frame_data.to_dict()
                for frame_idx, frame_data in self.frames.items()
            }
        }

    @classmethod
    def from_dict(cls, object_id: int, data: dict) -> "TrackedObject":
        frames = {
            int(frame_idx): FrameData.from_dict(frame_data)
            for frame_idx, frame_data in data["frames"].items()
        }
        return cls(object_id=object_id, frames=frames)


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
