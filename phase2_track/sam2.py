"""SAM2 wrapper for Phase 2.

Translated from src/bubble_tracking/sam2_adapter.py.

CV-keyframe prompt builder functions have been moved to
phase1_preprocess/cv_keyframes.py (Stage 5+6 of phase-boundary refactor).

frame_io helpers (sorted_frame_paths, save_mask) and all dataclasses are
imported from bubble_common/ as of the phase-boundary refactor.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from bubble_common.io import sorted_frame_paths, save_mask
from bubble_common.schemas import (
    BoundingBox,
    FrameData,
    PromptSpec,
    TrackedObject,
    TrackingResult,
)
from .schemas import SAM2_FILL_HOLE_AREA


def compute_mask_properties(mask: np.ndarray, object_id: int) -> Optional[FrameData]:
    """
    Compute centroid, area, and bounding box for an object in a mask.

    Args:
        mask: 2D array with instance IDs
        object_id: The object ID to analyze

    Returns:
        FrameData with computed properties, or None if object not in mask
    """
    binary = (mask == object_id)
    coords = np.where(binary)

    if len(coords[0]) == 0:
        return None

    area = int(np.sum(binary))
    centroid_y = float(np.mean(coords[0]))
    centroid_x = float(np.mean(coords[1]))

    y_min, y_max = int(np.min(coords[0])), int(np.max(coords[0]))
    x_min, x_max = int(np.min(coords[1])), int(np.max(coords[1]))

    return FrameData(
        centroid=(centroid_x, centroid_y),
        area=area,
        bbox=BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max),
    )


@dataclass
class SAM2Config:
    """Configuration for SAM2 runner."""
    checkpoint_path: str
    model_cfg: str = "sam2_hiera_t.yaml"
    device: str = "cpu"
    # SAM2's video predictor fills small background holes (area <= fill_hole_area)
    # in low-res masks. Default 8 matches upstream SAM2; set to 0 to preserve
    # genuine holes (e.g. porous bubbles whose interior holes are real signal).
    fill_hole_area: int = 8
    # Push frame tensors to CPU between accesses (saves GPU memory at ~5-15% latency cost).
    # v1 default: True (offload enabled by default).
    offload_video_to_cpu: bool = True


class SAM2Runner:
    """
    Runner for SAM2 video segmentation and tracking.

    Usage:
        runner = SAM2Runner(config)
        result = runner.run(frames, prompts)
    """

    def __init__(self, config: SAM2Config):
        self.config = config
        self.predictor = None
        self._load_model()

    def _load_model(self) -> None:
        """Load SAM2 model and predictor."""
        try:
            from sam2.build_sam import build_sam2_video_predictor
        except ImportError:
            raise ImportError(
                "SAM2 not installed. Please install with:\n"
                "  git clone https://github.com/facebookresearch/segment-anything-2.git\n"
                "  cd segment-anything-2 && pip install -e ."
            )

        self.predictor = build_sam2_video_predictor(
            self.config.model_cfg,
            self.config.checkpoint_path,
            device=self.config.device,
            hydra_overrides_extra=[
                f"++model.fill_hole_area={int(self.config.fill_hole_area)}",
            ],
        )

    def run(
        self,
        frames_dir: str,
        prompts: PromptSpec,
        output_dir: Optional[str] = None,
        roi_mask: Optional[np.ndarray] = None,
    ) -> Tuple[TrackingResult, Dict[int, np.ndarray]]:
        """
        Run SAM2 video tracking on a sequence of frames.

        Args:
            frames_dir: Directory containing frame images
            prompts: Prompt specification for object initialization
            output_dir: Optional directory to save masks

        Returns:
            Tuple of (TrackingResult, dict of frame_idx -> mask)
        """
        frame_paths = sorted_frame_paths(frames_dir)
        num_frames = len(frame_paths)

        if num_frames == 0:
            raise ValueError(f"No frames found in {frames_dir}")

        # Initialize predictor state with video
        inference_state = self.predictor.init_state(
            video_path=frames_dir,
            offload_video_to_cpu=bool(self.config.offload_video_to_cpu),
        )

        # Add prompts for each object
        object_ids = set()
        for prompt in prompts.prompts:
            object_ids.add(prompt.object_id)

            has_mask = getattr(prompt, "mask", None) is not None
            has_points = bool(prompt.points)
            has_box = prompt.box is not None

            # When both mask and point/box prompts are provided on the same frame,
            # call add_new_mask first so its prediction logits seed the subsequent
            # add_new_points_or_box call via SAM2's prev_sam_mask_logits path.
            # This lets negative points (e.g., placed inside holes) refine a
            # mask seed instead of replacing it.
            if has_mask:
                self.predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=prompt.frame_idx,
                    obj_id=prompt.object_id,
                    mask=prompt.mask,
                )

            if has_points or has_box:
                points = None
                labels = None
                box = None

                if has_points:
                    points = np.array([[p.x, p.y] for p in prompt.points])
                    labels = np.array([p.label for p in prompt.points])

                if has_box:
                    box = np.array(prompt.box.bbox.to_list())

                self.predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=prompt.frame_idx,
                    obj_id=prompt.object_id,
                    points=points,
                    labels=labels,
                    box=box,
                )

        # Propagate through video
        masks_by_frame: Dict[int, np.ndarray] = {}
        objects: Dict[int, TrackedObject] = {
            obj_id: TrackedObject(object_id=obj_id) for obj_id in object_ids
        }

        # Forward propagation
        for frame_idx, object_ids_out, masks_out in self.predictor.propagate_in_video(
            inference_state
        ):
            if torch.is_tensor(object_ids_out):
                object_ids_out = object_ids_out.tolist()
            if torch.is_tensor(masks_out):
                masks_out = masks_out.detach().cpu().numpy()
            # Combine masks into single instance mask
            combined_mask = np.zeros(masks_out.shape[-2:], dtype=np.int32)

            for i, obj_id in enumerate(object_ids_out):
                # masks_out shape: (num_objects, 1, H, W)
                obj_mask = masks_out[i, 0] > 0.0
                combined_mask[obj_mask] = obj_id

            if roi_mask is not None:
                if roi_mask.shape != combined_mask.shape:
                    raise ValueError("ROI mask size does not match mask output size")
                combined_mask[roi_mask == 0] = 0

            # Compute per-object properties from the final per-frame mask.
            for obj_id in object_ids_out:
                obj_id = int(obj_id)
                frame_data = compute_mask_properties(combined_mask, obj_id)
                if frame_data and obj_id in objects:
                    objects[obj_id].frames[frame_idx] = frame_data

            masks_by_frame[frame_idx] = combined_mask

        # Save masks if output directory specified
        if output_dir:
            masks_dir = os.path.join(output_dir, "masks")
            os.makedirs(masks_dir, exist_ok=True)
            for frame_idx, mask in masks_by_frame.items():
                save_mask(mask, os.path.join(masks_dir, f"{frame_idx:06d}.png"))

        # Build result
        result = TrackingResult(
            num_frames=num_frames,
            num_objects=len(objects),
            created_at=datetime.now(),
            objects=objects,
        )

        return result, masks_by_frame

    def reset(self) -> None:
        """Reset predictor state for a new video."""
        # SAM2 predictor state is managed per-video, no explicit reset needed
        pass


def create_runner(
    checkpoint_path: str,
    model_cfg: str = "sam2_hiera_t.yaml",
    device: Optional[str] = None,
    fill_hole_area: int = 8,
    offload_video_to_cpu: bool = True,
) -> SAM2Runner:
    """
    Factory function to create a SAM2 runner.

    Args:
        checkpoint_path: Path to SAM2 checkpoint file
        model_cfg: SAM2 model configuration name
        device: Device to use ("cpu", "cuda", or None for auto-detect)

    Returns:
        Configured SAM2Runner instance
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    config = SAM2Config(
        checkpoint_path=checkpoint_path,
        model_cfg=model_cfg,
        device=device,
        fill_hole_area=fill_hole_area,
        offload_video_to_cpu=offload_video_to_cpu,
    )
    return SAM2Runner(config)
