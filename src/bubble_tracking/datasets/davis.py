"""
DAVIS 2017 dataset loader for video object segmentation.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class DAVISSequence:
    """A single DAVIS video sequence."""
    name: str
    frames_dir: str
    masks_dir: str
    num_frames: int
    object_ids: List[int]


class DAVISDataset:
    """
    DAVIS 2017 dataset loader.

    Supports loading frames and ground truth masks for VOS evaluation.
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "val",
        resolution: str = "480p",
    ):
        """
        Initialize DAVIS dataset.

        Args:
            root_dir: Path to DAVIS dataset root (contains DAVIS/)
            split: Dataset split ("train", "val", or "trainval")
            resolution: Resolution variant ("480p" or "Full-Resolution")
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.resolution = resolution

        # Find the actual DAVIS directory
        davis_dir = self.root_dir
        if (self.root_dir / "DAVIS").exists():
            davis_dir = self.root_dir / "DAVIS"

        self.images_dir = davis_dir / "JPEGImages" / resolution
        self.masks_dir = davis_dir / "Annotations" / resolution
        self.imagesets_dir = davis_dir / "ImageSets" / "2017"

        self._sequences: Optional[List[str]] = None

    @property
    def sequences(self) -> List[str]:
        """Get list of sequence names for this split."""
        if self._sequences is None:
            self._sequences = self._load_split()
        return self._sequences

    def _load_split(self) -> List[str]:
        """Load sequence names from split file."""
        split_file = self.imagesets_dir / f"{self.split}.txt"
        if split_file.exists():
            with open(split_file, "r") as f:
                return [line.strip() for line in f if line.strip()]

        # Fallback: list directories
        if self.images_dir.exists():
            return sorted([d.name for d in self.images_dir.iterdir() if d.is_dir()])

        return []

    def get_sequence(self, name: str) -> DAVISSequence:
        """Get a sequence by name."""
        frames_dir = str(self.images_dir / name)
        masks_dir = str(self.masks_dir / name)

        if not os.path.exists(frames_dir):
            raise ValueError(f"Sequence not found: {name}")

        # Count frames
        frame_files = sorted([
            f for f in os.listdir(frames_dir)
            if f.endswith((".jpg", ".png"))
        ])
        num_frames = len(frame_files)

        # Get object IDs from first mask
        first_mask_path = os.path.join(masks_dir, frame_files[0].replace(".jpg", ".png"))
        object_ids = []
        if os.path.exists(first_mask_path):
            mask = cv2.imread(first_mask_path, cv2.IMREAD_UNCHANGED)
            if mask is not None:
                object_ids = sorted([int(x) for x in np.unique(mask) if x > 0])

        return DAVISSequence(
            name=name,
            frames_dir=frames_dir,
            masks_dir=masks_dir,
            num_frames=num_frames,
            object_ids=object_ids,
        )

    def load_frame(self, sequence: str, frame_idx: int) -> np.ndarray:
        """Load a frame image (BGR format)."""
        frame_path = self.images_dir / sequence / f"{frame_idx:05d}.jpg"
        frame = cv2.imread(str(frame_path))
        if frame is None:
            raise ValueError(f"Cannot load frame: {frame_path}")
        return frame

    def load_mask(self, sequence: str, frame_idx: int) -> np.ndarray:
        """Load a ground truth mask."""
        mask_path = self.masks_dir / sequence / f"{frame_idx:05d}.png"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            # Return empty mask if not available
            frame = self.load_frame(sequence, frame_idx)
            return np.zeros(frame.shape[:2], dtype=np.uint8)
        return mask

    def iter_sequence_frames(
        self,
        sequence: str,
        with_masks: bool = True,
    ) -> Iterator[Tuple[int, np.ndarray, Optional[np.ndarray]]]:
        """
        Iterate over frames in a sequence.

        Yields:
            Tuples of (frame_idx, frame, mask) where mask may be None
        """
        seq = self.get_sequence(sequence)

        for frame_idx in range(seq.num_frames):
            frame = self.load_frame(sequence, frame_idx)
            mask = None
            if with_masks:
                try:
                    mask = self.load_mask(sequence, frame_idx)
                except Exception:
                    pass
            yield frame_idx, frame, mask

    def get_first_frame_prompts(self, sequence: str) -> Dict[int, np.ndarray]:
        """
        Get prompts from first frame ground truth masks.

        Returns:
            Dict mapping object_id to binary mask for first frame
        """
        mask = self.load_mask(sequence, 0)
        prompts = {}
        for obj_id in np.unique(mask):
            if obj_id > 0:
                prompts[int(obj_id)] = (mask == obj_id).astype(np.uint8)
        return prompts

    def __len__(self) -> int:
        return len(self.sequences)

    def __iter__(self) -> Iterator[str]:
        return iter(self.sequences)

