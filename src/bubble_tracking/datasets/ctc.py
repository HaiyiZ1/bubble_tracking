"""
Cell Tracking Challenge dataset loader.

Supports loading sequences from CTC format (DIC-C2DH-HeLa, etc.)
"""

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class CTCSequence:
    """A single CTC sequence."""
    name: str
    frames_dir: str
    gt_seg_dir: Optional[str]
    gt_tra_dir: Optional[str]
    num_frames: int


class CTCDataset:
    """
    Cell Tracking Challenge dataset loader.

    CTC datasets have structure:
        DIC-C2DH-HeLa/
        ├── 01/             # Sequence 1 images
        │   ├── t000.tif
        │   ├── t001.tif
        │   └── ...
        ├── 01_GT/          # Ground truth for sequence 1
        │   ├── SEG/        # Segmentation masks
        │   └── TRA/        # Tracking masks + lineage
        ├── 02/             # Sequence 2
        └── 02_GT/
    """

    def __init__(self, root_dir: str):
        """
        Initialize CTC dataset.

        Args:
            root_dir: Path to dataset root (e.g., path/to/DIC-C2DH-HeLa)
        """
        self.root_dir = Path(root_dir)
        self._sequences: Optional[List[str]] = None

    @property
    def sequences(self) -> List[str]:
        """Get list of sequence names."""
        if self._sequences is None:
            self._sequences = self._find_sequences()
        return self._sequences

    def _find_sequences(self) -> List[str]:
        """Find all sequences in the dataset."""
        sequences = []
        for item in self.root_dir.iterdir():
            if item.is_dir() and re.match(r"^\d+$", item.name):
                sequences.append(item.name)
        return sorted(sequences)

    def get_sequence(self, name: str) -> CTCSequence:
        """Get a sequence by name."""
        frames_dir = self.root_dir / name
        gt_dir = self.root_dir / f"{name}_GT"

        if not frames_dir.exists():
            raise ValueError(f"Sequence not found: {name}")

        gt_seg_dir = gt_dir / "SEG" if (gt_dir / "SEG").exists() else None
        gt_tra_dir = gt_dir / "TRA" if (gt_dir / "TRA").exists() else None

        # Count frames
        num_frames = len([f for f in frames_dir.iterdir() if f.suffix.lower() in (".tif", ".tiff", ".png")])

        return CTCSequence(
            name=name,
            frames_dir=str(frames_dir),
            gt_seg_dir=str(gt_seg_dir) if gt_seg_dir else None,
            gt_tra_dir=str(gt_tra_dir) if gt_tra_dir else None,
            num_frames=num_frames,
        )

    def load_frame(self, sequence: str, frame_idx: int) -> np.ndarray:
        """
        Load a frame image.

        CTC uses t000.tif, t001.tif, ... naming convention.
        """
        seq_dir = self.root_dir / sequence

        # Try different extensions
        for ext in [".tif", ".tiff", ".png"]:
            frame_path = seq_dir / f"t{frame_idx:03d}{ext}"
            if frame_path.exists():
                # Use cv2 with IMREAD_UNCHANGED for 16-bit images
                frame = cv2.imread(str(frame_path), cv2.IMREAD_UNCHANGED)
                if frame is not None:
                    # Convert to 8-bit if needed
                    if frame.dtype == np.uint16:
                        frame = (frame / 256).astype(np.uint8)
                    # Convert grayscale to BGR for consistency
                    if len(frame.shape) == 2:
                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    return frame

        raise ValueError(f"Frame not found: sequence={sequence}, idx={frame_idx}")

    def load_gt_seg(self, sequence: str, frame_idx: int) -> Optional[np.ndarray]:
        """Load ground truth segmentation mask."""
        gt_dir = self.root_dir / f"{sequence}_GT" / "SEG"
        if not gt_dir.exists():
            return None

        # CTC SEG uses man_seg000.tif naming
        for ext in [".tif", ".tiff", ".png"]:
            mask_path = gt_dir / f"man_seg{frame_idx:03d}{ext}"
            if mask_path.exists():
                mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
                return mask

        return None

    def load_gt_tra(self, sequence: str, frame_idx: int) -> Optional[np.ndarray]:
        """Load ground truth tracking mask."""
        gt_dir = self.root_dir / f"{sequence}_GT" / "TRA"
        if not gt_dir.exists():
            return None

        # CTC TRA uses man_track000.tif naming
        for ext in [".tif", ".tiff", ".png"]:
            mask_path = gt_dir / f"man_track{frame_idx:03d}{ext}"
            if mask_path.exists():
                mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
                return mask

        return None

    def iter_sequence_frames(
        self,
        sequence: str,
        with_gt: bool = True,
    ) -> Iterator[Tuple[int, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]]:
        """
        Iterate over frames in a sequence.

        Yields:
            Tuples of (frame_idx, frame, seg_mask, tra_mask)
        """
        seq = self.get_sequence(sequence)

        for frame_idx in range(seq.num_frames):
            frame = self.load_frame(sequence, frame_idx)
            seg_mask = self.load_gt_seg(sequence, frame_idx) if with_gt else None
            tra_mask = self.load_gt_tra(sequence, frame_idx) if with_gt else None
            yield frame_idx, frame, seg_mask, tra_mask

    def get_object_ids_from_first_frame(self, sequence: str) -> List[int]:
        """Get object IDs from first frame ground truth."""
        tra_mask = self.load_gt_tra(sequence, 0)
        if tra_mask is None:
            seg_mask = self.load_gt_seg(sequence, 0)
            if seg_mask is None:
                return []
            tra_mask = seg_mask

        return sorted([int(x) for x in np.unique(tra_mask) if x > 0])

    def __len__(self) -> int:
        return len(self.sequences)

    def __iter__(self) -> Iterator[str]:
        return iter(self.sequences)

