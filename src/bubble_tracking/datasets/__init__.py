"""
Dataset loaders for biomedical and benchmark datasets.
"""

from .davis import DAVISDataset
from .ctc import CTCDataset

__all__ = ["DAVISDataset", "CTCDataset"]

