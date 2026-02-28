"""
DAVIS 2017 evaluation utilities.
"""

import os
from typing import Dict, List, Optional

import numpy as np

from .metrics import compute_iou, compute_boundary_f, compute_sequence_jf


def evaluate_davis_sequence(
    pred_masks_dir: str,
    gt_masks_dir: str,
) -> Dict[str, float]:
    """
    Evaluate a single DAVIS sequence.

    Args:
        pred_masks_dir: Directory with predicted masks (00000.png, ...)
        gt_masks_dir: Directory with ground truth masks

    Returns:
        Dict with J, F, and J&F scores
    """
    import cv2
    from PIL import Image

    def load_mask_as_index(path: str) -> np.ndarray:
        """Load mask as single-channel index array (handles palette PNGs)."""
        # Use PIL to properly load indexed/palette PNGs
        img = Image.open(path)
        # Convert to numpy - PIL handles palette correctly
        mask = np.array(img)
        # If still 3-channel (RGB), convert to single channel via unique colors
        if len(mask.shape) == 3:
            # Assume each unique RGB = unique object; convert to index
            h, w = mask.shape[:2]
            flat = mask.reshape(-1, mask.shape[-1])
            _, indices = np.unique(flat, axis=0, return_inverse=True)
            mask = indices.reshape(h, w).astype(np.uint8)
        return mask

    # Load masks
    pred_files = sorted([f for f in os.listdir(pred_masks_dir) if f.endswith(".png")])
    gt_files = sorted([f for f in os.listdir(gt_masks_dir) if f.endswith(".png")])

    pred_masks = []
    gt_masks = []

    for pred_file, gt_file in zip(pred_files, gt_files):
        pred_path = os.path.join(pred_masks_dir, pred_file)
        gt_path = os.path.join(gt_masks_dir, gt_file)

        if not os.path.exists(pred_path) or not os.path.exists(gt_path):
            continue

        pred = load_mask_as_index(pred_path)
        gt = load_mask_as_index(gt_path)

        pred_masks.append(pred)
        gt_masks.append(gt)

    return compute_sequence_jf(pred_masks, gt_masks)


def evaluate_davis_dataset(
    pred_dir: str,
    gt_dir: str,
    sequences: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate all sequences in a DAVIS dataset.

    Args:
        pred_dir: Root directory with prediction subdirectories per sequence
        gt_dir: Root directory with GT subdirectories per sequence
        sequences: Optional list of sequences to evaluate (all if None)

    Returns:
        Dict mapping sequence name to metrics, plus "mean" entry
    """
    if sequences is None:
        sequences = sorted([
            d for d in os.listdir(gt_dir)
            if os.path.isdir(os.path.join(gt_dir, d))
        ])

    results = {}
    all_j = []
    all_f = []

    for seq in sequences:
        pred_seq_dir = os.path.join(pred_dir, seq)
        gt_seq_dir = os.path.join(gt_dir, seq)

        if not os.path.exists(pred_seq_dir):
            print(f"Skipping {seq}: no predictions found")
            continue

        metrics = evaluate_davis_sequence(pred_seq_dir, gt_seq_dir)
        results[seq] = metrics
        all_j.append(metrics["J"])
        all_f.append(metrics["F"])

    # Compute mean across sequences
    if all_j:
        results["mean"] = {
            "J": float(np.mean(all_j)),
            "F": float(np.mean(all_f)),
            "J&F": float(np.mean(all_j) + np.mean(all_f)) / 2,
        }
    else:
        results["mean"] = {"J": 0.0, "F": 0.0, "J&F": 0.0}

    return results


def print_davis_results(results: Dict[str, Dict[str, float]]) -> None:
    """Print DAVIS evaluation results in a table format."""
    print("\n" + "=" * 50)
    print("DAVIS 2017 Evaluation Results")
    print("=" * 50)
    print(f"{'Sequence':<20} {'J':>8} {'F':>8} {'J&F':>8}")
    print("-" * 50)

    for seq, metrics in sorted(results.items()):
        if seq == "mean":
            continue
        print(f"{seq:<20} {metrics['J']:>8.3f} {metrics['F']:>8.3f} {metrics['J&F']:>8.3f}")

    print("-" * 50)
    if "mean" in results:
        m = results["mean"]
        print(f"{'Mean':<20} {m['J']:>8.3f} {m['F']:>8.3f} {m['J&F']:>8.3f}")
    print("=" * 50)

