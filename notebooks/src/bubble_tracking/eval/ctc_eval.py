"""
Cell Tracking Challenge evaluation utilities.

Implements simplified versions of CTC SEG and TRA metrics.
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from .metrics import compute_iou


def compute_seg_score(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """
    Compute CTC SEG score for a single frame.

    SEG measures segmentation accuracy using a weighted Jaccard index.
    For each GT object, we find the pred object with highest IoU > threshold.

    Args:
        pred_mask: Predicted instance mask
        gt_mask: Ground truth instance mask
        threshold: IoU threshold for matching

    Returns:
        SEG score in [0, 1]
    """
    gt_ids = [x for x in np.unique(gt_mask) if x > 0]
    pred_ids = [x for x in np.unique(pred_mask) if x > 0]

    if not gt_ids:
        return 1.0 if not pred_ids else 0.0

    # For each GT object, find best matching prediction
    weighted_sum = 0.0
    total_weight = 0.0

    for gt_id in gt_ids:
        gt_binary = gt_mask == gt_id
        gt_size = gt_binary.sum()

        best_iou = 0.0
        for pred_id in pred_ids:
            pred_binary = pred_mask == pred_id
            iou = compute_iou(pred_binary, gt_binary)
            best_iou = max(best_iou, iou)

        if best_iou >= threshold:
            weighted_sum += best_iou * gt_size
        total_weight += gt_size

    if total_weight == 0:
        return 1.0

    return float(weighted_sum / total_weight)


def compute_tra_score_simple(
    pred_tracks: Dict[int, Dict[int, int]],
    gt_tracks: Dict[int, Dict[int, int]],
) -> float:
    """
    Simplified TRA score based on tracking consistency.

    This is a simplified version - full TRA requires AOGM graph matching.

    Args:
        pred_tracks: Dict mapping pred_id -> {frame_idx: assigned_gt_id}
        gt_tracks: Dict mapping gt_id -> {frame_idx: 1}

    Returns:
        Simplified tracking accuracy in [0, 1]
    """
    if not gt_tracks:
        return 1.0 if not pred_tracks else 0.0

    # Count consistent track assignments
    consistent = 0
    total = 0

    for pred_id, assignments in pred_tracks.items():
        if not assignments:
            continue

        # Get most common GT assignment
        gt_counts: Dict[int, int] = {}
        for frame_idx, gt_id in assignments.items():
            gt_counts[gt_id] = gt_counts.get(gt_id, 0) + 1

        if gt_counts:
            most_common_count = max(gt_counts.values())
            consistent += most_common_count
            total += len(assignments)

    if total == 0:
        return 0.0

    return float(consistent / total)


def evaluate_ctc_sequence(
    pred_masks_dir: str,
    gt_masks_dir: str,
) -> Dict[str, float]:
    """
    Evaluate a CTC sequence.

    Args:
        pred_masks_dir: Directory with predicted masks
        gt_masks_dir: Directory with GT masks (man_seg*.tif or man_track*.tif)

    Returns:
        Dict with SEG and simplified TRA scores
    """
    import cv2

    # Load prediction masks
    pred_files = sorted([
        f for f in os.listdir(pred_masks_dir)
        if f.endswith((".png", ".tif", ".tiff"))
    ])

    seg_scores = []

    for pred_file in pred_files:
        # Extract frame index from filename
        # Try to match both our format (000000.png) and CTC format (mask000.tif)
        import re
        match = re.search(r"(\d+)", pred_file)
        if not match:
            continue
        frame_idx = int(match.group(1))

        # Load prediction
        pred_path = os.path.join(pred_masks_dir, pred_file)
        pred = cv2.imread(pred_path, cv2.IMREAD_UNCHANGED)
        if pred is None:
            continue

        # Try to find corresponding GT
        gt_path = None
        for gt_pattern in [f"man_seg{frame_idx:03d}.tif", f"man_track{frame_idx:03d}.tif"]:
            candidate = os.path.join(gt_masks_dir, gt_pattern)
            if os.path.exists(candidate):
                gt_path = candidate
                break

        if gt_path is None:
            continue

        gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
        if gt is None:
            continue

        # Compute SEG score for this frame
        score = compute_seg_score(pred, gt)
        seg_scores.append(score)

    mean_seg = float(np.mean(seg_scores)) if seg_scores else 0.0

    return {
        "SEG": mean_seg,
        "num_frames": len(seg_scores),
    }


def print_ctc_results(results: Dict[str, Dict[str, float]]) -> None:
    """Print CTC evaluation results."""
    print("\n" + "=" * 40)
    print("Cell Tracking Challenge Evaluation")
    print("=" * 40)
    print(f"{'Sequence':<15} {'SEG':>8} {'Frames':>8}")
    print("-" * 40)

    for seq, metrics in sorted(results.items()):
        if seq == "mean":
            continue
        print(f"{seq:<15} {metrics['SEG']:>8.3f} {metrics['num_frames']:>8}")

    print("-" * 40)
    if "mean" in results:
        m = results["mean"]
        print(f"{'Mean':<15} {m['SEG']:>8.3f}")
    print("=" * 40)

