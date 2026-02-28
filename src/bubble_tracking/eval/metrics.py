"""
Core evaluation metrics for segmentation and tracking.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np


def compute_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Compute Intersection over Union (Jaccard index).

    Args:
        pred: Predicted binary mask
        gt: Ground truth binary mask

    Returns:
        IoU score in [0, 1]
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    return float(intersection / union)


def compute_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Compute Dice coefficient (F1 score).

    Args:
        pred: Predicted binary mask
        gt: Ground truth binary mask

    Returns:
        Dice score in [0, 1]
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    intersection = np.logical_and(pred, gt).sum()
    pred_sum = pred.sum()
    gt_sum = gt.sum()

    if pred_sum + gt_sum == 0:
        return 1.0 if intersection == 0 else 0.0

    return float(2 * intersection / (pred_sum + gt_sum))


def compute_boundary_f(
    pred: np.ndarray,
    gt: np.ndarray,
    threshold: float = 2.0,
) -> float:
    """
    Compute boundary F-measure.

    Measures how well predicted boundaries match ground truth boundaries.

    Args:
        pred: Predicted binary mask
        gt: Ground truth binary mask
        threshold: Distance threshold for boundary matching (pixels)

    Returns:
        Boundary F-measure in [0, 1]
    """
    import cv2

    pred = pred.astype(np.uint8)
    gt = gt.astype(np.uint8)

    # Extract boundaries using morphological operations
    kernel = np.ones((3, 3), np.uint8)
    pred_boundary = pred - cv2.erode(pred, kernel)
    gt_boundary = gt - cv2.erode(gt, kernel)

    # Get boundary points
    pred_points = np.argwhere(pred_boundary > 0)
    gt_points = np.argwhere(gt_boundary > 0)

    if len(pred_points) == 0 and len(gt_points) == 0:
        return 1.0
    if len(pred_points) == 0 or len(gt_points) == 0:
        return 0.0

    # Compute distance transforms
    pred_dist = cv2.distanceTransform(1 - pred_boundary, cv2.DIST_L2, 3)
    gt_dist = cv2.distanceTransform(1 - gt_boundary, cv2.DIST_L2, 3)

    # Precision: fraction of pred boundary within threshold of gt
    precision = np.mean(gt_dist[pred_boundary > 0] <= threshold)

    # Recall: fraction of gt boundary within threshold of pred
    recall = np.mean(pred_dist[gt_boundary > 0] <= threshold)

    if precision + recall == 0:
        return 0.0

    return float(2 * precision * recall / (precision + recall))


def compute_instance_metrics(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
) -> Dict[str, float]:
    """
    Compute per-instance IoU metrics.

    Args:
        pred_mask: Instance segmentation mask (IDs > 0)
        gt_mask: Ground truth instance mask

    Returns:
        Dict with per-object and mean metrics
    """
    gt_ids = [x for x in np.unique(gt_mask) if x > 0]
    pred_ids = [x for x in np.unique(pred_mask) if x > 0]

    if not gt_ids:
        return {"mean_iou": 1.0 if not pred_ids else 0.0, "per_object": {}}

    # Match predictions to ground truth by IoU
    ious = {}
    for gt_id in gt_ids:
        gt_binary = gt_mask == gt_id
        best_iou = 0.0
        for pred_id in pred_ids:
            pred_binary = pred_mask == pred_id
            iou = compute_iou(pred_binary, gt_binary)
            best_iou = max(best_iou, iou)
        ious[gt_id] = best_iou

    mean_iou = float(np.mean(list(ious.values()))) if ious else 0.0

    return {
        "mean_iou": mean_iou,
        "per_object": ious,
    }


def compute_sequence_jf(
    pred_masks: List[np.ndarray],
    gt_masks: List[np.ndarray],
) -> Dict[str, float]:
    """
    Compute J&F metrics for a sequence (DAVIS-style).

    Args:
        pred_masks: List of predicted instance masks
        gt_masks: List of ground truth instance masks

    Returns:
        Dict with J (Jaccard), F (boundary), and J&F (mean)
    """
    if len(pred_masks) != len(gt_masks):
        raise ValueError("Mismatched number of frames")

    all_j = []
    all_f = []

    for pred, gt in zip(pred_masks, gt_masks):
        # Get object IDs from ground truth
        gt_ids = [x for x in np.unique(gt) if x > 0]

        for gt_id in gt_ids:
            gt_binary = (gt == gt_id).astype(np.uint8)

            # Find best matching prediction
            pred_ids = [x for x in np.unique(pred) if x > 0]
            best_j = 0.0
            best_pred_binary = np.zeros_like(gt_binary)

            for pred_id in pred_ids:
                pred_binary = (pred == pred_id).astype(np.uint8)
                j = compute_iou(pred_binary, gt_binary)
                if j > best_j:
                    best_j = j
                    best_pred_binary = pred_binary

            f = compute_boundary_f(best_pred_binary, gt_binary)

            all_j.append(best_j)
            all_f.append(f)

    mean_j = float(np.mean(all_j)) if all_j else 0.0
    mean_f = float(np.mean(all_f)) if all_f else 0.0
    mean_jf = (mean_j + mean_f) / 2

    return {
        "J": mean_j,
        "F": mean_f,
        "J&F": mean_jf,
    }

