"""
Hybrid keyframe matcher for re-prompting: IoU (Hungarian) + centroid fallback.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class MatchResult:
    matches: Dict[int, int]  # curr_idx -> prev_id
    births: List[int]        # curr_idx list
    deaths: List[int]        # prev_id list
    splits: Dict[int, List[int]]  # prev_id -> list of curr_idx (new IDs)


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    if inter == 0:
        return 0.0
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union > 0 else 0.0


def _centroid(mask: np.ndarray) -> Tuple[float, float]:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return (0.0, 0.0)
    return (float(xs.mean()), float(ys.mean()))


def _hungarian(cost: np.ndarray) -> List[int]:
    """
    Hungarian algorithm for square cost matrix (minimization).
    Returns assignment list where row i -> col assignment[i], or -1 if unassigned.
    """
    n, m = cost.shape
    size = max(n, m)
    max_cost = cost.max() if cost.size else 0.0
    pad = np.full((size, size), max_cost + 1.0, dtype=float)
    pad[:n, :m] = cost

    u = np.zeros(size + 1)
    v = np.zeros(size + 1)
    p = np.zeros(size + 1, dtype=int)
    way = np.zeros(size + 1, dtype=int)

    for i in range(1, size + 1):
        p[0] = i
        j0 = 0
        minv = np.full(size + 1, np.inf)
        used = np.zeros(size + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = np.inf
            j1 = 0
            for j in range(1, size + 1):
                if used[j]:
                    continue
                cur = pad[i0 - 1, j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(size + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    assignment = [-1] * n
    for j in range(1, size + 1):
        if p[j] <= n and j <= m:
            assignment[p[j] - 1] = j - 1
    return assignment


def match_keyframes(
    prev_ids: List[int],
    prev_masks: List[np.ndarray],
    curr_masks: List[np.ndarray],
    iou_threshold: float = 0.2,
    dist_threshold: float = 50.0,
) -> MatchResult:
    """
    Match current keyframe masks to previous IDs.
    Returns mapping of current mask indices to previous IDs, births/deaths/splits.
    """
    matches: Dict[int, int] = {}
    births: List[int] = []
    deaths: List[int] = []
    splits: Dict[int, List[int]] = {}

    if not prev_masks:
        births = list(range(len(curr_masks)))
        return MatchResult(matches=matches, births=births, deaths=deaths, splits=splits)

    if not curr_masks:
        deaths = prev_ids[:]
        return MatchResult(matches=matches, births=births, deaths=deaths, splits=splits)

    # IoU matrix
    iou = np.zeros((len(prev_masks), len(curr_masks)), dtype=float)
    for i, pm in enumerate(prev_masks):
        for j, cm in enumerate(curr_masks):
            iou[i, j] = _mask_iou(pm, cm)

    cost = 1.0 - iou
    assignment = _hungarian(cost)
    prev_matched = set()
    curr_matched = set()
    for i, j in enumerate(assignment):
        if j == -1:
            continue
        if iou[i, j] >= iou_threshold:
            prev_id = prev_ids[i]
            matches[j] = prev_id
            prev_matched.add(i)
            curr_matched.add(j)

    # centroid fallback for unmatched
    prev_centroids = [_centroid(m) for m in prev_masks]
    curr_centroids = [_centroid(m) for m in curr_masks]

    # build distance cost for unmatched
    unmatched_prev = [i for i in range(len(prev_masks)) if i not in prev_matched]
    unmatched_curr = [j for j in range(len(curr_masks)) if j not in curr_matched]
    if unmatched_prev and unmatched_curr:
        dist_cost = np.zeros((len(unmatched_prev), len(unmatched_curr)), dtype=float)
        for ii, i in enumerate(unmatched_prev):
            for jj, j in enumerate(unmatched_curr):
                dx = prev_centroids[i][0] - curr_centroids[j][0]
                dy = prev_centroids[i][1] - curr_centroids[j][1]
                dist_cost[ii, jj] = np.hypot(dx, dy)
        dist_assign = _hungarian(dist_cost)
        for ii, jj in enumerate(dist_assign):
            if jj == -1:
                continue
            if dist_cost[ii, jj] <= dist_threshold:
                i = unmatched_prev[ii]
                j = unmatched_curr[jj]
                prev_id = prev_ids[i]
                matches[j] = prev_id
                prev_matched.add(i)
                curr_matched.add(j)

    # splits: any remaining current masks that still overlap a prev mask
    for j in range(len(curr_masks)):
        if j in curr_matched:
            continue
        overlaps = iou[:, j]
        if overlaps.size == 0:
            births.append(j)
            continue
        i = int(overlaps.argmax())
        if overlaps[i] >= iou_threshold:
            prev_id = prev_ids[i]
            splits.setdefault(prev_id, []).append(j)
        else:
            births.append(j)

    # deaths
    for i, prev_id in enumerate(prev_ids):
        if i not in prev_matched:
            deaths.append(prev_id)

    return MatchResult(matches=matches, births=births, deaths=deaths, splits=splits)
