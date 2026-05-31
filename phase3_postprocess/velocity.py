"""Phase 3 velocity module.

Provides three public writer functions:
  - write_velocity_overlay_video  (from scripts/export_velocity_overlay.py)
  - write_velocity_metrics_csv    (from scripts/plot_velocity_norm_summary.py)
  - write_max_velocity_summary    (from scripts/export_bubble_max_velocity_summary.py)

All functions accept a ``run_dir`` that is the per-case run directory.
They auto-detect whether tracks.csv lives at ``<run_dir>/phase2/tracks.csv``
(new 3-phase layout) or ``<run_dir>/tracks.csv`` (legacy/oracle layout).
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb
from scipy import stats


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------


def _resolve_tracks_dir(run_dir: Path) -> Path:
    """Return the dir containing tracks.csv.

    Tries ``<run_dir>/phase2`` first, then ``<run_dir>``.
    """
    if (run_dir / "phase2" / "tracks.csv").exists():
        return run_dir / "phase2"
    if (run_dir / "tracks.csv").exists():
        return run_dir
    raise FileNotFoundError(
        f"No tracks.csv under {run_dir} (tried phase2/ and root)"
    )


# ---------------------------------------------------------------------------
# Inlined frame-IO helpers (from src/bubble_tracking/frame_io.py)
# ---------------------------------------------------------------------------


def _sorted_frame_paths(frames_dir: str) -> List[str]:
    """Get sorted list of frame image paths from a directory."""
    pattern = re.compile(r"(\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)
    paths: List[Tuple[int, str]] = []
    for fname in os.listdir(frames_dir):
        match = pattern.search(fname)
        if match:
            paths.append((int(match.group(1)), os.path.join(frames_dir, fname)))
    paths.sort(key=lambda x: x[0])
    return [p[1] for p in paths]


def _load_mask(mask_path: str) -> np.ndarray:
    """Load a segmentation mask from file (16-bit PNG)."""
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise ValueError(f"Cannot read mask: {mask_path}")
    return mask


# ---------------------------------------------------------------------------
# Inlined visualization helpers (from src/bubble_tracking/visualization.py)
# ---------------------------------------------------------------------------


def _generate_colors(n: int, seed: int = 42) -> List[Tuple[int, int, int]]:
    """Generate *n* distinct BGR colors."""
    np.random.seed(seed)
    colors: List[Tuple[int, int, int]] = []
    for i in range(n):
        h = i / max(n, 1)
        s = 0.7 + np.random.random() * 0.3
        v = 0.7 + np.random.random() * 0.3
        rgb = hsv_to_rgb([h, s, v])
        colors.append(tuple(int(c * 255) for c in rgb))
    return colors


def _draw_mask_overlay(
    frame: np.ndarray,
    mask: np.ndarray,
    colors: Optional[Dict[int, Tuple[int, int, int]]] = None,
    alpha: float = 0.5,
) -> np.ndarray:
    """Draw a colored instance-mask overlay on *frame* (BGR)."""
    result = frame.copy()
    unique_ids = np.unique(mask)
    unique_ids = unique_ids[unique_ids > 0]

    if colors is None:
        color_list = _generate_colors(len(unique_ids))
        colors = {int(uid): color_list[i] for i, uid in enumerate(unique_ids)}

    for obj_id in unique_ids:
        obj_mask = mask == obj_id
        color = colors.get(int(obj_id), (255, 255, 255))
        overlay = result.copy()
        overlay[obj_mask] = color
        result = cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0)
        contours, _ = cv2.findContours(
            obj_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(result, contours, -1, color, 2)

    return result


# ---------------------------------------------------------------------------
# Shared CSV helpers
# ---------------------------------------------------------------------------


def _to_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = value.strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Public function 1: velocity overlay video
# (from scripts/export_velocity_overlay.py)
# ---------------------------------------------------------------------------


def _read_tracks_by_frame(
    tracks_csv: Path,
) -> Tuple[Dict[int, List[dict]], List[int]]:
    by_frame: Dict[int, List[dict]] = {}
    object_ids: set[int] = set()
    with tracks_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                frame_idx = int(row["frame_idx"])
                obj_id = int(row["object_id"])
            except Exception:
                continue

            def _f(key: str, _row: dict = row) -> float | None:
                val = _row.get(key, "")
                if val is None or val == "":
                    return None
                try:
                    return float(val)
                except ValueError:
                    return None

            entry = {
                "object_id": obj_id,
                "centroid_x": _f("centroid_x"),
                "centroid_y": _f("centroid_y"),
                "vx_px_s": _f("velocity_x_px_s"),
                "vy_px_s": _f("velocity_y_px_s"),
                "vmag_px_s": _f("velocity_mag_px_s"),
                "vx_mm_s": _f("velocity_x_mm_s"),
                "vy_mm_s": _f("velocity_y_mm_s"),
                "vmag_mm_s": _f("velocity_mag_mm_s"),
            }
            by_frame.setdefault(frame_idx, []).append(entry)
            object_ids.add(obj_id)
    return by_frame, sorted(object_ids)


def write_velocity_overlay_video(
    *,
    run_dir: Path,
    frames_dir: Path,
    output: Path,
    fps: float,
    label_units: str = "mm_s",
    alpha: float = 0.35,
    arrow_seconds: float = 0.25,
    line_thickness: int = 2,
    tip_length: float = 0.3,
    overlay_dir: Optional[Path] = None,
    start_frame: int = 0,
    max_frames: int = 0,
) -> None:
    """Write a velocity-vector overlay video to *output*.

    Parameters
    ----------
    run_dir:
        Per-case run directory.  ``tracks.csv`` and ``masks/`` are looked up
        via :func:`_resolve_tracks_dir`.
    frames_dir:
        Directory containing the original video frames.
    output:
        Destination ``.mp4`` path.
    fps:
        Output video frame rate.
    label_units:
        ``"mm_s"`` (default) or ``"px_s"`` for the per-bubble speed label.
    """
    tracks_dir = _resolve_tracks_dir(run_dir)
    tracks_csv = tracks_dir / "tracks.csv"
    # masks/ lives next to tracks.csv in either layout
    masks_dir = tracks_dir / "masks"
    if not masks_dir.exists():
        # fall back to run_dir root
        masks_dir = run_dir / "masks"

    if not tracks_csv.exists():
        raise FileNotFoundError(f"tracks.csv not found: {tracks_csv}")
    if not masks_dir.exists():
        raise FileNotFoundError(f"masks dir not found: {masks_dir}")

    frame_paths = _sorted_frame_paths(str(frames_dir))
    if not frame_paths:
        raise ValueError(f"No frames found in {frames_dir}")

    by_frame, object_ids = _read_tracks_by_frame(tracks_csv)
    color_list = _generate_colors(max(1, len(object_ids)))
    colors = {obj_id: color_list[i % len(color_list)] for i, obj_id in enumerate(object_ids)}

    output.parent.mkdir(parents=True, exist_ok=True)
    if overlay_dir is not None:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    first = cv2.imread(frame_paths[0])
    if first is None:
        raise ValueError(f"Failed to read frame: {frame_paths[0]}")
    h, w = first.shape[:2]
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {output}")

    start = max(0, start_frame)
    end = len(frame_paths)
    if max_frames > 0:
        end = min(end, start + max_frames)

    for frame_idx in range(start, end):
        frame = cv2.imread(frame_paths[frame_idx])
        if frame is None:
            continue

        mask_path = masks_dir / f"{frame_idx:06d}.png"
        if mask_path.exists():
            mask = _load_mask(str(mask_path))
            frame = _draw_mask_overlay(frame, mask, colors=colors, alpha=alpha)

        for item in by_frame.get(frame_idx, []):
            cx = item["centroid_x"]
            cy = item["centroid_y"]
            vx_px_s = item["vx_px_s"]
            vy_px_s = item["vy_px_s"]
            if cx is None or cy is None or vx_px_s is None or vy_px_s is None:
                continue

            obj_id = int(item["object_id"])
            color = colors.get(obj_id, (255, 255, 255))
            p0 = (int(round(cx)), int(round(cy)))
            dx = vx_px_s * arrow_seconds
            dy = vy_px_s * arrow_seconds
            p1 = (int(round(cx + dx)), int(round(cy + dy)))

            cv2.arrowedLine(
                frame,
                p0,
                p1,
                color=color,
                thickness=max(1, line_thickness),
                tipLength=tip_length,
            )
            cv2.circle(frame, p0, 3, color, -1)

            if label_units == "mm_s":
                vmag = item["vmag_mm_s"]
                units = "mm/s"
            else:
                vmag = item["vmag_px_s"]
                units = "px/s"
            if vmag is not None:
                label = f"#{obj_id} {vmag:.2f} {units}"
                cv2.putText(
                    frame,
                    label,
                    (p0[0] + 6, p0[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    cv2.LINE_AA,
                )

        cv2.putText(
            frame,
            f"Frame {frame_idx}",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        writer.write(frame)
        if overlay_dir is not None:
            cv2.imwrite(str(overlay_dir / f"{frame_idx:06d}.jpg"), frame)

    writer.release()
    print(f"Wrote velocity overlay video: {output}")


# ---------------------------------------------------------------------------
# Public function 2: velocity metrics CSV + plots
# (from scripts/plot_velocity_norm_summary.py)
# ---------------------------------------------------------------------------


def _load_run_config_optional(run_dir: Path) -> dict:
    """Load run config; tries new run_config.json first, then legacy sam2_prompted_flow_config.json."""
    for name in ("run_config.json", "sam2_prompted_flow_config.json"):
        p = run_dir / name
        if p.exists():
            try:
                return json.loads(p.read_text())
            except Exception:
                continue
    return {}


def _resolve_roi_area(
    run_dir: Path, config: dict, explicit_roi_area: float | None
) -> float:
    if explicit_roi_area is not None and explicit_roi_area > 0:
        return float(explicit_roi_area)

    roi_mask_path = None
    roi_json_path = None
    if isinstance(config, dict):
        roi_mask_path = config.get("roi_mask")
        roi_json_path = config.get("roi_json")

    def _resolve_run_relative(raw: str) -> Path:
        # Run config paths can be:
        # - absolute (use as-is)
        # - project-relative (e.g. "outputs/case1/phase1/roi.json" — already includes run_dir prefix)
        # - run-dir-relative (e.g. "phase1/roi.json")
        # Try as-is first; if that exists, use it. Otherwise, join with run_dir.
        p = Path(raw)
        if p.is_absolute() or p.exists():
            return p
        return (run_dir / p).resolve()

    if roi_mask_path:
        p = _resolve_run_relative(roi_mask_path)
        if p.exists():
            mask = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                area = int((mask > 0).sum())
                if area > 0:
                    return float(area)

    if roi_json_path:
        p = _resolve_run_relative(roi_json_path)
        if p.exists():
            data = json.loads(p.read_text())
            if all(k in data for k in ("x1", "y1", "x2", "y2")):
                area = max(0, int(data["x2"]) - int(data["x1"])) * max(
                    0, int(data["y2"]) - int(data["y1"])
                )
                if area > 0:
                    return float(area)

    raise ValueError(
        "Failed to resolve ROI area. Provide explicit roi_area or ensure run config"
        " has valid roi_mask/roi_json."
    )


def _load_tracks_rows(tracks_csv: Path) -> List[dict]:
    rows: List[dict] = []
    with tracks_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _extract_velocity_vectors(
    rows: List[dict],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    vx_list: List[float] = []
    vy_list: List[float] = []
    vmag_list: List[float] = []
    obj_list: List[int] = []

    for row in rows:
        vx = _to_float(row.get("velocity_x_mm_s", ""))
        vy = _to_float(row.get("velocity_y_mm_s", ""))
        vmag = _to_float(row.get("velocity_mag_mm_s", ""))
        if vx is None or vy is None or vmag is None:
            continue
        obj_id = int(row["object_id"])
        vx_list.append(vx)
        vy_list.append(vy)
        vmag_list.append(vmag)
        obj_list.append(obj_id)

    return (
        np.asarray(vx_list, dtype=np.float64),
        np.asarray(vy_list, dtype=np.float64),
        np.asarray(vmag_list, dtype=np.float64),
        np.asarray(obj_list, dtype=np.int64),
    )


def _compute_object_area_stats(rows: List[dict], area_stat: str) -> Dict[int, float]:
    bucket: Dict[int, List[float]] = {}
    for row in rows:
        area = _to_float(row.get("area_px", ""))
        if area is None:
            continue
        obj_id = int(row["object_id"])
        bucket.setdefault(obj_id, []).append(area)

    result: Dict[int, float] = {}
    for obj_id, vals in bucket.items():
        arr = np.asarray(vals, dtype=np.float64)
        if area_stat == "max":
            result[obj_id] = float(np.max(arr))
        elif area_stat == "mean":
            result[obj_id] = float(np.mean(arr))
        elif area_stat == "median":
            result[obj_id] = float(np.median(arr))
        else:
            raise ValueError(f"Unsupported area_stat: {area_stat}")
    return result


def _plot_histograms(
    vx_mm_s: np.ndarray,
    vy_mm_s: np.ndarray,
    v_ref_1: float,
    v_ref_2: float,
    bins,
    clip_low_pct: float,
    clip_high_pct: float,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=150)

    plots = [
        (vx_mm_s / v_ref_1, f"v_x / v_ref_1 (v_ref_1={v_ref_1:g} mm/s)", axes[0, 0]),
        (vy_mm_s / v_ref_1, f"v_y / v_ref_1 (v_ref_1={v_ref_1:g} mm/s)", axes[0, 1]),
        (vx_mm_s / v_ref_2, f"v_x / v_ref_2 (v_ref_2={v_ref_2:g} mm/s)", axes[1, 0]),
        (vy_mm_s / v_ref_2, f"v_y / v_ref_2 (v_ref_2={v_ref_2:g} mm/s)", axes[1, 1]),
    ]

    for values, title, ax in plots:
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            ax.set_title(title + " (no data)")
            ax.grid(True, alpha=0.25)
            continue
        lo = float(np.percentile(finite, clip_low_pct))
        hi = float(np.percentile(finite, clip_high_pct))
        if hi <= lo:
            lo, hi = float(np.min(finite)), float(np.max(finite))
            if hi <= lo:
                hi = lo + 1.0
        in_range = (finite >= lo) & (finite <= hi)
        clipped = finite[in_range]
        ax.hist(
            clipped,
            bins=bins,
            range=(lo, hi),
            color="#1f77b4",
            alpha=0.8,
            edgecolor="white",
            linewidth=0.3,
        )
        ax.axvline(0.0, color="black", linewidth=1.0, alpha=0.6)
        ax.set_title(title)
        ax.set_xlabel("Normalized velocity")
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.25)
        dropped = int(finite.size - clipped.size)
        if dropped > 0:
            ax.text(
                0.99,
                0.95,
                f"outside range: {dropped}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                color="#444444",
            )

    fig.suptitle("Velocity Component Histograms (Normalized by Reference Velocity)")
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def _fit_distributions(
    data: np.ndarray,
    max_samples: int = 20000,
) -> List[dict]:
    if data.size < 8:
        return []

    sample = data
    if data.size > max_samples:
        rng = np.random.default_rng(42)
        sample = rng.choice(data, size=max_samples, replace=False)

    candidates = [
        ("norm", stats.norm),
        ("laplace", stats.laplace),
        ("logistic", stats.logistic),
        ("cauchy", stats.cauchy),
        ("t", stats.t),
    ]

    results: List[dict] = []
    for name, dist in candidates:
        try:
            params = dist.fit(sample)
            logpdf = dist.logpdf(sample, *params)
            if not np.all(np.isfinite(logpdf)):
                continue
            log_lik = float(np.sum(logpdf))
            k = int(len(params))
            aic = float(2 * k - 2 * log_lik)
            bic = float(k * math.log(sample.size) - 2 * log_lik)
            ks_stat, ks_p = stats.kstest(sample, name, args=params)
            results.append(
                {
                    "distribution": name,
                    "params": tuple(float(x) for x in params),
                    "aic": aic,
                    "bic": bic,
                    "ks_stat": float(ks_stat),
                    "ks_pvalue": float(ks_p),
                    "n_fit": int(sample.size),
                }
            )
        except Exception:
            continue

    results.sort(key=lambda r: r["aic"])
    return results


def _plot_pdf_with_fit(
    vx_mm_s: np.ndarray,
    vy_mm_s: np.ndarray,
    v_ref_1: float,
    v_ref_2: float,
    clip_low_pct: float,
    clip_high_pct: float,
    fit_max_samples: int,
    output_path: Path,
) -> List[dict]:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=150)

    plots = [
        ("vx_over_vref1", vx_mm_s / v_ref_1, f"v_x / v_ref_1 (v_ref_1={v_ref_1:g} mm/s)", axes[0, 0]),
        ("vy_over_vref1", vy_mm_s / v_ref_1, f"v_y / v_ref_1 (v_ref_1={v_ref_1:g} mm/s)", axes[0, 1]),
        ("vx_over_vref2", vx_mm_s / v_ref_2, f"v_x / v_ref_2 (v_ref_2={v_ref_2:g} mm/s)", axes[1, 0]),
        ("vy_over_vref2", vy_mm_s / v_ref_2, f"v_y / v_ref_2 (v_ref_2={v_ref_2:g} mm/s)", axes[1, 1]),
    ]

    all_fit_results: List[dict] = []
    for key, values, title, ax in plots:
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            ax.set_title(title + " (no data)")
            ax.grid(True, alpha=0.25)
            continue

        lo = float(np.percentile(finite, clip_low_pct))
        hi = float(np.percentile(finite, clip_high_pct))
        if hi <= lo:
            lo, hi = float(np.min(finite)), float(np.max(finite))
            if hi <= lo:
                hi = lo + 1.0

        in_range = (finite >= lo) & (finite <= hi)
        clipped = finite[in_range]
        ax.hist(
            clipped,
            bins="auto",
            range=(lo, hi),
            density=True,
            color="#9ecae1",
            alpha=0.65,
            edgecolor="white",
            linewidth=0.3,
            label="empirical density",
        )

        x_grid = np.linspace(lo, hi, 500)
        if clipped.size >= 8:
            try:
                kde = stats.gaussian_kde(clipped)
                ax.plot(x_grid, kde(x_grid), color="#2ca02c", linewidth=1.2, label="KDE")
            except Exception:
                pass

            fit_results = _fit_distributions(clipped, max_samples=fit_max_samples)
            for rank, row in enumerate(fit_results[:3], start=1):
                row_out = dict(row)
                row_out["variable"] = key
                row_out["rank"] = rank
                all_fit_results.append(row_out)

            if fit_results:
                best = fit_results[0]
                dist = getattr(stats, best["distribution"])
                y_pdf = dist.pdf(x_grid, *best["params"])
                ax.plot(
                    x_grid,
                    y_pdf,
                    color="#d62728",
                    linewidth=1.5,
                    label=f"best fit: {best['distribution']} (AIC={best['aic']:.1f})",
                )

        ax.axvline(0.0, color="black", linewidth=1.0, alpha=0.6)
        ax.set_title(title)
        ax.set_xlabel("Normalized velocity")
        ax.set_ylabel("Density")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", fontsize=8)

    fig.suptitle("Velocity PDFs: Empirical Density + Best-Fit Distribution")
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return all_fit_results


def _plot_scatter(
    obj_max_v_mm_s: Dict[int, float],
    obj_area_px: Dict[int, float],
    roi_area_px: float,
    v_ref_1: float,
    v_ref_2: float,
    output_path: Path,
    area_stat_label: str,
    log_x: bool = False,
    log_y: bool = False,
) -> None:
    common_ids = sorted(set(obj_max_v_mm_s.keys()) & set(obj_area_px.keys()))
    if not common_ids:
        raise ValueError("No overlapping object IDs between velocity and area statistics.")

    x1 = np.array([obj_max_v_mm_s[i] / v_ref_1 for i in common_ids], dtype=np.float64)
    x2 = np.array([obj_max_v_mm_s[i] / v_ref_2 for i in common_ids], dtype=np.float64)
    y = np.array([obj_area_px[i] / roi_area_px for i in common_ids], dtype=np.float64)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150, sharey=True)

    def _scatter_one(ax, xvals, yvals, title, color):
        valid = np.isfinite(xvals) & np.isfinite(yvals)
        if log_x:
            valid &= xvals > 0
        if log_y:
            valid &= yvals > 0
        xv = xvals[valid]
        yv = yvals[valid]

        if xv.size > 0:
            ax.scatter(xv, yv, s=10, alpha=0.7, color=color, label="bubbles")
        else:
            ax.text(
                0.5, 0.5, "No valid points for log scale",
                ha="center", va="center", transform=ax.transAxes,
            )

        if xv.size >= 2:
            x_fit = xv
            y_fit = yv
            if log_x:
                x_fit = np.log10(x_fit)
            if log_y:
                y_fit = np.log10(y_fit)

            if np.unique(x_fit).size >= 2:
                coeff = np.polyfit(x_fit, y_fit, deg=1)
                slope = float(coeff[0])
                intercept = float(coeff[1])
                y_pred = slope * x_fit + intercept
                ss_res = float(np.sum((y_fit - y_pred) ** 2))
                ss_tot = float(np.sum((y_fit - np.mean(y_fit)) ** 2))
                r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

                if log_x:
                    x_line = np.logspace(np.log10(np.min(xv)), np.log10(np.max(xv)), 256)
                else:
                    x_line = np.linspace(np.min(xv), np.max(xv), 256)
                y_line_fit = slope * (np.log10(x_line) if log_x else x_line) + intercept
                y_line = np.power(10.0, y_line_fit) if log_y else y_line_fit

                ax.plot(
                    x_line,
                    y_line,
                    color="black",
                    linewidth=1.3,
                    alpha=0.9,
                    label=f"best fit (R²={r2:.2f})",
                )

        if log_x:
            ax.set_xscale("log")
        if log_y:
            ax.set_yscale("log")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)

        dropped = int(len(xvals) - len(xv))
        if dropped > 0:
            ax.text(
                0.99, 0.95, f"dropped: {dropped}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=8, color="#444444",
            )

    _scatter_one(axes[0], x1, y, f"max(V / v_ref_1), v_ref_1={v_ref_1:g} mm/s", "#d62728")
    axes[0].set_xlabel("max(V / v_ref_1) over bubble lifecycle")
    axes[0].set_ylabel(f"{area_stat_label}(area_per_bubble) / roi_area")

    _scatter_one(axes[1], x2, y, f"max(V / v_ref_2), v_ref_2={v_ref_2:g} mm/s", "#2ca02c")
    axes[1].set_xlabel("max(V / v_ref_2) over bubble lifecycle")

    scale_label = "linear"
    if log_x and log_y:
        scale_label = "log-x/log-y"
    elif log_x:
        scale_label = "log-x"
    elif log_y:
        scale_label = "log-y"
    fig.suptitle(
        f"Bubble-level Scatter: Peak Normalized Speed vs Normalized Bubble Area ({scale_label})"
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def _write_object_metrics(
    output_csv: Path,
    obj_max_v_mm_s: Dict[int, float],
    obj_area_px: Dict[int, float],
    roi_area_px: float,
    v_ref_1: float,
    v_ref_2: float,
) -> None:
    common_ids = sorted(set(obj_max_v_mm_s.keys()) & set(obj_area_px.keys()))
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "object_id",
                "max_v_mm_s",
                "max_v_over_v_ref_1",
                "max_v_over_v_ref_2",
                "area_stat_px",
                "area_over_roi",
            ],
        )
        writer.writeheader()
        for obj_id in common_ids:
            max_v = obj_max_v_mm_s[obj_id]
            area_px = obj_area_px[obj_id]
            writer.writerow(
                {
                    "object_id": obj_id,
                    "max_v_mm_s": f"{max_v:.6f}",
                    "max_v_over_v_ref_1": f"{(max_v / v_ref_1):.6f}",
                    "max_v_over_v_ref_2": f"{(max_v / v_ref_2):.6f}",
                    "area_stat_px": f"{area_px:.6f}",
                    "area_over_roi": f"{(area_px / roi_area_px):.6f}",
                }
            )


def write_velocity_metrics_csv(
    *,
    run_dir: Path,
    output_dir: Path,
    v_ref_1: float,
    v_ref_2: float,
    roi_area: Optional[float] = None,
    bins: str | int = "auto",
    hist_clip_low: float = 0.5,
    hist_clip_high: float = 99.5,
    fit_max_samples: int = 20000,
    area_stat: str = "max",
) -> None:
    """Write velocity/area metrics CSV and companion PNGs to *output_dir*.

    Outputs written:
      - ``bubble_velocity_area_metrics.csv``
      - ``velocity_norm_histograms.png``
      - ``velocity_norm_pdf_fit.png``
      - ``velocity_area_scatter.png`` (and log-x / log-y / log-xy variants)
      - ``velocity_distribution_fit.csv``
      - ``velocity_distribution_best.json``

    Parameters
    ----------
    run_dir:
        Per-case run directory (layout auto-detected via
        :func:`_resolve_tracks_dir`).
    output_dir:
        Directory to write all output files.
    v_ref_1, v_ref_2:
        Reference velocities in mm/s for normalisation.
    roi_area:
        ROI area in pixels.  If *None*, resolved from the run config.
    """
    tracks_dir = _resolve_tracks_dir(run_dir)
    tracks_csv = tracks_dir / "tracks.csv"

    output_dir.mkdir(parents=True, exist_ok=True)
    config = _load_run_config_optional(run_dir)
    roi_area_px = _resolve_roi_area(run_dir, config, roi_area)

    if not (0.0 <= hist_clip_low < hist_clip_high <= 100.0):
        raise ValueError("hist_clip_low/hist_clip_high must satisfy 0 <= low < high <= 100")

    rows = _load_tracks_rows(tracks_csv)
    vx_mm_s, vy_mm_s, vmag_mm_s, obj_ids = _extract_velocity_vectors(rows)
    if vx_mm_s.size == 0:
        raise ValueError("No valid velocity rows found in tracks.csv (mm/s columns are empty).")

    # Histograms
    hist_path = output_dir / "velocity_norm_histograms.png"
    _plot_histograms(
        vx_mm_s=vx_mm_s,
        vy_mm_s=vy_mm_s,
        v_ref_1=v_ref_1,
        v_ref_2=v_ref_2,
        bins=bins,
        clip_low_pct=hist_clip_low,
        clip_high_pct=hist_clip_high,
        output_path=hist_path,
    )

    # PDF + fit
    pdf_path = output_dir / "velocity_norm_pdf_fit.png"
    fit_rows = _plot_pdf_with_fit(
        vx_mm_s=vx_mm_s,
        vy_mm_s=vy_mm_s,
        v_ref_1=v_ref_1,
        v_ref_2=v_ref_2,
        clip_low_pct=hist_clip_low,
        clip_high_pct=hist_clip_high,
        fit_max_samples=fit_max_samples,
        output_path=pdf_path,
    )

    # Per-object max velocity
    obj_max_v_mm_s: Dict[int, float] = {}
    for obj_id, vmag in zip(obj_ids.tolist(), vmag_mm_s.tolist()):
        obj_max_v_mm_s[obj_id] = max(float(vmag), obj_max_v_mm_s.get(obj_id, float("-inf")))

    obj_area_px = _compute_object_area_stats(rows, area_stat)

    # Scatter plots (4 scale variants)
    for log_x, log_y, suffix in [
        (False, False, ""),
        (True, False, "_logx"),
        (False, True, "_logy"),
        (True, True, "_logxy"),
    ]:
        _plot_scatter(
            obj_max_v_mm_s=obj_max_v_mm_s,
            obj_area_px=obj_area_px,
            roi_area_px=roi_area_px,
            v_ref_1=v_ref_1,
            v_ref_2=v_ref_2,
            output_path=output_dir / f"velocity_area_scatter{suffix}.png",
            area_stat_label=area_stat,
            log_x=log_x,
            log_y=log_y,
        )

    # Metrics CSV
    metrics_csv = output_dir / "bubble_velocity_area_metrics.csv"
    _write_object_metrics(
        output_csv=metrics_csv,
        obj_max_v_mm_s=obj_max_v_mm_s,
        obj_area_px=obj_area_px,
        roi_area_px=roi_area_px,
        v_ref_1=v_ref_1,
        v_ref_2=v_ref_2,
    )

    # Distribution fit CSV + JSON
    fit_csv = output_dir / "velocity_distribution_fit.csv"
    with fit_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "variable", "rank", "distribution",
                "aic", "bic", "ks_stat", "ks_pvalue", "n_fit", "params",
            ],
        )
        writer.writeheader()
        for row in fit_rows:
            writer.writerow(
                {
                    "variable": row["variable"],
                    "rank": row["rank"],
                    "distribution": row["distribution"],
                    "aic": f"{row['aic']:.6f}",
                    "bic": f"{row['bic']:.6f}",
                    "ks_stat": f"{row['ks_stat']:.6f}",
                    "ks_pvalue": f"{row['ks_pvalue']:.6g}",
                    "n_fit": row["n_fit"],
                    "params": json.dumps([float(v) for v in row["params"]]),
                }
            )

    best_by_var: Dict[str, str] = {}
    for row in fit_rows:
        if row["rank"] == 1:
            best_by_var[row["variable"]] = row["distribution"]
    (output_dir / "velocity_distribution_best.json").write_text(
        json.dumps(best_by_var, indent=2)
    )

    print(f"ROI area (px): {roi_area_px:.0f}")
    print(f"Wrote: {metrics_csv}")
    print(f"Wrote: {fit_csv}")
    print(f"Wrote: {hist_path}")
    print(f"Wrote: {pdf_path}")


# ---------------------------------------------------------------------------
# Public function 3: per-run max velocity summary
# (from scripts/export_bubble_max_velocity_summary.py — cross-case aggregation
#  has been dropped; this function writes one CSV per run_dir)
# ---------------------------------------------------------------------------


def _load_tracks_by_object(tracks_csv: Path) -> Dict[int, List[dict]]:
    by_object: Dict[int, List[dict]] = {}
    with tracks_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            obj_id = int(row["object_id"])
            by_object.setdefault(obj_id, []).append(row)
    return by_object


def _load_lineage(tracks_json: Path) -> Dict[int, dict]:
    if not tracks_json.exists():
        return {}
    data = json.loads(tracks_json.read_text())
    objects = data.get("objects", {})
    out: Dict[int, dict] = {}
    for obj_id_text, payload in objects.items():
        try:
            obj_id = int(obj_id_text)
        except (TypeError, ValueError):
            continue
        out[obj_id] = {
            "birth_reason": payload.get("birth_reason", ""),
            "birth_frame_idx": payload.get("birth_frame_idx", ""),
            "birth_parent_ids": [int(v) for v in payload.get("birth_parent_ids", [])],
            "death_reason": payload.get("death_reason", ""),
            "death_frame_idx": payload.get("death_frame_idx", ""),
            "death_child_ids": [int(v) for v in payload.get("death_child_ids", [])],
        }
    return out


def _join_ids(values: List[int]) -> str:
    if not values:
        return ""
    return "|".join(str(int(v)) for v in values)


def _summarize_object(
    rows: List[dict],
    fps: float,
    mm_per_pixel: float,
    lineage: dict | None,
) -> dict:
    frame_indices = sorted(int(row["frame_idx"]) for row in rows)
    areas_px = [float(row["area_px"]) for row in rows]

    vx_px_s = [abs(v) for row in rows if (v := _to_float(row.get("velocity_x_px_s"))) is not None]
    vy_px_s = [abs(v) for row in rows if (v := _to_float(row.get("velocity_y_px_s"))) is not None]
    vmag_px_s = [float(v) for row in rows if (v := _to_float(row.get("velocity_mag_px_s"))) is not None]

    mean_area_px = sum(areas_px) / len(areas_px)
    max_area_px = max(areas_px)
    max_vx_px_frame = (max(vx_px_s) / fps) if vx_px_s else 0.0
    max_vy_px_frame = (max(vy_px_s) / fps) if vy_px_s else 0.0
    max_vmag_px_frame = (max(vmag_px_s) / fps) if vmag_px_s else 0.0

    area_scale = mm_per_pixel ** 2
    return {
        "bubble_ID": int(rows[0]["object_id"]),
        "A_bubble_avg in pixel^2": mean_area_px,
        "A_bubble_avg in mm^2": mean_area_px * area_scale,
        "A_bubble_max in pixel^2": max_area_px,
        "A_bubble_max in mm^2": max_area_px * area_scale,
        "max_velocity_x in pixel/frame": max_vx_px_frame,
        "max_velocity_x in mm/s": max_vx_px_frame * fps * mm_per_pixel,
        "max_velocity_y in pixel/frame": max_vy_px_frame,
        "max_velocity_y in mm/s": max_vy_px_frame * fps * mm_per_pixel,
        "max_velocity in pixel/frame": max_vmag_px_frame,
        "max_velocity in mm/s": max_vmag_px_frame * fps * mm_per_pixel,
        "NO_start_frame": frame_indices[0],
        "NO_end_frame": frame_indices[-1],
        "frame_amount_of_appearance": len(rows),
        "time_of_appearance in s": len(rows) / fps,
        "birth_reason": str((lineage or {}).get("birth_reason", "")),
        "birth_frame_idx": (lineage or {}).get("birth_frame_idx", ""),
        "birth_parent_ids": _join_ids((lineage or {}).get("birth_parent_ids", [])),
        "death_reason": str((lineage or {}).get("death_reason", "")),
        "death_frame_idx": (lineage or {}).get("death_frame_idx", ""),
        "death_child_ids": _join_ids((lineage or {}).get("death_child_ids", [])),
    }


_MAX_VEL_FIELDNAMES = [
    "bubble_ID",
    "A_bubble_avg in pixel^2",
    "A_bubble_avg in mm^2",
    "A_bubble_max in pixel^2",
    "A_bubble_max in mm^2",
    "max_velocity_x in pixel/frame",
    "max_velocity_x in mm/s",
    "max_velocity_y in pixel/frame",
    "max_velocity_y in mm/s",
    "max_velocity in pixel/frame",
    "max_velocity in mm/s",
    "NO_start_frame",
    "NO_end_frame",
    "frame_amount_of_appearance",
    "time_of_appearance in s",
    "birth_reason",
    "birth_frame_idx",
    "birth_parent_ids",
    "death_reason",
    "death_frame_idx",
    "death_child_ids",
]


def write_max_velocity_summary(
    *,
    run_dir: Path,
    output: Path,
) -> None:
    """Write a per-bubble area + max-velocity summary CSV to *output*.

    Reads ``tracks.csv`` and ``tracks.json`` from *run_dir* (layout
    auto-detected).  FPS and mm_per_pixel are loaded from
    ``sam2_prompted_flow_config.json`` in *run_dir*.

    Parameters
    ----------
    run_dir:
        Per-case run directory.
    output:
        Destination CSV path (e.g. ``<run_dir>/phase3/bubble_max_velocity.csv``).
    """
    tracks_dir = _resolve_tracks_dir(run_dir)
    tracks_csv = tracks_dir / "tracks.csv"
    tracks_json = tracks_dir / "tracks.json"
    if not tracks_json.exists():
        # fall back to run_dir root
        tracks_json = run_dir / "tracks.json"

    config = _load_run_config_optional(run_dir)
    if not config:
        raise FileNotFoundError(
            f"run_config.json (or legacy sam2_prompted_flow_config.json) not found under {run_dir}"
        )
    tracking_cfg = config.get("tracking", {})
    # fps/mm_per_pixel live under tracking.* in the legacy config and at
    # the top level in the new run_config.json (phase1 writes them flat).
    fps = float(tracking_cfg.get("fps", config.get("fps", 4.0)))
    if "mm_per_pixel" in tracking_cfg:
        mm_per_pixel = float(tracking_cfg["mm_per_pixel"])
    elif "mm_per_pixel" in config:
        mm_per_pixel = float(config["mm_per_pixel"])
    else:
        raise KeyError(
            f"mm_per_pixel not found in run config at {run_dir} "
            "(tried tracking.mm_per_pixel and top-level mm_per_pixel)"
        )

    by_object = _load_tracks_by_object(tracks_csv)
    lineage_by_object = _load_lineage(tracks_json)
    summaries = [
        _summarize_object(
            rows=rows,
            fps=fps,
            mm_per_pixel=mm_per_pixel,
            lineage=lineage_by_object.get(obj_id, {}),
        )
        for obj_id, rows in sorted(by_object.items())
    ]

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_MAX_VEL_FIELDNAMES)
        writer.writeheader()
        for row in summaries:
            writer.writerow(
                {
                    "bubble_ID": row["bubble_ID"],
                    "A_bubble_avg in pixel^2": f"{row['A_bubble_avg in pixel^2']:.6f}",
                    "A_bubble_avg in mm^2": f"{row['A_bubble_avg in mm^2']:.6f}",
                    "A_bubble_max in pixel^2": f"{row['A_bubble_max in pixel^2']:.6f}",
                    "A_bubble_max in mm^2": f"{row['A_bubble_max in mm^2']:.6f}",
                    "max_velocity_x in pixel/frame": f"{row['max_velocity_x in pixel/frame']:.6f}",
                    "max_velocity_x in mm/s": f"{row['max_velocity_x in mm/s']:.6f}",
                    "max_velocity_y in pixel/frame": f"{row['max_velocity_y in pixel/frame']:.6f}",
                    "max_velocity_y in mm/s": f"{row['max_velocity_y in mm/s']:.6f}",
                    "max_velocity in pixel/frame": f"{row['max_velocity in pixel/frame']:.6f}",
                    "max_velocity in mm/s": f"{row['max_velocity in mm/s']:.6f}",
                    "NO_start_frame": row["NO_start_frame"],
                    "NO_end_frame": row["NO_end_frame"],
                    "frame_amount_of_appearance": row["frame_amount_of_appearance"],
                    "time_of_appearance in s": f"{row['time_of_appearance in s']:.6f}",
                    "birth_reason": row["birth_reason"],
                    "birth_frame_idx": row["birth_frame_idx"],
                    "birth_parent_ids": row["birth_parent_ids"],
                    "death_reason": row["death_reason"],
                    "death_frame_idx": row["death_frame_idx"],
                    "death_child_ids": row["death_child_ids"],
                }
            )

    print(f"Wrote: {output}")
