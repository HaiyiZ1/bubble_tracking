#!/usr/bin/env python3
"""Plot velocity-normalized histograms and per-bubble velocity-area scatter."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def _to_float(value: str) -> float | None:
    if value is None:
        return None
    value = value.strip()
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _load_run_config(run_dir: Path) -> dict:
    config_path = run_dir / "sam2_prompted_flow_config.json"
    if not config_path.exists():
        return {}
    try:
        return json.loads(config_path.read_text())
    except Exception:
        return {}


def _resolve_roi_area(run_dir: Path, config: dict, explicit_roi_area: float | None) -> float:
    if explicit_roi_area is not None and explicit_roi_area > 0:
        return float(explicit_roi_area)

    roi_mask_path = None
    roi_json_path = None
    if isinstance(config, dict):
        roi_mask_path = config.get("roi_mask")
        roi_json_path = config.get("roi_json")

    if roi_mask_path:
        p = Path(roi_mask_path)
        if not p.is_absolute():
            p = (run_dir / p).resolve()
        if p.exists():
            mask = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                area = int((mask > 0).sum())
                if area > 0:
                    return float(area)

    if roi_json_path:
        p = Path(roi_json_path)
        if not p.is_absolute():
            p = (run_dir / p).resolve()
        if p.exists():
            data = json.loads(p.read_text())
            if all(k in data for k in ("x1", "y1", "x2", "y2")):
                area = max(0, int(data["x2"]) - int(data["x1"])) * max(0, int(data["y2"]) - int(data["y1"]))
                if area > 0:
                    return float(area)

    raise ValueError(
        "Failed to resolve ROI area. Provide --roi-area or ensure run config has valid roi_mask/roi_json."
    )


def _load_tracks(tracks_csv: Path) -> List[dict]:
    rows: List[dict] = []
    with tracks_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _extract_velocity_vectors(rows: List[dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

    stats: Dict[int, float] = {}
    for obj_id, vals in bucket.items():
        arr = np.asarray(vals, dtype=np.float64)
        if area_stat == "max":
            stats[obj_id] = float(np.max(arr))
        elif area_stat == "mean":
            stats[obj_id] = float(np.mean(arr))
        elif area_stat == "median":
            stats[obj_id] = float(np.median(arr))
        else:
            raise ValueError(f"Unsupported area_stat: {area_stat}")
    return stats


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
            ax.text(0.5, 0.5, "No valid points for log scale", ha="center", va="center", transform=ax.transAxes)

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
                0.99,
                0.95,
                f"dropped: {dropped}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                color="#444444",
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
    fig.suptitle(f"Bubble-level Scatter: Peak Normalized Speed vs Normalized Bubble Area ({scale_label})")
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot normalized velocity histograms and per-bubble scatter from tracks.csv"
    )
    parser.add_argument("--run-dir", required=True, help="Run output directory (contains tracks.csv and config)")
    parser.add_argument("--tracks-csv", default=None, help="Optional explicit tracks.csv path")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: run-dir)")
    parser.add_argument("--roi-area", type=float, default=None, help="Optional ROI area in pixels")
    parser.add_argument("--v-ref-1", type=float, default=0.23, help="Reference velocity 1 (mm/s)")
    parser.add_argument("--v-ref-2", type=float, default=7.0, help="Reference velocity 2 (mm/s)")
    parser.add_argument(
        "--bins",
        default="auto",
        help="Histogram bins (integer or 'auto', default: auto)",
    )
    parser.add_argument(
        "--hist-clip-low",
        type=float,
        default=0.5,
        help="Lower percentile for histogram x-range clipping (default: 0.5)",
    )
    parser.add_argument(
        "--hist-clip-high",
        type=float,
        default=99.5,
        help="Upper percentile for histogram x-range clipping (default: 99.5)",
    )
    parser.add_argument(
        "--fit-max-samples",
        type=int,
        default=20000,
        help="Max samples per variable for parametric distribution fitting",
    )
    parser.add_argument(
        "--area-stat",
        choices=["max", "mean", "median"],
        default="max",
        help="Per-bubble area statistic for scatter y-axis",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise ValueError(f"Run dir not found: {run_dir}")

    tracks_csv = Path(args.tracks_csv) if args.tracks_csv else (run_dir / "tracks.csv")
    if not tracks_csv.exists():
        raise ValueError(f"tracks.csv not found: {tracks_csv}")

    output_dir = Path(args.output_dir) if args.output_dir else run_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    config = _load_run_config(run_dir)
    roi_area_px = _resolve_roi_area(run_dir, config, args.roi_area)

    bins_arg = args.bins
    bins = bins_arg
    if isinstance(bins_arg, str):
        b = bins_arg.strip().lower()
        if b != "auto":
            bins = int(b)
    if not (0.0 <= args.hist_clip_low < args.hist_clip_high <= 100.0):
        raise ValueError("--hist-clip-low/--hist-clip-high must satisfy 0 <= low < high <= 100")

    rows = _load_tracks(tracks_csv)
    vx_mm_s, vy_mm_s, vmag_mm_s, obj_ids = _extract_velocity_vectors(rows)
    if vx_mm_s.size == 0:
        raise ValueError("No valid velocity rows found in tracks.csv (mm/s columns are empty).")

    hist_path = output_dir / "velocity_norm_histograms.png"
    _plot_histograms(
        vx_mm_s=vx_mm_s,
        vy_mm_s=vy_mm_s,
        v_ref_1=float(args.v_ref_1),
        v_ref_2=float(args.v_ref_2),
        bins=bins,
        clip_low_pct=float(args.hist_clip_low),
        clip_high_pct=float(args.hist_clip_high),
        output_path=hist_path,
    )
    pdf_path = output_dir / "velocity_norm_pdf_fit.png"
    fit_rows = _plot_pdf_with_fit(
        vx_mm_s=vx_mm_s,
        vy_mm_s=vy_mm_s,
        v_ref_1=float(args.v_ref_1),
        v_ref_2=float(args.v_ref_2),
        clip_low_pct=float(args.hist_clip_low),
        clip_high_pct=float(args.hist_clip_high),
        fit_max_samples=int(args.fit_max_samples),
        output_path=pdf_path,
    )

    obj_max_v_mm_s: Dict[int, float] = {}
    for obj_id, vmag in zip(obj_ids.tolist(), vmag_mm_s.tolist()):
        obj_max_v_mm_s[obj_id] = max(float(vmag), obj_max_v_mm_s.get(obj_id, float("-inf")))

    obj_area_px = _compute_object_area_stats(rows, args.area_stat)

    scatter_path = output_dir / "velocity_area_scatter.png"
    _plot_scatter(
        obj_max_v_mm_s=obj_max_v_mm_s,
        obj_area_px=obj_area_px,
        roi_area_px=roi_area_px,
        v_ref_1=float(args.v_ref_1),
        v_ref_2=float(args.v_ref_2),
        output_path=scatter_path,
        area_stat_label=args.area_stat,
        log_x=False,
        log_y=False,
    )
    scatter_logx_path = output_dir / "velocity_area_scatter_logx.png"
    _plot_scatter(
        obj_max_v_mm_s=obj_max_v_mm_s,
        obj_area_px=obj_area_px,
        roi_area_px=roi_area_px,
        v_ref_1=float(args.v_ref_1),
        v_ref_2=float(args.v_ref_2),
        output_path=scatter_logx_path,
        area_stat_label=args.area_stat,
        log_x=True,
        log_y=False,
    )
    scatter_logy_path = output_dir / "velocity_area_scatter_logy.png"
    _plot_scatter(
        obj_max_v_mm_s=obj_max_v_mm_s,
        obj_area_px=obj_area_px,
        roi_area_px=roi_area_px,
        v_ref_1=float(args.v_ref_1),
        v_ref_2=float(args.v_ref_2),
        output_path=scatter_logy_path,
        area_stat_label=args.area_stat,
        log_x=False,
        log_y=True,
    )
    scatter_logxy_path = output_dir / "velocity_area_scatter_logxy.png"
    _plot_scatter(
        obj_max_v_mm_s=obj_max_v_mm_s,
        obj_area_px=obj_area_px,
        roi_area_px=roi_area_px,
        v_ref_1=float(args.v_ref_1),
        v_ref_2=float(args.v_ref_2),
        output_path=scatter_logxy_path,
        area_stat_label=args.area_stat,
        log_x=True,
        log_y=True,
    )

    metrics_csv = output_dir / "bubble_velocity_area_metrics.csv"
    _write_object_metrics(
        output_csv=metrics_csv,
        obj_max_v_mm_s=obj_max_v_mm_s,
        obj_area_px=obj_area_px,
        roi_area_px=roi_area_px,
        v_ref_1=float(args.v_ref_1),
        v_ref_2=float(args.v_ref_2),
    )

    fit_csv = output_dir / "velocity_distribution_fit.csv"
    with fit_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "variable",
                "rank",
                "distribution",
                "aic",
                "bic",
                "ks_stat",
                "ks_pvalue",
                "n_fit",
                "params",
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

    best_by_var = {}
    for row in fit_rows:
        var = row["variable"]
        if row["rank"] == 1:
            best_by_var[var] = row["distribution"]
    (output_dir / "velocity_distribution_best.json").write_text(json.dumps(best_by_var, indent=2))

    print(f"ROI area (px): {roi_area_px:.0f}")
    print(f"Wrote: {hist_path}")
    print(f"Wrote: {pdf_path}")
    print(f"Wrote: {fit_csv}")
    print(f"Wrote: {scatter_path}")
    print(f"Wrote: {scatter_logx_path}")
    print(f"Wrote: {scatter_logy_path}")
    print(f"Wrote: {scatter_logxy_path}")
    print(f"Wrote: {metrics_csv}")


if __name__ == "__main__":
    main()
