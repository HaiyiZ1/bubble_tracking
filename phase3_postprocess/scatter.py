"""Phase 3 scatter module.

Provides:
  write_velocity_area_scatter_with_pdfs  — log-log scatter + marginal PDFs PNG
"""

from __future__ import annotations

import csv
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from scipy.stats import gaussian_kde


# ---------------------------------------------------------------------------
# Layout helper
# ---------------------------------------------------------------------------


def _resolve_metrics_csv(run_dir: Path) -> Path:
    """Return path to bubble_velocity_area_metrics.csv.

    Checks ``<run_dir>/phase3/`` first, then ``<run_dir>/``.
    (The file is a Phase 3 output; fresh runs write it to phase3/,
    historical/oracle runs have it at the run-dir root.)
    """
    for subdir in ("phase3", ""):
        candidate = run_dir / subdir / "bubble_velocity_area_metrics.csv" if subdir else run_dir / "bubble_velocity_area_metrics.csv"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No bubble_velocity_area_metrics.csv under {run_dir} (tried phase3/ and root)"
    )


# ---------------------------------------------------------------------------
# Internal helpers (translated from scripts/plot_velocity_area_logxy_with_marginal_pdfs.py)
# ---------------------------------------------------------------------------


def _to_float(v: str) -> float | None:
    if v is None:
        return None
    s = v.strip()
    if s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _load_xy(metrics_csv: Path, x_col: str, y_col: str) -> tuple[np.ndarray, np.ndarray]:
    xs: list[float] = []
    ys: list[float] = []
    with metrics_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            x = _to_float(row.get(x_col, ""))
            y = _to_float(row.get(y_col, ""))
            if x is None or y is None:
                continue
            if not np.isfinite(x) or not np.isfinite(y):
                continue
            if x <= 0.0 or y <= 0.0:
                continue
            xs.append(float(x))
            ys.append(float(y))
    if not xs or not ys:
        raise ValueError("No valid positive points found for requested columns.")
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)


def _kde_pdf(values: np.ndarray, n: int = 400) -> tuple[np.ndarray, np.ndarray]:
    lo = float(np.min(values))
    hi = float(np.max(values))
    if hi <= lo:
        hi = lo * 1.01
    grid = np.logspace(np.log10(lo), np.log10(hi), n)
    kde = gaussian_kde(values)
    pdf = kde(grid)
    return grid, pdf


def _fmt_decimal(v: float) -> str:
    if not np.isfinite(v):
        return "nan"
    av = abs(float(v))
    if av >= 1.0:
        return f"{v:.2f}"
    if av >= 0.1:
        return f"{v:.3f}"
    if av >= 0.01:
        return f"{v:.4f}"
    return f"{v:.5f}"


def _fmt_ratio_decimal(v: float) -> str:
    if not np.isfinite(v):
        return "nan"
    return f"{float(v):.5f}"


def _fmt_pdf_scaled(v: float, scale: float = 0.01) -> str:
    if not np.isfinite(v):
        return "nan"
    out = f"{float(v) * float(scale):.3f}"
    out = out.rstrip("0").rstrip(".")
    return out if out else "0"


def _annotate_pdf_points(
    ax: plt.Axes,
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    color: str,
    x_fmt,
    y_fmt,
    offsets: list[tuple[int, int]],
    peak_axis: str = "y",
) -> None:
    if peak_axis == "x":
        idx_peak = int(np.argmax(x_vals))
    else:
        idx_peak = int(np.argmax(y_vals))
    idxs = [0, idx_peak, len(x_vals) - 1]
    for i, idx in enumerate(idxs):
        x = float(x_vals[idx])
        y = float(y_vals[idx])
        ax.scatter([x], [y], s=18, color=color, zorder=5)
        txt = f"({x_fmt(x)}, {y_fmt(y)})"
        dx, dy = offsets[i]
        ax.annotate(
            txt,
            xy=(x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8,
            color=color,
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def write_velocity_area_scatter_with_pdfs(
    *,
    run_dir: Path,
    output: Path,
    x_column: str,
    v_ref_index: int,
    v_ref_value: float,
    y_column: str = "area_over_roi",
    point_size: float = 14.0,
    alpha: float = 0.70,
) -> None:
    """Write a log-log scatter with marginal PDFs.

    Reads ``bubble_velocity_area_metrics.csv`` under run_dir (auto-detects
    phase3/ vs root layout, since the metrics CSV is a Phase 3 output).

    Parameters
    ----------
    run_dir:      Per-case run directory.
    output:       Full path to the output PNG.
    x_column:     Column name for the x-axis (e.g. 'max_v_over_v_ref_1').
    v_ref_index:  Reference velocity index (1 or 2); used in axis labels.
    v_ref_value:  Reference velocity value in mm/s; used in axis labels.
    y_column:     Column name for the y-axis (default: 'area_over_roi').
    point_size:   Scatter point size (default: 14.0).
    alpha:        Scatter alpha (default: 0.70).
    """
    metrics_csv = _resolve_metrics_csv(run_dir)

    plot_title = (
        r"$\mathrm{Bubble\ Scatter\ (log\!-\!log):}\ "
        + rf"\max\!\left(V/V_{{\mathrm{{ref}},{v_ref_index}}}\right)\ "
        + r"\mathrm{vs}\ "
        + r"\max\!\left(A_{\mathrm{bubble}}/A_{\mathrm{ROI}}\right)$"
    )

    x, y = _load_xy(metrics_csv=metrics_csv, x_col=x_column, y_col=y_column)
    xg, xpdf = _kde_pdf(x)
    yg, ypdf = _kde_pdf(y)

    fig = plt.figure(figsize=(10, 8), dpi=160, constrained_layout=True)
    gs = fig.add_gridspec(
        nrows=2,
        ncols=2,
        height_ratios=[1.2, 4.0],
        width_ratios=[4.0, 1.2],
        hspace=0.06,
        wspace=0.06,
    )
    ax_top = fig.add_subplot(gs[0, 0])
    ax_main = fig.add_subplot(gs[1, 0])
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

    # Main scatter
    ax_main.scatter(x, y, s=float(point_size), alpha=float(alpha), color="#1f77b4")
    ax_main.set_xscale("log")
    ax_main.set_yscale("log")
    ax_main.set_xlabel(
        rf"$\max\!\left(V/V_{{\mathrm{{ref}},{v_ref_index}}}\right),\ "
        + rf"V_{{\mathrm{{ref}},{v_ref_index}}}={v_ref_value:g}\ \mathrm{{mm/s}}$"
    )
    ax_main.set_ylabel(r"$\max\!\left(A_{\mathrm{bubble}}/A_{\mathrm{ROI}}\right)$")
    ax_main.grid(True, alpha=0.25)
    if x.size >= 2 and np.unique(x).size >= 2 and np.unique(y).size >= 2:
        lx = np.log10(x)
        ly = np.log10(y)
        slope, intercept = np.polyfit(lx, ly, deg=1)
        r_val = float(np.corrcoef(lx, ly)[0, 1])
        x_line = np.logspace(np.log10(float(np.min(x))), np.log10(float(np.max(x))), 256)
        y_line = np.power(10.0, slope * np.log10(x_line) + intercept)
        ax_main.plot(
            x_line,
            y_line,
            color="#111111",
            linewidth=1.4,
            alpha=0.90,
            label=rf"$\mathrm{{Trendline}}\ (R={r_val:.3f})$",
        )
        ax_main.legend(loc="best", fontsize=8)

    # Top marginal PDF (x)
    ax_top.plot(xg, xpdf, color="#d62728", linewidth=1.6)
    ax_top.fill_between(xg, 0.0, xpdf, color="#d62728", alpha=0.18)
    ax_top.set_xscale("log")
    ax_top.set_ylabel(
        rf"$\mathrm{{PDF\ of}}\ \max\!\left(V/V_{{\mathrm{{ref}},{v_ref_index}}}\right)$"
    )
    ax_top.grid(True, alpha=0.20)
    ax_top.tick_params(axis="x", labelbottom=False)
    _annotate_pdf_points(
        ax=ax_top,
        x_vals=xg,
        y_vals=xpdf,
        color="#d62728",
        x_fmt=_fmt_decimal,
        y_fmt=_fmt_decimal,
        offsets=[(6, -14), (-30, -14), (-55, 10)],
        peak_axis="y",
    )

    # Right marginal PDF (y)
    ax_right.plot(ypdf, yg, color="#2ca02c", linewidth=1.6)
    ax_right.fill_betweenx(yg, 0.0, ypdf, color="#2ca02c", alpha=0.18)
    ax_right.set_yscale("log")
    ax_right.set_xlabel(r"$\mathrm{PDF\ of}\ \max\!\left(A_{\mathrm{bubble}}/A_{\mathrm{ROI}}\right)$")
    ax_right.grid(True, alpha=0.20)
    ax_right.tick_params(axis="y", labelleft=False)
    ax_right.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: _fmt_ratio_decimal(float(v))))
    ax_right.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: _fmt_pdf_scaled(float(v), 0.01)))
    _annotate_pdf_points(
        ax=ax_right,
        x_vals=ypdf,
        y_vals=yg,
        color="#2ca02c",
        x_fmt=lambda v: _fmt_pdf_scaled(float(v), 0.01),
        y_fmt=_fmt_ratio_decimal,
        offsets=[(-70, -10), (-70, 6), (6, 6)],
        peak_axis="x",
    )

    fig.suptitle(plot_title, fontweight="bold")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)
