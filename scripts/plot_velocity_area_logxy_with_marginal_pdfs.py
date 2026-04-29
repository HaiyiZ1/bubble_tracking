#!/usr/bin/env python3
"""Plot log-log scatter (v_ref_2) with marginal PDFs on top/right."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from scipy.stats import gaussian_kde


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
    x_fmt: callable,
    y_fmt: callable,
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


def _infer_v_ref_index_from_column(x_column: str) -> int | None:
    m = re.search(r"_v_ref_(\d+)$", x_column)
    if not m:
        return None
    try:
        idx = int(m.group(1))
    except ValueError:
        return None
    return idx if idx in (1, 2) else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create log-log scatter for v_ref_2 with top/right marginal PDFs."
    )
    parser.add_argument("--run-dir", required=True, help="Run directory containing bubble_velocity_area_metrics.csv")
    parser.add_argument(
        "--metrics-csv",
        default=None,
        help="Optional explicit metrics CSV path (default: <run-dir>/bubble_velocity_area_metrics.csv)",
    )
    parser.add_argument(
        "--x-column",
        default="max_v_over_v_ref_2",
        help="X column from metrics CSV (default: max_v_over_v_ref_2)",
    )
    parser.add_argument(
        "--y-column",
        default="area_over_roi",
        help="Y column from metrics CSV (default: area_over_roi)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output image path (default: <run-dir>/velocity_area_scatter_logxy_vref2_with_pdfs.png)",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional figure title (default is auto-generated from V_ref index)",
    )
    parser.add_argument(
        "--v-ref-index",
        type=int,
        choices=[1, 2],
        default=None,
        help="Reference velocity index used in label text (default: infer from x-column)",
    )
    parser.add_argument(
        "--v-ref-value",
        type=float,
        default=None,
        help="Reference velocity value in mm/s for label text (default: 0.23 for idx=1, 7 for idx=2)",
    )
    parser.add_argument("--point-size", type=float, default=14.0, help="Scatter point size")
    parser.add_argument("--alpha", type=float, default=0.70, help="Scatter alpha")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    metrics_csv = Path(args.metrics_csv) if args.metrics_csv else (run_dir / "bubble_velocity_area_metrics.csv")
    if not metrics_csv.exists():
        raise ValueError(f"Metrics CSV not found: {metrics_csv}")

    v_ref_idx = int(args.v_ref_index) if args.v_ref_index is not None else _infer_v_ref_index_from_column(args.x_column)
    if v_ref_idx is None:
        v_ref_idx = 2
    if args.v_ref_value is not None:
        v_ref_value = float(args.v_ref_value)
    else:
        v_ref_value = 0.23 if v_ref_idx == 1 else 7.0

    output = Path(args.output) if args.output else (
        run_dir / f"velocity_area_scatter_logxy_vref{v_ref_idx}_with_pdfs.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    if args.title:
        plot_title = args.title
    else:
        plot_title = (
            r"$\mathrm{Bubble\ Scatter\ (log\!-\!log):}\ "
            + rf"\max\!\left(V/V_{{\mathrm{{ref}},{v_ref_idx}}}\right)\ "
            + r"\mathrm{vs}\ "
            + r"\max\!\left(A_{\mathrm{bubble}}/A_{\mathrm{ROI}}\right)$"
        )

    x, y = _load_xy(metrics_csv=metrics_csv, x_col=args.x_column, y_col=args.y_column)
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
    ax_main.scatter(x, y, s=float(args.point_size), alpha=float(args.alpha), color="#1f77b4")
    ax_main.set_xscale("log")
    ax_main.set_yscale("log")
    ax_main.set_xlabel(
        rf"$\max\!\left(V/V_{{\mathrm{{ref}},{v_ref_idx}}}\right),\ "
        + rf"V_{{\mathrm{{ref}},{v_ref_idx}}}={v_ref_value:g}\ \mathrm{{mm/s}}$"
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
        rf"$\mathrm{{PDF\ of}}\ \max\!\left(V/V_{{\mathrm{{ref}},{v_ref_idx}}}\right)$"
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
    # Annotate coordinates as (PDF, max(A_bubble/A_ROI)).
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
    fig.savefig(output)
    plt.close(fig)

    print(f"Wrote: {output}")
    print(f"Points plotted: {len(x)}")


if __name__ == "__main__":
    main()
