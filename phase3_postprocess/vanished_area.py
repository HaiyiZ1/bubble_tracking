"""Phase 3 vanished-area module.

Provides:
  write_vanished_area_plots  — moving-window CSV + 8 PNGs
"""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Layout helper
# ---------------------------------------------------------------------------

WINDOWS_SECONDS = [2.0, 5.0]


def _resolve_events_dir(run_dir: Path) -> Path:
    """Return the dir containing events.json.

    Tries ``<run_dir>/phase2`` first, then ``<run_dir>``.
    """
    if (run_dir / "phase2" / "events.json").exists():
        return run_dir / "phase2"
    if (run_dir / "events.json").exists():
        return run_dir
    raise FileNotFoundError(
        f"No events.json under {run_dir} (tried phase2/ and root)"
    )


# ---------------------------------------------------------------------------
# Internal helpers (translated from scripts/plot_vanished_area_moving_window.py)
# ---------------------------------------------------------------------------


def _load_events(events_path: Path) -> dict:
    return json.loads(events_path.read_text())


def _load_total_frames(area_path: Path, fallback_max_frame: int) -> int:
    if area_path.exists():
        area_rows = json.loads(area_path.read_text())
        if area_rows:
            max_frame = max(int(row.get("frame_idx", -1)) for row in area_rows)
            return max(0, max_frame + 1)
    return max(0, fallback_max_frame + 1)


def _extract_death_events(events_data: dict, max_frame_exclusive: int) -> List[Tuple[int, int]]:
    rows: List[Tuple[int, int]] = []
    for ev in events_data.get("events", []):
        if str(ev.get("type", "")) != "death":
            continue
        frame_idx = int(ev.get("frame_idx", -1))
        if frame_idx < 0 or frame_idx >= int(max_frame_exclusive):
            continue
        area_px = int(ev.get("area_px", 0))
        if area_px <= 0:
            continue
        rows.append((frame_idx, area_px))
    rows.sort(key=lambda x: x[0])
    return rows


def _moving_window_totals_per_second(
    deaths: List[Tuple[int, int]],
    fps: float,
    total_frames: int,
    window_seconds: float,
) -> Tuple[List[int], List[float], List[float]]:
    if fps <= 0:
        raise ValueError("FPS must be > 0")
    if total_frames <= 0:
        return [], [], []

    max_second = int(math.floor((total_frames - 1) / fps))
    seconds = list(range(max_second + 1))
    center_frames = [sec * fps for sec in seconds]
    half_window_frames = (window_seconds / 2.0) * fps

    totals: List[float] = []
    for center_frame in center_frames:
        total = 0.0
        for death_frame, death_area in deaths:
            if abs(float(death_frame) - float(center_frame)) <= half_window_frames:
                total += float(death_area)
        totals.append(total)
    return seconds, center_frames, totals


def _plot_series(
    x: List[float],
    y: List[float],
    xlabel: str,
    title: str,
    output_path: Path,
    log_y: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=160)
    if log_y:
        y_plot = np.asarray([float(v) if float(v) > 0.0 else np.nan for v in y], dtype=np.float64)
    else:
        y_plot = np.asarray(y, dtype=np.float64)
    ax.plot(x, y_plot, color="#1f77b4", linewidth=1.8)
    ax.set_xlabel(xlabel)
    if log_y:
        ax.set_yscale("log")
        ax.set_ylabel("Vanished Bubble Area (px, log scale)")
    else:
        ax.set_ylabel("Vanished Bubble Area (px)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _write_csv(
    output_path: Path,
    seconds: List[int],
    center_frames: List[float],
    totals_by_window: Dict[float, List[float]],
) -> None:
    fieldnames = [
        "second",
        "center_frame",
        "vanished_area_window_2s_px",
        "vanished_area_window_5s_px",
    ]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, second in enumerate(seconds):
            row = {
                "second": int(second),
                "center_frame": float(center_frames[idx]),
                "vanished_area_window_2s_px": float(totals_by_window.get(2.0, [0.0] * len(seconds))[idx]),
                "vanished_area_window_5s_px": float(totals_by_window.get(5.0, [0.0] * len(seconds))[idx]),
            }
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def write_vanished_area_plots(*, run_dir: Path, output_dir: Path, fps: float) -> None:
    """Write vanished_area_moving_window.csv + 8 PNGs (2s/5s × frames/seconds × linear/logy).

    Reads events.json (and optionally area_timeseries.json) from
    ``<run_dir>/phase2/`` (fresh runs) or ``<run_dir>/`` (legacy/oracle).
    All outputs are written to ``output_dir``.
    """
    events_dir = _resolve_events_dir(run_dir)
    events_path = events_dir / "events.json"
    area_path = events_dir / "area_timeseries.json"

    events_data = _load_events(events_path)

    if fps <= 0:
        raise ValueError("FPS must be > 0")

    fallback_max_frame = max(
        [int(ev.get("frame_idx", -1)) for ev in events_data.get("events", [])] + [-1]
    )
    total_frames = _load_total_frames(area_path, fallback_max_frame)
    deaths = _extract_death_events(events_data, max_frame_exclusive=total_frames)

    output_dir.mkdir(parents=True, exist_ok=True)

    totals_by_window: Dict[float, List[float]] = {}
    seconds: List[int] = []
    center_frames: List[float] = []
    for window_seconds in WINDOWS_SECONDS:
        sec, center_fr, totals = _moving_window_totals_per_second(
            deaths=deaths,
            fps=fps,
            total_frames=total_frames,
            window_seconds=window_seconds,
        )
        totals_by_window[window_seconds] = totals
        if not seconds:
            seconds = sec
            center_frames = center_fr

    csv_path = output_dir / "vanished_area_moving_window.csv"
    _write_csv(
        output_path=csv_path,
        seconds=seconds,
        center_frames=center_frames,
        totals_by_window=totals_by_window,
    )

    for window_seconds in WINDOWS_SECONDS:
        half = window_seconds / 2.0
        totals = totals_by_window[window_seconds]
        window_tag = f"{int(window_seconds)}s"

        _plot_series(
            x=[float(s) for s in seconds],
            y=totals,
            xlabel="Time (s)",
            title=f"Vanished Bubble Area ({window_tag} window, +/-{half:.1f}s)",
            output_path=output_dir / f"vanished_area_window_{window_tag}_x_seconds.png",
        )
        _plot_series(
            x=[float(s) for s in seconds],
            y=totals,
            xlabel="Time (s)",
            title=f"Vanished Bubble Area ({window_tag} window, +/-{half:.1f}s, log y)",
            output_path=output_dir / f"vanished_area_window_{window_tag}_x_seconds_logy.png",
            log_y=True,
        )
        _plot_series(
            x=center_frames,
            y=totals,
            xlabel="Frame Index",
            title=f"Vanished Bubble Area ({window_tag} window, +/-{half:.1f}s)",
            output_path=output_dir / f"vanished_area_window_{window_tag}_x_frames.png",
        )
        _plot_series(
            x=center_frames,
            y=totals,
            xlabel="Frame Index",
            title=f"Vanished Bubble Area ({window_tag} window, +/-{half:.1f}s, log y)",
            output_path=output_dir / f"vanished_area_window_{window_tag}_x_frames_logy.png",
            log_y=True,
        )
