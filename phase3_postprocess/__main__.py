"""CLI entry for Phase 3.

Usage:
    python -m phase3_postprocess --run-dir outputs/case1_air1_publish \\
        [--v-ref-1 0.23] [--v-ref-2 7.0]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .publishing import write_publishing_overlay_videos
from .scatter import write_velocity_area_scatter_with_pdfs
from .vanished_area import write_vanished_area_plots
from .velocity import (
    write_max_velocity_summary,
    write_velocity_metrics_csv,
    write_velocity_overlay_video,
)


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser("phase3_postprocess")
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--v-ref-1", type=float, default=0.23)
    p.add_argument("--v-ref-2", type=float, default=7.0)
    return p.parse_args()


def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from phase1_preprocess.run_config import load as load_run_config, mark_phase_done

    args = _parse()
    cfg = load_run_config(args.run_dir)
    if not cfg.phase2_done:
        sys.exit(f"{args.run_dir}/run_config.json: phase2_done=False. Run phase2 first.")

    phase3_dir = args.run_dir / "phase3"
    phase3_dir.mkdir(parents=True, exist_ok=True)

    write_velocity_overlay_video(
        run_dir=args.run_dir, output=phase3_dir / "velocity_overlay.mp4",
        fps=cfg.fps, label_units="mm_s",
    )
    write_velocity_metrics_csv(
        run_dir=args.run_dir, output_dir=phase3_dir,
        v_ref_1=args.v_ref_1, v_ref_2=args.v_ref_2,
    )
    write_max_velocity_summary(
        run_dir=args.run_dir, output=phase3_dir / "bubble_max_velocity.csv",
    )
    write_vanished_area_plots(run_dir=args.run_dir, output_dir=phase3_dir, fps=cfg.fps)
    write_velocity_area_scatter_with_pdfs(
        run_dir=args.run_dir,
        output=phase3_dir / "velocity_area_scatter_logxy_vref1_with_pdfs.png",
        x_column="max_v_over_v_ref_1", v_ref_index=1, v_ref_value=args.v_ref_1,
    )
    write_velocity_area_scatter_with_pdfs(
        run_dir=args.run_dir,
        output=phase3_dir / "velocity_area_scatter_logxy_vref2_with_pdfs.png",
        x_column="max_v_over_v_ref_2", v_ref_index=2, v_ref_value=args.v_ref_2,
    )
    write_publishing_overlay_videos(run_dir=args.run_dir, case=cfg.case, fps=cfg.fps)

    mark_phase_done(args.run_dir, 3)


if __name__ == "__main__":
    main()
