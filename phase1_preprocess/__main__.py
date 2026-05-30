"""CLI entry for Phase 1.

Usage:
    python -m phase1_preprocess \\
        --case case1.air1 \\
        --raw-frames data/bubbles/case1_air1_publish/frames \\
        --run-dir outputs/case1_air1_publish \\
        [--meta meta.yaml] [--downscale 0.5] \\
        [--roi-from <path>] [--roi-mask-from <path>]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

from .frame_prep import prepare
from .meta import load_case
from .preprocess import PreprocessConfig
from .roi import import_roi, interactive_select
from .run_config import RunConfig
from .visualize import DEFAULT_KEYFRAME_INTERVAL, keyframe_indices_for, render_previews


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser("phase1_preprocess")
    p.add_argument("--case", required=True, help="Case + air, e.g. case1.air1")
    p.add_argument("--raw-frames", required=True, type=Path)
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--meta", type=Path, default=Path("meta.yaml"))
    p.add_argument("--downscale", type=float, default=0.5)
    p.add_argument("--roi-from", type=Path, default=None)
    p.add_argument("--roi-mask-from", type=Path, default=None)
    p.add_argument(
        "--keyframe-interval", type=int, default=DEFAULT_KEYFRAME_INTERVAL,
        help="Spacing of preview keyframes (matches phase2 CV_KEYFRAME_INTERVAL).",
    )
    p.add_argument(
        "--keyframe-start", type=int, default=1,
        help="First preview keyframe index (1-based on relabeled frames).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse()

    case_spec = load_case(args.meta, args.case)
    mm_per_pixel = case_spec.mm_per_pixel(args.downscale)

    phase1_dir = args.run_dir / "phase1"
    frames_out = phase1_dir / "frames_0p5"

    prepared = prepare(
        raw_frames=args.raw_frames,
        out_dir=frames_out,
        downscale=args.downscale,
        start_frame=case_spec.start_frame,
    )

    # ROI: import vs interactive.
    if args.roi_from and args.roi_mask_from:
        roi_json_path, roi_mask_path = import_roi(args.roi_from, args.roi_mask_from, phase1_dir)
    elif args.roi_from or args.roi_mask_from:
        sys.exit("--roi-from and --roi-mask-from must be passed together.")
    else:
        roi_json_path, roi_mask_path = interactive_select(frames_out, phase1_dir)

    roi = json.loads(roi_json_path.read_text())
    roi_mask = cv2.imread(str(roi_mask_path), cv2.IMREAD_GRAYSCALE)
    if roi_mask is None:
        sys.exit(f"Could not read ROI mask {roi_mask_path}")

    # Keyframe previews. The default (light-map ON, contrast normalize per
    # flow_tracking defaults) matches the historical preprocess_frame
    # configuration; reviewers care about what the segmentation sees, so
    # use cfg.mode="default" rather than the lightweight "raw" mode.
    cfg = PreprocessConfig(mode="default")
    indices = keyframe_indices_for(
        prepared.n_frames,
        interval=args.keyframe_interval,
        start=args.keyframe_start,
    )
    written = render_previews(
        frames_dir=frames_out,
        reference_path=frames_out / "000001.jpg",
        roi=roi,
        roi_mask=roi_mask,
        out_dir=phase1_dir / "keyframe_previews",
        cfg=cfg,
        keyframe_indices=indices,
    )

    summary = {
        "n_frames": prepared.n_frames,
        "relabel_skipped": prepared.relabel_skipped,
        "downscale": args.downscale,
        "mm_per_pixel_at_downscale": mm_per_pixel,
        "fps": case_spec.fps,
        "case": args.case,
        "start_frame_meta": case_spec.start_frame,
        "n_preview_keyframes": len(indices),
        "n_preview_pngs_written": len(written),
    }
    (phase1_dir / "phase1_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    cfg_out = RunConfig(
        case=args.case,
        run_dir=str(args.run_dir),
        meta_path=str(args.meta),
        fps=case_spec.fps,
        mm_per_pixel=mm_per_pixel,
        downscale=args.downscale,
        start_frame=case_spec.start_frame,
        relabel_skipped=prepared.relabel_skipped,
        frames=str(frames_out),
        reference=str(frames_out / "000001.jpg"),
        roi_json=str(roi_json_path),
        roi_mask=str(roi_mask_path),
        phase1_done=True,
    )
    cfg_out.write()

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
