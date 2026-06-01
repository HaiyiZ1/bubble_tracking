"""CLI entry for Phase 1.

Usage:
    python -m phase1_preprocess \\
        --case case1.air1 \\
        --raw-frames data/bubbles/case1_air1_publish/frames \\
        --run-dir outputs/case1_air1_publish \\
        [--meta meta.yaml] [--downscale 0.5] \\
        [--roi-from <path>] [--roi-mask-from <path>] \\
        --cv-keyframe-start-idx 10 \\
        [--cv-keyframe-threshold-offset 7] \\
        [--cv-keyframe-min-area 50] \\
        [--cv-keyframe-normalize-lightmap] \\
        [--cv-keyframe-extra-indices N [N ...]]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

from .frame_prep import prepare
from .meta import load_case
from .roi import import_roi, interactive_select
from .cv_keyframes import build_cv_keyframes
from bubble_common.io import sorted_frame_paths
from bubble_common.run_config import RunConfig


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser("phase1_preprocess")
    p.add_argument("--case", required=True, help="Case + air, e.g. case1.air1")
    p.add_argument("--raw-frames", required=True, type=Path)
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--meta", type=Path, default=Path("meta.yaml"))
    p.add_argument("--downscale", type=float, default=0.5)
    p.add_argument("--roi-from", type=Path, default=None)
    p.add_argument("--roi-mask-from", type=Path, default=None)
    # CV-keyframe flags
    p.add_argument("--cv-keyframe-start-idx", required=True, type=int)
    p.add_argument("--cv-keyframe-threshold-offset", type=int, default=7)
    p.add_argument("--cv-keyframe-min-area", type=int, default=50)
    p.add_argument(
        "--cv-keyframe-normalize-lightmap", dest="cv_keyframe_normalize_lightmap",
        action="store_true", default=False,
    )
    p.add_argument("--cv-keyframe-extra-indices", nargs="+", type=int, default=None)
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

    # Load reference frame (first frame in the output directory).
    reference_path = frames_out / "000001.jpg"
    reference_full = cv2.imread(str(reference_path))
    if reference_full is None:
        sys.exit(f"Could not read reference frame {reference_path}")

    # Load sorted frame paths for the CV-keyframe builder.
    frame_paths = sorted_frame_paths(str(frames_out))
    if not frame_paths:
        sys.exit(f"No frames found in {frames_out}")

    # --- CV-keyframe prompts ---
    cv_summary = build_cv_keyframes(
        frame_paths=frame_paths,
        reference_full=reference_full,
        roi=roi,
        roi_mask=roi_mask,
        out_prompts_dir=phase1_dir / "cv_keyframe_prompts",
        out_debug_dir=phase1_dir / "cv_keyframe_debug",
        start_idx=args.cv_keyframe_start_idx,
        threshold_offset=args.cv_keyframe_threshold_offset,
        min_area=args.cv_keyframe_min_area,
        normalize_lightmap=args.cv_keyframe_normalize_lightmap,
        extra_indices=args.cv_keyframe_extra_indices,
    )

    cv_keyframes_config = {
        "start_idx": args.cv_keyframe_start_idx,
        "threshold_offset": args.cv_keyframe_threshold_offset,
        "min_area": args.cv_keyframe_min_area,
        "normalize_lightmap": args.cv_keyframe_normalize_lightmap,
        "extra_indices": [int(i) for i in (args.cv_keyframe_extra_indices or [])],
    }

    summary = {
        "n_frames": prepared.n_frames,
        "relabel_skipped": prepared.relabel_skipped,
        "downscale": args.downscale,
        "mm_per_pixel_at_downscale": mm_per_pixel,
        "fps": case_spec.fps,
        "case": args.case,
        "start_frame_meta": case_spec.start_frame,
        "cv_keyframes": cv_summary,
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
        reference=str(reference_path),
        roi_json=str(roi_json_path),
        roi_mask=str(roi_mask_path),
        phase1_done=True,
        cv_keyframes=cv_keyframes_config,
    )
    cfg_out.write()

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
