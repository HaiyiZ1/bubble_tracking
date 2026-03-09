#!/usr/bin/env python3
"""
Inspect CV preprocessing/segmentation on a single frame and export overlay artifacts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bubble_tracking.flow_tracking import (
    PreprocessConfig,
    SegmentationConfig,
    load_roi_json,
    preprocess_frame,
    preprocess_frame_with_debug,
    segment_frame,
    segment_frame_automatrix,
)
from bubble_tracking.visualization import draw_mask_overlay


def _resolve_frame_path(frames_dir: Path, frame: str) -> Path:
    frame = frame.strip()
    p = Path(frame)
    if p.suffix:
        return frames_dir / p.name
    return frames_dir / f"{int(frame):06d}.jpg"


def _preprocess_for_cv_prompt(
    frame_crop: np.ndarray,
    ref_crop: np.ndarray,
    cfg: PreprocessConfig,
    disable_light_map: bool,
    no_normalize: bool,
    use_clahe: bool,
) -> tuple[np.ndarray, str]:
    mode = (cfg.mode or "default").lower()

    if no_normalize:
        diff = frame_crop.astype(np.float32) - ref_crop.astype(np.float32)
        if cfg.diff_sigma > 0:
            diff = cv2.GaussianBlur(diff, (0, 0), sigmaX=cfg.diff_sigma, sigmaY=cfg.diff_sigma)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        gray = np.clip(diff_gray + 128.0, 0, 255).astype(np.uint8)
        if use_clahe:
            clahe = cv2.createCLAHE(
                clipLimit=cfg.clahe_clip_limit,
                tileGridSize=cfg.clahe_tile_grid,
            )
            gray = clahe.apply(gray)
        return gray, "diff_no_normalize"

    if disable_light_map and mode in {"default", "automatrix"}:
        debug = preprocess_frame_with_debug(frame_crop, ref_crop, cfg)
        adjusted = debug["adjusted"]
        gray = cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY) if adjusted.ndim == 3 else adjusted.astype(np.uint8)
        if use_clahe:
            clahe = cv2.createCLAHE(
                clipLimit=cfg.clahe_clip_limit,
                tileGridSize=cfg.clahe_tile_grid,
            )
            gray = clahe.apply(gray)
        return gray, "adjusted_no_light_map"

    gray = preprocess_frame(frame_crop, ref_crop, cfg)
    return gray, "normalized"


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect CV preprocessing/segmentation on one frame")
    parser.add_argument("--frames-dir", default="data/bubbles/frames_0p5", help="Frames directory")
    parser.add_argument("--frame", default="000100.jpg", help="Frame filename or numeric index")
    parser.add_argument("--reference", default="data/bubbles/frames_0p5/000001.jpg", help="Reference frame path")
    parser.add_argument("--roi", default="data/interim/bubbles_air1_072_271/roi_0p5.json", help="ROI json path")
    parser.add_argument("--roi-mask", default="data/interim/bubbles_air1_072_271/roi_mask_0p5.png", help="ROI mask path")
    parser.add_argument("--output-dir", default="outputs/debug_cv_single_frame", help="Output directory")
    parser.add_argument(
        "--preprocess-mode",
        choices=["raw", "default", "automatrix"],
        default="default",
        help="Preprocess mode",
    )
    parser.add_argument("--diff-sigma", type=float, default=0.5, help="Diff smoothing sigma")
    parser.add_argument("--use-clahe", action="store_true", help="Enable CLAHE")
    parser.add_argument("--disable-light-map", action="store_true", help="Disable light-map normalization")
    parser.add_argument("--no-normalize", action="store_true", help="No normalization at all (raw diff gray)")
    parser.add_argument("--min-area", type=int, default=100, help="Segmentation minimum area")
    parser.add_argument("--threshold-offset", type=int, default=7, help="Segmentation threshold offset")
    args = parser.parse_args()

    if args.no_normalize and args.preprocess_mode == "raw":
        parser.error("--no-normalize is only meaningful with reference-diff modes (default/automatrix)")

    frames_dir = Path(args.frames_dir)
    frame_path = _resolve_frame_path(frames_dir, args.frame)
    ref_path = Path(args.reference)
    roi_path = Path(args.roi)
    roi_mask_path = Path(args.roi_mask)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame = cv2.imread(str(frame_path))
    ref = cv2.imread(str(ref_path))
    if frame is None:
        raise SystemExit(f"Failed to read {frame_path}")
    if ref is None:
        raise SystemExit(f"Failed to read {ref_path}")
    if frame.shape[:2] != ref.shape[:2]:
        raise SystemExit("Frame/reference size mismatch")

    roi = load_roi_json(str(roi_path))
    roi_mask = cv2.imread(str(roi_mask_path), cv2.IMREAD_GRAYSCALE)
    if roi_mask is None:
        raise SystemExit(f"Failed to read {roi_mask_path}")
    if roi_mask.shape[:2] != frame.shape[:2]:
        raise SystemExit("ROI mask size mismatch")

    x1, y1, x2, y2 = roi["x1"], roi["y1"], roi["x2"], roi["y2"]
    frame_crop = frame[y1:y2, x1:x2]
    ref_crop = ref[y1:y2, x1:x2]

    pre_cfg = PreprocessConfig(
        mode=args.preprocess_mode,
        diff_sigma=args.diff_sigma,
        use_clahe=args.use_clahe,
    )
    seg_cfg = SegmentationConfig(
        min_area=args.min_area,
        threshold_offset=args.threshold_offset,
    )

    gray, stage = _preprocess_for_cv_prompt(
        frame_crop=frame_crop,
        ref_crop=ref_crop,
        cfg=pre_cfg,
        disable_light_map=args.disable_light_map,
        no_normalize=args.no_normalize,
        use_clahe=args.use_clahe,
    )

    if (args.preprocess_mode or "default").lower() == "automatrix":
        labels, _ = segment_frame_automatrix(gray, seg_cfg)
    else:
        labels, _ = segment_frame(gray, seg_cfg)

    mask_crop = (labels > 0).astype(np.uint8)
    mask_full = np.zeros(frame.shape[:2], dtype=np.uint8)
    mask_full[y1:y2, x1:x2] = mask_crop
    mask_full[roi_mask == 0] = 0

    inst_mask = np.zeros_like(mask_full, dtype=np.int32)
    inst_mask[mask_full > 0] = 1
    overlay = draw_mask_overlay(frame, inst_mask, alpha=0.45)

    roi_outline = np.zeros(frame.shape[:2], dtype=np.uint8)
    roi_outline[roi_mask > 0] = 255
    contours, _ = cv2.findContours(roi_outline, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 255, 0), 2)

    stem = Path(frame_path).stem
    raw_out = out_dir / f"frame_{stem}_raw.jpg"
    pre_out = out_dir / f"frame_{stem}_preprocessed_{stage}.png"
    mask_out = out_dir / f"frame_{stem}_mask.png"
    overlay_out = out_dir / f"frame_{stem}_overlay.png"
    meta_out = out_dir / f"frame_{stem}_stats.json"

    cv2.imwrite(str(raw_out), frame)
    cv2.imwrite(str(pre_out), gray)
    cv2.imwrite(str(mask_out), (mask_full * 255).astype(np.uint8))
    cv2.imwrite(str(overlay_out), overlay)

    num_labels, _ = cv2.connectedComponents(mask_full)
    num_components = max(0, int(num_labels - 1))
    area = int(mask_full.sum())
    stats = {
        "frame": frame_path.name,
        "preprocess_mode": args.preprocess_mode,
        "stage": stage,
        "use_clahe": bool(args.use_clahe),
        "disable_light_map": bool(args.disable_light_map),
        "no_normalize": bool(args.no_normalize),
        "min_area": int(args.min_area),
        "threshold_offset": int(args.threshold_offset),
        "components": num_components,
        "foreground_px": area,
        "outputs": {
            "raw": str(raw_out),
            "preprocessed": str(pre_out),
            "mask": str(mask_out),
            "overlay": str(overlay_out),
        },
    }
    meta_out.write_text(json.dumps(stats, indent=2))

    print(f"frame={frame_path.name}")
    print(f"stage={stage}")
    print(f"components={num_components}")
    print(f"foreground_px={area}")
    print(f"overlay={overlay_out}")
    print(f"mask={mask_out}")
    print(f"preprocessed={pre_out}")
    print(f"stats={meta_out}")


if __name__ == "__main__":
    main()
