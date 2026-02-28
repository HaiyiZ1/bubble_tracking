#!/usr/bin/env python3
"""
Generate SAM2 prompt YAML from a single frame using simple blob detection.

Example:
  python scripts/generate_auto_prompts.py \
      --frames data/interim/bubbles_air1_072_172/frames \
      --output data/interim/bubbles_air1_072_172/prompts_auto.yaml \
      --frame-idx 0
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import yaml
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def load_frame(frames_dir: str, frame_idx: int):
    from bubble_tracking.frame_io import sorted_frame_paths

    frame_paths = sorted_frame_paths(frames_dir)
    if not frame_paths:
        raise ValueError(f"No frames found in {frames_dir}")
    if frame_idx < 0 or frame_idx >= len(frame_paths):
        raise IndexError(f"frame_idx {frame_idx} out of range (0..{len(frame_paths)-1})")
    frame_path = frame_paths[frame_idx]
    frame = cv2.imread(frame_path)
    if frame is None:
        raise ValueError(f"Cannot read frame: {frame_path}")
    return frame, frame_path


def build_subset(source_dir: Path, out_dir: Path, start: int, end: int) -> None:
    import re
    import shutil

    pattern = re.compile(r"\((\d+)\)\.(jpg|jpeg|png)$", re.IGNORECASE)
    idx_to_file = {}
    for f in os.listdir(source_dir):
        m = pattern.search(f)
        if m:
            idx_to_file[int(m.group(1))] = f

    out_dir.mkdir(parents=True, exist_ok=True)
    map_lines = []
    missing = []
    for new_idx, orig_idx in enumerate(range(start, end + 1)):
        fname = idx_to_file.get(orig_idx)
        if not fname:
            missing.append(orig_idx)
            continue
        src_path = source_dir / fname
        dst_path = out_dir / f"{new_idx:06d}.jpg"
        shutil.copy2(src_path, dst_path)
        map_lines.append(f"{new_idx},{orig_idx},{fname}")

    (out_dir.parent / "map.csv").write_text(
        "new_idx,orig_idx,orig_name\n" + "\n".join(map_lines)
    )
    if missing:
        print(f"Warning: missing source frames: {missing[:10]} (count={len(missing)})", file=sys.stderr)


def connected_components(binary, min_area, max_area):
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    components = []
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area or area > max_area:
            continue
        x, y, w, h, _ = stats[i]
        cx, cy = centroids[i]
        components.append({
            "area": int(area),
            "bbox": (int(x), int(y), int(x + w), int(y + h)),
            "centroid": (int(cx), int(cy)),
        })
    return components


def choose_binary(gray, min_area, max_area):
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, th_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    comps = connected_components(th, min_area, max_area)
    comps_inv = connected_components(th_inv, min_area, max_area)

    # Prefer the variant with more valid components (but not empty).
    if comps_inv and (len(comps_inv) >= len(comps)):
        return th_inv, comps_inv, True
    if comps:
        return th, comps, False
    # Fallback if both are empty
    return th_inv, comps_inv, True


def main():
    parser = argparse.ArgumentParser(description="Generate auto-prompts from a frame")
    parser.add_argument("--frames", required=True, help="Directory of frames (numeric names)")
    parser.add_argument("--output", required=True, help="Output YAML path")
    parser.add_argument("--frame-idx", type=int, default=0, help="Frame index in sorted order")
    parser.add_argument(
        "--search-first",
        action="store_true",
        help="Search forward from --frame-idx until at least one prompt is found",
    )
    parser.add_argument(
        "--max-search-frames",
        type=int,
        default=200,
        help="Max number of frames to search when --search-first is enabled",
    )
    parser.add_argument("--min-area", type=int, default=200, help="Minimum component area")
    parser.add_argument(
        "--intensity-min",
        type=int,
        default=None,
        help="Optional min grayscale intensity to keep (0-255)",
    )
    parser.add_argument(
        "--intensity-max",
        type=int,
        default=None,
        help="Optional max grayscale intensity to keep (0-255)",
    )
    parser.add_argument(
        "--max-area-ratio",
        type=float,
        default=0.2,
        help="Max area as fraction of image area",
    )
    parser.add_argument(
        "--roi-x-center",
        type=float,
        default=0.5,
        help="ROI center x as fraction of width (0-1)",
    )
    parser.add_argument(
        "--roi-x-width",
        type=float,
        default=1.0,
        help="ROI width as fraction of width (0-1)",
    )
    parser.add_argument(
        "--roi-y-bottom",
        type=float,
        default=1.0,
        help="ROI height from bottom as fraction of height (0-1)",
    )
    parser.add_argument(
        "--roi-file",
        type=str,
        default=None,
        help="Optional ROI JSON file with x1,y1,x2,y2",
    )
    parser.add_argument(
        "--roi-mask",
        type=str,
        default=None,
        help="Optional ROI mask PNG (nonzero = keep)",
    )
    parser.add_argument(
        "--rebuild-subset",
        action="store_true",
        help="Rebuild frames subset into --frames from a source dir",
    )
    parser.add_argument(
        "--source-frames",
        type=str,
        default=None,
        help="Source frames directory (e.g., data/bubbles/Air1)",
    )
    parser.add_argument(
        "--subset-start",
        type=int,
        default=None,
        help="Start index in source frames (e.g., 72)",
    )
    parser.add_argument(
        "--subset-end",
        type=int,
        default=None,
        help="End index in source frames (e.g., 271)",
    )
    parser.add_argument(
        "--debug-image",
        default=None,
        help="Optional path to save debug overlay image",
    )
    parser.add_argument(
        "--debug-dir",
        default=None,
        help="Optional directory to save step-by-step debug images",
    )
    args = parser.parse_args()

    if args.rebuild_subset:
        if not args.source_frames or args.subset_start is None or args.subset_end is None:
            raise ValueError("--rebuild-subset requires --source-frames, --subset-start, --subset-end")
        build_subset(
            source_dir=Path(args.source_frames),
            out_dir=Path(args.frames),
            start=args.subset_start,
            end=args.subset_end,
        )

    from bubble_tracking.frame_io import sorted_frame_paths

    frame_paths = sorted_frame_paths(args.frames)
    if not frame_paths:
        raise ValueError(f"No frames found in {args.frames}")

    def process_frame(frame_idx: int, write_debug: bool):
        frame, frame_path = load_frame(args.frames, frame_idx)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        if args.roi_mask:
            roi_mask = cv2.imread(args.roi_mask, cv2.IMREAD_GRAYSCALE)
            if roi_mask is None:
                raise ValueError(f"Failed to read roi mask: {args.roi_mask}")
            if roi_mask.shape[:2] != gray.shape[:2]:
                raise ValueError("ROI mask size does not match frame size")
            roi_x1, roi_y1, roi_x2, roi_y2 = 0, 0, w, h
        elif args.roi_file:
            import json
            roi_data = json.loads(Path(args.roi_file).read_text())
            roi_x1 = int(roi_data["x1"])
            roi_y1 = int(roi_data["y1"])
            roi_x2 = int(roi_data["x2"])
            roi_y2 = int(roi_data["y2"])
            roi_mask = np.zeros_like(gray, dtype=np.uint8)
            roi_mask[roi_y1:roi_y2, roi_x1:roi_x2] = 255
        else:
            roi_half_w = max(1, int((args.roi_x_width * w) / 2))
            roi_cx = int(args.roi_x_center * w)
            roi_x1 = max(0, roi_cx - roi_half_w)
            roi_x2 = min(w, roi_cx + roi_half_w)
            roi_y1 = max(0, int((1.0 - args.roi_y_bottom) * h))
            roi_y2 = h
            roi_mask = np.zeros_like(gray, dtype=np.uint8)
            roi_mask[roi_y1:roi_y2, roi_x1:roi_x2] = 255

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        masked = blurred.copy()

        # Mask-out outside ROI before any thresholding
        masked[roi_mask == 0] = 255

        # Contrast stretch within ROI
        roi_vals = masked[roi_mask == 1]
        if roi_vals.size > 0:
            vmin = int(roi_vals.min())
            vmax = int(roi_vals.max())
            if vmax > vmin:
                scaled = (masked.astype(np.float32) - vmin) * (255.0 / (vmax - vmin))
                scaled = np.clip(scaled, 0, 255).astype(np.uint8)
                masked = scaled

        if args.intensity_min is not None or args.intensity_max is not None:
            lo = 0 if args.intensity_min is None else args.intensity_min
            hi = 255 if args.intensity_max is None else args.intensity_max
            keep = (masked >= lo) & (masked <= hi)
            masked[~keep] = 255

        max_area = int(gray.shape[0] * gray.shape[1] * args.max_area_ratio)
        _, components, inverted = choose_binary(masked, args.min_area, max_area)

        otsu_start = None
        if write_debug and args.debug_dir:
            os.makedirs(args.debug_dir, exist_ok=True)
            cv2.imwrite(os.path.join(args.debug_dir, "01_gray.jpg"), gray)
            cv2.imwrite(os.path.join(args.debug_dir, "02_blurred.jpg"), blurred)
            cv2.imwrite(os.path.join(args.debug_dir, "03_roi_masked.jpg"), masked)
            th_src = masked
            otsu_start = 4
            if args.intensity_min is not None or args.intensity_max is not None:
                cv2.imwrite(os.path.join(args.debug_dir, "04_intensity_mask.jpg"), masked)
                otsu_start = 5
            _, th = cv2.threshold(th_src, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            _, th_inv = cv2.threshold(th_src, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            cv2.imwrite(os.path.join(args.debug_dir, f"{otsu_start:02d}_otsu.jpg"), th)
            cv2.imwrite(os.path.join(args.debug_dir, f"{otsu_start+1:02d}_otsu_inv.jpg"), th_inv)

        return frame, frame_path, gray, components, inverted, (roi_x1, roi_y1, roi_x2, roi_y2), otsu_start

    if args.search_first:
        start = args.frame_idx
        end = min(len(frame_paths), start + args.max_search_frames)
        found = False
        for idx in range(start, end):
            frame, frame_path, gray, components, inverted, roi, otsu_start = process_frame(idx, False)
            if components:
                args.frame_idx = idx
                found = True
                break
        if not found:
            print("Warning: no components found in search window.", file=sys.stderr)
        frame, frame_path, gray, components, inverted, roi, otsu_start = process_frame(args.frame_idx, True)
    else:
        frame, frame_path, gray, components, inverted, roi, otsu_start = process_frame(args.frame_idx, True)
        if not components:
            print("Warning: no components found with current thresholds.", file=sys.stderr)

    # Build prompts
    from bubble_tracking.schemas import PromptSpec, ObjectPrompt, PointPrompt

    prompts = []
    for obj_id, comp in enumerate(components, start=1):
        cx, cy = comp["centroid"]
        prompts.append(ObjectPrompt(
            object_id=obj_id,
            frame_idx=args.frame_idx,
            points=[PointPrompt(x=cx, y=cy, label=1)],
        ))

    spec = PromptSpec(prompts=prompts)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        yaml.safe_dump(spec.to_dict(), f, default_flow_style=False)

    roi_x1, roi_y1, roi_x2, roi_y2 = roi
    h, w = gray.shape[:2]
    print(f"Frame: {frame_path}")
    print(f"Components kept: {len(components)} (inverted={inverted})")
    print(f"ROI: x=[{roi_x1},{roi_x2}] y=[{roi_y1},{roi_y2}] (w={w}, h={h})")
    print(f"Prompts saved: {args.output}")

    if args.debug_dir:
        dbg_all = frame.copy()
        for comp in components:
            x1, y1, x2, y2 = comp["bbox"]
            cv2.rectangle(dbg_all, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cx, cy = comp["centroid"]
            cv2.circle(dbg_all, (cx, cy), 4, (0, 0, 255), -1)
        comp_idx = (otsu_start or 4) + 2
        cv2.imwrite(os.path.join(args.debug_dir, f"{comp_idx:02d}_components_kept.jpg"), dbg_all)

        dbg_roi = frame.copy()
        cv2.rectangle(dbg_roi, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 255), 2)
        for comp in components:
            x1, y1, x2, y2 = comp["bbox"]
            cv2.rectangle(dbg_roi, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cx, cy = comp["centroid"]
            cv2.circle(dbg_roi, (cx, cy), 4, (0, 0, 255), -1)
        cv2.imwrite(os.path.join(args.debug_dir, f"{comp_idx+1:02d}_roi_prompts.jpg"), dbg_roi)

    if args.debug_image:
        dbg = frame.copy()
        for comp in components:
            x1, y1, x2, y2 = comp["bbox"]
            cv2.rectangle(dbg, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cx, cy = comp["centroid"]
            cv2.circle(dbg, (cx, cy), 4, (0, 0, 255), -1)
        cv2.rectangle(dbg, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 255), 2)
        os.makedirs(os.path.dirname(args.debug_image), exist_ok=True)
        cv2.imwrite(args.debug_image, dbg)
        print(f"Debug image saved: {args.debug_image}")


if __name__ == "__main__":
    main()
