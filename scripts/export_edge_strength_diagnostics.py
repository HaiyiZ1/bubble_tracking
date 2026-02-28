#!/usr/bin/env python3
"""
Export edge-strength diagnostics for selected frames.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bubble_tracking.flow_tracking import PreprocessConfig, SegmentationConfig, preprocess_frame
from bubble_tracking.frame_io import load_mask


def _parse_ranges(ranges: str) -> List[int]:
    frames: List[int] = []
    for part in ranges.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_str, end_str = part.split("-", 1)
            start = int(start_str)
            end = int(end_str)
            if end < start:
                start, end = end, start
            frames.extend(range(start, end + 1))
        else:
            frames.append(int(part))
    return sorted(set(frames))


def _load_config(events_path: Optional[str]) -> Tuple[PreprocessConfig, SegmentationConfig]:
    if not events_path:
        return PreprocessConfig(), SegmentationConfig()
    data = json.loads(Path(events_path).read_text())
    cfg = data.get("config", {})
    preprocess_cfg = cfg.get("preprocess", {})
    segment_cfg = cfg.get("segment", {})
    return PreprocessConfig(**preprocess_cfg), SegmentationConfig(**segment_cfg)


def _apply_roi(frame: np.ndarray, roi: Optional[dict]) -> np.ndarray:
    if roi is None:
        return frame
    return frame[roi["y1"]:roi["y2"], roi["x1"]:roi["x2"]]


def _boundary_ring(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask > 0
    ksize = radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    dilated = cv2.dilate(mask, kernel)
    eroded = cv2.erode(mask, kernel)
    return (dilated > 0) & (eroded == 0)


def _overlay_boundary_heat(
    base: np.ndarray,
    ring: np.ndarray,
    heat: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    output = base.copy()
    if not np.any(ring):
        return output
    blended = (1 - alpha) * output[ring] + alpha * heat[ring]
    output[ring] = blended.astype(np.uint8)
    return output


def _component_centroid(mask: np.ndarray) -> Tuple[int, int]:
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return 0, 0
    return int(xs.mean()), int(ys.mean())


def main() -> None:
    parser = argparse.ArgumentParser(description="Export edge-strength diagnostics")
    parser.add_argument("--frames", required=True, help="Frames directory")
    parser.add_argument("--reference", required=True, help="Reference image path")
    parser.add_argument("--roi", default=None, help="ROI JSON path")
    parser.add_argument("--masks", required=True, help="Masks directory from a run")
    parser.add_argument("--events", default=None, help="events.json from a run")
    parser.add_argument("--out-dir", required=True, help="Output directory for diagnostics")
    parser.add_argument(
        "--ranges",
        default="21-41,180-200",
        help="Frame ranges (e.g., '21-41,180-200')",
    )
    parser.add_argument("--alpha", type=float, default=0.4, help="Overlay alpha")
    parser.add_argument(
        "--edge-boundary-radius",
        type=int,
        default=None,
        help="Override boundary radius (default from config)",
    )
    args = parser.parse_args()

    roi = None
    if args.roi:
        roi_data = json.loads(Path(args.roi).read_text())
        roi = {
            "x1": int(roi_data["x1"]),
            "y1": int(roi_data["y1"]),
            "x2": int(roi_data["x2"]),
            "y2": int(roi_data["y2"]),
        }

    preprocess_cfg, segment_cfg = _load_config(args.events)
    if args.edge_boundary_radius is not None:
        segment_cfg.edge_boundary_radius = int(args.edge_boundary_radius)

    frame_indices = _parse_ranges(args.ranges)
    frames_dir = Path(args.frames)
    masks_dir = Path(args.masks)
    out_dir = Path(args.out_dir)
    overlay_dir = out_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    reference = cv2.imread(str(args.reference))
    if reference is None:
        raise ValueError(f"Failed to read reference image: {args.reference}")

    rows: List[Dict[str, object]] = []

    for frame_idx in frame_indices:
        frame_path = frames_dir / f"{frame_idx:06d}.jpg"
        mask_path = masks_dir / f"{frame_idx:06d}.png"
        if not frame_path.exists() or not mask_path.exists():
            continue

        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue
        mask = load_mask(str(mask_path))
        if mask is None:
            continue

        frame_roi = _apply_roi(frame, roi)
        ref_roi = _apply_roi(reference, roi)
        mask_roi = _apply_roi(mask, roi)

        normalized = preprocess_frame(frame_roi, ref_roi, preprocess_cfg)
        grad_x = cv2.Sobel(normalized, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(normalized, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = cv2.magnitude(grad_x, grad_y)
        grad_norm = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX)
        heat = cv2.applyColorMap(grad_norm.astype(np.uint8), cv2.COLORMAP_JET)

        overlay = frame_roi.copy()
        object_ids = [int(x) for x in np.unique(mask_roi) if x > 0]
        for obj_id in object_ids:
            obj_mask = (mask_roi == obj_id).astype(np.uint8)
            ring = _boundary_ring(obj_mask, segment_cfg.edge_boundary_radius)
            if np.any(ring):
                overlay = _overlay_boundary_heat(overlay, ring, heat, alpha=args.alpha)
                values = grad_mag[ring]
                edge_mean = float(values.mean())
                edge_median = float(np.median(values))
                edge_p10 = float(np.percentile(values, 10))
                edge_p90 = float(np.percentile(values, 90))
                edge_min = float(values.min())
                edge_max = float(values.max())
                boundary_px = int(values.size)
            else:
                edge_mean = edge_median = edge_p10 = edge_p90 = edge_min = edge_max = 0.0
                boundary_px = 0

            area_px = int(obj_mask.sum())
            cx, cy = _component_centroid(obj_mask)
            label = f"#{obj_id} {edge_median:.1f}"
            cv2.putText(
                overlay,
                label,
                (cx + 5, cy - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            rows.append({
                "frame_idx": frame_idx,
                "object_id": obj_id,
                "area_px": area_px,
                "boundary_px": boundary_px,
                "edge_mean": edge_mean,
                "edge_median": edge_median,
                "edge_p10": edge_p10,
                "edge_p90": edge_p90,
                "edge_min": edge_min,
                "edge_max": edge_max,
            })

        out_path = overlay_dir / f"{frame_idx:06d}.jpg"
        cv2.imwrite(str(out_path), overlay)

    csv_path = out_dir / "edge_strength.csv"
    json_path = out_dir / "edge_strength.json"
    out_dir.mkdir(parents=True, exist_ok=True)

    if rows:
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        json_path.write_text(json.dumps(rows, indent=2))

    meta = {
        "ranges": args.ranges,
        "frames_dir": str(frames_dir),
        "masks_dir": str(masks_dir),
        "roi": roi,
        "preprocess_mode": preprocess_cfg.mode,
        "edge_boundary_radius": segment_cfg.edge_boundary_radius,
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote overlays to {overlay_dir}")
    print(f"Wrote CSV to {csv_path}")
    print(f"Wrote JSON to {json_path}")


if __name__ == "__main__":
    main()
