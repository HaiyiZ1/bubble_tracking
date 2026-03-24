#!/usr/bin/env python3
"""
Relabel Air1 case datasets for publishing runs by discarding pre-bubble frames.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
import sys
from typing import Dict, List, Optional

import cv2

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bubble_tracking.frame_io import sorted_frame_paths
from bubble_tracking.publishing import PUBLISHING_CASES, get_publishing_case


def _frame_number(path: str) -> int:
    match = re.search(r"(\d+)\.(?:jpg|jpeg|png)$", Path(path).name, re.IGNORECASE)
    if not match:
        raise ValueError(f"Failed to parse frame number from {path}")
    return int(match.group(1))


def _selected_frame_map(frames_dir: Path, first_frame_number: int) -> Dict[int, str]:
    paths = sorted_frame_paths(str(frames_dir))
    mapping = {
        _frame_number(path): path
        for path in paths
        if _frame_number(path) >= int(first_frame_number)
    }
    if not mapping:
        raise ValueError(
            f"No frames found in {frames_dir} at or after frame {first_frame_number}"
        )
    return mapping


def _ensure_output_root_is_clean(output_root: Path) -> None:
    if not output_root.exists():
        return
    for subdir_name in ("frames", "frames_0p5"):
        subdir = output_root / subdir_name
        if subdir.exists() and any(subdir.iterdir()):
            raise FileExistsError(
                f"Output directory already contains files: {subdir}. "
                "Use a new destination root for relabeled publishing data."
            )


def _write_jpeg(image, output_path: Path, quality: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(output_path), image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError(f"Failed to write image: {output_path}")


def _read_image(path: str):
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    return image


def relabel_case(case_id: str, jpeg_quality: int) -> dict:
    case = get_publishing_case(case_id)
    source_root = case.source_root_path()
    output_root = case.relabeled_root_path()
    source_frames = source_root / "frames"
    source_frames_half = source_root / "frames_0p5"
    output_frames = output_root / "frames"
    output_frames_half = output_root / "frames_0p5"

    if not source_frames.exists():
        raise FileNotFoundError(f"Source frames dir not found: {source_frames}")

    _ensure_output_root_is_clean(output_root)

    selected_full = _selected_frame_map(source_frames, case.first_frame_number)
    selected_half: Optional[Dict[int, str]] = None
    if source_frames_half.exists():
        selected_half = _selected_frame_map(source_frames_half, case.first_frame_number)
        missing_half = sorted(set(selected_full) - set(selected_half))
        if missing_half:
            raise ValueError(
                f"frames_0p5 is missing relabeled source frames for {case_id}: "
                f"{missing_half[:10]}"
            )

    frame_map_rows: List[dict] = []
    for new_frame_number, old_frame_number in enumerate(sorted(selected_full), start=1):
        full_image = _read_image(selected_full[old_frame_number])
        full_out = output_frames / f"{new_frame_number:06d}.jpg"
        _write_jpeg(full_image, full_out, quality=jpeg_quality)

        if selected_half is not None:
            half_image = _read_image(selected_half[old_frame_number])
        else:
            h, w = full_image.shape[:2]
            half_image = cv2.resize(
                full_image,
                (max(1, w // 2), max(1, h // 2)),
                interpolation=cv2.INTER_AREA,
            )
        half_out = output_frames_half / f"{new_frame_number:06d}.jpg"
        _write_jpeg(half_image, half_out, quality=jpeg_quality)

        frame_map_rows.append(
            {
                "new_frame_number": int(new_frame_number),
                "new_frame_idx_zero_based": int(new_frame_number - 1),
                "old_frame_number": int(old_frame_number),
                "source_full_path": str(Path(selected_full[old_frame_number]).resolve()),
                "source_half_path": (
                    str(Path(selected_half[old_frame_number]).resolve())
                    if selected_half is not None
                    else ""
                ),
            }
        )

    output_root.mkdir(parents=True, exist_ok=True)
    frame_map_path = output_root / "frame_map.csv"
    with frame_map_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "new_frame_number",
                "new_frame_idx_zero_based",
                "old_frame_number",
                "source_full_path",
                "source_half_path",
            ],
        )
        writer.writeheader()
        writer.writerows(frame_map_rows)

    metadata = {
        "case_id": case.case_id,
        "title": case.title,
        "source_root": str(source_root.resolve()),
        "output_root": str(output_root.resolve()),
        "first_original_frame_number": int(case.first_frame_number),
        "num_output_frames": int(len(frame_map_rows)),
        "legacy_prompt_start_idx": int(case.legacy_prompt_start_idx),
        "relabeled_prompt_start_idx": int(case.relabeled_prompt_start_idx),
        "reference_frame_full": str((output_frames / "000001.jpg").resolve()),
        "reference_frame_half": str((output_frames_half / "000001.jpg").resolve()),
        "frame_map_csv": str(frame_map_path.resolve()),
    }
    metadata_path = output_root / "publishing_relabel_config.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Relabel Air1 case datasets for publishing runs")
    parser.add_argument(
        "--case",
        nargs="+",
        choices=sorted(PUBLISHING_CASES),
        default=sorted(PUBLISHING_CASES),
        help="Publishing case preset(s) to relabel",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality for written relabeled frames",
    )
    args = parser.parse_args()

    all_metadata = []
    for case_id in args.case:
        metadata = relabel_case(case_id=case_id, jpeg_quality=args.jpeg_quality)
        all_metadata.append(metadata)
        print(
            f"[{case_id}] wrote {metadata['num_output_frames']} frames to "
            f"{metadata['output_root']}"
        )
        print(
            f"[{case_id}] new pipeline reference: {metadata['reference_frame_half']}"
        )
        print(
            f"[{case_id}] adjusted --cv-keyframe-start-idx: "
            f"{metadata['relabeled_prompt_start_idx']}"
        )

    print(json.dumps({"cases": all_metadata}, indent=2))


if __name__ == "__main__":
    main()
