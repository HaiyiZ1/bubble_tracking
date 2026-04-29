#!/usr/bin/env python3
"""
Round-trip utilities for case1 relabeled frame 0234 manual mask editing.

Workflow:
1) export: convert ID mask PNG into two full-frame binary CSV matrices for IDs #48 and #49.
2) import: rebuild edited ID mask from the two edited binary CSV matrices.

Import preserves numbering by design:
- IDs other than 48/49 remain unchanged.
- Pixels set in edited #48/#49 matrices update only the editable region (background/48/49).
- Overlap policy: #49 wins.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


MANUAL_DIR = Path("outputs/case1_air1_publish_original_0304_relabeled_0234_manual_change")
DEFAULT_BASE_MASK = MANUAL_DIR / "000233_manual_mask.png"
DEFAULT_ID48_CSV = MANUAL_DIR / "000233_manual_mask_id48_matrix.csv"
DEFAULT_ID49_CSV = MANUAL_DIR / "000233_manual_mask_id49_matrix.csv"
DEFAULT_META_JSON = MANUAL_DIR / "000233_manual_mask_matrix_meta.json"
DEFAULT_EDITED_MASK = MANUAL_DIR / "000233_manual_mask_edited.png"
DEFAULT_ID48_VOID_FILLED_MASK = MANUAL_DIR / "000233_manual_mask_id48_void_filled.png"
DEFAULT_ID49_CROP_CSV = MANUAL_DIR / "000233_manual_mask_id49_crop_10pct_before_manual_change.csv"
DEFAULT_ID49_CROP_META_JSON = MANUAL_DIR / "000233_manual_mask_id49_crop_10pct_meta.json"


def _load_mask(mask_path: Path) -> np.ndarray:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Failed to read mask: {mask_path}")
    if mask.ndim != 2:
        raise ValueError(f"Mask must be single-channel, got shape={mask.shape}")
    if mask.dtype != np.uint16:
        # Preserve existing values but normalize storage to uint16.
        mask = mask.astype(np.uint16)
    return mask


def _write_binary_matrix_csv(matrix: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        for row in matrix:
            writer.writerow(row.astype(np.uint8).tolist())


def _fill_background_holes_inside_id(mask: np.ndarray, target_id: int) -> np.ndarray:
    target = mask == np.uint16(target_id)
    if not np.any(target):
        return np.zeros_like(target, dtype=bool)

    inv = np.logical_not(target).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(inv, connectivity=8)
    if num_labels <= 1:
        return np.zeros_like(target, dtype=bool)

    border_labels = np.unique(
        np.concatenate([labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]])
    )
    border_region = np.isin(labels, border_labels)
    holes_any = np.logical_and(inv == 1, np.logical_not(border_region))
    holes_background = np.logical_and(holes_any, mask == 0)
    return holes_background


def _read_binary_matrix_csv(csv_path: Path, expected_shape: Tuple[int, int]) -> np.ndarray:
    h, w = expected_shape
    out = np.zeros((h, w), dtype=np.uint8)

    row_idx = 0
    extra_rows = 0
    row_length_mismatch = 0
    missing_cells = 0
    extra_cells = 0
    empty_cells = 0
    nan_cells = 0
    non_int_cells = 0
    non_binary_cells = 0

    with csv_path.open("r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if row_idx >= h:
                extra_rows += 1
                row_idx += 1
                continue

            if len(row) != w:
                row_length_mismatch += 1
                if len(row) < w:
                    missing_cells += w - len(row)
                else:
                    extra_cells += len(row) - w

            cols = min(len(row), w)
            for col_idx in range(cols):
                token = row[col_idx].strip()
                if token == "":
                    empty_cells += 1
                    continue
                if token.lower() == "nan":
                    nan_cells += 1
                    continue
                try:
                    value = int(token)
                except ValueError:
                    non_int_cells += 1
                    continue
                if value not in (0, 1):
                    non_binary_cells += 1
                    continue
                out[row_idx, col_idx] = np.uint8(value)

            row_idx += 1

    missing_rows = max(0, h - row_idx)

    has_error = any(
        [
            extra_rows > 0,
            missing_rows > 0,
            row_length_mismatch > 0,
            missing_cells > 0,
            extra_cells > 0,
            empty_cells > 0,
            nan_cells > 0,
            non_int_cells > 0,
            non_binary_cells > 0,
        ]
    )

    if has_error:
        raise ValueError(
            "Invalid binary matrix CSV "
            f"({csv_path}). Errors: extra_rows={extra_rows}, missing_rows={missing_rows}, "
            f"row_length_mismatch={row_length_mismatch}, missing_cells={missing_cells}, "
            f"extra_cells={extra_cells}, empty_cells={empty_cells}, nan_cells={nan_cells}, "
            f"non_int_cells={non_int_cells}, non_binary_cells={non_binary_cells}"
        )

    return out


def export_matrices(base_mask_path: Path, id48_csv: Path, id49_csv: Path, meta_json: Path) -> None:
    mask = _load_mask(base_mask_path)
    id48 = (mask == 48).astype(np.uint8)
    id49 = (mask == 49).astype(np.uint8)

    _write_binary_matrix_csv(id48, id48_csv)
    _write_binary_matrix_csv(id49, id49_csv)

    meta = {
        "base_mask_path": str(base_mask_path),
        "shape": [int(mask.shape[0]), int(mask.shape[1])],
        "dtype": str(mask.dtype),
        "id_values_present": [int(v) for v in np.unique(mask).tolist()],
        "editable_ids": {"id48_csv": 48, "id49_csv": 49},
        "csv_format": {"header": False, "delimiter": ",", "allowed_values": [0, 1]},
    }
    meta_json.parent.mkdir(parents=True, exist_ok=True)
    meta_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Wrote: {id48_csv}")
    print(f"Wrote: {id49_csv}")
    print(f"Wrote: {meta_json}")
    print(f"shape={mask.shape[0]}x{mask.shape[1]}")


def import_matrices(
    base_mask_path: Path,
    id48_csv: Path,
    id49_csv: Path,
    output_mask_path: Path,
) -> None:
    base = _load_mask(base_mask_path)
    shape = (int(base.shape[0]), int(base.shape[1]))
    m48 = _read_binary_matrix_csv(id48_csv, shape)
    m49 = _read_binary_matrix_csv(id49_csv, shape)

    editable_region = np.isin(base, np.array([0, 48, 49], dtype=base.dtype))
    paint48 = m48.astype(bool)
    paint49 = m49.astype(bool)

    conflict48 = np.logical_and(paint48, np.logical_not(editable_region))
    conflict49 = np.logical_and(paint49, np.logical_not(editable_region))
    conflict_any = np.logical_or(conflict48, conflict49)
    if np.any(conflict_any):
        blocked_ids = [int(v) for v in np.unique(base[conflict_any]).tolist() if int(v) not in (0, 48, 49)]
        raise ValueError(
            "Edited matrices paint onto protected non-48/49 object pixels. "
            f"conflict_pixels_total={int(np.count_nonzero(conflict_any))}, "
            f"conflict_from_id48={int(np.count_nonzero(conflict48))}, "
            f"conflict_from_id49={int(np.count_nonzero(conflict49))}, "
            f"blocked_ids_sample={blocked_ids[:10]}"
        )

    edited = base.copy()
    edited[(base == 48) | (base == 49)] = 0
    edited[paint48] = np.uint16(48)
    edited[paint49] = np.uint16(49)  # #49 wins on overlap by assignment order.

    output_mask_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_mask_path), edited.astype(np.uint16)):
        raise RuntimeError(f"Failed to write edited mask: {output_mask_path}")

    print(f"Wrote: {output_mask_path}")
    print(f"shape={edited.shape[0]}x{edited.shape[1]}")
    print(f"ids_present={sorted(int(v) for v in np.unique(edited).tolist())}")
    print(f"pixels_id48={int(np.count_nonzero(edited == 48))}")
    print(f"pixels_id49={int(np.count_nonzero(edited == 49))}")


def _expanded_crop_bounds(
    y_min: int,
    y_max: int,
    x_min: int,
    x_max: int,
    shape: Tuple[int, int],
    expand_ratio: float,
) -> Tuple[int, int, int, int]:
    h, w = shape
    box_h = y_max - y_min + 1
    box_w = x_max - x_min + 1

    pad_y = int(math.ceil(float(box_h) * float(expand_ratio)))
    pad_x = int(math.ceil(float(box_w) * float(expand_ratio)))

    row_start = max(0, y_min - pad_y)
    row_end_exclusive = min(h, y_max + pad_y + 1)
    col_start = max(0, x_min - pad_x)
    col_end_exclusive = min(w, x_max + pad_x + 1)
    return row_start, row_end_exclusive, col_start, col_end_exclusive


def prepare_compact_matrices(
    base_mask_path: Path,
    output_mask_path: Path,
    id48_crop_filled_target: int,
    id49_target: int,
    id49_crop_csv_path: Path,
    id49_crop_meta_path: Path,
    expand_ratio: float,
) -> None:
    if expand_ratio < 0:
        raise ValueError(f"expand_ratio must be >= 0, got {expand_ratio}")

    base = _load_mask(base_mask_path)
    filled = base.copy()

    holes = _fill_background_holes_inside_id(base, id48_crop_filled_target)
    filled[holes] = np.uint16(id48_crop_filled_target)

    output_mask_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_mask_path), filled.astype(np.uint16)):
        raise RuntimeError(f"Failed to write mask: {output_mask_path}")

    ys, xs = np.where(filled == np.uint16(id49_target))
    if xs.size == 0:
        raise ValueError(f"ID {id49_target} not found in mask after #48 hole fill: {base_mask_path}")

    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    row_start, row_end_exclusive, col_start, col_end_exclusive = _expanded_crop_bounds(
        y_min=y_min,
        y_max=y_max,
        x_min=x_min,
        x_max=x_max,
        shape=(int(filled.shape[0]), int(filled.shape[1])),
        expand_ratio=expand_ratio,
    )

    crop49 = (
        filled[row_start:row_end_exclusive, col_start:col_end_exclusive] == np.uint16(id49_target)
    ).astype(np.uint8)
    _write_binary_matrix_csv(crop49, id49_crop_csv_path)

    meta = {
        "base_mask_path": str(base_mask_path),
        "output_mask_path": str(output_mask_path),
        "frame_shape": [int(filled.shape[0]), int(filled.shape[1])],
        "dtype": str(filled.dtype),
        "id48_target": int(id48_crop_filled_target),
        "id49_target": int(id49_target),
        "id48_hole_fill_pixels_added": int(np.count_nonzero(holes)),
        "expand_ratio_each_side": float(expand_ratio),
        "id49_bbox_in_full_frame": {
            "row_min": y_min,
            "row_max": y_max,
            "col_min": x_min,
            "col_max": x_max,
            "height": y_max - y_min + 1,
            "width": x_max - x_min + 1,
        },
        "id49_crop_in_full_frame": {
            "row_start": row_start,
            "row_end_exclusive": row_end_exclusive,
            "col_start": col_start,
            "col_end_exclusive": col_end_exclusive,
            "height": row_end_exclusive - row_start,
            "width": col_end_exclusive - col_start,
        },
        "id49_crop_csv_path": str(id49_crop_csv_path),
        "id49_crop_csv_format": {"header": False, "delimiter": ",", "allowed_values": [0, 1]},
    }
    id49_crop_meta_path.parent.mkdir(parents=True, exist_ok=True)
    id49_crop_meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Wrote: {output_mask_path}")
    print(f"Wrote: {id49_crop_csv_path}")
    print(f"Wrote: {id49_crop_meta_path}")
    print(f"id48_hole_fill_pixels_added={int(np.count_nonzero(holes))}")
    print(
        "id49_crop_row_col="
        f"row_start:{row_start},row_end_exclusive:{row_end_exclusive},"
        f"col_start:{col_start},col_end_exclusive:{col_end_exclusive}"
    )
    print(f"id49_crop_shape={crop49.shape[0]}x{crop49.shape[1]}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Round-trip case1 frame 0234 mask to/from editable binary matrices")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_export = sub.add_parser("export", help="Export id48/id49 full-frame binary matrices from base mask")
    p_export.add_argument("--base-mask", default=str(DEFAULT_BASE_MASK), help="Base uint16 ID mask PNG")
    p_export.add_argument("--id48-csv", default=str(DEFAULT_ID48_CSV), help="Output CSV for ID #48 (0/1)")
    p_export.add_argument("--id49-csv", default=str(DEFAULT_ID49_CSV), help="Output CSV for ID #49 (0/1)")
    p_export.add_argument("--meta-json", default=str(DEFAULT_META_JSON), help="Output metadata JSON")

    p_import = sub.add_parser("import", help="Import edited id48/id49 matrices and rebuild edited ID mask")
    p_import.add_argument("--base-mask", default=str(DEFAULT_BASE_MASK), help="Base uint16 ID mask PNG")
    p_import.add_argument("--id48-csv", default=str(DEFAULT_ID48_CSV), help="Edited CSV for ID #48 (0/1)")
    p_import.add_argument("--id49-csv", default=str(DEFAULT_ID49_CSV), help="Edited CSV for ID #49 (0/1)")
    p_import.add_argument("--output-mask", default=str(DEFAULT_EDITED_MASK), help="Output edited uint16 ID mask PNG")

    p_prepare_compact = sub.add_parser(
        "prepare-compact",
        help="Fill enclosed #48 holes and export a 10%%-expanded cropped #49 binary matrix with crop offsets",
    )
    p_prepare_compact.add_argument("--base-mask", default=str(DEFAULT_BASE_MASK), help="Base uint16 ID mask PNG")
    p_prepare_compact.add_argument(
        "--output-mask",
        default=str(DEFAULT_ID48_VOID_FILLED_MASK),
        help="Output mask PNG after enclosed #48 hole fill",
    )
    p_prepare_compact.add_argument(
        "--id49-crop-csv",
        default=str(DEFAULT_ID49_CROP_CSV),
        help="Output cropped CSV matrix for ID #49 (0/1)",
    )
    p_prepare_compact.add_argument(
        "--id49-crop-meta",
        default=str(DEFAULT_ID49_CROP_META_JSON),
        help="Output metadata JSON containing crop row/column offsets",
    )
    p_prepare_compact.add_argument(
        "--expand-ratio",
        type=float,
        default=0.10,
        help="Expansion ratio added on each side of #49 bbox for the crop",
    )
    p_prepare_compact.add_argument("--id48", type=int, default=48, help="ID used for hole filling target")
    p_prepare_compact.add_argument("--id49", type=int, default=49, help="ID used for cropped export target")

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.cmd == "export":
        export_matrices(
            base_mask_path=Path(args.base_mask),
            id48_csv=Path(args.id48_csv),
            id49_csv=Path(args.id49_csv),
            meta_json=Path(args.meta_json),
        )
        return

    if args.cmd == "import":
        import_matrices(
            base_mask_path=Path(args.base_mask),
            id48_csv=Path(args.id48_csv),
            id49_csv=Path(args.id49_csv),
            output_mask_path=Path(args.output_mask),
        )
        return

    if args.cmd == "prepare-compact":
        prepare_compact_matrices(
            base_mask_path=Path(args.base_mask),
            output_mask_path=Path(args.output_mask),
            id48_crop_filled_target=int(args.id48),
            id49_target=int(args.id49),
            id49_crop_csv_path=Path(args.id49_crop_csv),
            id49_crop_meta_path=Path(args.id49_crop_meta),
            expand_ratio=float(args.expand_ratio),
        )
        return

    raise ValueError(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
