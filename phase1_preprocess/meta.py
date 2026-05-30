"""Load and interpret meta.yaml for the bubble-tracking pipeline.

The file's top level has a global `fps` and a `cases` dict. Each case has a
`scale: {mm, pixel}` block (calibration at full resolution) and an `airs`
dict where each air sub-case has a `start_frame` (1-indexed frame number
in the raw acquisition stream — used to relabel raw data so frame
start_frame becomes 000001).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass(frozen=True)
class CaseSpec:
    case: str            # e.g. "case1"
    air: str             # e.g. "air1"
    scale_mm: float      # cases.<case>.scale.mm
    scale_pixel: float   # cases.<case>.scale.pixel
    start_frame: int     # cases.<case>.airs.<air>.start_frame
    fps: float           # top-level fps

    def mm_per_pixel(self, downscale: float) -> float:
        """Effective mm/pixel after downscaling.

        Raw mm/pixel = scale_mm / scale_pixel.
        Downscaling by `downscale` (e.g. 0.5) multiplies mm/pixel by 1/downscale.
        """
        if downscale <= 0:
            raise ValueError(f"downscale must be positive, got {downscale}")
        return (self.scale_mm / self.scale_pixel) / downscale


def load_case(meta_path: Path, case_air: str) -> CaseSpec:
    """Load `cases.<case>.<airs>.<air>` plus top-level fps from meta.yaml.

    `case_air` uses dotted form `case1.air1`.
    """
    if "." not in case_air:
        raise ValueError(
            f"--case must be in the form '<case>.<air>' (e.g. 'case1.air1'), got {case_air!r}"
        )
    case_name, air_name = case_air.split(".", 1)

    data: Dict[str, Any] = yaml.safe_load(Path(meta_path).read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{meta_path}: top level must be a mapping")

    if "fps" not in data:
        raise KeyError(f"{meta_path}: missing top-level `fps` key")
    fps = float(data["fps"])

    cases = data.get("cases")
    if not isinstance(cases, dict) or case_name not in cases:
        raise KeyError(f"{meta_path}: cases.{case_name} not found")
    case_block = cases[case_name]

    scale = case_block.get("scale")
    if not isinstance(scale, dict) or "mm" not in scale or "pixel" not in scale:
        raise KeyError(f"{meta_path}: cases.{case_name}.scale must have mm and pixel")
    scale_mm = float(scale["mm"])
    scale_pixel = float(scale["pixel"])

    airs = case_block.get("airs")
    if not isinstance(airs, dict) or air_name not in airs:
        raise KeyError(f"{meta_path}: cases.{case_name}.airs.{air_name} not found")
    air_block = airs[air_name]

    if "start_frame" not in air_block:
        raise KeyError(
            f"{meta_path}: cases.{case_name}.airs.{air_name}.start_frame missing"
        )
    start_frame = int(air_block["start_frame"])

    return CaseSpec(
        case=case_name,
        air=air_name,
        scale_mm=scale_mm,
        scale_pixel=scale_pixel,
        start_frame=start_frame,
        fps=fps,
    )
