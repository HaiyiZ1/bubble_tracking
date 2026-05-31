"""Read and write `<run-dir>/run_config.json`.

This is the contract between Phase 1 and Phases 2/3. Phase 1 writes the
full file (with `phase1_done: true`). Phases 2 and 3 update their own
`phaseN_done` flag on completion.

Moved from phase1_preprocess/run_config.py.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class RunConfig:
    case: str               # "case1.air1"
    run_dir: str            # "outputs/case1_air1_publish"
    meta_path: str          # "meta.yaml"
    fps: float
    mm_per_pixel: float
    downscale: float
    start_frame: int
    relabel_skipped: bool   # True when input was already relabeled
    frames: str             # path to phase1/frames_0p5
    reference: str          # path to phase1/frames_0p5/000001.jpg
    roi_json: str
    roi_mask: str
    phase1_done: bool = False
    phase2_done: bool = False
    phase3_done: bool = False

    def path(self) -> Path:
        return Path(self.run_dir) / "run_config.json"

    def write(self) -> None:
        p = self.path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(self), indent=2) + "\n")


def load(run_dir: str | Path) -> RunConfig:
    p = Path(run_dir) / "run_config.json"
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. Run `python -m phase1_preprocess` first."
        )
    data = json.loads(p.read_text())
    return RunConfig(**data)


def mark_phase_done(run_dir: str | Path, phase: int) -> None:
    """Update phase{phase}_done in run_config.json (used by Phase 2/3)."""
    if phase not in (1, 2, 3):
        raise ValueError(f"phase must be 1, 2, or 3; got {phase}")
    cfg = load(run_dir)
    setattr(cfg, f"phase{phase}_done", True)
    cfg.write()
