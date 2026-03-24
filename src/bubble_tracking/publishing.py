"""
Publishing-specific case presets and helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict


@dataclass(frozen=True)
class PublishingCase:
    case_id: str
    source_root: str
    relabeled_root: str
    title: str
    first_frame_number: int
    legacy_prompt_start_idx: int

    @property
    def relabeled_prompt_start_idx(self) -> int:
        return max(0, int(self.legacy_prompt_start_idx) - int(self.first_frame_number) + 1)

    def source_root_path(self) -> Path:
        return Path(self.source_root)

    def relabeled_root_path(self) -> Path:
        return Path(self.relabeled_root)


PUBLISHING_CASES: Dict[str, PublishingCase] = {
    "case1_air1": PublishingCase(
        case_id="case1_air1",
        source_root="data/bubbles",
        relabeled_root="data/bubbles/case1_air1_publish",
        title="High Porosity-Air1",
        first_frame_number=71,
        legacy_prompt_start_idx=80,
    ),
    "case2_air1": PublishingCase(
        case_id="case2_air1",
        source_root="data/bubbles/case2_air1",
        relabeled_root="data/bubbles/case2_air1_publish",
        title="Low Porosity-Air1",
        first_frame_number=127,
        legacy_prompt_start_idx=127,
    ),
    "case3_air1": PublishingCase(
        case_id="case3_air1",
        source_root="data/bubbles/case3_air1",
        relabeled_root="data/bubbles/case3_air1_publish",
        title="Medium Porosity-Air1",
        first_frame_number=215,
        legacy_prompt_start_idx=224,
    ),
}


def get_publishing_case(case_id: str) -> PublishingCase:
    try:
        return PUBLISHING_CASES[case_id]
    except KeyError as exc:
        valid = ", ".join(sorted(PUBLISHING_CASES))
        raise KeyError(f"Unknown publishing case '{case_id}'. Valid choices: {valid}") from exc
