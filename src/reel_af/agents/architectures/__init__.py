"""Story-generation architectures — each takes (app, summary) → ArchOutput.

Run them all on the same article via `reel-af stories URL` to compare.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from reel_af.agents.distiller import ArticleSummary
from reel_af.agents.reel_composer import ReelDraft


@dataclass
class ArchOutput:
    """Uniform return shape across architectures — apples-to-apples comparison."""

    arch_id: str
    arch_name: str
    bet: str                                 # one-line "what this arch is betting on"
    draft: ReelDraft                         # winning script + declared tricks
    self_score: float                        # arch's own composite, 1-10
    wall_time_s: float
    trace: list[str] = field(default_factory=list)   # per-step audit lines

    def script(self) -> str:
        return self.draft.script

    def word_count(self) -> int:
        return len(self.draft.script.split())
