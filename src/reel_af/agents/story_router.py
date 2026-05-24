"""Story Router — picks the best architecture for THIS article, then runs it.

Routing rule (derived empirically from the stories comparison):

  counterintuitive / discovery / inspiration → I  (Hook-First + Pairwise)
                                                  ↳ wins on analytical / contrarian
                                                    / wonder content (B+H ensemble)

  breakdown / tutorial / explainer            → F  (Few-Shot Cloning)
                                                  ↳ wins on genre-bound content
                                                    where exemplars dominate

The router itself is one cheap direction-picker call (~10s), then the
appropriate full architecture. Net wall-time average: 80-120s with
better quality than any single architecture on its own.

Context strategy:
  IN  : ArticleSummary (no source body needed at this layer)
  OUT : ScriptOutput (the winning ReelDraft + audit trail)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from reel_af.agents.architectures import ArchOutput
from reel_af.agents.architectures.arch_f_fewshot import _pick_direction
from reel_af.agents.architectures.arch_f_fewshot import run as run_f
from reel_af.agents.architectures.arch_i_hybrid import run as run_i
from reel_af.agents.distiller import ArticleSummary
from reel_af.agents.reel_composer import Direction, ReelDraft

# Direction → architecture choice. The default lands on I (more general).
_ARCH_FOR_DIRECTION: dict[Direction, str] = {
    "counterintuitive": "I",
    "discovery":        "I",
    "inspiration":      "F",   # F clones the inspiration_story exemplar shape
    "breakdown":        "F",   # F clones the breakdown_news exemplar shape
    "tutorial":         "F",
    "explainer":        "F",
}


@dataclass
class RouterResult:
    """What the router returns to the caller (and the video pipeline)."""

    draft: ReelDraft
    direction: Direction
    chosen_arch: str          # "I" or "F" — for logging
    arch_output: ArchOutput   # full output including trace
    wall_time_s: float


async def route_and_run(app: Any, summary: ArticleSummary) -> RouterResult:
    t0 = time.time()

    # One cheap call to pick the direction.
    direction = await _pick_direction(app, summary)
    chosen = _ARCH_FOR_DIRECTION.get(direction, "I")

    if chosen == "F":
        arch_out = await run_f(app, summary)
    else:
        arch_out = await run_i(app, summary)

    # Force the direction on the resulting draft so downstream stages see the
    # routed direction (the underlying arch may pick a slightly different one).
    final_draft = arch_out.draft.model_copy(update={"direction": direction})

    return RouterResult(
        draft=final_draft,
        direction=direction,
        chosen_arch=chosen,
        arch_output=arch_out,
        wall_time_s=time.time() - t0,
    )
