"""Architecture H — Tournament (pairwise elimination).

Bet: pairwise judgments beat absolute scoring (Bradley-Terry preference
theory). A critic comparing two candidates side-by-side is more reliable
than the same critic putting an absolute 1-10 on each.

Stages:
  1. Generate 8 drafts (forced (direction, hook_trick) variety like arch A)
  2. Round of 8 → 4: four parallel pairwise critic calls
  3. Round of 4 → 2: two parallel pairwise critic calls
  4. Final: 2 → 1
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from reel_af.agents.architectures import ArchOutput
from reel_af.agents.distiller import ArticleSummary
from reel_af.agents.draft_pool import _draft_one, plan_pool, POOL_SIZE
from reel_af.agents.reel_composer import ReelDraft

NUM_DRAFTS = 8  # power of 2 for the bracket


class _Match(BaseModel):
    model_config = ConfigDict(extra="forbid")
    winner_index: int = Field(..., ge=0, le=1, description="0 = left, 1 = right.")
    why: str = Field(..., description="1 sentence: what beat what.")


_MATCH_SYSTEM = """You are a vertical-reel editor picking the BETTER of two
scripts. Pairwise comparison only — no absolute scoring.

Pick by, in this priority order:
  1. Hook strength (would a thumb stop in <1s?)
  2. Faithfulness to the source article
  3. Trope execution (declared tricks visibly present)
  4. Close strength (rewatch / save / comment trigger)
  5. Spoken flow

If they're genuinely tied, prefer the one with the more SPECIFIC content
(named entities, numbers, real examples beat abstractions)."""


async def _match(app: Any, left: ReelDraft, right: ReelDraft, summary: ArticleSummary) -> int:
    user = (
        f"ARTICLE THESIS: {summary.one_line_thesis}\n\n"
        f"=== LEFT [0] ===\n"
        f"direction={left.direction}  hook_trick={left.hook_trick}\n"
        f"{left.script}\n\n"
        f"=== RIGHT [1] ===\n"
        f"direction={right.direction}  hook_trick={right.hook_trick}\n"
        f"{right.script}"
    )
    m = await app.ai(system=_MATCH_SYSTEM, user=user, schema=_Match)
    return m.winner_index


async def _round(
    app: Any, contestants: list[ReelDraft], summary: ArticleSummary
) -> list[ReelDraft]:
    """One bracket round: pair adjacent contestants, run matches in parallel."""
    assert len(contestants) % 2 == 0, "round needs even number"
    pairs = list(zip(contestants[0::2], contestants[1::2]))
    winners_idx = await asyncio.gather(
        *(_match(app, left, right, summary) for left, right in pairs)
    )
    return [pair[w] for pair, w in zip(pairs, winners_idx)]


async def _generate_eight(app: Any, summary: ArticleSummary) -> list[ReelDraft]:
    """Plan pool size 4 (from draft_pool's planner), then draft 8 by running
    each slot twice in parallel for extra variety."""
    slots = await plan_pool(app, summary)
    # Draft each slot twice (parallel) → 8 total with paired styles.
    drafts = await asyncio.gather(
        *(_draft_one(app, summary, slot) for slot in slots * 2),
        return_exceptions=True,
    )
    return [d for d in drafts if isinstance(d, ReelDraft)]


async def run(app: Any, summary: ArticleSummary) -> ArchOutput:
    t0 = time.time()
    trace: list[str] = []

    drafts = await _generate_eight(app, summary)
    trace.append(f"generated {len(drafts)} drafts (pool plan × 2)")
    for d in drafts:
        trace.append(f"  - [{d.direction:18s} / {d.hook_trick:20s}]")

    # Pad to a power of 2 if some drafts failed.
    n = len(drafts)
    if n < 2:
        raise RuntimeError("arch_h: not enough drafts for a tournament.")
    # Trim to nearest power of 2 ≤ n.
    pow2 = 1
    while pow2 * 2 <= n:
        pow2 *= 2
    contestants = drafts[:pow2]
    trace.append(f"tournament bracket size: {pow2}")

    round_num = 0
    while len(contestants) > 1:
        round_num += 1
        contestants = await _round(app, contestants, summary)
        trace.append(f"round {round_num}: {len(contestants)} survivors")

    winner = contestants[0]
    trace.append(
        f"champion: direction={winner.direction}  hook_trick={winner.hook_trick}"
    )
    return ArchOutput(
        arch_id="H",
        arch_name="Tournament (pairwise elimination)",
        bet="Pairwise judgment beats absolute scoring — Bradley-Terry preference theory.",
        draft=winner,
        # Self-score is the model's own viral_score on the winner — pairwise
        # arch doesn't produce a composite, so use this as the surrogate.
        self_score=float(winner.viral_score),
        wall_time_s=time.time() - t0,
        trace=trace,
    )
