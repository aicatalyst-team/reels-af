"""Architecture I — Hook-First Cascade + Pairwise Final Pick (B+H hybrid).

Best of B and H: generate 8 cheap hooks → critic picks top 2 → write 2 full
scripts (B's strength: hooks are pre-optimized) → pairwise critic picks
winner (H's strength: pairwise > absolute scoring).

One extra critic call vs B alone (~10-15s). Quality should be ≥ B on
borderline cases where the body-quality decides the winner.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from reel_af.agents.architectures import ArchOutput
from reel_af.agents.architectures.arch_b_hook_first import (
    NUM_FINALISTS,
    NUM_HOOKS,
    _body_from_hook,
    _gen_hooks,
    _rank_hooks,
)
from reel_af.agents.distiller import ArticleSummary
from reel_af.agents.reel_composer import ReelDraft


class _PairwiseMatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    winner_index: int = Field(..., ge=0, le=1)
    composite_score: float = Field(..., ge=1, le=10)
    why: str = Field(..., description="1-2 sentences: what beat what.")


_PAIRWISE_SYSTEM = """You are picking the BETTER of two complete vertical-reel
scripts. Pairwise comparison — not absolute scoring; compare them directly.

Priority order (apply in sequence; only proceed to the next tie-breaker if
the two are roughly even on the previous):

  1. Hook strength — would a thumb stop in <1 second?
  2. Faithfulness — uses the article's actual content (names, numbers)?
  3. Trope execution — declared tricks visibly present?
  4. Close strength — drives rewatch / save / comment?
  5. Spoken flow — varied sentence length, conversational rhythm?

Ties go to the more SPECIFIC script (named entities and numbers beat
abstractions).

Score the winner's composite 1-10."""


async def _pairwise_pick(
    app: Any, scripts: list[ReelDraft], summary: ArticleSummary
) -> tuple[int, float, str]:
    listing = "\n\n".join(
        f"=== SCRIPT [{i}] ===\n"
        f"direction={d.direction}  hook_trick={d.hook_trick}\n"
        f"retention_trick={d.retention_trick}  close_trick={d.close_trick}\n"
        f"\n{d.script}"
        for i, d in enumerate(scripts)
    )
    user = f"ARTICLE THESIS: {summary.one_line_thesis}\n\n{listing}"
    m = await app.ai(system=_PAIRWISE_SYSTEM, user=user, schema=_PairwiseMatch)
    return m.winner_index, m.composite_score, m.why


async def run(app: Any, summary: ArticleSummary) -> ArchOutput:
    t0 = time.time()
    trace: list[str] = []

    # 1. Hooks (8 candidates).
    hooks = await _gen_hooks(app, summary)
    trace.append(f"generated {len(hooks)} hooks")

    # 2. Rank.
    ranked = await _rank_hooks(app, hooks, summary)
    finalists = [hooks[r.hook_index] for r in ranked[:NUM_FINALISTS]]
    trace.append("top hooks chosen:")
    for r in ranked[:NUM_FINALISTS]:
        trace.append(f"  {r.score}/10  {hooks[r.hook_index].hook!r}")

    # 3. Bodies in parallel.
    bodies = await asyncio.gather(
        *(_body_from_hook(app, h, summary) for h in finalists),
        return_exceptions=True,
    )
    bodies = [b for b in bodies if isinstance(b, ReelDraft)]
    if not bodies:
        raise RuntimeError("arch_i: all body generations failed.")

    # 4. Pairwise pick (H-style) instead of absolute scoring.
    if len(bodies) == 1:
        winner_idx, score, why = 0, float(bodies[0].viral_score), "only one survived"
    else:
        winner_idx, score, why = await _pairwise_pick(app, bodies, summary)
    trace.append(f"pairwise winner: script[{winner_idx}]  composite={score:.1f}  ({why})")

    return ArchOutput(
        arch_id="I",
        arch_name="Hook-First + Pairwise Final (B+H hybrid)",
        bet="Best of B and H: pre-optimize hook, then pairwise-judge the bodies.",
        draft=bodies[winner_idx],
        self_score=score,
        wall_time_s=time.time() - t0,
        trace=trace,
    )
