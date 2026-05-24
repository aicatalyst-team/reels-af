"""Bounded refinement loop — pool → critic → refine → critic → ship.

Encapsulates the HUNT-PROVE pattern for script generation. Drafts are
generated in parallel (hunters), critic scores all (prover), winner is
refined surgically, critic re-scores; on second-pass failure the loop
either does ONE more refinement OR ships with a low-score warning.

The hard caps prevent runaway cost — at most 2 refinement passes total.

Returns the winning ReelDraft + the final scorecard so the caller can
log / surface viral_score, which trick beat which, etc.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from reel_af.agents.distiller import ArticleSummary
from reel_af.agents.draft_pool import generate_draft_pool
from reel_af.agents.reel_composer import ReelDraft
from reel_af.agents.script_critic import (
    ACCEPT_THRESHOLD,
    DraftScorecard,
    composite_score,
    critique,
)
from reel_af.agents.script_refiner import refine

MAX_REFINE_PASSES = 2


@dataclass
class LoopResult:
    """What the loop returns to the caller."""

    draft: ReelDraft
    scorecard: DraftScorecard
    composite: float
    pool_size: int
    refine_passes: int
    pool_history: list[tuple[ReelDraft, DraftScorecard, float]]  # for debug / surfacing


async def run_script_loop(app: Any, summary: ArticleSummary) -> LoopResult:
    """Generate a pool, critique, refine until acceptable or capped."""

    # ── Round 0 — parallel draft pool ────────────────────────────
    drafts = await generate_draft_pool(app, summary)
    if not drafts:
        raise RuntimeError("script_loop: empty draft pool — nothing to critique.")

    verdict = await critique(app, drafts, summary)
    history: list[tuple[ReelDraft, DraftScorecard, float]] = []
    for card in verdict.scorecards:
        history.append((drafts[card.draft_index], card, composite_score(card)))

    # Pick the highest composite (don't trust the model's winner_index alone).
    history.sort(key=lambda t: -t[2])
    best_draft, best_card, best_score = history[0]

    # ── Refinement passes — bounded ──────────────────────────────
    passes = 0
    while best_score < ACCEPT_THRESHOLD and passes < MAX_REFINE_PASSES:
        refined = await refine(app, best_draft, best_card, summary)
        # Re-critique JUST the refined draft against the same article.
        verdict2 = await critique(app, [refined], summary)
        refined_card = verdict2.scorecards[0]
        refined_card = refined_card.model_copy(update={"draft_index": 0})
        refined_score = composite_score(refined_card)
        passes += 1

        # If refinement made it worse, revert. Refiner sometimes oversteps.
        if refined_score >= best_score:
            best_draft, best_card, best_score = refined, refined_card, refined_score
            history.append((refined, refined_card, refined_score))
        else:
            # Stop — refinement isn't helping.
            break

    return LoopResult(
        draft=best_draft,
        scorecard=best_card,
        composite=best_score,
        pool_size=len(drafts),
        refine_passes=passes,
        pool_history=history,
    )
