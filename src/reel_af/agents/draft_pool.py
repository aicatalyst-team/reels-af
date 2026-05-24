"""Draft Pool — generates 4 parallel script candidates with FORCED variety.

Each drafter is locked into a different (direction, hook_trick) combo so the
pool actually explores the angle space instead of converging. The planner
picks which 4 combos to try based on the article's domain (a tutorial-shaped
article won't try `counterintuitive`; a science finding won't try `tutorial`).

Context strategy:
  IN  : ArticleSummary + the assigned (direction, hook_trick)
  OUT : one ScriptDraft with declared tricks
  NOT : other drafts (no cross-contamination), article body (filtered already)
"""

from __future__ import annotations

import asyncio
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from reel_af.agents.creator_playbook import (
    CLOSE_TRICKS,
    HOOK_TRICKS,
    RETENTION_TRICKS,
    format_menu,
)
from reel_af.agents.distiller import ArticleSummary
from reel_af.agents.reel_composer import (
    CloseTrickId,
    Direction,
    HookTrickId,
    ReelDraft,
    RetentionTrickId,
    VoiceTone,
)

# How many candidates to generate. 4 is the sweet spot — enough variety to
# pick from, few enough to fan-out cheaply.
POOL_SIZE = 4


# ───────────────────────────────────────────────────────────────────────
# Direction planner — picks N (direction, hook_trick) combos for the pool.
# ───────────────────────────────────────────────────────────────────────


class _DraftSlot(BaseModel):
    model_config = ConfigDict(extra="forbid")
    direction: Direction
    hook_trick: HookTrickId
    rationale: str = Field(
        ...,
        description="One sentence: why this combo could work for this article.",
    )


class _DraftPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    slots: list[_DraftSlot] = Field(..., min_length=POOL_SIZE, max_length=POOL_SIZE)


_PLAN_SYSTEM = f"""You're picking {POOL_SIZE} DISTINCT (direction, hook_trick)
combinations to try for this article. The reel composer will write one
script per combination in parallel; a critic will pick the best.

DIRECTIONS:
  explainer        — teach how something works
  discovery        — surprising fact / finding
  counterintuitive — flip a common belief
  tutorial         — numbered steps to do something
  inspiration      — someone did X — you can too
  breakdown        — explain a recent event/situation

{format_menu(HOOK_TRICKS, "HOOK TRICKS")}

RULES:
  • All {POOL_SIZE} slots must use DIFFERENT direction OR different hook_trick.
    Ideally both. Pure duplicates wasted.
  • Pick combinations that ACTUALLY fit the article's domain and content.
    Don't propose `tutorial` for an essay; don't propose `discovery` for an
    opinion piece. Mismatched combos waste a slot.
  • Include ONE risky / unusual combo. If 3 obvious choices feel safe,
    burn the 4th slot on something the article could surprisingly support.
  • One-sentence rationale per slot — why this combo could land here.
"""


async def plan_pool(app: Any, summary: ArticleSummary) -> list[_DraftSlot]:
    user = (
        f"ARTICLE\n"
        f"  domain   : {summary.domain}\n"
        f"  thesis   : {summary.one_line_thesis}\n"
        f"  takeaway : {summary.intended_takeaway}\n\n"
        f"  key points:\n"
        + "\n".join(f"    {i+1}. {p}" for i, p in enumerate(summary.key_points))
        + (
            f"\n\n  concrete examples:\n"
            + "\n".join(f"    - {e}" for e in summary.concrete_examples)
            if summary.concrete_examples
            else ""
        )
    )
    plan = await app.ai(system=_PLAN_SYSTEM, user=user, schema=_DraftPlan)
    return plan.slots


# ───────────────────────────────────────────────────────────────────────
# Per-slot drafter — generates one ReelDraft with a LOCKED hook trick.
# ───────────────────────────────────────────────────────────────────────


def _drafter_system(direction: Direction, hook_trick: HookTrickId) -> str:
    hook_desc = HOOK_TRICKS[hook_trick]
    retention_menu = format_menu(RETENTION_TRICKS, "RETENTION TRICKS")
    close_menu = format_menu(CLOSE_TRICKS, "CLOSE TRICKS")
    return f"""You write ONE vertical-reel script for a faithful condensation
of an article. You have been ASSIGNED:

  direction  = {direction}
  hook_trick = {hook_trick}

Use this hook technique. Description:

{hook_desc}

The FIRST SENTENCE of your script must execute this exact hook trick.
You may not pick a different one.

You pick your own retention_trick and close_trick from the menus below:

{retention_menu}

{close_menu}

You also pick voice_tone and score viral_score yourself.

CONTENT RULES (non-negotiable):
  • FAITHFUL to the article — use its actual examples, names, numbers from
    the summary's "concrete examples" list. NEVER invent metaphors not
    present in the source.
  • Spoken English. Varied sentence length. Active voice.
  • 42-58 words total. One paragraph. Em-dashes for one dramatic pause.
  • No AI tells ("Did you know", "In this video", "Let's talk about").
  • Second person where natural.

You return a ReelDraft. The script field must visibly execute the assigned
hook_trick in sentence 1, your chosen retention_trick in the body, and your
chosen close_trick in the last sentence."""


async def _draft_one(
    app: Any,
    summary: ArticleSummary,
    slot: _DraftSlot,
) -> ReelDraft:
    system = _drafter_system(slot.direction, slot.hook_trick)
    user = (
        f"ARTICLE SUMMARY\n"
        f"  domain   : {summary.domain}\n"
        f"  thesis   : {summary.one_line_thesis}\n"
        f"  takeaway : {summary.intended_takeaway}\n\n"
        f"  key points:\n"
        + "\n".join(f"    {i+1}. {p}" for i, p in enumerate(summary.key_points))
        + "\n\n  concrete examples (use these — don't invent new ones):\n"
        + (
            "\n".join(f"    - {e}" for e in summary.concrete_examples)
            if summary.concrete_examples
            else "    (none — keep abstract-but-faithful)"
        )
    )
    draft = await app.ai(system=system, user=user, schema=ReelDraft)
    # Force the assigned direction + hook trick — the model sometimes
    # overrides what was instructed. Lock them post-hoc.
    return draft.model_copy(update={
        "direction": slot.direction,
        "hook_trick": slot.hook_trick,
    })


# ───────────────────────────────────────────────────────────────────────
# Public entrypoint — plan + parallel draft.
# ───────────────────────────────────────────────────────────────────────


async def generate_draft_pool(
    app: Any,
    summary: ArticleSummary,
) -> list[ReelDraft]:
    """Plan 4 distinct (direction, hook) combos, then draft them in parallel."""
    slots = await plan_pool(app, summary)
    drafts = await asyncio.gather(
        *(_draft_one(app, summary, s) for s in slots),
        return_exceptions=True,
    )
    # Drop failures rather than aborting — critic only needs 2+ to compare.
    return [d for d in drafts if isinstance(d, ReelDraft)]
