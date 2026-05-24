"""Architecture E — Reverse (close-first).

Bet: most reels die at the close. A weak close kills rewatch/save behavior,
which is what the algorithm actually rewards. So build the close FIRST,
write the hook that sets up that close as a callback target, then bridge.

Stages:
  1. Generate 5 candidate closes (each 4-6 words, complete on its own)
  2. Pick the strongest close
  3. Generate the hook that this close will call back to
  4. Fill the middle that takes us from hook → close
"""

from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from reel_af.agents.architectures import ArchOutput
from reel_af.agents.creator_playbook import CLOSE_TRICKS, format_menu
from reel_af.agents.distiller import ArticleSummary
from reel_af.agents.reel_composer import (
    CloseTrickId,
    Direction,
    HookTrickId,
    ReelDraft,
    RetentionTrickId,
    VoiceTone,
)

NUM_CLOSES = 5


# ───── Step 1 — close candidates ─────────────────────────────────────


class _CloseCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    close_line: str = Field(..., description="The final spoken sentence (4-7 words).")
    close_trick: CloseTrickId


class _CloseBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    closes: list[_CloseCandidate] = Field(..., min_length=NUM_CLOSES, max_length=NUM_CLOSES)


def _close_gen_system() -> str:
    return f"""You are generating {NUM_CLOSES} CLOSING LINES for a vertical reel
about an article. ONLY the close — the final 4-7 spoken words. The hook
and body will be written later to set this up.

A great close does ONE of these jobs (pick varied across the 5):
  • loop_closure   — echoes a noun or verb from a hypothetical hook
  • cliffhanger    — leaves the thought unfinished, on a question
  • save_bait      — frames the reel as reference ("…next time you…")
  • comment_bait   — invites debate ("…would you?")
  • callback_punch — flips a word's meaning at the end

{format_menu(CLOSE_TRICKS, "CLOSE TRICKS")}

Each close must be IMMEDIATELY USABLE — not depend on context the viewer
doesn't have. Reference the article's actual content (names, numbers,
specific framings). NO generic ("And that's all.", "Hope this helped.").

Produce exactly {NUM_CLOSES}, each using a different trick."""


async def _gen_closes(app: Any, summary: ArticleSummary) -> list[_CloseCandidate]:
    user = (
        f"ARTICLE\n"
        f"  thesis  : {summary.one_line_thesis}\n"
        f"  takeaway: {summary.intended_takeaway}\n"
        f"  examples: {summary.concrete_examples}"
    )
    out = await app.ai(system=_close_gen_system(), user=user, schema=_CloseBatch)
    return out.closes


# ───── Step 2 — pick the best close ──────────────────────────────────


class _ClosePick(BaseModel):
    model_config = ConfigDict(extra="forbid")
    winner_index: int = Field(..., ge=0, le=NUM_CLOSES - 1)
    why: str = Field(..., description="1 sentence: what makes this close the strongest.")


_CLOSE_CRITIC_SYSTEM = """You are picking the strongest CLOSE for a vertical
reel. Strength = drives one of {rewatch, save, share, comment}.

Penalties:
  • Generic CTAs ("comment below", "save for later") that don't tie to content.
  • Trailing off / ellipsis / hedge.
  • Doesn't actually execute its declared trick.

Rewards:
  • Specific to the article — uses an actual word/concept from the source.
  • Open loop or callback that demands rewatch.
  • Last word is a strong noun or verb.

Pick exactly one winner."""


async def _pick_close(
    app: Any, closes: list[_CloseCandidate], summary: ArticleSummary
) -> tuple[int, str]:
    listing = "\n".join(
        f"  [{i}] trick={c.close_trick:20s} line={c.close_line!r}"
        for i, c in enumerate(closes)
    )
    user = f"ARTICLE THESIS: {summary.one_line_thesis}\n\nCLOSES:\n{listing}"
    p = await app.ai(system=_CLOSE_CRITIC_SYSTEM, user=user, schema=_ClosePick)
    return p.winner_index, p.why


# ───── Step 3 — hook that sets up the close ──────────────────────────


class _HookForClose(BaseModel):
    model_config = ConfigDict(extra="forbid")
    hook_line: str = Field(..., description="First 4-7 spoken words of the script.")
    hook_trick: HookTrickId
    direction: Direction
    why_pairs: str = Field(..., description="1 sentence: how this hook sets up the close.")


def _hook_for_close_system(close: _CloseCandidate) -> str:
    return f"""You are writing the OPENING HOOK for a vertical reel whose
CLOSE is already locked:

  close: {close.close_line!r}
  close_trick: {close.close_trick}

Write a hook (4-7 words) such that the close lands as a payoff / callback /
loop closure. If the close repeats a word, the hook must introduce that
word. If the close flips a meaning, the hook must establish the original
meaning. If the close asks a question, the hook should hint at a stance.

The hook must execute a named hook_trick (contradiction / number_shock /
mid_thought / direct_threat / loop_bait / name_drop / question_curiosity /
pattern_interrupt). Pick the one most natural for setting up this close.

Spoken English. First 3 words carry the load. No AI tells."""


async def _hook_for_close(
    app: Any, close: _CloseCandidate, summary: ArticleSummary
) -> _HookForClose:
    user = (
        f"ARTICLE\n"
        f"  thesis  : {summary.one_line_thesis}\n"
        f"  examples: {summary.concrete_examples}"
    )
    return await app.ai(system=_hook_for_close_system(close), user=user, schema=_HookForClose)


# ───── Step 4 — fill the middle ──────────────────────────────────────


def _middle_system(hook: _HookForClose, close: _CloseCandidate) -> str:
    return f"""You are filling the MIDDLE of a vertical-reel script. Hook and
close are locked:

  HOOK (must appear as opening line verbatim): {hook.hook_line!r}
  CLOSE (must appear as closing line verbatim): {close.close_line!r}

  hook_trick = {hook.hook_trick}
  close_trick = {close.close_trick}
  direction   = {hook.direction}

Write the body that bridges them. The full script (hook + body + close)
should be 58-72 words total (~22-27s spoken). The body must:
  • Pay off the hook's promise.
  • Set up the close so it lands as payoff/callback.
  • Use the article's actual examples — no invented metaphors.
  • Pick a retention_trick (open_loop / stakes_ladder / rehook_mid /
    promise_payoff / concrete_specifics) and visibly execute it in the body.

Return a ReelDraft. Set:
  direction       = {hook.direction}
  hook_trick      = {hook.hook_trick}
  close_trick     = {close.close_trick}
  retention_trick : your pick
  voice_tone      : your pick
  script          : "<hook>. <body>. <close>" as one paragraph.
  viral_score     : honest 1-10."""


async def _fill_middle(
    app: Any,
    hook: _HookForClose,
    close: _CloseCandidate,
    summary: ArticleSummary,
) -> ReelDraft:
    user = (
        f"ARTICLE\n"
        f"  thesis   : {summary.one_line_thesis}\n"
        f"  takeaway : {summary.intended_takeaway}\n"
        f"  examples : {summary.concrete_examples}\n"
        f"  key points:\n"
        + "\n".join(f"    {i+1}. {p}" for i, p in enumerate(summary.key_points))
    )
    return await app.ai(system=_middle_system(hook, close), user=user, schema=ReelDraft)


# ───── Public entrypoint ────────────────────────────────────────────


async def run(app: Any, summary: ArticleSummary) -> ArchOutput:
    t0 = time.time()
    trace: list[str] = []

    closes = await _gen_closes(app, summary)
    trace.append(f"generated {len(closes)} closes")
    for c in closes:
        trace.append(f"  - [{c.close_trick:18s}] {c.close_line!r}")

    winner_idx, why_close = await _pick_close(app, closes, summary)
    chosen_close = closes[winner_idx]
    trace.append(f"picked close[{winner_idx}]: {chosen_close.close_line!r}  ({why_close})")

    hook = await _hook_for_close(app, chosen_close, summary)
    trace.append(
        f"built hook for close: {hook.hook_line!r}  ({hook.hook_trick})  → {hook.why_pairs}"
    )

    full = await _fill_middle(app, hook, chosen_close, summary)
    trace.append(f"filled middle  composite={full.viral_score}/10")

    return ArchOutput(
        arch_id="E",
        arch_name="Reverse (close-first)",
        bet="Close drives rewatch/save — build it first, write hook to set it up.",
        draft=full,
        self_score=float(full.viral_score),
        wall_time_s=time.time() - t0,
        trace=trace,
    )
