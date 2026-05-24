"""Architecture J — Adversarial Debate (Writer vs Skeptic, judge picks).

Bet: focused iteration with adversarial pressure refines a single draft
lineage more than parallel candidates can. The skeptic is forced to find
SPECIFIC weaknesses (quoted phrases). The writer must defend or address
each one in the rewrite.

Stages:
  v0 = Writer A drafts a complete script.
  v1 = Skeptic identifies N weaknesses → Writer A produces v1.
  v2 = Skeptic re-attacks → Writer A produces v2 (final).
  Judge picks BEST of {v0, v1, v2} — refinement isn't always improvement.

Hard cap: 3 versions. Each version costs ~1 call from writer + 1 from
skeptic, so the whole arch is ~6 calls total (~80-120s).

Why this might win:
  • Parallel architectures (B, H, I) can't address SPECIFIC weaknesses —
    they can only pick between candidates. Debate can fix them.
  • The Judge step (J's twist) means a bad refinement doesn't sink the run —
    the original wins by default.
"""

from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from reel_af.agents.architectures import ArchOutput
from reel_af.agents.distiller import ArticleSummary
from reel_af.agents.reel_composer import ReelDraft

MAX_DEBATE_ROUNDS = 2  # → 3 total versions (v0, v1, v2)


# ───── Skeptic — finds SPECIFIC weaknesses ──────────────────────────


class _Weakness(BaseModel):
    model_config = ConfigDict(extra="forbid")
    weak_phrase: str = Field(
        ...,
        description=(
            "EXACT phrase from the draft that is weak. Verbatim — the writer "
            "will substring-search to find and replace it."
        ),
    )
    why_weak: str = Field(..., description="1 sentence: what's wrong with this phrase.")
    suggested_fix: str = Field(
        ..., description="A concrete replacement for the weak phrase."
    )


class _SkepticVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    is_strong_enough: bool = Field(
        ...,
        description=(
            "True if the draft is genuinely ship-ready (no concrete weaknesses "
            "found). The skeptic should set this to False if there's anything "
            "actionable to improve."
        ),
    )
    weaknesses: list[_Weakness] = Field(
        ...,
        min_length=0,
        max_length=5,
        description=(
            "0-5 SPECIFIC weaknesses. Empty list ONLY if is_strong_enough=True. "
            "Otherwise list at least 2 — the skeptic's job is to find problems."
        ),
    )
    overall_critique: str = Field(
        ...,
        description="2-3 sentences: holistic read on what's failing.",
    )


_SKEPTIC_SYSTEM = """You are the SKEPTIC in a debate about a vertical-reel
script. Your job is to find SPECIFIC, ACTIONABLE weaknesses. You are not
nice. You don't write essays about how the draft is "decent" — you quote
the exact phrases that are weak and propose fixes.

WHAT COUNTS AS A WEAKNESS:
  • Generic openers ("Did you know", "In this video", "Today we…")
  • Hook that explains instead of provokes
  • Body that doesn't pay off the hook
  • Invented metaphors not in the source article
  • Long compound sentences (robotic in spoken English)
  • Weak close that hedges or trails off
  • Declared trick (e.g., loop_closure) that isn't actually executed
  • Generic claims when the source has specific examples available

RULES:
  • Quote the SPECIFIC phrase that's weak. Verbatim. The writer needs to
    substring-search and replace it.
  • Propose a concrete fix. "Make it punchier" is not feedback.
  • Find at least 2 weaknesses unless the script is genuinely strong.
  • If you really can't find anything actionable, set is_strong_enough=True
    and explain why. But don't rubber-stamp — that defeats the debate."""


async def _skeptic_pass(
    app: Any, draft: ReelDraft, summary: ArticleSummary
) -> _SkepticVerdict:
    user = (
        f"ARTICLE SUMMARY (for faithfulness check):\n"
        f"  thesis  : {summary.one_line_thesis}\n"
        f"  examples: {summary.concrete_examples}\n\n"
        f"DRAFT TO ATTACK:\n"
        f"  direction      : {draft.direction}\n"
        f"  hook_trick     : {draft.hook_trick}\n"
        f"  retention_trick: {draft.retention_trick}\n"
        f"  close_trick    : {draft.close_trick}\n"
        f"  script:\n{draft.script}"
    )
    return await app.ai(system=_SKEPTIC_SYSTEM, user=user, schema=_SkepticVerdict)


# ───── Writer — drafts v0, then refines with skeptic's feedback ─────


_WRITER_V0_SYSTEM = """You write the FIRST DRAFT of a vertical-reel script
for a faithful condensation of an article. This will go through adversarial
debate — a skeptic will attack it, you'll get a chance to revise.

Don't hold back. Write the strongest possible v0 you can.

RULES:
  • 58-72 words (~22-27s spoken). One continuous paragraph.
  • Spoken English. Varied sentence length. Active voice. Second person.
  • Use the article's actual examples — names, numbers — from the summary's
    concrete examples list. Don't invent metaphors not in the source.
  • Em-dashes (—) for one dramatic pause. Commas for breath.
  • No AI tells ("Did you know", "In this video").
  • Pick direction, hook_trick, retention_trick, close_trick, voice_tone
    that fit the article. Declare them so the skeptic can verify they were
    executed.

You're writing for a vertical scroll feed. First sentence has to stop a
thumb in <1 second."""


_WRITER_REFINE_SYSTEM_TEMPLATE = """You are REVISING a vertical-reel script
the skeptic has attacked. For each weakness, either:
  (a) Replace the quoted weak phrase with the suggested fix (or a better
      rewrite of your own).
  (b) If the skeptic is wrong, defend by writing a stronger replacement
      that addresses the underlying concern in a different way.

DO NOT discard the parts the skeptic didn't attack. Preserve them.
DO NOT change the declared tricks unless a weakness REQUIRES it.
KEEP the script at 58-72 words.

SKEPTIC'S VERDICT:
  overall: {overall}

WEAKNESSES TO ADDRESS (quoted phrases + suggested fixes):
{weakness_block}

Return a refined ReelDraft. Keep direction, hook_trick, retention_trick,
close_trick, voice_tone unchanged unless explicitly invalidated."""


def _format_weaknesses(v: _SkepticVerdict) -> str:
    if not v.weaknesses:
        return "  (skeptic found no specific weaknesses)"
    lines = []
    for i, w in enumerate(v.weaknesses):
        lines.append(f"  [{i+1}] weak: {w.weak_phrase!r}")
        lines.append(f"      why : {w.why_weak}")
        lines.append(f"      fix : {w.suggested_fix}")
    return "\n".join(lines)


async def _writer_v0(app: Any, summary: ArticleSummary) -> ReelDraft:
    user = (
        f"ARTICLE\n"
        f"  domain   : {summary.domain}\n"
        f"  thesis   : {summary.one_line_thesis}\n"
        f"  takeaway : {summary.intended_takeaway}\n"
        f"  examples : {summary.concrete_examples}\n"
        f"  key points:\n"
        + "\n".join(f"    {i+1}. {p}" for i, p in enumerate(summary.key_points))
    )
    return await app.ai(system=_WRITER_V0_SYSTEM, user=user, schema=ReelDraft)


async def _writer_refine(
    app: Any,
    prior: ReelDraft,
    verdict: _SkepticVerdict,
    summary: ArticleSummary,
) -> ReelDraft:
    system = _WRITER_REFINE_SYSTEM_TEMPLATE.format(
        overall=verdict.overall_critique,
        weakness_block=_format_weaknesses(verdict),
    )
    user = (
        f"PRIOR DRAFT (refine in place):\n"
        f"  direction      : {prior.direction}\n"
        f"  hook_trick     : {prior.hook_trick}\n"
        f"  retention_trick: {prior.retention_trick}\n"
        f"  close_trick    : {prior.close_trick}\n"
        f"  voice_tone     : {prior.voice_tone}\n"
        f"  script:\n{prior.script}\n\n"
        f"ARTICLE EXAMPLES (don't drift):\n"
        f"  {summary.concrete_examples}"
    )
    refined = await app.ai(system=system, user=user, schema=ReelDraft)
    # Preserve structural declarations unless the model deliberately changed
    # them in response to a weakness flagged on those fields.
    return refined.model_copy(update={
        "direction": prior.direction,
        "hook_trick": prior.hook_trick,
        "retention_trick": prior.retention_trick,
        "close_trick": prior.close_trick,
        "voice_tone": prior.voice_tone,
    })


# ───── Judge — picks the best of all versions ───────────────────────


class _JudgeVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")
    winner_version: int = Field(
        ..., ge=0, description="0-based index of the best version."
    )
    composite_score: float = Field(..., ge=1, le=10)
    why: str = Field(..., description="1-2 sentences: what made this version win.")


_JUDGE_SYSTEM = """You are the JUDGE deciding which version of a script is
best after a writer/skeptic debate. You see ALL versions (v0, v1, v2…) and
pick the strongest, by composite quality:

  hook (30%) > faithfulness (20%) > tropes (20%) > flow (15%) > close (15%)

CRITICAL: refinement isn't always improvement. Sometimes a writer
over-corrects and v2 is worse than v0. Pick honestly — the original wins
by default if nothing later actually beat it.

Score the winner's composite 1-10."""


async def _judge(
    app: Any, versions: list[ReelDraft], summary: ArticleSummary
) -> tuple[int, float, str]:
    listing = "\n\n".join(
        f"=== v{i} ===\n"
        f"direction={d.direction}  hook={d.hook_trick}\n"
        f"retention={d.retention_trick}  close={d.close_trick}\n"
        f"\n{d.script}"
        for i, d in enumerate(versions)
    )
    user = f"ARTICLE THESIS: {summary.one_line_thesis}\n\n{listing}"
    v = await app.ai(system=_JUDGE_SYSTEM, user=user, schema=_JudgeVerdict)
    idx = max(0, min(v.winner_version, len(versions) - 1))
    return idx, v.composite_score, v.why


# ───── Public entrypoint ────────────────────────────────────────────


async def run(app: Any, summary: ArticleSummary) -> ArchOutput:
    t0 = time.time()
    trace: list[str] = []
    versions: list[ReelDraft] = []

    # v0 — initial draft.
    v0 = await _writer_v0(app, summary)
    versions.append(v0)
    trace.append(f"v0: direction={v0.direction} hook={v0.hook_trick} (writer's first cut)")

    # Debate rounds.
    current = v0
    for round_num in range(1, MAX_DEBATE_ROUNDS + 1):
        verdict = await _skeptic_pass(app, current, summary)
        if verdict.is_strong_enough:
            trace.append(f"round {round_num}: skeptic concedes — draft is strong enough.")
            break
        trace.append(
            f"round {round_num}: skeptic found {len(verdict.weaknesses)} weakness(es)"
        )
        for w in verdict.weaknesses[:3]:
            trace.append(f"  - {w.weak_phrase!r} → {w.suggested_fix!r}")
        refined = await _writer_refine(app, current, verdict, summary)
        versions.append(refined)
        trace.append(f"v{round_num}: writer revised")
        current = refined

    # Judge picks the best across all versions.
    if len(versions) == 1:
        winner_idx, score, why = 0, float(versions[0].viral_score), "only one version"
    else:
        winner_idx, score, why = await _judge(app, versions, summary)
    trace.append(
        f"judge picked v{winner_idx} of {len(versions)}  composite={score:.1f}  ({why})"
    )

    return ArchOutput(
        arch_id="J",
        arch_name="Adversarial Debate (Writer vs Skeptic + Judge)",
        bet="Iteration on a single lineage with concrete weaknesses beats parallel candidates.",
        draft=versions[winner_idx],
        self_score=score,
        wall_time_s=time.time() - t0,
        trace=trace,
    )
