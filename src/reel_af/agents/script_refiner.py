"""Script Refiner — surgical rewrite of weak phrases.

Takes the winning draft + the critic's per-dimension weak phrases and
proposed rewrites. Produces a refined script that:
  • Keeps everything the critic didn't flag (preserves the wins).
  • Replaces flagged phrases with the proposed rewrites (or better ones).
  • Maintains the declared tricks (hook, retention, close).

Context strategy:
  IN  : winning draft + critic's scorecard for it + the article summary
  OUT : a refined ReelDraft (same shape — refinement may bump self-scores)
  NOT : other drafts (irrelevant — winning draft is now ground truth)
"""

from __future__ import annotations

from typing import Any

from reel_af.agents.distiller import ArticleSummary
from reel_af.agents.reel_composer import ReelDraft
from reel_af.agents.script_critic import DraftScorecard


def _format_weakness_block(card: DraftScorecard) -> str:
    """Render the critic's feedback as actionable instructions for the refiner."""
    lines = []
    for dim_name, dim in [
        ("hook_strength", card.hook_strength),
        ("faithfulness", card.faithfulness),
        ("trope_execution", card.trope_execution),
        ("spoken_flow", card.spoken_flow),
        ("close_landing", card.close_landing),
    ]:
        if dim.score >= 8:
            continue
        lines.append(f"\n  {dim_name.upper()} (score {dim.score}/10)")
        lines.append(f"    why weak: {dim.reasoning}")
        for phrase in dim.weak_phrases:
            lines.append(f"    weak phrase: {phrase!r}")
        if dim.suggested_rewrite:
            lines.append(f"    suggested rewrite: {dim.suggested_rewrite}")
    if not lines:
        return "  (no weaknesses identified — all dims scored >= 8)"
    return "".join(lines)


_SYSTEM = """You're refining a vertical-reel script that scored well overall
but has identified weaknesses. You operate SURGICALLY — replace ONLY the
flagged phrases. Keep everything else exactly as it is. The script's
declared tricks (hook, retention, close) MUST stay intact.

Rules:
  • For each weak phrase the critic quoted, write a stronger replacement.
    Use the critic's suggested rewrite as a starting point, but you may
    write something better if you see one.
  • Don't rewrite the whole script. Don't add new sentences. Don't change
    structure. Don't change direction or declared tricks.
  • Preserve the article's faithfulness — don't introduce new examples or
    metaphors that aren't in the article summary.
  • Word count of the refined script must stay within 42-58 words.

You return a ReelDraft. All fields (direction, hook_trick, retention_trick,
close_trick, voice_tone) must match the input draft exactly — only the
script text and viral_score may change."""


async def refine(
    app: Any,
    draft: ReelDraft,
    scorecard: DraftScorecard,
    summary: ArticleSummary,
) -> ReelDraft:
    user = (
        f"WINNING DRAFT (refine in place):\n"
        f"  direction      : {draft.direction}\n"
        f"  hook_trick     : {draft.hook_trick}\n"
        f"  retention_trick: {draft.retention_trick}\n"
        f"  close_trick    : {draft.close_trick}\n"
        f"  tone           : {draft.voice_tone}\n"
        f"  script:\n{draft.script}\n\n"
        f"IDENTIFIED WEAKNESSES (critic feedback):"
        f"{_format_weakness_block(scorecard)}\n\n"
        f"ARTICLE SUMMARY (for faithfulness — don't drift):\n"
        f"  thesis  : {summary.one_line_thesis}\n"
        f"  examples: {summary.concrete_examples}"
    )
    refined = await app.ai(system=_SYSTEM, user=user, schema=ReelDraft)
    # Force-preserve the structural fields. The model occasionally drifts.
    return refined.model_copy(update={
        "direction": draft.direction,
        "hook_trick": draft.hook_trick,
        "retention_trick": draft.retention_trick,
        "close_trick": draft.close_trick,
        "voice_tone": draft.voice_tone,
    })
