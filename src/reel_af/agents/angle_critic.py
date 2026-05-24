"""Angle Critic — picks the winning angle from the proposer fan-out.

HUNT-PROVE pattern from CLAUDE.md: the proposers are the hunters (broad,
varied, biased). This critic is the prover — it reasons adversarially about
scroll-stop probability, explicitly down-weights clichés, and ignores the
proposers' self-scores (which are uncalibrated and cluster around 7-8).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from reel_af.models import AngleProposal, SourceContent

# Common patterns that show up in low-performing AI-generated hooks. The
# critic is explicitly told to penalise these in its system prompt so it
# doesn't rubber-stamp them. Editable as we learn what fails in production.
_CLICHE_PHRASES = (
    "you won't believe",
    "the secret is",
    "this changes everything",
    "in this video",
    "today we will",
    "let's talk about",
    "have you ever wondered",
    "did you know that",
    "i'll show you",
    "the truth about",
)


class _Verdict(BaseModel):
    chosen_index: int = Field(..., description="0-based index of the winning angle.")
    reasoning: str = Field(..., description="Why this one beats the others, in 2-3 sentences.")
    revised_hook: str = Field(
        ...,
        description=(
            "If the winning hook can be tightened (cut filler, sharpen verb), rewrite it "
            "here. Otherwise return an empty string."
        ),
    )


async def pick_angle(
    app: Any,
    angles: list[AngleProposal],
    source: SourceContent,
) -> AngleProposal:
    if not angles:
        raise RuntimeError("Critic: no angle proposals to choose from.")
    if len(angles) == 1:
        return angles[0]

    listing = "\n\n".join(
        f"[{i}] frame={a.frame}\n    hook: {a.hook_line!r}\n    angle: {a.angle}\n    self-score: {a.predicted_score}/10"
        for i, a in enumerate(angles)
    )

    system = (
        "You're a vertical-video editor whose only job is to predict which hook will "
        "stop someone scrolling TikTok / Reels / Shorts in <1 second.\n\n"
        "Hard rules:\n"
        "• Reject any hook that uses generic AI-content clichés ("
        + ", ".join(repr(p) for p in _CLICHE_PHRASES)
        + ").\n"
        "• Reject hooks longer than ~7 words — viewers don't read.\n"
        "• Reject hooks that explain instead of provoke.\n"
        "• Prefer hooks that contradict, surprise, or open a curiosity gap "
        "(viewer must watch the next 2 seconds to resolve).\n"
        "• Ignore the proposers' self-scores. They're consistently overconfident.\n\n"
        "Return the index of the strongest hook + a 2-3 sentence justification. "
        "Optionally tighten the chosen hook (only if it's clearly improvable)."
    )

    user = (
        f"Source title: {source.title}\n"
        f"Audience: {source.audience_hints}\n"
        f"Source surprise quality: {source.surprise_score}/10\n\n"
        f"CANDIDATES:\n{listing}"
    )

    verdict = await app.ai(system=system, user=user, schema=_Verdict)

    idx = max(0, min(verdict.chosen_index, len(angles) - 1))
    winner = angles[idx]
    if verdict.revised_hook and verdict.revised_hook.strip():
        winner = winner.model_copy(update={"hook_line": verdict.revised_hook.strip()})
    # Bake critic reasoning into why_works so downstream stages see it.
    winner = winner.model_copy(
        update={"why_works": f"{winner.why_works} | critic: {verdict.reasoning}"}
    )
    return winner
