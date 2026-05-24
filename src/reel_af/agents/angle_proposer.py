"""Angle Proposers — 5 parallel .ai() agents propose distinct viral angles.

Each angle frame is its own small structured-output call so we can run all
5 concurrently with asyncio.gather. The Archei "parallel hunters" pattern
applied to ideation: each hunter is forced to commit to a different angle
template, then a downstream critic chooses among them. This produces more
varied output than a single "give me 5 angles" call which tends to converge.
"""

from __future__ import annotations

import asyncio
from typing import Any

from pydantic import BaseModel, Field

from reel_af.models import AngleFrame, AngleProposal, SourceContent


class _SingleAngle(BaseModel):
    """What each proposer returns. Flat schema, .ai()-shaped."""

    hook_line: str = Field(..., description="Literal first words spoken on screen. <10 words.")
    angle: str = Field(..., description="One-sentence framing of the take.")
    why_works: str = Field(..., description="Why this would stop a thumb on a vertical feed.")
    predicted_score: int = Field(..., ge=1, le=10)


# Per-frame system prompts. Each one biases the model toward a different
# rhetorical move. Kept terse so the agent has room to fill the flat schema.
_PROMPTS: dict[AngleFrame, str] = {
    "contrarian": (
        "You write the CONTRARIAN angle: the take that goes against the conventional "
        "wisdom in the source. Find what most people would say, then say the opposite "
        "(but defensible). Hook line must signal contradiction in the first 3 words."
    ),
    "didnt_know": (
        "You write the 'I BET YOU DIDN'T KNOW' angle: find the most surprising "
        "fact / detail in the source that a smart person would still get wrong. "
        "Hook line frames it as a knowledge gap the viewer wants closed."
    ),
    "personal": (
        "You write the PERSONAL/RELATABLE angle: reframe the source's idea as something "
        "happening to YOU or the viewer right now, in concrete daily life. Make it feel "
        "like a friend telling you a story, not a journalist summarizing."
    ),
    "surprising_stat": (
        "You write the SURPRISING-STAT angle: lead with a specific number from the "
        "source that creates an information gap. If no hard number exists, surface a "
        "scale comparison or ratio that lands the same way."
    ),
    "pattern_interrupt": (
        "You write the PATTERN-INTERRUPT angle: open with a sentence that breaks the "
        "viewer's expectation of what a video about this topic would say. The hook line "
        "should feel like a record scratch — unexpected, slightly funny, or off-script."
    ),
}


async def _propose(app: Any, frame: AngleFrame, source: SourceContent) -> AngleProposal:
    system = (
        _PROMPTS[frame]
        + "\n\nReturn ONLY the structured fields. Be specific to the source — do not "
        "use generic hooks. Predicted_score is YOUR honest read of scroll-stop odds, 1-10."
    )
    user = (
        f"TITLE: {source.title}\n"
        f"AUDIENCE: {source.audience_hints}\n"
        f"KEY CLAIMS:\n- "
        + "\n- ".join(source.key_claims)
        + f"\n\nFULL BODY EXCERPT (for context only):\n{source.body[:6_000]}"
    )

    out = await app.ai(system=system, user=user, schema=_SingleAngle)
    return AngleProposal(
        frame=frame,
        hook_line=out.hook_line,
        angle=out.angle,
        why_works=out.why_works,
        predicted_score=out.predicted_score,
    )


async def propose_angles(app: Any, source: SourceContent) -> list[AngleProposal]:
    """Run all 5 angle proposers in parallel."""
    frames: list[AngleFrame] = [
        "contrarian",
        "didnt_know",
        "personal",
        "surprising_stat",
        "pattern_interrupt",
    ]
    results = await asyncio.gather(
        *(_propose(app, f, source) for f in frames),
        return_exceptions=True,
    )
    # Drop any failures rather than failing the whole pipeline. The critic
    # only needs 2+ candidates to pick from.
    return [r for r in results if isinstance(r, AngleProposal)]
