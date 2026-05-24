"""Script Critic — adversarial scoring + concrete weakness identification.

The PROVER in our HUNT-PROVE pair. It scores each candidate script on five
weighted dimensions, identifies SPECIFIC failing phrases (quoted verbatim
so the refiner has something concrete to operate on), and ranks the pool.

Critical design choice: critic is forced to quote weaknesses verbatim. This
prevents "rubber stamp" mode where the critic vaguely says "make it
punchier" without identifying what's actually weak.

Context strategy:
  IN  : ALL candidate drafts + the article summary (for faithfulness check)
  OUT : ranked list with per-draft scores + per-dimension feedback
  NOT : article body (script is the artifact under test, not the source)
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from reel_af.agents.distiller import ArticleSummary
from reel_af.agents.reel_composer import ReelDraft

# Weights for each scoring dimension. Sum = 1.0. Adjustable as we learn
# which dims best predict actual viral performance.
WEIGHTS = {
    "hook_strength": 0.30,
    "faithfulness": 0.20,
    "trope_execution": 0.20,
    "spoken_flow": 0.15,
    "close_landing": 0.15,
}

# Acceptance threshold — composite score above this = ship as-is.
ACCEPT_THRESHOLD = 8.0


class DimensionScore(BaseModel):
    """One critic dim's score + specific weak phrases identified."""

    model_config = ConfigDict(extra="forbid")

    score: int = Field(..., ge=1, le=10)
    reasoning: str = Field(
        ...,
        description=(
            "1-2 sentences explaining the score. Reference SPECIFIC phrases "
            "from the draft when possible."
        ),
    )
    weak_phrases: list[str] = Field(
        ...,
        description=(
            "EXACT phrases from the draft that cost points for this dimension. "
            "Quote verbatim — the refiner will substring-search and rewrite them. "
            "Empty list if score >= 8 (nothing actionably weak)."
        ),
    )
    suggested_rewrite: str = Field(
        ...,
        description=(
            "If the dim scored < 8, propose a concrete replacement for the weakest "
            "phrase. Empty string if score >= 8."
        ),
    )


class DraftScorecard(BaseModel):
    """All-dim scoring + composite for ONE draft."""

    model_config = ConfigDict(extra="forbid")

    draft_index: int = Field(..., description="0-based index into the input draft pool.")
    hook_strength: DimensionScore
    faithfulness: DimensionScore
    trope_execution: DimensionScore
    spoken_flow: DimensionScore
    close_landing: DimensionScore
    overall_notes: str = Field(
        ...,
        description="2-3 sentences on what works and what kills this draft overall.",
    )


class CriticVerdict(BaseModel):
    """Critic's full output across the pool."""

    model_config = ConfigDict(extra="forbid")

    scorecards: list[DraftScorecard] = Field(..., description="One per input draft.")
    winner_index: int = Field(..., description="0-based index of the highest-composite draft.")
    winner_reasoning: str = Field(
        ...,
        description="1-2 sentences: why this draft beats the others.",
    )


def composite_score(card: DraftScorecard) -> float:
    """Apply the weight table to dim scores → single 1-10 composite."""
    return (
        card.hook_strength.score * WEIGHTS["hook_strength"]
        + card.faithfulness.score * WEIGHTS["faithfulness"]
        + card.trope_execution.score * WEIGHTS["trope_execution"]
        + card.spoken_flow.score * WEIGHTS["spoken_flow"]
        + card.close_landing.score * WEIGHTS["close_landing"]
    )


_SYSTEM = """You're scoring vertical-reel script candidates. You're not nice.
A "good" script is one that would actually stop a thumb mid-scroll and hold
attention for 20 seconds; everything else is failure.

SCORE EACH DRAFT ON FIVE DIMENSIONS (1-10 each):

(1) HOOK STRENGTH (weight 30%)
    "Would a thumb actually stop in <1 second?"
    PENALTIES (deduct 2-3 each):
      • Generic openers ("Did you know", "In this video", "Today we…")
      • Soft setup ("So basically…", "Let me tell you about…")
      • Hook explains the topic instead of provoking curiosity
      • > 7 words before the first verb
    REWARDS:
      • Contradiction, surprise number, mid-thought drop, direct threat
      • First word is a verb, noun, or named entity

(2) FAITHFULNESS (weight 20%)
    "Does the script use the article's actual content?"
    PENALTIES:
      • Invented metaphors not present in the source
      • Generic claims that ignore the specific examples in the summary
      • Author's argument simplified to the point of misrepresentation
    REWARDS:
      • Uses the source's own names, numbers, examples verbatim or near-verbatim
      • Preserves the author's actual framing of the idea

(3) TROPE EXECUTION (weight 20%)
    "Are the declared tricks visibly present in the script?"
    The drafter declared a hook_trick, retention_trick, close_trick. Check
    each one is actually executed:
      • hook_trick = "contradiction" → first sentence must contradict
      • retention_trick = "open_loop" → body must promise X and delay
      • close_trick = "loop_closure" → last sentence must echo the hook
    Declared-but-absent tricks = 1-3 penalty per missing trick.

(4) SPOKEN FLOW (weight 15%)
    "Reads as spoken English, not written prose?"
    PENALTIES:
      • Long compound sentences with multiple commas
      • Uniform sentence length (all the same → robotic)
      • Written-only words ("furthermore", "however", "thus")
      • No em-dashes / commas where a human would breathe
    REWARDS:
      • Wildly varied sentence length (short. then long-and-flowing.)
      • Active voice, second person
      • Punctuation that steers TTS delivery

(5) CLOSE LANDING (weight 15%)
    "Does the ending actually land?"
    PENALTIES:
      • Trailing off, ellipsis, hedge ("…or something")
      • Generic CTA ("Like and subscribe", "Stay tuned")
      • Close that doesn't execute the declared close_trick
    REWARDS:
      • Loop closure that echoes the hook
      • Save-bait or comment-bait that fires naturally from the content
      • Last sentence ends on a noun or verb, not a hedge

CRITICAL OUTPUT RULES:
  • When you deduct points on a dimension, you MUST quote the SPECIFIC
    phrase from the draft that caused the deduction (verbatim — the
    refiner will substring-search). Lists may be empty only if score >= 8.
  • When dim score < 8, propose a CONCRETE replacement for the worst
    phrase. "Make it punchier" is not feedback; "replace 'Did you know
    that X' with 'X. Today.'" is feedback.
  • Pick the winner by composite, not by gut. Composite weights are
    enforced downstream — your job is honest per-dim scoring."""


async def critique(
    app: Any,
    drafts: list[ReelDraft],
    summary: ArticleSummary,
) -> CriticVerdict:
    if not drafts:
        raise RuntimeError("script_critic: no drafts to score.")

    listing = "\n\n".join(
        f"=== DRAFT [{i}] ===\n"
        f"direction      : {d.direction}\n"
        f"hook_trick     : {d.hook_trick}\n"
        f"retention_trick: {d.retention_trick}\n"
        f"close_trick    : {d.close_trick}\n"
        f"tone           : {d.voice_tone}\n"
        f"self-score     : {d.viral_score}/10\n"
        f"SCRIPT:\n{d.script}"
        for i, d in enumerate(drafts)
    )
    user = (
        f"ARTICLE SUMMARY (for faithfulness check):\n"
        f"  thesis   : {summary.one_line_thesis}\n"
        f"  takeaway : {summary.intended_takeaway}\n"
        f"  examples : {summary.concrete_examples}\n\n"
        f"CANDIDATES:\n{listing}"
    )
    return await app.ai(system=_SYSTEM, user=user, schema=CriticVerdict)
