"""Take Picker — picks ONE viral take from a source.

Replaces the old 5-angle-fan-out + critic pattern. The ensemble was clever
but always-pick-one is closer to how real editors work: commit to one take,
write it well. We use DeepSeek V4 Pro because the work is reasoning-heavy
(map article claims → human-stakes reframing) and the output is small.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from reel_af.models import SourceContent

AnchorTone = Literal["urgent", "wonder", "deadpan", "earnest", "playful"]


class TakeBrief(BaseModel):
    """The single take the rest of the pipeline will execute against."""

    model_config = ConfigDict(extra="forbid")

    take: str = Field(
        ...,
        description=(
            "One sentence (max 18 words) reframing the article's core insight through a "
            "human-stakes lens. NOT a summary. The viewer reading this alone should feel "
            "an instant 'wait, what?'"
        ),
    )
    human_stakes: str = Field(
        ...,
        description=(
            "One sentence explaining WHY this matters to the viewer's daily life, in "
            "second person ('you', 'your'). This is the bridge from article-world to "
            "viewer-world. Concrete, not abstract."
        ),
    )
    key_revelation: str = Field(
        ...,
        description=(
            "The single most surprising claim from the source — verbatim or near-verbatim "
            "if quotable. This is what the reel hinges on; everything else sets it up."
        ),
    )
    callback_phrase: str = Field(
        ...,
        description=(
            "A 3-6 word phrase the script will land on at the end, ideally echoing the "
            "hook. Creates the loop-and-save instinct."
        ),
    )
    voice_tone: AnchorTone = Field(
        ...,
        description=(
            "Emotional register of the narration. Determines voice + delivery: "
            "urgent (warn/threat), wonder (science/discovery), deadpan (irony/contrarian), "
            "earnest (advice/coaching), playful (humor/quirky)."
        ),
    )


_SYSTEM = """You are picking ONE take for a 20-second vertical reel.

The user just pasted an article. Your job is NOT to summarize it. Your job
is to find the single most viral-shaped reframing of its core insight, in
the form of a one-sentence take that would stop a thumb mid-scroll.

A good take has three properties:
1. It surprises — it says something most viewers would NOT guess.
2. It hooks human stakes — the viewer should feel "this is about me".
3. It is specific — it commits to one claim, not a general observation.

Anti-patterns to AVOID:
• Summary: "Researchers found that octopi can taste with their suckers."
• Listicle: "5 surprising facts about octopi."
• Generic AI hook: "Did you know that octopi..."
• Soft observation: "Octopi are more complex than people realize."

What works instead:
• Stakes-flip: "Imagine your hand tasting everything it touched."
• Paradox: "Octopi can't see color. They camouflage perfectly anyway."
• Superlative: "This is the only animal that tastes with its skin."

Pick the tone that best fits the source material — don't force 'wonder' onto
serious content or 'urgent' onto something playful.

Output the take, the human stakes bridge, the key revelation it rests on,
a callback phrase the script can land on, and the right voice tone."""


async def pick_take(app: Any, source: SourceContent) -> TakeBrief:
    """Pick one take for the reel. Single reasoning-heavy call."""
    user = (
        f"SOURCE\n"
        f"  title    : {source.title}\n"
        f"  audience : {source.audience_hints}\n"
        f"  surprise : {source.surprise_score}/10\n\n"
        f"KEY CLAIMS (verbatim from the source — these are your raw material):\n"
        + "\n".join(f"  - {c}" for c in source.key_claims)
        + f"\n\nFULL BODY EXCERPT (for context only; do NOT summarize):\n{source.body[:8000]}"
    )
    return await app.ai(system=_SYSTEM, user=user, schema=TakeBrief)
