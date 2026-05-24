"""Narrative Architect — writes the full reel script as flowing spoken English.

This is the brain step. It receives the take + source and produces a single
continuous narration with the 5 micro-beat structure enforced. The output is
*spoken* English: short clauses, varied lengths, second-person, em-dashes for
dramatic pauses, commas for breath, deliberate punctuation Kokoro can act on.

Why one continuous script and not per-beat lines: the old per-beat approach
made each line stand alone, which destroyed the narrative arc and made the
TTS sound disconnected. Writing the script first, then deriving segments
from sentence boundaries, gives Kokoro a flowing utterance and gives the
shot director real context for each visual choice.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from reel_af.agents.take_picker import TakeBrief
from reel_af.models import SourceContent

# Tight word budget so the reel lands in the 18-22s sweet spot for completion.
# Kokoro speaks at ~2.6 words/sec → 50 words ≈ 19s.
TARGET_WORDS_MIN = 45
TARGET_WORDS_MAX = 58


class _Script(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hook: str = Field(
        ...,
        description=(
            "4-7 words. A pattern-interrupt opening sentence. Must contain a verb in "
            "the first 3 words. NO 'did you know' / 'in this video' / 'today'."
        ),
    )
    stakes: str = Field(
        ...,
        description=(
            "8-12 words. One sentence in second person ('you', 'your') that makes the "
            "viewer feel this is about them. Concrete imagery beats abstraction."
        ),
    )
    revelation: str = Field(
        ...,
        description=(
            "12-18 words. The surprising claim from the source, delivered as a single "
            "sentence (one comma allowed). Use specific numbers / verbs / nouns. Active voice."
        ),
    )
    consequence: str = Field(
        ...,
        description=(
            "8-12 words. 'Which means…' — what changes for the viewer now that they "
            "know the revelation. Must be a real consequence, not a platitude."
        ),
    )
    callback: str = Field(
        ...,
        description=(
            "4-6 words. Lands back on the hook's energy. Open-ended or quotable so the "
            "viewer rewatches / comments. End on a noun or a verb, not a hedge."
        ),
    )


class NarrativeScript(BaseModel):
    """The final script. .full() returns the contiguous spoken-English version."""

    hook: str
    stakes: str
    revelation: str
    consequence: str
    callback: str

    def full(self) -> str:
        # Join with single spaces. Sentences already end in their own punctuation
        # so this produces natural spoken-prose.
        return " ".join(
            s.strip() for s in (self.hook, self.stakes, self.revelation, self.consequence, self.callback)
        )

    def word_count(self) -> int:
        return len(self.full().split())


def _build_system(tone: str) -> str:
    return f"""You are writing the SPOKEN narration for a vertical reel.

The narration will be read aloud by a TTS engine that respects punctuation
(commas = small breath, em-dashes = dramatic pause, periods = full stop,
question marks = rising inflection). Punctuate deliberately.

STRUCTURE — five micro-beats in this exact order:
  HOOK         — 4-7 words. Pattern-interrupt. Make a thumb stop.
  STAKES       — 8-12 words. Second-person bridge ("you", "your").
  REVELATION   — 12-18 words. The surprising claim from the source.
  CONSEQUENCE  — 8-12 words. "Which means..." — real implication.
  CALLBACK     — 4-6 words. Land on the hook's energy. Quotable.

Total target: {TARGET_WORDS_MIN}-{TARGET_WORDS_MAX} words. Sum the budgets
above and you'll see they fit.

STYLE RULES (non-negotiable):
• Spoken English, NOT written. Read each line aloud in your head.
• Vary sentence length. Robotic prose has uniform sentence lengths.
• Concrete > abstract. "Three weeks" not "a short time".
• Active voice. "Octopi taste light." Not "Light is detected by..."
• Second person where possible. "Your" beats "people's".
• No filler: "actually", "basically", "the fact that", "it's interesting".
• No AI tells: "Let's talk about", "In this video", "Did you know".

PUNCTUATION (steers TTS delivery):
• Use em-dashes (—) for one or two dramatic pauses. Don't overuse.
• Use commas where a human speaker would breathe.
• End the CALLBACK with a period — never an ellipsis or hedge.

TONE: {tone}. Match the source material — don't force a register.

Return the five fields. Each is ONE sentence (or one phrase for hook/callback).
The full script must read as one continuous spoken thought, not five labeled
chunks. The reader will not see the labels."""


async def write_script(
    app: Any,
    take: TakeBrief,
    source: SourceContent,
) -> NarrativeScript:
    """Write the full reel script as flowing spoken English."""
    user = (
        f"YOUR TAKE\n"
        f"  take          : {take.take}\n"
        f"  human stakes  : {take.human_stakes}\n"
        f"  key revelation: {take.key_revelation}\n"
        f"  callback hint : {take.callback_phrase}\n"
        f"  tone          : {take.voice_tone}\n\n"
        f"SOURCE CONTEXT (for grounding only — do not paraphrase wholesale)\n"
        f"  title    : {source.title}\n"
        f"  audience : {source.audience_hints}\n"
        f"  claims   :\n"
        + "\n".join(f"    - {c}" for c in source.key_claims)
    )

    draft = await app.ai(
        system=_build_system(take.voice_tone),
        user=user,
        schema=_Script,
    )
    return NarrativeScript(
        hook=draft.hook,
        stakes=draft.stakes,
        revelation=draft.revelation,
        consequence=draft.consequence,
        callback=draft.callback,
    )
