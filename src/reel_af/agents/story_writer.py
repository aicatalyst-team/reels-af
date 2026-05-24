"""Story Writer — writes ONE continuous viral script.

Replaces take_picker + narrative_architect. The previous 2-step + 5-field
schema produced fragmented copy because each field was optimized in isolation
and the model never wrote the whole thing as one breath.

This agent gets ONLY the source's filtered claims + audience hint — never
the raw article body. Filtering already happened in the navigator; passing
the body again is pure context pollution.

Output is FLAT — `script` is one continuous string the rest of the pipeline
treats as the source of truth. Voice tone is a side-output so the TTS layer
can route to the right Kokoro voice.

The prompt is the entire product here. Few-shot viral examples are baked in.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from reel_af.models import SourceContent

VoiceTone = Literal["urgent", "wonder", "deadpan", "earnest", "playful"]

# Target: ~50 spoken words → ~19s at Kokoro's 2.6 words/sec. Hard cap to
# prevent runaway scripts.
TARGET_WORDS_MIN = 42
TARGET_WORDS_MAX = 58


class _Script(BaseModel):
    model_config = ConfigDict(extra="forbid")

    script: str = Field(
        ...,
        description=(
            f"The COMPLETE narration as ONE continuous spoken paragraph. "
            f"{TARGET_WORDS_MIN}-{TARGET_WORDS_MAX} words. NOT a list, NOT bullet "
            f"points — one flowing thought. Use commas for breaths, em-dashes (—) "
            f"for dramatic pauses, periods to end statements. Spoken English, "
            f"second person where possible. Each sentence should hook into the next."
        ),
    )
    voice_tone: VoiceTone = Field(
        ...,
        description=(
            "Emotional register: urgent (alert/threat), wonder (discovery), "
            "deadpan (irony/contrarian), earnest (advice), playful (humor). "
            "Pick what fits the source — don't force 'wonder' on serious content."
        ),
    )


_SYSTEM = """You write VIRAL SHORT-FORM SCRIPTS for vertical reels (TikTok, Reels, Shorts).

Your script will be read aloud by a TTS engine and shown over generated video.
You're writing for a thumb that's already moving — they'll swipe in 0.5s if
nothing lands. The script has to grab and not let go.

OUTPUT: one continuous paragraph of 42-58 words, with this narrative spine:

  • HOOK         (4-7 words)   — pattern-interrupt opening
  • STAKES       (8-12 words)  — second-person bridge: why YOU care
  • REVELATION   (12-18 words) — the surprising claim from the source
  • CONSEQUENCE  (8-12 words)  — "which means..." — what changes for you
  • CALLBACK     (4-6 words)   — land on the hook's energy

Don't label these sections — write one flowing paragraph. The reader hears it
as one continuous thought. The arc is implicit.

LINKING RULES (this is what makes it cohere):
  • Each sentence picks up a word or image from the previous one.
  • If sentence N ends on a noun, sentence N+1 starts with that noun or its
    consequence. Continuity beats novelty inside the same script.
  • Do NOT switch metaphors mid-script. Pick one frame and ride it.

SPOKEN-ENGLISH RULES:
  • Read each line aloud in your head. If your jaw stalls, rewrite it.
  • Vary sentence length wildly. Short. Then long-and-flowing-like-this.
  • Concrete > abstract. "Three weeks" beats "a short time".
  • Active voice. "Octopi taste light." Not "Light is detected by..."
  • Second person where possible.
  • NO filler: "actually", "basically", "the fact that".
  • NO AI tells: "Let's talk about", "In this video", "Today we", "Did you know".

PUNCTUATION DRIVES TTS DELIVERY:
  • Em-dash (—) → dramatic pause. Use 1-2 max per script.
  • Comma → small breath.
  • Period → full stop.
  • End the script with a period — not an ellipsis, not a hedge.

EXAMPLES (read these to calibrate):

source: "Octopi can taste with their suckers"
GOOD: "Your hand can't taste a thing. An octopus's can. Every sucker is a
       tongue — it tastes the rock, the prey, the predator before it
       sees them. Which means an octopus knows your skin's chemistry
       on contact. Your hand is missing something."
BAD:  "Did you know that scientists have discovered octopi can taste with
       their suckers? This is a fascinating ability that helps them..."

source: "Most viral essays were drafts the author thought were bad"
GOOD: "You delete the bad drafts. Don't. The essays people actually share
       were the ones their writers nearly trashed. The brain rejects what
       feels obvious — even when it's the most original thing you'll
       write. Stop trusting the editor."
BAD:  "Writers often underestimate their best work. Studies show that..."

source: "Mathematicians proved a 100-year-old conjecture using AI"
GOOD: "A computer just closed a math problem older than your grandparents.
       Not a brute force. A proof — elegant, short, the kind that wins
       a Fields Medal. Which means math itself just got a new collaborator.
       And it doesn't sleep."
BAD:  "Recent breakthroughs in AI-assisted mathematics have led to..."

Now: read the source and write the script. Pick ONE take, commit, write it
in one breath."""


async def write_story(app: Any, source: SourceContent) -> _Script:
    """Write the complete viral script. One .ai call."""
    user = (
        f"SOURCE\n"
        f"  title    : {source.title}\n"
        f"  audience : {source.audience_hints}\n"
        f"  surprise : {source.surprise_score}/10\n\n"
        f"KEY CLAIMS (these are your only raw material — pick what lands hardest):\n"
        + "\n".join(f"  - {c}" for c in source.key_claims)
    )
    return await app.ai(system=_SYSTEM, user=user, schema=_Script)
