"""Scene Breaker — splits the script into scenes + picks per-scene captions.

Why .ai() instead of regex: regex can split on punctuation but it can't pick
the LOAD-BEARING WORDS of each line for the burned caption. Doing both at
once (where to break + what to caption) keeps the two choices coherent.

Context strategy: this agent sees ONLY the full script. No source body, no
audience hint, no take. The script is now the source of truth — anything
else is pollution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# Same speaking rate the rest of the pipeline assumes.
WORDS_PER_SECOND = 2.6
MIN_SCENES = 5
MAX_SCENES = 8


@dataclass
class Scene:
    """One scene of the reel."""

    idx: int
    sentence: str         # exact spoken text for this scene (Kokoro input)
    caption: str          # 2-4 load-bearing words burned on-screen
    est_duration_s: float # estimated spoken duration (real duration from .wav at assembly)
    role: str = "body"    # hook | body | callback — inferred from position


class _BrokenScene(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sentence: str = Field(
        ...,
        description=(
            "Verbatim slice of the original script. Sum of all sentences across "
            "scenes must equal the full script (punctuation preserved). Each "
            "sentence should end at a natural breath: period, question mark, or "
            "em-dash boundary."
        ),
    )
    caption: str = Field(
        ...,
        description=(
            "2-4 words taken FROM the sentence above (or distilling its meaning). "
            "These are the words a viewer with sound off needs to follow the reel. "
            "Pick the most surprising / load-bearing words — verbs and concrete "
            "nouns over articles and adverbs. Normal case; assembly uppercases."
        ),
    )


class _Scenes(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenes: list[_BrokenScene] = Field(
        ...,
        min_length=MIN_SCENES,
        max_length=MAX_SCENES,
        description=f"{MIN_SCENES}-{MAX_SCENES} scenes covering the entire script in order.",
    )


_SYSTEM = """You're splitting a vertical reel's narration script into scenes.
Each scene becomes one ~3-second shot with a generated video and a burned-in
caption.

DO TWO THINGS IN ONE PASS:

1. SPLIT the script into 5-8 scenes at COMPLETE SENTENCE boundaries only —
   period, question mark, or exclamation point. NEVER split at an em-dash
   or comma. The sentence on its OWN must be a complete spoken thought.

2. CAPTION each scene with 2-4 load-bearing words from the sentence. Pick
   the most surprising verbs and concrete nouns. NOT "INTERESTING FACT" or
   "WATCH THIS". The caption must mean the SAME THING as the sentence in
   compressed form.

NON-NEGOTIABLE RULES:

  • NO orphan fragments. "The problem.", "But why?", "And then." are
    fragments — they MUST be attached to the preceding or following
    sentence. Every scene's sentence must stand alone as a complete thought.
  • NO trailing em-dashes. "42 investigations locked —" is broken;
    combine with the next clause.
  • The LAST scene must be the script's actual closing sentence — never a
    fragment, never a question mark unless it's a complete rhetorical
    question that READS as the close.
  • Sentences must concatenate back to the script verbatim (punctuation
    preserved). Don't paraphrase, drop, or add words. Pure slicing.

Example:
  script: "Your hand can't taste a thing. An octopus's can. Every sucker
           is a tongue — it tastes the rock, the prey, the predator before
           it sees them. Save this for next time."

  scenes:
    [1] sentence: "Your hand can't taste a thing."
        caption : "hand can't taste"
    [2] sentence: "An octopus's can."
        caption : "octopus can"
    [3] sentence: "Every sucker is a tongue — it tastes the rock, the prey,
                   the predator before it sees them."
        caption : "every sucker tastes"
    [4] sentence: "Save this for next time."
        caption : "save for later"

Notice: scenes split at periods only. The em-dash phrase stays attached
to its parent sentence — not broken into a fragment."""


def _est_duration(text: str) -> float:
    words = max(len(text.split()), 1)
    return max(1.5, min(words / WORDS_PER_SECOND, 5.0))


async def break_scenes(app: Any, script: str) -> list[Scene]:
    """Break the script into scenes with captions, via one .ai() call."""
    user = f"SCRIPT TO SPLIT:\n\n{script.strip()}"
    out = await app.ai(system=_SYSTEM, user=user, schema=_Scenes)
    n = len(out.scenes)
    scenes: list[Scene] = []
    for i, bs in enumerate(out.scenes):
        # Role inference: first = hook, last = callback, others = body.
        # Used downstream for per-scene speed (callback slower for emphasis)
        # and visual arc shaping (hook gets the most arresting visual).
        if i == 0:
            role = "hook"
        elif i == n - 1:
            role = "callback"
        elif i == 1 and n >= 4:
            role = "stakes"
        elif i == n - 2 and n >= 5:
            role = "consequence"
        else:
            role = "revelation" if n >= 4 else "body"
        scenes.append(
            Scene(
                idx=i,
                sentence=bs.sentence.strip(),
                caption=bs.caption.strip(),
                est_duration_s=_est_duration(bs.sentence),
                role=role,
            )
        )
    return scenes
