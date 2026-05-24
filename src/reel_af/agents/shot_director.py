"""Shot Director — per-segment cinematic plan: image, motion, caption.

Runs DeepSeek V4 Pro once per segment (parallel). For each segment it picks
an anchor TYPE first (literal / metaphor / contrast), then writes:
  • image_prompt    — for grok-imagine (first frame, cinematic, vertical)
  • motion_prompt   — for Veo (what should move; camera + subject)
  • on_screen_text  — a 2-4 word distillation of the VO line being spoken now
                      (NOT a fresh thought — the load-bearing words of THIS line)

Why per-segment and not single-call: each shot needs to reason about the
specific spoken line and the surrounding context. A single call to "design
all 6 shots" produces visually monotonous output because the model averages.
Parallel single-shot calls give variety AND speed (the calls run in 3-5s
each, all together via asyncio.gather).
"""

from __future__ import annotations

import asyncio
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from reel_af.agents.script_segmenter import Segment

AnchorType = Literal["literal", "metaphor", "contrast"]


class ShotPlan(BaseModel):
    """Plan for one segment's visual + caption. All three carefully chosen."""

    model_config = ConfigDict(extra="forbid")

    anchor_type: AnchorType = Field(
        ...,
        description=(
            "literal = show the thing being said (builds trust). "
            "metaphor = show a human analog (builds empathy). "
            "contrast = show the OPPOSITE of what's said (builds surprise). "
            "Pick the one that makes THIS specific line land hardest."
        ),
    )
    image_prompt: str = Field(
        ...,
        description=(
            "ONE concrete scene, ONE subject, vertical 9:16 framing. Specify camera "
            "(close-up / mid / wide), lighting, and a real environment (not concept art). "
            "DO NOT include any text/words/captions. DO NOT use abstract metaphors like "
            "'brain made of code' or 'glowing neural network'. Real people, real things."
        ),
    )
    motion_prompt: str = Field(
        ...,
        description=(
            "What should MOVE in this 4-second clip. Be specific: 'subject turns head', "
            "'camera dollies in', 'rain starts falling', 'liquid pours'. Keep camera "
            "moves subtle (slow push, gentle pan). Subject motion can be more dramatic."
        ),
    )
    on_screen_text: str = Field(
        ...,
        description=(
            "2-4 words BURNED on screen during this segment. These MUST be the most "
            "load-bearing words FROM the VO line — the words that carry the meaning "
            "if the rest were silent. ALL CAPS in the burn step is automatic, write "
            "them in normal case. NEVER restate the whole line."
        ),
    )


_DIRECTOR_SYSTEM = """You direct ONE shot of a vertical reel. The narration
for this shot is already written — your job is to translate it into:

  1. An ANCHOR TYPE — literal / metaphor / contrast.
     • literal   : show the thing being said. Anchors trust.
     • metaphor  : show a human-life analog of the idea. Anchors empathy.
     • contrast  : show the opposite of what's said. Anchors surprise.
     Pick the one that makes THIS line hit hardest.

  2. An IMAGE PROMPT for the first frame (grok-imagine).
     Concrete subject. Real environment. Specific framing. Vertical 9:16.
     Cinematic — like a single frame from a polished documentary or short
     film. NO text/words/letters in the frame.

  3. A MOTION PROMPT for the video (Veo image-to-video).
     What should move in the 4 seconds. Subject motion + subtle camera.

  4. ON-SCREEN TEXT — 2-4 words that get burned over the shot.
     These are the LOAD-BEARING WORDS OF THE VO LINE BELOW.
     If the VO says "Octopi taste with their skin", on-screen is "TASTE = SKIN"
     or "SKIN TASTES" — the actual revelation. Not "INTERESTING FACT".
     The viewer with sound off should get the same beat from these words alone.

PREVIOUS SEGMENTS (for visual variety — avoid repeating subjects / framings):
{prev_summary}

This shot has role={role}. The overall reel TONE is {tone}. Keep the visual
style consistent across shots (warm cinematic documentary unless tone says
otherwise)."""


async def _direct_one(
    app: Any,
    seg: Segment,
    prev_summaries: list[str],
    tone: str,
) -> ShotPlan:
    prev = "\n".join(f"  - {s}" for s in prev_summaries) if prev_summaries else "  (none — this is the first shot)"
    system = _DIRECTOR_SYSTEM.format(prev_summary=prev, role=seg.role, tone=tone)
    user = (
        f"VO LINE FOR THIS SHOT (the line being spoken now):\n"
        f"  {seg.text!r}\n\n"
        f"Duration: ~{seg.est_duration_s:.1f}s\n"
        f"Role in arc: {seg.role}"
    )
    return await app.ai(system=system, user=user, schema=ShotPlan)


async def direct_shots(
    app: Any,
    segments: list[Segment],
    tone: str = "wonder",
) -> list[ShotPlan]:
    """Direct all shots. Sequential because each shot sees prior summaries
    so the director can avoid visual repetition."""
    plans: list[ShotPlan] = []
    summaries: list[str] = []
    for seg in segments:
        plan = await _direct_one(app, seg, summaries, tone)
        plans.append(plan)
        summaries.append(f"shot {seg.idx} ({plan.anchor_type}): {plan.image_prompt[:90]}")
    return plans


async def direct_shots_parallel(
    app: Any,
    segments: list[Segment],
    tone: str = "wonder",
) -> list[ShotPlan]:
    """Faster, less visually-aware alternative — run all directors in parallel.
    Use this when wall-time matters more than visual variety control."""
    results = await asyncio.gather(*(_direct_one(app, s, [], tone) for s in segments))
    return list(results)
