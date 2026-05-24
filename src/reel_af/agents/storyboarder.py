"""Storyboarder — converts the chosen angle into a 4-7 beat shot list.

This is a single .ai() call because:
 - input is bounded (one angle + source summary, < 2K tokens)
 - output is bounded (4-7 beats × ~4 fields each, ~700 tokens)
 - the work is structural, not navigational

If we needed to read the full source body again to ground individual beats
we'd switch to a .harness(). For v0 the angle + key_claims is enough.
"""

from __future__ import annotations

import asyncio
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from reel_af.models import AngleProposal, Beat, SourceContent, Storyboard

MIN_BEATS = 6
MAX_BEATS = 9
TARGET_DURATION_S = 20.0  # sweet spot for completion-rate on Reels / Shorts


class _BeatDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")  # OpenAI strict-schema requirement

    duration_s: float = Field(
        ...,
        ge=1.4,
        le=3.2,
        description="Vertical scroll content cuts every 2-3s; longer beats die.",
    )
    image_prompt: str = Field(
        ...,
        description=(
            "Concrete visual prompt for an image generator. Describe ONE specific scene "
            "with subject, setting, framing (vertical 9:16), and mood. Avoid text-on-image "
            "instructions — captions are burned in separately. Avoid abstract nouns."
        ),
    )
    caption: str = Field(
        ...,
        description="3-5 punchy words burned across the middle of the frame.",
    )
    vo_line: str = Field(
        ...,
        description="Exact words the narrator speaks during this beat. Match duration_s × ~2.5 words/sec.",
    )
    motion_hint: Literal["zoom_in", "zoom_out", "pan_left", "pan_right", "static"] = Field(
        ..., description="Camera move applied via ffmpeg ken-burns."
    )


class _StoryboardDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    beats: list[_BeatDraft] = Field(..., min_length=MIN_BEATS, max_length=MAX_BEATS)
    style_notes: str = Field(
        ...,
        description=(
            "1-2 sentence visual consistency hint applied to every image prompt — e.g. "
            "'cinematic teal-and-orange grading, shallow depth of field, 35mm film grain'."
        ),
    )


async def storyboard(
    app: Any,
    angle: AngleProposal,
    source: SourceContent,
) -> Storyboard:
    system = (
        f"You design vertical short-form reels (TikTok/Reels/Shorts). "
        f"Convert the chosen angle into {MIN_BEATS}-{MAX_BEATS} beats totalling "
        f"~{TARGET_DURATION_S:.0f} seconds.\n\n"
        "PACING (critical — vertical-scroll viewers cut at every shot):\n"
        f"• Each beat is 1.4–3.2 seconds. Most should be 1.8–2.4s. NEVER 4s+.\n"
        "• Sum of durations: ["
        f"{TARGET_DURATION_S - 4}, {TARGET_DURATION_S + 4}] seconds.\n"
        "• vo_line word count = duration_s × 2.6  (so a 2s beat = 5-6 words).\n"
        "• Total VO should feel like spoken thought, not lecture.\n\n"
        "STRUCTURE:\n"
        "• Beat 0 is the HOOK. vo_line MUST start with the angle's hook_line verbatim "
        "(or a tighter version). Image must be visually arresting in <1s.\n"
        "• Middle beats DEVELOP the angle with CONCRETE imagery. Real people, real "
        "objects, real settings. No abstract concept art ('brain made of code', "
        "'glowing neural network'). Documentary or cinematic.\n"
        "• Final beat is the PAYOFF — leaves a quotable line, a question, or a hard "
        "cut that begs a rewatch.\n\n"
        "CAPTIONS (burned across the screen):\n"
        "• 2-4 words MAX. They're emphasis, not subtitles. The viewer reads them in "
        "0.3 seconds.\n"
        "• ALL CAPS. Punchy verbs and nouns, no filler ('the', 'is', 'of').\n"
        "• Should make sense on their own when scrolled past with sound off.\n\n"
        "Visual consistency: define style_notes once (look/lighting/grading) so every "
        "image prompt can inherit it implicitly. Don't repeat style words inside each "
        "image_prompt."
    )

    user = (
        f"CHOSEN ANGLE\n"
        f"  frame: {angle.frame}\n"
        f"  hook : {angle.hook_line!r}\n"
        f"  take : {angle.angle}\n"
        f"  why  : {angle.why_works}\n\n"
        f"SOURCE\n"
        f"  title   : {source.title}\n"
        f"  audience: {source.audience_hints}\n"
        f"  claims  :\n    - " + "\n    - ".join(source.key_claims)
    )

    draft = await app.ai(system=system, user=user, schema=_StoryboardDraft)

    # Re-index beats and clamp the total duration as a defense.
    beats = [
        Beat(
            idx=i,
            duration_s=b.duration_s,
            image_prompt=b.image_prompt,
            caption=b.caption,
            vo_line=b.vo_line,
            motion_hint=b.motion_hint,
        )
        for i, b in enumerate(draft.beats)
    ]
    total = sum(b.duration_s for b in beats)
    return Storyboard(
        angle=angle,
        beats=beats,
        total_duration_s=total,
        style_notes=draft.style_notes,
    )


# Convenience for tests / debugging.
async def storyboard_pretty(app: Any, angle: AngleProposal, source: SourceContent) -> str:
    sb = await storyboard(app, angle, source)
    lines = [
        f"Total: {sb.total_duration_s:.1f}s   style: {sb.style_notes}",
        "",
    ]
    for b in sb.beats:
        lines.append(
            f"[{b.idx}] {b.duration_s:.1f}s  cap={b.caption!r}\n"
            f"    vo : {b.vo_line!r}\n"
            f"    img: {b.image_prompt[:140]}{'…' if len(b.image_prompt) > 140 else ''}"
        )
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover — local smoke

    async def _main() -> None:
        import os
        from agentfield import Agent, AIConfig
        from reel_af.agents.angle_proposer import propose_angles
        from reel_af.agents.angle_critic import pick_angle
        from reel_af.agents.navigator import navigate

        app = Agent(
            node_id="reel-af-test",
            version="0.1.0",
            ai_config=AIConfig(
                model="openrouter/openai/gpt-5-mini",
                api_key=os.environ["OPENROUTER_API_KEY"],
                api_base="https://openrouter.ai/api/v1",
            ),
        )
        sc = await navigate(
            app, "https://www.santoshkumarradha.com/writing/atomic-unit-of-intelligence"
        )
        winner = await pick_angle(app, await propose_angles(app, sc), sc)
        print(await storyboard_pretty(app, winner, sc))

    asyncio.run(_main())
