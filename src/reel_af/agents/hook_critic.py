"""Hook Critic — predicts scroll-stop probability on the opening beat.

v0 is a single .ai() pass that judges the hook text + caption + a one-line
description of the opening image. v1 would feed the actual first-frame image
in via vision (Gemini / GPT image), but for this example we keep the input
text-only so it runs against any chat model.

The critic only *surfaces* its verdict. Regeneration is the caller's call —
we don't want a runaway loop.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from reel_af.models import Beat, HookVerdict, Storyboard


class _CriticOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    score: int = Field(..., ge=1, le=10)
    reasoning: str
    suggested_fix: str = Field(
        ..., description="Concrete rewrite of the hook if score < 7. Empty string otherwise."
    )


async def judge_hook(app: Any, storyboard: Storyboard) -> HookVerdict:
    if not storyboard.beats:
        raise RuntimeError("hook-critic: empty storyboard")
    opener: Beat = storyboard.beats[0]

    system = (
        "You're predicting whether the opening 1 second of a vertical reel will stop a "
        "thumb. Score 1-10. 7+ means it would hold a viewer at least 2 more seconds.\n\n"
        "Heuristics that score HIGH:\n"
        "• Hook contradicts the viewer's expectation or opens a curiosity gap.\n"
        "• Specific concrete imagery, not abstract concept art.\n"
        "• <7 spoken words in first second.\n\n"
        "Heuristics that score LOW:\n"
        "• Generic AI-content openers ('let's talk', 'in this video', 'today we').\n"
        "• Hook explains what's coming instead of provoking it.\n"
        "• Imagery is a stock metaphor (brain/lightbulb/gears/networks-of-glowing-lines).\n\n"
        "Return the score, your reasoning in 1-2 sentences, and a concrete rewrite of the "
        "hook IF score is below 7 (otherwise empty string)."
    )

    user = (
        f"ANGLE FRAME: {storyboard.angle.frame}\n"
        f"HOOK LINE  : {storyboard.angle.hook_line!r}\n"
        f"OPENER BEAT\n"
        f"  duration : {opener.duration_s:.1f}s\n"
        f"  vo line  : {opener.vo_line!r}\n"
        f"  caption  : {opener.caption!r}\n"
        f"  imagery  : {opener.image_prompt}\n"
        f"  motion   : {opener.motion_hint}\n"
    )

    out = await app.ai(system=system, user=user, schema=_CriticOut)
    return HookVerdict(
        score=out.score,
        passes=out.score >= 7,
        reasoning=out.reasoning,
        suggested_fix=out.suggested_fix.strip() or None,
    )
