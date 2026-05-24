"""Pydantic models that flow between pipeline stages.

Naming follows the Archei rule from the project CLAUDE.md:
- Outputs consumed by downstream code (routing, scoring) live in flat schemas
  → Structured JSON (these models).
- Outputs consumed by downstream LLMs as context live in string fields on the
  same models (e.g. `reasoning`, `vo_line`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field

AngleFrame = Literal[
    "contrarian",
    "didnt_know",
    "personal",
    "surprising_stat",
    "pattern_interrupt",
]


class SourceContent(BaseModel):
    """Cleaned, structured representation of whatever URL the user pasted.

    Produced by the navigator harness — fed downstream as natural-language
    strings for the angle proposers / storyboarder to reason over.
    """

    url: str
    title: str
    body: str = Field(..., description="Cleaned main text, capped to ~50KB.")
    key_claims: list[str] = Field(
        default_factory=list,
        description="3-5 surprising or quotable claims pulled from the body.",
    )
    audience_hints: str = Field(
        default="",
        description="One-sentence guess at who this is for (devs, founders, marketers...).",
    )
    surprise_score: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Navigator's gut take on how surprising / scroll-stop-worthy the source is.",
    )


class AngleProposal(BaseModel):
    """One viral angle, proposed by a single .ai() agent."""

    frame: AngleFrame
    hook_line: str = Field(..., description="The literal first line spoken on screen (<10 words).")
    angle: str = Field(..., description="One-sentence framing of the take.")
    why_works: str = Field(..., description="Why this would stop a scroll, in plain English.")
    predicted_score: int = Field(ge=1, le=10)


class Beat(BaseModel):
    """A single beat (~3-5s) inside the storyboard."""

    idx: int
    duration_s: float = Field(ge=1.5, le=6.0)
    image_prompt: str = Field(..., description="Prompt for grok-imagine. Specific, visual, vertical-friendly.")
    caption: str = Field(..., description="3-5 words burned on-screen.")
    vo_line: str = Field(..., description="What Kokoro says during this beat.")
    motion_hint: Literal["zoom_in", "zoom_out", "pan_left", "pan_right", "static"] = "zoom_in"


class Storyboard(BaseModel):
    """The full plan for the reel before any media is generated."""

    angle: AngleProposal
    beats: list[Beat]
    total_duration_s: float
    style_notes: str = Field(default="", description="Visual consistency hints fed to every image prompt.")


class BeatArtifact(BaseModel):
    """Generated media for a single beat."""

    idx: int
    image_path: Optional[Path] = None
    audio_path: Optional[Path] = None

    model_config = {"arbitrary_types_allowed": True}


class HookVerdict(BaseModel):
    """Hook critic's read on the first beat."""

    score: int = Field(ge=1, le=10, description="Predicted scroll-stop probability.")
    passes: bool = Field(..., description="True if score >= 7.")
    reasoning: str
    suggested_fix: Optional[str] = None


class ReelResult(BaseModel):
    """Final result returned to the caller / CLI."""

    output_path: Path
    storyboard: Storyboard
    hook_verdict: Optional[HookVerdict] = None
    duration_s: float
    suggested_caption: str = ""
    suggested_hashtags: list[str] = Field(default_factory=list)
    run_id: str
    cost_usd_est: float = 0.0

    model_config = {"arbitrary_types_allowed": True}
