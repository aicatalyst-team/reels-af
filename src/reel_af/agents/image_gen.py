"""Image fan-out — one grok-imagine call per beat, run in parallel.

Each prompt is augmented with the storyboard's style_notes so visuals stay
visually consistent across beats. Vertical 9:16 framing is enforced via
image_config.aspect_ratio (xAI accepts it).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from agentfield.media_providers import OpenRouterProvider

from reel_af.models import Beat, BeatArtifact, Storyboard

IMAGE_MODEL = "openrouter/x-ai/grok-imagine-image-quality"


def _augment_prompt(prompt: str, style_notes: str) -> str:
    """Append global style hints + vertical framing to each beat's prompt."""
    parts = [prompt.strip().rstrip(".")]
    if style_notes:
        parts.append(style_notes.strip().rstrip("."))
    parts.append("vertical 9:16 framing, fills the frame, no text or captions")
    return ". ".join(parts) + "."


async def _gen_one(
    provider: OpenRouterProvider,
    beat: Beat,
    style_notes: str,
    out_dir: Path,
) -> BeatArtifact:
    prompt = _augment_prompt(beat.image_prompt, style_notes)
    result = await provider.generate_image(
        prompt=prompt,
        model=IMAGE_MODEL,
        image_config={"aspect_ratio": "9:16"},
    )
    if not result.images:
        raise RuntimeError(f"image-gen: no image returned for beat {beat.idx}")

    out_path = out_dir / f"beat-{beat.idx:02d}.jpg"
    result.images[0].save(str(out_path))
    return BeatArtifact(idx=beat.idx, image_path=out_path)


async def generate_images(storyboard: Storyboard, out_dir: Path) -> list[BeatArtifact]:
    """Fan-out image generation across all beats. Failures are surfaced loudly."""
    out_dir.mkdir(parents=True, exist_ok=True)
    provider = OpenRouterProvider()

    results = await asyncio.gather(
        *(_gen_one(provider, b, storyboard.style_notes, out_dir) for b in storyboard.beats),
        return_exceptions=True,
    )
    errors = [r for r in results if isinstance(r, Exception)]
    if errors:
        raise RuntimeError(
            f"image-gen: {len(errors)}/{len(storyboard.beats)} beats failed. "
            f"First error: {errors[0]}"
        )
    return [r for r in results if isinstance(r, BeatArtifact)]
