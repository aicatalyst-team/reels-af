"""TTS fan-out — one Kokoro call per beat, run in parallel with image-gen.

Uses the OpenRouter /audio/speech path that shipped in PR #579. We request
WAV format and the SDK wraps the raw PCM client-side, so each beat ends up
as a playable per-beat .wav ready for ffmpeg to overlay.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from agentfield.media_providers import OpenRouterProvider

from reel_af.models import Beat, BeatArtifact, Storyboard

TTS_MODEL = "openrouter/hexgrad/kokoro-82m"
DEFAULT_VOICE = "af_bella"


async def _tts_one(
    provider: OpenRouterProvider,
    beat: Beat,
    voice: str,
    out_dir: Path,
) -> BeatArtifact:
    result = await provider.generate_audio(
        text=beat.vo_line,
        model=TTS_MODEL,
        voice=voice,
        format="wav",
    )
    if not result.has_audio:
        raise RuntimeError(f"tts: no audio returned for beat {beat.idx}")

    out_path = out_dir / f"beat-{beat.idx:02d}.wav"
    result.audio.save(str(out_path))
    return BeatArtifact(idx=beat.idx, audio_path=out_path)


async def generate_audio(storyboard: Storyboard, out_dir: Path) -> list[BeatArtifact]:
    out_dir.mkdir(parents=True, exist_ok=True)
    voice = os.environ.get("REEL_AF_TTS_VOICE", DEFAULT_VOICE)
    provider = OpenRouterProvider()

    results = await asyncio.gather(
        *(_tts_one(provider, b, voice, out_dir) for b in storyboard.beats),
        return_exceptions=True,
    )
    errors = [r for r in results if isinstance(r, Exception)]
    if errors:
        raise RuntimeError(
            f"tts: {len(errors)}/{len(storyboard.beats)} beats failed. "
            f"First error: {errors[0]}"
        )
    return [r for r in results if isinstance(r, BeatArtifact)]
