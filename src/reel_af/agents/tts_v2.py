"""TTS v2 — per-segment Kokoro with tone-driven voice routing.

Differences from v0:
  • Voice is chosen per reel based on the take's voice_tone, not hardcoded.
  • Each segment becomes its own .wav so the final assembly can sync them
    to per-segment Veo videos (not one continuous stream).
  • The text passed to Kokoro is the exact segment text — punctuation
    intact, including em-dashes (Kokoro respects them for dramatic pauses).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from agentfield.media_providers import OpenRouterProvider

from reel_af.agents.scene_breaker import Scene
from reel_af.models import BeatArtifact

TTS_MODEL = "openrouter/hexgrad/kokoro-82m"

# Voice routing: pick a Kokoro voice that matches the script's tone. These
# are all standard Kokoro voice ids on OpenRouter.
_VOICE_BY_TONE: dict[str, str] = {
    "urgent":   "am_michael",   # deeper male voice — gravitas
    "wonder":   "af_nicole",    # warm female voice — discovery feel
    "deadpan":  "am_adam",      # flatter male voice — irony
    "earnest":  "af_bella",     # warm female voice — coaching
    "playful":  "af_sarah",     # brighter female voice — fun
}


def voice_for_tone(tone: str) -> str:
    return _VOICE_BY_TONE.get(tone, "af_bella")


# Per-role speech speed — slows the close for emphasis, slight lift on the
# hook for energy. Bodies stay at 1.0. Real creators do this with takes;
# we do it with TTS rate control.
_SPEED_BY_ROLE: dict[str, float] = {
    "hook":        1.02,   # slight lift — energy
    "stakes":      1.00,
    "revelation":  1.00,
    "consequence": 0.97,   # leaning into the implication
    "callback":    0.92,   # the close LANDS
}


async def _tts_one(
    provider: OpenRouterProvider,
    seg: Scene,
    voice: str,
    out_dir: Path,
) -> BeatArtifact:
    speed = _SPEED_BY_ROLE.get(seg.role, 1.0)
    result = await provider.generate_audio(
        text=seg.sentence,
        model=TTS_MODEL,
        voice=voice,
        format="wav",
        speed=speed,
    )
    if not result.has_audio:
        raise RuntimeError(f"tts_v2: kokoro returned no audio for segment {seg.idx}")
    out = out_dir / f"seg-{seg.idx:02d}.wav"
    result.audio.save(str(out))
    return BeatArtifact(idx=seg.idx, audio_path=out)


async def generate_audio_v2(
    segments: list[Scene],
    voice: str,
    out_dir: Path,
) -> list[BeatArtifact]:
    out_dir.mkdir(parents=True, exist_ok=True)
    provider = OpenRouterProvider()
    results = await asyncio.gather(
        *(_tts_one(provider, s, voice, out_dir) for s in segments),
        return_exceptions=True,
    )
    errs = [r for r in results if isinstance(r, Exception)]
    if errs:
        raise RuntimeError(f"tts_v2: {len(errs)}/{len(segments)} failed. First: {errs[0]}")
    return [r for r in results if isinstance(r, BeatArtifact)]
