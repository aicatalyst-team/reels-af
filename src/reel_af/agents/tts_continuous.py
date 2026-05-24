"""Continuous TTS — one TTS call for the full script, split on silences.

This solves TWO problems the per-segment approach had:

(1) DISCRETE WORDS — per-segment calls have no prosody continuity, so every
    sentence dies flat at the period. Generating the whole script in one
    call carries intonation across sentence boundaries.

(2) MODEL RESPONDS INSTEAD OF READING — the SDK's generate_audio helper
    sends ONLY a user message, so gpt-audio-mini treats it as a chat
    prompt and may RESPOND ("Sure, I can help with that…") instead of
    reading it. We bypass the SDK here and call OpenRouter directly with a
    SYSTEM message that tells the model to read verbatim, plus EMOTION
    direction (pace, emphasis, where to pause).

Approach:
  1. Direct chat-completions call with system + user messages, audio modality.
  2. Stream the SSE response; accumulate base64 PCM16 chunks.
  3. Wrap raw PCM as WAV with our SDK helper.
  4. Use ffmpeg `silencedetect` to find sentence-boundary silences.
  5. Split at silence midpoints; assign per-scene clips.
  6. Fall back to proportional-by-word-count splits if silence count is off.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Optional

import aiohttp

# SDK helper for wrapping raw PCM16 → WAV (the one we built in PR #579).
from agentfield.media_providers import _wrap_pcm16_bytes_as_wav

from reel_af.agents.scene_breaker import Scene
from reel_af.models import BeatArtifact

# Default model: gpt-audio-mini via chat-completions audio modality.
DEFAULT_TTS_MODEL = "openrouter/openai/gpt-audio-mini"
OPENROUTER_BASE = "https://openrouter.ai/api/v1"

# OpenAI audio TTS sample rate (gpt-audio family streams PCM16 @ 24kHz).
TTS_SAMPLE_RATE = 24000

# Voice map by tone — gpt-audio supports the OpenAI TTS voice set.
_VOICE_BY_TONE: dict[str, str] = {
    "urgent":  "onyx",      # deep, serious — gravitas
    "wonder":  "nova",      # warm, curious
    "deadpan": "echo",      # neutral, dry
    "earnest": "alloy",     # friendly, warm
    "playful": "shimmer",   # bright, conversational
}


def voice_for_tone(tone: str) -> str:
    """Pick a gpt-audio voice that matches the script's tone."""
    return _VOICE_BY_TONE.get(tone, "nova")


# ───── Direct OpenRouter chat-audio call (with system prompt) ────────


def _build_narrator_system_prompt(tone: str) -> str:
    """The system message that turns a chat model into a narrator.

    CRITICAL: explicitly tells the model not to respond to the user text —
    just READ it. Plus emotion direction the model uses to shape delivery.
    """
    return f"""You are a professional voiceover artist reading a vertical
short-form video narration aloud.

ABSOLUTE RULE: Read the user message EXACTLY AS WRITTEN, word-for-word.
DO NOT respond to it, greet, explain, apologize, comment, or add ANY
words of your own. The user message IS the script. Read it.

DELIVERY DIRECTION:
  • Tone: {tone}. Match this emotional register throughout.
  • Pace: vary deliberately. Open with energy. Slow on key revelations.
    Bring the final sentence in WEIGHTY and SLOW — let it LAND.
  • Em-dashes (—): a deliberate pause, then emphasise what follows.
  • Single-word periods ("Wrong.", "Dead.", "Down."): punchy beats. Brief,
    impactful, then pause.
  • ALL CAPS WORDS: lean in — slightly higher pitch + volume + intent.
  • Commas: small breath. Don't ignore them; they're pacing.
  • Question marks: rising inflection that genuinely asks.

You're not reading documentation. You're a narrator telling a story for a
20-second vertical video that needs to STOP a thumb mid-scroll. Bring
real performance. Vary your pitch and pace like a human would."""


async def _direct_chat_audio(
    full_script: str,
    voice: str,
    tone: str,
    model: str,
    api_key: str,
    timeout: float = 300.0,
) -> bytes:
    """Direct call to OpenRouter /chat/completions with audio modality.

    Returns raw PCM16 bytes (24kHz mono) that the caller wraps in WAV.

    Why direct instead of SDK: SDK's generate_audio sends only a user
    message, so chat models RESPOND instead of READ. We need a system
    message for both correctness (read verbatim) AND emotion.
    """
    payload = {
        "model": model.removeprefix("openrouter/"),
        "messages": [
            {"role": "system", "content": _build_narrator_system_prompt(tone)},
            {"role": "user", "content": full_script},
        ],
        "modalities": ["text", "audio"],
        "audio": {"voice": voice, "format": "pcm16"},
        "stream": True,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    client_timeout = aiohttp.ClientTimeout(total=timeout)

    pcm_chunks: list[bytes] = []
    async with aiohttp.ClientSession(timeout=client_timeout) as session:
        async with session.post(
            f"{OPENROUTER_BASE}/chat/completions", json=payload, headers=headers,
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise RuntimeError(
                    f"tts_continuous: chat-audio call failed ({resp.status}): "
                    f"{body[:500]}"
                )
            # Stream SSE; accumulate audio delta chunks.
            buf = b""
            done = False
            async for raw in resp.content.iter_any():
                if done:
                    break
                buf += raw
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    line_str = line.decode("utf-8", errors="replace").strip()
                    if not line_str.startswith("data: "):
                        continue
                    data = line_str[6:]
                    if data == "[DONE]":
                        done = True
                        break
                    try:
                        event = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    for choice in event.get("choices", []):
                        delta = choice.get("delta", {})
                        audio_delta = delta.get("audio") or {}
                        b64 = audio_delta.get("data")
                        if b64:
                            try:
                                pcm_chunks.append(base64.b64decode(b64))
                            except Exception:
                                continue

    if not pcm_chunks:
        raise RuntimeError("tts_continuous: no audio chunks returned by model")
    return b"".join(pcm_chunks)


# ───── Silence-detection helpers ─────────────────────────────────────


def _detect_silences(audio_path: Path, threshold_db: int = -32, min_dur: float = 0.25) -> list[tuple[float, float]]:
    """Run ffmpeg silencedetect; return list of (silence_start, silence_end) tuples."""
    cmd = [
        "ffmpeg", "-i", str(audio_path),
        "-af", f"silencedetect=noise={threshold_db}dB:d={min_dur}",
        "-f", "null", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    # silencedetect writes results to stderr in the form:
    #   [silencedetect @ 0x…] silence_start: 3.14
    #   [silencedetect @ 0x…] silence_end: 3.52
    starts = [float(m) for m in re.findall(r"silence_start: ([\d.]+)", proc.stderr)]
    ends = [float(m) for m in re.findall(r"silence_end: ([\d.]+)", proc.stderr)]
    return list(zip(starts, ends))


def _probe_duration(audio_path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(audio_path)],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip())


def _proportional_splits(scenes: list[Scene], total_duration: float) -> list[float]:
    """Fallback: split times by per-scene word-count proportion."""
    total_words = sum(max(len(s.sentence.split()), 1) for s in scenes)
    cuts: list[float] = []
    acc = 0
    for s in scenes[:-1]:
        acc += max(len(s.sentence.split()), 1)
        cuts.append(total_duration * acc / total_words)
    return cuts


def _pick_split_points(
    scenes: list[Scene],
    silences: list[tuple[float, float]],
    total_duration: float,
) -> list[float]:
    """Pick (n_scenes - 1) split points — silence midpoints when possible."""
    n_needed = len(scenes) - 1
    if n_needed == 0:
        return []
    # Use silence MIDPOINTS as split points so we cut between words, not on top of one.
    silence_mids = [(start + end) / 2 for start, end in silences]
    if len(silence_mids) >= n_needed:
        # If the model emitted MORE silences than sentences (e.g. internal commas),
        # pick the n_needed silences whose times best match the word-proportional split.
        target_cuts = _proportional_splits(scenes, total_duration)
        chosen: list[float] = []
        used = set()
        for t in target_cuts:
            best_idx = min(
                (i for i in range(len(silence_mids)) if i not in used),
                key=lambda i: abs(silence_mids[i] - t),
                default=None,
            )
            if best_idx is None:
                chosen.append(t)
            else:
                chosen.append(silence_mids[best_idx])
                used.add(best_idx)
        return sorted(chosen)
    # Not enough silences — fall back to proportional.
    return _proportional_splits(scenes, total_duration)


async def _split_audio(
    full_audio: Path,
    cuts: list[float],
    total_duration: float,
    out_dir: Path,
    n_scenes: int,
) -> list[Path]:
    """Cut the full audio into per-scene WAVs at the given cut points."""
    bounds = [0.0] + cuts + [total_duration]
    out_paths: list[Path] = []
    tasks = []
    for i in range(n_scenes):
        start, end = bounds[i], bounds[i + 1]
        out_path = out_dir / f"seg-{i:02d}.wav"
        out_paths.append(out_path)
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(full_audio),
            "-ss", f"{start:.3f}",
            "-to", f"{end:.3f}",
            "-c", "copy",
            str(out_path),
        ]
        tasks.append(asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        ))
    procs = await asyncio.gather(*tasks)
    for i, proc in enumerate(procs):
        _, err = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(
                f"tts_continuous: failed to extract segment {i}: "
                f"{err.decode(errors='replace')[-300:]}"
            )
    return out_paths


# ───── Public entrypoint ────────────────────────────────────────────


async def generate_continuous_audio(
    full_script: str,
    scenes: list[Scene],
    voice: str,
    out_dir: Path,
    tone: str = "wonder",
    model: Optional[str] = None,
) -> tuple[list[BeatArtifact], Path]:
    """Generate the full script as ONE TTS call, split into per-scene WAVs.

    Returns (per-scene artifacts, full continuous wav path) so callers can
    debug by listening to the full continuous track.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    tts_model = model or os.environ.get("REEL_AF_TTS_MODEL", DEFAULT_TTS_MODEL)
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set")

    # Direct OpenRouter call WITH a system prompt — telling the model to
    # READ verbatim (not respond to the user message) and giving it
    # explicit emotion direction.
    pcm = await _direct_chat_audio(
        full_script=full_script,
        voice=voice,
        tone=tone,
        model=tts_model,
        api_key=api_key,
    )
    full_audio_path = out_dir / "full.wav"
    wav_bytes = _wrap_pcm16_bytes_as_wav(pcm, sample_rate=TTS_SAMPLE_RATE)
    full_audio_path.write_bytes(wav_bytes)

    # Find silences and split.
    total_dur = _probe_duration(full_audio_path)
    silences = _detect_silences(full_audio_path)
    cuts = _pick_split_points(scenes, silences, total_dur)
    seg_paths = await _split_audio(
        full_audio_path, cuts, total_dur, out_dir, n_scenes=len(scenes)
    )

    artifacts = [
        BeatArtifact(idx=s.idx, audio_path=p)
        for s, p in zip(scenes, seg_paths)
    ]
    return artifacts, full_audio_path
