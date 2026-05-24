"""Script Segmenter — splits the full narration into per-beat segments.

Pure code, no LLM. Each micro-beat from the narrative architect becomes one
segment; longer beats can be split further at em-dash or comma boundaries.
Returns segments paired with estimated speaking duration (words / 2.6 s).

We segment by *speech beats* rather than asking the LLM to label its own
segments because the LLM-derived structure is already there — we just slice
on it. Pure code is faster, deterministic, and easier to debug.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from reel_af.agents.narrative_architect import NarrativeScript

# Average Kokoro speaking rate. Used to estimate per-segment duration; the
# real audio length is taken from the .wav at assembly time, but having an
# estimate lets the shot director pre-budget durations for Veo.
WORDS_PER_SECOND = 2.6
MIN_SEG_DURATION = 1.8
MAX_SEG_DURATION = 4.5


@dataclass
class Segment:
    idx: int
    text: str
    role: str                # hook | stakes | revelation | consequence | callback
    est_duration_s: float


def _split_long(text: str, role: str, max_words: int = 11) -> list[str]:
    """If a beat is long, split at em-dash or comma boundaries."""
    words = text.split()
    if len(words) <= max_words:
        return [text]

    # Prefer em-dash splits (architect uses these for dramatic pauses).
    parts = re.split(r"\s*—\s*", text)
    if len(parts) > 1:
        # Re-attach em-dash to the front of each part except the first so
        # Kokoro still produces the pause; or just let segmenter dropping
        # it be fine since the per-segment boundary IS the pause.
        return [p.strip() for p in parts if p.strip()]

    # Fall back to comma splits at the longest internal comma.
    parts = re.split(r",\s*", text)
    if len(parts) > 1:
        return [p.strip().rstrip(",") + "," if i < len(parts) - 1 else p.strip()
                for i, p in enumerate(parts)]

    # No natural split — return as one segment.
    return [text]


def _estimate_duration(text: str) -> float:
    """Estimate spoken duration in seconds, clamped to [MIN, MAX]."""
    words = max(len(text.split()), 1)
    raw = words / WORDS_PER_SECOND
    return max(MIN_SEG_DURATION, min(raw, MAX_SEG_DURATION))


def segment_script(script: NarrativeScript) -> list[Segment]:
    """Turn the 5-beat script into 5-8 timed segments."""
    raw: list[tuple[str, str]] = [
        ("hook", script.hook),
        ("stakes", script.stakes),
        ("revelation", script.revelation),
        ("consequence", script.consequence),
        ("callback", script.callback),
    ]

    segments: list[Segment] = []
    idx = 0
    for role, text in raw:
        text = text.strip()
        if not text:
            continue
        for piece in _split_long(text, role):
            segments.append(
                Segment(
                    idx=idx,
                    text=piece,
                    role=role,
                    est_duration_s=_estimate_duration(piece),
                )
            )
            idx += 1
    return segments
