"""Caption fitting — measure ACTUAL rendered text width with Pillow,
then pick a fontsize that fits within the safe area.

Why this exists: sizing captions by character count is wrong. "WWW" is way
wider than "iii" at the same length. Pillow has the same font file the
ffmpeg drawtext filter will use, so measurements match what gets rendered.

Two-line auto-wrap: when even the smallest single-line fontsize would still
overflow, we split into 2 balanced lines at a word boundary and use the
single-line measurement on the longer half.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from PIL import ImageFont

# Safe text area = full width minus a generous margin per side. 1080-wide
# frame → 120px margins → 840px usable. This keeps text clear of platform
# UI overlays and gives the eye some air.
SAFE_MARGIN_PX = 120
TARGET_W = 1080

# Fontsize ladder we'll try, biggest first. Stops at 56pt — smaller than
# that is illegible at thumbnail size, so we'd rather wrap to 2 lines.
SIZE_LADDER: tuple[int, ...] = (132, 120, 108, 96, 86, 78, 70, 62, 56)
MIN_SIZE = SIZE_LADDER[-1]


@dataclass
class CaptionLayout:
    """Result of fitting a caption to the safe text area."""

    lines: list[str]        # 1 or 2 lines, each fits the safe width
    fontsize: int           # ffmpeg drawtext fontsize


@lru_cache(maxsize=64)
def _font(font_path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(font_path, size)


def _text_width(text: str, font: ImageFont.FreeTypeFont) -> int:
    """Pixel width of `text` rendered with `font` (no anchor padding)."""
    bbox = font.getbbox(text)
    return bbox[2] - bbox[0]


def _split_two_lines(text: str) -> tuple[str, str]:
    """Split `text` at the word boundary that produces the most-balanced
    two lines (closest to equal length)."""
    words = text.split()
    if len(words) < 2:
        return text, ""
    best_diff = float("inf")
    best_split = 1
    total_len = sum(len(w) for w in words)
    for i in range(1, len(words)):
        left = sum(len(words[j]) for j in range(i)) + (i - 1)
        right = total_len - left + (len(words) - i - 1)
        diff = abs(left - right)
        if diff < best_diff:
            best_diff = diff
            best_split = i
    return " ".join(words[:best_split]), " ".join(words[best_split:])


def fit_caption(text: str, font_path: str) -> CaptionLayout:
    """Pick the biggest fontsize from the ladder that fits within the safe
    width. Falls back to a two-line layout if even MIN_SIZE overflows."""
    safe_w = TARGET_W - 2 * SAFE_MARGIN_PX

    # Try single-line at each ladder size, biggest first.
    for size in SIZE_LADDER:
        if _text_width(text, _font(font_path, size)) <= safe_w:
            return CaptionLayout(lines=[text], fontsize=size)

    # Single-line at MIN_SIZE still overflows → wrap.
    left, right = _split_two_lines(text)
    # Pick the largest size at which BOTH halves fit.
    for size in SIZE_LADDER:
        font = _font(font_path, size)
        if (
            _text_width(left, font) <= safe_w
            and _text_width(right, font) <= safe_w
        ):
            return CaptionLayout(lines=[left, right], fontsize=size)

    # Last resort — return the wrapped layout at MIN_SIZE even if one half
    # still overflows. Better to clip a little than crash.
    return CaptionLayout(lines=[left, right], fontsize=MIN_SIZE)
