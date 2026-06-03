"""Pure-helper invariants downstream of the providers.

None of these touch a provider, but they sit on the same render path and
must keep behaving identically through the refactor. They double as a spec
for the audio/visual contract the README describes.
"""

from __future__ import annotations

from PIL import Image
from util import make_fake_provider, requires_ffmpeg, silence_pcm

# ───── tts text helpers ──────────────────────────────────────────────


def test_voice_for_tone_mapping():
    from reel_af.render.tts import voice_for_tone

    assert voice_for_tone("urgent") == "Charon"
    assert voice_for_tone("wonder") == "Kore"
    assert voice_for_tone("deadpan") == "Schedar"
    assert voice_for_tone("earnest") == "Aoede"
    assert voice_for_tone("playful") == "Puck"
    assert voice_for_tone("unknown-tone") == "Kore"  # documented default


def test_strip_tts_tags_removes_directions_and_collapses_space():
    from reel_af.render.tts import strip_tts_tags

    assert strip_tts_tags("[curious] Why   do [emphasis] we sleep?") == "Why do we sleep?"
    assert strip_tts_tags("No tags here.") == "No tags here."


def test_split_sentences_keeps_tags_and_splits_on_terminal_punct():
    from reel_af.render.tts import _split_sentences_with_tags

    sents = _split_sentences_with_tags("[curious] Why do we dream? Nobody knows — yet.")
    assert sents == ["[curious] Why do we dream?", "Nobody knows — yet."]
    # Em-dashes and commas do NOT split.
    assert _split_sentences_with_tags("A, b — c.") == ["A, b — c."]


# ───── video helpers ─────────────────────────────────────────────────


def test_motion_clause_static_vs_movement():
    from reel_af.render.video import _motion_clause

    assert _motion_clause("static") == "Camera: static, no movement"
    assert _motion_clause("slow_zoom_in") == "Camera: slow zoom in"
    assert _motion_clause("pan_left") == "Camera: pan left"


def test_image_to_data_url_normalises_jpg_to_jpeg(tmp_path):
    from reel_af.render.video import _image_to_data_url

    jpg = tmp_path / "frame.jpg"
    Image.new("RGB", (8, 8), (10, 20, 30)).save(jpg, format="JPEG")
    url = _image_to_data_url(jpg)
    assert url.startswith("data:image/jpeg;base64,")


def test_crop_to_9x16_outputs_native_vertical(tmp_path):
    from reel_af.render.images import _crop_to_9x16

    src = tmp_path / "src.png"
    Image.new("RGB", (1024, 1024), (200, 200, 200)).save(src, format="PNG")
    dest = _crop_to_9x16(src, tmp_path / "out.jpg")
    with Image.open(dest) as im:
        assert im.size == (720, 1280)


# ───── word timing (audio is master) ─────────────────────────────────


@requires_ffmpeg
async def test_word_timings_cover_every_spoken_word_in_order(tmp_path, monkeypatch):
    import reel_af.render.tts as tts

    fake = make_fake_provider(speech_pcm=silence_pcm(0.6))
    monkeypatch.setattr(tts, "OpenRouterProvider", fake)

    narration = "[curious] Why do we sleep? [emphasis] Nobody fully knows yet."
    full_path, timings = await tts.synthesize_audio(narration, "Kore", tmp_path)

    # One timing per *spoken* word — tags excluded, punctuation kept on words.
    expected_words = tts.strip_tts_tags(narration).split()
    assert [t.word for t in timings] == expected_words

    # Monotonic, non-overlapping, and bounded by the measured audio length.
    assert all(t.start_s <= t.end_s for t in timings)
    starts = [t.start_s for t in timings]
    assert starts == sorted(starts)
    assert timings[0].start_s >= 0
    assert full_path.exists()
