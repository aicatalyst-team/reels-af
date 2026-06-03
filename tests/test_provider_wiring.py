"""Lock how the renderers drive the media provider.

Today every media call goes through ``OpenRouterProvider``. These tests
pin the *call contract* — which method, which model, which arguments — so
that when provider selection becomes configurable, the default OpenRouter
path is provably unchanged.
"""

from __future__ import annotations

from PIL import Image
from util import (
    make_beat,
    make_fake_provider,
    make_visual,
    requires_ffmpeg,
    silence_pcm,
    square_png_bytes,
)

# ───── TTS ───────────────────────────────────────────────────────────


@requires_ffmpeg
async def test_tts_calls_generate_speech_with_resolved_model(tmp_path, monkeypatch):
    import reel_af.render.tts as tts

    monkeypatch.delenv("REEL_AF_TTS_MODEL", raising=False)
    fake = make_fake_provider(speech_pcm=silence_pcm(0.4))
    monkeypatch.setattr(tts, "OpenRouterProvider", fake)

    path, dur = await tts.synthesize_audio_single(
        narration="[curious] Why do we dream?",
        voice="Kore",
        out_dir=tmp_path,
    )

    assert path.exists() and path.suffix == ".wav"
    assert dur > 0
    speech_calls = [kw for method, kw in fake.calls if method == "speech"]
    assert len(speech_calls) == 1
    kw = speech_calls[0]
    assert kw["model"] == tts.DEFAULT_TTS_MODEL          # falls back to default
    assert kw["voice"] == "Kore"
    assert kw["response_format"] == "pcm"                # dedicated TTS route
    assert kw["text"] == "[curious] Why do we dream?"    # tags passed through verbatim


@requires_ffmpeg
async def test_tts_respects_tts_model_env(tmp_path, monkeypatch):
    import reel_af.render.tts as tts

    monkeypatch.setenv("REEL_AF_TTS_MODEL", "google/custom-tts-model")
    fake = make_fake_provider(speech_pcm=silence_pcm(0.4))
    monkeypatch.setattr(tts, "OpenRouterProvider", fake)

    await tts.synthesize_audio_single("Hello there.", "Kore", tmp_path)

    speech_calls = [kw for method, kw in fake.calls if method == "speech"]
    assert speech_calls[0]["model"] == "google/custom-tts-model"


# ───── image ─────────────────────────────────────────────────────────


async def test_image_generation_uses_image_model_and_crops_9x16(tmp_path):
    import reel_af.render.images as images

    fake = make_fake_provider(image_data=square_png_bytes(512))
    out = await images.generate_first_frame(
        provider=fake(),
        image_prompt="a microscope on a lab bench",
        idx=0,
        out_dir=tmp_path,
    )

    assert out.exists()
    with Image.open(out) as im:
        assert im.size == (720, 1280)   # documented Veo-native vertical res

    image_calls = [kw for method, kw in fake.calls if method == "image"]
    assert len(image_calls) == 1
    assert image_calls[0]["model"] == images.IMAGE_MODEL
    assert image_calls[0]["n"] == 1


async def test_image_prompt_carries_content_mode_style(tmp_path):
    import reel_af.render.images as images

    fake = make_fake_provider(image_data=square_png_bytes(256))
    await images.generate_first_frame(
        provider=fake(),
        image_prompt="a plot",
        idx=1,
        out_dir=tmp_path,
        content_mode="scientific",
    )
    prompt = [kw for m, kw in fake.calls if m == "image"][0]["prompt"]
    assert "research lab" in prompt          # scientific style note appended
    assert prompt.startswith("a plot")


# ───── video ─────────────────────────────────────────────────────────


@requires_ffmpeg
async def test_video_kenburns_default_makes_no_veo_call(tmp_path, monkeypatch):
    import reel_af.render.video as video

    monkeypatch.setattr(video, "USE_VEO", False)
    fake = make_fake_provider(image_data=square_png_bytes(512))
    monkeypatch.setattr(video, "OpenRouterProvider", fake)

    artifacts = await video.generate_beat_videos(
        beats=[make_beat(0), make_beat(1)],
        visuals=[make_visual(), make_visual()],
        out_dir=tmp_path,
    )

    assert len(artifacts) == 2
    for art in artifacts:
        assert art.video_path.exists() and art.video_path.suffix == ".mp4"
        assert art.video_path.stat().st_size > 0
    assert not any(method == "video" for method, _ in fake.calls)  # no Veo when off


async def test_video_veo_call_shape_when_enabled(tmp_path, monkeypatch):
    import reel_af.render.video as video

    monkeypatch.setattr(video, "USE_VEO", True)
    fake = make_fake_provider(
        image_data=square_png_bytes(512), video_bytes=b"FAKE-MP4-BYTES"
    )
    monkeypatch.setattr(video, "OpenRouterProvider", fake)

    artifacts = await video.generate_beat_videos(
        beats=[make_beat(0, veo_duration=6)],
        visuals=[make_visual("slow_zoom_in")],
        out_dir=tmp_path,
    )

    assert artifacts[0].video_path.read_bytes() == b"FAKE-MP4-BYTES"
    video_calls = [kw for method, kw in fake.calls if method == "video"]
    assert len(video_calls) == 1
    kw = video_calls[0]
    assert kw["model"] == video.VIDEO_MODEL
    assert kw["first_frame"].startswith("data:image/")  # frame sent as data URL
    assert kw["duration"] == 6


@requires_ffmpeg
async def test_video_image_failure_falls_back_to_placeholder(tmp_path, monkeypatch):
    import reel_af.render.video as video

    monkeypatch.setattr(video, "USE_VEO", False)
    fake = make_fake_provider(image_error=RuntimeError("image gen down"))
    monkeypatch.setattr(video, "OpenRouterProvider", fake)

    artifacts = await video.generate_beat_videos(
        beats=[make_beat(0)], visuals=[make_visual()], out_dir=tmp_path
    )
    # A single failed image must not crash the reel — a clip still exists.
    assert artifacts[0].video_path.exists()
    assert artifacts[0].video_path.stat().st_size > 0


@requires_ffmpeg
async def test_video_veo_failure_falls_back_to_kenburns(tmp_path, monkeypatch):
    import reel_af.render.video as video

    monkeypatch.setattr(video, "USE_VEO", True)
    fake = make_fake_provider(
        image_data=square_png_bytes(512), video_error=RuntimeError("veo 401")
    )
    monkeypatch.setattr(video, "OpenRouterProvider", fake)

    artifacts = await video.generate_beat_videos(
        beats=[make_beat(0)], visuals=[make_visual()], out_dir=tmp_path
    )
    # Veo failed, but the still-frame ken-burns fallback produced a clip.
    assert artifacts[0].video_path.exists()
    assert artifacts[0].video_path.stat().st_size > 0
    assert any(method == "video" for method, _ in fake.calls)  # it *did* try Veo
