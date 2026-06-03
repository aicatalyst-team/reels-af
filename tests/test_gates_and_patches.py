"""API-key gates and the SDK runtime patches.

The provider refactor will relax the hard ``OPENROUTER_API_KEY`` requirement,
so these pin the *current* gate behaviour to make any change deliberate and
visible. The patch tests guard the end-to-end-callability fixes the README
depends on (data-URL images, video-download auth, the /audio/speech route).
"""

from __future__ import annotations

import base64
import io

import pytest
from PIL import Image

# ───── missing-key gates ─────────────────────────────────────────────


async def test_article_to_reel_errors_without_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    import reel_af.app as app

    result = await app.article_to_reel(url="https://example.com/x")
    assert result == {"error": "OPENROUTER_API_KEY not set in env."}


async def test_topic_to_reel_errors_without_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    import reel_af.app as app

    result = await app.topic_to_reel(topic="the placebo effect")
    assert result == {"error": "OPENROUTER_API_KEY not set in env."}


def test_cli_require_key_raises_when_unset(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    import reel_af.cli as cli

    with pytest.raises(SystemExit):
        cli._require_key()


def test_cli_require_key_passes_when_set(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    import reel_af.cli as cli

    cli._require_key()  # must not raise


# ───── sdk patches ───────────────────────────────────────────────────


def test_generate_speech_method_is_added():
    from agentfield.media_providers import OpenRouterProvider

    import reel_af.sdk_patches  # noqa: F401

    assert hasattr(OpenRouterProvider, "generate_speech")


def test_generate_video_is_patched():
    from agentfield.media_providers import OpenRouterProvider

    import reel_af.sdk_patches  # noqa: F401

    assert getattr(OpenRouterProvider.generate_video, "__reel_af_patched__", False) is True


def test_image_output_save_decodes_data_urls(tmp_path):
    from agentfield.multimodal_response import ImageOutput

    import reel_af.sdk_patches  # noqa: F401

    # A 4x4 red PNG encoded as a data: URL — Gemini returns these.
    buf = io.BytesIO()
    Image.new("RGB", (4, 4), (255, 0, 0)).save(buf, format="PNG")
    raw = buf.getvalue()
    data_url = "data:image/png;base64," + base64.b64encode(raw).decode()

    out = tmp_path / "decoded.png"
    ImageOutput(url=data_url).save(out)

    assert out.read_bytes() == raw  # decoded locally, no network fetch


def test_apply_all_is_idempotent():
    from agentfield.media_providers import OpenRouterProvider

    import reel_af.sdk_patches as patches

    before = OpenRouterProvider.generate_video
    patches.apply_all()  # second application
    patches.apply_all()  # third
    after = OpenRouterProvider.generate_video

    assert after is before  # no re-wrapping
    assert getattr(after, "__reel_af_patched__", False) is True
    assert hasattr(OpenRouterProvider, "generate_speech")
