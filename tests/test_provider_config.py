"""Lock the documented provider/config contract (README "Customize" table).

These characterise the env-var → config resolution that the upcoming
"bring your own provider" change will refactor. They must keep passing
afterwards: the new behaviour is purely additive, so the OpenRouter
defaults and the already-documented overrides cannot regress.
"""

from __future__ import annotations

from util import run_config_probe

# README-documented defaults (Customize table + .env.example).
DEFAULT_MODEL = "openrouter/deepseek/deepseek-v4-pro"
DEFAULT_API_BASE = "https://openrouter.ai/api/v1"
DEFAULT_TTS = "google/gemini-3.1-flash-tts-preview"
DEFAULT_IMAGE = "openrouter/google/gemini-2.5-flash-image"
DEFAULT_VIDEO = "openrouter/google/veo-3.1-lite"

# Every model-selecting env var, cleared so we observe true defaults.
_CLEAR_MODELS = {
    "REEL_AF_MODEL": None,
    "REEL_AF_TTS_MODEL": None,
    "REEL_AF_IMAGE_MODEL": None,
    "REEL_AF_VIDEO_MODEL": None,
    "REEL_AF_USE_VEO": None,
}


def test_clean_env_resolves_documented_defaults():
    cfg = run_config_probe(_CLEAR_MODELS)
    assert cfg["model"] == DEFAULT_MODEL
    assert cfg["api_base"] == DEFAULT_API_BASE
    assert cfg["tts_default"] == DEFAULT_TTS
    assert cfg["image_model"] == DEFAULT_IMAGE
    assert cfg["video_model"] == DEFAULT_VIDEO
    assert cfg["use_veo"] is False


def test_api_key_sourced_from_openrouter_key():
    cfg = run_config_probe({**_CLEAR_MODELS, "OPENROUTER_API_KEY": "sk-probe-123"})
    assert cfg["api_key"] == "sk-probe-123"


def test_reel_af_model_override():
    # Exact example from the README quick-start.
    override = "openrouter/anthropic/claude-sonnet-4"
    cfg = run_config_probe({**_CLEAR_MODELS, "REEL_AF_MODEL": override})
    assert cfg["model"] == override
    # Overriding the reasoning model leaves the endpoint untouched.
    assert cfg["api_base"] == DEFAULT_API_BASE


def test_media_model_overrides():
    cfg = run_config_probe(
        {
            **_CLEAR_MODELS,
            "REEL_AF_TTS_MODEL": "google/some-tts",
            "REEL_AF_IMAGE_MODEL": "openrouter/black-forest-labs/flux-1.1-pro",
            "REEL_AF_VIDEO_MODEL": "openrouter/google/veo-3.1",
        }
    )
    assert cfg["tts_default"] == DEFAULT_TTS  # constant is the *fallback*, not the override
    assert cfg["image_model"] == "openrouter/black-forest-labs/flux-1.1-pro"
    assert cfg["video_model"] == "openrouter/google/veo-3.1"


def test_use_veo_truthy_values_enable_veo():
    for truthy in ("true", "TRUE", "True", "1", "yes"):
        cfg = run_config_probe({**_CLEAR_MODELS, "REEL_AF_USE_VEO": truthy})
        assert cfg["use_veo"] is True, truthy


def test_use_veo_falsey_values_keep_kenburns():
    for falsey in ("false", "0", "no", "", "off"):
        cfg = run_config_probe({**_CLEAR_MODELS, "REEL_AF_USE_VEO": falsey})
        assert cfg["use_veo"] is False, falsey
