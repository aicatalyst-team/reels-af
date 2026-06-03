"""Tier A — bring-your-own reasoning endpoint.

Locks the additive override path introduced by ``REEL_AF_API_BASE`` /
``REEL_AF_API_KEY``. These coexist with the OpenRouter defaults pinned in
``test_provider_config.py``: with both vars unset, config resolves to
OpenRouter exactly as before. Each test maps to a validation-contract item.
"""

from __future__ import annotations

from util import run_config_probe

DEFAULT_API_BASE = "https://openrouter.ai/api/v1"

# Clear the BYO vars by default so the host environment can't leak in; each
# test sets only what it exercises.
_CLEAR = {"REEL_AF_API_BASE": None, "REEL_AF_API_KEY": None}


def test_api_base_override():
    # CA2 — a custom endpoint is honoured verbatim.
    cfg = run_config_probe({**_CLEAR, "REEL_AF_API_BASE": "http://localhost:8000/v1"})
    assert cfg["api_base"] == "http://localhost:8000/v1"


def test_api_key_override_takes_precedence_over_openrouter_key():
    # CA3 — REEL_AF_API_KEY wins over OPENROUTER_API_KEY when both are set.
    cfg = run_config_probe(
        {**_CLEAR, "OPENROUTER_API_KEY": "sk-openrouter", "REEL_AF_API_KEY": "sk-byo-endpoint"}
    )
    assert cfg["api_key"] == "sk-byo-endpoint"


def test_api_key_falls_back_to_openrouter_when_byo_unset():
    # CA4 — without REEL_AF_API_KEY, OPENROUTER_API_KEY still supplies the key.
    cfg = run_config_probe({**_CLEAR, "OPENROUTER_API_KEY": "sk-openrouter"})
    assert cfg["api_key"] == "sk-openrouter"


def test_empty_api_base_falls_back_to_openrouter():
    # CA5 — Docker passes `${REEL_AF_API_BASE:-}` as "" when unset; an empty
    # string must NOT clobber the default endpoint.
    cfg = run_config_probe({**_CLEAR, "REEL_AF_API_BASE": ""})
    assert cfg["api_base"] == DEFAULT_API_BASE
