"""Source Navigator — fetches a URL, cleans it, extracts structure.

Pattern from CLAUDE.md: deterministic work belongs in code, intelligence
belongs in agents. URL fetching + HTML cleaning is deterministic, so it
runs as pure code (aiohttp + readability). The intelligence step — picking
key claims, guessing audience, scoring surprise — is a single .ai() call
on the already-cleaned body (which fits comfortably in the context).
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

import aiohttp
from pydantic import BaseModel, Field
from readability import Document

from reel_af.models import SourceContent

# Hard caps so a hostile URL can't blow up the pipeline.
FETCH_TIMEOUT_S = 30.0
MAX_BODY_CHARS = 50_000
USER_AGENT = "reel-af/0.1 (+https://github.com/Agent-Field/agentfield)"


class _Extracted(BaseModel):
    """Flat schema the LLM fills out for one source. Kept tiny per .ai() rules."""

    key_claims: list[str] = Field(
        ..., description="3-5 surprising, quotable, scroll-stop claims pulled verbatim."
    )
    audience_hints: str = Field(
        ..., description="One sentence: who is this content for?"
    )
    surprise_score: int = Field(
        ..., ge=1, le=10, description="How scroll-stop-worthy this content is."
    )


async def _fetch(url: str) -> tuple[str, str]:
    """Fetch URL, return (raw_html, final_url) — caller cleans."""
    timeout = aiohttp.ClientTimeout(total=FETCH_TIMEOUT_S)
    headers = {"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"}
    async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
        async with session.get(url, allow_redirects=True, max_redirects=5) as resp:
            resp.raise_for_status()
            text = await resp.text(errors="replace")
            return text, str(resp.url)


def _clean(html: str) -> tuple[str, str]:
    """Run readability to get a clean title + body. Pure CPU."""
    doc = Document(html)
    title = (doc.short_title() or doc.title() or "").strip()
    content_html = doc.summary(html_partial=True)
    # Strip tags, collapse whitespace.
    text = re.sub(r"<[^>]+>", " ", content_html)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > MAX_BODY_CHARS:
        text = text[:MAX_BODY_CHARS]
    return title, text


async def navigate(app: Any, url: str) -> SourceContent:
    """Public entry point. Returns a SourceContent ready for angle proposers."""
    html, final_url = await _fetch(url)
    title, body = await asyncio.get_event_loop().run_in_executor(None, _clean, html)

    if not body:
        raise RuntimeError(f"Navigator: could not extract readable text from {url}")

    # One structured-output call — cheap, fast, covers the intelligence work.
    extracted = await app.ai(
        system=(
            "You're picking the most scroll-stop-worthy material from web content for a "
            "short-form vertical video. Return literal verbatim claims from the body "
            "(no paraphrasing) — pick the 3-5 most surprising or quotable. Then guess "
            "the audience in one sentence and rate the source's surprise quality 1-10."
        ),
        user=f"TITLE: {title}\n\nBODY:\n{body[:12_000]}",
        schema=_Extracted,
    )

    return SourceContent(
        url=final_url,
        title=title or final_url,
        body=body,
        key_claims=extracted.key_claims,
        audience_hints=extracted.audience_hints,
        surprise_score=extracted.surprise_score,
    )
