"""Story comparison runner — runs ALL architectures on the same article.

  reel-af stories URL
    → navigate + distill once (shared work)
    → run arch A, B, E, H in parallel on the same ArticleSummary
    → dump side-by-side markdown to stories.md
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from reel_af.agents.architectures import ArchOutput
from reel_af.agents.architectures import (
    arch_a_pool,
    arch_b_hook_first,
    arch_e_reverse,
    arch_f_fewshot,
    arch_h_tournament,
    arch_i_hybrid,
    arch_j_debate,
)
from reel_af.agents.distiller import ArticleSummary, distill
from reel_af.agents.navigator import navigate

# Edit this list to subset which architectures run.
ARCHITECTURES = [
    arch_b_hook_first,    # B — won PG essay
    arch_f_fewshot,       # F — won TechCrunch
    arch_i_hybrid,        # I — B+H combined
    arch_j_debate,        # J — NEW: adversarial debate w/ judge
]


async def run_all(app: Any, url: str) -> tuple[ArticleSummary, list[ArchOutput]]:
    """Navigate + distill once, then run every architecture in parallel."""
    source = await navigate(app, url)
    summary = await distill(app, source)

    outputs = await asyncio.gather(
        *(mod.run(app, summary) for mod in ARCHITECTURES),
        return_exceptions=True,
    )

    # Surface failures as visible-but-empty entries so the comparison shows
    # which arch broke.
    cleaned: list[ArchOutput] = []
    for mod, out in zip(ARCHITECTURES, outputs):
        if isinstance(out, ArchOutput):
            cleaned.append(out)
        else:
            cleaned.append(
                ArchOutput(
                    arch_id="?",
                    arch_name=mod.__name__,
                    bet="(failed)",
                    draft=None,  # type: ignore[arg-type]
                    self_score=0,
                    wall_time_s=0,
                    trace=[f"FAILED: {type(out).__name__}: {out}"],
                )
            )
    return summary, cleaned


def render_markdown(url: str, summary: ArticleSummary, outputs: list[ArchOutput]) -> str:
    """Render a side-by-side comparison as a single markdown document."""
    lines: list[str] = []
    lines.append(f"# Story comparison — {summary.one_line_thesis}")
    lines.append("")
    lines.append(f"**Source:** {url}")
    lines.append(f"**Domain:** {summary.domain}")
    lines.append(f"**Takeaway:** {summary.intended_takeaway}")
    lines.append("")
    lines.append("## Article summary (shared input)")
    lines.append(f"**Thesis:** {summary.one_line_thesis}")
    lines.append("")
    lines.append("**Key points:**")
    for p in summary.key_points:
        lines.append(f"  - {p}")
    if summary.concrete_examples:
        lines.append("")
        lines.append("**Examples available:**")
        for e in summary.concrete_examples:
            lines.append(f"  - {e}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Per-architecture sections.
    for out in outputs:
        lines.append(f"## {out.arch_id}. {out.arch_name}")
        lines.append(f"**Bet:** {out.bet}")
        if out.draft is None:
            lines.append("")
            lines.append("```")
            lines.append("\n".join(out.trace))
            lines.append("```")
            lines.append("")
            continue
        lines.append("")
        wc = out.word_count()
        lines.append(
            f"- **self-score:** {out.self_score:.1f}/10  "
            f"**wall:** {out.wall_time_s:.1f}s  "
            f"**words:** {wc} (~{wc/2.6:.1f}s spoken)"
        )
        lines.append(
            f"- direction `{out.draft.direction}` · "
            f"hook `{out.draft.hook_trick}` · "
            f"retention `{out.draft.retention_trick}` · "
            f"close `{out.draft.close_trick}` · "
            f"tone `{out.draft.voice_tone}`"
        )
        lines.append("")
        lines.append("**Script:**")
        lines.append("")
        lines.append(f"> {out.draft.script}")
        lines.append("")
        lines.append("<details><summary>trace</summary>")
        lines.append("")
        lines.append("```")
        lines.extend(out.trace)
        lines.append("```")
        lines.append("</details>")
        lines.append("")
        lines.append("---")
        lines.append("")
    return "\n".join(lines)
