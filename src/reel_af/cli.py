"""reel-af CLI — turn a URL or a topic into a vertical reel.

Two subcommands mirror the two entry reasoners:

  reel-af article URL                # article_to_reel
  reel-af topic "topic phrase"       # topic_to_reel

The CLI runs the orchestrator function directly (no AgentField server
required). For production use with the control-plane DAG, run
``python -m reel_af.app`` and POST to the ``/execute/async/...`` API.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Annotated, Optional

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_PROJECT_ROOT / ".env")

app = typer.Typer(
    name="reel-af",
    help="URL or topic → vertical reel (OpenRouter only).",
    no_args_is_help=True,
    add_completion=False,
)
console = Console(stderr=True)


def _require_key() -> None:
    if "OPENROUTER_API_KEY" not in os.environ:
        raise SystemExit("OPENROUTER_API_KEY not set (put it in .env)")


def _summarize(result: dict, run_id: str, out_path: Path) -> None:
    table = Table(title=f"reel-af run {run_id}", show_header=False)
    table.add_column("field", style="bold cyan")
    table.add_column("value")
    for k in (
        "source", "video_path", "duration_s", "beat_count", "card_count",
        "accent_count", "hook", "tease", "topic", "url",
        "content_mode", "domain", "voice_id",
    ):
        if k in result and result[k] is not None:
            table.add_row(k, str(result[k]))
    console.print(table)
    if "timings_s" in result:
        console.print(
            Panel(
                "\n".join(
                    f"  {k:14s}{v:>6.1f}s"
                    for k, v in result["timings_s"].items()
                ),
                title="timings",
                border_style="dim",
            )
        )
    sidecar = out_path / "result.json"
    sidecar.write_text(json.dumps(result, indent=2, default=str))
    console.print(f"\n[green]→ {result.get('video_path')}[/green]")
    console.print(f"[dim]   result.json → {sidecar}[/dim]")


@app.command("article")
def article(
    url: Annotated[str, typer.Argument(help="The article URL.")],
    out_dir: Annotated[
        Optional[Path],
        typer.Option("--out", help="Output directory.", show_default=False),
    ] = None,
) -> None:
    """Turn an article URL into a vertical viral reel."""
    _require_key()
    from reel_af.app import article_to_reel

    run_id = uuid.uuid4().hex[:8]
    out_path = out_dir or (Path.cwd() / "output" / f"article-{run_id}")
    out_path.mkdir(parents=True, exist_ok=True)

    console.rule(f"[bold]article_to_reel — {run_id}")
    console.print(f"  url: [cyan]{url}[/cyan]")
    console.print(f"  out: [dim]{out_path}[/dim]\n")

    result = asyncio.run(article_to_reel(url=url, out_dir=str(out_path)))
    if "error" in result:
        console.print(f"[red]error:[/red] {result['error']}")
        sys.exit(1)
    _summarize(result, run_id, out_path)


@app.command("topic")
def topic(
    topic: Annotated[str, typer.Argument(help="The topic phrase.")],
    out_dir: Annotated[
        Optional[Path],
        typer.Option("--out", help="Output directory.", show_default=False),
    ] = None,
) -> None:
    """Turn a topic into a vertical viral reel (multi-reasoner cascade)."""
    _require_key()
    from reel_af.app import topic_to_reel

    run_id = uuid.uuid4().hex[:8]
    out_path = out_dir or (Path.cwd() / "output" / f"topic-{run_id}")
    out_path.mkdir(parents=True, exist_ok=True)

    console.rule(f"[bold]topic_to_reel — {run_id}")
    console.print(f"  topic: [cyan]{topic}[/cyan]")
    console.print(f"  out:   [dim]{out_path}[/dim]\n")

    result = asyncio.run(topic_to_reel(topic=topic, out_dir=str(out_path)))
    if "error" in result:
        console.print(f"[red]error:[/red] {result['error']}")
        sys.exit(1)
    _summarize(result, run_id, out_path)


@app.command("serve")
def serve() -> None:
    """Run the AgentField node so the reasoners register with the control plane."""
    _require_key()
    from reel_af.app import main as _main
    _main()


def main() -> None:
    app()


if __name__ == "__main__":
    main()
