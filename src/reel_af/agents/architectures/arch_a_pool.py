"""Architecture A — Pool + Critic + Refine (baseline).

The current production pipeline. 4 parallel drafts with forced (direction,
hook_trick) variety → critic scores on 5 weighted dims → refine if below
threshold. Wrapped here in the uniform ArchOutput shape for comparison.
"""

from __future__ import annotations

import time
from typing import Any

from reel_af.agents.architectures import ArchOutput
from reel_af.agents.distiller import ArticleSummary
from reel_af.agents.script_loop import run_script_loop


async def run(app: Any, summary: ArticleSummary) -> ArchOutput:
    t0 = time.time()
    loop = await run_script_loop(app, summary)
    trace = [
        f"pool size: {loop.pool_size}",
        f"refinement passes: {loop.refine_passes}",
    ]
    # Surface ranking from the pool history.
    for d, card, comp in loop.pool_history:
        trace.append(
            f"  {comp:.1f}  {d.direction:18s} / {d.hook_trick:20s} "
            f"(hook={card.hook_strength.score} faith={card.faithfulness.score} "
            f"tropes={card.trope_execution.score} flow={card.spoken_flow.score} "
            f"close={card.close_landing.score})"
        )
    return ArchOutput(
        arch_id="A",
        arch_name="Pool + Critic + Refine",
        bet="Quantity of forced-distinct candidates + structured critic finds the best.",
        draft=loop.draft,
        self_score=loop.composite,
        wall_time_s=time.time() - t0,
        trace=trace,
    )
