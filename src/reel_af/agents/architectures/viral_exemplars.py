"""Curated exemplar reel scripts for few-shot imitation.

These are short-form scripts in the style of documented high-performing
creators (Hormozi value-bombs, Ali Abdaal explainer Shorts, Hank Green
science explainers, MrBeast hook engineering, Justin Welsh contrarian
takes). Each is annotated with the hook/retention/close trick it executes
so the model can imitate the STRUCTURE, not the surface words.

A few are direct paraphrases of patterns the creators have explicitly
taught (Hormozi's "$100M Offers" 1-min hooks, MrBeast's hook teardowns).
Others are synthesized in the same idiom. The point isn't provenance — it's
giving the model a small set of TRUE viral structures to ground on.

Tagged so we can pick exemplars matching the target article's direction.
"""

from __future__ import annotations

from dataclasses import dataclass

from reel_af.agents.reel_composer import (
    CloseTrickId,
    Direction,
    HookTrickId,
    RetentionTrickId,
)


@dataclass
class Exemplar:
    id: str
    direction: Direction
    hook_trick: HookTrickId
    retention_trick: RetentionTrickId
    close_trick: CloseTrickId
    style_note: str        # one-line vibe descriptor for matching
    script: str            # the actual exemplar script


EXEMPLARS: list[Exemplar] = [
    # ───────────────────────────────────────────────────────────────
    Exemplar(
        id="hormozi_pricing",
        direction="counterintuitive",
        hook_trick="contradiction",
        retention_trick="open_loop",
        close_trick="callback_punch",
        style_note="Confident, blunt, business-coach voice. Short punchy sentences.",
        script=(
            "Stop discounting your product. Wrong move every time. The second you cut "
            "price, you teach the buyer your old price was a lie — and they'll never "
            "pay full again. There's a fix coming, but first: discount once, lose "
            "trust forever. The fix? Add value. Don't cut price. Cutting price kills "
            "everything."
        ),
    ),
    # ───────────────────────────────────────────────────────────────
    Exemplar(
        id="abdaal_study",
        direction="tutorial",
        hook_trick="number_shock",
        retention_trick="promise_payoff",
        close_trick="save_bait",
        style_note="Friendly tutor voice. Numbered steps. Concrete examples.",
        script=(
            "Three reasons your studying isn't sticking. One: you re-read instead of "
            "recall — passive doesn't work. Two: you study in long blocks — your brain "
            "needs spacing. Three: you skip the test — testing yourself IS the learning. "
            "Save this. Next exam, run these three, watch what happens."
        ),
    ),
    # ───────────────────────────────────────────────────────────────
    Exemplar(
        id="science_discovery",
        direction="discovery",
        hook_trick="number_shock",
        retention_trick="concrete_specifics",
        close_trick="cliffhanger",
        style_note="Hank Green-style wonder voice. Numbers, names, fast pacing.",
        script=(
            "Twenty-five years. That's how long it took to figure out why octopi can "
            "taste with their skin. Each sucker has receptors that read chemicals "
            "directly — they don't need to see prey, they LICK the rock and know what "
            "was there an hour ago. Scientists call it touch-taste. What else are "
            "they tasting that we can't?"
        ),
    ),
    # ───────────────────────────────────────────────────────────────
    Exemplar(
        id="welsh_contrarian",
        direction="counterintuitive",
        hook_trick="direct_threat",
        retention_trick="stakes_ladder",
        close_trick="loop_closure",
        style_note="Justin Welsh-style direct address. Real consequence, no metaphors.",
        script=(
            "If you're working 60 hours a week, you're falling behind. Not because of "
            "effort — because effort without leverage compounds nothing. The people "
            "passing you spent year one building distribution. Year two building "
            "systems. By year three, they work 20 hours and out-earn you. Stop working. "
            "Start leveraging."
        ),
    ),
    # ───────────────────────────────────────────────────────────────
    Exemplar(
        id="tech_explainer",
        direction="explainer",
        hook_trick="pattern_interrupt",
        retention_trick="rehook_mid",
        close_trick="callback_punch",
        style_note="Tech-curious voice. Surprising mental model. Clear payoff.",
        script=(
            "Your phone's GPS doesn't know where you are. Satellites guess — and you "
            "do the math. Each satellite sends the exact time. Your phone compares "
            "delays from four of them. The answer is where the four delays only make "
            "sense if you're standing right HERE. But here's the kicker: a microsecond "
            "off and you're 300 meters off. Your GPS is doing relativity math. Every "
            "second."
        ),
    ),
    # ───────────────────────────────────────────────────────────────
    Exemplar(
        id="inspiration_story",
        direction="inspiration",
        hook_trick="name_drop",
        retention_trick="concrete_specifics",
        close_trick="callback_punch",
        style_note="Documentary voice. Specific person, specific moment, transferable lesson.",
        script=(
            "Stripe was rejected by every VC for six months. Patrick Collison was 19. "
            "He kept sending the deck — same deck — to anyone who'd take a call. By "
            "month seven, Sequoia funded them at a $20 million valuation. Stripe is "
            "worth $90 billion now. The trick wasn't the deck. The trick was month seven."
        ),
    ),
    # ───────────────────────────────────────────────────────────────
    Exemplar(
        id="breakdown_news",
        direction="breakdown",
        hook_trick="mid_thought",
        retention_trick="concrete_specifics",
        close_trick="comment_bait",
        style_note="Newsy explainer voice. Drops you mid-story, walks back to context.",
        script=(
            "…and that's when OpenAI quietly removed the safety clause. The change "
            "went live Thursday — one line deleted from the model usage policy. The "
            "old line said no military applications. The new policy says they'll work "
            "with the DoD on cybersecurity. Three years ago they called that crossing "
            "a red line. Are they wrong now, or were they wrong then?"
        ),
    ),
    # ───────────────────────────────────────────────────────────────
    Exemplar(
        id="psych_curiosity",
        direction="discovery",
        hook_trick="question_curiosity",
        retention_trick="open_loop",
        close_trick="save_bait",
        style_note="Curious science voice. Question hook, methodical reveal.",
        script=(
            "Why do you remember exactly where you were on 9/11 but not last "
            "Tuesday? It's not what you think. The brain doesn't store more memory "
            "during emotional events — it stores them DIFFERENTLY. Stress hormones "
            "tag them as 'urgent.' Decades later they replay in HD while last week "
            "blurs. Save this next time you can't remember a meeting."
        ),
    ),
]


def select_exemplars(direction: Direction, n: int = 3) -> list[Exemplar]:
    """Pick N exemplars that best fit a target direction.

    Always include 1 direct match if available; fill the rest with diverse
    techniques so the model sees structural variety, not just one template.
    """
    direct = [e for e in EXEMPLARS if e.direction == direction]
    others = [e for e in EXEMPLARS if e.direction != direction]
    picks: list[Exemplar] = []
    if direct:
        picks.append(direct[0])
    # Fill with non-duplicates, preferring varied hook tricks.
    seen_hooks = {p.hook_trick for p in picks}
    for e in others:
        if e.hook_trick in seen_hooks:
            continue
        picks.append(e)
        seen_hooks.add(e.hook_trick)
        if len(picks) >= n:
            break
    return picks[:n]


def format_for_prompt(exemplars: list[Exemplar]) -> str:
    """Render the exemplars as a structured prompt block."""
    lines = ["=== VIRAL REEL EXEMPLARS (study the STRUCTURE, not the words) ==="]
    for e in exemplars:
        lines.append("")
        lines.append(f"--- exemplar: {e.id} ---")
        lines.append(
            f"direction={e.direction} · hook_trick={e.hook_trick} · "
            f"retention_trick={e.retention_trick} · close_trick={e.close_trick}"
        )
        lines.append(f"vibe: {e.style_note}")
        lines.append("script:")
        lines.append(f"  {e.script}")
    return "\n".join(lines)
