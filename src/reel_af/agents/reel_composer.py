"""Reel Composer — given a faithful summary, write the script.

This replaces the old story_writer. Different job:
  - Receives a structured summary (not the article body — that pollution
    was making earlier versions invent off-source metaphors).
  - Judges if it's reel-worthy and picks ONE presentation direction.
  - Writes a 40-55 word script that is a FAITHFUL CONDENSATION of the
    summary in the chosen direction. Engaging through pacing and
    structure, not by inventing new metaphors or examples.

Six supported directions, each with its own micro-structure:
  • explainer       — HOOK → core concept → 2 mechanics → takeaway
  • discovery       — fact HOOK → context → implication → so-what
  • counterintuitive→ common belief → reversal → evidence → flip
  • tutorial        — promise → 3 steps → close
  • inspiration     — situation → action → result → reframe-for-you
  • breakdown       — event → cause → effect → what's next

Context strategy:
  IN  : ArticleSummary (thesis, points, examples, takeaway, domain)
  OUT : { direction, voice_tone, viral_score, script }
  No source body, no claims list — the summary is now the source of truth.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from reel_af.agents.creator_playbook import (
    CLOSE_TRICKS,
    HOOK_TRICKS,
    RETENTION_TRICKS,
    format_menu,
)
from reel_af.agents.distiller import ArticleSummary

Direction = Literal[
    "explainer",
    "discovery",
    "counterintuitive",
    "tutorial",
    "inspiration",
    "breakdown",
]

VoiceTone = Literal["urgent", "wonder", "deadpan", "earnest", "playful"]

# Literals must match playbook keys exactly. Kept in sync by hand because
# Python Literal can't generate from a dict at module load.
HookTrickId = Literal[
    "contradiction", "number_shock", "mid_thought", "direct_threat",
    "loop_bait", "name_drop", "question_curiosity", "pattern_interrupt",
]
RetentionTrickId = Literal[
    "open_loop", "stakes_ladder", "rehook_mid", "promise_payoff",
    "concrete_specifics",
]
CloseTrickId = Literal[
    "loop_closure", "cliffhanger", "save_bait", "comment_bait", "callback_punch",
]

# Word budget tuned for ~19-22s spoken at Kokoro's 2.6 words/sec.
TARGET_WORDS_MIN = 42
TARGET_WORDS_MAX = 58


class ReelDraft(BaseModel):
    """The full presentation plan + final script. One .ai call produces this."""

    model_config = ConfigDict(extra="forbid")

    direction: Direction = Field(
        ...,
        description=(
            "Which presentation shape fits THIS article best. "
            "explainer: teach how something works. "
            "discovery: lead with a fact and unfold it. "
            "counterintuitive: flip a common belief. "
            "tutorial: numbered steps to do something. "
            "inspiration: someone did X — you can too. "
            "breakdown: explain a recent event/situation. "
            "Pick the one that matches what the AUTHOR is doing in the article."
        ),
    )
    voice_tone: VoiceTone = Field(
        ...,
        description=(
            "Emotional register for narration. urgent=alert/warning, "
            "wonder=discovery/science, deadpan=contrarian/irony, "
            "earnest=teaching/advice, playful=humor/quirky."
        ),
    )
    viral_score: int = Field(
        ...,
        ge=1,
        le=10,
        description=(
            "Honest read on how scroll-stop-worthy this material is, 1-10. "
            "Be ruthless. Dense academic content with no concrete examples = 3-4. "
            "Surprising claim with vivid examples = 8-9. Caller uses this to "
            "decide whether to publish."
        ),
    )
    hook_trick: HookTrickId = Field(
        ...,
        description=(
            "Pick ONE hook technique from the menu. The first sentence of the "
            "script MUST execute this trick. Declare it so the caller can audit."
        ),
    )
    retention_trick: RetentionTrickId = Field(
        ...,
        description=(
            "Pick ONE retention technique to thread through the middle of the "
            "script. The body must execute it (open_loop = promise X then delay; "
            "rehook_mid = drop a 'but here's the crazy part' at scene 3-4; etc.)."
        ),
    )
    close_trick: CloseTrickId = Field(
        ...,
        description=(
            "Pick ONE close technique. The last sentence MUST execute it. If "
            "close_trick=loop_closure, the last sentence must echo the hook."
        ),
    )
    script: str = Field(
        ...,
        description=(
            f"ONE continuous spoken paragraph, {TARGET_WORDS_MIN}-{TARGET_WORDS_MAX} "
            f"words. A FAITHFUL CONDENSATION of the article in the chosen direction, "
            f"executing the declared hook/retention/close tricks. Use the article's "
            f"own examples — never invent metaphors not in the source. Spoken English "
            f"(commas for breath, em-dashes for pause, periods for stops). Second "
            f"person where natural."
        ),
    )


def _build_system() -> str:
    hooks_menu = format_menu(HOOK_TRICKS, "HOOK TRICKS (pick ONE — the first sentence executes it)")
    retention_menu = format_menu(RETENTION_TRICKS, "RETENTION TRICKS (pick ONE — the body executes it)")
    close_menu = format_menu(CLOSE_TRICKS, "CLOSE TRICKS (pick ONE — the last sentence executes it)")
    return f"""You're turning a real article's summary into a faithful vertical-reel
narration that uses NAMED VIRAL TECHNIQUES. This is not creative writing — it
is structured compression. Every viral reel you've seen uses 3-4 specific
techniques from a known menu; you pick from that menu explicitly.

NON-NEGOTIABLE RULES:
  1. FAITHFUL TO SOURCE. Use the article's own examples, names, numbers from
     the summary's "concrete examples" list. NEVER invent metaphors that
     aren't in the source (no "elephants in rowboats" unless the article
     mentioned them).
  2. PICK AND DECLARE three named techniques: hook_trick, retention_trick,
     close_trick. The script MUST visibly execute each.
  3. SCORE VIRAL POTENTIAL honestly (1-10). Dense academic content with no
     examples = 3-4. Surprising fact + concrete example = 8-9.

STEP 1 — PICK A DIRECTION (shape that matches what the AUTHOR is doing):

  explainer        — author is teaching how something works
                     structure: HOOK → core idea → 2 mechanics → takeaway
  discovery        — author is reporting a surprising fact or finding
                     structure: fact HOOK → context → implication → so-what
  counterintuitive — author is flipping a common belief
                     structure: common belief → reversal → evidence → flip
  tutorial         — author gives steps / a how-to
                     structure: promise → 3 steps → close
  inspiration      — author shows what someone did, applicable to you
                     structure: situation → action → result → reframe-for-you
  breakdown        — author explains an event or current situation
                     structure: event → cause → effect → what's next

STEP 2 — PICK YOUR THREE TECHNIQUES from these menus:

{hooks_menu}

{retention_menu}

{close_menu}

STEP 3 — WRITE THE SCRIPT executing those techniques.

Style:
  • Spoken English. Read each line aloud in your head.
  • Vary sentence length wildly. Short. Then long-and-flowing.
  • Second person where natural ("you", "your").
  • Active voice. Concrete > abstract.
  • Em-dashes (—) for one dramatic pause. Commas for breath. Period to close.

Forbidden:
  • Metaphors not in the source. Use the article's actual material.
  • Generic AI tells: "Let's talk about", "In this video", "Did you know".
  • Hedging fillers: "actually", "basically", "the fact that".

EXAMPLES (technique-tagged):

article: PG essay — best essays are discoveries made while writing
direction: counterintuitive
hook_trick: contradiction
retention_trick: open_loop
close_trick: callback_punch
script: "You can't write your best essay on purpose. The best ones surprise
         the writer mid-sentence — and there's a trick to triggering that.
         When your draft lands exactly where you planned, you wrote the safe
         one. Throw it out. The good one starts from the thing that startled
         you. On purpose, never."
NOTES: contradiction opens it. open_loop = "there's a trick" promised, then
       delayed. callback_punch = "on purpose" reappears with new meaning.

article: Quanta — octopi taste with their suckers
direction: discovery
hook_trick: number_shock
retention_trick: concrete_specifics
close_trick: save_bait
script: "Two thousand suckers per octopus. Each one is a tongue. They read
         the rock, the prey, the predator's skin chemistry before they see
         it. Scientists call it touch-taste. Save this — your hand is
         missing something that every octopus has."
NOTES: number_shock = "two thousand" up front. concrete_specifics = "rock,
       prey, predator skin chemistry". save_bait = explicit save prompt.

Now: read the summary, pick your techniques, and compose."""


_SYSTEM_TEMPLATE = _build_system()


async def compose_reel(app: Any, summary: ArticleSummary) -> ReelDraft:
    """Pick direction + tone + viral score + script, in one .ai call."""
    user = (
        f"ARTICLE SUMMARY\n"
        f"  domain   : {summary.domain}\n"
        f"  thesis   : {summary.one_line_thesis}\n"
        f"  takeaway : {summary.intended_takeaway}\n\n"
        f"  key points (in source order):\n"
        + "\n".join(f"    {i+1}. {p}" for i, p in enumerate(summary.key_points))
        + "\n\n  concrete examples / names / numbers from the article "
        "(use these — don't invent new ones):\n"
        + (
            "\n".join(f"    - {e}" for e in summary.concrete_examples)
            if summary.concrete_examples
            else "    (none in source — keep the script abstract-but-faithful)"
        )
    )
    return await app.ai(system=_SYSTEM_TEMPLATE, user=user, schema=ReelDraft)
