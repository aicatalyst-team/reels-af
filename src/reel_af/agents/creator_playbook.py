"""Creator Playbook — the named bag of tricks every viral short-form video uses.

The downstream agents (composer, scene breaker, shot director, captioner)
pick techniques from this menu and DECLARE which they're using. That way:
  • Every reel's output is auditable — you can see which trick produced
    which moment.
  • A/B testing is structured — swap a hook trick, regenerate, compare.
  • The model isn't told "be viral" (vague) but "pick a hook from this menu
    and use it" (specific, copy-from-the-best).

Sources for these tricks: dissection of high-performing reels from creators
like MrBeast (hook engineering), Ali Abdaal (educational reels), Hamish
Hodder (open loops), the_dnt (caption pacing). Generalized into
content-agnostic moves any reel can apply.

Each entry is a string the agent can paste into its own reasoning. Don't
remove items lightly — even rarely-used tricks are picked by the agents
when the article calls for them.
"""

from __future__ import annotations

# ────────────────────────────────────────────────────────────────────────
# HOOK TRICKS — the first 0.5-1.5s. Pick exactly ONE per reel.
# ────────────────────────────────────────────────────────────────────────

HOOK_TRICKS: dict[str, str] = {
    "contradiction": (
        "Open with a sentence that contradicts a common belief.\n"
        "  Template: \"[Common belief]. Wrong.\"  or  \"Everyone tells you X. They're lying.\"\n"
        "  Why it works: the brain must resolve the contradiction → keeps watching."
    ),
    "number_shock": (
        "Lead with a specific, surprising number from the source.\n"
        "  Template: \"[Specific number] [unit] [unexpected verb].\"\n"
        "  Why it works: concrete numbers are unskippable; vague claims aren't."
    ),
    "mid_thought": (
        "Drop the viewer into the MIDDLE of a thought, no setup.\n"
        "  Template: \"…and that's when I realized [X].\"\n"
        "  Why it works: the brain back-fills context → can't swipe away."
    ),
    "direct_threat": (
        "Address the viewer directly with a stakes-shaped warning.\n"
        "  Template: \"If you [common behavior], you're losing [thing].\"\n"
        "  Why it works: personal threat overrides general curiosity."
    ),
    "loop_bait": (
        "Open with a visual or phrase the close will explicitly call back to.\n"
        "  Template: a striking opener line + planned callback at the end.\n"
        "  Why it works: viewers feel closure → triggers rewatch / save."
    ),
    "name_drop": (
        "Lead with a recognizable name doing something unexpected.\n"
        "  Template: \"[Known person] just [unexpected verb].\"\n"
        "  Why it works: recognition is instant comprehension; unexpected verb hooks."
    ),
    "question_curiosity": (
        "Open with a question that PROMISES the answer is in the reel.\n"
        "  Template: \"Why does [common thing] [unexpected pattern]?\"\n"
        "  Why it works: open question = open loop. Brain must close it."
    ),
    "pattern_interrupt": (
        "Open with a sentence that breaks the viewer's expectation of what a "
        "video on this topic would say.\n"
        "  Template: an absurd-feeling but defensible opener.\n"
        "  Why it works: record-scratch effect; the brain pauses to verify."
    ),
}

# ────────────────────────────────────────────────────────────────────────
# NAME-THEN-HOOK OPENER TEMPLATES — for OBSCURE topics only.
# Pay-the-clarity-tax-upfront patterns documented across retention research.
# When `topic_familiarity == "obscure"`, the writer MUST use one of these
# instead of the cold-open HOOK_TRICKS above.
# ────────────────────────────────────────────────────────────────────────

VISUAL_ACCESSIBILITY_GUIDE = """\
═══════════════════════════════════════════════════════════════════════════
VISUAL GUIDE FOR THE TIKTOK SCROLLER — PRINCIPLES, NOT TEMPLATES
═══════════════════════════════════════════════════════════════════════════

You're designing visuals that play under a 25-second vertical-video
narration. The viewer is on a phone, glancing while doing something else.
They read the frame in 200 milliseconds. If the visual doesn't COMMUNICATE
what's being talked about in that 200ms, they swipe.

These are principles to think with, not a menu to pick from.

──────────────────────────────────────────────────────────────────────────
PRINCIPLE 1 — VISUALS MUST MAKE THE TOPIC VISIBLE
──────────────────────────────────────────────────────────────────────────
For HOT topics (audience already knows): cinematic mood / atmosphere
shots are fine. The frame can be evocative because the viewer already
has the referent.

For OBSCURE topics (audience needs introduction): visuals must HELP
DEFINE the subject. If the article is about a piece of software, show
the software's interface — not "a person at a laptop". If it's about a
research finding, show the actual phenomenon — not "a scientist with a
clipboard". Visuals that explain reduce the audience's cognitive load
and reinforce what the narration says.

──────────────────────────────────────────────────────────────────────────
PRINCIPLE 2 — RECOGNISABLE BEATS SURREAL
──────────────────────────────────────────────────────────────────────────
A close-up of a real keyboard with a key being pressed beats "abstract
keyboard floating in a digital void". A specific brand-recognisable
object (an iPhone, a Tesla, an actual cargo plane) beats a generic
stand-in. People react emotionally to things they recognise; they swipe
past things they need to decode.

NEVER generate: glowing brains, neural network nodes, abstract data
streams, generic businesspeople shaking hands, "AI" rendered as
glowing geometry, generic server rooms with blue lights.

──────────────────────────────────────────────────────────────────────────
PRINCIPLE 3 — FACES, HANDS, OR REAL OBJECTS — PICK ONE PER SHOT
──────────────────────────────────────────────────────────────────────────
The eye locks onto faces, hands doing things, and specific real-world
objects in that order. Every shot should anchor on at least one of these,
NOT on architectural / atmospheric / abstract elements.

  Faces      — emotion you can read in 0.5s.
  Hands      — POV; viewer's mirror neurons fire.
  Real objs  — specific, recognisable, named.

──────────────────────────────────────────────────────────────────────────
PRINCIPLE 4 — MOTION DOES NARRATIVE WORK
──────────────────────────────────────────────────────────────────────────
Motion isn't just "make the still feel less static". Motion should
TRANSFORM or REVEAL something the still didn't show. A hand opening to
reveal what was inside. A door swinging shut on the close. A keyboard
being pressed as the line lands on the key. Bad: "slow camera push on
generic background." Good: "the keystroke completes as the voice says
'enter'."

──────────────────────────────────────────────────────────────────────────
PRINCIPLE 5 — VARY BUT MAINTAIN VISUAL COHERENCE
──────────────────────────────────────────────────────────────────────────
Across the reel, vary scale (close-up → mid → wide), subject (object →
face → environment), and treatment (literal → analogous → contrasting).
But maintain ONE visual style across the whole reel (lighting, grading,
era of imagery). The reel should feel like one continuous piece, not a
slideshow of stock footage.

──────────────────────────────────────────────────────────────────────────
CALIBRATION — quick read
──────────────────────────────────────────────────────────────────────────

Script line: "Omarchy is Arch Linux with one developer's keyboard shortcuts."

✗ BAD visuals (assumes context, abstract):
  - "Glowing terminal in a dark room, code flowing like rain."
  - "A penguin (Linux mascot) standing in a void."

✓ GOOD visuals (defines the subject):
  - "Real laptop with a clearly-visible terminal window, fingers
    typing 'SUPER+SHIFT+C' as the keys press."
  - "Hand holding a USB drive labelled 'Omarchy' next to a
    second drive labelled 'Arch Linux' — the second is much bigger."

Script line: "Quantum cryptography promises perfectly secure messages."

✗ BAD: "Glowing entangled particles on a starfield."
✓ GOOD: "A locked envelope on a desk; a hand reaching for it — the
   lock dissolves as the camera pushes in."
   (The envelope IS the encrypted message; the dissolving lock is the
    threat. A viewer who knows nothing about quantum mechanics still
    gets it.)
══════════════════════════════════════════════════════════════════════════
"""


OBSCURE_WRITING_GUIDE = """\
═══════════════════════════════════════════════════════════════════════════
WRITING GUIDE FOR OBSCURE TOPICS — PRINCIPLES, NOT TEMPLATES
═══════════════════════════════════════════════════════════════════════════

You are writing for a TikTok / Reels scroller. Picture: someone on a lunch
break, scrolling fast, no specialty knowledge of your topic. Maybe a high
school education, maybe a college one — but NOT in your field. They have
literally zero context about what your article is about. If they can't
follow the first 8 seconds in plain English, they swipe.

These are principles to think with, not slots to fill. Pick the SHAPE
that fits THIS specific article — the right shape almost never matches a
generic template.

──────────────────────────────────────────────────────────────────────────
PRINCIPLE 1 — DEFINE BEFORE YOU JUDGE
──────────────────────────────────────────────────────────────────────────
The viewer cannot react to a take about X if they don't know what X is.
For an obscure topic, the FIRST job is to establish the subject in a
sentence so plain that a stranger overhearing the audio could follow it.
THEN deliver the take.

The contrarian payoff comes second, not first. Resist the urge to lead
with the punch — it lands as confusion, not surprise, when the viewer
has no referent.

──────────────────────────────────────────────────────────────────────────
PRINCIPLE 2 — TALK LIKE A FRIEND, NOT LIKE A PITCH
──────────────────────────────────────────────────────────────────────────
For obscure topics, the register is CONVERSATIONAL EXPLAINER — like
telling a friend about something interesting you read — not the punchy
contrarian creator voice that works for hot topics.

Cues: rhetorical questions, "you know how…", "have you heard of…",
"turns out…", "here's the weird part…", "I'll explain". Address the
viewer as a peer who's curious but uninformed, not someone you're trying
to dunk on.

──────────────────────────────────────────────────────────────────────────
PRINCIPLE 3 — PLAIN LANGUAGE OR INLINE DEFINITION
──────────────────────────────────────────────────────────────────────────
Every specialist term must either (a) be REPLACED with the plain-English
equivalent from the jargon glossary you'll receive, or (b) defined
inline in 4 words MAX, e.g. "entanglement — two particles linked as one".

Test: read the script to your aunt. If she'd ask "what does X mean?",
rewrite or define.

NEVER use a specialist term without resolution. NEVER assume the viewer
knows what a "dotfile", "no-signaling", or "ricing" is.

──────────────────────────────────────────────────────────────────────────
PRINCIPLE 4 — ANCHOR EVERY ABSTRACT IDEA TO SOMETHING CONCRETE
──────────────────────────────────────────────────────────────────────────
If a concept needs explaining, anchor it to something the viewer
already knows from daily life: phones, food, money, their own body,
their commute, a tool they've used. Concrete anchors carry abstract
ideas. "Encryption" alone is abstract. "The padlock icon in your
browser" is concrete.

──────────────────────────────────────────────────────────────────────────
PRINCIPLE 5 — STILL HOOK, JUST ACCESSIBLY
──────────────────────────────────────────────────────────────────────────
"Accessible" does not mean "boring". Use rhetorical questions, open
loops, surprise reveals — all of the creator-toolkit moves — but route
them through plain language and a friendly register. The viral
machinery still applies; only the vocabulary changes.

──────────────────────────────────────────────────────────────────────────
CALIBRATION — read these aloud
──────────────────────────────────────────────────────────────────────────

Topic: "Omarchy is a Linux distribution that's just one developer's dotfiles"

✗ BAD (assumes context, jargon, punchy creator-style):
  "Omarchy isn't a real distro. Wrong label. It's Arch Linux with DHH's
   personal dotfiles, monetised. Ships zero packages. Install Debian."
  (Why bad: "distro", "Arch Linux", "DHH", "dotfiles", "packages",
   "Debian" — six specialist terms in the first 20 seconds. Viewer
   swipes at second 2.)

✓ GOOD (defines first, then take, plain English):
  "Have you heard of Omarchy? It's been called a new Linux operating
   system — Linux being the free alternative to Windows. Except it
   isn't a new operating system at all. It's basically one developer's
   personal keyboard shortcuts, repackaged and sold. Imagine if your
   buddy renamed their custom Windows setup and called it 'BillOS'.
   That's Omarchy. Funny part — people are paying for it."
  (Why good: names + defines Omarchy in the first 14 words. Anchors
   Linux to Windows. Anchors the whole thing to a friend-renaming-
   Windows analogy. Take comes in sentence 4. Close is a quotable.)

Topic: "Quantum jamming may break quantum cryptography"

✗ BAD: "Quantum crypto is unbreakable. Jamming disagrees. Jim the Jammer
   shifts correlations from opposite to same — and you'd never know."
  (Why bad: "quantum crypto", "no-signaling", "correlations" — viewer
   has no idea what's being claimed. Sounds technical, says nothing.)

✓ GOOD:
  "Quantum cryptography is the holy grail of secret messages — codes
   even a supercomputer can't crack. Or so we thought. Physicists just
   showed a way to fake the signal so cleanly that the receiver would
   never know it was hacked. They call it jamming. It doesn't break
   any laws of physics — it breaks our intuition about what 'secure'
   even means. Which raises a question: is anything truly unhackable?"
  (Why good: defines quantum crypto in plain words first. The take
   comes in sentence 2. No jargon survives the opener.)

──────────────────────────────────────────────────────────────────────────
END OF GUIDE — apply with judgment, not as a template
══════════════════════════════════════════════════════════════════════════
"""


# ────────────────────────────────────────────────────────────────────────
# RETENTION TRICKS — used during the body (3-15s). Pick ONE main + may layer.
# ────────────────────────────────────────────────────────────────────────

RETENTION_TRICKS: dict[str, str] = {
    "open_loop": (
        "Promise a payoff early, deliver it late.\n"
        "  How: in scene 1-2, say \"…and the kicker is X — but first…\". Don't\n"
        "  reveal X until scene 4-5. Viewer waits.\n"
        "  Why it works: open loops are the strongest known retention device."
    ),
    "stakes_ladder": (
        "Each scene escalates the stakes of the previous one.\n"
        "  How: small consequence → bigger consequence → biggest.\n"
        "  Why it works: monotone stakes = boring; escalation = momentum."
    ),
    "rehook_mid": (
        "Re-hook around scene 3-4 with a 'but here's the crazy part' line.\n"
        "  How: a deliberate second hook AFTER the initial drop-in.\n"
        "  Why it works: catches viewers who were about to swipe at 6-8s."
    ),
    "promise_payoff": (
        "Tell viewer exactly what's coming, then deliver it.\n"
        "  How: \"I'll show you 3 things that…\" → walk through them.\n"
        "  Why it works: makes viewer feel in control → less swipe instinct."
    ),
    "concrete_specifics": (
        "Layer in specific names/numbers/times in EVERY sentence.\n"
        "  How: replace 'a few weeks' with '17 weeks', 'researchers' with\n"
        "  'a Stanford team led by [name]'.\n"
        "  Why it works: specifics signal expertise → viewer commits more time."
    ),
}

# ────────────────────────────────────────────────────────────────────────
# CLOSE TRICKS — the last 2-3s. Pick ONE per reel.
# ────────────────────────────────────────────────────────────────────────

CLOSE_TRICKS: dict[str, str] = {
    "loop_closure": (
        "Explicitly reference the opening image or phrase.\n"
        "  How: a closing line that echoes the hook's noun or verb.\n"
        "  Why it works: viewers replay the opening to verify → rewatch."
    ),
    "cliffhanger": (
        "End mid-thought, on a question, or with the implication unspoken.\n"
        "  How: stop the script one sentence early, on a hook.\n"
        "  Why it works: incompleteness triggers comments, debate, shares."
    ),
    "save_bait": (
        "End with a line that frames the reel as REFERENCE material.\n"
        "  How: \"…remember this next time you [common situation].\"\n"
        "  Why it works: prompts the save → save = algorithmic ranking."
    ),
    "comment_bait": (
        "End with a question that splits the audience into camps.\n"
        "  How: \"…would you do it? Tell me below.\"\n"
        "  Why it works: comments = engagement = distribution."
    ),
    "callback_punch": (
        "End by punching the hook's word with new meaning.\n"
        "  How: same word, completely different context after the body.\n"
        "  Why it works: linguistic loop closure → satisfaction → rewatch."
    ),
}

# ────────────────────────────────────────────────────────────────────────
# VISUAL TRICKS — per-scene; the shot director picks one per beat.
# ────────────────────────────────────────────────────────────────────────

VISUAL_TRICKS: dict[str, str] = {
    "face_fill": (
        "Tight face-fills-the-frame close-up.\n"
        "  Why: faces are unskippable; eye contact + emotion = stop."
    ),
    "pov_hands": (
        "First-person POV — viewer sees their own hands doing the thing.\n"
        "  Why: feels personal; viewer's mirror neurons fire."
    ),
    "transformation": (
        "Mid-action transformation: before → after, build → reveal.\n"
        "  Why: brain locks until transformation completes."
    ),
    "scale_contrast": (
        "Dramatic scale comparison in one frame.\n"
        "  Why: visual size shock = instant emotional response."
    ),
    "movement_into_frame": (
        "Subject enters or exits the frame; not static.\n"
        "  Why: motion catches peripheral vision = no swipe."
    ),
    "isolated_object": (
        "ONE object, centered, against neutral background, dramatic light.\n"
        "  Why: cinematic framing = premium = trust → keeps watching."
    ),
    "human_scale_detail": (
        "Show the thing being discussed at a real-life human scale next to a\n"
        "  recognizable object (hand, phone, coin).\n"
        "  Why: makes abstract concrete = comprehension = retention."
    ),
}

# ────────────────────────────────────────────────────────────────────────
# CAPTION TRICKS — assembly-time text treatment.
# ────────────────────────────────────────────────────────────────────────

CAPTION_TRICKS: dict[str, str] = {
    "scale_pop": "Caption scales from 80% to 100% in 0.2s on entry (bounce).",
    "color_flash": "Caption white normally; key word flashes yellow for 0.15s.",
    "single_word_reveal": "One word at a time, synced to audio emphasis.",
    "soft_fade": "Caption fades in over 0.25s, no animation. (Default; least loud.)",
}


# ────────────────────────────────────────────────────────────────────────
# Helpers to inject into prompts.
# ────────────────────────────────────────────────────────────────────────

def format_menu(menu: dict[str, str], header: str) -> str:
    """Render a dict[id, description] as a numbered menu for prompts."""
    lines = [f"=== {header} ==="]
    for tid, desc in menu.items():
        lines.append(f"\n[{tid}]")
        lines.append(desc)
    return "\n".join(lines)
