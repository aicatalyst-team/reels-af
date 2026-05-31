"""LLM-driven agents.

Article-to-reel agents:
  extract  — URL → Essence (one harness call)
  compose  — Essence → ScriptDraft (Hook → Mechanism → Payoff)

Topic-to-reel agents:
  hunters  — topic → 12 EssenceCandidates (4 angles × 3 each)
  critic   — rank candidates, pick top N
  narrator — EssenceCandidate → ConversationalScript (delayed-reveal)
  judge    — pairwise pick of the best narration

Shared downstream:
  visual   — per-beat first-frame image prompt + motion hint
  accent   — per-beat optional editorial overlay
"""
