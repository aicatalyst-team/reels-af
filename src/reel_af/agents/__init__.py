"""Per-stage agents in the reel-af pipeline.

Each module exports one async function that takes typed input and returns
typed output. Stages are composed by the orchestrator in `reel_af.cli`.
"""
