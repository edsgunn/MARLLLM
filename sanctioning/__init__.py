"""Sanctioning Measurement Framework (Rungs 0-1).

A validated, provider-agnostic pipeline for measuring *sanctioning* —
approving/disapproving reactions between agents — in forum-style multi-agent
training transcripts. See README.md in this package for the full design.

The load-bearing component is the per-post sanctioning-event extractor
(`judge.py`); everything in `rung0.py` / `rung1.py` is deterministic
aggregation on top of the persisted event records. Detection (the LLM judge)
is kept strictly separate from interpretation (the aggregation code).
"""

SCHEMA_VERSION = "1"
PROMPT_VERSION = "sanction_judge_v1"
