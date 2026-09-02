# Qwen2.5-14B — submitted twice, never trained

Two runs, 8 and 16 agents on `study_group_ashbourne_gc`, the model-scale step up
from the campaign's 7B reference.

## Status: NO DATA

Neither wrote a `metrics.jsonl`. Both logs show the population starting, the
iteration-0 checkpoint traces being written, and nothing after — the 8-agent run
twice on 2026-05-11, thirty-six minutes apart.

## What this cost, and what it means for the scaling question

The campaign has **no model-scaling data above 7B**. `cultural_emergence_32B/`
did not even get as far as a `config.json`. So every claim in this project about
what populations become is a claim about Qwen2.5-7B and, at short horizons,
Qwen3-4B/8B.

The failure signature — dies immediately after the iteration-0 trace dump — is
shared with the Qwen3.5, packed and async-attic attempts, which suggests one
cause rather than four. It was never traced.

## Never delete a run

These stay.
