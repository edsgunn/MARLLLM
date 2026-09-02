# Sequence packing — submitted twice, never trained

Two runs, `Qwen3-4B` at 8 and 16 agents, testing the sequence-packing path from
Task 3 of `RLvr_rollout_optimisation_audit.md`: concatenate one agent's
trajectories into a single `(1, T_total)` row with per-token `seq_ids` and
`position_ids` resetting at each boundary, and let the attention implementation
build a block-diagonal causal mask instead of a dense 4D one.

## Status: NO DATA

Neither run wrote a `metrics.jsonl`. Both logs stop after the iteration-0
checkpoint traces. The earlier attempts, in
[`../cultural_emergence_qwen3_packed_attic/`](../cultural_emergence_qwen3_packed_attic/),
failed the same way.

**Sequence packing has never produced a training iteration in this project.**
Four submissions, zero data.

The configs record two constraints that are worth keeping in view when this is
next attempted, because either could be the cause:

- `seq_chunk_size` must be **removed** — packing and chunked loss are mutually
  exclusive in this release, since chunked+packed needs `cu_seqlens`-aware
  per-chunk masks.
- `max_episode_tokens` was cut 24576 → 12288 so per-agent packed totals do not
  blow activation memory without chunking.

Removing chunked loss is exactly the change that OOMed `social_terminal/run03`,
so the first hypothesis to test is memory.

## Never delete a run

These stay.
