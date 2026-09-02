# Attic: async rollout, first attempts (2026-05-15)

Two directories, the rotated-aside first attempts at the asynchronous-rollout
variant. Their successors
([`../cultural_emergence_qwen3_async/`](../cultural_emergence_qwen3_async/)) ran
and produced the equivalence result.

## Why they cannot be cited

Neither trained. The logs end on repeated
`async rollout worker started (buffer_size=2, max_staleness=1)` lines — one per
rank — with no iteration ever completing. The workers start and the trainer
never consumes.

Unlike the Qwen3.5 and packing failures, this one *was* fixed: the resubmission
under the same configuration ran to iterations 112 and 123 and matched its
synchronous siblings on every metric.

## Never delete a run

These stay.
