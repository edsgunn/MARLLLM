# Attic: Qwen3.5 (2026-05-14) — twelve attempts, no training

Twelve directories: `Qwen3.5-4B` and `Qwen3.5-9B`, 8 and 16 agents, each
resubmitted three times on 2026-05-14 (`_111249`, `_123841`, `_132433` — the
timestamp suffix is the previous attempt rotated aside on resubmission).

## Why they cannot be cited

**None of them trained.** The deepest reached iteration 19; most wrote one or
five metric records or none at all, and every log ends the same way — the
population starts, the iteration-0 checkpoint traces are written, and then
nothing. The failure is in the first training step and it is reproducible: it
happened three times per configuration.

KL from base is ≤ 0.001 in every run that logged anything, which confirms no
optimisation step took effect.

## What was learned

That the same-day Qwen3.5 attempt cost twelve submissions and produced no data,
and that resubmitting an identically-configured job three times is not a
diagnosis. The campaign moved to Qwen3 (`../cultural_emergence_qwen3/`), which
runs.

**What was not learned is why**, and the failure was never traced. If Qwen3.5 is
wanted, that trace is the first job, not another submission.

## Never delete a run

These stay — twelve silent directories are exactly the gap between job ids that
`PROCESS.md` exists to prevent.
