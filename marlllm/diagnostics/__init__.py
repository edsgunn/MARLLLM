"""
Post-training diagnostics for the distillation-vs-grounding question.

Each module in this package exposes a `run(config, output_dir)` function and
writes structured outputs (JSON / Parquet) to `output_dir`. They are designed
to be invoked independently or as a suite via `scripts/run_diagnostics.py`.

Modules
-------
mimicry  — n-gram overlap, embedding similarity, BLEU/chrF, move-distribution.
ood      — task-outcome metrics on held-out scenarios and adversarial partners.
probing  — linear probes for environment state vs partner identity.
hull     — fraction of utterances outside the strong-partner convex hull.
"""
from marlllm.diagnostics import mimicry, ood, probing, hull
from marlllm.diagnostics.eval_rollout import collect_eval_rollouts

__all__ = ["mimicry", "ood", "probing", "hull", "collect_eval_rollouts"]
