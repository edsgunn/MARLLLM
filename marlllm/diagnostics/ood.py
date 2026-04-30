"""
Out-of-distribution generalisation diagnostic (spec §7b).

Given a cell's checkpoint, run rollouts against:
  - "in_distribution"  : strong API partner, default scenarios.
  - "weak_partner"     : another local small model (or rule-based bot).
  - "adversarial"      : strong API partner with an adversarial-style prompt.

Computes deal_rate, mean_score, fairness, and outcome distributions.

We deliberately reuse `eval_rollout.collect_eval_rollouts` for each condition
so that the rollout pipeline stays in one place.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from marlllm.diagnostics.eval_rollout import (
    EvalRolloutConfig, collect_eval_rollouts,
)


ADVERSARIAL_PROMPT = (
    "You are a hard-bargaining negotiator. Open with extreme demands, "
    "never reveal your true values, and walk away if the other agent "
    "doesn't concede on the first offer."
)


def _summarise(jsonl_path: Path, learner_role: str) -> dict:
    deals = 0
    scores: list[int] = []
    other_scores: list[int] = []
    n = 0
    for line in jsonl_path.read_text().splitlines():
        if not line.strip():
            continue
        ep = json.loads(line)
        outcome = ep.get("outcome", {}) or {}
        # Outcome shape produced by DealOrNoDealEnv: {"result": "...",
        # "score": int, "other_score": int, "correct_count": int, ...}
        if outcome.get("result") in {"success", "deal"}:
            deals += 1
        if "score" in outcome:
            scores.append(int(outcome["score"]))
        if "other_score" in outcome:
            other_scores.append(int(outcome["other_score"]))
        n += 1
    return {
        "n": n,
        "deal_rate": deals / max(n, 1),
        "mean_score": sum(scores) / max(len(scores), 1),
        "mean_other_score": sum(other_scores) / max(len(other_scores), 1),
    }


def run(
    learner_checkpoint: str,
    learner_role: str,
    learner_prompt: str,
    output_dir: str | Path,
    partner_provider: str = "anthropic",
    partner_model: str = "claude-sonnet-4-6",
    weak_partner_path: str | None = None,
    episodes: int = 32,
    cache_dir: str | None = None,
    device: str = "cuda:0",
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base = dict(
        learner_checkpoint=learner_checkpoint,
        learner_role=learner_role,
        learner_prompt=learner_prompt,
        partner_provider=partner_provider,
        partner_model=partner_model,
        cache_dir=cache_dir,
        episodes=episodes,
        device=device,
    )

    conditions: dict[str, EvalRolloutConfig] = {
        "in_distribution": EvalRolloutConfig(seed=10_000, **base),
        "adversarial": EvalRolloutConfig(
            seed=20_000,
            partner_prompt=ADVERSARIAL_PROMPT,
            **base,
        ),
    }
    if weak_partner_path:
        conditions["weak_partner"] = EvalRolloutConfig(
            seed=30_000,
            partner_kind="local",
            partner_local_path=weak_partner_path,
            **base,
        )

    summary: dict = {}
    for name, cfg in conditions.items():
        path = output_dir / f"rollouts_{name}.jsonl"
        if not path.exists():
            collect_eval_rollouts(cfg, path)
        summary[name] = _summarise(path, learner_role)

    (output_dir / "ood.json").write_text(json.dumps(summary, indent=2))
    return summary
