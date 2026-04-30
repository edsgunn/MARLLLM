"""
Run the full diagnostic harness across cell × seed checkpoints.

Reads a YAML config of the form:

    output_dir: runs/distillation_ablation_v1
    cell_a_transcripts: runs/distillation_ablation_v1/cell_A/transcripts.jsonl
    learner_role: agent_0
    learner_prompt: "You are Agent A, ..."
    partner:
      provider: anthropic
      model: claude-sonnet-4-6
      cache_dir: .cache/anthropic
    eval:
      episodes: 32
      device: cuda:0
    cells: [B, C, D, E, C0]
    seeds: [0, 1, 2]
    diagnostics: [mimicry, ood, probing, hull]

For each (cell, seed) we run an eval rollout against the strong partner
(in-distribution + hidden states), then dispatch the requested diagnostics.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from marlllm.diagnostics import mimicry, ood, probing, hull
from marlllm.diagnostics.eval_rollout import EvalRolloutConfig, collect_eval_rollouts


def _checkpoint_path(root: Path, cell: str, seed: int) -> Path | None:
    """Locate a HF-loadable checkpoint for the given cell/seed."""
    if cell == "A":
        return root / "cell_A" / f"seed_{seed}" / "checkpoint"
    # train_negotiation saves .pt under checkpoints/. We export the final HF
    # weights only when training completes via `model.save_pretrained` —
    # for non-A cells the ablation runner currently saves .pt only. The
    # diagnostic loader expects a HF directory, so cells B-E need an extra
    # step (export from the latest .pt) — see below.
    cell_dir = root / f"cell_{cell}" / f"seed_{seed}"
    hf_dir = cell_dir / "hf_export"
    if hf_dir.exists() and (hf_dir / "config.json").exists():
        return hf_dir
    return None


def _maybe_export_hf(root: Path, cell: str, seed: int, base_model: str) -> Path | None:
    """Convert the trainer's .pt checkpoint to a HF directory.

    The trainer saves agent_0's _backbone state-dict inside payload['agent_states'].
    Here we reload the base model, apply the saved weights, and write a HF dir.
    """
    if cell == "A":
        return _checkpoint_path(root, cell, seed)
    cell_dir = root / f"cell_{cell}" / f"seed_{seed}"
    pt = cell_dir / "checkpoints" / "latest.pt"
    if not pt.exists():
        return None
    hf_dir = cell_dir / "hf_export"
    if (hf_dir / "config.json").exists():
        return hf_dir

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"  Exporting HF checkpoint for cell {cell} seed {seed}")
    payload = torch.load(pt, map_location="cpu")
    states = payload.get("agent_states", {})
    learner_state = states.get("agent_0") or next(iter(states.values()), None)
    if learner_state is None:
        print(f"  ! no agent state in {pt}")
        return None
    model = AutoModelForCausalLM.from_pretrained(base_model)
    try:
        model.load_state_dict(learner_state["backbone"], strict=False)
    except Exception as e:
        print(f"  ! state-dict load failed ({e}); skipping cell {cell} seed {seed}")
        return None
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    hf_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(hf_dir)
    tokenizer.save_pretrained(hf_dir)
    return hf_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--base-model", required=True,
                   help="HF base model used for cells B-E so we can re-load .pt checkpoints.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    root = Path(cfg["output_dir"])
    cell_a = Path(cfg["cell_a_transcripts"])
    learner_role = cfg.get("learner_role", "agent_0")
    learner_prompt = cfg.get(
        "learner_prompt",
        "You are Agent A, negotiating to maximise your score.",
    )
    partner = cfg.get("partner", {}) or {}
    eval_cfg = cfg.get("eval", {}) or {}
    diagnostics_to_run = cfg.get("diagnostics", ["mimicry", "ood", "probing", "hull"])

    summary: dict = {}
    for cell in cfg.get("cells", ["B", "C", "D", "E", "C0"]):
        for seed in cfg.get("seeds", [0]):
            print(f"=== Cell {cell} seed {seed} ===")
            ckpt = _maybe_export_hf(root, cell, seed, args.base_model)
            if ckpt is None:
                print(f"  no checkpoint; skipping")
                continue

            diag_root = root / "diagnostics" / f"cell_{cell}_seed_{seed}"
            diag_root.mkdir(parents=True, exist_ok=True)
            roll_path = diag_root / "rollouts_eval.jsonl"

            if not roll_path.exists():
                eval_rc = EvalRolloutConfig(
                    learner_checkpoint=str(ckpt),
                    learner_role=learner_role,
                    learner_prompt=learner_prompt,
                    partner_provider=partner.get("provider", "anthropic"),
                    partner_model=partner.get("model", "claude-sonnet-4-6"),
                    cache_dir=partner.get("cache_dir"),
                    budget_state=partner.get("budget_state"),
                    budget_cap_usd=partner.get("budget_cap_usd"),
                    episodes=eval_cfg.get("episodes", 32),
                    seed=eval_cfg.get("seed", 99) + seed,
                    dialogue_turns=eval_cfg.get("dialogue_turns", 10),
                    token_budget=eval_cfg.get("token_budget", 128),
                    temperature=eval_cfg.get("temperature", 0.7),
                    max_tokens=eval_cfg.get("max_tokens", 256),
                    device=eval_cfg.get("device", "cuda:0"),
                    dump_hidden_states=("probing" in diagnostics_to_run),
                )
                collect_eval_rollouts(eval_rc, roll_path)

            cell_summary: dict = {}
            if "mimicry" in diagnostics_to_run:
                cell_summary["mimicry"] = mimicry.run(
                    roll_path, cell_a, learner_role, diag_root,
                )
            if "hull" in diagnostics_to_run:
                cell_summary["hull"] = hull.run(
                    roll_path, cell_a, learner_role, diag_root,
                )
            if "probing" in diagnostics_to_run:
                cell_summary["probing"] = probing.run(roll_path, diag_root)
            if "ood" in diagnostics_to_run:
                cell_summary["ood"] = ood.run(
                    learner_checkpoint=str(ckpt),
                    learner_role=learner_role,
                    learner_prompt=learner_prompt,
                    output_dir=diag_root / "ood",
                    partner_provider=partner.get("provider", "anthropic"),
                    partner_model=partner.get("model", "claude-sonnet-4-6"),
                    weak_partner_path=cfg.get("weak_partner_path"),
                    episodes=eval_cfg.get("episodes", 32),
                    cache_dir=partner.get("cache_dir"),
                    device=eval_cfg.get("device", "cuda:0"),
                )

            summary.setdefault(cell, {})[seed] = cell_summary

    (root / "diagnostics" / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote {root / 'diagnostics' / 'summary.json'}")


if __name__ == "__main__":
    main()
