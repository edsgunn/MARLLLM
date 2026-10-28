#!/usr/bin/env python
"""Final-hidden-state norm across training, measured on fresh rollouts.

For each saved checkpoint of a population run (iter 0 = the untrained policy,
i.e. zero-initialised LoRA), load the adapters, push them to vLLM, roll out
fresh episodes exactly as the trainer does, then teacher-force each agent's
trajectory back through its own adapter and record, per position:

  norm_post   ||h||  of the final hidden state the trainer feeds lm_head
              (`out.hidden_states[-1]`, i.e. AFTER the final RMSNorm)
  norm_pre    ||h||  of the residual stream entering the final RMSNorm
  entropy     H(softmax(lm_head(h_post)))  — the trainer's `<agent>/entropy`
              is the mean of this over ACT positions (marlllm/loss.py)
  logit_std   std of the logits over the vocabulary (the logit scale)

Positions are split by what they predict: ACT (a token the agent generated)
and OBS (a token inserted into its context) — the trainer's σ split.

Two data sets per checkpoint:
  onpolicy    fresh episodes rolled out by THIS checkpoint. Norm changes here
              mix weight change with the change in what the policy writes.
  fixed       the iter-0 trajectories, re-scored under this checkpoint. Inputs
              are held fixed, so any change is the weights alone.

The trainer is built by `train_population.main()` from the run's own
experiment_config.yaml, so env, roster, prompts and vLLM settings match
training; only `PopulationTrainer.train` is replaced by the measurement loop.
Env seeds are reset to the same value before every checkpoint's rollout, so
checkpoints are compared on common random numbers.

Outputs (in --output):
  episodes.jsonl            one row per (iter, data, agent, episode, subset)
  tokens_iter{N}.npz        per-ACT-token arrays for the on-policy set
  meta.json                 agent names, iters, run dir

Usage:
  .venv/bin/python scripts/rollout_hidden_norms.py \\
      --run-dir runs/cultural_emergence/run7_8agent_7B_study_group_ashbourne_gc \\
      --output  runs/cultural_emergence/_hidden_norms/run7_8agent_7B_study_group_ashbourne_gc
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

_V_CHUNK = 512   # positions per lm_head chunk; (512, 152k) fp32 ≈ 300 MiB
_ROLLOUT_SEED = 1_000_003


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--iters", default=None,
                   help="Comma-separated checkpoint iters (default: 0 + every iter_* checkpoint).")
    p.add_argument("--episodes-per-ckpt", type=int, default=32)
    p.add_argument("--fixed-episodes", type=int, default=8,
                   help="Iter-0 trajectories per agent re-scored under every checkpoint.")
    p.add_argument("--vllm-gpu-mem-util", type=float, default=0.30)
    return p.parse_args()


def checkpoint_iters(run_dir: Path) -> list[int]:
    its = sorted(int(d.name.split("_")[1]) for d in (run_dir / "checkpoints").glob("iter_*"))
    return [0] + its


@torch.no_grad()
def position_stats(agent, token_ids: list[int], captured: dict) -> dict[str, torch.Tensor]:
    """Per-position stats for positions 0..T-2 (position p predicts token p+1)."""
    inp = torch.tensor([token_ids], dtype=torch.long, device=agent.device)
    post, _ = agent.evaluate_hidden(inp, torch.ones_like(inp))
    post = post[0, :-1]
    pre = captured.pop("pre")[0, :-1]
    ent, lstd = [], []
    for s in range(0, post.shape[0], _V_CHUNK):
        logits = agent.lm_head(post[s:s + _V_CHUNK]).float()
        logp = F.log_softmax(logits, dim=-1)
        ent.append(-(logp.exp() * logp).sum(-1))
        lstd.append(logits.std(-1))
    return {
        "norm_post": post.float().norm(dim=-1),
        "norm_pre": pre.float().norm(dim=-1),
        "entropy": torch.cat(ent),
        "logit_std": torch.cat(lstd),
    }


def summarise(stats: dict[str, torch.Tensor], mask: torch.Tensor) -> dict:
    row = {"n": int(mask.sum())}
    if row["n"] == 0:
        return row
    for k, v in stats.items():
        x = v[mask]
        row[f"{k}_mean"] = float(x.mean())
        row[f"{k}_median"] = float(x.median())
        row[f"{k}_p90"] = float(torch.quantile(x, 0.9)) if x.numel() < 16_000_000 else float("nan")
    return row


def measure(trainer, args) -> None:
    from marlllm.types import TokenType

    run_dir = Path(args.run_dir)
    out = Path(args.output)
    names = list(trainer.population)
    first = trainer.population[names[0]]
    norm_mod = first._backbone.get_base_model().model.norm
    captured: dict = {}
    norm_mod.register_forward_hook(lambda m, inp, o: captured.__setitem__("pre", inp[0].detach()))

    iters = ([int(x) for x in args.iters.split(",")] if args.iters
             else checkpoint_iters(run_dir))
    assert iters[0] == 0, "iter 0 must come first: it is the only state with fresh adapters"
    (out / "meta.json").write_text(json.dumps(
        {"run_dir": str(run_dir), "agents": names, "iters": iters,
         "episodes_per_ckpt": args.episodes_per_ckpt}, indent=2))

    rows_f = (out / "episodes.jsonl").open("w")
    fixed: dict[str, list] = {}
    checked_norm = False
    t0 = time.time()

    def score(it, data, pop_name, ep_idx, traj, dump):
        nonlocal checked_norm
        ids = traj.token_ids()
        types = traj.type_mask()
        stats = position_stats(trainer.population[pop_name], ids, captured)
        if not checked_norm:
            # Sanity: the hook must be catching the residual stream that the
            # final RMSNorm maps to hidden_states[-1].
            w = norm_mod.weight.float()
            pre = stats["norm_pre"]
            print(f"[check] ||h_pre|| mean {pre.mean():.1f}, ||h_post|| mean "
                  f"{stats['norm_post'].mean():.1f}, ||rmsnorm weight|| {w.norm():.1f}", flush=True)
            checked_norm = True
        tgt = torch.tensor([int(t) for t in types[1:]], device=stats["entropy"].device)
        masks = {"act": tgt == int(TokenType.ACT), "obs": tgt == int(TokenType.OBS)}
        for subset, m in masks.items():
            row = {"iter": it, "data": data, "agent": pop_name, "episode": ep_idx,
                   "subset": subset, "seq_len": len(ids), **summarise(stats, m)}
            rows_f.write(json.dumps(row) + "\n")
        if dump is not None:
            m = masks["act"]
            for k, v in stats.items():
                dump.setdefault(f"{pop_name}|{k}", []).append(v[m].cpu().numpy().astype(np.float32))
            dump.setdefault(f"{pop_name}|episode", []).append(
                np.full(int(m.sum()), ep_idx, dtype=np.int32))

    for it in iters:
        if it > 0:
            ck = run_dir / "checkpoints" / f"iter_{it:06d}"
            got = trainer.load_checkpoint(str(ck))
            assert got == it, (got, it)
            if trainer.sampling_engine is not None:
                trainer.sampling_engine.sync_all_adapters(trainer._sampling_engine_peft_model)
        for agent in trainer.population.values():
            agent.eval_mode()

        trainer._rng_counter = _ROLLOUT_SEED
        trainer._rng = random.Random(_ROLLOUT_SEED)
        random.seed(_ROLLOUT_SEED)
        np.random.seed(_ROLLOUT_SEED)
        torch.manual_seed(_ROLLOUT_SEED)
        t_roll = time.time()
        raw = trainer._collect_episodes_batched(args.episodes_per_ckpt)
        t_roll = time.time() - t_roll

        dump: dict[str, list] = {}
        for pop_name in names:
            eps = [ep for ep in raw if pop_name in ep[2]]
            trajs = trainer._build_agent_trajectories(eps, pop_name)
            if it == 0:
                fixed[pop_name] = trajs[: args.fixed_episodes]
            for e, traj in enumerate(trajs):
                score(it, "onpolicy", pop_name, e, traj, dump)
            for e, traj in enumerate(fixed.get(pop_name, [])):
                score(it, "fixed", pop_name, e, traj, None)
        rows_f.flush()
        np.savez_compressed(out / f"tokens_iter{it:06d}.npz",
                            **{k: np.concatenate(v) for k, v in dump.items()})
        print(f"[iter {it:4d}] {len(raw)} episodes, rollout {t_roll:.0f}s, "
              f"elapsed {time.time() - t0:.0f}s", flush=True)
        torch.cuda.empty_cache()

    rows_f.close()
    print(f"done in {time.time() - t0:.0f}s → {out}", flush=True)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    args.run_dir, args.output = str(run_dir), str(out)

    import train_population
    from marlllm.population import PopulationTrainer
    PopulationTrainer.train = lambda self, start_iteration=1: measure(self, args)
    sys.argv = [
        "train_population.py",
        "--config", str(run_dir / "experiment_config.yaml"),
        "--output-dir", str(out / "_trainer"),
        "--vllm-gpu-mem-util", str(args.vllm_gpu_mem_util),
    ]
    train_population.main()


if __name__ == "__main__":
    main()
