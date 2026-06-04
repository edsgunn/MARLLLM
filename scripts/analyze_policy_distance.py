"""Direct policy-distance + entropy measurement (Diagnostics 2 and 3).

For each (dataset, character slot, iter) work unit, evaluate all 5 seeds'
per-character LoRA adapters on the same set of contexts and measure:

  - Per-position entropy under each adapter
  - Per-position NLL of the gold action token
  - Pairwise symmetric KL between every pair of adapters

Contexts: we use each seed's own held-out evaluation episodes as the
context source — i.e. for every (eval_source ∈ 5 seeds) we score all 5
adapters on that source's traces. This is more work than picking a single
canonical source, but it gives:

  D2 (entropy under own vs sibling at the agent's own action positions):
      take rows where (eval_source == adapter_seed) → "own entropy"
      versus rows where (eval_source == agent_seed) and adapter ∈ other 4
        → "sibling entropy on agent's own samples"

  D3 (policy distance at matched contexts):
      sym_kl rows aggregated over all eval_sources and episodes give
      symmetric KL averaged over a broad context distribution.

Output JSONL (one row per measurement):
  type=entropy: dataset, character, iter, eval_source, episode, adapter,
                n_action_tokens, mean_entropy, nll_sum, mean_kl_to_uniform
  type=sym_kl:  dataset, character, iter, eval_source, episode,
                adapter_a, adapter_b, n_action_tokens, mean_sym_kl
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from analyze_character_absorption import (  # noqa: E402
    rank, world_size, log,
    list_iters_with_traces, checkpoint_dir,
    build_assistant_pattern, find_assistant_spans,
    load_adapter_weights,
)


# Dataset config — three panels of 5 seeds each.
DATASETS = {
    "innerloop": {
        "seeds": [
            "run7_8agent_7B_inner_loop",
            "run7_8agent_7B_inner_loop_seed2",
            "run7_8agent_7B_inner_loop_seed3",
            "run7_8agent_7B_inner_loop_seed4",
            "run7_8agent_7B_inner_loop_seed5",
        ],
        "iters": [25, 50, 75, 100],
    },
    "marginnotes": {
        "seeds": [
            "run7_8agent_7B_margin_notes",
            "run7_8agent_7B_margin_notes_seed2",
            "run7_8agent_7B_margin_notes_seed3",
            "run7_8agent_7B_margin_notes_seed4",
            "run7_8agent_7B_margin_notes_seed5",
        ],
        "iters": [25, 50, 75, 100, 125, 150],
    },
    "ashbourne16": {
        "seeds": [
            "run8_16agent_7B_study_group_ashbourne_gc",
            "run8_16agent_7B_study_group_ashbourne_gc_seed2",
            "run8_16agent_7B_study_group_ashbourne_gc_seed3",
            "run8_16agent_7B_study_group_ashbourne_gc_seed4",
            "run8_16agent_7B_study_group_ashbourne_gc_seed5",
        ],
        "iters": [25, 50, 75, 100, 125, 150],
    },
}


def seed_label(run_name: str) -> str:
    import re
    m = re.search(r"_seed(\d+)$", run_name)
    return f"seed{m.group(1)}" if m else "seed1"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default="innerloop,marginnotes,ashbourne16")
    p.add_argument("--episodes-per-source", type=int, default=4)
    p.add_argument("--output", required=True)
    p.add_argument("--runs-root", default="runs/cultural_emergence")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--attn-impl", default="sdpa")
    return p.parse_args()


def build_work_units(runs_root: Path, dataset_names: list[str]) -> list[dict]:
    """Work unit: (dataset, character, iter). One per (slot, ckpt) pair
    where AT LEAST one seed has the adapter."""
    units = []
    for ds_name in dataset_names:
        cfg = DATASETS[ds_name]
        # Use first seed as character roster reference
        first_seed = runs_root / cfg["seeds"][0]
        for it in cfg["iters"]:
            ck = checkpoint_dir(first_seed, it)
            if not ck.exists():
                continue
            char_files = sorted(p for p in ck.iterdir()
                                if p.suffix == ".pt" and p.stem != "meta")
            for cp in char_files:
                # Locate per-seed adapter path for this (char, iter); skip seeds
                # that don't have it.
                ckpt_paths = {}
                for seed_run in cfg["seeds"]:
                    sd = runs_root / seed_run
                    candidate = checkpoint_dir(sd, it) / cp.name
                    if candidate.exists():
                        ckpt_paths[seed_label(seed_run)] = str(candidate)
                if len(ckpt_paths) < 2:
                    continue
                units.append({"dataset": ds_name,
                              "iter": it,
                              "character": cp.stem,
                              "ckpt_paths": ckpt_paths,
                              "seed_runs": {seed_label(r): str(runs_root / r)
                                            for r in cfg["seeds"]}})
    return units


def first_assistant_token_positions(token_ids, prefix_ids, im_end_id):
    spans = find_assistant_spans(token_ids, prefix_ids, im_end_id)
    positions = []
    for s, e in spans:
        # in shifted-token indexing: predicting position p means logits at p-1
        # we want positions [s, e) → logits at [s-1, e-1)
        positions.extend(range(max(s - 1, 0), max(e - 1, 0)))
    return spans, positions


def compute_pos_logp(model, token_ids, positions, device):
    """Return [n_pos, vocab] log-softmax + targets [n_pos]."""
    inp = torch.tensor([token_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(input_ids=inp, use_cache=False).logits[0, :-1, :].float()
        targets = inp[0, 1:]
        sel = torch.tensor(positions, dtype=torch.long, device=device)
        sub_logits = logits.index_select(0, sel)
        logp = F.log_softmax(sub_logits, dim=-1)
        tgt = targets.index_select(0, sel)
    return logp, tgt


def main():
    args = parse_args()
    out_dir = Path(args.output); out_dir.mkdir(parents=True, exist_ok=True)
    runs_root = REPO / args.runs_root
    dataset_names = [d.strip() for d in args.datasets.split(",")]

    units = build_work_units(runs_root, dataset_names)
    my_units = units[rank()::world_size()]
    log(f"total work units: {len(units)}, this rank: {len(my_units)}")
    if not my_units:
        return

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    # Use first unit's first available run to find model+lora config.
    first_unit = my_units[0]
    first_seed_dir = Path(next(iter(first_unit["seed_runs"].values())))
    cfg = json.loads((first_seed_dir / "config.json").read_text())
    model_name = cfg["model_name_or_path"]
    lora_r, lora_alpha = cfg["lora_r"], cfg["lora_alpha"]
    lora_target_modules = cfg.get("lora_target_modules", ["q_proj", "v_proj"])
    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16,
                 "float16": torch.float16}
    torch_dtype = dtype_map[args.dtype]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log(f"loading {model_name} on {device}")
    base = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch_dtype, attn_implementation=args.attn_impl
    ).to(device)
    base.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    target_adapter = "active"
    lora_cfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha,
                          target_modules=lora_target_modules,
                          lora_dropout=0.0, bias="none")
    peft_model = get_peft_model(base, lora_cfg, adapter_name=target_adapter)
    peft_model.set_adapter(target_adapter); peft_model.eval()

    prefix_ids, im_end_id = build_assistant_pattern(tokenizer)

    # Cache traces by (seed_run_path, iter)
    trace_cache: dict[tuple[str, int], list] = {}

    def load_traces_for(seed_run: Path, it: int):
        key = (str(seed_run), it)
        if key in trace_cache:
            return trace_cache[key]
        iter_traces = list_iters_with_traces(seed_run)
        exact = [tp for (i, tp) in iter_traces if i == it]
        if not exact:
            trace_cache[key] = []
            return []
        episodes = json.loads(Path(exact[0]).read_text())
        trace_cache[key] = episodes
        return episodes

    out_path = out_dir / f"policy_distance_rank{rank():02d}.jsonl"
    log(f"writing → {out_path}")
    f_out = out_path.open("w")
    t0 = time.time()

    for u_idx, u in enumerate(my_units):
        char = u["character"]
        it = u["iter"]
        ds = u["dataset"]
        adapter_seeds = sorted(u["ckpt_paths"].keys())  # subset of 5

        # For each eval source (each seed's traces at this iter), iterate
        # episodes and score under each adapter.
        for eval_seed_label, eval_run_str in u["seed_runs"].items():
            episodes = load_traces_for(Path(eval_run_str), it)
            if not episodes:
                continue
            for ep_idx, ep in enumerate(episodes):
                if ep_idx >= args.episodes_per_source:
                    break
                agents = ep.get("agents", {})
                if char not in agents:
                    continue
                ctx_tokens = list(agents[char]["context_tokens"])
                spans, positions = first_assistant_token_positions(
                    ctx_tokens, prefix_ids, im_end_id)
                if not positions:
                    continue
                # Score under each adapter, keeping logp in memory.
                logp_by_seed: dict[str, torch.Tensor] = {}
                target_tokens = None
                for adapter in adapter_seeds:
                    try:
                        load_adapter_weights(
                            peft_model, Path(u["ckpt_paths"][adapter]),
                            target_adapter)
                        peft_model.set_adapter(target_adapter)
                    except Exception as e:
                        log(f"  adapter load failed {adapter} {char} it{it}: {e}")
                        continue
                    try:
                        logp, tgt = compute_pos_logp(
                            peft_model, ctx_tokens, positions, device)
                    except torch.cuda.OutOfMemoryError:
                        log(f"  OOM ds={ds} char={char} it={it} ep{ep_idx}; skip")
                        torch.cuda.empty_cache()
                        continue
                    logp_by_seed[adapter] = logp  # [n_pos, vocab]
                    target_tokens = tgt
                if not logp_by_seed:
                    continue
                n_tok = len(positions)
                # Write per-adapter entropy rows
                for adapter, logp in logp_by_seed.items():
                    p = logp.exp()
                    entropy = -(p * logp).sum(dim=-1)         # [n_pos]
                    nll = -logp.gather(1, target_tokens.unsqueeze(1)).squeeze(1)
                    f_out.write(json.dumps({
                        "type": "entropy",
                        "dataset": ds, "character": char, "iter": it,
                        "eval_source": eval_seed_label,
                        "episode": ep.get("episode", ep_idx),
                        "adapter": adapter,
                        "n_action_tokens": n_tok,
                        "mean_entropy": float(entropy.mean().item()),
                        "nll_sum": float(nll.sum().item()),
                    }) + "\n")
                # Pairwise sym KL
                for a, b in itertools.combinations(adapter_seeds, 2):
                    if a not in logp_by_seed or b not in logp_by_seed:
                        continue
                    la, lb = logp_by_seed[a], logp_by_seed[b]
                    pa, pb = la.exp(), lb.exp()
                    kl_ab = (pa * (la - lb)).sum(dim=-1)  # [n_pos]
                    kl_ba = (pb * (lb - la)).sum(dim=-1)
                    sym_kl = 0.5 * (kl_ab + kl_ba)
                    f_out.write(json.dumps({
                        "type": "sym_kl",
                        "dataset": ds, "character": char, "iter": it,
                        "eval_source": eval_seed_label,
                        "episode": ep.get("episode", ep_idx),
                        "adapter_a": a, "adapter_b": b,
                        "n_action_tokens": n_tok,
                        "mean_sym_kl": float(sym_kl.mean().item()),
                    }) + "\n")
                # free memory
                logp_by_seed.clear()
                del target_tokens
            f_out.flush()

        if u_idx % 4 == 0:
            log(f"  unit {u_idx+1}/{len(my_units)} (ds={ds} char={char} it={it}) "
                f"elapsed={time.time()-t0:.0f}s")

    f_out.close()
    log(f"finished in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
