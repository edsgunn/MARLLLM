"""Trajectory-level KL between parallel populations.

For each (target_seed, character, iter) work unit, load the target seed's
LoRA adapter for that character at that iter, then score the action-token
NLL of that character's assistant spans across episodes from *every* seed
in the panel. The aggregator turns these into per-trajectory log-likelihood
ratios summed across all 8 character slots.

Output (one row per (target_seed, source_seed, iter, character, episode))
JSONL shards `trajectory_kl_rank{N}.jsonl`:

  target_run, source_run, iter, character, episode,
  nll_sum, n_action_tokens

Aggregation script computes
  per_traj_loglik(target, source, iter, ep) = sum over chars of nll_sum
  D̂(source → target)_iter = mean over eps of [LL_target(ep) - LL_source(ep)]
                          = mean of [-LL_source - (-LL_target)]
                          = mean of [NLL_target - NLL_source]
(matching the KL definition E_{τ~source}[log p_source(τ) - log p_target(τ)] ).
"""
from __future__ import annotations

import argparse
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


def score_nll_union(model, token_ids: list[int],
                    spans: list[tuple[int, int]], device) -> tuple[float, int]:
    """NLL over the union of `spans` (each [s, e) half-open)."""
    if not spans:
        return 0.0, 0
    inp = torch.tensor([token_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model(input_ids=inp, use_cache=False)
        logits = out.logits[0, :-1, :].float()
        targets = inp[0, 1:]
        mask = torch.zeros(logits.shape[0], dtype=torch.bool, device=device)
        for s, e in spans:
            lo, hi = max(s - 1, 0), max(e - 1, 0)
            if hi > lo:
                mask[lo:hi] = True
        if not mask.any():
            return 0.0, 0
        logp = F.log_softmax(logits[mask], dim=-1)
        tgt = targets[mask]
        nll = -logp.gather(1, tgt.unsqueeze(1)).squeeze(1)
    return float(nll.sum().item()), int(nll.numel())


def parse_args():
    p = argparse.ArgumentParser()
    # Each --seed-run-dir is one population in the panel. The first one's
    # config.json is used to pick the base model + LoRA shape.
    p.add_argument("--seed-run-dir", action="append", required=True,
                   help="Paths to the population runs (target+source pool).")
    p.add_argument("--iters", default="25,50,75,100",
                   help="Comma-separated iters to score.")
    p.add_argument("--output", required=True)
    p.add_argument("--episodes-per-iter", type=int, default=None)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--attn-impl", default="sdpa")
    return p.parse_args()


def build_work_units(seed_dirs: list[Path], iters: list[int]) -> list[dict]:
    """Work units: (target_run, char, iter). One per character-checkpoint
    pair that exists. The rank loads target's adapter and scores across
    all source seeds' traces."""
    units = []
    for target in seed_dirs:
        for it in iters:
            ck = checkpoint_dir(target, it)
            if not ck.exists():
                continue
            char_files = sorted(p for p in ck.iterdir()
                                if p.suffix == ".pt" and p.stem != "meta")
            for cp in char_files:
                units.append({"target_run": str(target),
                              "iter": it,
                              "character": cp.stem,
                              "ckpt_path": str(cp)})
    return units


def main():
    args = parse_args()
    out_dir = Path(args.output); out_dir.mkdir(parents=True, exist_ok=True)
    seed_dirs = [Path(s).resolve() for s in args.seed_run_dir]
    iters = [int(x) for x in args.iters.split(",") if x.strip()]
    log(f"seeds ({len(seed_dirs)}): {[s.name for s in seed_dirs]}")
    log(f"iters: {iters}")

    units = build_work_units(seed_dirs, iters)
    my_units = units[rank()::world_size()]
    log(f"total units: {len(units)}, this rank: {len(my_units)}")
    if not my_units:
        return

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    first_run = Path(my_units[0]["target_run"])
    cfg = json.loads((first_run / "config.json").read_text())
    model_name = cfg["model_name_or_path"]
    lora_r, lora_alpha = cfg["lora_r"], cfg["lora_alpha"]
    lora_target_modules = cfg.get("lora_target_modules", ["q_proj", "v_proj"])
    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16,
                 "float16": torch.float16}
    torch_dtype = dtype_map[args.dtype]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    log(f"loading {model_name}")
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

    # Cache traces by (source_run, iter) -> episodes list
    trace_cache: dict[tuple[str, int], list] = {}

    def load_source_traces(source: Path, it: int):
        key = (str(source), it)
        if key in trace_cache:
            return trace_cache[key]
        iter_traces = list_iters_with_traces(source)
        exact = [tp for (i, tp) in iter_traces if i == it]
        if not exact:
            trace_cache[key] = []
            return []
        episodes = json.loads(Path(exact[0]).read_text())
        trace_cache[key] = episodes
        return episodes

    out_path = out_dir / f"trajectory_kl_rank{rank():02d}.jsonl"
    log(f"writing → {out_path}")
    f_out = out_path.open("w")
    t0 = time.time()

    for u_idx, u in enumerate(my_units):
        target_run = u["target_run"]
        it = u["iter"]
        char = u["character"]
        # Load target's char-LoRA at this iter
        load_adapter_weights(peft_model, Path(u["ckpt_path"]), target_adapter)
        peft_model.set_adapter(target_adapter)

        for source in seed_dirs:
            episodes = load_source_traces(source, it)
            if not episodes:
                continue
            scored_eps = 0
            for ep_idx, ep in enumerate(episodes):
                if args.episodes_per_iter is not None and scored_eps >= args.episodes_per_iter:
                    break
                agents = ep.get("agents", {})
                if char not in agents:
                    continue
                ctx_tokens = list(agents[char]["context_tokens"])
                spans = find_assistant_spans(ctx_tokens, prefix_ids, im_end_id)
                if not spans:
                    continue
                try:
                    nll_sum, n_tok = score_nll_union(peft_model, ctx_tokens, spans, device)
                except torch.cuda.OutOfMemoryError:
                    log(f"  OOM {target_run.split('/')[-1]} ↔ {source.name} ep{ep_idx} char={char}; skipping")
                    torch.cuda.empty_cache()
                    continue
                if n_tok == 0:
                    continue
                row = {
                    "target_run": target_run,
                    "source_run": str(source),
                    "iter": it,
                    "character": char,
                    "episode": ep.get("episode", ep_idx),
                    "nll_sum": nll_sum,
                    "n_action_tokens": n_tok,
                    "mean_nll": nll_sum / n_tok,
                }
                f_out.write(json.dumps(row) + "\n")
                scored_eps += 1
            f_out.flush()

        if u_idx % 4 == 0:
            log(f"  done unit {u_idx+1}/{len(my_units)} "
                f"(target={Path(target_run).name} it={it} char={char}) "
                f"elapsed={time.time()-t0:.0f}s")

    f_out.close()
    log(f"finished in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
