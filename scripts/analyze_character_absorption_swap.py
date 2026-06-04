"""
Swapped-character PMI control for the absorption analysis.

For each (iter-checkpoint, character A, episode) we score the agent's
assistant-token NLL under A's LoRA under three prompt variants:

  with    : A's own system prompt as recorded (env preamble + A's bullets)
  without : same prompt with `{character_bullets}` replaced by "(none)"
  swap    : same prompt with A's bullets replaced by partner B's bullets
            (B = swap partner from a fixed within-population pairing)

Hypothesis: under genuine absorption of *own* character, swap-vs-none
shrinkage stays near zero (a different character actively mispredicts A's
actions) while own-vs-none shrinkage falls. Under generic drift from base,
both shrinkages track each other.

Swap pairing
------------
Per (run, iter): within the trace's character set, sort names and pair
each character with the next (cyclic). Pairing is stable across iters for
a given population (deterministic from the sorted character list, which
doesn't change).

Output
------
JSONL shards `absorption_swap_rank{N}.jsonl` with fields:

  run_dir, iter, character, swap_partner, episode,
  n_action_tokens, context_len_with, context_len_without, context_len_swap,
  mean_nll_with, mean_nll_without, mean_nll_swap,
  gap         = mean_nll_without - mean_nll_with        (own-vs-none)
  gap_swap    = mean_nll_without - mean_nll_swap        (swap-vs-none)
  gap_own_vs_swap = mean_nll_swap - mean_nll_with       (own better than swap?)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn.functional as F
import yaml

# Reuse helpers from the original analyzer.
import sys
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from analyze_character_absorption import (  # noqa: E402
    rank, world_size, log,
    list_iters_with_traces, checkpoint_dir,
    _BULLETS_RE, build_assistant_pattern, find_assistant_spans,
    find_system_boundary, score_nll, load_adapter_weights,
)


# ---------------------------------------------------------------------------
# Bullet extraction
# ---------------------------------------------------------------------------
def extract_bullets_payload(system_block_text: str) -> tuple[str, str, int, int] | None:
    """Return (header, bullets_payload, start, end) for the bullets section
    of a system block, or None if the regex doesn't match.

    Header is `Your background and beliefs:` (or similar); payload is the
    block of consecutive `- ...\\n` lines that follow. The caller can
    substitute payload while keeping header intact.
    """
    m = _BULLETS_RE.search(system_block_text)
    if not m:
        return None
    return m.group(1), m.group(2), m.start(2), m.end(2)


def swap_bullets(system_block_text: str, new_payload: str) -> tuple[str, bool]:
    """Replace the bullets payload in a system block with `new_payload`.
    Keeps the original header. Returns (text, found)."""
    info = extract_bullets_payload(system_block_text)
    if info is None:
        return system_block_text, False
    _, _, s, e = info
    return system_block_text[:s] + new_payload + system_block_text[e:], True


def strip_bullets_payload(system_block_text: str) -> tuple[str, bool]:
    info = extract_bullets_payload(system_block_text)
    if info is None:
        return system_block_text, False
    _, _, s, e = info
    return system_block_text[:s] + "(none)\n" + system_block_text[e:], True


# ---------------------------------------------------------------------------
# Swap-partner pairing
# ---------------------------------------------------------------------------
def build_swap_map(character_names: list[str]) -> dict[str, str]:
    """Deterministic cyclic pairing within a population."""
    names = sorted(character_names)
    n = len(names)
    return {names[i]: names[(i + 1) % n] for i in range(n)}


def extract_partner_bullets_for_trace(
    trace_episodes: list[dict],
    char_names: list[str],
) -> dict[str, str]:
    """For each character in `char_names`, pull their bullets payload from
    the first episode in `trace_episodes` that contains them. Returns
    {character: payload_string}. Missing characters are absent from the
    dict; caller skips swap scoring for those."""
    out: dict[str, str] = {}
    needed = set(char_names)
    for ep in trace_episodes:
        for ch in list(needed):
            agents = ep.get("agents", {})
            if ch not in agents:
                continue
            ct = agents[ch]["context_text"]
            sys_end_idx = ct.find("<|im_end|>")
            if sys_end_idx < 0:
                continue
            sys_block = ct[: sys_end_idx + len("<|im_end|>\n")]
            info = extract_bullets_payload(sys_block)
            if info is None:
                continue
            _, payload, _, _ = info
            out[ch] = payload
            needed.discard(ch)
        if not needed:
            break
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", action="append", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-iters", type=int, default=None)
    p.add_argument("--episodes-per-iter", type=int, default=None)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--attn-impl", default="sdpa")
    p.add_argument("--include-iter-zero", action="store_true")
    return p.parse_args()


def build_work_units(args) -> list[dict]:
    units: list[dict] = []
    for rd in args.run_dir:
        run_dir = Path(rd).resolve()
        iter_traces = list_iters_with_traces(run_dir)
        if not args.include_iter_zero:
            iter_traces = [(it, p) for it, p in iter_traces if it > 0]
        iter_traces = [
            (it, p) for it, p in iter_traces if checkpoint_dir(run_dir, it).exists()
        ]
        if args.max_iters is not None:
            iter_traces = iter_traces[: args.max_iters]
        for it, trace_path in iter_traces:
            ck = checkpoint_dir(run_dir, it)
            char_files = sorted(
                p for p in ck.iterdir() if p.suffix == ".pt" and p.stem != "meta"
            )
            for char_pt in char_files:
                units.append({
                    "run_dir": str(run_dir),
                    "iter": it,
                    "character": char_pt.stem,
                    "trace_path": str(trace_path),
                    "ckpt_path": str(char_pt),
                })
    return units


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    units = build_work_units(args)
    my_units = units[rank()::world_size()]
    log(f"total work units: {len(units)}, this rank: {len(my_units)}")
    if not my_units:
        return

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    first_run = Path(my_units[0]["run_dir"])
    cfg = json.loads((first_run / "config.json").read_text())
    model_name = cfg["model_name_or_path"]
    lora_r = cfg["lora_r"]
    lora_alpha = cfg["lora_alpha"]
    lora_target_modules = cfg.get("lora_target_modules", ["q_proj", "v_proj"])

    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    torch_dtype = dtype_map[args.dtype]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    log(f"loading {model_name} on {device} (dtype={args.dtype})")
    base = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch_dtype, attn_implementation=args.attn_impl,
    ).to(device)
    base.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    target_adapter = "active"
    lora_cfg = LoraConfig(
        r=lora_r, lora_alpha=lora_alpha,
        target_modules=lora_target_modules, lora_dropout=0.0, bias="none",
    )
    peft_model = get_peft_model(base, lora_cfg, adapter_name=target_adapter)
    peft_model.set_adapter(target_adapter)
    peft_model.eval()

    prefix_ids, im_end_id = build_assistant_pattern(tokenizer)

    # Per-trace cache, plus per-trace partner-bullets cache and swap map.
    trace_cache: dict[str, list] = {}
    partner_payloads_cache: dict[str, dict[str, str]] = {}
    swap_map_cache: dict[str, dict[str, str]] = {}

    out_path = out_dir / f"absorption_swap_rank{rank():02d}.jsonl"
    log(f"writing → {out_path}")
    f_out = out_path.open("w")
    t0 = time.time()

    for u_idx, u in enumerate(my_units):
        run_dir = u["run_dir"]
        trace_path = u["trace_path"]

        if trace_path not in trace_cache:
            trace_cache.clear()
            partner_payloads_cache.clear()
            swap_map_cache.clear()
            episodes = json.loads(Path(trace_path).read_text())
            trace_cache[trace_path] = episodes
            # Discover character set present in this trace.
            char_set: set[str] = set()
            for ep in episodes:
                char_set.update(ep.get("agents", {}).keys())
            chars_in_trace = sorted(char_set)
            swap_map_cache[trace_path] = build_swap_map(chars_in_trace)
            partner_payloads_cache[trace_path] = extract_partner_bullets_for_trace(
                episodes, chars_in_trace,
            )
            log(f"  trace {Path(trace_path).name}: {len(chars_in_trace)} chars, "
                f"{len(partner_payloads_cache[trace_path])} payloads extracted")

        episodes = trace_cache[trace_path]
        swap_map = swap_map_cache[trace_path]
        partner_payloads = partner_payloads_cache[trace_path]

        load_adapter_weights(peft_model, Path(u["ckpt_path"]), target_adapter)
        peft_model.set_adapter(target_adapter)

        char = u["character"]
        partner = swap_map.get(char)
        partner_payload = partner_payloads.get(partner) if partner else None
        if partner is None or partner_payload is None:
            log(f"  skip {char}: no swap partner / payload available")
            continue

        scored = 0
        for ep in episodes:
            agents = ep.get("agents", {})
            if char not in agents:
                continue
            if args.episodes_per_iter is not None and scored >= args.episodes_per_iter:
                break
            agent_blob = agents[char]
            ctx_tokens = list(agent_blob["context_tokens"])
            ctx_text = agent_blob["context_text"]

            try:
                sys_end = find_system_boundary(ctx_text, ctx_tokens, tokenizer)
            except Exception as e:
                log(f"  skip ep{ep.get('episode')} char={char}: {e}")
                continue

            sys_block_text = ctx_text[: ctx_text.find("<|im_end|>") + len("<|im_end|>\n")]

            sys_without_text, found_w = strip_bullets_payload(sys_block_text)
            sys_swap_text, found_s = swap_bullets(sys_block_text, partner_payload)
            if not (found_w and found_s):
                log(f"  skip ep{ep.get('episode')} char={char}: bullets section not detected")
                continue

            sys_without_tokens = tokenizer.encode(sys_without_text, add_special_tokens=False)
            sys_swap_tokens = tokenizer.encode(sys_swap_text, add_special_tokens=False)

            spans_with = find_assistant_spans(ctx_tokens, prefix_ids, im_end_id)
            if not spans_with:
                continue

            without_tokens = sys_without_tokens + ctx_tokens[sys_end:]
            delta_wo = len(sys_without_tokens) - sys_end
            spans_without = [(s + delta_wo, e + delta_wo) for s, e in spans_with]

            swap_tokens = sys_swap_tokens + ctx_tokens[sys_end:]
            delta_sw = len(sys_swap_tokens) - sys_end
            spans_swap = [(s + delta_sw, e + delta_sw) for s, e in spans_with]

            try:
                nll_with_sum, n_with = score_nll(peft_model, ctx_tokens, spans_with, device)
                nll_wo_sum, n_wo = score_nll(peft_model, without_tokens, spans_without, device)
                nll_sw_sum, n_sw = score_nll(peft_model, swap_tokens, spans_swap, device)
            except torch.cuda.OutOfMemoryError:
                log(f"  OOM ep{ep.get('episode')} char={char}; skipping")
                torch.cuda.empty_cache()
                continue

            assert n_with == n_wo == n_sw, (n_with, n_wo, n_sw)
            n = max(1, n_with)
            row = {
                "run_dir": run_dir,
                "iter": u["iter"],
                "character": char,
                "swap_partner": partner,
                "episode": ep.get("episode"),
                "n_action_tokens": n_with,
                "context_len_with": len(ctx_tokens),
                "context_len_without": len(without_tokens),
                "context_len_swap": len(swap_tokens),
                "mean_nll_with": nll_with_sum / n,
                "mean_nll_without": nll_wo_sum / n,
                "mean_nll_swap": nll_sw_sum / n,
                "gap":          (nll_wo_sum - nll_with_sum) / n,
                "gap_swap":     (nll_wo_sum - nll_sw_sum) / n,
                "gap_own_vs_swap": (nll_sw_sum - nll_with_sum) / n,
            }
            f_out.write(json.dumps(row) + "\n")
            f_out.flush()
            scored += 1

        if u_idx % 5 == 0:
            log(f"  done unit {u_idx + 1}/{len(my_units)} "
                f"(iter={u['iter']} char={char} partner={partner} scored={scored}) "
                f"elapsed={time.time()-t0:.0f}s")

    f_out.close()
    log(f"finished in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
