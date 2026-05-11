"""
Character-prompt absorption diagnostic.

For each (iteration-checkpoint, character, episode) we compute the NLL of the
agent's assistant tokens twice under that character's trained LoRA adapter:

  with    : the original system prompt (env preamble + character bullets)
  without : the same system prompt with `{character_description_bullets}`
            replaced by "(none)". Everything else — env description, the
            character's own name, any trailing instructions — is preserved.

The hypothesis: if the character has been "absorbed" into the adapter weights,
the bullets become redundant and `gap = nll_without - nll_with` shrinks toward
zero across training iterations. (See conversation 2026-05-05 for derivation.)

Inputs that already exist on disk
---------------------------------
  - <run>/checkpoints/iter_NNNNNN/<character>.pt   per-character LoRA weights
  - <run>/traces/{ckpt_NNNNNN/traces.json | iter_NNNNNN.json}
       each episode contains agents[<character>].context_tokens (raw IDs the
       model actually saw) and context_text (same content as a string, used
       only to locate the system-block boundary).
  - <run>/experiment_config.yaml                   names env template + chars

Parallelism
-----------
SLURM-driven data parallelism. Each task loads its own base model and processes
a stride of (run, iter, character) work units indexed by SLURM_PROCID /
SLURM_NTASKS, writing a per-rank shard `absorption_rank{N}.jsonl`. No NCCL.

Usage
-----
  srun --ntasks=4 --gpus-per-task=1 \\
       uv run python scripts/analyze_character_absorption.py \\
           --run-dir runs/cultural_emergence/run7_8agent_7B_margin_notes \\
           --output  runs/cultural_emergence/run7_8agent_7B_margin_notes/absorption \\
           --episodes-per-iter 8
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


# ---------------------------------------------------------------------------
# Rank / work partitioning
# ---------------------------------------------------------------------------
def rank() -> int:
    return int(os.environ.get("SLURM_PROCID", os.environ.get("RANK", "0")))


def world_size() -> int:
    return int(os.environ.get("SLURM_NTASKS", os.environ.get("WORLD_SIZE", "1")))


def log(msg: str) -> None:
    print(f"[rank {rank()}/{world_size()}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Run discovery
# ---------------------------------------------------------------------------
TRACE_NAME_RE = re.compile(r"(?:ckpt|iter)_(\d+)")


def list_iters_with_traces(run_dir: Path) -> list[tuple[int, Path]]:
    """Return [(iter_int, trace_json_path), ...] sorted, deduped by iter.

    Some runs have both `ckpt_NNNNNN/traces.json` (the canonical traces
    rolled out at the checkpoint) and `iter_NNNNNN.json` (rolling training
    traces). When both exist for the same iteration we prefer the ckpt-dir
    one, since those rollouts use exactly the checkpoint's adapter.
    """
    traces_dir = run_dir / "traces"
    out: dict[int, Path] = {}
    flat: dict[int, Path] = {}
    if not traces_dir.exists():
        return []
    for entry in traces_dir.iterdir():
        m = TRACE_NAME_RE.match(entry.name)
        if not m:
            continue
        it = int(m.group(1))
        if entry.is_dir():
            tp = entry / "traces.json"
            if tp.exists():
                out[it] = tp
        elif entry.suffix == ".json":
            flat[it] = entry
    for it, p in flat.items():
        out.setdefault(it, p)
    return sorted(out.items())


def checkpoint_dir(run_dir: Path, it: int) -> Path:
    return run_dir / "checkpoints" / f"iter_{it:06d}"


# ---------------------------------------------------------------------------
# System-block surgery
# ---------------------------------------------------------------------------
# Matches the bullets section in the trace's own system prompt:
#   "Your background and beliefs:\n- bullet\n- bullet\n"
#   "your background:\n- bullet\n"
#   "your character:\n- bullet\n"
# Header is any line ending in ":" with the word "background", "beliefs",
# or "character" in it; bullets are consecutive lines starting with "- ".
_BULLETS_RE = re.compile(
    r"^([Yy]our [^\n:]*(?:background|beliefs|character)[^\n:]*:)\s*\n((?:- .*\n)+)",
    re.MULTILINE,
)


def strip_character_bullets(system_block_text: str) -> tuple[str, bool]:
    """Replace the character-bullets section with `<header>\\n(none)\\n`.

    Returns (modified_text, found_bullets). `found_bullets=False` means the
    regex didn't match — the caller should skip that episode rather than
    score a useless ablation.
    """
    m = _BULLETS_RE.search(system_block_text)
    if not m:
        return system_block_text, False
    header = m.group(1)
    new = system_block_text[: m.start()] + f"{header}\n(none)\n" + system_block_text[m.end():]
    return new, True


# ---------------------------------------------------------------------------
# Token-level work
# ---------------------------------------------------------------------------
def build_assistant_pattern(tokenizer) -> tuple[list[int], int]:
    """Return (`<|im_start|>assistant\\n` token ids, `<|im_end|>` token id)."""
    prefix = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    end_ids = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    if len(end_ids) != 1:
        raise RuntimeError(f"Expected `<|im_end|>` to be a single token, got {end_ids}")
    return prefix, end_ids[0]


def find_assistant_spans(
    token_ids: list[int],
    prefix: list[int],
    im_end_id: int,
) -> list[tuple[int, int]]:
    """Return [(body_start, body_end), ...] half-open ranges in token_ids.

    `body_start` is the first content token after the `<|im_start|>assistant\\n`
    prefix; `body_end` is the position of the closing `<|im_end|>` (exclusive).
    """
    spans: list[tuple[int, int]] = []
    n, k = len(token_ids), len(prefix)
    i = 0
    while i + k <= n:
        if token_ids[i : i + k] == prefix:
            body_start = i + k
            j = body_start
            while j < n and token_ids[j] != im_end_id:
                j += 1
            spans.append((body_start, j))
            i = j + 1
        else:
            i += 1
    return spans


def find_system_boundary(
    context_text: str,
    context_tokens: list[int],
    tokenizer,
) -> int:
    """Return n such that context_tokens[:n] re-encodes to the system block.

    The system block is `<|im_start|>system\\n...<|im_end|>\\n` — i.e. up to
    and including the newline after the closing `<|im_end|>`.
    """
    sys_close = context_text.find("<|im_end|>")
    if sys_close < 0:
        raise ValueError("no <|im_end|> found in context_text")
    sys_block = context_text[: sys_close + len("<|im_end|>\n")]
    sys_tokens = tokenizer.encode(sys_block, add_special_tokens=False)
    n = len(sys_tokens)
    # Verify the boundary matches the recorded tokens. If the tokenizer
    # doesn't round-trip exactly we widen the search ±2 tokens.
    if context_tokens[:n] == sys_tokens:
        return n
    for delta in (-2, -1, 1, 2):
        m = n + delta
        if 0 < m <= len(context_tokens) and context_tokens[:m] == tokenizer.encode(
            tokenizer.decode(context_tokens[:m], skip_special_tokens=False),
            add_special_tokens=False,
        ):
            return m
    raise RuntimeError(
        f"could not align system boundary: re-tokenized len={n} "
        f"but context_tokens[:{n}] differs"
    )


def score_nll(
    model,
    token_ids: list[int],
    span_positions: list[tuple[int, int]],
    device: torch.device,
) -> tuple[float, int]:
    """Mean NLL (nats/token) over the union of span positions.

    Returns (sum_nll, n_tokens). For each span [s, e) we score positions
    [s, e) — i.e. predict body token at position p from logits at p-1.
    """
    if not span_positions:
        return 0.0, 0
    inp = torch.tensor([token_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model(input_ids=inp, use_cache=False)
        logits = out.logits[0, :-1, :].float()       # [T-1, V]
        targets = inp[0, 1:]                          # [T-1]
        # We want NLL on positions p in [s, e) with target=token_ids[p],
        # logits-from-position p-1 → logits[p-1] predicts targets[p-1] in
        # the shifted view. So in shifted indexing, score position (p-1).
        mask = torch.zeros(logits.shape[0], dtype=torch.bool, device=device)
        for s, e in span_positions:
            # predicting positions [max(s,1), e) → shifted indices [s-1, e-1)
            lo, hi = max(s - 1, 0), max(e - 1, 0)
            if hi > lo:
                mask[lo:hi] = True
        if not mask.any():
            return 0.0, 0
        logp = F.log_softmax(logits[mask], dim=-1)
        tgt = targets[mask]
        nll = -logp.gather(1, tgt.unsqueeze(1)).squeeze(1)
    return float(nll.sum().item()), int(nll.numel())


# ---------------------------------------------------------------------------
# Adapter loader
# ---------------------------------------------------------------------------
def load_adapter_weights(
    peft_model,
    adapter_pt_path: Path,
    target_adapter_name: str,
) -> int:
    payload = torch.load(adapter_pt_path, map_location="cpu", weights_only=False)
    src_adapter = payload.get("adapter_name") or target_adapter_name
    adapter_state = payload["adapter"]
    if src_adapter != target_adapter_name:
        adapter_state = {
            k.replace(f".{src_adapter}.", f".{target_adapter_name}."): v
            for k, v in adapter_state.items()
        }
    target_params = dict(peft_model.named_parameters())
    loaded = 0
    for k, v in adapter_state.items():
        tgt = target_params.get(k)
        if tgt is None:
            continue
        with torch.no_grad():
            tgt.data.copy_(v.to(tgt.device, dtype=tgt.dtype))
        loaded += 1
    return loaded


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1] if __doc__ else "")
    p.add_argument("--run-dir", action="append", required=True,
                   help="Run directory (may be passed multiple times).")
    p.add_argument("--output", required=True,
                   help="Output directory for per-rank JSONL shards.")
    p.add_argument("--max-iters", type=int, default=None,
                   help="Cap on number of iteration checkpoints per run.")
    p.add_argument("--episodes-per-iter", type=int, default=None,
                   help="Cap on episodes scored per (iter, character).")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--attn-impl", default="sdpa")
    p.add_argument("--include-iter-zero", action="store_true",
                   help="Also score the iter_0 (untrained) checkpoint as a control.")
    return p.parse_args()


def build_work_units(args) -> list[dict]:
    units: list[dict] = []
    for rd in args.run_dir:
        run_dir = Path(rd).resolve()
        cfg = yaml.safe_load((run_dir / "experiment_config.yaml").read_text())
        characters = cfg.get("characters")
        if isinstance(characters, str):  # references a character file by name
            characters = None
        iter_traces = list_iters_with_traces(run_dir)
        if not args.include_iter_zero:
            iter_traces = [(it, p) for it, p in iter_traces if it > 0]
        # Filter to iterations that actually have a checkpoint, *then* cap.
        iter_traces = [
            (it, p) for it, p in iter_traces if checkpoint_dir(run_dir, it).exists()
        ]
        if args.max_iters is not None:
            iter_traces = iter_traces[: args.max_iters]
        for it, trace_path in iter_traces:
            ck = checkpoint_dir(run_dir, it)
            char_files = sorted(p for p in ck.iterdir() if p.suffix == ".pt" and p.stem != "meta")
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

    # ── Load base model + scaffold one LoRA adapter ─────────────────────────
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    # Use the first run's config as authoritative for model + LoRA dims.
    first_run = Path(my_units[0]["run_dir"])
    cfg = json.loads((first_run / "config.json").read_text())
    model_name = cfg["model_name_or_path"]
    lora_r = cfg["lora_r"]
    lora_alpha = cfg["lora_alpha"]
    lora_target_modules = cfg.get("lora_target_modules", ["q_proj", "v_proj"])

    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    torch_dtype = dtype_map[args.dtype]

    if torch.cuda.is_available():
        device = torch.device("cuda:0")  # SLURM gives each task its own GPU
    else:
        device = torch.device("cpu")

    log(f"loading {model_name} on {device} (dtype={args.dtype})")
    base = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        attn_implementation=args.attn_impl,
    ).to(device)
    base.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    target_adapter = "active"
    lora_cfg = LoraConfig(
        r=lora_r, lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=0.0, bias="none",
    )
    peft_model = get_peft_model(base, lora_cfg, adapter_name=target_adapter)
    peft_model.set_adapter(target_adapter)
    peft_model.eval()

    prefix_ids, im_end_id = build_assistant_pattern(tokenizer)

    # ── Per-trace cache (one trace file at a time to bound memory) ─────────
    trace_cache: dict[str, list] = {}

    out_path = out_dir / f"absorption_rank{rank():02d}.jsonl"
    log(f"writing → {out_path}")
    f_out = out_path.open("w")
    t0 = time.time()

    for u_idx, u in enumerate(my_units):
        run_dir = u["run_dir"]

        if u["trace_path"] not in trace_cache:
            trace_cache.clear()  # one trace file at a time to bound memory
            trace_cache[u["trace_path"]] = json.loads(Path(u["trace_path"]).read_text())
        episodes = trace_cache[u["trace_path"]]

        load_adapter_weights(peft_model, Path(u["ckpt_path"]), target_adapter)
        peft_model.set_adapter(target_adapter)

        char = u["character"]
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

            # Surgery on the actual system block as it appeared in the trace,
            # not on a re-rendered template. Robust to env-template variations.
            sys_block_text = ctx_text[: ctx_text.find("<|im_end|>") + len("<|im_end|>\n")]
            sys_without_text, found = strip_character_bullets(sys_block_text)
            if not found:
                log(f"  skip ep{ep.get('episode')} char={char}: bullets section not detected")
                continue
            sys_without_tokens = tokenizer.encode(sys_without_text, add_special_tokens=False)

            spans_with = find_assistant_spans(ctx_tokens, prefix_ids, im_end_id)
            if not spans_with:
                continue

            # Without-character: swap system block, shift spans by Δ.
            without_tokens = sys_without_tokens + ctx_tokens[sys_end:]
            delta = len(sys_without_tokens) - sys_end
            spans_without = [(s + delta, e + delta) for s, e in spans_with]

            try:
                nll_with_sum, n_with = score_nll(peft_model, ctx_tokens, spans_with, device)
                nll_without_sum, n_without = score_nll(peft_model, without_tokens, spans_without, device)
            except torch.cuda.OutOfMemoryError:
                log(f"  OOM on ep{ep.get('episode')} char={char} (T={len(ctx_tokens)}); skipping")
                torch.cuda.empty_cache()
                continue

            assert n_with == n_without, (n_with, n_without)
            row = {
                "run_dir": run_dir,
                "iter": u["iter"],
                "character": char,
                "episode": ep.get("episode"),
                "n_action_tokens": n_with,
                "context_len_with": len(ctx_tokens),
                "context_len_without": len(without_tokens),
                "mean_nll_with": nll_with_sum / max(1, n_with),
                "mean_nll_without": nll_without_sum / max(1, n_without),
                "gap": (nll_without_sum - nll_with_sum) / max(1, n_with),
            }
            f_out.write(json.dumps(row) + "\n")
            f_out.flush()
            scored += 1

        if u_idx % 5 == 0:
            elapsed = time.time() - t0
            log(f"  done unit {u_idx + 1}/{len(my_units)} "
                f"(iter={u['iter']} char={char} scored={scored}) "
                f"elapsed={elapsed:.0f}s")

    f_out.close()
    log(f"finished in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
