"""PMI decomposition analyzer.

For each (run, iter-checkpoint, character A, episode), scores action-token
NLL under up to 8 prompt-construction conditions in order to decompose the
"absorption" signal into character-, environment-, and population-history
components.

Conditions
----------
  own                       — A's system + transcript as recorded
  none_char                 — A's bullets replaced by "(none)"
  swap_char_within          — A's bullets replaced by within-population
                              partner's bullets (cyclic pairing)
  none_env                  — env paragraph replaced by neutral preamble;
                              applied in BOTH system block and first user
                              block (which echoes the env description)
  swap_env                  — env paragraph replaced by another substrate's
                              env paragraph (tool-compat. pairings)
  none_history              — transcript turns dropped (system only)
  swap_history_within_run   — transcript replaced by another episode's
                              transcript from the same run, same iter
  swap_history_cross_run    — transcript replaced by a parallel-seed run's
                              transcript at the same iter (Inner-Loop seeds
                              only; provided via --cross-run-roots)

Scoring scope
-------------
For comparability across conditions, we score *only the first assistant
span* per episode. Multi-turn contexts make "history" well-defined only up
to A's first turn; later turns depend on A's own intervening posts, which
the history swap would have to reconstruct. The original PMI analysis
scored all spans — absolute magnitudes here therefore differ from the
published numbers; compare deltas, not absolutes.

Output
------
JSONL shards `pmi_decomp_rank{N}.jsonl`, one row per (run, iter, char, ep):

  run_dir, iter, character, episode, swap_partner_char, swap_env_substrate,
  n_action_tokens, n_history_tokens_own,
  nll_own, nll_none_char, nll_swap_char_within,
  nll_none_env, nll_swap_env,
  nll_none_history, nll_swap_history_within_run, nll_swap_history_cross_run

Missing conditions are written as null when unavailable (e.g. cross-run
history when the run has no parallel seeds).
"""
from __future__ import annotations

import argparse
import json
import os
import re
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
    find_system_boundary, load_adapter_weights,
)
from analyze_character_absorption_swap import (  # noqa: E402
    extract_bullets_payload, swap_bullets, strip_bullets_payload,
    build_swap_map, extract_partner_bullets_for_trace,
)


# ---------------------------------------------------------------------------
# Env-preamble parsing
# ---------------------------------------------------------------------------
# Env preamble = the paragraph between "You are X, a member of FORUM." and
# "Your background and beliefs:" (or similar). It is **followed by** the
# bullets header.
_ENV_RE = re.compile(
    r"(You are [^\n]+,\s*a member of [^\n]+?\.\s*\n+)"   # group 1: identity line
    r"(.*?)"                                              # group 2: env paragraph
    r"(\n+[Yy]our\s+[^:\n]*?(?:background|beliefs|character)[^:\n]*:\s*\n)",  # group 3: bullets header (anchor)
    re.DOTALL,
)


def parse_env_paragraph(system_block_text: str):
    """Return (text_before, env_paragraph, text_from_bullets_header_onward)
    or None if anchors not found. The env paragraph excludes the trailing
    bullets header — it's the bare descriptive paragraph(s)."""
    m = _ENV_RE.search(system_block_text)
    if not m:
        return None
    pre = system_block_text[: m.start(2)]
    env = m.group(2)
    post = system_block_text[m.start(3):]
    return pre, env, post


def replace_env_paragraph(system_block_text: str, new_env: str) -> tuple[str, bool]:
    """Replace the env paragraph in the system block with `new_env`."""
    parsed = parse_env_paragraph(system_block_text)
    if parsed is None:
        return system_block_text, False
    pre, _, post = parsed
    return pre + new_env + post, True


# The env paragraph is *also* echoed in the first user block of every
# episode (as the "transcript header" preamble). For env-swap conditions
# we replace both occurrences. The user-block version is anchored
# differently: it's the entire text from "<|im_start|>user\n" up to the
# first "\n\n" — i.e. one paragraph.
def replace_env_in_user_block(prompt_text: str,
                              original_env: str,
                              new_env: str) -> tuple[str, bool]:
    """Replace the duplicated env paragraph that appears in the first user
    block. We do this by direct string substitution of the original env
    text, restricted to the first occurrence (which is in the user block —
    the system-block one was already replaced before calling)."""
    idx = prompt_text.find(original_env)
    if idx < 0:
        return prompt_text, False
    return prompt_text[:idx] + new_env + prompt_text[idx + len(original_env):], True


# ---------------------------------------------------------------------------
# Generic / swap env preambles
# ---------------------------------------------------------------------------
GENERIC_ENV = (
    "This is an online discussion forum where members post messages in "
    "turn. Members are knowledgeable, articulate, and direct. They reply "
    "to one another, building on prior posts. Tone varies between "
    "members. It's 2026."
)


# Substrate -> env-swap pairing partner (within tool-compat group).
# Cyclic so the mapping is deterministic and symmetric-ish.
ENV_SWAP_PARTNER = {
    "inner_loop":              "margin_notes",
    "margin_notes":            "inner_loop",
    "study_group":             "study_group_ashbourne_gc",
    "study_group_ashbourne_gc": "study_group_strathearn_server",
    "study_group_strathearn_server": "conjecture",
    "conjecture":              "robotic_athanor",
    "robotic_athanor":         "study_group",
}

# Identity-line forum name needs swapping too in env-swap mode so the
# "You are X, a member of FORUM" sentence doesn't reference the wrong forum.
# We extract FORUM from each substrate's traces lazily; see substrate_assets
# below.


_RUN_ENV_RE = re.compile(r"run\d+_\d+agent_\d+B_(.+?)(?:_seed\d+)?$")


def env_name_from_run(run_dir: Path) -> str:
    m = _RUN_ENV_RE.match(run_dir.name)
    return m.group(1) if m else run_dir.name


def load_substrate_env_assets(run_dir: Path) -> dict:
    """Pull this substrate's env paragraph and forum name from its first
    available trace. Returns {env_paragraph, forum_phrase, sample_block}."""
    iter_traces = list_iters_with_traces(run_dir)
    if not iter_traces:
        return {}
    _, tp = iter_traces[0]
    episodes = json.loads(Path(tp).read_text())
    ep = episodes[0]
    a = next(iter(ep["agents"].keys()))
    ct = ep["agents"][a]["context_text"]
    sys_end = ct.find("<|im_end|>")
    if sys_end < 0:
        return {}
    sys_block = ct[: sys_end + len("<|im_end|>\n")]
    parsed = parse_env_paragraph(sys_block)
    if parsed is None:
        return {}
    _, env_para, _ = parsed
    # Forum phrase: text after "a member of " up to "." in the identity line.
    m = re.search(r"a member of ([^.\n]+)\.", sys_block)
    forum = m.group(1).strip() if m else ""
    return {"env_paragraph": env_para, "forum_phrase": forum,
            "sample_sys_block": sys_block}


# ---------------------------------------------------------------------------
# History extraction / replacement
# ---------------------------------------------------------------------------
def first_assistant_span(token_ids: list[int],
                         prefix: list[int],
                         im_end_id: int) -> tuple[int, int] | None:
    """Return (body_start, body_end) of the FIRST assistant span."""
    spans = find_assistant_spans(token_ids, prefix, im_end_id)
    return spans[0] if spans else None


def first_assistant_offset_in_text(text: str) -> int | None:
    """Character offset of the first `<|im_start|>assistant\\n` in `text`."""
    pat = "<|im_start|>assistant\n"
    idx = text.find(pat)
    if idx < 0:
        return None
    return idx


def extract_history_text(prompt_text: str, sys_end_char: int) -> str | None:
    """Return text between end-of-system-block and first assistant prefix."""
    rest = prompt_text[sys_end_char:]
    first_a = first_assistant_offset_in_text(rest)
    if first_a is None:
        return None
    return rest[:first_a]


def find_sys_end_char(ctx_text: str) -> int:
    """Char offset just past the system block's terminating newline."""
    i = ctx_text.find("<|im_end|>")
    if i < 0:
        return -1
    return i + len("<|im_end|>\n")


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def score_nll_at_span(model, token_ids: list[int],
                      span: tuple[int, int], device) -> tuple[float, int]:
    """NLL on tokens in [span[0], span[1])."""
    s, e = span
    if e <= s:
        return 0.0, 0
    inp = torch.tensor([token_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model(input_ids=inp, use_cache=False)
        logits = out.logits[0, :-1, :].float()
        targets = inp[0, 1:]
        mask = torch.zeros(logits.shape[0], dtype=torch.bool, device=device)
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
# Condition constructors — produce token id list + assistant span to score
# ---------------------------------------------------------------------------
def build_condition_tokens(
    cond: str,
    *,
    tokenizer,
    ctx_tokens: list[int],
    ctx_text: str,
    sys_block_text: str,
    sys_end_tokens: int,
    sys_end_chars: int,
    first_span_with: tuple[int, int],   # in own ctx_tokens
    own_action_tokens: list[int],
    own_env_paragraph: str,
    partner_bullets: str | None,
    foreign_env_paragraph: str | None,
    foreign_forum_phrase: str | None,
    own_forum_phrase: str | None,
    foreign_history_text: str | None,
):
    """Return (token_ids, span_to_score) for the requested condition, or
    (None, None) if the condition cannot be built (e.g. swap_history_*
    when a foreign history isn't available)."""

    # --- character family --------------------------------------------------
    if cond == "own":
        return ctx_tokens, first_span_with

    if cond == "none_char":
        new_sys, ok = strip_bullets_payload(sys_block_text)
        if not ok:
            return None, None
        new_sys_tokens = tokenizer.encode(new_sys, add_special_tokens=False)
        new_tokens = new_sys_tokens + ctx_tokens[sys_end_tokens:]
        delta = len(new_sys_tokens) - sys_end_tokens
        return new_tokens, (first_span_with[0] + delta, first_span_with[1] + delta)

    if cond == "swap_char_within":
        if not partner_bullets:
            return None, None
        new_sys, ok = swap_bullets(sys_block_text, partner_bullets)
        if not ok:
            return None, None
        new_sys_tokens = tokenizer.encode(new_sys, add_special_tokens=False)
        new_tokens = new_sys_tokens + ctx_tokens[sys_end_tokens:]
        delta = len(new_sys_tokens) - sys_end_tokens
        return new_tokens, (first_span_with[0] + delta, first_span_with[1] + delta)

    # --- env family -------------------------------------------------------
    if cond in ("none_env", "swap_env"):
        if cond == "none_env":
            new_env = GENERIC_ENV
            forum_replacement = None  # keep original "You are X, a member of FORUM"
        else:
            if not foreign_env_paragraph:
                return None, None
            new_env = foreign_env_paragraph
            forum_replacement = foreign_forum_phrase
        # System block: replace env paragraph (and optionally forum phrase).
        new_sys, ok = replace_env_paragraph(sys_block_text, new_env)
        if not ok:
            return None, None
        if forum_replacement and own_forum_phrase:
            new_sys = new_sys.replace(
                f"a member of {own_forum_phrase}.",
                f"a member of {forum_replacement}.",
                1,
            )
        # The remainder of the prompt (after system) contains a duplicate
        # env paragraph at the start of the first user block. Replace it
        # there too.
        rest_text = ctx_text[sys_end_chars:]
        rest_new, _ = replace_env_in_user_block(rest_text, own_env_paragraph,
                                                new_env)
        new_full_text = new_sys + rest_new
        new_tokens = tokenizer.encode(new_full_text, add_special_tokens=False)
        # Locate the first assistant span in the new token sequence and
        # score the same character body, which is at the same string
        # location and now has the same length (we only altered env, not
        # bullets / assistant). To be robust we relocate via the prefix.
        prefix_ids, im_end_id = build_assistant_pattern(tokenizer)
        sp = first_assistant_span(new_tokens, prefix_ids, im_end_id)
        if sp is None:
            return None, None
        body_start, _body_end = sp
        return new_tokens, (body_start, body_start + len(own_action_tokens))

    # --- history family ---------------------------------------------------
    if cond == "none_history":
        # Drop the entire transcript: system + first assistant prefix +
        # A's action tokens.
        prefix_ids, im_end_id = build_assistant_pattern(tokenizer)
        sys_tokens = ctx_tokens[:sys_end_tokens]
        new_tokens = sys_tokens + prefix_ids + own_action_tokens + [im_end_id]
        body_start = len(sys_tokens) + len(prefix_ids)
        return new_tokens, (body_start, body_start + len(own_action_tokens))

    if cond in ("swap_history_within_run", "swap_history_cross_run"):
        if not foreign_history_text:
            return None, None
        prefix_ids, im_end_id = build_assistant_pattern(tokenizer)
        sys_tokens = ctx_tokens[:sys_end_tokens]
        new_text = ctx_text[:sys_end_chars] + foreign_history_text + \
                   "<|im_start|>assistant\n"
        head_tokens = tokenizer.encode(new_text, add_special_tokens=False)
        new_tokens = head_tokens + own_action_tokens + [im_end_id]
        body_start = len(head_tokens)
        return new_tokens, (body_start, body_start + len(own_action_tokens))

    raise ValueError(f"unknown condition {cond!r}")


# ---------------------------------------------------------------------------
# Foreign-history sourcing
# ---------------------------------------------------------------------------
class HistorySource:
    """Picks foreign histories (within-run + cross-run) per (run, iter, char).

    - Within-run: from the same trace's other episodes where the character
      appears.
    - Cross-run: from the parallel-seed runs' traces at the same iter.
    """

    def __init__(self, cross_run_roots: list[Path]):
        # Map env_name -> list of run dirs that share that env (potential
        # parallel-seed sources). The current run is excluded at query time.
        self.cross_run_roots = [Path(r) for r in cross_run_roots]
        self.env_to_runs: dict[str, list[Path]] = {}
        for r in self.cross_run_roots:
            env = env_name_from_run(r)
            self.env_to_runs.setdefault(env, []).append(r)
        # Trace caches by (run_dir, iter)
        self.cache: dict[tuple[str, int], list[dict]] = {}

    def _load_iter_traces(self, run_dir: Path, target_iter: int) -> list[dict]:
        key = (str(run_dir), target_iter)
        if key in self.cache:
            return self.cache[key]
        # Find exact iter, else nearest.
        iter_traces = list_iters_with_traces(run_dir)
        if not iter_traces:
            self.cache[key] = []
            return []
        # Exact match preferred
        exact = [p for (i, p) in iter_traces if i == target_iter]
        if exact:
            episodes = json.loads(Path(exact[0]).read_text())
        else:
            # Nearest iter
            nearest = min(iter_traces, key=lambda ip: abs(ip[0] - target_iter))
            episodes = json.loads(Path(nearest[1]).read_text())
        self.cache[key] = episodes
        return episodes

    def within_run(self, episodes: list[dict], own_ep_idx: int,
                   character: str) -> str | None:
        """Return foreign history from a *different* episode in the same
        traces.json that also contains `character`."""
        for j, ep in enumerate(episodes):
            if j == own_ep_idx:
                continue
            if character not in ep.get("agents", {}):
                continue
            ct = ep["agents"][character]["context_text"]
            sys_end = find_sys_end_char(ct)
            if sys_end < 0:
                continue
            hist = extract_history_text(ct, sys_end)
            if hist:
                return hist
        return None

    def cross_run(self, run_dir: Path, target_iter: int,
                  character: str) -> tuple[str | None, str | None]:
        """Return (history_text, source_run_name) from a parallel-seed run
        with the same env, at the closest available iter, containing
        `character`."""
        env = env_name_from_run(run_dir)
        candidates = [r for r in self.env_to_runs.get(env, [])
                      if r != run_dir]
        for other in candidates:
            episodes = self._load_iter_traces(other, target_iter)
            for ep in episodes:
                if character not in ep.get("agents", {}):
                    continue
                ct = ep["agents"][character]["context_text"]
                sys_end = find_sys_end_char(ct)
                if sys_end < 0:
                    continue
                hist = extract_history_text(ct, sys_end)
                if hist:
                    return hist, other.name
        return None, None


# ---------------------------------------------------------------------------
# Work building
# ---------------------------------------------------------------------------
def build_work_units(run_dirs, max_iters=None, include_iter_zero=False):
    units = []
    for rd in run_dirs:
        run_dir = Path(rd).resolve()
        iter_traces = list_iters_with_traces(run_dir)
        if not include_iter_zero:
            iter_traces = [(it, p) for it, p in iter_traces if it > 0]
        iter_traces = [(it, p) for it, p in iter_traces
                       if checkpoint_dir(run_dir, it).exists()]
        if max_iters is not None:
            iter_traces = iter_traces[:max_iters]
        for it, trace_path in iter_traces:
            ck = checkpoint_dir(run_dir, it)
            char_files = sorted(
                p for p in ck.iterdir()
                if p.suffix == ".pt" and p.stem != "meta"
            )
            for cp in char_files:
                units.append({"run_dir": str(run_dir), "iter": it,
                              "character": cp.stem,
                              "trace_path": str(trace_path),
                              "ckpt_path": str(cp)})
    return units


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
ALL_CONDITIONS = [
    "own", "none_char", "swap_char_within",
    "none_env", "swap_env",
    "none_history", "swap_history_within_run", "swap_history_cross_run",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", action="append", required=True)
    p.add_argument("--cross-run-roots", action="append", default=[],
                   help="Additional run dirs available as cross-run history "
                        "sources (typically the same set of inner-loop seeds).")
    p.add_argument("--substrate-asset-run-dirs", action="append", default=[],
                   help="Run dirs to source per-substrate env paragraphs from "
                        "for swap_env. Includes the union of all substrates "
                        "we might swap to.")
    p.add_argument("--conditions", default=",".join(ALL_CONDITIONS))
    p.add_argument("--output", required=True)
    p.add_argument("--max-iters", type=int, default=None)
    p.add_argument("--episodes-per-iter", type=int, default=None)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--attn-impl", default="sdpa")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output); out_dir.mkdir(parents=True, exist_ok=True)
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    log(f"conditions: {conditions}")

    units = build_work_units(args.run_dir)
    my_units = units[rank()::world_size()]
    log(f"total work units: {len(units)}, this rank: {len(my_units)}")
    if not my_units:
        return

    # Substrate env assets (for swap_env)
    asset_dirs = args.substrate_asset_run_dirs or args.run_dir
    substrate_assets: dict[str, dict] = {}
    for rd in asset_dirs:
        rdp = Path(rd).resolve()
        env = env_name_from_run(rdp)
        if env in substrate_assets:
            continue
        substrate_assets[env] = load_substrate_env_assets(rdp)
    log(f"substrate env assets: {list(substrate_assets.keys())}")

    # Model load
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    first_run = Path(my_units[0]["run_dir"])
    cfg = json.loads((first_run / "config.json").read_text())
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

    # Per-trace caches
    trace_cache: dict[str, list] = {}
    partner_payloads_cache: dict[str, dict[str, str]] = {}
    swap_map_cache: dict[str, dict[str, str]] = {}

    # Cross-run history source (uses any --cross-run-roots provided)
    hist_src = HistorySource(args.cross_run_roots)

    out_path = out_dir / f"pmi_decomp_rank{rank():02d}.jsonl"
    log(f"writing → {out_path}")
    f_out = out_path.open("w")
    t0 = time.time()

    for u_idx, u in enumerate(my_units):
        trace_path = u["trace_path"]
        if trace_path not in trace_cache:
            trace_cache.clear(); partner_payloads_cache.clear()
            swap_map_cache.clear()
            episodes = json.loads(Path(trace_path).read_text())
            trace_cache[trace_path] = episodes
            char_set = set()
            for ep in episodes:
                char_set.update(ep.get("agents", {}).keys())
            chars = sorted(char_set)
            swap_map_cache[trace_path] = build_swap_map(chars)
            partner_payloads_cache[trace_path] = \
                extract_partner_bullets_for_trace(episodes, chars)
        episodes = trace_cache[trace_path]
        swap_map = swap_map_cache[trace_path]
        partner_payloads = partner_payloads_cache[trace_path]

        load_adapter_weights(peft_model, Path(u["ckpt_path"]), target_adapter)
        peft_model.set_adapter(target_adapter)

        char = u["character"]
        run_dir_obj = Path(u["run_dir"])
        env_name = env_name_from_run(run_dir_obj)
        own_assets = substrate_assets.get(env_name, {})
        own_env_paragraph = own_assets.get("env_paragraph")
        own_forum_phrase = own_assets.get("forum_phrase")

        swap_env_name = ENV_SWAP_PARTNER.get(env_name)
        swap_env_assets = substrate_assets.get(swap_env_name, {}) if swap_env_name else {}
        foreign_env_para = swap_env_assets.get("env_paragraph")
        foreign_forum_phrase = swap_env_assets.get("forum_phrase")

        partner = swap_map.get(char)
        partner_payload = partner_payloads.get(partner) if partner else None

        scored = 0
        for ep_idx, ep in enumerate(episodes):
            agents = ep.get("agents", {})
            if char not in agents:
                continue
            if args.episodes_per_iter is not None and scored >= args.episodes_per_iter:
                break
            blob = agents[char]
            ctx_tokens = list(blob["context_tokens"])
            ctx_text = blob["context_text"]

            try:
                sys_end_tok = find_system_boundary(ctx_text, ctx_tokens, tokenizer)
            except Exception as e:
                log(f"  skip ep{ep.get('episode')} {char}: sys boundary {e}")
                continue
            sys_end_char = find_sys_end_char(ctx_text)
            sys_block_text = ctx_text[:sys_end_char]

            first_span = first_assistant_span(ctx_tokens, prefix_ids, im_end_id)
            if first_span is None:
                continue
            own_action_tokens = ctx_tokens[first_span[0]:first_span[1]]
            if not own_action_tokens:
                continue

            # Foreign histories
            hist_within = hist_src.within_run(episodes, ep_idx, char)
            hist_cross, cross_src = hist_src.cross_run(
                run_dir_obj, u["iter"], char,
            )

            row = {
                "run_dir": str(run_dir_obj),
                "iter": u["iter"],
                "character": char,
                "episode": ep.get("episode", ep_idx),
                "swap_partner_char": partner,
                "swap_env_substrate": swap_env_name,
                "swap_hist_within_run_available": hist_within is not None,
                "swap_hist_cross_run_source": cross_src,
                "n_action_tokens": len(own_action_tokens),
            }

            ok_any = False
            for cond in conditions:
                tokens, span = build_condition_tokens(
                    cond,
                    tokenizer=tokenizer,
                    ctx_tokens=ctx_tokens,
                    ctx_text=ctx_text,
                    sys_block_text=sys_block_text,
                    sys_end_tokens=sys_end_tok,
                    sys_end_chars=sys_end_char,
                    first_span_with=first_span,
                    own_action_tokens=own_action_tokens,
                    own_env_paragraph=own_env_paragraph,
                    partner_bullets=partner_payload,
                    foreign_env_paragraph=foreign_env_para,
                    foreign_forum_phrase=foreign_forum_phrase,
                    own_forum_phrase=own_forum_phrase,
                    foreign_history_text=hist_within if cond == "swap_history_within_run"
                                         else hist_cross if cond == "swap_history_cross_run"
                                         else None,
                )
                if tokens is None:
                    row[f"nll_{cond}"] = None
                    continue
                try:
                    nll_sum, n_tok = score_nll_at_span(peft_model, tokens, span, device)
                except torch.cuda.OutOfMemoryError:
                    log(f"  OOM ep{ep_idx} {char} cond={cond}; skip cond")
                    torch.cuda.empty_cache()
                    row[f"nll_{cond}"] = None
                    continue
                if n_tok == 0:
                    row[f"nll_{cond}"] = None
                    continue
                row[f"nll_{cond}"] = nll_sum / n_tok
                row[f"n_tok_{cond}"] = n_tok
                ok_any = True

            if ok_any:
                f_out.write(json.dumps(row) + "\n")
                f_out.flush()
                scored += 1

        if u_idx % 5 == 0:
            log(f"  done unit {u_idx+1}/{len(my_units)} "
                f"(iter={u['iter']} char={char} scored={scored}) "
                f"elapsed={time.time()-t0:.0f}s")

    f_out.close()
    log(f"finished in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
