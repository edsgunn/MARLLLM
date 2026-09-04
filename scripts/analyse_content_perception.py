#!/usr/bin/env python3
"""
PERCEPTION LOSS ON POST CONTENT ONLY, ACROSS CHECKPOINTS.

WHAT THIS CHANGES
    `perc_loss` averages over every observation token, and most of an
    observation is framing the environment writes deterministically ("Sam
    Pritchard wrote:", "Several people posted:"). So when a population stops
    posting, the stream it observes becomes mostly that framing, the loss falls
    for free, and the regime classifier — whose primary gate is
    `perc_loss < 1.7` — reads the silence as coherence. At iteration 900 of the
    extended 8-agent reference, five well-formed posts in sixty-four turns
    scored `alive` 0.62, with four of five "alive" agents posting nothing
    (runs/cultural_emergence/c6_extended_horizon/NOTES.md).

    This splits the loss. Content tokens are the ones another agent actually
    generated; scaffolding is everything the environment wrote. A health
    measure built on the first cannot be gamed by silence, because silence
    removes the tokens rather than making them easy.

WHY IT NEEDS NO TRAINING
    The adapters and the traces are on disk. Contexts are rebuilt by replaying
    each recorded thread through a ForumEnv on the run's own spec, and each
    observation block is scored under the observing agent's adapter. One
    forward per block.

HOW CONTENT IS LOCATED
    NOT via `observation_segments`: that labels a span "post" only when
    provenance resolved, and provenance does not resolve in TAGGED mode at all
    (the delivered text is the content between the tags while the action tokens
    include them), so on every tagged run it returns pure scaffolding. Instead
    the block text is tokenised with offset mapping and each post's text is
    located in it by character span, which is exact and needs no provenance.

PRE-REGISTERED PREDICTIONS, AND WHAT WOULD FALSIFY EACH
    P1  content NLL is HIGHER than scaffolding NLL at every checkpoint — other
        agents' words are less predictable than a fixed template. FALSIFIED IF
        not, which would mean the split is mislocated rather than meaningful.
    P2  the all-token mean tracks the scaffolding as delivery falls, because
        scaffolding comes to dominate the token count. That is the artefact,
        stated as a prediction.
    P3  content NLL does NOT fall with delivery the way the all-token mean
        does. FALSIFIED IF it falls in step, which would mean the agents really
        are getting better at predicting each other and the artefact reading is
        wrong.

WHAT THIS CANNOT SETTLE
    It is a reconstruction: contexts are rebuilt rather than replayed from the
    trainer's own buffers, and the recorded post text is what the environment
    delivered rather than what the generator emitted. Both are stated in the
    output. It measures the metric, not the populations.

Usage
    .venv/bin/python scripts/analyse_content_perception.py \
        --spec run7:configs/.../run7_....yaml:25,100,200,300 \
        --spec c6:configs/.../c6_extend_8agent....yaml:400,600,900 \
        --device cuda:0 --out runs/_tier_b/content_perception
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "tier_b"))

from c9_post_hoc_env_term import build_population, build_env  # noqa: E402


def replay_with_text(env, names, thread, tokeniser, formatters):
    """Like c9's replay, but keeps each observation block's ids and the
    observer's context, so the block can be scored and split."""
    order = []
    for p in thread:
        sp = p.get("speaker")
        if sp in names and sp not in order:
            order.append(sp)
        if len(order) == len(names):
            break
    env.reset(seed=0, options={"order_seed": 0})
    if order and set(order) == set(env.agents):
        env.agents = list(order)

    po, pc = env._post_open_tag, env._post_close_tag
    ctx = {n: list(formatters[n].wrap_prompt(
        tokeniser.encode_prompt(env._personas[n]))) for n in names}
    blocks = []
    it = iter([p for p in thread if p.get("speaker") in names])
    while env.agents:
        role = env.agent_selection
        obs, _r, term, trunc, _i = env.last()
        obs_ids = tokeniser.encode_observation(obs)
        if obs_ids:
            blocks.append(dict(observer=role,
                               ids=list(obs_ids), ctx=list(ctx[role])))
            ctx[role].extend(formatters[role].wrap_observation(obs_ids))
        if term or trunc:
            break
        try:
            post = next(it)
        except StopIteration:
            break
        txt = post.get("text") or ""
        # Re-wrap for tagged substrates: the trace stores the EXTRACTED text,
        # so feeding it back bare would produce no post block at all.
        payload = (f"{po}{txt}{pc}" if (po and pc and txt.strip()) else txt)
        act_ids = tokeniser.encode_prompt(payload)
        env.step(act_ids)
        if act_ids:
            ctx[role].extend(formatters[role].wrap_action(act_ids))
    return blocks


def content_mask(tok, ids, post_texts):
    """True where a token lies inside a post's content.

    Works in TOKEN space, because ForumEnv's observation is a list of ids
    rather than a string. Each id is decoded individually and the pieces
    concatenate to the whole decode exactly (verified on Qwen byte-BPE), which
    gives every token an exact character span without re-tokenising anything.

    Not via `observation_segments`: that labels a span "post" only when
    provenance resolved, and provenance does not resolve in tagged mode at all,
    so on every tagged run it would report pure scaffolding.
    """
    pieces = tok.batch_decode([[i] for i in ids])
    spans, pos = [], 0
    for pc in pieces:
        spans.append((pos, pos + len(pc)))
        pos += len(pc)
    text = "".join(pieces)
    hits, cursor = [], 0
    for t in post_texts:
        t = (t or "").strip()
        if not t:
            continue
        i = text.find(t, cursor)
        if i < 0:                      # truncated by the post budget, or reframed
            continue
        hits.append((i, i + len(t)))
        cursor = i + len(t)
    mask = [any(a < e and b > st_ for st_, e in hits) for (a, b) in spans]
    return mask, len(ids)


def score_block(agent, ctx_ids, block_ids, device):
    """Per-token log-probs of the block, conditioned on the observer's context."""
    import torch
    ids = torch.tensor([ctx_ids + block_ids], device=device)
    am = torch.ones_like(ids)
    with torch.no_grad():
        logits, _ = agent.evaluate(ids, am)
        lp = torch.log_softmax(logits[0, :-1].float(), dim=-1)
        tgt = ids[0, 1:]
        tok_lp = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    start = len(ctx_ids) - 1
    return tok_lp[start:].tolist()


def measure(pop, env, names, tokeniser, formatters, tok, episodes, device):
    content, scaffold = [], []
    for thread in episodes:
        posts = [p.get("text") or "" for p in thread if p.get("speaker") in names]
        blocks = replay_with_text(env, names, thread, tokeniser, formatters)
        for b in blocks:
            lps = score_block(pop[b["observer"]], b["ctx"], b["ids"], device)
            mask, n = content_mask(tok, b["ids"], posts)
            for i in range(min(n, len(lps))):
                (content if mask[i] else scaffold).append(-lps[i])
    return content, scaffold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", action="append", required=True,
                    help="label:config_path:iters  (iters comma-separated)")
    ap.add_argument("--max-episodes", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="runs/_tier_b/content_perception")
    a = ap.parse_args()

    import yaml
    from marlllm.checkpoint_utils import load_population_checkpoint
    from marlllm.tokeniser import TextTokeniser

    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    pop = env = names = tok = None
    for spec in a.spec:
        label, cfg_path, iters = spec.split(":")
        cfg = yaml.safe_load((ROOT / cfg_path).read_text())
        if pop is None:                      # one backbone for all specs
            pop, tok, _ = build_population(cfg, a.device)
            env, names = build_env(cfg, tok)
            tokeniser = TextTokeniser(tok)
            formatters = {n: pop[n].context_formatter for n in names}
        run = ROOT / cfg["output_dir"]
        for it in [int(x) for x in iters.split(",")]:
            ck = run / "checkpoints" / f"iter_{it:06d}"
            tr = run / "traces" / f"ckpt_{it:06d}" / "traces.json"
            if not ck.exists() or not tr.exists():
                print(f"  {label} iter {it}: missing, skipped")
                continue
            load_population_checkpoint(population=pop, optimizer=None,
                                       ckpt_dir=ck, device=a.device,
                                       load_optimizer=False)
            d = json.loads(tr.read_text())
            recs = d if isinstance(d, list) else [d]
            eps = [(r.get("env_trace") or {}).get("thread") or [] for r in recs]
            eps = [e for e in eps if e][:a.max_episodes]
            # delivery, for the comparison this exists to make
            succ = turns = 0
            for r in recs:
                for _a2, c in ((r.get("env_trace") or {}).get(
                        "tool_counts_per_agent") or {}).items():
                    succ += c.get("post_success", 0)
                    turns += c.get("post_turns", 0)
            con, sca = measure(pop, env, names, tokeniser, formatters, tok,
                               eps, a.device)
            allt = con + sca
            row = dict(
                label=label, iteration=it,
                delivery=(succ / turns if turns else 0.0),
                n_content=len(con), n_scaffold=len(sca),
                content_frac=(len(con) / len(allt) if allt else 0.0),
                nll_content=(st.mean(con) if con else None),
                nll_scaffold=(st.mean(sca) if sca else None),
                nll_all=(st.mean(allt) if allt else None))
            results[f"{label}@{it}"] = row
            print(f"  {label:<6} iter {it:>3}: delivery {row['delivery']:.3f}  "
                  f"content {row['content_frac']*100:>4.1f}% of tokens  "
                  f"NLL content {row['nll_content'] or float('nan'):.3f}  "
                  f"scaffold {row['nll_scaffold'] or float('nan'):.3f}  "
                  f"all {row['nll_all']:.3f}")
            (out / "summary.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out/'summary.json'}")


if __name__ == "__main__":
    main()
