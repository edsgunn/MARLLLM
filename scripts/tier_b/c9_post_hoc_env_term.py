#!/usr/bin/env python3
"""
C9 POST HOC — RECOVER THE DROPPED ENVIRONMENT TERM FROM ARM A's CHECKPOINTS.

WHAT THIS CHANGES
    Arms A and B of the environment-likelihood experiment ran 300 iterations
    each and logged none of the quantities they existed to produce: no-tag mode
    delivered `generation.strip()` while the provenance matcher required the
    unstripped decode, so `env_log_probs` never reached the loss
    (runs/env_likelihood_bias/NOTES.md). The runtime measurement is gone.

    The raw material is not. Arm A's checkpoints hold all eight adapters at
    every 25 iterations, and its traces hold every episode's thread. This scores
    each delivered post under its GENERATOR's adapter and under each OBSERVER's,
    which is what the runtime path was supposed to do, and recovers the two
    decision-relevant numbers post hoc without retraining anything.

WHY NO-TAG MODE MAKES THIS EXACT
    With the post tags blanked, everything an agent generates is its post. There
    is no private prefix, so the correction I(z; y | u) that the derivation needs
    in general is identically zero here and p_env is exactly the generator's own
    log-probability of the delivered text. That is why arm A is the right run to
    do this on, and why no marginalisation over latents is required.

WHAT IS MEASURED, PER DELIVERED POST s BY GENERATOR j
    E_s = ln p_theta_j(y_s | context j held)      the environment log-likelihood
    M_s^i = ln p_theta_i(y_s | context i held)    observer i's surprise, negated

    then per observer i, over that episode's posts:
        G_t   = sum_{s>t} gamma^(s-t) (-M_s^i)        the implemented surrogate
        env_t = sum_{s>t} gamma^(s-t) E_s            the dropped term
        R_t   = G_t + env_t                          the stated objective

    env_share_of_return = mean |env_t| / (|G_t| + |env_t|)
    return_rank_corr    = Spearman(G_t, R_t) over action positions

    The decision rests on the second: if the two returns rank actions alike, the
    bias does not matter for learning however large the first is.

STRING LEVEL, NOT TOKEN LEVEL
    Scored per post as a total log-probability under each model's own
    tokenisation, following the derivation (section 6) and B5, which shows the
    generator's and the observer's token sequences differ for the same string.
    Per-token alignment is not attempted and is not needed: the returns are sums
    over posts.

PRE-REGISTERED PREDICTIONS, AND WHAT WOULD FALSIFY EACH
    P1  env_share_of_return is bounded by the share of an observer's context
        that is other agents' post content — ~32% at 8 agents (the config
        header's ceiling). FALSIFIED IF materially above that, which would mean
        the accounting is wrong rather than the term being large.
    P2  return_rank_corr is high early, when agents are near-identical copies of
        the base model and E_s tracks M_s closely. FALSIFIED IF low early, which
        would make the bias matter from the start.
    P3  return_rank_corr FALLS over training as adapters diverge, because E_s is
        the generator's log-prob and M_s the observer's, and those separate as
        the policies do. FALSIFIED IF flat or rising — the substitution would
        then be safe at exactly the point the paper cares about.
    P4  E_s > M_s^i on average: a post is less surprising to its author than to
        its readers. A violation is an estimator error, not a finding.

WHAT THIS CANNOT SETTLE
    It is a RECONSTRUCTION. Contexts are rebuilt by replaying each recorded
    thread through a ForumEnv with arm A's own environment spec and the same
    context formatter, which reproduces what the trainer built but is not the
    trainer's own buffer. And the recorded post text is the STRIPPED text the
    original run delivered, so the whitespace the generator actually emitted is
    not recoverable; E_s is the adapter's log-prob of the delivered string,
    scored fresh, rather than the fp8 sampling log-prob. Both are stated in the
    report.

    It measures the size and the direction-preservingness of the dropped term.
    It does not test the corrected OBJECTIVE — that is what the resubmitted arm
    B is for.

OUTPUT  runs/_tier_b/c9_post_hoc_env_term/
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

GAMMA = 0.99


def _spearman(a, b):
    """Spearman rho with average ranks for ties."""
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    if len(a) < 2:
        return float("nan")
    ra, rb = rank(a), rank(b)
    ma, mb = st.mean(ra), st.mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = math.sqrt(sum((x - ma) ** 2 for x in ra))
    db = math.sqrt(sum((y - mb) ** 2 for y in rb))
    return num / (da * db) if da > 0 and db > 0 else float("nan")


# ------------------------------------------------------------------ setup
def build_population(cfg: dict, device: str):
    """Eight LoRA adapters on one shared frozen backbone, as arm A had them."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from marlllm import LoRASharedBaseAgent

    names = list(cfg["characters"])
    tok = AutoTokenizer.from_pretrained(cfg["model"], local_files_only=True)
    base = AutoModelForCausalLM.from_pretrained(
        cfg["model"], dtype=torch.bfloat16, attn_implementation="sdpa",
        local_files_only=True).to(device)
    base.eval()
    lc = LoraConfig(
        r=int(cfg["lora_r"]), lora_alpha=int(cfg["lora_alpha"]),
        target_modules=[m.strip() for m in str(cfg["lora_modules"]).split(",")],
        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")
    peft = get_peft_model(base, lc, adapter_name=names[0])
    for n in names[1:]:
        peft.add_adapter(n, lc)
    prompts = cfg.get("character_prompts") or {}
    pop = {}
    for n in names:
        pr = prompts.get(n, f"You are {n}.")
        pop[n] = LoRASharedBaseAgent(
            agent_id=n, character_prompt=(pr[0] if isinstance(pr, list) else pr),
            shared_backbone=peft, adapter_name=n, tokenizer=tok, device=device,
            keep_ref_model=False)
    return pop, tok, peft


def build_env(cfg: dict, tok):
    """A ForumEnv on arm A's own spec, so contexts render as the run rendered
    them. Construction mirrors train_population.py's `_build_forum_env`; the
    persona strings in particular come from the environment's own
    `render_persona` over the character pack's memories, which is not
    reproducible by hand.
    """
    from envs.forum import load_characters, load_environment
    from envs.forum.env import ForumEnv
    spec = cfg["environments"][0]
    environment = load_environment(spec["environment"])
    characters = load_characters(spec["characters"])
    names = characters.get_character_set(spec.get("character_set", "canonical_4"))
    personas = {
        n: environment.render_persona(n, characters.get_memories(n)) for n in names
    }
    env = ForumEnv(
        agent_names=names, agent_personas=personas, tokenizer=tok,
        forum_description=spec.get("forum_description") or environment.forum_description,
        initial_invitation=spec.get("initial_invitation") or environment.default_invitation,
        post_length_note=spec.get("post_length_note"),
        post_open_tag=spec.get("post_open_tag", "<post>"),
        post_close_tag=spec.get("post_close_tag", "</post>"),
        provenance=bool(spec.get("provenance", False)),
        post_token_budget=spec.get("post_token_budget"),
        total_token_budget=spec.get("total_token_budget"),
        max_posts=int(spec.get("max_posts", 12)),
        post_order=spec.get("post_order", "round_robin"), seed=0)
    return env, names


# ------------------------------------------------------------------ scoring
def score_span(agent, ctx_ids: list[int], span_ids: list[int], device) -> float:
    """Total ln p_theta_agent(span | ctx), under this agent's own adapter.

    One forward over ctx+span; the log-prob of position i is read off the
    logits at i-1, so only the span's own positions contribute and the context
    is pure conditioning.
    """
    import torch
    ids = torch.tensor([ctx_ids + span_ids], device=device)
    am = torch.ones_like(ids)
    with torch.no_grad():
        logits, _ = agent.evaluate(ids, am)
        lp = torch.log_softmax(logits[0, :-1].float(), dim=-1)
        tgt = ids[0, 1:]
        tok_lp = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    start = len(ctx_ids) - 1          # first span token is predicted from here
    return float(tok_lp[start:].sum())


def replay_episode(env, names, thread, tokeniser, formatters):
    """Rebuild the episode turn by turn, following the trainer's own pattern.

    The trainer does NOT deliver each post to each observer as it happens. It
    calls ``env.last()`` for the agent whose turn it is, and only that agent —
    so an agent receives everything accumulated since its previous turn as one
    OBSERVATION BLOCK, then acts. With eight agents on a sixteen-post thread
    that is a block of several posts per turn, not one.

    That granularity is the measurement's unit, so it has to be reproduced:
    M is the observer's log-probability of the block it actually received, and
    E is the sum over the posts inside that block of each generator's own
    log-probability of its own post.

    Returns (turns, posts):
      turns  — per (observer, block): the block's token ids and the observer's
               context before it, plus which post indices the block contained.
      posts  — per post: the generator, its tokens, and its context before it.
    """
    order = []
    for post in thread:                    # the realised turn order
        sp = post.get("speaker")
        if sp in names and sp not in order:
            order.append(sp)
        if len(order) == len(names):
            break
    env.reset(seed=0, options={"order_seed": 0})
    if order and set(order) == set(env.agents):
        env.agents = list(order)           # impose the recorded round-robin

    ctx = {n: list(formatters[n].wrap_prompt(
        tokeniser.encode_prompt(env._personas[n]))) for n in names}
    turns, posts = [], []
    pending_posts = {n: [] for n in names}   # post indices seen since last turn
    by_index = {p.get("post_index"): p for p in thread}
    delivered = set()
    it = iter([p for p in thread if p.get("speaker") in names])

    while env.agents:
        role = env.agent_selection
        obs, _r, term, trunc, _i = env.last()
        obs_ids = tokeniser.encode_observation(obs)
        if obs_ids:
            turns.append(dict(observer=role, obs_ids=list(obs_ids),
                              ctx=list(ctx[role]),
                              contains=list(pending_posts[role])))
            ctx[role].extend(formatters[role].wrap_observation(obs_ids))
        pending_posts[role] = []
        if term or trunc:
            break
        try:
            post = next(it)
        except StopIteration:
            break
        act_ids = tokeniser.encode_prompt(post.get("text") or "")
        if act_ids:
            posts.append(dict(post_index=post.get("post_index"), speaker=role,
                              n_tokens=len(act_ids), act_ids=list(act_ids),
                              ctx=list(ctx[role])))
        env.step(act_ids)
        if act_ids:
            ctx[role].extend(formatters[role].wrap_action(act_ids))
        idx = post.get("post_index")
        if idx is not None and idx not in delivered:
            delivered.add(idx)
            for other in names:
                if other != role:
                    pending_posts[other].append(idx)
    return turns, posts


def measure_checkpoint(pop, env, names, tokeniser, formatters, episodes, device):
    """E and M per observation block, and the two return statistics."""
    all_blocks, all_posts, rows = [], [], []
    for thread in episodes:
        turns, posts = replay_episode(env, names, thread, tokeniser, formatters)
        # E per post, under its own generator's adapter. Exact in no-tag mode:
        # the whole generation is the post, so there is no private prefix.
        E = {}
        for r in posts:
            E[r["post_index"]] = score_span(
                pop[r["speaker"]], r["ctx"], r["act_ids"], device)
            all_posts.append(dict(post=r["post_index"], gen=r["speaker"],
                                  n_tok=r["n_tokens"], E=E[r["post_index"]]))
        # M per observation block, under the observer's adapter, plus the E of
        # the posts that block delivered.
        for t in turns:
            M = score_span(pop[t["observer"]], t["ctx"], t["obs_ids"], device)
            e = sum(E[i] for i in t["contains"] if i in E)
            all_blocks.append(dict(observer=t["observer"], M=M, E=e,
                                   n_tok=len(t["obs_ids"]),
                                   n_posts=len(t["contains"]),
                                   n_posts_scored=sum(1 for i in t["contains"] if i in E)))
        # Returns per observer, within this episode only.
        for i in names:
            seq = [(b["E"], b["M"]) for b in all_blocks[-len(turns):]
                   if b["observer"] == i]
            if len(seq) < 2:
                continue
            G, env_r = [], []
            for t_ in range(len(seq)):
                G.append(sum(GAMMA ** (s - t_) * (-seq[s][1])
                             for s in range(t_ + 1, len(seq))))
                env_r.append(sum(GAMMA ** (s - t_) * seq[s][0]
                                 for s in range(t_ + 1, len(seq))))
            rows.append(dict(observer=i, G=G, env=env_r,
                             R=[g + e for g, e in zip(G, env_r)]))
    return all_blocks, all_posts, rows


def summarise(blocks, posts, rows):
    Es = [p["E"] for p in posts]
    Ms = [b["M"] for b in blocks]
    per_tok_E = [p["E"] / p["n_tok"] for p in posts if p["n_tok"]]
    per_tok_M = [b["M"] / b["n_tok"] for b in blocks if b["n_tok"]]
    shares, corrs = [], []
    for r in rows:
        for g, e in zip(r["G"], r["env"]):
            d = abs(g) + abs(e)
            if d > 0:
                shares.append(abs(e) / d)
        c = _spearman(r["G"], r["R"])
        if not math.isnan(c):
            corrs.append(c)
    # Fidelity: blocks whose posts were not all scorable (empty generations are
    # skipped, so a block can deliver a post with no E).
    tot_p = sum(b["n_posts"] for b in blocks)
    got_p = sum(b["n_posts_scored"] for b in blocks)
    return dict(
        n_posts=len(posts), n_blocks=len(blocks), n_observer_streams=len(rows),
        posts_scored_frac=(got_p / tot_p) if tot_p else None,
        E_mean_per_post=st.mean(Es) if Es else None,
        M_mean_per_block=st.mean(Ms) if Ms else None,
        E_per_token=st.mean(per_tok_E) if per_tok_E else None,
        M_per_token=st.mean(per_tok_M) if per_tok_M else None,
        env_share_of_return=st.mean(shares) if shares else None,
        env_share_median=st.median(shares) if shares else None,
        return_rank_corr=st.mean(corrs) if corrs else None,
        return_rank_corr_min=min(corrs) if corrs else None,
        # P4: a post should be less surprising per token to its author than the
        # block is to its reader.
        E_minus_M_per_token=((st.mean(per_tok_E) - st.mean(per_tok_M))
                             if per_tok_E and per_tok_M else None),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/env_likelihood_bias/A_surrogate_seed1")
    ap.add_argument("--config", default="configs/env_likelihood_bias/A_surrogate_seed1.yaml")
    ap.add_argument("--checkpoints", default="25,100,200,300")
    ap.add_argument("--max-episodes", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="runs/_tier_b/c9_post_hoc_env_term")
    a = ap.parse_args()

    import yaml
    from marlllm.checkpoint_utils import load_population_checkpoint
    from marlllm.context_formatter import make_formatter

    cfg = yaml.safe_load(Path(a.config).read_text())
    run = ROOT / a.run
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)

    pop, tok, _ = build_population(cfg, a.device)
    env, names = build_env(cfg, tok)
    from marlllm.tokeniser import TextTokeniser
    tokeniser = TextTokeniser(tok)
    # Each agent's own formatter, as the trainer uses.
    formatters = {n: pop[n].context_formatter for n in names}
    print(f"population {len(names)} | no-tag {env._no_tag_mode} | "
          f"formatter {formatters[names[0]]}")

    results = {}
    for it in [int(x) for x in a.checkpoints.split(",")]:
        ck = run / "checkpoints" / f"iter_{it:06d}"
        tr = run / "traces" / f"ckpt_{it:06d}" / "traces.json"
        if not ck.exists() or not tr.exists():
            print(f"  iter {it}: missing checkpoint or traces, skipped")
            continue
        load_population_checkpoint(population=pop, optimizer=None, ckpt_dir=ck,
                                   device=a.device, load_optimizer=False)
        d = json.loads(tr.read_text())
        recs = d if isinstance(d, list) else [d]
        episodes = [(r.get("env_trace") or {}).get("thread") or []
                    for r in recs][:a.max_episodes]
        episodes = [e for e in episodes if e]
        blocks, pp, rows = measure_checkpoint(
            pop, env, names, tokeniser, formatters, episodes, a.device)
        s = summarise(blocks, pp, rows)
        results[it] = s
        print(f"  iter {it:>3}: posts {s['n_posts']:>3} blocks {s['n_blocks']:>3}  "
              f"E/tok {s['E_per_token']:+.3f}  M/tok {s['M_per_token']:+.3f}  "
              f"env_share {s['env_share_of_return']:.3f}  "
              f"rank_corr {s['return_rank_corr']:+.3f}  "
              f"scored {s['posts_scored_frac']:.2f}")
        (out / "summary.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out/'summary.json'}")


if __name__ == "__main__":
    main()
