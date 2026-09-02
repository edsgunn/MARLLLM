#!/usr/bin/env python3
"""
B5a — CAN WE RECOVER p_env FOR AGENT-PRODUCED OBSERVATION TOKENS?

WHAT THIS CHANGES
    Tests the premise of REWRITE.md §5, which is in turn the premise of the
    paper's whole "we must drop the environment term" argument. Appendix A drops
    Sum_s sigma_s ln p_env(x_s | x_<s) because "p_env is only ever sampled, not
    evaluated". In a multi-agent LLM population that is only true of some
    observation tokens. An agent's observation stream is:
        persona prefix        deterministic -> ln p_env = 0
        structural markers    deterministic -> ln p_env = 0
        tool output           deterministic given the code -> ln p_env = 0
        OTHER AGENTS' POSTS   sampled from another LoRA adapter over the same
                              frozen base, whose per-token log-probs we compute
                              and store (RolloutBatch.act_log_probs_old)
    If the third case is token-identifiable, the dropped term is recoverable for
    the only part of the observation stream that carries any entropy.

WHAT WE EXPECT TO LEARN
    Whether agent j's posted tokens appear in agent i's context as the SAME
    token sequence j sampled. Two checks:
        (1) receiver-side  — does j's post appear as a contiguous token span in
                             i's context?
        (2) producer-side  — is that span token-identical to the span in j's own
                             context, i.e. to what j actually sampled?
    (1) alone gives the bias MEASUREMENT (B5b). (2) is required for the stronger
    proposal of restoring the term per-token, because a per-token log-prob can
    only be attached to a token sequence that matches.

WHAT WOULD FALSIFY THE PROPOSAL
    If (1) fails, agent provenance cannot be resolved to token spans from logged
    artefacts and the bias measurement needs new instrumentation.
    If (1) holds but (2) fails, the environment re-encodes text between producer
    and receiver, the two token sequences differ, and restoring p_env per-token
    requires either aligning the tokenisations or scoring j's adapter on the
    RECEIVER's tokenisation — a real cost, not a free lunch. The bias measurement
    would survive; the restored-return arm would get more expensive.

WHAT THIS DOES NOT DO
    It does not measure the covariance. That is B5b and it needs a GPU scoring
    pass over checkpoints, because no logged artefact carries per-token
    log-probabilities — the trace JSON holds only `context_tokens` and
    `context_text` per agent. B5b is therefore Tier C, not Tier B.

OUTPUT  runs/_tier_b/b5_env_likelihood/
"""
from __future__ import annotations
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C

os.environ.setdefault("HF_HOME", "/lus/lfs1aip2/projects/a5l/egunn/hf_cache")

TRACES = [
    ("runs/cultural_emergence/run8_16agent_7B_study_group_ashbourne_gc", "iter_000300.json"),
    ("runs/cultural_emergence/run8_16agent_7B_study_group_ashbourne_gc", "iter_000025.json"),
    ("runs/cultural_emergence/run7_8agent_7B_inner_loop/", "iter_000100.json"),
]


def find_span(hay, needle):
    """Index of contiguous `needle` in `hay`, or -1. Both lists of ints."""
    n, m = len(hay), len(needle)
    if m == 0 or m > n:
        return -1
    first = needle[0]
    for i in range(n - m + 1):
        if hay[i] == first and hay[i:i + m] == needle:
            return i
    return -1


def main():
    from transformers import AutoTokenizer
    out = C.outdir("b5_env_likelihood")
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", local_files_only=True)
    TRIM = 2   # drop this many tokens at each end before matching: BPE re-tokenises
               # at a span boundary depending on the surrounding text, so the edge
               # tokens differ even when the content is identical.

    results = []
    for rel, fname in TRACES:
        p = os.path.join(C.ROOT, rel, "traces", fname)
        if not os.path.exists(p):
            print(f"  (missing) {p}")
            continue
        ep = json.load(open(p))[0]
        thread = ep["env_trace"].get("thread", [])
        agents = ep["agents"]
        ctx = {a: v["context_tokens"] for a, v in agents.items()}
        txt = {a: v["context_text"] for a, v in agents.items()}

        n = dict(posts=0, empty=0, prod_str=0, prod_tok=0, prod_n=0,
                 recv_str=0, recv_tok=0, recv_n=0)
        for post in thread:
            spk, text = post.get("speaker"), (post.get("text") or "")
            n["posts"] += 1
            if not text.strip():
                n["empty"] += 1
                continue
            ids = tok.encode(text, add_special_tokens=False)
            interior = ids[TRIM:-TRIM] if len(ids) > 2 * TRIM + 4 else ids
            if spk in ctx:
                n["prod_n"] += 1
                n["prod_str"] += text in txt.get(spk, "")
                n["prod_tok"] += find_span(ctx[spk], interior) >= 0
            for other in ctx:
                if other == spk:
                    continue
                n["recv_n"] += 1
                n["recv_str"] += text in txt.get(other, "")
                n["recv_tok"] += find_span(ctx[other], interior) >= 0

        row = {"run": rel.rstrip("/").split("/")[-1], "trace": fname,
               "posts": n["posts"], "empty_posts": n["empty"],
               "producer_string_rate": round(n["prod_str"] / n["prod_n"], 4) if n["prod_n"] else None,
               "producer_token_rate": round(n["prod_tok"] / n["prod_n"], 4) if n["prod_n"] else None,
               "receiver_string_rate": round(n["recv_str"] / n["recv_n"], 4) if n["recv_n"] else None,
               "receiver_token_rate": round(n["recv_tok"] / n["recv_n"], 4) if n["recv_n"] else None,
               "n_producer": n["prod_n"], "n_receiver": n["recv_n"]}
        results.append(row)
        print(f"{row['run'][:42]:<44}{fname:<18}posts {row['posts']:>3} "
              f"empty {row['empty_posts']:>2} | producer str {row['producer_string_rate']} "
              f"tok {row['producer_token_rate']} | receiver str {row['receiver_string_rate']} "
              f"tok {row['receiver_token_rate']}")

    json.dump(results, open(os.path.join(out, "recoverability.json"), "w"), indent=1)
    if not results:
        return
    rs = [r["receiver_string_rate"] for r in results]
    rt = [r["receiver_token_rate"] for r in results]
    ps = [r["producer_string_rate"] for r in results]
    pt = [r["producer_token_rate"] for r in results]
    print()
    print(f"receiver: post text present as a string in {min(rs):.1%}-{max(rs):.1%} of "
          f"(post, other-agent) pairs;\n          as a contiguous interior token span in "
          f"{min(rt):.1%}-{max(rt):.1%}")
    print(f"producer: string {min(ps):.1%}-{max(ps):.1%}; interior token span "
          f"{min(pt):.1%}-{max(pt):.1%}")
    print()
    if min(rt) > 0.95 and min(pt) > 0.95:
        print("VERDICT: agent-produced observation tokens are token-identifiable on both\n"
              "         sides. p_env is recoverable per token from an agent's own stored\n"
              "         action log-probs, and the dropped term is measurable in-training.")
    elif min(rs) > 0.95:
        print("VERDICT: the content is preserved verbatim end to end, so provenance is\n"
              "         resolvable and the BIAS MEASUREMENT (B5b) is well posed. Whether it\n"
              "         resolves to identical TOKENS is the number above; where it does not,\n"
              "         restoring p_env per token requires scoring the producer's adapter on\n"
              "         the receiver's tokenisation rather than reusing stored log-probs.")
    else:
        print("VERDICT: the post text is not reliably present in other agents' contexts.\n"
              "         Recovering p_env from logged artefacts is not possible.")


if __name__ == "__main__":
    main()
