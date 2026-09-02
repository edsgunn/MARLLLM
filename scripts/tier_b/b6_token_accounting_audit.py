#!/usr/bin/env python3
"""
B6 — WHAT DOES `thinking_frac` ACTUALLY CONTAIN?

WHAT THIS CHANGES
    Nothing about the runs. It asks what the quantity we call "thinking tokens"
    is made of, before any claim rests on it.

    The framework's definition is deliberate, not accidental. The forum
    environment routes by tag, not by intent: `<post>` is broadcast, `<python>`
    is executed and its OUTPUT broadcast, and "everything else — private
    thinking; never shown to" other agents (envs/forum/env.py:290, 518, 689).
    `<think>` has NO special status here: it exists only so that a `<post>`
    nested inside it does not count (env.py:677). So
    `thinking_tokens = max(0, n_act - post_tokens)` (population.py:721) is the
    correct implementation of the environment's own notion of private, and the
    absence of `<think>` tags says nothing about whether the private channel is
    being used.

WHAT WE EXPECT TO LEARN
    Two things the name hides.

    (1) COMPOSITION. Non-post tokens are not one thing. They are `<python>`
        blocks — which are *public*, since pybot broadcasts their output — plus
        tag tokens, plus prose before the first `<post>`, plus anything trailing
        after the last `</post>`. On a tool-bearing substrate the first of these
        could dominate, in which case `thinking_frac` is largely measuring
        public tool use.

    (2) CONTENT. Where a genuine pre-post block exists, is it deliberation, or
        is it text the model meant as a post and failed to tag? The environment
        cannot tell these apart and neither can the metric; only reading it can.

WHAT WOULD FALSIFY A "PRIVATE DELIBERATION" READING OF `thinking_frac`
    If most non-post tokens are `<python>`, the metric is measuring the public
    channel and the name is misleading.
    If the pre-post block is degenerate — token loops, tag spam — it is neither
    deliberation nor an untagged post, and no intent reading is available.
    If it is coherent prose addressed to other members, the model meant to post
    and the tag failed, which is a substrate problem rather than a policy one.

OUTPUT  runs/_tier_b/b6_token_accounting/
"""
from __future__ import annotations
import csv, glob, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C

os.environ.setdefault("HF_HOME", "/lus/lfs1aip2/projects/a5l/egunn/hf_cache")

RUNS = [
    ("margin_notes 8a (tool)", "runs/cultural_emergence/run7_8agent_7B_margin_notes"),
    ("inner_loop 8a (tool)", "runs/cultural_emergence/run7_8agent_7B_inner_loop_post_format"),
    ("ashbourne_gc 16a", "runs/cultural_emergence/run8_16agent_7B_study_group_ashbourne_gc"),
    ("strathearn 16a", "runs/cultural_emergence/run8_16agent_7B_study_group_strathearn_server"),
    ("ashbourne_gc 32a", "runs/cultural_emergence/run12_32agent_7B_study_group_ashbourne_gc"),
]
DEGENERATE = re.compile(r"(</?\w*[>?]){6,}")   # six or more tags in a row


def generations(path):
    txt = open(path, errors="replace").read()
    out = []
    for block in txt.split("AGENT ")[1:]:
        name = block.split("\n", 1)[0].strip()
        for g in re.findall(r"<\|im_start\|>assistant\n(.*?)<\|im_end\|>", block, re.S):
            out.append((name, g))
    return out


def main():
    from transformers import AutoTokenizer
    out = C.outdir("b6_token_accounting")
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", local_files_only=True)
    n = lambda s: len(tok.encode(s, add_special_tokens=False)) if s else 0

    rows = []
    print(f"{'run':<26}{'ckpt':>6}{'gens':>6}{'total':>7}{'in post':>9}"
          f"{'python':>8}{'pre-post':>10}{'trailing':>10}{'degen%':>8}")
    for label, rel in RUNS:
        cks = sorted(glob.glob(os.path.join(C.ROOT, rel, "traces/ckpt_*/traces.txt")))
        if len(cks) < 2:
            continue
        for f in ([cks[1]] + [cks[-1]]):
            G = generations(f)
            if not G:
                continue
            acc = {k: 0 for k in ("total", "post", "python", "pre", "trail")}
            degen = 0
            for _, g in G:
                acc["total"] += n(g)
                acc["post"] += sum(n(p) for p in re.findall(r"<post>(.*?)</post>", g, re.S))
                acc["python"] += sum(n(p) for p in re.findall(r"<python>(.*?)</python>", g, re.S))
                pre = g.split("<post>")[0] if "<post>" in g else g
                acc["pre"] += n(pre)
                trail = g.rsplit("</post>", 1)[1] if "</post>" in g else ""
                acc["trail"] += n(trail)
                if DEGENERATE.search(pre) or DEGENERATE.search(trail):
                    degen += 1
            it = int(re.search(r"ckpt_(\d+)", f).group(1))
            m = {k: v / len(G) for k, v in acc.items()}
            dg = degen / len(G)
            print(f"{label:<26}{it:>6}{len(G):>6}{m['total']:>7.0f}{m['post']:>9.0f}"
                  f"{m['python']:>8.0f}{m['pre']:>10.0f}{m['trail']:>10.0f}{dg:>7.0%}")
            recs = C.load_metrics(rel)
            rec = C.record_at(recs, it) or {}
            env = C.env_of(recs)
            rows.append({"run": rel.split("/")[-1], "label": label, "ckpt": it,
                         "n_generations": len(G),
                         **{f"mean_{k}_tokens": round(v, 1) for k, v in m.items()},
                         "nonpost_that_is_python": round(m["python"] /
                             max(1e-9, m["total"] - m["post"]), 3),
                         "degenerate_generation_rate": round(dg, 3),
                         "thinking_frac_logged": rec.get(f"gen/{env}/thinking_frac")})
        print()

    with open(os.path.join(out, "nonpost_composition.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    early = [r for r in rows if r["ckpt"] <= 50]
    late = [r for r in rows if r["ckpt"] > 200]
    print("share of NON-POST tokens that is <python> (a PUBLIC channel — pybot "
          "broadcasts its output):")
    for tag, rs in (("early checkpoints", early), ("late checkpoints", late)):
        if rs:
            print(f"   {tag:<20}" + "  ".join(
                f"{r['label'].split()[0]} {r['nonpost_that_is_python']:.0%}" for r in rs))
    print("\ngenerations whose non-post region is degenerate tag/token loops:")
    for tag, rs in (("early checkpoints", early), ("late checkpoints", late)):
        if rs:
            print(f"   {tag:<20}" + "  ".join(
                f"{r['label'].split()[0]} {r['degenerate_generation_rate']:.0%}" for r in rs))
    # Dump a fair sample of late pre-post regions for reading. The composition
    # table cannot tell deliberation from an untagged post; only this can.
    import random
    random.seed(0)
    with open(os.path.join(out, "pre_post_excerpts.txt"), "w") as fh:
        for label, rel in RUNS:
            cks = sorted(glob.glob(os.path.join(C.ROOT, rel, "traces/ckpt_*/traces.txt")))
            if not cks:
                continue
            G = generations(cks[-1])
            cand = [(nm, g) for nm, g in G
                    if len((g.split("<post>")[0] if "<post>" in g else g).strip()) > 40]
            fh.write(f"\n{'='*78}\n{label} @ {cks[-1].split('ckpt_')[1][:6]} — "
                     f"{len(cand)}/{len(G)} generations with >40 chars before any <post>\n")
            for nm, g in random.sample(cand, min(6, len(cand))):
                pre = (g.split("<post>")[0] if "<post>" in g else g).strip()
                m = re.search(r"<post>(.*?)(?:</post>|$)", g, re.S)
                fh.write(f"\n--- {nm} | pre-post {len(pre)} chars | "
                         f"in-post: {'yes' if m else 'NONE'}\n{pre[:600]}\n")
    print(f"\nfair sample of late pre-post regions written to "
          f"{os.path.relpath(os.path.join(out, 'pre_post_excerpts.txt'), C.ROOT)}")

    json.dump(rows, open(os.path.join(out, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
