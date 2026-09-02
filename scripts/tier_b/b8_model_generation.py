#!/usr/bin/env python3
"""
B8 — DOES ANY OF THIS DEPEND ON THE BASE MODEL?

WHAT THIS CHANGES
    Nothing. It puts numbers behind the answer to Reviewer pd7C, who objected
    that Qwen2.5-7B-Instruct is not a "pure pre-training" model and suggested
    Olmo-3. The rebuttal conceded the epistemics — the prior is to be specified,
    not eliminated — but offered no data. We have four Qwen3 runs on the
    campaign's reference substrate.

WHAT WE EXPECT TO LEARN
    Whether the phenomena the paper rests on — movement from base, the
    appearance of a high-drift stratum, perception improvement — are properties
    of the objective or of Qwen2.5-7B-Instruct. Compared at MATCHED ITERATIONS,
    because the Qwen3 runs are all truncated below iteration 125 and the
    Qwen2.5 runs are not (PROCESS.md: never compare runs at different iterations).

WHAT WOULD FALSIFY A "PROPERTY OF THE OBJECTIVE" READING
    If Qwen3 populations reach comparable KL from base at matched iterations and
    produce a comparable high-drift stratum, the phenomena are model-general at
    this scale.
    If they stay pinned near base with no high-drift agents, then "commitment" —
    the regime the paper's absorption result lives in — is not something the
    objective reliably produces, it is something this base model does under it,
    and every claim conditioned on that stratum inherits the dependence.

WHAT THIS CANNOT SETTLE
    Anything about a genuinely open-corpus base such as Olmo-3, which we have
    never run. Qwen3 is a different instruction-tuned model, not a different
    kind of prior. It bounds model-specificity within a family; it does not
    address the reviewer's point about instruction tuning itself.

OUTPUT  runs/_tier_b/b8_model_generation/
"""
from __future__ import annotations
import csv, json, os, statistics, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C

GRID = [25, 50, 75, 100]
SUBSTRATE = "study_group_ashbourne_gc"


def main():
    out = C.outdir("b8_model_generation")
    rows = []
    for rel in C.live_runs():
        recs = C.load_metrics(rel)
        if not recs or C.env_of(recs) != SUBSTRATE:
            continue
        cfgp = os.path.join(C.ROOT, rel, "config.json")
        if not os.path.exists(cfgp):
            continue
        cfg = json.load(open(cfgp))
        model = (cfg.get("model_name_or_path") or "").replace("Qwen/", "")
        n_agents = len(cfg.get("character_prompts") or {})
        # hold regularisation fixed at the campaign default
        if not (cfg.get("kl_coef") == 0.05 and cfg.get("beta") == 0.01):
            continue
        for t in GRID:
            pop = C.population_at(recs, t)
            if pop is None:
                continue
            kls = [a["kl"] for a in pop["agents"] if a["kl"] is not None]
            rows.append({
                "run": rel, "model": model, "n_agents": n_agents, "target": t,
                "iter": pop["iter"], "alive": round(pop["alive"], 3),
                "mean_kl": round(statistics.mean(kls), 4) if kls else None,
                "max_kl": round(max(kls), 4) if kls else None,
                "mean_perc": round(C.mean([a["perc_loss"] for a in pop["agents"]]), 4),
                "n_committed": pop["counts"].get("committed", 0),
                "n_high_drift": sum(1 for k in kls if k >= 1.0),
                "n": pop["n"]})

    with open(os.path.join(out, "by_model.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ["run"])
        w.writeheader(); w.writerows(rows)

    fams = {}
    for r in rows:
        fam = ("Qwen2.5-7B" if r["model"].startswith("Qwen2.5-7B") else
               "Qwen3-4B" if r["model"].startswith("Qwen3-4B") else
               "Qwen3-8B" if r["model"].startswith("Qwen3-8B") else r["model"])
        fams.setdefault((fam, r["target"]), []).append(r)

    print(f"substrate `{SUBSTRATE}`, kl_coef 0.05, beta 0.01, matched iterations\n")
    print(f"{'model':<14}{'iter':>5}{'runs':>6}{'mean KL':>10}{'max KL':>9}"
          f"{'mean perc':>11}{'alive':>7}{'agents KL>=1':>14}")
    summary = {}
    for (fam, t), rs in sorted(fams.items()):
        mk = statistics.mean(r["mean_kl"] for r in rs if r["mean_kl"] is not None)
        xk = max(r["max_kl"] for r in rs if r["max_kl"] is not None)
        mp = statistics.mean(r["mean_perc"] for r in rs)
        al = statistics.mean(r["alive"] for r in rs)
        hd = sum(r["n_high_drift"] for r in rs); tot = sum(r["n"] for r in rs)
        print(f"{fam:<14}{t:>5}{len(rs):>6}{mk:>10.3f}{xk:>9.3f}{mp:>11.3f}"
              f"{al:>7.2f}{f'{hd}/{tot}':>14}")
        summary[f"{fam}@{t}"] = {"runs": len(rs), "mean_kl": round(mk, 4),
                                 "max_kl": round(xk, 4), "mean_perc": round(mp, 4),
                                 "alive": round(al, 3), "high_drift": hd, "agents": tot}

    print()
    for t in GRID:
        a = summary.get(f"Qwen2.5-7B@{t}")
        for fam in ("Qwen3-4B", "Qwen3-8B"):
            b = summary.get(f"{fam}@{t}")
            if a and b and b["mean_kl"]:
                print(f"iter {t:>3}: Qwen2.5-7B mean KL is {a['mean_kl']/b['mean_kl']:.1f}x "
                      f"{fam}'s ({a['mean_kl']:.3f} vs {b['mean_kl']:.3f})")
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
