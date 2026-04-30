"""
Aggregate diagnostic outputs across cells × seeds into the report tables and
figures referenced in the spec (§8).

Reads `<output_dir>/diagnostics/summary.json` and produces:
  - tables.md    : Tables 1-3 (headline performance, mimicry, grounding).
  - summary.json : programmatic aggregate.
  - fig_generalisation.{png,pdf} : OOD-vs-ID scatter (Figure 2).
  - fig_hull.{png,pdf}            : Hull novelty vs deal_rate (Figure 3).

Hypothesis verdicts (H1..H4) are reported with seed-level mean/std.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev


def _agg(values: list[float]) -> dict:
    if not values:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {"mean": mean(values), "std": pstdev(values) if len(values) > 1 else 0.0,
            "n": len(values)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--diagnostics-summary", required=True,
                   help="Path to <output_dir>/diagnostics/summary.json.")
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    summary = json.loads(Path(args.diagnostics_summary).read_text())
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Aggregate across seeds ────────────────────────────────────────────────
    table1: dict[str, dict] = {}   # task outcomes
    table2: dict[str, dict] = {}   # mimicry
    table3: dict[str, dict] = {}   # grounding
    hull_per_cell: dict[str, dict] = {}

    for cell, seeds in summary.items():
        deal_id, deal_ad, deal_weak = [], [], []
        ngram3, embed_cos, kl = [], [], []
        env_r2, phase_acc = [], []
        hull_frac = []
        for seed, data in seeds.items():
            ood_data = data.get("ood", {}) or {}
            if "in_distribution" in ood_data:
                deal_id.append(ood_data["in_distribution"].get("deal_rate", math.nan))
            if "adversarial" in ood_data:
                deal_ad.append(ood_data["adversarial"].get("deal_rate", math.nan))
            if "weak_partner" in ood_data:
                deal_weak.append(ood_data["weak_partner"].get("deal_rate", math.nan))

            mim = data.get("mimicry", {}) or {}
            if "ngram_jaccard_3" in mim:
                ngram3.append(mim["ngram_jaccard_3"])
            if "embed_mean_max_cos" in mim:
                embed_cos.append(mim["embed_mean_max_cos"])
            if "move_distribution_kl" in mim:
                kl.append(mim["move_distribution_kl"])

            pr = data.get("probing", {}) or {}
            if "env_state_r2" in pr:
                env_r2.append(pr["env_state_r2"])
            if "phase_accuracy" in pr:
                phase_acc.append(pr["phase_accuracy"])

            h = data.get("hull", {}) or {}
            if "fraction_out_of_hull" in h:
                hull_frac.append(h["fraction_out_of_hull"])

        table1[cell] = {
            "deal_rate_id": _agg(deal_id),
            "deal_rate_adversarial": _agg(deal_ad),
            "deal_rate_weak_partner": _agg(deal_weak),
        }
        table2[cell] = {
            "ngram_jaccard_3": _agg(ngram3),
            "embed_mean_max_cos": _agg(embed_cos),
            "move_distribution_kl": _agg(kl),
        }
        table3[cell] = {
            "env_state_r2": _agg(env_r2),
            "phase_accuracy": _agg(phase_acc),
        }
        hull_per_cell[cell] = _agg(hull_frac)

    # ── Hypothesis tests (descriptive) ────────────────────────────────────────
    def _m(table, cell, key):
        return table.get(cell, {}).get(key, {}).get("mean", math.nan)

    hypotheses = {
        "H1_distillation_null": {
            "description": "Cell C ≈ Cell A on in-distribution deal rate.",
            "C_id": _m(table1, "C", "deal_rate_id"),
            "A_id": _m(table1, "A", "deal_rate_id"),
            "delta": _m(table1, "C", "deal_rate_id") - _m(table1, "A", "deal_rate_id"),
        },
        "H2_stabilisation": {
            "description": "Cells C and D outperform Cell C0 on in-distribution.",
            "C_id":  _m(table1, "C",  "deal_rate_id"),
            "D_id":  _m(table1, "D",  "deal_rate_id"),
            "C0_id": _m(table1, "C0", "deal_rate_id"),
        },
        "H3_grounding": {
            "description": "Cell C beats Cell A on OOD and on env_state_r2.",
            "C_ad": _m(table1, "C", "deal_rate_adversarial"),
            "A_ad": _m(table1, "A", "deal_rate_adversarial"),
            "C_env_r2": _m(table3, "C", "env_state_r2"),
            "A_env_r2": _m(table3, "A", "env_state_r2"),
        },
        "H4_rl_beyond_imitation": {
            "description": "Cell C > Cell B on task; more out-of-hull utterances.",
            "C_id": _m(table1, "C", "deal_rate_id"),
            "B_id": _m(table1, "B", "deal_rate_id"),
            "C_hull": hull_per_cell.get("C", {}).get("mean", math.nan),
            "B_hull": hull_per_cell.get("B", {}).get("mean", math.nan),
        },
    }

    aggregate = {
        "table1_performance": table1,
        "table2_mimicry": table2,
        "table3_grounding": table3,
        "hull_per_cell": hull_per_cell,
        "hypotheses": hypotheses,
    }
    (out / "summary.json").write_text(json.dumps(aggregate, indent=2))

    # ── Markdown report ───────────────────────────────────────────────────────
    def fmt(d: dict) -> str:
        if "mean" not in d:
            return "—"
        if math.isnan(d.get("mean", math.nan)):
            return "—"
        return f"{d['mean']:.3f} ± {d['std']:.3f} (n={d['n']})"

    cells = sorted(table1)
    md_lines: list[str] = []
    md_lines.append("# Distillation ablation — diagnostic report\n")

    md_lines.append("## Table 1 — Headline performance (deal rate)\n")
    md_lines.append("| Cell | In-dist | Adversarial | Weak partner |")
    md_lines.append("|------|---------|-------------|--------------|")
    for c in cells:
        t = table1[c]
        md_lines.append(f"| {c} | {fmt(t['deal_rate_id'])} | "
                        f"{fmt(t['deal_rate_adversarial'])} | "
                        f"{fmt(t['deal_rate_weak_partner'])} |")
    md_lines.append("")

    md_lines.append("## Table 2 — Mimicry signatures\n")
    md_lines.append("| Cell | n-gram J@3 | Embed cos | Move-KL |")
    md_lines.append("|------|-----------|-----------|---------|")
    for c in cells:
        t = table2[c]
        md_lines.append(f"| {c} | {fmt(t['ngram_jaccard_3'])} | "
                        f"{fmt(t['embed_mean_max_cos'])} | "
                        f"{fmt(t['move_distribution_kl'])} |")
    md_lines.append("")

    md_lines.append("## Table 3 — Grounding signatures\n")
    md_lines.append("| Cell | env_state R² | phase acc | hull novelty |")
    md_lines.append("|------|--------------|-----------|--------------|")
    for c in cells:
        t = table3[c]
        md_lines.append(f"| {c} | {fmt(t['env_state_r2'])} | "
                        f"{fmt(t['phase_accuracy'])} | "
                        f"{fmt(hull_per_cell.get(c, {}))} |")
    md_lines.append("")

    md_lines.append("## Hypothesis snapshot\n")
    for h_name, h in hypotheses.items():
        md_lines.append(f"### {h_name}")
        md_lines.append(f"_{h['description']}_")
        for k, v in h.items():
            if k == "description":
                continue
            md_lines.append(f"- **{k}**: {v}")
        md_lines.append("")

    (out / "tables.md").write_text("\n".join(md_lines))

    # ── Figures (matplotlib if available) ────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5, 4))
        for c in cells:
            id_m = table1[c]["deal_rate_id"]["mean"]
            ad_m = table1[c]["deal_rate_adversarial"]["mean"]
            if math.isnan(id_m) or math.isnan(ad_m):
                continue
            ax.scatter([id_m], [ad_m], s=80)
            ax.annotate(c, (id_m, ad_m), textcoords="offset points", xytext=(5, 5))
        ax.set_xlabel("In-distribution deal rate")
        ax.set_ylabel("Adversarial-partner deal rate")
        ax.set_title("Generalisation slope")
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(out / f"fig_generalisation.{ext}")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(5, 4))
        for c in cells:
            id_m = table1[c]["deal_rate_id"]["mean"]
            hl_m = hull_per_cell.get(c, {}).get("mean", math.nan)
            if math.isnan(id_m) or math.isnan(hl_m):
                continue
            ax.scatter([hl_m], [id_m], s=80)
            ax.annotate(c, (hl_m, id_m), textcoords="offset points", xytext=(5, 5))
        ax.set_xlabel("Fraction of utterances out-of-hull")
        ax.set_ylabel("In-distribution deal rate")
        ax.set_title("Strategic novelty vs performance")
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(out / f"fig_hull.{ext}")
        plt.close(fig)
    except ImportError:
        print("matplotlib unavailable; skipping figures")

    print(f"Wrote report to {out}")


if __name__ == "__main__":
    main()
