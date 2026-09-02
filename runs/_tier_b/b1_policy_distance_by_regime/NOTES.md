# B1 — cross-population policy distance, restricted by regime

*Is the divergence culture, or collapse?*

|  |  |
|---|---|
| Script | [`scripts/tier_b/b1_policy_distance_by_regime.py`](../../../scripts/tier_b/b1_policy_distance_by_regime.py) |
| Inputs | `_diagnostics_d2d3/d3_sym_kl_pairs.csv` (1,632 pair-rows), `_diagnostics_d1_coherent_only/regimes_*.csv` (160 labels) |
| Ran | 2026-09-01, no compute |

## What it was for

Diagnostic 3's policy-distance headline — the number `REWRITE.md` proposes the
paper should lead with — is computed over full populations. Two pages earlier,
Diagnostic 2 established that collapsed adapters are far more confident than
their siblings (policy-collapse mean entropy difference −0.212). Two adapters
that have each collapsed to a *different* near-deterministic mode are genuinely
far apart in KL. D1 applied the equivalent restriction to the realisation-level
trajectory KL and the signal rose 20–40%. D3 was never restricted.

## Headline

**The headline survives on two of three datasets and loses 45% of its magnitude
on the third.** Restricting to pairs where the character is alive in *both*
populations, at the end of training:

| dataset | full | both alive | ratio | pairs kept |
|---|---:|---:|---:|---|
| Inner Loop 8a (iter 100) | 0.585 | 0.585 | **1.00** | 48/48 |
| Ashbourne 16a (iter 150) | 1.349 | 1.282 | **0.95** | 137/160 |
| Margin Notes 8a (iter 150) | 1.359 | 0.748 | **0.55** | 36/48 |

Margin Notes is the dataset that loses half its population by iteration 300 and
the one carrying D2's two extreme seed-1 collapse outliers. Its 1.359 — one of
the two largest values in the published D3 table — is the least trustworthy
number in it.

**The qualitative claim is untouched.** Under the restriction, growth from the
iteration-25 baseline is still ×27 (Ashbourne), ×64 (Inner Loop) and ×112
(Margin Notes), monotone throughout, from a baseline of 0.005–0.048.

## Core metrics

Mean symmetric per-character policy distance, `both_alive` filter:

| iteration | Inner Loop | Margin Notes | Ashbourne 16a |
|---:|---:|---:|---:|
| 25 | 0.0092 | 0.0067 | 0.0481 |
| 50 | 0.0871 | 0.0679 | 0.2719 |
| 75 | 0.2795 | 0.2552 | 0.5715 |
| 100 | 0.5854 | 0.5033 | 0.7074 |
| 150 | — | 0.7476 | 1.2817 |

The strictest filter — coherent in both populations — gives 0.193 / 0.366 /
0.558, that is 27–41% of the full value. That is expected and is *not* the
control: coherent agents sit close to base and therefore close to each other, so
the strict filter measures within-coherent-cluster divergence, a different
quantity. D1 made the same observation for trajectory KL.

## What this settles

- **Cross-population divergence is not an artefact of collapsed adapters.**
  Removing every pair involving a failed adapter changes the estimate by 0–5% on
  two of three datasets. — *likely*
- **The pooled magnitudes are inflated where failure regimes are present.**
  Margin Notes falls 45%. — *established*
- **`both_alive` should be the reported primary**, not `full`. — a
  recommendation, not a finding

## What it does not settle

The regime labels are **end-of-training**, so an agent that collapsed late is
treated as collapsed throughout — the same limitation D1 carries. A
contemporaneous version would need per-iteration labels joined to the pair rows,
which the shards support and this script does not do.

Nor does it address whether policy distance is the right object at all: it
measures that two adapters differ, not what they differ *about*.

## Status

`CLOSED`. Report `both_alive` as primary and state the Margin Notes sensitivity.

---
*Analysed 2026-09-01. Machine output stays local.*
