# B7 — is absorption measured while the population is still healthy?

|  |  |
|---|---|
| Script | [`scripts/tier_b/b7_absorption_horizon.py`](../../../scripts/tier_b/b7_absorption_horizon.py) |
| Inputs | `_absorption_by_regime/absorption_{cross,seeds}_with_regime.csv` (1,632 + 640 rows), joined to alive-fraction curves |
| Ran | 2026-09-01, no compute |

## What it was for

Absorption is a first-minus-last difference. [B2](../b2_regime_by_iteration/NOTES.md)
shows populations peak around iteration 100 and never recover after 150, while
the absorption window runs 25 → 300. If a run's last checkpoint is deep into its
decline, the "shrinkage" is partly the difference between a healthy population
and a failing one — the artefact the original report identified for the
text-degenerate stratum. Contemporaneous labelling fixes the *label*; it does not
stop the last checkpoint from being late.

A measurement counts as taken during decline if the population's alive fraction
at that iteration is strictly below its own maximum.

## Headline

**The contrast is intact where the window is clean and largely horizon-driven
where it is not.**

The reproduction matches the published values exactly — committed-minus-coherent
shrinkage +0.199 (cross) and +0.160 (seeds), the numbers quoted in the
threshold-sweep rebuttal.

| dataset | measurements during decline | contrast, all | contrast, healthy window |
|---|---:|---:|---:|
| cross-environment | 512/1,632 (31%) | **+0.199** | **+0.024** |
| Inner Loop seeds | 0/640 (0%) | **+0.160** | **+0.160** |

Per stratum on the cross-environment set, committed shrinkage falls from +0.254
to +0.067 and coherent from +0.056 to +0.044 when the declining measurements are
dropped.

The Inner Loop 5-seed replication never declines inside its window — its runs
end at iterations 100–144, before the decline sets in — so its contrast is
untouched. **That dataset is the clean one, and it carries the result.**

## What this settles

- **The cross-environment pooled contrast is substantially inflated by
  measurements taken during population decline.** +0.199 → +0.024. — *likely*
- **The Inner Loop seed replication is horizon-clean and its contrast stands
  unchanged.** — *established*
- **The paper should lead with the seed replication, not the cross-environment
  pool** — which is what the rebuttal already said on other grounds. — a
  recommendation

## What it does not settle

**The ambiguity was pre-registered and it bites.** Restricting to the healthy
window also shortens the window, and absorption needs training to happen in it.
A smaller contrast on a shorter window is ambiguous between "artefact" and "less
time to absorb". The seeds dataset is what breaks the tie — there the window is
not shortened at all and the contrast holds — but that is an argument, not a
control.

A cleaner version would hold the iteration range fixed across strata rather than
using each agent's own first and last checkpoint. That is not done here.

## Status

`CLOSED`. Cite the seed replication; report the cross-environment pool with this
sensitivity attached.

---
*Analysed 2026-09-01. Machine output stays local.*
