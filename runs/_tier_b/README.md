# Tier-B analyses

Eight questions answered from data already on disk, with no new training. Named
B1–B8 after the ledger in [`REWRITE.md`](../../REWRITE.md).

Each has a script in [`scripts/tier_b/`](../../scripts/tier_b/) whose module
docstring is the **pre-registration** — what the analysis changes, what we expect
to learn, and what would falsify it — written before it was run, in the role
`PROCESS.md` gives a config header. Each has a `NOTES.md` here with what it
found. The `.csv` and `.json` outputs stay local.

| | question | verdict |
|---|---|---|
| [B1](b1_policy_distance_by_regime/NOTES.md) | Is cross-population policy distance culture, or collapse? | Survives on 2 of 3 datasets; Margin Notes loses 45% |
| [B2](b2_regime_by_iteration/NOTES.md) | Is a regime an outcome, or a snapshot? | A snapshot. No population improves after iteration 150 |
| [B3](b3_substrate_relative_regimes/NOTES.md) | Does `perc ≥ 2.0` measure breakdown or difficulty? | Two substrates start above the cut; 50% of labels move |
| [B4](b4_public_channel/NOTES.md) | Does the public channel close? | Yes, in 63/65 runs, ~150 iterations before the classifier notices |
| [B5](b5_env_likelihood/NOTES.md) | Is `p_env` recoverable? | Provenance yes (100%); stored log-probs no |
| [B6](b6_token_accounting/NOTES.md) | Does `thinking_frac` measure deliberation? | No. Zero `<think>` tags anywhere; it is a residual |
| [B7](b7_absorption_horizon/NOTES.md) | Is absorption measured while populations are healthy? | Seeds yes; cross-env contrast falls +0.199 → +0.024 |
| [B8](b8_model_generation/NOTES.md) | Does this depend on the base model? | Strongly. No Qwen3 agent ever reaches KL ≥ 1 |

Two of these overturned something. B6 falsified the private-deliberation reading
of the rising token-mix share that `REWRITE.md` §3.5 had built on, and B5
falsified the claim that restoring the dropped environment term would be nearly
free. Both corrections are carried into `REWRITE.md` and `STATE.md`.

## Reproducing

```bash
for s in scripts/tier_b/b*.py; do .venv/bin/python "$s"; done
```

No GPU. B5 needs the Qwen2.5-7B tokeniser from the local HF cache.
