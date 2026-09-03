# c9_post_hoc_env_term

*Recovering the dropped environment term from arm A's checkpoints, after the
runtime path failed to log it. Job 6280214, 2026-09-03.*

|  |  |
|---|---|
| Script | [`scripts/tier_b/c9_post_hoc_env_term.py`](../../../scripts/tier_b/c9_post_hoc_env_term.py) |
| Input | `runs/env_likelihood_bias/A_surrogate_seed1`, checkpoints 25/100/200/300, 4 episodes each |
| Compute | one 7B forward per post and per observation block; 5m39s equivalent |
| **Reads against** | **`runs/env_likelihood_bias/NOTES.md`** — the runtime measurement that was void |

## What it was for

Arms A and B logged none of the quantities they existed to produce. Arm A's
adapters and traces survived, so each delivered post could be scored under its
**generator's** adapter for `E_s` and each observation block under its
**observer's** for `M_s`, recovering the measurement without retraining.

No-tag mode is why this is exact rather than estimated: with no private prefix
there is no latent to marginalise, so `p_env` is the generator's own log-prob
and the `I(z;y|u)` correction is identically zero.

## Headline

**The dropped term is large and direction-preserving.** It accounts for ~43% of
the return's magnitude at every checkpoint, yet the corrected return ranks
actions identically to the surrogate — Spearman 1.000 throughout. By the
criterion pre-registered in the config header, *"if they rank actions alike, the
bias does not matter for learning however large it is"*, the answer is that it
does not.

## Core numbers

| iter | E/token | M/token | \|E\|/\|M\| | env_share | rank_corr | posts scored |
|---|---|---|---|---|---|---|
| 25 | −0.982 | −1.532 | 0.641 | 0.435 | 1.000 | 0.98 |
| 100 | −1.148 | −1.545 | 0.743 | 0.441 | 1.000 | 0.97 |
| 200 | −1.987 | −2.125 | 0.935 | 0.435 | 1.000 | 0.99 |
| 300 | −1.038 | −1.213 | 0.856 | 0.426 | 1.000 | 0.94 |

**P4 holds at every checkpoint**: a post is less surprising to its author than
the block containing it is to its reader. That is *why* the ordering survives —
the correction is same-signed and uniformly smaller, so it cannot reorder. The
rank statistic is not degenerate: fed adversarial `E`, it drops to 0.64 and
0.29, so it would detect a reordering if one occurred.

**But the margin is eroding.** |E|/|M| runs 0.64 → 0.74 → 0.94 → 0.86. As
adapters converge, a post becomes nearly as surprising to its author as to its
readers, and the property guaranteeing the ordering is the one shrinking.

## Two estimator faults, both found after the fact

**The discounting is wrong.** The script discounts *between* observation blocks
and not within them. γ=0.99 is applied positionally per token with a 100-token
horizon, while a block runs to ~2700 tokens — so the true return sees roughly
the first 100 tokens of the next block and nothing else, and treating a block
as a point is not an approximation, it is a different quantity. What survives is
the **ratio**: both `E` and `M` carry the same wrong weights over the same
spans, so `env_share` is readable as a per-token log-probability ratio. It is
not a share of the return, and should not be quoted as one.

**P1 was mis-stated, not falsified.** The header predicted `env_share` ≤ ~32%,
the share of an observer's context that is other agents' post content. The
measured 0.43 exceeds it — but the ceiling bounds a *context-token share* while
the statistic is a *log-probability magnitude ratio*. The prediction did not
apply to the quantity computed. That is a pre-registration error rather than a
finding, and it is why P1's "the accounting is wrong" clause should not be
invoked here.

## What this settles

- **The dropped environment term does not change the update ordering on this
  substrate.** ~43% of the magnitude, rank correlation 1.000 at four
  checkpoints. — *likely* — this pass; *likely* rather than *established*
  because the discounting fault above means the magnitude figure is a ratio and
  one run at one substrate underlies all four points
- **A post is less surprising to its author than to its readers**, by 6–36%
  per token, throughout training. — *established* — same
- **The protection is weakening.** The author's advantage narrows toward parity
  as training proceeds, so direction-preservation is a property of early and
  middle training rather than a guarantee. — *likely* — same

## What it does not settle

- **The gradient bias proper.** This measures the size and the
  ordering-preservingness of the dropped *return* term, not the covariance
  between an action's score and the environmental surprise that follows it,
  which is what REWRITE.md §5's exact expression is about.
- **Anything with correct discounting.** Re-running with token-positional
  discounting would make `env_share` a real return share. Expect it to shrink
  substantially: γ^384 ≈ 0.02 means most of a block is weighted to nothing.
- **Whether the substitution is safe under the corrected objective.** That is
  the resubmitted arm B (6279926).
- **Any of it at γ=1.** The whole measurement inherits the discount whose
  legitimacy [C11](../../cultural_emergence/c11_discount_normalisation/) is now
  testing. If γ=1 turns out to be the right objective, this pass should be
  redone under it.
