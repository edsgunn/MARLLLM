# B5a — is `p_env` recoverable for agent-produced observation tokens?

*The premise of the whole "we must drop the environment term" argument.*

|  |  |
|---|---|
| Script | [`scripts/tier_b/b5_env_likelihood_recoverability.py`](../../../scripts/tier_b/b5_env_likelihood_recoverability.py) |
| Inputs | episode traces (canonical thread + per-agent `context_tokens`), Qwen2.5-7B tokeniser, 3 checkpoints |
| Ran | 2026-09-01, no compute |

## What it was for

Appendix A drops `Σ_s σ_s ln p_env(x_s | x_<s)` because "`p_env` is only ever
sampled, not evaluated". In a multi-agent LLM population that is only true of
*some* observation tokens: the persona prefix, structural markers and tool
outputs are deterministic (`ln p_env = 0`), and the rest is other agents' posts —
sampled from LoRA adapters over the same frozen base. `REWRITE.md` §5 proposed
that the dropped term is therefore recoverable. This tests the premise.

## Headline

**Half true, and the half that fails is the half that made it look cheap.**

- **Receiver side: 100%.** Every non-empty post appears verbatim, as a
  contiguous interior token span, in every other agent's context — in all three
  checkpoints, at both iteration 25 and iteration 300. Observation-token
  provenance is fully resolvable from logged artefacts.
- **Producer side: 42–88%**, and it degrades over training: 87.5% at iteration
  25 falls to 41.7% at iteration 300 in the same run.

The producer-side failure is not noise. The text placed on the thread is the
*extracted* post — chunks joined and stripped — so for a malformed generation it
is not a contiguous span of what the agent actually emitted. The match rate
tracks post-format breakdown, which is an independent corroboration of
[B4](../b4_public_channel/NOTES.md) and [B6](../b6_token_accounting/NOTES.md).

## What this means for §5

**The bias measurement (B5b) is well posed.** Provenance resolves at 100%, so
for every observation token we can identify the producing agent and score that
agent's adapter on it. Nothing new needs instrumenting in the trainer.

**Reusing stored log-probs does not work.** `REWRITE.md` §5.1 claimed the
dropped term was "recoverable at close to zero marginal cost" because
`act_log_probs_old` is already in the buffer. That is wrong: the producer's
token sequence is not the receiver's token sequence, so the stored per-token
log-probs do not align. Restoring `p_env` means **re-scoring the producing
agent's adapter on the receiver's tokenisation** — an extra forward pass per
agent per episode, not a lookup. Feasible, since all adapters share the frozen
base and are swapped in anyway; not free.

A second caveat worth stating in the paper: the formalism conditions `p_env` on
the *receiver's* context, while the generating distribution was the producer's
policy on the *producer's* context. Same event, different conditioning sets.

## What this settles

- **Observation-token provenance is fully resolvable from logged traces.**
  100% receiver-side match at every checkpoint tested. — *established*
- **`p_env` is not a black box in this substrate class.** — *likely*
- **The zero-cost version of the restored-return proposal does not exist.** —
  *established*

## What it does not settle

The covariance itself. No logged artefact carries per-token log-probabilities —
the trace JSON holds only `context_tokens` and `context_text` — so **B5b needs a
GPU scoring pass and is Tier C, not Tier B.** `REWRITE.md` has been corrected.

## Status

`CLOSED` as a feasibility check. B5b remains open and is now correctly costed.

---
*Analysed 2026-09-01. Machine output stays local.*
