# RLVR Rollout Throughput — Codebase Audit Checklist

> **Purpose:** Audit guide for Claude Code. The training loop suffers from rollout
> latency that scales badly with trajectory length in RLVR training. The cause is
> a mix of genuinely quadratic attention cost and — more likely dominant at this
> scale — variable-length stragglers and padding waste. The tasks below are
> ordered by expected leverage. For each, inspect the relevant code, report what
> the codebase currently does, and propose concrete changes.
>
> **Scope note:** This is model-agnostic. Do not special-case any particular model
> or environment — the training harness should handle variable-length rollouts
> well regardless of which model/env is plugged in.

---

## How to use this document

For each task:
1. Locate the relevant code paths (suggested search terms are given).
2. Report the **current behaviour** — don't assume, verify in the source.
3. State whether the optimisation is **absent / partial / present**.
4. If absent or partial, propose a specific change with file/function references.
5. Flag any change that would alter training semantics (e.g. introduces policy
   staleness) so it can be reviewed before merging.

---

## Task 1 — Determine if the rollout→train loop is synchronous (highest leverage)

**Problem:** If the loop is `generate batch → train on exactly that batch →
repeat`, then step time is gated by the *slowest* trajectory in every batch.
GPUs idle while waiting for long rollouts to finish.

**What to look for:**
- The main training loop. Search: `for ... in range(num_steps)`, `rollout`,
  `generate`, `.step()`, `train_step`, trainer entry points.
- Whether generation and the optimiser update happen in strict lockstep on the
  same batch, or whether there is any buffering / queue between them.

**Target state:** Disaggregated generation and training — rollout workers
produce trajectories into a buffer; training workers consume whatever is ready.
Bounded staleness of ~1–4 policy versions is acceptable and standard.

**Report:**
- Is the loop synchronous or already buffered/asynchronous?
- If synchronous: identify the smallest change that introduces a rollout buffer
  with bounded staleness. Note this changes on-policy → mildly off-policy; flag
  for review.

---

## Task 2 — Length bucketing and token-budget batching

**Problem:** Microbatches assembled from randomly-mixed trajectory lengths waste
compute on padding and reintroduce the straggler effect *within* each microbatch.

**What to look for:**
- Batch/microbatch assembly. Search: `collate`, `pad`, `batch_size`,
  `DataLoader`, `sampler`, `microbatch`.
- Whether batch size is defined as a **fixed number of sequences** (bad for
  variable length) or a **fixed token budget** (good).

**Target state:**
- Trajectories sorted/bucketed by length so each microbatch is roughly uniform.
- Batch size expressed as a total token budget, not a sequence count, so GPU
  utilisation stays flat as length distribution shifts.

**Report:**
- Current batching strategy and where it's defined.
- Whether length-aware bucketing exists. If not, propose a length-bucketed
  sampler and a token-budget batch size.

---

## Task 3 — Sequence packing with block-diagonal attention masks

**Problem:** Padding variable-length sequences to the batch max burns FLOPs on
pad tokens. With a long-tailed length distribution this waste is large.

**What to look for:**
- The training forward pass and how sequences enter the model. Search:
  `attention_mask`, `pad`, `pack`, `cu_seqlens`, `varlen`.
- Whether sequences are padded-to-max or concatenated into dense packed blocks.

**Target state:** Variable-length trajectories concatenated into fixed-length
packed blocks with a block-diagonal mask (no cross-trajectory attention),
using FlashAttention's varlen API (`cu_seqlens`). Eliminates padding FLOPs.

**Report:**
- Is packing used, or are sequences padded?
- If padded: propose a packing implementation, noting any loss-masking or
  per-trajectory reduction code that must be updated to respect block boundaries
  (advantage/return computation must not bleed across packed trajectories).

---

## Task 4 — Confirm FlashAttention (or equivalent) on the *training* pass

**Problem:** vLLM uses an efficient attention kernel for *generation*, but the
*training* forward/backward pass may fall back to vanilla attention, which
materialises the O(L²) attention matrix in memory — forcing tiny batch sizes or
OOM at long L.

**What to look for:**
- Model instantiation / config for the training pass. Search:
  `attn_implementation`, `flash_attention`, `sdpa`, `eager`.
- Confirm the *trainer's* model — not just the vLLM inference engine — uses
  FlashAttention or memory-efficient SDPA.

**Target state:** Training forward/backward uses FlashAttention (varlen variant
if Task 3 is implemented). This doesn't change O(L²) compute asymptotics but
removes the O(L²) memory materialisation.

**Report:**
- Which attention implementation the training model uses.
- If eager/vanilla: propose the config change and check varlen compatibility.

---

## Task 5 — Gradient checkpointing for long-context feasibility

**Problem:** Long sequences OOM, which forces batch-size collapse — an indirect
but large throughput loss.

**What to look for:**
- Trainer config / model setup. Search: `gradient_checkpointing`,
  `checkpoint`, `use_reentrant`.

**Target state:** Gradient checkpointing available and enabled when sequence
length is large. Costs ~30% extra compute but prevents OOM-driven batch-size
collapse.

**Report:**
- Whether gradient checkpointing is wired in and toggleable.
- If absent: propose enabling it, ideally conditioned on a length threshold.

---

## Task 6 — Max length cap and over-length trajectory handling

**Problem:** Unbounded rollout length means a single runaway trajectory can
dominate a step.

**What to look for:**
- Rollout/generation config. Search: `max_tokens`, `max_new_tokens`,
  `max_length`, `truncat`.
- How over-length trajectories are handled: discarded, truncated, or
  marked-as-negative-reward.

**Target state:** An explicit max rollout length, with a defined and consistent
policy for over-length trajectories (discard vs. truncate vs. penalise).

**Report:**
- Current max length setting and over-length handling.
- Whether handling is consistent across environments.

---

## Task 7 — Diagnose *why* traces lengthen over training (semantic, not just perf)

**Problem:** If trace length grows *during* training, it may be verbosity drift
or length-hacking rather than genuine additional reasoning. This is a
reward-shaping issue, not just a compute issue — and it interacts with KL
regularisation.

**What to look for:**
- Reward computation. Search: `reward`, `length_penalty`, `kl`, `kl_coef`,
  `kl_penalty`.
- Whether the reward includes any length term, and whether KL is configured to
  discourage drift from the reference policy.
- Any logging of mean/median/p95 trace length over training steps.

**Target state:**
- Trace-length distribution logged per step (mean, median, p95) so drift is
  visible.
- A decision point: if length growth is verbosity drift, a mild length penalty
  or appropriately-tuned KL term is doing real work, not papering over a
  compute problem.

**Report:**
- Whether length is logged over training.
- Whether reward/KL currently constrain length.
- Recommend adding length-distribution logging if absent, so this can be
  diagnosed empirically.

---

## Summary table

| # | Optimisation | Type | Expected leverage |
|---|---|---|---|
| 1 | Disaggregate rollout/train (buffer + bounded staleness) | Latency | High |
| 2 | Length bucketing + token-budget batching | Latency / waste | High |
| 3 | Sequence packing + block-diagonal mask | Waste | Med–High |
| 4 | FlashAttention on training pass | Memory | Med (High if absent) |
| 5 | Gradient checkpointing | Memory | Med (indirect) |
| 6 | Max length cap + over-length policy | Latency | Med |
| 7 | Diagnose length drift (reward/KL/logging) | Semantic | Med |

**Expected dominant cause at small model scale:** stragglers + padding waste
(Tasks 1–3), not raw O(L²) attention FLOPs. Prioritise accordingly, but verify
against the actual codebase rather than assuming.