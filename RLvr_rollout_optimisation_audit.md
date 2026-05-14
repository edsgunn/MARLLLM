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

---

## Audit findings against this codebase (May 2026)

Per-task state after walking the source. References point at `main`.

| # | Optimisation | State | Reference |
|---|---|---|---|
| 1 | Async rollout/train | **Absent** | `marlllm/population.py:586-746` — strict `collect → forward+backward → optimizer.step → vLLM weight sync` per iteration. `time/rollout_s` (already logged) typically reports 70-85% of iter wall time spent in rollout. |
| 2 | Length bucketing / token-budget batching | **Absent** | `marlllm/trainer.py:201-204` (and the parallel block in `population.py`) — `grad_accum_steps` defines microbatches as a *fixed sequence count*; no length sorting; no token-budget batching. One mitigation at `trainer.py:218-223` trims each microbatch back to its local max length post-slice. |
| 3 | Sequence packing | **Absent** | `marlllm/types.py:117-128` — `RolloutBatch.from_trajectories` right-pads every trajectory to `T_max`. No `cu_seqlens`, no varlen. Trajectory lengths within one agent's batch vary 3-5× empirically, so pad tokens typically dominate FLOPs. |
| 4 | FlashAttention on training | **Partial** | `marlllm/agent.py:406-407` — `attn_impl` is a config knob; configs default to `sdpa`. SDPA on Hopper avoids the O(L²) attention-matrix materialisation but is ~20-30% slower than `flash_attention_2`. |
| 5 | Gradient checkpointing | **Present** | `marlllm/agent.py:420-423` — wired in, toggleable, enabled in every long-context config. |
| 6 | Max-length cap + over-length policy | **Present** | `marlllm/population.py:1046-1048` — `max_episode_tokens` is a hard cap; over-budget episodes get `null_indices` (step with `None` until termination) unless the env signals `must_act`. Consistent across forum/Concordia envs. |
| 7 | Length-drift logging | **Present** | `marlllm/population.py:1620-1640` — per-env logs include `mean_post_tokens`, `total_truncation_rate`, `post_truncation_rate`, `act_tokens`, `think_tokens`, `post_tokens`, `max_episode_total_tokens`. `kl_coef` / `beta` are configurable. No explicit length penalty on reward — visibility without an active controller. |

### What's actually on the critical path for us

We had to learn this from OOMs at long sequences with native-thinking RL. Two findings the generic audit wouldn't reveal:

- **Per-agent serial loop** (`population.py:608-...`). The population trainer iterates agents one at a time and does each agent's forward + backward sequentially even when they share a LoRA base. For population size N this serialises N forward+backward passes inside every step. Per-agent wall-time logging is now added (`per_agent/<name>/fb_s` / `per_agent/fb_s_max` / `per_agent/fb_s_sum`) so we can quantify the dominance of this loop over training time.
- **Stragglers inside one rollout iteration**, not just across iterations. As episodes terminate independently, the active set shrinks, and vLLM batching shrinks with it; tail iterations spend a long time generating with batch sizes of 1-2. A "soft async" mode — start the next iteration's episodes when the active set drops below threshold — would recover much of this without the full Task 1 build-out.

### Recommendations, by effort × leverage

1. **Task 3 — packing + flash_attention_2** *(in flight as of this write-up)*. Highest leverage we can get without rearchitecting the loop. Removes pad-token FLOPs (40-60% of training compute at our agent counts) and the O(L²) memory materialisation.
2. **Per-agent serial loop**: once `per_agent/fb_s_*` makes the cost visible, the natural follow-up is a packed batch carrying the existing `agent_id_mask` with per-token adapter routing — collapses N forward+backwards into one packed pass on the shared LoRA base. Larger win than Task 4, less invasive than Task 1.
3. **Soft-async rollout** (intermediate before full Task 1): when active set in `_collect_episodes_batched` drops below a threshold, start the next iteration's episodes in parallel. Keeps vLLM batches saturated, no buffer / staleness bookkeeping.
4. **Full Task 1 (rollout buffer + bounded staleness)**: defer until the above three exhaust their leverage. Our PPO already carries `act_log_probs_old` per token, so the importance-ratio machinery is in place — the engineering is purely orchestration, not loss math.
5. **Task 2 (length bucketing)**: a free dividend once packing is in (microbatches packed by token budget rather than sequence count).
6. **Task 4 (flash_attention_2)**: a one-line config change once `flash-attn` is in the venv; ships with packing.
7. **Length controller** (Task 7 extension): the regime tracker already gives us length visibility; if length grows while reward stays flat, add a mild length penalty or per-trajectory token budget. Don't add it pre-emptively — only when the signal demands it.

### What changed in this audit pass

- `_extract_tagged` in `envs/forum/env.py` rewritten so only outermost tags are functional (a `<post>` inside `<think>` is now just text).
- Per-agent wall-time logging added in `marlllm/population.py` so the serial-agent-loop hypothesis can be tested empirically before we spend effort on the packed-multi-adapter forward. Metrics: `per_agent/<name>/{collate_s,fb_s,total_s,n_traj,n_micros,T_max}` plus aggregates `per_agent/fb_s_{max,mean,sum}`.
- `marlllm/context_formatter.py` auto-injects the Qwen3-family `<think>\n` primer on the OBS side of the OBS/ACT boundary (loss-masked-out by construction).
- **Task 3 (sequence packing) implemented** with opt-in config flag `pack_sequences: true` (CLI `--pack-sequences`):
  - New `RolloutBatch.pack_trajectories(...)` in `marlllm/types.py` builds a `(1, T_total)` batch with per-token `seq_ids`, `position_ids` (reset at each trajectory boundary), and `cu_seqlens`.
  - The forward path passes `attention_mask=None` + `position_ids` to transformers' built-in packed-sequence detection (`find_packed_sequence_indices` in `transformers/masking_utils.py`, torch ≥ 2.6, transformers ≥ 4.55) — the block-diagonal causal mask is constructed as a mask-function in attention, **not** materialised as a dense 4D tensor. No O(T²) memory cost on the autograd graph.
  - Boundary-aware reductions in `marlllm/loss.py`: `_compute_returns` resets the running discount sum at trajectory boundaries; the cross-boundary shifted ACT position is excluded from policy/value losses; the cross-boundary OBS position is kept in `obs_mask` but its surprise is zeroed so the mean denominator matches the padded path (verified bit-exact equivalence in `tests/test_packed_vs_padded.py` style probe).
  - Constraint: mutually exclusive with `seq_chunk_size` in this release — chunked + packed together needs cu_seqlens-aware per-chunk masks, follow-up work.
  - Constraint: needs `transformers ≥ 4.55` (legacy `.venv` ships 4.57 → ok; flagged for the qwen3p5 venv at 5.8 → ok).