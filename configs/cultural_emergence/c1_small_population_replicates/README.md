# C1 — seed replicates of the small-population corner

Twenty configs: **5 seeds × 2 substrates × 2 population sizes**, 300 iterations
each, checkpoints every 25. This is [REWRITE.md](../../../REWRITE.md) Tier C
item **C1**, and it exists to answer one question — *is the N=2 / N=4 corner of
Figure 1 a property of the cell, or of the one seed that was run there?*

Everything is field-for-field identical to the four `run10_*` configs one
directory up. Only the seed moves.

## The cells, and what one seed said

| substrate | N | reference run | `alive` at 25 / 100 / 150 / 200 / 300 |
|---|---:|---|---|
| `study_group_ashbourne_gc` | 2 | [`run10_2agent_…_ashbourne_gc`](../../../runs/cultural_emergence/run10_2agent_7B_study_group_ashbourne_gc/NOTES.md) | 0.00 · 0.00 · 1.00 · 1.00 · 0.00 |
| `study_group_ashbourne_gc` | 4 | [`run10_4agent_…_ashbourne_gc`](../../../runs/cultural_emergence/run10_4agent_7B_study_group_ashbourne_gc/NOTES.md) | 0.00 · 1.00 · 1.00 · 0.75 · 0.75 |
| `study_group_strathearn_server` | 2 | [`run10_2agent_…_strathearn_server`](../../../runs/cultural_emergence/run10_2agent_7B_study_group_strathearn_server/NOTES.md) | 1.00 · 1.00 · 0.00 · 0.00 · 0.00 |
| `study_group_strathearn_server` | 4 | [`run10_4agent_…_strathearn_server`](../../../runs/cultural_emergence/run10_4agent_7B_study_group_strathearn_server/NOTES.md) | 1.00 · 1.00 · 1.00 · 0.75 · 0.75 |

Every one of those rows is **n = 1**. The figure reads the 2-vs-4 contrast on
Strathearn (0.00 vs 0.75 at iteration 300) as a population-size effect; nothing
in the data so far separates it from seed noise. Note also that the 2-agent
Ashbourne cell is non-monotone in iteration — dead at 100, alive at 150–200,
dead at 300 — so *which iteration the figure is drawn at* is doing work too.
That is why the checkpoint grid matters as much as the seed count.

## The seed ladder

`seed1`→1, `seed2`→42, `seed3`→625, `seed4`→8192, `seed5`→123 — the same
label→value mapping as the `run7_*_seedN` and `run8_*_seedN` families, so a seed
label means the same thing across families.

The run directory suffix is what
[`scripts/aggregate_seed_replicates.py`](../../../scripts/aggregate_seed_replicates.py)
parses (`_seed(\d+)$`), so the names have to keep that form.

## Why seed 1 is re-run rather than reused

The four reference runs executed on **2026-05-06** against a working tree that
was ahead of the last commit — their configs already use the
`characters:`/`environment:` split that was not committed until `7ca25a6` on
05-11 — so git cannot establish what code they ran. `marlllm/` has since gained
regime tracking, the chunked/packed loss paths, the async rollout path and the
environment-likelihood work.

Rather than assume that drift is inert, all five seeds are run fresh. The family
is then self-contained, and the 05-06 run becomes a **free code-drift check**
instead of a load-bearing member of the comparison: if fresh seed 1 lands inside
the band of the other four, the historical run is comparable and can be pooled;
if it lands outside, the band is the answer and the old run is set aside.

## What was verified before writing these

- **The substrate is the same substrate.** The 05-06 traces carry the current
  `<post>` / `<python>` instruction text, byte-identical to what
  `ForumEnv._render_initial_ctx` emits today, and contain no `<think>` tag.
- **`thinking_enabled` was already inert** when those runs executed —
  [`train_population.py`](../../../train_population.py) builds `ForumEnv` from
  explicit `spec_dict.get()` calls and never reads that key. It is dropped here
  rather than carried forward as a dead field that misdescribes the substrate.
  See [REWRITE.md](../../../REWRITE.md) B6, which is the same mis-naming.
- **The rollout pathway is pinned**, not inherited: `async_rollout: false`,
  `ppo_clip: 0.0`, `pack_sequences: false`, `seq_chunk_size` unset. The
  reference runs' `config.json` predates all four keys, so they ran the
  synchronous, single-shot, plain-REINFORCE path. These are also the current
  defaults — pinning changes no behaviour, it stops a later change of default
  from moving a replicate off the pathway it is compared against.
- **What is new and left on:** `regime_tracking_enabled` and
  `regime_pre_collapse_checkpoint` default to true and did not exist on 05-06.
  Both are observation-only — labels, plots and extra checkpoints, no gradient
  path — and the regime table is step 2 of the analysis protocol in
  [PROCESS.md](../../../PROCESS.md), so they stay on.

## Falsification condition

Pre-registered, before the data: **if the within-cell spread of `alive` across
the five seeds is comparable to the between-cell difference the figure reads as
a population-size effect, the N=2 and N=4 points carry no information about
population size and the corner comes out of the figure.**

Tier D has already withdrawn the causal claim ("population size determines which
regime is accessible"). This decides whether the descriptive one survives.

A run is **void** if it reaches fewer than 300 iterations, or if its
`tool/post_*` accounting differs structurally from the reference run's — the
substrate has to be the same substrate for a replication to mean anything.

## Cost

The reference runs took 2.5–5.5 h each on 4 × GH200. Twenty runs is roughly
80 GPU-hours of walltime across 20 single-node jobs, all inside the 24 h
`workq` limit. This is the cheapest item in Tier C.

## Status

**Submitted 2026-09-02 as jobs 6260359–6260378**, queued behind the two
environment-likelihood arms (6256587/6256588). Job-to-run mapping is in each
config's `STATUS:` line and in `slurm_logs/submission_20260902_192435.txt`.

## Submitting

```bash
export HF_TOKEN="hf_..."
.venv/bin/python submit_batch.py \
    configs/cultural_emergence/c1_small_population_replicates/ \
    --account brics.a5l \
    --hf-home /lus/lfs1aip2/projects/a5l/egunn/hf_cache \
    --scripts-dir slurm_scripts/
```

`--dry-run` first to print the twenty job scripts without submitting.

**Use `.venv/bin/python`, never `uv run`** — see
[PROCESS.md](../../../PROCESS.md).

Outputs land in
`runs/cultural_emergence/c1_small_population_replicates/<name>/`.

## When they finish

The unit of inference is the **family**, not the run, so the write-up is one
`runs/cultural_emergence/c1_small_population_replicates/NOTES.md` rather than
twenty. Per [PROCESS.md](../../../PROCESS.md):

```bash
.venv/bin/python scripts/plot_results.py \
    runs/cultural_emergence/c1_small_population_replicates
.venv/bin/python scripts/plot_population_regime_figures.py
```

Then read traces at iteration 200 for at least one seed per cell — a policy that
has collapsed into a `</post>` / `</python>` loop still scores as healthy on
every token-count metric, and that exact failure is what the 2-agent Strathearn
cell turned out to be at seed 1.
