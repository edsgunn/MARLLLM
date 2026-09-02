# C6 — does the decline bottom out?

Two runs, each resuming a completed reference population from its
iteration-300 checkpoint and continuing.
[REWRITE.md](../../../REWRITE.md) Tier C item **C6**.

| config | reference | 300 → | est. walltime |
|---|---|---:|---:|
| `c6_extend_8agent_ashbourne_gc_to900` | [`run7_8agent_…`](../../../runs/cultural_emergence/run7_8agent_7B_study_group_ashbourne_gc/NOTES.md) | 900 | ~10.5 h |
| `c6_extend_16agent_ashbourne_gc_to600` | [`run8_16agent_…`](../../../runs/cultural_emergence/run8_16agent_7B_study_group_ashbourne_gc/NOTES.md) | 600 | ~7.6 h |

The deeper probe goes on the cheaper run. Both fit the 24 h `workq` limit with
better than 2× margin, using the references' own measured rates (5.25 h and
7.6 h per 300 iterations).

## Why this matters more than its size suggests

Two things the project cannot see at *any* horizon:

- **Whether cultural divergence keeps growing.** It has not plateaued at any
  measured horizon, and no run goes far enough to find out. The divergence
  result is one of the paper's strongest, and "it was still rising when we
  stopped looking" is a weak place to leave it.
- **Whether the decline bottoms out.** The reference populations are already
  past their regime decline at 300. Nobody knows whether that is a floor, a
  trough, or a slope that keeps going.

And one that follows from both: **every population-level claim in this project
is a claim about iteration 200**, chosen before the decline was visible, and
now known by the runs themselves not to be a converged state. A reviewer who
notices that will ask what happens later, and right now there is no answer.

## Falsification

A flat or recovering `alive` fraction and a plateauing KL from base over the
further iterations. That would make iteration 300 a floor rather than a
waypoint, and the decline a transient the paper has been reading as a trend.

## The void condition — read this before trusting the curve

The reference runs trained on **2026-05 code**; this continues them on current
code. A step change at iteration 301 in perception loss, entropy or KL is
**code drift, not dynamics**, and the extension then cannot be read as one
trajectory.

This is directly checkable: compare iterations 295–300 against 301–306. Do that
first, before reading anything else off the extension. (The same drift question
is why C1 re-runs its own seed 1 rather than reusing the 05-06 originals; here
reuse is unavoidable, since extending an existing dataset is the point.)

The optimiser state does carry across: the reference checkpoints hold 912
optimiser entries, which is exactly what 8 agents × (112 LoRA tensors + 2 value
head) needs, so momentum resumes rather than restarting.

## Output goes to a new directory

Deliberately. `runs/cultural_emergence/c6_extended_horizon/<name>/` rather than
appending into the reference run's directory, so the originals stay exactly as
the existing analyses found them — several reports and figures are built on
those files. To read the two halves as one trajectory:

```bash
.venv/bin/python scripts/concat_metrics_summary.py   # see its --help for paths
```

## Submitting

```bash
.venv/bin/python submit_batch.py \
    configs/cultural_emergence/c6_extended_horizon/ \
    --account brics.a5l \
    --hf-home /lus/lfs1aip2/projects/a5l/egunn/hf_cache \
    --scripts-dir slurm_scripts/
```

`resume_from` is set in each YAML, so no `--extra-args` is needed and the whole
directory submits uniformly.

If the decline still has not bottomed out at the end, extend again from the new
checkpoint — resumption is now a proven path rather than a hopeful one.
