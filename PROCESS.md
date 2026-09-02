# Process

How findings get recorded, so that expensive runs are not wasted and no part of
the project goes stale.

`STATE.md` is the control room. This document is how it stays true.

This process was adopted retrospectively on 2026-09-01, after five months of
runs had accumulated with their conclusions living only in analysis scripts,
figure directories and memory. The back-fill is described at the end.

---

## The documents and what each is for

| document | role | changes when |
|---|---|---|
| **`STATE.md`** | Control room. Everything we believe, how strongly, what is running, what we still do not know. Not chronological — a synthesis. | after every set of runs |
| **`runs/TIMELINE.md`** | Chronology. Every campaign in order, one line of outcome each. | when a run finishes |
| **`runs/README.md`** | Per-run index: a **map** of what exists and where. Holds no beliefs. | when a run is added or attic'd |
| **`runs/<family>/<run>/NOTES.md`** | The analysis of one run. The only place numbers and argument live. | once, when that run is analysed |
| **`runs/<family>/NOTES.md`** | The analysis of a *family* whose unit of inference is the family, not the cell — sweeps, seed replicates, matched designs. | when the family completes |
| **`runs/<analysis>/REPORT.md`** | The analysis of a question that spans runs. Written by the analysis that produced it; cites its own `.csv`s. | when that analysis runs |
| **`runs/<attic>/README.md`** | Why these runs cannot be cited. | when runs are invalidated |
| **`configs/**/*.yaml` headers** | What a run is FOR and what would falsify it — written *before* it runs. | at submission |
| **analysis script docstrings** | The same commitment for an analysis that spends no compute: what it changes, what we expect to learn, what would falsify it — written before it is run. See `scripts/tier_b/`. | at analysis |
| **`README.md`** | The machinery. How to run things and what the code does. | when code changes |

The direction of flow is one way:

> config header (a commitment) → run → `metrics.jsonl` (machine) →
> `NOTES.md` (interpretation) → `TIMELINE.md` (chronology) → `STATE.md` (belief)

Beliefs are never edited from memory. They are edited from a `NOTES.md` or a
`REPORT.md`.

### What is in git and what is not

`.gitignore` keeps `runs/**` out of git and lets `runs/**/*.md` back in. Run
data is ~28 TB — checkpoints, traces, `metrics.jsonl`, figures, and the `.csv`
tables the reports link to all stay on the filesystem. The write-ups are the
project's transmissible record and they are versioned.

A consequence to be honest about: **a `REPORT.md` links to `.csv` files that a
clone will not have.** That is deliberate — the report has to carry enough of
its own numbers to be read without them. If a report is unreadable without its
tables, the tables are doing work that belongs in the prose.

---

## When a job is submitted

Write the config header first. It must state:

1. **What this changes**, and against which run it should be read.
2. **What we expect to learn** — the specific numbers expected to move.
3. **What would falsify it** — what result would mean the idea is wrong.

This is a commitment made before the data. It is the main defence against
reading a result as confirmation of whatever we hoped.

**An analysis gets the same header**, in its script's docstring, even when it
spends no compute. Re-analysis of data already on disk is where the temptation to
fit the answer is strongest, because the answer is already knowable. Two of the
eight Tier-B analyses overturned a claim the author had already written down;
neither would have if the falsification condition had been chosen afterwards.

Add the job to the *What we are running* table in `STATE.md` **and**
`runs/TIMELINE.md`.

Submission is `submit_batch.py`, which generates one Slurm job per config:

```bash
.venv/bin/python submit_batch.py configs/<family>/ --account <project> \
    --hf-home /projects/<proj>/hf_cache
```

**Use `.venv/bin/python`, never `uv run`.** `uv run` re-syncs from
`pyproject.toml` and replaces the torch build that the hand-installed vLLM was
compiled against, silently breaking the rollout path. Rebuild with
`sbatch slurm_scripts/install_vllm.sh`; add packages with `uv pip install`.

---

## When a job finishes

**Every finished job gets analysed. No exceptions**, including crashes — a run
that died at iteration 14 still produced data, and in this project a truncated
run twice carried the headline (`social_terminal/run03`, `run05`).

1. **Plot it.**
   `.venv/bin/python scripts/plot_results.py runs/<family>` — per-run training
   curves plus the suite comparison.
2. **Classify the population.** For any forum/population run, the regime table
   is the first thing to look at:
   `.venv/bin/python scripts/plot_population_regime_figures.py`. Regimes and
   their thresholds are defined below; never hand-roll the classifier in a
   one-off script, or two runs stop being comparable.
3. **Read the traces.** `runs/<run>/traces/ckpt_NNNNNN/traces.txt`, or
   `scripts/format_ckpt_traces_for_print.py`. This is the only check that can
   find a failure mode nobody thought of — every metric counts tokens against
   categories chosen in advance, so a policy that has collapsed into a
   `</post>\n</python>` loop while still emitting a readable post scores as
   healthy. That exact case is why `degenerate` was split in two
   (`_population_figures/trace_verification_iter200.md`).
4. **Write `NOTES.md`** from `runs/_TEMPLATE_NOTES.md`. Fill in every core
   metric, even where the answer is "not measured" — a visible gap is worth
   more than a silent one.
5. **Compare against exactly one run**, named in the header. A run compared
   against nothing establishes nothing.
6. **Move it out of *What we are running*** in both `STATE.md` and
   `TIMELINE.md`, into the timeline under the day it ended.
7. **Fold what it establishes into `STATE.md`**, with confidence set by what the
   run actually showed — not by what it was hoping to show.

   **Mark what is new, and demote what was.** `STATE.md` is a synthesis, not a
   diff, so nothing about it says what changed since you last read it. Novelty
   markers carry that, and they age out on their own:

   | marker | meaning |
   |---|---|
   | 🆕 `(YYYY-MM-DD)` | added or materially changed in this update |
   | 🔶 | was 🆕 in the previous update |
   | 🔸 | was 🔶 in the update before that |
   | *(none)* | settled — three updates old or more |

   **Demote first, then add.** At the start of every `STATE.md` edit: strip 🔸,
   turn 🔶 into 🔸, turn 🆕 into 🔶 and drop its date. Only then mark the new
   entries 🆕. Doing it the other way round leaves two generations wearing the
   same marker.

   A marker moves back to 🆕 when a claim's **substance or confidence** changes
   — a *likely* promoted to *established*, a claim retracted — not when its
   wording is tidied.

   **`STATE.md` holds claims, not evidence.** Every entry is **one claim, its
   confidence, and a pointer to where the evidence lives** — and nothing else.

   | belongs in `STATE.md` | belongs in the cited document |
   |---|---|
   | the claim, in a sentence | the table it rests on |
   | how strongly we believe it | the derivation |
   | where the evidence is | the caveats and biases |
   | what it changes about how we work | the per-run numbers |

   Concretely:

   - **No data tables.** The only tables allowed are the confidence key and the
     currently-running list, which are a legend and a status board.
   - **Numbers only where the number IS the claim.** "β ≥ 0.1 is fatal at any
     KL" needs its number. "Mean perception loss went 1.29 → 1.68 → 2.48 across
     these three arms" is a result, and the claim is what it means.
   - **A bullet is at most a few lines.** A claim that cannot be stated in a few
     lines is not yet one claim.
   - **Every claim carries a confidence and a citation.** An entry with neither
     is a hunch, and it belongs under *Open questions*.

   **Put each finding in the category it belongs to. Do not add a section for
   the run.** `STATE.md` is organised by subject and its headings are standing
   topics. If a finding fits nowhere, the taxonomy is wrong: revise the
   categories and move the existing entries with them. `TIMELINE.md` already
   carries the chronology; `STATE.md` must not duplicate it.
8. **Keep open questions open.** An entry under *Open questions* states what is
   unknown and what would settle it, in a few lines. The moment it starts
   explaining what we found, the found part is a belief and belongs above.
9. **Add it to the index** in `runs/README.md`, and give its config a
   `STATUS:` line saying what became of it. A config left reading "queued as
   job X" long after that job finished is how a finished arm gets resubmitted
   by accident.
10. **If it is invalid or truncated beyond use**, move it to an attic directory
    with a README saying why. **Never delete a run.**

    **A job that produced no data at all still gets an entry** — the failure
    must be findable rather than being a silent gap. `cultural_emergence_14B/`,
    `cultural_emergence_32B/` and `cultural_emergence_qwen3_packed/` are each a
    directory whose only content is the fact that the job died before writing
    a metric, and each of those facts cost GPU-hours to learn.

---

## Regimes: the classifier every population run is read through

Assigned per (run, agent, iteration) from three metrics in `metrics.jsonl` —
`kl` (from the frozen base policy), `perc_loss` (perception loss), `entropy`:

| regime | condition |
|---|---|
| `text_degenerate` | `perc_loss ≥ 2.0` |
| `policy_collapse` | `entropy < 0.2` or `> 1.8`, with `perc_loss < 2.0` |
| `committed` | `kl ≥ 1.0`, `perc_loss < 1.7` |
| `mid_range` | `perc_loss ∈ [1.7, 2.0)` |
| `coherent_low_kl` | `kl < 1.0`, `perc_loss < 1.7` |

*alive* = coherent + committed + mid-range.

Three things about this classifier are load-bearing and were each learned the
hard way:

- **It is a descriptive partition of a continuous, bimodally-occupied space**,
  not a set of mechanistic phases. Absorption responds to KL linearly, not in a
  step at the boundary; what makes the labels non-arbitrary is that the
  covariate is bimodal, with the threshold sitting in the low-density valley.
  See `runs/cultural_emergence/_regime_threshold_sweep_3d/REBUTTAL_NOTES.md`.
- **The two failure modes are not one bucket.** `policy_collapse` agents often
  emit perfectly readable text over a policy that has narrowed to
  near-determinism. Merging them under-sells the population-size result.
- **The high-entropy branch of `policy_collapse` has never fired.** All 7
  policy-collapse agents in 220 labelled agents classify via `entropy < 0.2`;
  the highest entropy observed anywhere is 1.59. Keep the upper bound as a
  guard, but do not describe it as something we see.

---

## Core metrics: report these every time

Common across all population runs so they stay comparable.

| | why it is on the list |
|---|---|
| **regime composition at the evaluation iter** | the population-level outcome; the unit most claims are made in |
| **alive fraction** | the single number that says whether the run is usable at all |
| **per-agent `kl` from base, and its max** | how far the policy travelled; the committed/coherent axis |
| **`perc_loss`, first and last** | whether the model can still predict its world |
| **`entropy`, first and last** | whether the policy has narrowed to determinism |
| **iteration reached vs configured** | truncation is never silent |
| **PMI gap (own − none), first and last** | character absorption, where it was measured |
| **swap-vs-none differential** | absorption *specific to the agent's own character*, as opposed to generic drift |
| **cross-population trajectory KL, with its iter-25 baseline** | cultural divergence, and the proof the construction is clean |

Where an analysis spans runs, add the metric to the script in `scripts/` rather
than computing a table row inline — every run's table has to stay comparable.

---

## Standing rules

These exist because each was learned by getting it wrong.

- **Agents within a population are not independent samples.** One-way ICC
  across seeds for perception loss is 0.56–0.88, a design effect of 4.9–13.7:
  a 16-agent population is worth about 6 independent agents. Use the
  **population** as the unit of analysis, with t- or exact-binomial CIs.
- **Say whether a regime label is contemporaneous or end-of-training.** 38% of
  cross-environment agents passed through two or more regimes during the
  absorption window; labelling a mid-training measurement with an
  end-of-training regime mis-attributes it.
- **A realisation-level statistic is not a policy-level claim.** The
  "committed agents converge across seeds" reading survived three datasets and
  was still an artefact — per-character trajectory-KL contribution correlates
  +0.55 with the agent's own entropy deficit and −0.09 with actual inter-seed
  policy distance. Measure the policy directly before claiming anything about
  it.
- **Every swap or counterfactual construction needs a baseline at iter 25.**
  If the gap is already large before training has done anything, the
  construction is measuring off-distributionness, not learning. The cross-run
  history swap failed exactly this test (+1.10 nats/token at iter 25) and the
  cultural-learning measurement built on it does not stand.
- **A shrinking PMI gap is not absorption until the swapped-character control
  is run.** Own-vs-none shrinking while swap-vs-none stays flat is absorption;
  both shrinking together is drift from base.
- **Sweep any threshold before resting a claim on it**, and report the label
  stability, not only the headline.
- **Never compare runs at different iterations.** Substrates differ in how fast
  they move; state the iter with every number.
- **Truncation is never silent.** Record the iteration a run stopped at and
  what that biases. Most of this project's runs are truncated.
- **Rates with no denominator are NaN, not zero**, and every n gets stated.
- **A single run cannot rank two schemes.** The population-size story needed
  15 replicate populations before it could be stated without causal language.
- **`uv run` breaks the venv.** `.venv/bin/python`, always.

---

## Keeping it from going stale

Check these when updating `STATE.md`:

- [ ] Every job in *What we are running* is actually running. Finished ones moved.
- [ ] Every live run directory has a `NOTES.md`, or sits under a family
      `NOTES.md`, or is in an attic with a README giving the reason.
- [ ] Every claim in `STATE.md` points at a document that exists.
- [ ] Confidence levels still match the evidence — a *likely* since replicated
      should be promoted; one contradicted should be demoted or cut.
- [ ] Every config's `STATUS:` matches what actually became of that job.
- [ ] Every finished run appears in `runs/README.md`, and every attic directory
      is listed there with the reason its runs cannot be cited.
- [ ] `STATE.md` contains no data table, no bullet longer than a few lines, and
      no claim without a confidence and a citation.
- [ ] Every *Open question* is still a question. Any that has grown a partial
      answer has had the answer moved up and the residue left behind.
- [ ] Novelty markers were demoted **before** new ones were added.
- [ ] `STATE.md`'s headings are still subjects, not events.
- [ ] `runs/README.md` still states no beliefs.

---

## When a belief turns out to be wrong

Correct it in `STATE.md` and say so in the `NOTES.md` or `REPORT.md` that
overturned it. Do not quietly delete the old claim — that we believed it, and
what changed, is part of the record. Two of this project's headline readings
have already been reversed by later analysis (the regime reinterpretation of
trajectory KL; the LoRA-capacity explanation of Robotic Athanor), and knowing
which were reversed is how the confidence levels stay honest.

---

## About the retrospective back-fill

The `NOTES.md` files written on 2026-09-01 were reconstructed from
`metrics.jsonl`, `config.json`, the analysis reports under `runs/cultural_emergence/_*`,
and the config headers — not from notes taken at the time. They are therefore
weaker than notes written when a run finished, in one specific way: **they
record what the data shows, not what was expected before the run.** A
back-filled note cannot tell you whether a result confirmed or surprised its
author, which is exactly the information a pre-registered header exists to
preserve.

Back-filled notes say so in their status line. Notes written under this process
from now on do not need to.
