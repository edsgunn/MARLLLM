# sanctioning

*Rungs 0–1 of the sanctioning measurement framework, run over the two confirmed
five-seed forum families.*

|  |  |
|---|---|
| Code | [`sanctioning/`](../../sanctioning/) — the pipeline and its design commitments |
| Config | [`configs/sanctioning.yaml`](../../configs/sanctioning.yaml) |
| Corpus | `ashbourne_gc` (`run8_16agent_7B_study_group_ashbourne_gc` + seeds 2–5) and `margin_notes` (`run7_8agent_7B_margin_notes` + seeds 2–5) |
| Judge | LLM per-event labelling only; all "is this a norm" logic is deterministic aggregation |
| Ran | `extract` (6,403 events), `rung0` (79-row table + figures), `cluster` (4,662 assignments), `label` (199 hand labels) |
| Not run | `validate` — and `rung1` is gated on it |

## What it was for

Reducing *"do cultural norms emerge?"* to something measurable. A norm, in
Leibo's theory of appropriateness, is a stable mapping from a behaviour-in-context
to a consistent sanctioning response by others. So:

- **Rung 0** — does sanctioning occur at all? Base rate of non-neutral reactions
  per iteration, approval and disapproval reported separately.
- **Rung 1** — is sanctioning *targeted* rather than mood? Does the same
  behaviour type reliably draw the same valence?

The framework's fourth design commitment is that the judge is an instrument and
must be validated before Rung-1 numbers are trusted, so `rung1` refuses to run
without a passing gate report.

## Headline

**Rung 0 answers its question and the answer is lopsided: these populations
approve constantly and barely disapprove.** Across the 79 (run, iteration)
cells, approval runs at 0.172–1.344 reactions per post (median 0.656) while
disapproval spans 0.000–0.172 with a median of **0.062** — roughly one
disapproving reaction per sixteen posts, against ten approving ones.

**Training does not move it.** Mean disapproval rate by iteration is 0.069 at
iteration 0, before any training, and 0.071, 0.077, 0.057, 0.079, 0.073, 0.041
at iterations 25 through 150. All ten iteration-0 cells are already in the same
band (0.031–0.109). The asymmetry is the base model's, not something these
populations learned.

The framework's own caveat anticipated this: "Disapproval ≈ 0 is a result... the
chatbot-base finding, not a pipeline failure." The data says it more sharply —
it is not merely a low rate, it is a rate with no trend.

**Rung 1 has not been run**, because the judge has not been validated. 199 hand
labels exist; `runs/sanctioning/validation/` is empty.

## What this settles

- **Sanctioning is detectable in these transcripts at high volume.** 6,403
  judged posts, 4,662 clustered trigger descriptions. — *established*
- **Disapproval runs at a median of 0.062 per post against approval's 0.656**,
  in both substrates and at every iteration sampled. — *established*
- **The asymmetry is a property of the base model, not an outcome of training.**
  It is at its iteration-0 value throughout, with no trend across 300
  iterations. — *likely*; the iteration-0 cells were not analysed separately at
  the time
- **A norm requires a sanctioning response that can go either way.** With
  disapproval at the floor, the Rung-1 question — does the same behaviour draw
  the same valence — has almost no valence variation to work with. —
  *suspected*; this is a reading of the Rung-0 table, not a measured result

## What it does not settle

- **Rung 1, entirely.** The gate is not passed, so the targeting question is
  open. That is a deliberate refusal by the pipeline, not an omission.
- **Whether an RLHF'd base model can sanction at all.** Disapproval being rare in
  a Qwen2.5-Instruct forum is unsurprising, and nothing here separates "the
  model will not disapprove" from "there is nothing in these transcripts to
  disapprove of".
- **Rungs 2–4** — cross-sanctioner consistency, second-order sanctioning,
  evolution — are out of scope by design, though the event schema retains
  `target_agent_id`, `agent_id` and `iteration` so they are pure aggregation
  additions later.

## Status

`OPEN`. Rung 0 is complete and citable. The next step is `validate`, which is
cheap, needs no GPU, and unblocks Rung 1. See the open questions in `STATE.md`.

---
*Analysed retrospectively on 2026-09-01 from `rung0_table.csv`, `events.jsonl`
and the pipeline's own README. Machine output stays local.*
