# content_perception

*Does splitting perception loss into post content and environment
scaffolding give a health measure that silence cannot game? **No.**
Job 6312502, 2026-09-03.*

|  |  |
|---|---|
| Script | [`scripts/analyse_content_perception.py`](../../../scripts/analyse_content_perception.py) |
| Plots | [`content_perception.png`](content_perception.png), [`content_perception_scatter.png`](content_perception_scatter.png) |
| Input | `run7` checkpoints 25–300 and its extension 400–900, 4 episodes each |
| **Reads against** | **`runs/cultural_emergence/c6_extended_horizon/NOTES.md`** — the artefact this was built to repair |

## What it was for

`perc_loss` averages over every observation token, most of which is framing the
environment writes. When a population stops posting, that framing dominates, the
loss falls for free, and the classifier reads silence as coherence. The proposed
repair was to measure the loss on **post content only**, on the reasoning that
silence removes those tokens rather than making them easy.

## Headline

**The repair fails, and the pre-registered check caught it.** P1 predicted
content would be *less* predictable than scaffolding; it is *more* predictable
at **12 of 13** checkpoints. And content-only loss is at its **lowest** where
delivery is lowest — 0.78 nats/token at iteration 700, delivery 0.234.

The reason is that **degenerate output is content**. At iteration 900, Priya
Shah's post is 1143 characters of `/n?</post?</?</?</?…` against two real posts
of 120 and 123 — 82% of everything delivered. A repeating `</?` is about as
predictable as text gets, so it drags content NLL *down* exactly when the
population is worst.

So the problem was never which tokens the average runs over. **Any measure of
how well the model predicts the stream rewards degeneration**, because the
agents control what is there to predict and degenerate text is the most
predictable thing they can emit.

## The numbers

| iter | delivery | content % | NLL content | NLL scaffold | NLL all |
|---:|---:|---:|---:|---:|---:|
| 25 | 0.984 | 55.7 | 1.455 | 2.664 | 1.990 |
| 100 | 0.766 | 65.6 | 1.449 | 2.362 | 1.763 |
| 200 | 0.609 | 35.0 | 1.073 | 2.214 | 1.814 |
| 300 | 0.562 | 43.9 | 2.326 | 2.476 | 2.410 |
| 500 | 0.281 | 45.0 | **2.975** | 2.721 | 2.835 |
| 700 | 0.234 | 46.6 | **0.779** | 1.842 | 1.346 |
| 900 | **0.078** | 47.6 | 1.401 | 2.018 | 1.724 |

Correlations with delivery across all 13 checkpoints:

| quantity | r |
|---|---:|
| all tokens (`perc_loss`) | +0.01 |
| content only | −0.05 |
| scaffolding only | +0.51 |
| content share of the stream | +0.48 |

## Why scaffolding is *harder* than content

The framing is not deterministic. It carries the **speaker's name** — which of
eight agents posts next — and that is the least predictable thing in the
stream, several tokens long. "Deterministic scaffolding" was wrong: the
template is fixed, the name slot is not. Scaffolding NLL sits at 1.8–2.7
throughout, above content at nearly every point.

## A limitation that matters for reading the rest

This reconstruction is **underpowered**. At 4 episodes per checkpoint it gives
r = +0.47 between delivery and the all-token loss where the trainer's own
`metrics.jsonl`, averaging 16 episodes per iteration, gives **+0.85**; the two
measurements correlate at +0.74. The C6 artefact finding rests on the trainer's
metric and is unaffected — but the correlations in the table above should be
read as noisy, and the +0.01 for the all-token loss should **not** be taken as
contradicting the +0.85. More episodes per checkpoint would settle it.

## What this settles

- **Content-only perception loss does not repair the classifier.** It is lowest
  at lowest delivery, for the concrete reason that a degenerate loop is content
  and is maximally predictable. — *established* — this pass
- **Predictability is the wrong target for a health measure in this setting.**
  The agents control the stream, so any loss over it can be reduced by emitting
  less or by emitting something trivially repetitive. This generalises past the
  content/scaffolding split to any variant of the idea. — *likely* — same
- **The scaffolding is not deterministic**, because the speaker name is not.
  Any future split has to account for that rather than assume framing is free.
  — *established* — same
- **Delivery rate is monotone across the whole range**, 0.98 at iteration 25 to
  0.08 at 900, with no recovery anywhere. It is the one measure here that
  neither silence nor degeneration can move in the wrong direction. —
  *established* — same

## What it does not settle

- **What the health measure should be.** Delivery rate is the obvious anchor,
  but it says nothing about whether the posts are any *good* — a population
  emitting fluent nonsense at 100% delivery would score perfectly. The likely
  shape is delivery as a gate with a quality measure conditional on it, and
  the quality measure cannot be a predictability score.
- **The skill-score alternative** (`CE_θ / CE_base` on the same tokens), which
  is untested. It is more resistant than content-only loss, because a base
  model also finds a `</?` loop predictable so the ratio does not improve — but
  that argument needs measuring, not asserting.
- **Anything at more than 4 episodes per checkpoint**, or on more than this one
  run.
