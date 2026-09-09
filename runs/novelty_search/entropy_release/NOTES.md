# entropy_release

*novelty search — release the entropy penalty and see whether the behaviour is
interesting or noisy TV. **The family is the unit of inference.*** **RUNNING.**

|  |  |
|---|---|
| Configs | [`configs/novelty_search/entropy_release/`](../../../configs/novelty_search/entropy_release/) |
| Model | `Qwen2.5-7B-Instruct`, LoRA r64, 8 agents on `study_group_ashbourne_gc` |
| Grid | `beta` ∈ {0.03, 0.06, 0.10} × seeds {1, 2} |
| Held fixed | `return_sign` **−1.0**, `kl_coef` **0.05**, lr 5e-6, 16 episodes/iter, 300 iterations, γ 0.99 |
| Submitted | 2026-09-09, jobs 6434469–6434474, 06:00:00 each |
| **Differs from** | **[`../c13_novelty_search_8agent_ashbourne_gc_seed1`](../NOTES.md)** — `beta` only, from its 0.01. |

## The question

Under novelty search, is entropy the **action arm asks for** structured where
entropy a **uniform bonus imposes** is not?

## Why the previous attempt could not answer it

[`entropy_anchor_ridge`](../../cultural_emergence/entropy_anchor_ridge_8agent_ashbourne_gc/NOTES.md)
paired a large `beta` with a large `kl_coef`. Its headline is retracted, and it
was the wrong instrument twice over:

1. **It was surprise minimisation** (`return_sign` +1.0), not novelty search.
2. **`kl_coef` 2.0–5.0 pinned the policy.** KL from base ended at 0.02–0.46
   against 1.20–1.95 at `kl_coef` 0, and end entropy never exceeded 1.36 — while
   the regime where entropy opens up is 1.5–7.0. The anchor removed the thing
   being studied.
3. And structurally: **`beta` and `kl_coef` act through roughly their ratio**
   (ρ = +0.831, p = 7.8 × 10⁻¹¹, n = 21), so raising both together cannot reach
   a new regime by construction.

This family therefore holds `kl_coef` at the novelty reference's 0.05 and moves
`beta` alone. **Nothing is pinned: KL from base is free.**

## What the minimisation ladder already shows

The 28 regularisation runs were read on trace *content* for the first time on
2026-09-09. Under minimisation, as end entropy rises:

| end entropy | what the transcripts look like |
|---|---|
| 0.9–1.1 | coherent throughout |
| 1.5–1.8 | **mixed within one episode** — some agents holding coherent mathematics while others emit `beat, beat, beat! beatrice, absolute beatrice` or a bare `</?</?</?` loop |
| ~3.4 | coherent **opening**, noise **tail** — real content (`modular arithmetic properties with small primes`) then hundreds of repeated emoji |
| 6.6–7.0 | word salad — `sweet baked contentious oats propaganda zipper bepace correlcusAuthenticate` |

So under minimisation the answer is **both, at different points, and the
transition is compositional rather than global** — openings survive longest,
tails go first, and agents collapse individually rather than together. The
usable window is around 1.1–1.8 and no run was ever aimed at it.

One thing inside the collapse is worth more than the collapse: at end entropy
6.6–7.0 some agents comment on their peers' degeneration — *"can someone tame
this beast lang mess?"*, *"figured since everyone else babbled cryptograms"*.
Coherence enough to observe that the channel has become noise.

## Why the sign changes the question

Under minimisation the two arms agree about entropy — the action arm wants
predictable futures and a narrow policy delivers them — so entropy collapses
unless `beta` forces it open, and what `beta` forces open is unstructured.
Under maximisation the action arm is **already paid** for unpredictable futures
and raises entropy on its own, 8 of 8 arms above their own reference. Whether
that entropy is structured is the whole question, and it is only askable here.

## Pre-registered readings

| | criterion |
|---|---|
| **Prediction** | the collapse threshold sits at a **lower** `beta` than under minimisation, since the action arm already pushes the same way — the novelty reference ends at entropy 0.97/0.98 against minimisation's 0.80 at identical `beta` |
| **Interesting** | end entropy in 1.3–2.5 with `stopword` ≥ 0.28, or in 0.10–0.28 and confirmed coherent **by reading** |
| **Noisy TV** | `stopword` < 0.10 with long non-repetitive posts |
| **The loop** | high delivery with short posts and `repeat` > 0.2 — the artefact that scored delivery 0.926 in the size sweep |

## How it must be read

On the traces, with [`scripts/analyse_novelty_traces.py`](../../../scripts/analyse_novelty_traces.py),
and specifically on `stopword`. **Every other trace measure in that script was
fooled by high-entropy collapse**: at end entropy 6.59 a word-salad population
scored 15 posts, 724 mean characters, `repeat` 0.00 and `frag` 0.00.

`stopword` is a **screen, not a verdict**. Below 0.10 is reliably salad; 0.28
and above is coherent; **between those it must be read by hand**, because dense
technical prose scores 0.118 while genuine salad scores 0.139. Do not report a
cell in that band without quoting from it.

Do not read this family on `alive`, and do not read delivery without a
repetition measure beside it.

## Status

`RUNNING`. Nothing to cite yet.
