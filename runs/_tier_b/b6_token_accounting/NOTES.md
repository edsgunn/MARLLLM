# B6 — token accounting audit

*Does `thinking_frac` measure private deliberation?*

|  |  |
|---|---|
| Script | [`scripts/tier_b/b6_token_accounting_audit.py`](../../../scripts/tier_b/b6_token_accounting_audit.py) |
| Inputs | checkpoint traces + `metrics.jsonl` for 7 long forum runs, 2 checkpoints each |
| Ran | 2026-09-01, no compute |

## What it was for

`REWRITE.md` §3.5 read a rising `thinking_frac` as agents moving generation into
a private channel. That reading needs `mean_thinking_tokens` to count tokens
inside `<think>…</think>`. The audit checks whether it does.

## Headline

**It does not, and the reading it supported is wrong.**

`marlllm/population.py:721` computes `thinking_tokens = max(0, n_act -
post_tokens)`, where `post_tokens` is the *re-encoded* text extracted from
`<post>` blocks. It is a residual, not a measurement. The source comment says so
for one of its components: `<python>` code "shows up in (n_act - post_tokens)
along with thinking". A turn containing no well-formed `<post>` at all scores as
100% "thinking".

**There are zero `<think>` tags in any checkpoint of any forum run inspected** —
14 checkpoints across 7 runs, early and late. The forum substrate never produces
private deliberation. What `thinking_frac` rises with is the share of action
tokens outside a well-formed post block.

The arithmetic confirms the definition: `act == think + post` exactly in 14/14
logged records.

## Core metrics

`<post>` opens per `</post>` close, early checkpoint → late:

| run | early | late |
|---|---:|---:|
| `run12_32agent_ashbourne_gc` | 1.78 | **87.03** |
| `run8_16agent_margin_notes` | 1.77 | **28.82** |
| `run7_8agent_margin_notes` | 1.50 | 6.88 |
| `run8_16agent_ashbourne_gc` | 1.72 | 4.05 |
| `run8_16agent_strathearn_server` | 1.69 | 2.68 |
| `run7_8agent_inner_loop_post_format` | 1.78 | **0.95** |
| `run7_8agent_ashbourne_gc` | 1.47 | **0.32** |

Two pathologies, not one. Most runs emit `<post>` repeatedly without closing it;
`run7_8agent_ashbourne_gc` does the reverse and emits three closes per open —
the `</post>\n</python>` loop already documented qualitatively in
[`../../cultural_emergence/_population_figures/trace_verification_iter200.md`](../../cultural_emergence/_population_figures/trace_verification_iter200.md).
`inner_loop_post_format` is the only run that *improves*, to 0.95, which is the
post-format change working.

A representative late generation shows what the breakdown looks like — a closing
tag with a curly quotation mark: `…mr davies.</post“`.

## What this settles

- **`thinking_frac` and `mean_thinking_tokens` do not measure private
  deliberation on any forum substrate.** Zero `<think>` tags in 14 checkpoints.
  — *established*
- **Structural tag emission degrades badly over training**, in both directions,
  and by up to 87 opens per close. — *established*
- **The §3.5 private-channel reading is retracted.** — *established*

## What it does not settle

Whether the degradation matters — that is [B4](../b4_public_channel/NOTES.md),
which uses the environment's own post accounting instead.

## Status

`CLOSED`. **Do not use `thinking_frac` as a private-deliberation measure.** For
a non-forum substrate with a genuine `<think>` channel the audit would need
repeating; the Qwen3 accounting is separately unverified (see B4).

---
*Analysed 2026-09-01. Machine output stays local.*
