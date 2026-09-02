# run05_8agent_7B_ashbourne_school

*social terminal — the turn-ending upgrade, on pre-fix code*

|  |  |
|---|---|
| Config | [`configs/social_terminal/run05_8agent_7B_ashbourne_school.yaml`](../../../configs/social_terminal/run05_8agent_7B_ashbourne_school.yaml) |
| Model | `Qwen2.5-7B-Instruct` |
| Population | 8 agents on `ashbourne_school` |
| Settings | kl_coef 0.05, β 0.01, lr 5e-06, LoRA r 64 |
| Ran to | iteration 48 of 300 — **truncated** · 1 checkpoints |
| **Differs from** | **`run03` — information verbs now end the turn, sampler stop-strings added, tags inside `<think>` no longer fire, and framework-inserted turn-close scaffolding is tagged OBS rather than ACT** |

## What it was for

The first run of the observation-principle upgrade: only model-generated tokens
count as actions, and any verb that queues a perception ends the turn so the
actor cannot invent what it has not yet been told.

## Headline

**It ran 48 iterations on incomplete code.** `<go>` and the self-closing
`<look/>` / `<where/>` forms were not yet generation stop-points, so the policy
could still walk into a room and confabulate its contents — the exact failure
the upgrade was written to remove. The upgrade's intended effect *is* visible in
what data there is: generations shortened markedly against `run02`.

## Core metrics

Population means over the first and last 10% of logged iterations (blocks of 4 of 48):

| metric | first block | last block |
|---|---:|---:|
| perception loss | 1.547 | 1.397 |
| KL from base | 0.001 | 0.043 |
| policy entropy | 0.519 | 0.569 |

Regime composition (paper thresholds; see `PROCESS.md`):

| iteration | n | alive | composition |
|---:|---:|---:|---|
| ~25 | 8 | 1.00 | coherent_low_kl 8 |

## What this settles

- **The turn-ending change shortens generations.** — *provisional*, 48 iterations

## What it does not settle

Whether the upgrade works, since the version that ran was incomplete. `run06` is
the run that answers this.

## Status

`SUPERSEDED` by `run06`, which carries the complete fix. Do not cite as the upgrade.

---
*Analysed retrospectively on 2026-09-01 from `metrics.jsonl` and `config.json`; see the note in `PROCESS.md` on what a back-filled note cannot tell you. Machine output stays local.*
