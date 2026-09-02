# Attic: `prompt_as_observation` was off

Six cultural-emergence runs from before 2026-05-03, when the character prompt
prefix was typed `PAD` rather than `OBS`.

## Why they cannot be cited

`marlllm/config.py` now defaults `prompt_as_observation: True`, and
`marlllm/trainer.py:505` and `marlllm/population.py:1570` type the character
prompt accordingly. `PAD` is excluded from every loss term. So in these runs the
agent was **never trained to predict its own character prompt**, and perception
loss was computed over a strictly smaller set of tokens than in every run since.

That makes their perception loss numerically incomparable with anything current
— it is a different quantity, not a worse measurement of the same one — and it
removes the mechanism the campaign's central result depends on. Character
absorption is measured as the shrinking predictive value of the character
description over the agent's own actions; with the prompt outside the loss, the
thing being absorbed was not in the objective.

Every run in this attic has `prompt_as_observation` absent from its
`config.json`; every current run has it `True`.

## What is here

| directory | what it was |
|---|---|
| `run1_2agent_seed1`, `run1_2agent_seed1_kl05` | the original 2-agent Robotic Athanor pair, with and without a KL anchor at 0.5 |
| `run2_8agent_seedA` | the first 8-agent population |
| `run4_2agent_7B` | the first 7B run |
| `limited_run1`, `too_short_context` | exploratory; the names record their own verdicts |
| `calibration` | the Day-1 calibration scripts' output |

## Never delete a run

These stay. They are the record of the campaign's first three weeks, and the
reason the perception-loss scale changed under everything.
