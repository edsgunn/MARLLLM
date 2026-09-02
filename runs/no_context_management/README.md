# Attic: no chat-format context management

67 runs from before 2026-04-27 — five negotiation and population families
predating commit `e760344`, *"Add context formatter to delineate actions from
observations"*.

## Why they cannot be cited

`marlllm/context_formatter.py` states the problem these runs have, in its own
docstring:

> Without formatting, OBS and ACT tokens are concatenated as a flat stream with
> no delimiter — the model has no in-context signal indicating which tokens it
> generated versus which came from the environment.

So in every run here the model read its context as an undifferentiated token
stream, outside the chat framing it was pretrained on. The formatter wraps each
observation block in the model's user-turn tokens and primes the context with
the assistant-turn start before each `act()`, in both the live rollout buffer
and the training forward passes.

This is the single largest change in the project's history. A model reading
out-of-distribution context is not doing a worse version of the current
experiment; it is doing a different one.

## What is here

| directory | runs |
|---|---:|
| `negotiation_experiments/` | 20 |
| `negotiation_lora_shared_experiments/` | 12 |
| `negotiation_prompt_experiments/` | 12 |
| `negotiation_role_experiments/` | 12 |
| `population_experiments/` | 11 |

Each family was re-run after the fix; the re-runs live under
[`../old_delimiters/`](../old_delimiters/) and were then themselves superseded.

## Never delete a run

These stay.
