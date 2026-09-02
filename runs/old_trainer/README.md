# Attic: pre-population-trainer

75 runs from before 2026-04-24, when population-based training replaced the
original per-experiment trainers (commits `9b2ed9e`, *"Add population based
training"*, and `1df5560`, *"Add environment diversity"*).

## Why they cannot be cited

These runs predate the trainer that every subsequent result uses. They also
predate both context-management fixes — see
[`../no_context_management/`](../no_context_management/) and
[`../old_delimiters/`](../old_delimiters/) — so they carry those problems as
well as running a different training loop.

Concretely, they have no per-agent LoRA on a shared base, no random pairing, no
population-level metric namespacing, and no snapshot evaluation. Nothing in
their `metrics.jsonl` has the same meaning as a current run's.

## What is here

| directory | runs | what it was |
|---|---:|---|
| `negotiation_experiments/` | 19 | the model/size/LR sweep on deal-or-no-deal |
| `concordia_experiments/` | 13 | the Concordia scenario set — round-robin, haggling, labour, reality show, pub coordination, state formation |
| `negotiation_lora_shared_experiments/` | 12 | shared-base LoRA at r16 and r64 |
| `negotiation_prompt_experiments/` | 12 | character-prompt variation and role shuffling |
| `negotiation_role_experiments/` | 12 | role shuffling × learning rate × KL |
| `experiments/` | 4 | the earliest scratch runs |
| `negotiation/` | 1 | the first negotiation run in the project |

## The Concordia line ends here

`configs/concordia_experiments/` still holds 15 configs and this is the only
place any of them ever ran. The Concordia environment was superseded by the
purpose-built forum environments (`envs/forum/`), which is why the substrate
names in the cultural-emergence campaign are Study Group and Inner Loop rather
than Concordia scenarios — but that transition was never written down at the
time, and Robotic Athanor, the one Concordia scenario that survived into the
forum campaign, is also the one substrate that never worked.

## Never delete a run

These stay.
