# Attic: special tokens in the wrong positions

79 runs from between 2026-04-27 and 2026-04-30 — after the context formatter was
added but before commit `5fa0b3f`, *"Fix special token positions and task
formatting"*.

## Why they cannot be cited

These runs have chat-format context management, so they are not affected by the
problem that invalidates [`../no_context_management/`](../no_context_management/).
But the special tokens were placed wrongly, so the boundaries between the
character prompt, each observation block and each action span did not fall where
the model's chat template expects them. Task formatting changed in the same
commit.

The consequence is the same in kind and smaller in degree: the context the model
read during rollout, and the context the training forward pass reconstructed,
are not the ones any current run produces. Their loss values are not on the same
scale as anything since.

## What is here

| directory | runs |
|---|---:|
| `negotiation_experiments/` | 20 |
| `negotiation_lora_shared_experiments/` | 12 |
| `negotiation_prompt_experiments/` | 12 |
| `negotiation_role_experiments/` | 12 |
| `negotiation_population_experiments/` | 11 |
| `negotiation_entropy_kl_ablation/` | 9 |
| `phase_a_substrate_validation/` | 3 |
| `02_ccsm_focal_competitive/` | 1 |

## What this cost

The negotiation families were run three times — once before context management,
once with broken delimiters, and once after the fix — and the third pass is the
only one that can be cited. That is roughly 150 jobs on two framework bugs, both
of which would have been caught by reading a single rollout trace against the
model's chat template before submitting a batch.

That is why `PROCESS.md` puts *read the traces* on the list for every finished
job, and why the config header has to say what the run is being read against.

## Never delete a run

These stay.
