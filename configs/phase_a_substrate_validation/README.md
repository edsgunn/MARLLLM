# Phase A — Substrate Validation for CCSM

Configs for the minimum viable experiment from the Phase A spec.  The single
claim being tested is:

> Character-Conditioned Surprise Minimisation, applied to a pretrained
> language model in a multi-agent text environment, produces non-trivial
> agentic behaviour that (a) differs from the pretrained baseline,
> (b) varies with the character prompt, and (c) improves on environment-
> relevant external metrics over training.

## The matrix

Each condition is run twice — once under the **cooperative** character prompt
and once under the **competitive** character prompt.  The pair of runs is the
prompt-swap (§2.3 condition 5), which is the most important diagnostic for the
substrate-validation claim.

| # | Condition (file prefix)   | What it tests                                                            |
|---|---------------------------|---------------------------------------------------------------------------|
| 1 | `01_pretrained_baseline_` | No training — reference behaviour for the pretrained model.              |
| 2 | `02_ccsm_focal_`          | Full unified loss, asymmetric (focal vs frozen pretrained partner).      |
| 3 | `03_perception_only_`     | NTP on OBS only — α_act = α_val = 0.  Isolates the policy-gradient role. |
| 4 | `04_action_only_`         | REINFORCE on ACT only — α_perc = 0.  Isolates the perception role.       |
| 5 | `05_symmetric_ccsm_`      | Full loss, both agents trained simultaneously.  Supporting condition.    |
| 6 | `06_qwen_7b_lora_`        | Scale datapoint at 7B with LoRA — only one prompt, run if compute fits.  |

`02_ccsm_focal_*_seed2.yaml` provides a seed replicate of the headline
condition for variance estimation.

## How to run

Training conditions (everything except condition 1):

```bash
uv run python submit_batch.py \
    configs/phase_a_substrate_validation/ \
    --account <project> --hf-home /lus/.../hf_cache
```

Pretrained-baseline evaluation (condition 1) does not train; submit it
through the same dispatcher — `script: evaluate_negotiation` in the YAML
points the runner at the eval entry point:

```bash
uv run python submit_batch.py \
    configs/phase_a_substrate_validation/01_pretrained_baseline_*.yaml \
    --account <project>
```

After every training run finishes, score it with the same evaluator and
then run the prompt-conditional comparison.  Two helper invocations:

```bash
# Capability metrics for a trained checkpoint.
uv run python evaluate_negotiation.py \
    --model Qwen/Qwen2.5-1.5B \
    --checkpoint runs/phase_a_substrate_validation/02_ccsm_focal_cooperative/checkpoints/final.pt \
    --prompt-0 "$(yq .prompt_0 configs/phase_a_substrate_validation/02_ccsm_focal_cooperative.yaml)" \
    --prompt-1 "$(yq .prompt_1 configs/phase_a_substrate_validation/02_ccsm_focal_cooperative.yaml)" \
    --output runs/phase_a_substrate_validation/02_ccsm_focal_cooperative/eval_final.json

# Behavioural divergence + cross-character surprise (Phase A §3.2).
uv run python analyze_prompt_conditional.py \
    --model Qwen/Qwen2.5-1.5B \
    --checkpoint-a runs/phase_a_substrate_validation/02_ccsm_focal_cooperative/checkpoints/final.pt \
    --checkpoint-b runs/phase_a_substrate_validation/02_ccsm_focal_competitive/checkpoints/final.pt \
    --prompt-a "$(yq .prompt_0 configs/phase_a_substrate_validation/02_ccsm_focal_cooperative.yaml)" \
    --prompt-b "$(yq .prompt_0 configs/phase_a_substrate_validation/02_ccsm_focal_competitive.yaml)" \
    --output runs/phase_a_substrate_validation/prompt_conditional_post.json
```

The same `analyze_prompt_conditional.py` invocation, run with no checkpoints,
produces the **pre-training** sensitivity baseline.  Phase A §7 requires this
to be inspected on day 2 before committing to the full matrix.

## Day-2 fast check

Before kicking off the matrix, check that the pretrained 1.5B model has
*any* prompt sensitivity at all on this task:

```bash
uv run python analyze_prompt_conditional.py \
    --model Qwen/Qwen2.5-1.5B \
    --prompt-a "<cooperative>" \
    --prompt-b "<competitive>" \
    --n-episodes 32 \
    --output runs/phase_a_substrate_validation/baseline_prompt_sensitivity.json
```

If the divergence statistics are essentially zero, the substrate is too small
and the week's plan needs to change (spec §7).

## Pre-registered failure modes

See the spec §4.  The metrics emitted by `evaluate_negotiation.py` and
`analyze_prompt_conditional.py` map directly to the diagnostic axes.
