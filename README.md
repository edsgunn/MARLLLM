# MARLLLM

A framework for training language models through multi-agent interaction using **Character-Conditioned Surprise Minimisation (CCSM)** — a reward-free training objective derived from the Action Perception Divergence framework (Hafner et al., 2022).

Agents learn by minimising the KL divergence between the trajectories they actually produce and the trajectories their own model predicts. A **character prompt** is the sole specification of desired behaviour; no reward function is needed.

The framework uses [PettingZoo](https://pettingzoo.farama.org/) as its environment interface. Any PettingZoo-compatible environment works without modification.

---

## Contents

- [Installation](#installation)
- [Quickstart](#quickstart)
- [The Algorithm](#the-algorithm)
- [Training Scripts](#training-scripts)
  - [train.py — single-agent counting](#trainpy--single-agent-counting)
  - [train_negotiation.py — two-agent deal-or-no-deal](#train_negotiationpy--two-agent-deal-or-no-deal)
  - [train_concordia.py — concordia multi-agent simulation](#train_concordiapy--concordia-multi-agent-simulation)
- [Running Experiments](#running-experiments)
  - [Config files](#config-files)
  - [Submitting to SLURM](#submitting-to-slurm)
  - [Resuming jobs](#resuming-jobs)
- [Inspecting Results](#inspecting-results)
  - [Trace files](#trace-files)
  - [Regenerating traces](#regenerating-traces)
  - [Regenerating traces in bulk via SLURM](#regenerating-traces-in-bulk-via-slurm)
  - [Plots](#plots)
  - [Concatenated reports](#concatenated-reports)
- [Core Abstractions](#core-abstractions)
- [Configuration Reference](#configuration-reference)
- [Writing a New Environment](#writing-a-new-environment)
- [Writing a New Agent](#writing-a-new-agent)
- [Token Type Mask](#token-type-mask)

---

## Installation

Requires Python ≥ 3.11.

```bash
git clone https://github.com/you/MARLLLM
cd MARLLLM
uv sync          # creates .venv and installs all dependencies
```

Or install as a library into another project:

```bash
uv add git+https://github.com/you/MARLLLM
```

---

## Quickstart

```bash
# Run the built-in counting task demo (GPT-2 on CPU)
uv run python train.py

# Two-agent negotiation with a larger model on GPU
uv run python train_negotiation.py \
  --model Qwen/Qwen2.5-1.5B \
  --device cuda \
  --dtype bfloat16 \
  --iters 500 \
  --output-dir runs/my_first_run
```

Output is written to `--output-dir`:

```
runs/my_first_run/
  config.json           # full hyperparameter snapshot
  experiment_config.yaml  # copy of the YAML config, if one was used
  system_info.json      # hardware, package versions, git commit
  train.log             # timestamped debug log
  metrics.jsonl         # one JSON record per iteration
  checkpoints/
    iter_000050.pt
    iter_000100.pt
    latest.pt           # symlink to most recent
  traces/
    iter_000010.txt     # decoded episode trace (written every log_every iters)
    iter_000020.txt
    ...
```

Resume a killed job:

```bash
uv run python train_negotiation.py \
  --config configs/negotiation_role_experiments/01_no_shuffle_lr3e5_kl0.yaml \
  --resume
```

---

## The Algorithm

CCSM is derived in full in [`surprise_minimisation_derivation.md`](surprise_minimisation_derivation.md). The short version:

A trajectory interleaves **observation tokens** (produced by the environment, type `σ=1`) and **action tokens** (sampled by the model, type `σ=0`). The loss has two terms that emerge from a single KL objective:

**Perception loss** — next-token prediction on observation tokens:
$$\mathcal{L}^{\text{perc}} = -\frac{1}{N_{\text{obs}}} \sum_{t:\,\sigma_t=1} \ln p_\theta(x_t \mid c, x_{<t})$$

**Action loss** — REINFORCE weighted by future observation surprise:
$$\mathcal{L}^{\text{act}} = -\frac{1}{N_{\text{act}}} \sum_{t:\,\sigma_t=0} \ln p_\theta(x_t \mid c, x_{<t}) \cdot \hat{A}_t \;-\; \beta \cdot H[p_\theta(\cdot \mid c, x_{<t})]$$

where the advantage $\hat{A}_t = -(G_t - V_\phi)$ and the intrinsic return $G_t = \sum_{s>t} \sigma_s \gamma^{s-t}(-\ln p_\theta(x_s \mid c, x_{<s}))$ is the discounted future observation surprise. No extrinsic reward. The character prompt $c$ is the only specification of what the agent should do.

---

## Training Scripts

### `train.py` — single-agent counting

Trains one agent on `CountingEnv`: the agent must output the next integer in sequence. Useful for sanity-checking the framework and for quick iteration on hyperparameters.

```
uv run python train.py [OPTIONS]

Options:
  --config PATH             YAML experiment config (all keys map to CLI flags;
                            explicit CLI args override config values)
  --model MODEL             HuggingFace model ID or local path (default: gpt2)
  --device DEVICE           PyTorch device: cpu, cuda, cuda:0, mps (default: cpu)
  --dtype DTYPE             float32 | bfloat16 | float16 | auto (default: auto)
  --device-map DEVICE_MAP   Pass 'auto' to shard across GPUs via accelerate
  --iters N                 Training iterations (default: 100)
  --rollouts N              Episodes collected per iteration (default: 8)
  --lr LR                   AdamW learning rate (default: 3e-5)
  --max-count N             CountingEnv max_count target (default: 10)
  --log-every N             Print metrics every N iterations (default: 10)
  --checkpoint-every N      Save checkpoint every N iterations (default: 100)
  --output-dir PATH         Directory for logs, metrics, checkpoints (default: runs/default)
  --resume                  Resume from checkpoints/latest.pt in output-dir
  --kl-coef FLOAT           KL penalty vs frozen reference model (default: 0.0)
  --seed N                  Random seed (default: 42)
  --prompt TEXT             Character prompt for agent_0
```

**Examples:**

```bash
# Quick CPU smoke test
uv run python train.py

# GPU run with a larger model
uv run python train.py \
  --model Qwen/Qwen2.5-3B \
  --device cuda \
  --dtype bfloat16 \
  --iters 500 \
  --rollouts 16 \
  --output-dir runs/qwen3b-counting

# Resume after preemption
uv run python train.py --output-dir runs/qwen3b-counting --resume
```

---

### `train_negotiation.py` — two-agent deal-or-no-deal

Trains two agents on `DealOrNoDealEnv` (Lewis et al. 2017): agents negotiate over books, hats, and balls with private valuations. Each agent's utterance becomes the other agent's observation. A deal is reached when both agents submit compatible allocation proposals.

```
uv run python train_negotiation.py [OPTIONS]

Model options:
  --model MODEL             HuggingFace model for both agents (default: gpt2)
  --model-0 MODEL           Override model for agent_0 only
  --model-1 MODEL           Override model for agent_1 only
  --dtype DTYPE             float32 | bfloat16 | float16 | auto
  --device DEVICE           Default device for both agents
  --device-0 DEVICE         Device for agent_0 (overrides --device)
  --device-1 DEVICE         Device for agent_1 (overrides --device)
  --device-map DEVICE_MAP   Pass 'auto' to shard a large model across GPUs
  --attn-impl IMPL          attn_implementation for from_pretrained
                            (use 'eager' for GPT-2 and older architectures)
  --compile                 torch.compile() the backbone (~20-40% speedup
                            after 1-2 iteration warm-up on Ampere/Hopper)

Weight-sharing options (pick at most one):
  --shared-weights          Both agents use the same model weights. Gradients
                            from both trajectories update the same parameters.
  --lora-shared-base        One frozen backbone, two independent LoRA adapters.
                            Requires --lora-r > 0. Halves backbone VRAM vs two
                            full independent models.

LoRA options:
  --lora-r N                LoRA rank. 0 = full fine-tuning (default: 0)
  --lora-alpha N            LoRA alpha scaling factor (default: 16)
  --lora-modules NAMES      Comma-separated target module names. None = PEFT
                            auto-detect (q_proj,v_proj for most architectures)

Environment options:
  --dialogue-turns N        Max total dialogue turns per episode (default: 10)
  --token-budget N          Tokens each agent generates per turn (default: 64)
  --max-episode-tokens N    Token budget for the full episode (default: 1024)
  --role-shuffle            Randomly assign which agent goes first each
                            episode, balancing first-mover pressure

Training options:
  --iters N                 Training iterations (default: 500)
  --rollouts N              Episodes per iteration (default: 8)
  --lr LR                   AdamW learning rate (default: 3e-5)
  --kl-coef FLOAT           KL penalty vs frozen reference model (default: 0.0)
  --grad-accum N            Gradient accumulation steps (default: 8).
                            Splits each batch into N micro-batches to reduce
                            peak VRAM without changing effective batch size.
  --gradient-checkpointing  Recompute activations during backward to trade
                            ~30% compute for significant VRAM savings
  --seed N                  Random seed (default: 42)
  --log-every N             Print metrics every N iterations (default: 10)
  --checkpoint-every N      Save checkpoint every N iterations (default: 100)
  --output-dir PATH         Directory for logs, metrics, checkpoints

Prompt options:
  --prompt-0 TEXT           Character prompt for agent_0
  --prompt-1 TEXT           Character prompt for agent_1

Misc:
  --config PATH             YAML experiment config
  --resume                  Resume from checkpoints/latest.pt in output-dir
```

**Examples:**

```bash
# Minimal run on two GPUs (agent_0 → cuda:0, agent_1 → cuda:1 automatically)
uv run python train_negotiation.py \
  --model Qwen/Qwen2.5-1.5B \
  --dtype bfloat16

# Shared weights — single model, both agents' gradients update it
uv run python train_negotiation.py \
  --model Qwen/Qwen2.5-1.5B \
  --dtype bfloat16 \
  --shared-weights \
  --role-shuffle \
  --output-dir runs/shared-weights-shuffle

# LoRA shared base — one frozen 8B backbone, two independent rank-16 adapters
# (fits on a single GH200 at ~16 GB vs ~32 GB for two full models)
uv run python train_negotiation.py \
  --model Qwen/Qwen3-8B \
  --dtype bfloat16 \
  --lora-shared-base \
  --lora-r 16 \
  --kl-coef 0.1 \
  --output-dir runs/lora-shared-8b

# Distinct character prompts, separate models
uv run python train_negotiation.py \
  --model Qwen/Qwen2.5-1.5B \
  --dtype bfloat16 \
  --role-shuffle \
  --kl-coef 0.1 \
  --prompt-0 "You are Marcus, a wily merchant who never reveals what he wants." \
  --prompt-1 "You are Sophia, a pragmatic trader who always proposes first." \
  --output-dir runs/marcus-vs-sophia

# Resume after preemption
uv run python train_negotiation.py \
  --config configs/negotiation_role_experiments/08_shuffle_lr1e5_kl05.yaml \
  --resume
```

---

### `train_concordia.py` — Concordia multi-agent simulation

Trains agents within a [Concordia](https://github.com/google-deepmind/concordia) generative agent simulation. The Concordia engine runs in a background thread; CCSM training hooks into it via a stub agent that blocks until the Trainer produces an action.

Usage is analogous to `train_negotiation.py`. See the script's `--help` and the configs in `configs/concordia_experiments/` for details.

---

## Running Experiments

### Config files

All CLI flags can be set in a YAML config file. The file is passed with `--config`; explicit CLI flags override any value from the file.

```yaml
# configs/my_experiment.yaml
name: my_experiment
description: >
  Brief description of what this tests.
script: train_negotiation        # which training script to use

slurm:                           # resource directives for submit_batch.py
  time: "08:00:00"
  nodes: 1
  gpus_per_node: 1
  cpus_per_gpu: 72
  partition: workq

# All other keys map directly to CLI flag names (dashes or underscores)
model: Qwen/Qwen2.5-1.5B
dtype: bfloat16
iters: 500
rollouts: 8
lr: 1.0e-5
dialogue_turns: 10
token_budget: 64
max_episode_tokens: 1024
grad_accum: 4
kl_coef: 0.1
role_shuffle: true
shared_weights: false
seed: 42
log_every: 10
checkpoint_every: 50
output_dir: runs/my_experiment
prompt_0: "You are Agent A, negotiating to maximise your score."
prompt_1: "You are Agent B, negotiating to maximise your score."
```

Run directly from a config:

```bash
uv run python train_negotiation.py --config configs/my_experiment.yaml
```

Configs are saved into the run directory as `experiment_config.yaml` at the start of training for full reproducibility.

---

### Submitting to SLURM

`submit_batch.py` reads one or more YAML config files and submits each as a separate SLURM job. It is tuned for **Isambard AI Phase 2** (GH200 Grace Hopper nodes, `workq` partition, `--gpus=N` syntax).

```
uv run python submit_batch.py [OPTIONS] CONFIGS...

Positional:
  CONFIGS     One or more YAML config files or directories containing them.
              Directories are scanned for *.yaml / *.yml files.

Options:
  --account ACCOUNT     SLURM project account (required on Isambard AI)
  --partition PART      Override the partition from config (default: workq)
  --qos QOS             SLURM QOS
  --dry-run             Print generated job scripts without submitting
  --scripts-dir PATH    Save generated job scripts here for inspection
  --log-dir PATH        Directory for SLURM stdout/stderr (default: slurm_logs/)
  --hf-home PATH        HuggingFace model cache directory
                        (default: /lus/lfs1aip2/projects/a5l/egunn/hf_cache)
  --mail-user EMAIL     Email address for job-status notifications
  --mail-type EVENTS    SLURM mail events (default: END,FAIL)
  --module MODULE       Extra 'module load' line (repeat for multiple)
  --no-default-modules  Skip default modules (cuda/12.6, brics/nccl)
  --venv PATH           Path to a venv activate script (default: auto-detect
                        .venv/ or fall back to 'uv run')
  --output-dir-prefix PATH  Prepend this to each config's output_dir, e.g.
                            to redirect all runs to project storage
```

**Examples:**

```bash
# Submit all experiments in a directory
uv run python submit_batch.py configs/negotiation_role_experiments/ \
  --account my_project

# Dry-run to inspect scripts without submitting
uv run python submit_batch.py configs/negotiation_role_experiments/ --dry-run

# Submit a single config
uv run python submit_batch.py configs/negotiation_role_experiments/05_no_shuffle_lr1e5_kl01.yaml \
  --account my_project

# Submit with email notification
uv run python submit_batch.py configs/negotiation_role_experiments/ \
  --account my_project \
  --mail-user you@example.com

# Save generated scripts for inspection before submitting
uv run python submit_batch.py configs/negotiation_role_experiments/ \
  --account my_project \
  --scripts-dir slurm_scripts/role_experiments/
```

Before submitting, set your HuggingFace token if you are using gated models (Llama etc.):

```bash
export HF_TOKEN="hf_..."
uv run python submit_batch.py configs/ --account my_project
```

The token is forwarded to jobs via `--export` and is never written into the job scripts.

---

### Resuming jobs

Every training script supports `--resume`. It loads `checkpoints/latest.pt` from `--output-dir` and continues from where training stopped. The flag is safe to include on first run — it is a no-op if no checkpoint exists.

To resume a SLURM job, resubmit the same script with the same config:

```bash
uv run python submit_batch.py configs/negotiation_role_experiments/08_shuffle_lr1e5_kl05.yaml \
  --account my_project
```

The config already has `output_dir` set, so `--resume` in the job script picks up the latest checkpoint automatically. The same job script is used for first run and all restarts.

---

## Inspecting Results

### Trace files

Every `log_every` iterations the trainer writes a human-readable trace of one episode to `{output_dir}/traces/iter_XXXXXX.txt`.

Traces have two sections:

**ENVIRONMENT OVERVIEW** — the chronological event log from the environment's perspective, showing who sends what to whom:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ENVIRONMENT OVERVIEW  (chronological event log)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ┌─ [PROMPT: agent_0]
  │  'You are Marcus, a merchant...'
  └─ ids=[...]

  ┌─ [ENV → agent_0]  observation
  │  'Negotiation. Items: 2 books, 2 hats, 1 balls. Your values: books=0, hats=10, balls=0...'
  └─ ids=[...]

  ┌─ [agent_0 → ENV]  action
  │  ' I would be happy to trade the books...'
  └─ ids=[...]
     ↓  routed as observation to agent_1

  ┌─ [agent_0+ENV → agent_1]  observation
  │  'Negotiation. Items: 2 books... Your values: books=9... I would be happy to trade...'
  └─ ids=[...]
     ├─ ENV prefix (38 tokens):  'Negotiation. Items: 2 books... Your values: books=9...'
     └─ from agent_0 (12 tokens): ' I would be happy to trade the books...'
```

**PER-AGENT CONTEXT** — what each agent's full context window contained during inference, including a `[BROADCAST]` annotation where the trainer's shared-context design causes one agent to see the other's private observations:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONTEXT: agent_0  (tokens in this agent's context window)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  [PROMPT]
    'You are Marcus, a merchant...'

  [OBS]
    origin: from ENV
    'Negotiation. Items: 2 books, 2 hats, 1 balls. Your values: books=0, hats=10, balls=0...'

  [ACT]
    ' I would be happy to trade the books...'

  [OBS]  [BROADCAST — addressed to agent_1]
    origin: routed from agent_0 + env context prefix
    'Negotiation. Items: 2 books... Your values: books=9, hats=0, balls=1...'
```

---

### Regenerating traces

`generate_traces.py` regenerates traces from any saved checkpoint without running the full training loop. It reads the experiment config saved in the run directory to reconstruct the environment and agents, then runs episodes and writes traces in the same two-section format.

```
uv run python generate_traces.py [OPTIONS]

Required:
  --run-dir PATH        Experiment output directory (must contain config.json
                        and experiment_config.yaml)

Checkpoint options:
  --checkpoint RELPATH  Path to a .pt file, relative to --run-dir or absolute.
                        Default: checkpoints/latest.pt
  --no-checkpoint       Skip loading any checkpoint (uses base pretrained
                        weights; useful for sanity-checking the trace format)

Output options:
  --n-traces N          Number of episodes to collect and save (default: 10)
  --output-dir PATH     Where to write trace files
                        (default: <run-dir>/traces_regen/)

Override options:
  --device DEVICE       Override the device from config.json
  --seed N              Override the RNG seed
  --temperature FLOAT   Override the sampling temperature
  --dialogue-turns N    Override max dialogue turns
  --token-budget N      Override tokens per turn
  --max-episode-tokens N  Override max episode token budget
```

**Examples:**

```bash
# Use latest checkpoint, write 10 traces to <run-dir>/traces_regen/
uv run python generate_traces.py \
  --run-dir runs/negotiation_role_experiments/08_shuffle_lr1e5_kl05

# Specific checkpoint, 20 traces
uv run python generate_traces.py \
  --run-dir runs/negotiation_role_experiments/08_shuffle_lr1e5_kl05 \
  --checkpoint checkpoints/iter_000300.pt \
  --n-traces 20

# Compare a specific checkpoint against an earlier one
uv run python generate_traces.py \
  --run-dir runs/negotiation_role_experiments/08_shuffle_lr1e5_kl05 \
  --checkpoint checkpoints/iter_000100.pt \
  --output-dir runs/negotiation_role_experiments/08_shuffle_lr1e5_kl05/traces_iter100

# Sanity-check trace format without loading any checkpoint
uv run python generate_traces.py \
  --run-dir runs/negotiation_role_experiments/08_shuffle_lr1e5_kl05 \
  --no-checkpoint \
  --n-traces 3
```

Trace files are named `ckpt{iteration}_ep{N:04d}.txt`, e.g. `ckpt500_ep0000.txt`.

---

### Regenerating traces in bulk via SLURM

`submit_traces.py` submits one SLURM job per run directory. Each job calls `generate_traces.py`. Trace generation is inference-only so jobs are short (default 1 hour) and need only one GPU.

```
uv run python submit_traces.py [OPTIONS] RUNS...

Positional:
  RUNS          One or more run directories, or a parent directory containing
                multiple run directories. Each immediate subdirectory with a
                config.json and at least one checkpoint is included.

Selection:
  --filter SUBSTR     Only include run directories whose name contains SUBSTR

SLURM options:
  --account ACCOUNT   SLURM project account (required on Isambard AI)
  --partition PART    SLURM partition (default: workq)
  --qos QOS           SLURM QOS
  --time HH:MM:SS     Walltime per job (default: 01:00:00)
  --gpus N            GPUs per job (default: 1)
  --cpus-per-gpu N    CPUs per GPU (default: 72)
  --mail-user EMAIL   Email for job-status notifications
  --mail-type EVENTS  SLURM mail events (default: END,FAIL)

Trace options:
  --n-traces N        Episodes per run (default: 10)
  --checkpoint PATH   Checkpoint relative to each run dir
                      (default: checkpoints/latest.pt)
  --output-subdir NAME  Subdirectory inside each run dir for output
                        (default: traces_regen). Use a distinct name when
                        generating from multiple checkpoints, e.g. traces_iter300
  --temperature FLOAT Sampling temperature override

Misc:
  --dry-run           Print scripts without submitting
  --scripts-dir PATH  Save job scripts here for inspection
  --log-dir PATH      SLURM stdout/stderr directory (default: slurm_logs/)
  --hf-home PATH      HuggingFace model cache directory
  --module MODULE     Extra 'module load' line (repeat for multiple)
  --no-default-modules  Skip default modules (cuda/12.6, brics/nccl)
  --venv PATH         Path to a venv activate script
```

**Examples:**

```bash
# Regenerate traces for all runs in a suite (latest checkpoint, 10 traces each)
uv run python submit_traces.py \
  runs/negotiation_role_experiments/ \
  --account my_project

# Dry-run first to check what would be submitted
uv run python submit_traces.py runs/negotiation_role_experiments/ --dry-run

# Only runs matching a substring
uv run python submit_traces.py runs/negotiation_role_experiments/ \
  --account my_project \
  --filter shared

# Specific checkpoint across all runs, named output subdir
uv run python submit_traces.py runs/negotiation_role_experiments/ \
  --account my_project \
  --checkpoint checkpoints/iter_000300.pt \
  --output-subdir traces_iter300 \
  --n-traces 20

# Compare checkpoints at two points in training for a whole suite
uv run python submit_traces.py runs/negotiation_role_experiments/ \
  --account my_project --checkpoint checkpoints/iter_000100.pt \
  --output-subdir traces_iter100
uv run python submit_traces.py runs/negotiation_role_experiments/ \
  --account my_project --checkpoint checkpoints/iter_000500.pt \
  --output-subdir traces_iter500
```

A timestamped manifest of all submitted job IDs is written to `slurm_logs/`.

---

### Plots

`scripts/plot_results.py` reads `metrics.jsonl` from each run and produces:
- `training_curves.png` per experiment — loss, return, entropy, KL over iterations
- `suite_comparison.png` — bar charts of final values across all experiments
- `suite_training.png` — all experiments' smoothed return on one axes

```
uv run python scripts/plot_results.py SUITE_DIR [OPTIONS]

Positional:
  SUITE_DIR     Parent directory containing run subdirectories

Options:
  --exp PREFIX...   Only include experiments matching these prefixes
                    (e.g. --exp 01 05 09 to include runs starting with those)
  --no-per-exp      Skip per-experiment training_curves.png
  --no-suite        Skip suite-level comparison plots
```

**Examples:**

```bash
# All experiments in a suite
uv run python scripts/plot_results.py runs/negotiation_role_experiments/

# Subset of experiments
uv run python scripts/plot_results.py runs/negotiation_role_experiments/ \
  --exp 07 08 11 12

# Suite-level comparison only (no per-experiment plots)
uv run python scripts/plot_results.py runs/negotiation_role_experiments/ \
  --no-per-exp
```

---

### Concatenated reports

Two scripts produce text-format reports suitable for pasting into analysis notes or sending to a language model for review.

**`scripts/concat_results.py`** — full report including configs, logs, and trace snapshots:

```
uv run python scripts/concat_results.py SUITE_DIR [OPTIONS]

Options:
  -o, --output PATH     Output file (default: stdout)
  -n, --snapshots N     Number of evenly-spaced metric and trace snapshots
                        per experiment (default: 6)
  --no-traces           Omit trace files from the report
  --exp PREFIX...       Only include experiments matching these prefixes
```

**`scripts/concat_metrics_summary.py`** — compact summary of configs and metric snapshots only (no traces, no logs):

```
uv run python scripts/concat_metrics_summary.py SUITE_DIR [OPTIONS]

Options:
  -n, --snapshots N     Number of metric snapshots per experiment (default: 6)
  -o, --output PATH     Output file (default: stdout)
  --exp PREFIX...       Only include experiments matching these prefixes
```

**Examples:**

```bash
# Full report for a suite
uv run python scripts/concat_results.py runs/negotiation_role_experiments/ \
  -o reports/role_experiments.txt

# Compact metrics summary
uv run python scripts/concat_metrics_summary.py runs/negotiation_role_experiments/ \
  -o reports/role_metrics.txt

# Just two experiments, more metric snapshots
uv run python scripts/concat_results.py runs/negotiation_role_experiments/ \
  --exp 07 08 -n 10 -o reports/kl05_comparison.txt
```

---

## Core Abstractions

The framework has five pluggable interfaces. Each is an ABC with a concrete default implementation.

### Agent

Wraps a model. The training loop calls `act()` during rollouts and `evaluate()` during the update step.

```python
from marlllm import Agent, IndependentAgent

# Built-in: any HuggingFace CausalLM
agent = IndependentAgent(
    agent_id="agent_0",
    character_prompt="You are a chess grandmaster playing white.",
    model_name_or_path="gpt2",        # or "Qwen/Qwen2.5-7B-Instruct", etc.
    device="cuda",
    torch_dtype=torch.bfloat16,       # "auto" lets HF choose
    device_map="auto",                # shard across GPUs
)
```

`IndependentAgent` attaches a learned value head (a single linear layer) on top of the LM's hidden states. Hidden states are detached before the value head so the value loss never propagates into the LM backbone.

`LoRASharedBaseAgent` wraps a single PEFT model with two independent LoRA adapters — one per agent. The frozen base is shared in memory; only the adapters are trained. This halves backbone VRAM when training two agents on one GPU.

### Environment

Any PettingZoo AEC environment. The framework passes observations to agents unmodified — all social structure (turn order, partial observability, communication) is the environment's concern.

```python
from envs.counting_env import CountingEnv
from envs.deal_or_no_deal_env import DealOrNoDealEnv

env = CountingEnv(tokenizer=agent.tokenizer, max_count=20)

env = DealOrNoDealEnv(
    tokenizer=agent.tokenizer,
    max_dialogue_turns=10,
    action_token_budget=64,
    role_shuffle=True,
)
```

The env must expose one custom attribute used by the Trainer:
- `action_token_budget: int` — how many tokens the agent produces per turn

### Loss

Computes the scalar loss and diagnostics from a batched forward pass.

```python
from marlllm import CCSMLoss

loss = CCSMLoss()   # REINFORCE with intrinsic surprise return
```

### Tokeniser

Converts between raw environment observations/actions and token IDs, and assembles trajectories.

```python
from marlllm import TextTokeniser

tokeniser = TextTokeniser(agent.tokenizer)
```

### Trajectory Store

Buffers rollouts between collection and the update step.

```python
from marlllm import OnPolicyStore

store = OnPolicyStore()   # simple FIFO buffer, cleared after each update
```

---

## Configuration Reference

All hyperparameters live in `TrainingConfig`. Every field has a default.

| Field | Default | Description |
|---|---|---|
| `model_name_or_path` | `"gpt2"` | HuggingFace model ID or local path |
| `character_prompts` | `{"agent_0": "..."}` | Character prompt per agent ID |
| `gamma` | `0.99` | Discount factor for future observation surprise |
| `beta` | `0.01` | Entropy regularisation coefficient |
| `alpha_perc` | `1.0` | Weight on perception loss |
| `alpha_act` | `1.0` | Weight on action loss |
| `normalise_returns` | `True` | Standardise G_t to zero mean / unit variance per batch |
| `kl_coef` | `0.0` | KL penalty coefficient vs frozen reference model |
| `episodes_per_iter` | `8` | Rollouts collected before each gradient update |
| `max_episode_tokens` | `128` | Token budget per episode (excluding prompt) |
| `num_iterations` | `500` | Total training iterations |
| `lr` | `3e-5` | AdamW learning rate |
| `temperature` | `1.0` | Sampling temperature during rollouts |
| `grad_accum_steps` | `1` | Gradient accumulation micro-batches |
| `log_every` | `10` | Log to stdout every N iterations |
| `checkpoint_every` | `100` | Save checkpoint every N iterations |
| `output_dir` | `"runs/default"` | Directory for logs, metrics, checkpoints |
| `device` | `"cpu"` | PyTorch device string |
| `seed` | `42` | Random seed |
| `lora_r` | `0` | LoRA rank (0 = full fine-tuning) |
| `lora_alpha` | `16` | LoRA alpha scaling factor |
| `lora_target_modules` | `None` | LoRA target module names (None = auto-detect) |
| `gradient_checkpointing` | `False` | Activation checkpointing for VRAM savings |

---

## Writing a New Environment

Implement the PettingZoo AEC interface and add `action_token_budget`:

```python
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector

class MyEnv(AECEnv):
    metadata = {"name": "my_env_v0"}

    def __init__(self, tokenizer):
        super().__init__()
        self._tok = tokenizer
        self.possible_agents = ["agent_0", "agent_1"]
        self.action_token_budget = 32  # tokens the agent generates per turn

    def reset(self, seed=None, options=None):
        self.agents = list(self.possible_agents)
        self._selector = agent_selector.AgentSelector(self.agents)
        self.agent_selection = self._selector.next()
        self._pending_obs = {a: [] for a in self.agents}
        self._terminations = {a: False for a in self.agents}
        self._truncations  = {a: False for a in self.agents}
        self.rewards = {a: 0.0 for a in self.agents}
        self.infos   = {a: {} for a in self.agents}

    def last(self):
        agent = self.agent_selection
        return (
            self._pending_obs[agent],
            self.rewards[agent],
            self._terminations[agent],
            self._truncations[agent],
            self.infos[agent],
        )

    def step(self, action):
        # action is list[int] of length action_token_budget, or None
        if action is None or self._terminations[self.agent_selection]:
            self._was_dead_step(action)
            return
        # ... update state, set observations for other agents, check termination
        self.agent_selection = self._selector.next()

    # Also required: observe(), observation_space(), action_space(),
    #                terminations, truncations, rewards, infos properties,
    #                render(), close()
```

Framework contracts:
1. `last()` returns the observation as a string or `list[int]` — `TextTokeniser` handles both.
2. `step()` receives `list[int]` of length `action_token_budget`, or `None` for terminated agents.
3. Terminal observations must be placed in `_pending_obs` **before** setting `_terminations[agent] = True` — the Trainer reads the terminal observation via `env.last()` on the same cycle as termination.

---

## Writing a New Agent

```python
from marlllm import Agent
from typing import Iterable
import torch
import torch.nn as nn

class MyAgent(Agent):
    def __init__(self, agent_id, character_prompt, model, value_head):
        self._id = agent_id
        self._prompt = character_prompt
        self._model = model
        self._value_head = value_head

    @property
    def agent_id(self): return self._id

    @property
    def character_prompt(self): return self._prompt

    def act(self, context_token_ids, n_tokens, temperature=1.0):
        # Sample n_tokens autoregressively from context_token_ids.
        # Returns (token_ids: list[int], log_probs: list[float]).
        # Called under torch.no_grad(); use self._model.eval() beforehand.
        ...

    def evaluate(self, input_ids, attention_mask):
        # Full differentiable forward pass used during the training update.
        # Returns (logits: Tensor[B, T, V], values: Tensor[B, T]).
        # Values should be 0.0 everywhere if this agent has no value head.
        ...

    def evaluate_ref(self, input_ids, attention_mask):
        # Optional: frozen reference model forward pass for KL penalty.
        # Return None if kl_coef == 0 or this agent has no reference model.
        return None

    def parameters(self) -> Iterable[nn.Parameter]:
        # Yield only the parameters to be optimised.
        yield from self._model.parameters()
        yield from self._value_head.parameters()

    def train_mode(self): self._model.train()
    def eval_mode(self):  self._model.eval()
```

---

## Token Type Mask

The `TokenType` enum is the central routing mechanism:

| Value | Name | Meaning |
|---|---|---|
| `0` | `PAD` | Character prompt prefix and right-padding — excluded from all losses |
| `1` | `OBS` | Environment observation token — contributes to `L_perc` (NTP) |
| `2` | `ACT` | Agent action token — contributes to `L_act` and `L_val` (REINFORCE + value) |

`PAD` (value 0) doubles as the right-padding fill in batched tensors, so no separate padding mask is needed in loss functions.

---

## Background

MARLLLM implements the CCSM algorithm described in [`surprise_minimisation_derivation.md`](surprise_minimisation_derivation.md), which derives a unified training objective from the Action Perception Divergence framework (Hafner et al., 2022). The key insight is that joint KL minimisation between the actual trajectory distribution and the model's own predictive distribution produces both a perception signal (next-token prediction on observations) and an action signal (REINFORCE with intrinsic surprise return) from a single principle, with no extrinsic reward.

The framework design is specified in [`framework_spec.md`](framework_spec.md).
