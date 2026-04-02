# MARLLLM

A framework for training language models through multi-agent interaction using **Character-Conditioned Surprise Minimisation (CCSM)** — a reward-free training objective derived from the Action Perception Divergence framework (Hafner et al., 2022).

Agents learn by minimising the KL divergence between the trajectories they actually produce and the trajectories their own model predicts. A **character prompt** is the sole specification of desired behaviour; no reward function is needed.

The framework uses [PettingZoo](https://pettingzoo.farama.org/) as its environment interface. Any PettingZoo-compatible environment works without modification.

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

# Larger model on GPU
uv run python train.py \
  --model Qwen/Qwen2.5-3B-Instruct \
  --device cuda \
  --dtype bfloat16 \
  --iters 500 \
  --output-dir runs/qwen-counting
```

Output is written to `runs/default/` (or `--output-dir`):
```
runs/default/
  config.json          # full hyperparameter snapshot
  train.log            # timestamped log
  metrics.jsonl        # one JSON record per iteration
  checkpoints/
    iter_000100.pt
    final.pt
    latest.pt          # symlink to most recent
```

Resume a killed job:
```bash
uv run python train.py --output-dir runs/qwen-counting --resume
```

---

## The Algorithm

CCSM is derived in detail in [`surprise_minimisation_derivation.md`](surprise_minimisation_derivation.md). The short version:

A trajectory interleaves **observation tokens** (produced by the environment, type `σ=1`) and **action tokens** (sampled by the model, type `σ=0`). The loss has two terms that emerge from a single KL objective:

**Perception loss** — next-token prediction on observation tokens:
$$\mathcal{L}^{\text{perc}} = -\frac{1}{N_{\text{obs}}} \sum_{t:\,\sigma_t=1} \ln p_\theta(x_t \mid c, x_{<t})$$

**Action loss** — REINFORCE weighted by future observation surprise:
$$\mathcal{L}^{\text{act}} = -\frac{1}{N_{\text{act}}} \sum_{t:\,\sigma_t=0} \ln p_\theta(x_t \mid c, x_{<t}) \cdot \hat{A}_t \;-\; \beta \cdot H[p_\theta(\cdot \mid c, x_{<t})]$$

where the advantage $\hat{A}_t = -(G_t - V_\phi)$ and the intrinsic return $G_t = \sum_{s>t} \sigma_s \gamma^{s-t}(-\ln p_\theta(x_s \mid c, x_{<s}))$ is the discounted future observation surprise. No extrinsic reward. The character prompt $c$ is the only specification of what the agent should do.

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

# Custom agent: subclass Agent and implement act() + evaluate() + parameters()
class MyAgent(Agent):
    ...
```

The `IndependentAgent` attaches a learned value head (a single linear layer) on top of the LM's hidden states. Hidden states are detached before the value head so the value loss never propagates into the LM backbone.

### Environment

Any PettingZoo AEC environment. The framework passes observations to agents unmodified — all social structure (turn order, partial observability, communication) is the environment's concern.

```python
from envs.counting_env import CountingEnv

env = CountingEnv(tokenizer=agent.tokenizer, max_count=20)

# Or any PettingZoo AEC env:
from pettingzoo.classic import chess_v6
env = chess_v6.env()
```

The env must expose one custom attribute used by the Trainer:
- `action_token_budget: int` — how many tokens the agent produces per turn (1 for token-by-token interaction)

### Loss

Computes the scalar loss and diagnostics from a batched forward pass.

```python
from marlllm import CCSMLoss, Loss

loss = CCSMLoss()   # REINFORCE with intrinsic surprise return

# Custom loss: subclass Loss and implement compute_loss()
class MyLoss(Loss):
    requires_value_function = True
    requires_old_log_probs = False

    def compute_loss(self, logits, values, input_ids,
                     token_type_mask, agent_id_mask,
                     target_agent_idx, config) -> tuple[Tensor, dict]:
        ...
```

### Tokeniser

Converts between raw environment observations/actions and token IDs, and assembles trajectories.

```python
from marlllm import TextTokeniser

# Wraps any HuggingFace tokenizer
tokeniser = TextTokeniser(agent.tokenizer)

# Custom tokeniser for multimodal or structured action spaces:
class VLMTokeniser(Tokeniser):
    def encode_observation(self, obs) -> list[int]: ...
    def decode_action(self, ids) -> Any: ...
    def encode_prompt(self, prompt) -> list[int]: ...
    def build_trajectory(self, history, agent_ids) -> Trajectory: ...
```

### Trajectory Store

Buffers rollouts between collection and the update step.

```python
from marlllm import OnPolicyStore

store = OnPolicyStore()   # simple FIFO buffer, cleared after each update

# Custom store: subclass TrajectoryStore and implement store() / sample() / clear()
```

---

## Training

Wire everything together with `Trainer`:

```python
from marlllm import (
    IndependentAgent, CCSMLoss, TextTokeniser,
    OnPolicyStore, Trainer, TrainingConfig,
)

config = TrainingConfig(
    model_name_or_path="gpt2",
    character_prompts={"agent_0": "You are a helpful assistant."},
    episodes_per_iter=8,
    num_iterations=500,
    lr=3e-5,
    gamma=0.99,
    beta=0.01,
    output_dir="runs/my_experiment",
    checkpoint_every=100,
    device="cpu",
)

agent = IndependentAgent(
    agent_id="agent_0",
    character_prompt=config.character_prompts["agent_0"],
    model_name_or_path=config.model_name_or_path,
    device=config.device,
)

trainer = Trainer(
    agents={"agent_0": agent},
    env=env,
    loss=CCSMLoss(),
    tokeniser=TextTokeniser(agent.tokenizer),
    store=OnPolicyStore(),
    config=config,
)

trainer.train()
```

### Checkpointing and resuming

```python
# Save manually at any point
trainer.save_checkpoint(iteration=42)

# Load and continue
iteration = trainer.load_checkpoint("runs/my_experiment/checkpoints/iter_000042.pt")
trainer.train(start_iteration=iteration + 1)
```

---

## Configuration Reference

All hyperparameters live in `TrainingConfig`. Every field has a default.

| Field | Default | Description |
|---|---|---|
| `model_name_or_path` | `"gpt2"` | HuggingFace model ID or local path |
| `character_prompts` | `{"agent_0": ...}` | Character prompt per agent ID |
| `gamma` | `0.99` | Discount factor for future obs surprise |
| `beta` | `0.01` | Entropy regularisation coefficient |
| `alpha_perc` | `1.0` | Weight on perception loss |
| `alpha_act` | `1.0` | Weight on action loss |
| `normalise_returns` | `True` | Standardise G_t to zero mean / unit variance per batch |
| `episodes_per_iter` | `8` | Rollouts collected before each gradient update |
| `max_episode_tokens` | `128` | Token budget per episode (excluding prompt) |
| `num_iterations` | `500` | Total training iterations |
| `lr` | `3e-5` | AdamW learning rate |
| `log_every` | `10` | Print to stdout every N iterations |
| `checkpoint_every` | `100` | Save checkpoint every N iterations |
| `output_dir` | ``"runs/default"` | Directory for logs, metrics, checkpoints |
| `device` | `"cpu"` | PyTorch device string |
| `temperature` | `1.0` | Sampling temperature during rollouts |
| `seed` | `42` | Random seed |

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
        self.possible_agents = ["agent_0"]
        self.action_token_budget = 1   # tokens the agent produces per turn

    def reset(self, seed=None, options=None):
        self.agents = list(self.possible_agents)
        self._selector = agent_selector.AgentSelector(self.agents)
        self.agent_selection = self._selector.next()
        self._obs = {"agent_0": []}
        self._terminations = {"agent_0": False}
        self._truncations = {"agent_0": False}
        # ...

    def observe(self, agent):
        # Return a list of token IDs (or a string — TextTokeniser handles both)
        return self._obs[agent]

    def step(self, action):
        # action is a list[int] of token IDs, length == action_token_budget
        # Set self._obs, self._terminations, etc.
        # Call self._was_dead_step(action) if agent is already terminated
        ...

    # Also required: terminations, truncations, rewards, infos properties
    # observation_space(), action_space(), render(), close()
```

The only framework-specific contract:
1. `observe()` returns something `TextTokeniser.encode_observation()` can handle (string, list of ints, or bytes).
2. `action` passed to `step()` is `list[int]` of length `action_token_budget`, or `None` for terminated agents.
3. Terminal observations must be set **before** `self._terminations[agent] = True` — the Trainer calls `env.last()` to read the terminal observation on the same cycle as termination.

---

## Writing a New Agent

```python
from marlllm import Agent
from typing import Iterable
import torch
import torch.nn as nn

class LoRAAgent(Agent):
    def __init__(self, agent_id, character_prompt, base_model, lora_weights):
        self._id = agent_id
        self._prompt = character_prompt
        # ... set up LoRA-adapted model

    @property
    def agent_id(self): return self._id

    @property
    def character_prompt(self): return self._prompt

    def act(self, context_token_ids, n_tokens, temperature=1.0):
        # Sample n_tokens autoregressively; return (token_ids, log_probs)
        ...

    def evaluate(self, input_ids, attention_mask):
        # Full differentiable forward pass
        # Returns (logits: (B, T, V), values: (B, T))
        # Values should be 0.0 everywhere if this agent has no value head
        ...

    def parameters(self) -> Iterable[nn.Parameter]:
        # Return only the LoRA parameters, not the frozen backbone
        yield from self._lora_params

    def train_mode(self): self._model.train()
    def eval_mode(self): self._model.eval()
```

---

## Token Type Mask

The `TokenType` enum is the central routing mechanism in the framework:

| Value | Name | Meaning |
|---|---|---|
| `0` | `PAD` | Padding / character prompt prefix — excluded from all losses |
| `1` | `OBS` | Environment observation token — contributes to `L_perc` |
| `2` | `ACT` | Agent action token — contributes to `L_act` and `L_val` |

`PAD` (value 0) doubles as the right-padding fill in batched tensors, so no separate padding mask is needed in loss functions.

---

## SLURM Example

```bash
#!/bin/bash
#SBATCH --job-name=marlllm
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=slurm_%j.log

cd /path/to/MARLLLM
source .venv/bin/activate

python train.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --device cuda \
  --dtype bfloat16 \
  --iters 2000 \
  --rollouts 16 \
  --checkpoint-every 50 \
  --output-dir runs/qwen7b-chess \
  --prompt "You are a chess grandmaster playing white." \
  --resume   # safe to include on first run; no-op if no checkpoint exists
```

If the job is preempted, resubmit the same script — `--resume` picks up from the last checkpoint automatically.

---

## Background

MARLLLM implements the CCSM algorithm described in [`surprise_minimisation_derivation.md`](surprise_minimisation_derivation.md), which derives a unified training objective from the Action Perception Divergence framework (Hafner et al., 2022). The key insight is that joint KL minimisation between the actual trajectory distribution and the model's own predictive distribution produces both a perception signal (next-token prediction on observations) and an action signal (REINFORCE with intrinsic surprise return) from a single principle, with no extrinsic reward.

The framework design follows [`framework_spec.md`](framework_spec.md).
