"""Training configuration for a MARLLLM run."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TrainingConfig:
    # Model
    model_name_or_path: str = "gpt2"
    character_prompts: dict[str, str] = field(
        default_factory=lambda: {
            "agent_0": "You are a counter that always outputs the next integer in sequence."
        }
    )

    # CCSM hyperparameters (§9.2 of surprise_minimisation_derivation.md)
    gamma: float = 0.99       # discount factor for future obs surprise
    beta: float = 0.01        # entropy regularisation coefficient
    alpha_perc: float = 1.0   # weight on L_perc
    alpha_act: float = 1.0    # weight on L_act
    normalise_returns: bool = True  # standardise G_t within batch (§9.3)
    kl_coef: float = 0.0      # KL penalty weight: λ * KL(π_θ || π_ref) at ACT positions

    # Training loop
    episodes_per_iter: int = 8
    max_episode_tokens: int = 128  # non-prompt tokens per episode
    num_iterations: int = 500
    lr: float = 3e-5
    log_every: int = 10

    # Device
    device: str = "cpu"

    # Sampling
    temperature: float = 1.0
    seed: int = 42

    # Output / checkpointing
    output_dir: str = "runs/default"
    checkpoint_every: int = 100  # save a checkpoint every N iterations

    # Distribution / memory
    grad_accum_steps: int = 1          # split each batch into N micro-batches, accumulate grads
    gradient_checkpointing: bool = False  # recompute activations during backward to save VRAM

    # LoRA (PEFT) — set lora_r > 0 to enable; requires `peft` package
    lora_r: int = 0                    # LoRA rank; 0 = full fine-tuning
    lora_alpha: int = 16               # LoRA scaling factor
    lora_target_modules: list[str] | None = None  # None = PEFT auto-detect
