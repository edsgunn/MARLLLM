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
    alpha_perc: float = 1.0   # weight on L_perc (set to 0 for action-only ablation)
    alpha_act: float = 1.0    # weight on L_act  (set to 0 for perception-only ablation)
    alpha_val: float = 1.0    # weight on L_val  (set to 0 for perception-only ablation)
    normalise_returns: bool = True  # standardise G_t within batch (§9.3)
    kl_coef: float = 0.0      # KL penalty weight: λ * KL(π_θ || π_ref) at ACT positions

    # Asymmetric multi-agent training (§2.4 of Phase A spec).
    # IDs in this list participate in rollouts but receive NO gradient updates.
    # E.g. frozen_agents=["agent_1"] turns agent_1 into a fixed pretrained
    # partner while agent_0 is trained — the "focal vs fixed partner" setup.
    frozen_agents: list[str] = field(default_factory=list)

    # Per-agent perception masking (Cells B/C/D/E of the distillation ablation).
    # When non-empty, L_perc only counts OBS tokens whose *source* agent
    # (agent_id_mask at that position) is in this list. Default = empty list,
    # which means "include all agents' OBS tokens" (existing behaviour).
    # Setting this to e.g. ["agent_1"] focuses perception on the strong
    # partner's tokens only — a clean way to express "perception-active" in
    # the spec's per-agent loss-flag matrix.
    perception_agents: list[str] = field(default_factory=list)

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
    num_checkpoint_traces: int = 4  # traces saved alongside each checkpoint

    # Distribution / memory
    grad_accum_steps: int = 1          # split each batch into N micro-batches, accumulate grads
    gradient_checkpointing: bool = False  # recompute activations during backward to save VRAM

    # LoRA (PEFT) — set lora_r > 0 to enable; requires `peft` package
    lora_r: int = 0                    # LoRA rank; 0 = full fine-tuning
    lora_alpha: int = 16               # LoRA scaling factor
    lora_target_modules: list[str] | None = None  # None = PEFT auto-detect

    # Held-out behavioural-distribution snapshots
    # When ``snapshot_eval_path`` is set, at every checkpoint the population
    # samples ``snapshot_samples_per_context`` continuations per held-out
    # context and writes results to ``{output_dir}/snapshots/iter_NNNNNN.json``.
    snapshot_eval_path: str | None = None
    snapshot_samples_per_context: int = 8
    snapshot_max_new_tokens: int = 128
