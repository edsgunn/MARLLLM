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
    always_log_kl: bool = False  # If True, compute & log KL each step even when kl_coef == 0 (no gradient applied; requires keep_ref_model=True on agents).

    # Asymmetric multi-agent training (§2.4 of Phase A spec).
    # IDs in this list participate in rollouts but receive NO gradient updates.
    # E.g. frozen_agents=["agent_1"] turns agent_1 into a fixed pretrained
    # partner while agent_0 is trained — the "focal vs fixed partner" setup.
    frozen_agents: list[str] = field(default_factory=list)

    # Treat the character prompt as an observation the agent emits about itself,
    # rather than as masked exogenous conditioning. When True (default), prompt
    # tokens are tagged TokenType.OBS — they contribute to L_perc (next-token
    # prediction) but never to the policy gradient. Over training the model
    # learns to predict its own character prompt, absorbing it into the weights.
    # When False, prompt tokens are tagged TokenType.PAD (legacy behaviour):
    # they condition generation but receive no loss of any kind.
    prompt_as_observation: bool = True

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
    # Use bitsandbytes' 8-bit AdamW (halves optimizer state vs fp32 m/v).
    # Saves ~5-10 GB on a 7B-LoRA setup with no expressiveness loss.
    use_8bit_adam: bool = False

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
    # Sequence-dim chunked loss step. When set, the per-episode forward+backward
    # is split into chunks of this many tokens along the sequence axis and
    # backward runs once per chunk. Reduces peak activation memory roughly by
    # T / seq_chunk_size at the cost of running a full no-grad forward first
    # to compute global returns, plus per-chunk re-forwards. Cross-chunk
    # attention gradients are dropped (each chunk's autograd graph attends
    # over a detached KV cache from prior chunks); this is an approximation,
    # exact within a chunk. Set to None to disable (default; uses the original
    # single-shot path).
    seq_chunk_size: int | None = None

    # Sequence packing: concatenate one agent's K trajectories into a single
    # (1, T_total) row with a block-diagonal causal attention mask instead of
    # right-padding them all to T_max. Removes pad-token FLOPs (which dominate
    # at our typical 3-5x length variance per agent), at the cost of building
    # a 4D (1, 1, T_total, T_total) additive attention mask — 128 MiB in bf16
    # at T_total=8192, kept live on the autograd graph. For very long T_total
    # use flash-attention varlen instead (one-line follow-up once flash-attn
    # is in the venv; see RLvr_rollout_optimisation_audit.md).
    #
    # Constraints (May 2026):
    #   * Mutually exclusive with seq_chunk_size — chunked + packed together
    #     needs cu_seqlens-aware per-chunk masks, not yet implemented.
    #   * Only the non-chunked compute_loss path is wired (single forward
    #     over the packed row), so total tokens per agent must fit memory.
    pack_sequences: bool = False

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

    # Variance-decomposition diagnostic.
    # Estimates Var[G_t] = Var_a[E[G|a]] + E_a[Var[G|a]] on a fixed held-out
    # context set, so we can detect dark-room collapse (signal ↓ noise ↓) and
    # entrenched diffuse non-learning (signal ↓ noise ↑) directly rather than
    # inferring them from entropy curves. See marlllm/variance_decomposition.py.
    var_decomp_enabled: bool = False
    var_decomp_eval_contexts_path: str | None = None  # auto-built if missing
    var_decomp_n_contexts: int = 32
    var_decomp_K: int = 8                        # actions sampled per context
    var_decomp_M: int = 4                        # env-response samples per (context, action)
    var_decomp_max_continuation_steps: int = 0   # 0 = run to env termination
    var_decomp_n_eval_early: int = 5             # cadence while iter <= switch
    var_decomp_n_eval_late: int = 25             # cadence after the switch
    var_decomp_switch_iter: int = 50             # iter at which cadence relaxes

    # ------------------------------------------------------------------
    # Live regime tracking. Stamps a five-way regime label on each
    # metrics.jsonl record using the (kl, perc_loss, entropy) thresholds
    # calibrated in scripts/plot_population_regime_figures.py, plus
    # population aggregates, slope-based early warnings, and live PNG
    # visualisations under <output_dir>/regime/. See marlllm/regime_tracking.py
    # for the implementation and the calibration rationale.
    # ------------------------------------------------------------------
    regime_tracking_enabled: bool = True
    regime_kl_threshold: float = 1.0
    regime_perc_coherent: float = 1.7
    regime_perc_degen: float = 2.0
    regime_ent_low: float = 0.2
    regime_ent_high: float = 1.8
    regime_slope_window: int = 5
    # Render the live regime composition + per-agent strip plots every N iters.
    regime_viz_every: int = 25
    # Compute Tier-3 token-level signals (tail_loop_rate, non_ascii_rate,
    # top_token_frac) from rollout traces. Off by default since it touches
    # the rollout hot path; enable for debugging suspected collapses.
    regime_token_signals: bool = False
    # If True, copy the most recent on-disk checkpoint into
    # checkpoints/pre_collapse_iter_<N>/ when any agent first enters
    # text_degenerate. Useful for keeping the snapshot just before collapse.
    regime_pre_collapse_checkpoint: bool = True
