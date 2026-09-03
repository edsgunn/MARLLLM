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
    # Discount on future observation surprise, applied POSITIONALLY per token
    # (the exponent is s-t in the token sequence, gaps included) — see
    # marlllm/loss.py _compute_returns.
    #
    # This is NOT part of the derived objective. surprise_minimisation_derivation.md
    # introduces it in section 6.2 as variance reduction, "reduces variance at
    # the cost of bias", over gamma in (0, 1] — so gamma = 1 IS the derived
    # return and anything below it is a deliberate bias-for-variance trade.
    #
    # At 0.99 per token the horizon is 1/(1-gamma) = 100 tokens, against a
    # post_token_budget of 128-384. The consequence is not a smaller return —
    # normalise_returns divides the scale out — but a positional one: on a real
    # 386-token post the future-observation mass at an action token runs from
    # 2.0 at the first token to 97.1 at the last, a 48x swing, where gamma = 1
    # gives 2057 flat across the whole post. Credit that should be uniform over
    # a post is concentrated on its final tokens.
    #
    # 0.99 was the hardcoded default for every run before 2026-09-03 and no
    # config could reach it; it was never a chosen value.
    gamma: float = 0.99
    beta: float = 0.01        # entropy regularisation coefficient
    alpha_perc: float = 1.0   # weight on L_perc (set to 0 for action-only ablation)
    alpha_act: float = 1.0    # weight on L_act  (set to 0 for perception-only ablation)
    alpha_val: float = 1.0    # weight on L_val  (set to 0 for perception-only ablation)
    normalise_returns: bool = True  # standardise G_t within batch (§9.3)

    # Fit the value head to the RAW return even when normalise_returns is on.
    #
    # By default the head regresses onto the already-standardised return, so it
    # is fitted to a z-score: it can learn within-batch structure but not the
    # LEVEL, which is precisely the drift a baseline exists to absorb. And the
    # level here is non-stationary for a reason RL does not have — G_t is built
    # from the same per-token surprises the perception arm is simultaneously
    # driving down, so the return shrinks globally as perception improves,
    # independently of the policy.
    #
    # Fitting on raw returns lets the baseline absorb that drift. The cost is
    # the optimisation problem that motivated normalising the target in the
    # first place: raw returns run to ~250 at gamma=0.99 and ~2300 at gamma=1,
    # and a plain MSE head over that range is badly conditioned. If this option
    # trains poorly, that is the signal to reach for PopArt or a two-hot /
    # symlog head rather than to standardise the target again.
    value_target_raw: bool = False

    # Denominator for the action arm's mean.
    #
    # surprise_minimisation_derivation.md 7.1-7.2 writes BOTH arms as sums over
    # t, so their relative weight is 1:1 and is fixed by the gradient
    # decomposition rather than chosen. The implementation takes a mean of each
    # arm over its own token count, which multiplies the action arm by
    # N_obs/N_act — about 6.4 on the reference substrate, drifting to 7.05 by
    # iteration 300 as agents post less. That factor is not constant and was
    # never selected.
    #
    #   "act"  — divide by the number of action tokens (the historical default)
    #   "obs"  — divide by the number of observation tokens, the same
    #            denominator as the perception arm, restoring the derived 1:1
    #            ratio up to a global 1/N_obs that is only a learning-rate scale
    action_loss_denominator: str = "act"
    kl_coef: float = 0.0      # KL penalty weight: λ * KL(π_θ || π_ref) at ACT positions
    always_log_kl: bool = False  # If True, compute & log KL each step even when kl_coef == 0 (no gradient applied; requires keep_ref_model=True on agents).

    # ---------------------------------------------------------------- #
    # Environment likelihood (see environment_likelihood_derivation.md)   #
    # ---------------------------------------------------------------- #
    # The practical loss drops Sum_s sigma_s ln p_env(x_s | x_<s) because the
    # environment "can be sampled but not evaluated". In a population of
    # LoRA agents over one frozen base that is false for the only observation
    # tokens carrying entropy — other agents' posts — whose per-token
    # log-probs we already compute when sampling them.
    #
    # env_likelihood_logging: carry the generating agent's log-probs through
    #   to every observer, so E_s = ln p_env(x_s) is available at training
    #   time. Diagnostics only; the return is unchanged. Requires an env that
    #   exposes observation provenance (forum env with provenance=True).
    env_likelihood_logging: bool = False
    # use_env_likelihood_return: put E_s back into the REINFORCE return, so
    #   R_t = Sum_{s>t} gamma^(s-t) (E_s - M_s) instead of -G_t = Sum M_s.
    #   Requires env_likelihood_logging.
    use_env_likelihood_return: bool = False
    # bias_probe_enabled: train a SECOND value head on the corrected return
    #   while the policy trains on whichever return is configured. Its input
    #   is detached, exactly as the main value head's is, so it cannot leak
    #   into the trunk. Used to centre E_s when estimating the gradient bias.
    bias_probe_enabled: bool = False
    # bias_probe_every: iterations between the expensive gradient-direction
    #   probe (two backward passes over the action loss). 0 disables it.
    bias_probe_every: int = 25

    # Asymmetric multi-agent training (§2.4 of Phase A spec).
    # IDs in this list participate in rollouts but receive NO gradient updates.
    # E.g. frozen_agents=["agent_1"] turns agent_1 into a fixed pretrained
    # partner while agent_0 is trained — the "focal vs fixed partner" setup.
    frozen_agents: list[str] = field(default_factory=list)

    # Fixed partner graph (REWRITE.md 6.3). None means every population member
    # takes part in every episode, which is what a forum substrate has always
    # done and remains the default.
    #
    # Set to k and the population is partitioned once, deterministically, into
    # disjoint groups of k+1; each episode runs with one group. Every agent then
    # has exactly k interaction partners and always the SAME k, while the
    # population stays size N. That is what separates partner count from
    # population size — the two are otherwise varied together by every
    # population-size comparison in this project.
    #
    # Requires (k+1) to divide N. Note the graph is a disjoint union of
    # (k+1)-cliques rather than an arbitrary k-regular graph: within an episode
    # everyone present is everyone else's partner, which is the property that
    # makes "partner count" well defined here.
    partner_graph_degree: int | None = None

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

    # Async rollout (off-policy with mild staleness). When True a background
    # thread continuously fills a bounded buffer; the training thread
    # consumes whatever's ready. Trajectories carry a policy_version stamp
    # so the loss path can apply PPO importance correction.
    #
    # See "A Guide to Training Agentic LLMs on Off-Policy Surprise
    # Minimisation.md" for the math. Key correctness items:
    #   * Returns are recomputed under current θ at consumption time — we
    #     already do this in compute_loss (and compute_loss_chunked) so no
    #     extra change is required.
    #   * Action loss must use PPO-clip on the stored behaviour log-probs
    #     (act_log_probs_old). Set ppo_clip > 0 to enable.
    #   * Perception loss has no importance-weighting story; keep staleness
    #     shallow (default max_staleness=1) per the guide's §4.3.
    async_rollout: bool = False
    replay_buffer_size: int = 2  # max iter-batches queued ahead of trainer
    max_staleness: int = 1       # drop batches sampled >= this many policy versions ago
    # PPO clip threshold on action log-prob ratios. 0 = REINFORCE (sync only).
    # Recommended starting value: 0.2, but the doc notes our pretrained-KL
    # anchor (kl_coef) already constrains drift, so we may be able to run
    # looser — sweep jointly with kl_coef.
    ppo_clip: float = 0.0

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
