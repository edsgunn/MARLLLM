"""
Loss ABC and CCSMLoss implementation.

CCSMLoss implements the three-term loss from surprise_minimisation_derivation.md §7:

    L_perc = mean(-log p_θ(x_t | x_<t))                    at OBS positions
    L_act  = mean(-log p_θ(x_t | x_<t) * stop_grad(A_t))
             - β * mean(H[p_θ(· | x_<t)])                  at ACT positions
    L_val  = mean((V_t - stop_grad(G_t))²)                  at ACT positions
    L      = α_perc * L_perc + α_act * L_act + L_val

where G_t = Σ_{s>t, σ_s=OBS} γ^(s-t) * surprise_s  (positional discount, per paper)
      A_t = -(G_t - V_t.detach())

Stop-gradient contract (all enforced here, not in the model):
- V_t is detached when computing advantages (policy gradient doesn't flow into value head)
- G_t (returns) is always pre-computed data — never in the autograd graph
- V_t is NOT detached for L_val so gradients flow through the value head's linear layer
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import torch
import torch.nn.functional as F

from marlllm.config import TrainingConfig
from marlllm.types import TokenType


# Chunk sizes for the lm_head + log-softmax / cross-entropy passes. The
# (B, T, V) logits tensor is several GiB at long T (Qwen2.5 vocab is 152k);
# materialising it pushed the 8-agent study_group runs over GPU memory in
# backward. We never form the full tensor — instead we apply lm_head in
# chunks and reduce to scalars / (B, T) tensors immediately.
#
# T_CHUNK governs the per-token CE pass (drives obs surprises + act log-probs);
# 2048 keeps a chunk's logits at ~600 MiB in bf16 for B=1.
# N_ACT_CHUNK governs the per-act-position passes (entropy + KL); these
# materialise (CHUNK, V) twice and live in fp32 inside log_softmax, so we
# keep the chunk small.
_T_CHUNK = 2048
_N_ACT_CHUNK = 256


class Loss(ABC):
    @property
    @abstractmethod
    def requires_value_function(self) -> bool: ...

    @property
    @abstractmethod
    def requires_old_log_probs(self) -> bool: ...

    @abstractmethod
    def compute_loss(
        self,
        last_hidden: torch.Tensor,           # (B, T, H)
        lm_head: Callable[[torch.Tensor], torch.Tensor],
        values: torch.Tensor,                # (B, T)
        input_ids: torch.Tensor,             # (B, T)
        token_type_mask: torch.Tensor,       # (B, T)
        agent_id_mask: torch.Tensor,         # (B, T)
        target_agent_idx: int,
        config: TrainingConfig,
        last_hidden_ref: torch.Tensor | None = None,    # (B, T, H) frozen reference
        lm_head_ref: Callable[[torch.Tensor], torch.Tensor] | None = None,
        perception_source_indices: list[int] | None = None,
    ) -> tuple[torch.Tensor, dict]: ...


class CCSMLoss(Loss):
    """
    Character-Conditioned Surprise Minimisation loss.
    Implements REINFORCE (not PPO) for the MVP.
    """

    @property
    def requires_value_function(self) -> bool:
        return True

    @property
    def requires_old_log_probs(self) -> bool:
        return False

    def compute_loss(
        self,
        last_hidden: torch.Tensor,
        lm_head: Callable[[torch.Tensor], torch.Tensor],
        values: torch.Tensor,
        input_ids: torch.Tensor,
        token_type_mask: torch.Tensor,
        agent_id_mask: torch.Tensor,
        target_agent_idx: int,
        config: TrainingConfig,
        last_hidden_ref: torch.Tensor | None = None,
        lm_head_ref: Callable[[torch.Tensor], torch.Tensor] | None = None,
        perception_source_indices: list[int] | None = None,
        seq_ids: torch.Tensor | None = None,
        act_log_probs_old: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        # Single chunked pass through lm_head produces shifted per-token
        # log p(input_ids[t+1] | hidden[t]). Both surprises and act log-probs
        # are derived from this same (B, T-1) tensor, so we never re-run the
        # head over the same hidden states twice.
        token_lp = _token_log_probs_shifted_chunked(
            last_hidden, lm_head, input_ids, _T_CHUNK,
        )
        surprises = _surprises_from_token_lp(token_lp, token_type_mask)
        # In packed mode, positions where the causal shift crosses a
        # trajectory boundary have meaningless predictions (logits[t-1] came
        # from a different trajectory than input_ids[t]). Zero those
        # surprises so they don't enter the perception loss or returns.
        if seq_ids is not None:
            boundary_shifted = _seq_boundary_shifted(seq_ids)             # (B, T-1)
            # _surprises_from_token_lp returns (B, T) with position 0 zero-padded;
            # invalid shifted positions correspond to indices 1..T-1, i.e. surprises[:, 1:].
            B = surprises.shape[0]
            zero_pad = torch.zeros(B, 1, device=surprises.device, dtype=surprises.dtype)
            valid_shift = (~boundary_shifted).to(surprises.dtype)
            surprises = torch.cat(
                [zero_pad, surprises[:, 1:] * valid_shift], dim=1
            )
        returns = _compute_returns(
            surprises, token_type_mask, config.gamma, seq_ids=seq_ids
        )

        obs_mask = token_type_mask == int(TokenType.OBS)
        if perception_source_indices is not None and len(perception_source_indices) > 0:
            allowed = torch.zeros_like(agent_id_mask, dtype=torch.bool)
            for idx in perception_source_indices:
                allowed = allowed | (agent_id_mask == idx)
            obs_mask = obs_mask & allowed
        # ACT mask: only this agent's action tokens contribute to L_act and L_val
        act_mask = (token_type_mask == int(TokenType.ACT)) & (agent_id_mask == target_agent_idx)
        if seq_ids is not None:
            # Drop ACT positions whose shifted prediction would cross a trajectory
            # boundary (logits at the boundary's preceding position came from a
            # different trajectory, so the predicted log-prob is meaningless).
            # We DO NOT drop these positions from ``obs_mask`` — their surprise
            # contribution is already zeroed above, but keeping them in the mask
            # preserves the denominator of ``surprises.mean()`` to match how the
            # padded path counts position-0-of-each-row (whose surprise is
            # similarly zero-padded by _surprises_from_token_lp).
            B, T = token_type_mask.shape
            invalid = torch.zeros_like(act_mask)
            invalid[:, 1:] = _seq_boundary_shifted(seq_ids)
            act_mask = act_mask & ~invalid

        # ---- Perception loss ----
        obs_surprises = surprises[obs_mask]
        if obs_surprises.numel() == 0:
            l_perc = torch.tensor(0.0, device=last_hidden.device, dtype=last_hidden.dtype, requires_grad=True)
        else:
            l_perc = obs_surprises.mean()

        # ---- Action loss ----
        # act_mask excludes position 0: the Trainer always prepends the character
        # prompt (as OBS or PAD depending on prompt_as_observation), so position 0
        # is never ACT. token_lp uses the causal shift (hidden[t] predicts
        # input_ids[t+1]), which matches act_mask[:,1:] — both exclude pos 0.
        if act_mask.any():
            act_returns = returns[act_mask]  # G_t at act positions

            if config.normalise_returns and act_returns.numel() > 1:
                act_returns = (act_returns - act_returns.mean()) / (act_returns.std() + 1e-8)

            # Advantages: negate so less surprise = positive advantage (§7.2 step 4)
            act_values_det = values[act_mask].detach()
            advantages = -(act_returns - act_values_det)

            # Log-probs of the sampled action tokens (from the training forward pass)
            shifted_act_mask = act_mask[:, 1:]
            act_log_probs = token_lp[shifted_act_mask]  # flat (N_act,)

            # On-policy (REINFORCE) vs off-policy (PPO-clipped importance
            # weighting). The PPO path is required under async rollout
            # because action tokens were sampled under a stale policy
            # version. The behaviour log-probs are stored per-token in
            # ``batch.act_log_probs_old`` (0 at non-ACT positions); we
            # gather the same positions used for ``act_log_probs`` above.
            ppo_clip = float(getattr(config, "ppo_clip", 0.0) or 0.0)
            if ppo_clip > 0.0 and act_log_probs_old is not None:
                act_lp_old_flat = act_log_probs_old[:, 1:][shifted_act_mask]  # (N_act,)
                # ratio = π_θ(a|s) / π_θ_old(a|s) at action tokens
                ratio = (act_log_probs - act_lp_old_flat.detach()).exp()
                adv_det = advantages.detach()
                unclipped = ratio * adv_det
                clipped = torch.clamp(ratio, 1.0 - ppo_clip, 1.0 + ppo_clip) * adv_det
                # PPO surrogate: take the *pessimistic* (min) — for both
                # signs of advantage, this is the correct "discourage
                # off-policy drift" bound.
                policy_loss = -torch.min(unclipped, clipped).mean()
                # Diagnostics: surface ratio stats + clip-fraction so we
                # can verify staleness isn't blowing the ratio horizon out.
                clip_frac = ((ratio < 1.0 - ppo_clip) | (ratio > 1.0 + ppo_clip)).float().mean().item()
                ratio_mean = ratio.detach().mean().item()
                ratio_max = ratio.detach().abs().max().item()
            else:
                policy_loss = -(act_log_probs * advantages.detach()).mean()
                clip_frac = 0.0
                ratio_mean = 1.0
                ratio_max = 1.0

            # Entropy bonus: H[p_θ] at action positions (§6.1).
            # We index hidden states at the act positions first, then push only
            # those through lm_head in chunks — never materialising the full
            # (B, T-1, V) entropy tensor.
            act_hidden_cur = last_hidden[:, :-1, :][shifted_act_mask]  # (N_act, H)
            entropy = _action_entropy_from_hidden_chunked(
                act_hidden_cur, lm_head, _N_ACT_CHUNK,
            )

            l_act = policy_loss - config.beta * entropy

            # ---- Value loss ----
            # Use the same act_returns tensor as the advantage computation — if
            # normalise_returns is True, act_returns is already normalised here,
            # which keeps the value-head target on the same scale as the advantages.
            # Using the raw returns[act_mask] when normalise_returns=True would train
            # the value head to predict ~250 while advantages are computed in ~[-1,1],
            # making the baseline actively harmful.
            act_values = values[act_mask]
            l_val = F.mse_loss(act_values, act_returns.detach())
        else:
            l_act = torch.tensor(0.0, device=last_hidden.device, dtype=last_hidden.dtype)
            l_val = torch.tensor(0.0, device=last_hidden.device, dtype=last_hidden.dtype)
            advantages = torch.zeros(0, device=last_hidden.device, dtype=last_hidden.dtype)
            entropy = torch.tensor(0.0, device=last_hidden.device, dtype=last_hidden.dtype)
            act_returns = torch.zeros(0, device=last_hidden.device, dtype=last_hidden.dtype)
            act_hidden_cur = None
            clip_frac = 0.0
            ratio_mean = 1.0
            ratio_max = 1.0

        # ---- KL penalty ----
        # KL(π_θ || π_ref) at ACT positions, using the causal shift.
        # Only computed when a reference model was provided and kl_coef > 0.
        if (
            last_hidden_ref is not None
            and lm_head_ref is not None
            and (config.kl_coef > 0.0 or getattr(config, "always_log_kl", False))
            and act_mask.any()
        ):
            shifted_act_mask = act_mask[:, 1:]
            act_hidden_ref = last_hidden_ref[:, :-1, :][shifted_act_mask]  # (N_act, H)
            kl = _kl_at_act_chunked(
                act_hidden_cur, lm_head, act_hidden_ref, lm_head_ref, _N_ACT_CHUNK,
            )
        else:
            kl = torch.tensor(0.0, device=last_hidden.device, dtype=last_hidden.dtype)

        # Surprise distribution percentiles (Phase A §3.3): cheap to compute and
        # essential for diagnosing dark-room collapse vs healthy compression.
        if obs_surprises.numel() > 0:
            surp_q = torch.quantile(
                obs_surprises.float().detach(),
                torch.tensor([0.25, 0.5, 0.75, 0.95], device=obs_surprises.device),
            ).tolist()
            surp_p25, surp_p50, surp_p75, surp_p95 = surp_q
            surp_max = obs_surprises.detach().max().item()
        else:
            surp_p25 = surp_p50 = surp_p75 = surp_p95 = surp_max = 0.0

        metrics = {
            "mean_surprise": obs_surprises.mean().item() if obs_surprises.numel() > 0 else 0.0,
            "surprise_p25": surp_p25,
            "surprise_p50": surp_p50,
            "surprise_p75": surp_p75,
            "surprise_p95": surp_p95,
            "surprise_max": surp_max,
            "mean_return": act_returns.mean().item() if act_returns.numel() > 0 else 0.0,
            "return_abs_max": act_returns.abs().max().item() if act_returns.numel() > 0 else 0.0,
            "mean_advantage": advantages.mean().item() if advantages.numel() > 0 else 0.0,
            "entropy": entropy.item() if isinstance(entropy, torch.Tensor) else float(entropy),
            "perc_loss": l_perc.item(),
            "act_loss": l_act.item(),
            "value_loss": l_val.item(),
            "kl": kl.item(),
            "ppo/clip_frac": clip_frac,
            "ppo/ratio_mean": ratio_mean,
            "ppo/ratio_max": ratio_max,
        }

        total = (
            config.alpha_perc * l_perc
            + config.alpha_act * l_act
            + config.alpha_val * l_val
            + config.kl_coef * kl
        )
        return total, metrics

    def compute_loss_chunked(
        self,
        agent,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_mask: torch.Tensor,
        agent_id_mask: torch.Tensor,
        target_agent_idx: int,
        config: TrainingConfig,
        backward_scale: float,
        perception_source_indices: list[int] | None = None,
    ) -> tuple[float, dict]:
        """Sequence-chunked counterpart of ``compute_loss``.

        Splits the per-episode forward+backward into chunks of
        ``config.seq_chunk_size`` tokens. A no-grad streaming pass first
        computes global surprises / returns / advantages / counts; a second
        pass forwards each chunk with autograd, computes that chunk's
        contribution to the global mean of each loss term, and calls
        ``(loss * backward_scale).backward()`` immediately so the chunk's
        autograd graph can be freed before the next chunk forwards.

        The KV cache from prior chunks is detached, so cross-chunk attention
        gradients are not captured. Within a chunk the gradient is exact.
        """
        chunk_size = config.seq_chunk_size
        assert chunk_size is not None and chunk_size > 0, "seq_chunk_size must be set"
        device = input_ids.device
        dtype_h = next(agent._backbone.parameters()).dtype
        B, T = input_ids.shape

        # ---------------- Pass 1: no-grad streaming -----------------------
        # Accumulate token_lp (B, T-1) and values (B, T) across chunks.
        token_lp_full = input_ids.new_zeros((B, T - 1), dtype=dtype_h) if T > 1 else input_ids.new_zeros((B, 0), dtype=dtype_h)
        values_full = input_ids.new_zeros((B, T), dtype=dtype_h)
        with torch.no_grad():
            for s, e, hidden_chunk, values_chunk in agent.evaluate_hidden_chunked(
                input_ids, attention_mask, chunk_size,
            ):
                # token_lp for shifted predictions hidden[s..min(e,T-1)) → input[s+1..min(e,T))
                pred_end = min(e, T - 1)
                if pred_end > s:
                    L = pred_end - s
                    targets = input_ids[:, s + 1 : pred_end + 1]
                    hidden_pred = hidden_chunk[:, :L, :]
                    for vs in range(0, L, _T_CHUNK):
                        ve = min(vs + _T_CHUNK, L)
                        logits_chunk = agent.lm_head(hidden_pred[:, vs:ve, :])
                        V = logits_chunk.shape[-1]
                        nll = F.cross_entropy(
                            logits_chunk.reshape(-1, V),
                            targets[:, vs:ve].reshape(-1),
                            reduction="none",
                        ).reshape(B, ve - vs)
                        token_lp_full[:, s + vs : s + ve] = -nll.to(dtype_h)
                values_full[:, s:e] = values_chunk.to(dtype_h)

        # Surprises, returns, masks (no_grad — these are pre-computed data).
        surprises = _surprises_from_token_lp(token_lp_full, token_type_mask)
        returns = _compute_returns(surprises, token_type_mask, config.gamma)

        obs_mask = token_type_mask == int(TokenType.OBS)
        if perception_source_indices is not None and len(perception_source_indices) > 0:
            allowed = torch.zeros_like(agent_id_mask, dtype=torch.bool)
            for idx in perception_source_indices:
                allowed = allowed | (agent_id_mask == idx)
            obs_mask = obs_mask & allowed
        act_mask = (token_type_mask == int(TokenType.ACT)) & (agent_id_mask == target_agent_idx)

        # Shifted-view (positions 1..T-1) masks for slicing per-chunk
        # contributions. Note: ``n_obs_total`` matches the single-shot
        # path's denominator, which is ``obs_mask.sum()`` over *absolute*
        # positions — that count includes any OBS at position 0 (whose
        # surprise is 0 by construction). Using the shifted-view count
        # would silently drop that off-by-one and bias L_perc upwards.
        # ``act_mask[:, 0]`` is always False (position 0 is never ACT),
        # so absolute and shifted counts coincide for the ACT side.
        shifted_obs_mask = obs_mask[:, 1:]
        shifted_act_mask = act_mask[:, 1:]
        n_obs_total = int(obs_mask.sum().item())
        n_act_total = int(act_mask.sum().item())

        # Global act_returns + advantages, optionally normalised.
        if act_mask.any():
            act_returns_global = returns[act_mask]
            if config.normalise_returns and act_returns_global.numel() > 1:
                act_returns_global = (act_returns_global - act_returns_global.mean()) / (act_returns_global.std() + 1e-8)
            act_values_global_det = values_full[act_mask].detach()
            advantages_global = -(act_returns_global - act_values_global_det)
            # Per-position scatter buffers so each chunk can slice [s:e] cheaply.
            advantages_per_pos = torch.zeros(B, T, device=device, dtype=dtype_h)
            advantages_per_pos[act_mask] = advantages_global.to(dtype_h)
            returns_per_pos_norm = torch.zeros(B, T, device=device, dtype=dtype_h)
            returns_per_pos_norm[act_mask] = act_returns_global.to(dtype_h)
        else:
            act_returns_global = torch.zeros(0, device=device, dtype=dtype_h)
            advantages_global = torch.zeros(0, device=device, dtype=dtype_h)
            advantages_per_pos = torch.zeros(B, T, device=device, dtype=dtype_h)
            returns_per_pos_norm = torch.zeros(B, T, device=device, dtype=dtype_h)

        # OBS surprises (for diagnostics). Mirror the single-shot path's
        # ``surprises[obs_mask]`` selection — over absolute positions, so
        # any OBS at position 0 contributes a 0 to both sum and count.
        obs_surprises_flat = surprises[obs_mask]

        # ---------------- Pass 2: per-chunk grad + backward ---------------
        agent.train_mode()
        total_loss_scalar = 0.0
        l_perc_acc = 0.0
        l_act_policy_acc = 0.0
        entropy_acc = 0.0
        kl_acc = 0.0
        l_val_acc = 0.0

        keep_ref = (config.kl_coef > 0.0 or getattr(config, "always_log_kl", False)) and bool(act_mask.any()) and getattr(agent, "_keep_ref_model", False)
        train_iter = agent.evaluate_hidden_chunked(input_ids, attention_mask, chunk_size)
        ref_iter = (
            agent.evaluate_hidden_ref_chunked(input_ids, attention_mask, chunk_size)
            if keep_ref else None
        )

        denom_obs = max(n_obs_total, 1)
        denom_act = max(n_act_total, 1)

        for s, e, hidden_chunk, values_chunk in train_iter:
            ref_chunk = None
            if ref_iter is not None:
                _rs, _re, ref_chunk = next(ref_iter)
                assert (_rs, _re) == (s, e), "train/ref chunk iterators desynchronised"

            chunk_loss = hidden_chunk.new_zeros(())

            # Predictor side: this chunk's hidden states at positions
            # [s, predictor_end) drive log-prob / entropy / KL for predicted
            # absolute positions [s+1, predictor_end+1). The very last
            # position T-1 has no target and is dropped.
            predictor_end = min(e, T - 1)
            L_pred = max(0, predictor_end - s)
            if L_pred > 0:
                hidden_pred = hidden_chunk[:, :L_pred, :]
                targets = input_ids[:, s + 1 : predictor_end + 1]
                token_lp_chunk = hidden_chunk.new_empty((B, L_pred))
                for vs in range(0, L_pred, _T_CHUNK):
                    ve = min(vs + _T_CHUNK, L_pred)
                    logits_chunk = agent.lm_head(hidden_pred[:, vs:ve, :])
                    V = logits_chunk.shape[-1]
                    nll = F.cross_entropy(
                        logits_chunk.reshape(-1, V),
                        targets[:, vs:ve].reshape(-1),
                        reduction="none",
                    ).reshape(B, ve - vs)
                    token_lp_chunk[:, vs:ve] = -nll

                shifted_obs_chunk_local = shifted_obs_mask[:, s : s + L_pred]
                shifted_act_chunk_local = shifted_act_mask[:, s : s + L_pred]

                # L_perc contribution (predicted absolute positions [s+1, predictor_end+1))
                if shifted_obs_chunk_local.any():
                    perc_sum = (-token_lp_chunk * shifted_obs_chunk_local.to(token_lp_chunk.dtype)).sum()
                    l_perc_chunk = perc_sum / denom_obs
                    chunk_loss = chunk_loss + config.alpha_perc * l_perc_chunk
                    l_perc_acc += float(l_perc_chunk.detach().item())

                if shifted_act_chunk_local.any():
                    chunk_adv = advantages_per_pos[:, s + 1 : predictor_end + 1].to(token_lp_chunk.dtype)
                    chunk_act_float = shifted_act_chunk_local.to(token_lp_chunk.dtype)
                    policy_sum = -(token_lp_chunk * chunk_adv * chunk_act_float).sum()
                    l_policy_chunk = policy_sum / denom_act
                    l_act_policy_acc += float(l_policy_chunk.detach().item())

                    act_hidden_chunk = hidden_pred[shifted_act_chunk_local]  # (N_chunk_pred, H)
                    ent_sum_chunk = act_hidden_chunk.new_zeros(())
                    for ns in range(0, act_hidden_chunk.shape[0], _N_ACT_CHUNK):
                        ne = min(ns + _N_ACT_CHUNK, act_hidden_chunk.shape[0])
                        log_p = F.log_softmax(agent.lm_head(act_hidden_chunk[ns:ne]), dim=-1)
                        ent_sum_chunk = ent_sum_chunk - (log_p.exp() * log_p).sum()
                    l_ent_chunk = ent_sum_chunk / denom_act
                    entropy_acc += float(l_ent_chunk.detach().item())

                    chunk_loss = chunk_loss + config.alpha_act * (l_policy_chunk - config.beta * l_ent_chunk)

                    if keep_ref and ref_chunk is not None:
                        ref_pred = ref_chunk[:, :L_pred, :]
                        act_hidden_ref_chunk = ref_pred[shifted_act_chunk_local]
                        # lm_head is not LoRA-wrapped in any current config, so
                        # ``lm_head_ref`` (which wraps the call in disable_adapter)
                        # is functionally identical to ``lm_head``. Avoid the
                        # context-manager toggle inside this inner loop — it
                        # caused a "backward through graph twice" error when the
                        # input pattern interacted with gradient checkpointing.
                        kl_sum_chunk = act_hidden_chunk.new_zeros(())
                        for ns in range(0, act_hidden_chunk.shape[0], _N_ACT_CHUNK):
                            ne = min(ns + _N_ACT_CHUNK, act_hidden_chunk.shape[0])
                            log_p = F.log_softmax(agent.lm_head(act_hidden_chunk[ns:ne]), dim=-1)
                            log_p_ref = F.log_softmax(agent.lm_head(act_hidden_ref_chunk[ns:ne]), dim=-1)
                            kl_sum_chunk = kl_sum_chunk + (log_p.exp() * (log_p - log_p_ref)).sum()
                        l_kl_chunk = kl_sum_chunk / denom_act
                        kl_acc += float(l_kl_chunk.detach().item())
                        chunk_loss = chunk_loss + config.kl_coef * l_kl_chunk

            # Value side: this chunk's values at absolute positions [s, e)
            # train the value head on returns at any act positions it owns.
            # Disjoint from the predictor side (predictor is one position
            # earlier), so we account for them separately.
            chunk_act_abs_mask = act_mask[:, s:e]
            if chunk_act_abs_mask.any():
                chunk_values_at_act = values_chunk[chunk_act_abs_mask]
                chunk_returns_at_act = returns_per_pos_norm[:, s:e][chunk_act_abs_mask].to(chunk_values_at_act.dtype)
                val_sum = ((chunk_values_at_act - chunk_returns_at_act.detach()) ** 2).sum()
                l_val_chunk = val_sum / denom_act
                l_val_acc += float(l_val_chunk.detach().item())
                chunk_loss = chunk_loss + config.alpha_val * l_val_chunk

            if chunk_loss.requires_grad:
                (chunk_loss * backward_scale).backward()
                total_loss_scalar += float(chunk_loss.detach().item())
            # Free chunk-local tensors before next iteration.
            del hidden_chunk, values_chunk
            if ref_chunk is not None:
                del ref_chunk

        # Drain reference iterator if any chunks were skipped.
        if ref_iter is not None:
            for _ in ref_iter:
                pass

        # Diagnostics — mirror the single-shot path.
        if obs_surprises_flat.numel() > 0:
            surp_q = torch.quantile(
                obs_surprises_flat.float().detach(),
                torch.tensor([0.25, 0.5, 0.75, 0.95], device=obs_surprises_flat.device),
            ).tolist()
            surp_p25, surp_p50, surp_p75, surp_p95 = surp_q
            surp_max = obs_surprises_flat.detach().max().item()
            mean_surprise = obs_surprises_flat.mean().item()
        else:
            surp_p25 = surp_p50 = surp_p75 = surp_p95 = surp_max = 0.0
            mean_surprise = 0.0

        l_act_total = l_act_policy_acc - config.beta * entropy_acc
        metrics = {
            "mean_surprise": mean_surprise,
            "surprise_p25": surp_p25,
            "surprise_p50": surp_p50,
            "surprise_p75": surp_p75,
            "surprise_p95": surp_p95,
            "surprise_max": surp_max,
            "mean_return": float(act_returns_global.mean().item()) if act_returns_global.numel() > 0 else 0.0,
            "return_abs_max": float(act_returns_global.abs().max().item()) if act_returns_global.numel() > 0 else 0.0,
            "mean_advantage": float(advantages_global.mean().item()) if advantages_global.numel() > 0 else 0.0,
            "entropy": entropy_acc,
            "perc_loss": l_perc_acc,
            "act_loss": l_act_total,
            "value_loss": l_val_acc,
            "kl": kl_acc,
        }
        return total_loss_scalar, metrics


# ------------------------------------------------------------------ #
# Internal helpers                                                     #
# ------------------------------------------------------------------ #

def _compute_obs_surprises(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    token_type_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute per-token negative log-probability at OBS positions; 0 elsewhere.

    Causal shift: logits[:, t] predicts input_ids[:, t+1].
    We shift left by one and align: surprise[t] = -log p(input_ids[t] | x_{<t}).
    Position 0 has no ground-truth predecessor — it's left as 0.
    """
    B, T, V = logits.shape

    # Shift: logits[:, :-1] predicts input_ids[:, 1:]
    shifted_logits = logits[:, :-1, :]      # (B, T-1, V)
    shifted_targets = input_ids[:, 1:]      # (B, T-1)
    shifted_types = token_type_mask[:, 1:]  # (B, T-1)

    # Use cross_entropy (fused NLL kernel) instead of log_softmax + gather.
    # log_softmax would allocate a full (B, T-1, V) output tensor (~9 GB for
    # Qwen vocab at typical batch sizes); cross_entropy never materialises it.
    token_lp = -F.cross_entropy(
        shifted_logits.reshape(-1, V),
        shifted_targets.reshape(-1),
        reduction="none",
    ).reshape(B, T - 1)  # (B, T-1)

    # Mask to OBS positions only
    obs_mask = shifted_types == int(TokenType.OBS)
    surprises_shifted = torch.where(obs_mask, -token_lp, torch.zeros_like(token_lp))

    # Pad back to (B, T) with 0 at position 0
    surprises = torch.cat(
        [torch.zeros(B, 1, device=logits.device, dtype=logits.dtype), surprises_shifted], dim=1
    )
    return surprises  # (B, T)


def _compute_returns(
    surprises: torch.Tensor,
    token_type_mask: torch.Tensor,
    gamma: float,
    seq_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute discounted future observation surprise G_t at ACT positions.

    G_t = Σ_{s > t, σ_s = OBS} γ^(s-t) · surprises[s]

    Discount exponent is positional (s-t in token sequence), as written in the
    paper — includes gaps where action or prompt tokens appear between t and s.

    Implemented via a backwards scan over the sequence dimension.

    When ``seq_ids`` is provided (packed mode), the running accumulator is
    reset at trajectory boundaries so returns never bleed across packed
    trajectories. Whenever ``seq_ids[:, t] != seq_ids[:, t+1]`` the running
    sum is zeroed before incorporating position ``t`` (backward scan).
    """
    B, T = surprises.shape
    returns = torch.zeros_like(surprises)

    running = torch.zeros(B, device=surprises.device, dtype=surprises.dtype)

    for t in range(T - 1, -1, -1):
        if seq_ids is not None and t < T - 1:
            same_seq = (seq_ids[:, t] == seq_ids[:, t + 1]).to(dtype=surprises.dtype)
            running = running * same_seq

        obs_here = (token_type_mask[:, t] == int(TokenType.OBS)).to(dtype=surprises.dtype)
        act_here = (token_type_mask[:, t] == int(TokenType.ACT)).to(dtype=surprises.dtype)

        # If OBS: add this surprise to the running discounted sum
        running = obs_here * surprises[:, t] + gamma * running

        # If ACT: record running sum as the return for this position
        # (running already accumulated obs surprises strictly after t due to
        #  the backward direction — here we capture what's been accumulated
        #  so far, which corresponds to future positions relative to t)
        returns[:, t] = act_here * running

    return returns  # (B, T) — non-zero only at ACT positions


def _seq_boundary_shifted(seq_ids: torch.Tensor) -> torch.Tensor:
    """Boolean (B, T-1) mask: True where the shifted prediction (logits[t-1]
    predicts input_ids[t]) crosses a packed-trajectory boundary and must be
    excluded from any shifted loss. Returns all-False if ``seq_ids`` is
    constant (degenerate single-trajectory packed batch)."""
    return seq_ids[:, 1:] != seq_ids[:, :-1]


def _build_block_diagonal_causal_mask(
    seq_ids: torch.Tensor,
    *,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """4D additive attention mask for a packed batch.

    Returns shape ``(B, 1, T, T)``. Position ``(i, j)`` is 0 when token ``i``
    may attend to token ``j`` (same trajectory *and* ``j <= i``); otherwise
    ``-inf``. Memory cost is ``B × T × T`` of the chosen dtype: at T=8192
    that's 128 MiB in bf16 per batch, kept live on the autograd graph until
    backward releases it. For longer T prefer flash-attn varlen via
    ``cu_seqlens`` (one-line swap once ``flash-attn`` is in the venv).
    """
    if dtype is None:
        dtype = torch.float32
    device = seq_ids.device
    B, T = seq_ids.shape
    arange = torch.arange(T, device=device)
    causal = arange.unsqueeze(0) <= arange.unsqueeze(1)              # (T, T) — j <= i
    same_seq = seq_ids.unsqueeze(2) == seq_ids.unsqueeze(1)          # (B, T, T)
    allow = same_seq & causal                                         # (B, T, T)
    mask = torch.zeros(B, T, T, dtype=dtype, device=device)
    mask.masked_fill_(~allow, float("-inf"))
    return mask.unsqueeze(1)                                          # (B, 1, T, T)


def _act_log_probs(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    act_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Log-probabilities of sampled action tokens from the training forward pass.
    Uses the causal shift: logits[:, t-1] predicts input_ids[:, t].
    Returns a flat 1-D tensor of log-probs at act_mask positions.
    """
    B, T, V = logits.shape
    # Shift: logits[:, :-1] predicts input_ids[:, 1:]
    shifted_logits = logits[:, :-1, :]    # (B, T-1, V)
    shifted_targets = input_ids[:, 1:]    # (B, T-1)
    shifted_act_mask = act_mask[:, 1:]    # (B, T-1)

    # Same fused cross_entropy trick as _compute_obs_surprises — avoids
    # allocating the full (B, T-1, V) log_softmax output.
    token_lp = -F.cross_entropy(
        shifted_logits.reshape(-1, V),
        shifted_targets.reshape(-1),
        reduction="none",
    ).reshape(B, T - 1)  # (B, T-1)

    return token_lp[shifted_act_mask]  # flat (N_act,)


def _token_log_probs_shifted_chunked(
    last_hidden: torch.Tensor,
    lm_head: Callable[[torch.Tensor], torch.Tensor],
    input_ids: torch.Tensor,
    chunk_t: int,
) -> torch.Tensor:
    """Per-token log p(input_ids[t+1] | hidden[t]) for t = 0 .. T-2.

    Applies lm_head in T-axis chunks so the (B, T, V) logits tensor is never
    materialised. Returns a (B, T-1) tensor in the hidden-state dtype.
    """
    B, T, _H = last_hidden.shape
    targets = input_ids[:, 1:]  # (B, T-1)
    out = last_hidden.new_empty((B, T - 1))
    for s in range(0, T - 1, chunk_t):
        e = min(s + chunk_t, T - 1)
        logits_chunk = lm_head(last_hidden[:, s:e, :])  # (B, e-s, V)
        V = logits_chunk.shape[-1]
        # cross_entropy is fused: never allocates the full log_softmax tensor.
        nll = F.cross_entropy(
            logits_chunk.reshape(-1, V),
            targets[:, s:e].reshape(-1),
            reduction="none",
        ).reshape(B, e - s)
        out[:, s:e] = -nll  # log p
    return out


def _surprises_from_token_lp(
    token_lp: torch.Tensor,           # (B, T-1) log-probs
    token_type_mask: torch.Tensor,    # (B, T)
) -> torch.Tensor:
    """Negative log-prob at OBS positions, padded to (B, T) with 0 at pos 0."""
    B, _ = token_lp.shape
    shifted_types = token_type_mask[:, 1:]
    obs_mask = shifted_types == int(TokenType.OBS)
    surprises_shifted = torch.where(
        obs_mask, -token_lp, torch.zeros_like(token_lp)
    )
    return torch.cat(
        [
            torch.zeros(B, 1, device=token_lp.device, dtype=token_lp.dtype),
            surprises_shifted,
        ],
        dim=1,
    )


def _action_entropy_from_hidden_chunked(
    act_hidden: torch.Tensor,                                    # (N_act, H)
    lm_head: Callable[[torch.Tensor], torch.Tensor],
    n_chunk: int,
) -> torch.Tensor:
    """Mean entropy at the act positions, computing logits in N-axis chunks."""
    N = act_hidden.shape[0]
    if N == 0:
        return act_hidden.new_zeros(())
    ent_acc = act_hidden.new_zeros(())
    for start in range(0, N, n_chunk):
        end = min(start + n_chunk, N)
        log_p = F.log_softmax(lm_head(act_hidden[start:end]), dim=-1)
        ent_acc = ent_acc - (log_p.exp() * log_p).sum()
    return ent_acc / N


def _kl_at_act_chunked(
    act_hidden_cur: torch.Tensor,                                # (N_act, H)
    lm_head: Callable[[torch.Tensor], torch.Tensor],
    act_hidden_ref: torch.Tensor,                                # (N_act, H)
    lm_head_ref: Callable[[torch.Tensor], torch.Tensor],
    n_chunk: int,
) -> torch.Tensor:
    """KL(π_θ || π_ref) averaged over act positions, in N-axis chunks.

    Mathematically identical to a single-shot computation; chunking only
    changes the order of floating-point summation across positions.
    """
    N = act_hidden_cur.shape[0]
    if N == 0:
        return act_hidden_cur.new_zeros(())
    kl_acc = act_hidden_cur.new_zeros(())
    for start in range(0, N, n_chunk):
        end = min(start + n_chunk, N)
        log_p = F.log_softmax(lm_head(act_hidden_cur[start:end]), dim=-1)
        log_p_ref = F.log_softmax(lm_head_ref(act_hidden_ref[start:end]), dim=-1)
        kl_acc = kl_acc + (log_p.exp() * (log_p - log_p_ref)).sum(dim=-1).sum()
    return kl_acc / N


def _action_entropy(
    logits: torch.Tensor,
    act_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Mean entropy H[p_θ(· | x_{<t})] at act_mask positions.
    Uses shifted logits so position t sees the logits that generated x_t.
    """
    # Shift: logits[:, :-1] predicts the token at position t (which is act_mask[:, 1:])
    shifted_logits = logits[:, :-1, :]
    shifted_act_mask = act_mask[:, 1:]

    act_logits = shifted_logits[shifted_act_mask]  # (N_act, V)
    N = act_logits.shape[0]
    if N == 0:
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

    # Process in chunks to avoid allocating (N_act, V) × 2 for log_probs + probs.
    # With Qwen vocab (150k) even a few thousand act positions is several GB.
    CHUNK = 256
    entropy_acc = act_logits.new_zeros(())
    for start in range(0, N, CHUNK):
        chunk = act_logits[start : start + CHUNK]  # (CHUNK, V)
        log_p = F.log_softmax(chunk, dim=-1)       # (CHUNK, V) — freed each iter
        entropy_acc = entropy_acc + -(log_p.exp() * log_p).sum()
    return entropy_acc / N
