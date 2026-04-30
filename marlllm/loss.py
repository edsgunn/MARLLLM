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

import torch
import torch.nn.functional as F

from marlllm.config import TrainingConfig
from marlllm.types import TokenType


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
        logits: torch.Tensor,           # (B, T, V)
        values: torch.Tensor,           # (B, T)
        input_ids: torch.Tensor,        # (B, T)
        token_type_mask: torch.Tensor,  # (B, T)
        agent_id_mask: torch.Tensor,    # (B, T)
        target_agent_idx: int,
        config: TrainingConfig,
        ref_logits: torch.Tensor | None = None,  # (B, T, V) frozen reference, optional
        perception_source_indices: list[int] | None = None,  # source-agent gating for L_perc
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
        logits: torch.Tensor,
        values: torch.Tensor,
        input_ids: torch.Tensor,
        token_type_mask: torch.Tensor,
        agent_id_mask: torch.Tensor,
        target_agent_idx: int,
        config: TrainingConfig,
        ref_logits: torch.Tensor | None = None,
        perception_source_indices: list[int] | None = None,
    ) -> tuple[torch.Tensor, dict]:
        surprises = _compute_obs_surprises(logits, input_ids, token_type_mask)
        returns = _compute_returns(surprises, token_type_mask, config.gamma)

        obs_mask = token_type_mask == int(TokenType.OBS)
        if perception_source_indices is not None and len(perception_source_indices) > 0:
            allowed = torch.zeros_like(agent_id_mask, dtype=torch.bool)
            for idx in perception_source_indices:
                allowed = allowed | (agent_id_mask == idx)
            obs_mask = obs_mask & allowed
        # ACT mask: only this agent's action tokens contribute to L_act and L_val
        act_mask = (token_type_mask == int(TokenType.ACT)) & (agent_id_mask == target_agent_idx)

        # ---- Perception loss ----
        obs_surprises = surprises[obs_mask]
        if obs_surprises.numel() == 0:
            l_perc = torch.tensor(0.0, device=logits.device, dtype=logits.dtype, requires_grad=True)
        else:
            l_perc = obs_surprises.mean()

        # ---- Action loss ----
        # act_mask already excludes position 0 because the Trainer always prepends
        # the character prompt as PAD-typed tokens, so position 0 is never ACT.
        # _act_log_probs uses the causal shift (logits[:,:-1] → input_ids[:,1:]),
        # which matches act_mask[:,1:] — both exclude position 0 naturally.
        if act_mask.any():
            act_returns = returns[act_mask]  # G_t at act positions

            if config.normalise_returns and act_returns.numel() > 1:
                act_returns = (act_returns - act_returns.mean()) / (act_returns.std() + 1e-8)

            # Advantages: negate so less surprise = positive advantage (§7.2 step 4)
            act_values_det = values[act_mask].detach()
            advantages = -(act_returns - act_values_det)

            # Log-probs of the sampled action tokens (from the training forward pass)
            act_log_probs = _act_log_probs(logits, input_ids, act_mask)

            policy_loss = -(act_log_probs * advantages.detach()).mean()

            # Entropy bonus: H[p_θ] at action positions (§6.1)
            entropy = _action_entropy(logits, act_mask)

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
            l_act = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
            l_val = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
            advantages = torch.zeros(0, device=logits.device, dtype=logits.dtype)
            entropy = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
            act_returns = torch.zeros(0, device=logits.device, dtype=logits.dtype)

        # ---- KL penalty ----
        # KL(π_θ || π_ref) at ACT positions, using the causal shift.
        # Only computed when a reference model was provided and kl_coef > 0.
        if ref_logits is not None and config.kl_coef > 0.0 and act_mask.any():
            shifted_act_mask = act_mask[:, 1:]
            act_logits_cur = logits[:, :-1, :][shifted_act_mask]       # (N_act, V)
            act_logits_ref = ref_logits[:, :-1, :][shifted_act_mask]   # (N_act, V)
            log_p = F.log_softmax(act_logits_cur, dim=-1)
            log_p_ref = F.log_softmax(act_logits_ref, dim=-1)
            kl = (log_p.exp() * (log_p - log_p_ref)).sum(dim=-1).mean()
        else:
            kl = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

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
        }

        total = (
            config.alpha_perc * l_perc
            + config.alpha_act * l_act
            + config.alpha_val * l_val
            + config.kl_coef * kl
        )
        return total, metrics


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
) -> torch.Tensor:
    """
    Compute discounted future observation surprise G_t at ACT positions.

    G_t = Σ_{s > t, σ_s = OBS} γ^(s-t) · surprises[s]

    Discount exponent is positional (s-t in token sequence), as written in the
    paper — includes gaps where action or prompt tokens appear between t and s.

    Implemented via a backwards scan over the sequence dimension.
    """
    B, T = surprises.shape
    returns = torch.zeros_like(surprises)

    running = torch.zeros(B, device=surprises.device, dtype=surprises.dtype)

    for t in range(T - 1, -1, -1):
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
