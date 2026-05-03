"""
Tests for marlllm.variance_decomposition.

The three required cases (fixed deterministic, deterministic-given-action,
action-independent stochastic) are exercised against the pure variance
estimator with synthetic G_t arrays. This isolates the (M-1)/(K-1)
Bessel-corrected formula from any model/env machinery — if the
estimator distinguishes signal from noise here, it does in the full
pipeline too (the rollout layer just feeds different G arrays into
this same function).

Two additional tests cover the persistence layer (round-trip JSONL)
and Bessel correction edge cases (K=1, M=1).
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

import torch

from marlllm.variance_decomposition import (
    EvalContext,
    aggregate_decomposition,
    compute_decomposition_from_returns,
    load_eval_contexts,
    write_eval_contexts,
)


# --------------------------------------------------------------------------- #
# Three required cases                                                         #
# --------------------------------------------------------------------------- #


def test_deterministic_partner_zero_signal_zero_noise():
    """Partner always emits the same response → all G_t identical → signal≈0, noise≈0.

    Mirrors a 'dark room' collapse: every (action, environment-response) pair
    produces the same return, so REINFORCE has no information to act on.
    """
    K, M = 8, 4
    returns_KM = torch.full((K, M), 5.0)
    signal, noise = compute_decomposition_from_returns(returns_KM)
    assert signal == 0.0, f"expected zero signal, got {signal}"
    assert noise == 0.0, f"expected zero noise, got {noise}"


def test_deterministic_given_action_signal_positive_noise_zero():
    """Partner response is a deterministic function of action → signal>0, noise≈0.

    This is the regime where REINFORCE has clean gradient: each action has
    a distinct expected return, with no within-action variance to wash it out.
    """
    K, M = 8, 4
    # K distinct expected returns, replicated across M (no within-action variance).
    per_action_means = torch.linspace(1.0, 8.0, K)
    returns_KM = per_action_means.unsqueeze(1).expand(K, M).contiguous()

    signal, noise = compute_decomposition_from_returns(returns_KM)
    assert noise == 0.0, f"expected zero noise, got {noise}"
    # Bessel-corrected sample variance of [1,2,...,8]:
    expected_signal = float(per_action_means.var(unbiased=True).item())
    assert math.isclose(signal, expected_signal, rel_tol=1e-9), (
        f"signal {signal} != expected {expected_signal}"
    )
    assert signal > 0.0


def test_action_independent_stochastic_signal_zero_noise_positive():
    """Partner response is independent of action, fully stochastic → signal≈0, noise>0.

    The 'entrenched diffuse non-learning' regime: returns vary, but the
    variation has nothing to do with action choice, so REINFORCE updates
    are pure noise.

    Using K large enough and a sample size big enough that the law of large
    numbers makes signal/noise estimable to within a small fraction of noise.
    """
    rng = torch.Generator().manual_seed(20260503)
    K, M = 64, 64
    sigma = 1.0
    # Each branch is an i.i.d. draw from N(0, sigma^2). Action choice has
    # no influence on the distribution.
    returns_KM = torch.randn(K, M, generator=rng) * sigma

    signal, noise = compute_decomposition_from_returns(returns_KM)
    assert noise > 0.5 * sigma**2, f"noise {noise} should be ~sigma^2={sigma**2}"
    assert noise < 1.5 * sigma**2, f"noise {noise} should be ~sigma^2={sigma**2}"
    # E[signal] = sigma^2 / M  for action-independent draws (variance of the
    # mean of M iid samples). With M=64 that's ~0.016 — should be << noise.
    assert signal < noise * 0.1, (
        f"signal {signal} should be << noise {noise} when action is irrelevant"
    )


# --------------------------------------------------------------------------- #
# Estimator edge cases                                                         #
# --------------------------------------------------------------------------- #


def test_bessel_correction_applied_for_signal():
    """Verify (K-1) divisor on signal: hand-computed for a 2x1 case is exact."""
    # K=4, M=1: noise must be 0 (no M-variance); signal uses (K-1)=3.
    returns = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    signal, noise = compute_decomposition_from_returns(returns)
    assert noise == 0.0
    # Bessel-corrected variance of [1,2,3,4] = sum((x-2.5)^2)/3 = 5/3.
    assert math.isclose(signal, 5.0 / 3.0, rel_tol=1e-9), signal


def test_bessel_correction_applied_for_noise():
    """Verify (M-1) divisor on noise: hand-computed for a 1x4 case is exact."""
    # K=1, M=4: signal must be 0 (no K-variance); noise uses (M-1)=3.
    returns = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    signal, noise = compute_decomposition_from_returns(returns)
    assert signal == 0.0
    assert math.isclose(noise, 5.0 / 3.0, rel_tol=1e-9), noise


def test_degenerate_singleton_returns_zero():
    """K=1, M=1 → both terms are 0 (no Bessel-corrected variance of one sample)."""
    signal, noise = compute_decomposition_from_returns(torch.tensor([[7.0]]))
    assert signal == 0.0 and noise == 0.0


# --------------------------------------------------------------------------- #
# Aggregation + log-scale handling                                             #
# --------------------------------------------------------------------------- #


def test_aggregate_logs_finite_when_signal_or_noise_zero():
    """log(0) must not appear in the aggregated metrics; eps clamp should kick in."""
    agg = aggregate_decomposition(per_context_signal=[0.0, 0.0], per_context_noise=[0.0, 0.0])
    for k in ("log_signal", "log_noise", "log_snr"):
        assert math.isfinite(agg[k]), f"{k} = {agg[k]} should be finite"
    assert agg["snr"] == 0.0  # 0 / eps == 0 in float


def test_aggregate_snr_matches_signal_over_noise():
    agg = aggregate_decomposition(
        per_context_signal=[2.0, 4.0],
        per_context_noise=[1.0, 1.0],
    )
    assert math.isclose(agg["signal"], 3.0)
    assert math.isclose(agg["noise"], 1.0)
    assert math.isclose(agg["snr"], 3.0)
    assert math.isclose(agg["log_snr"], math.log(3.0))


# --------------------------------------------------------------------------- #
# Persistence: EvalContext JSONL round-trip                                    #
# --------------------------------------------------------------------------- #


def test_eval_context_jsonl_round_trip(tmp_path: Path):
    contexts = [
        EvalContext(
            context_id="env_seed42_t3",
            env_name="forum",
            env_seed=42,
            role_to_name={"agent_0": "Marcus", "agent_1": "Sophia"},
            focal_env_role="agent_0",
            focal_pop_name="Marcus",
            prompts={"Marcus": "You are Marcus.", "Sophia": "You are Sophia."},
            prefix_actions=[
                {"env_role": "agent_0", "tokens": [1, 2, 3]},
                {"env_role": "agent_1", "tokens": [4, 5]},
                {"env_role": "agent_0", "tokens": [6]},
            ],
        ),
        EvalContext(
            context_id="env_seed43_t1",
            env_name="forum",
            env_seed=43,
            role_to_name={"agent_0": "Marcus", "agent_1": "Sophia"},
            focal_env_role="agent_1",
            focal_pop_name="Sophia",
            prompts={"Marcus": "", "Sophia": ""},
            prefix_actions=[{"env_role": "agent_0", "tokens": [99]}],
        ),
    ]
    path = tmp_path / "ctx.jsonl"
    write_eval_contexts(contexts, path)
    loaded = load_eval_contexts(path)
    assert len(loaded) == 2
    for orig, got in zip(contexts, loaded):
        assert got.to_json() == orig.to_json()
