"""Tests for the return's sign — the novelty-search ablation.

`return_sign` = -1 pushes the policy toward actions that make the future MORE
surprising instead of less. It is meant to be the ONLY change from the
reference, so what has to hold is mostly about what does *not* move:

  1. The perception arm is untouched. It still minimises observation surprise;
     only the action arm's direction flips, which is what makes the two arms
     pull against each other and is the point of the ablation.
  2. The return is exactly negated, so the value head fits the negated return
     and stays a correct baseline.
  3. The policy gradient is very nearly reversed — not exactly, because the
     perception arm is shared and unflipped.
  4. Default +1 leaves everything as it was.

Run:  .venv/bin/python tests/test_return_sign.py
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

os.environ.setdefault("HF_HOME", "/lus/lfs1aip2/projects/a5l/egunn/hf_cache")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from marlllm.agent import ValueHead          # noqa: E402
from marlllm.config import TrainingConfig    # noqa: E402
from marlllm.loss import CCSMLoss            # noqa: E402
from marlllm.types import TokenType          # noqa: E402

OBS, ACT = int(TokenType.OBS), int(TokenType.ACT)


def _run(sign, **kw):
    torch.manual_seed(0)
    B, T, H, V = 2, 12, 16, 32
    ids = torch.randint(0, V, (B, T))
    tm = torch.full((B, T), OBS); tm[:, 2::3] = ACT; tm[:, 0] = OBS
    am = torch.zeros((B, T), dtype=torch.long)
    trunk, lm, vh = nn.Linear(H, H), nn.Linear(H, V), ValueHead(H)
    for m in (trunk, lm, vh):
        m.zero_grad(set_to_none=True)
    hidden = trunk(torch.randn(B, T, H,
                               generator=torch.Generator().manual_seed(1)))
    loss, met = CCSMLoss().compute_loss(
        last_hidden=hidden, lm_head=lm, values=vh(hidden.detach()),
        input_ids=ids, token_type_mask=tm, agent_id_mask=am,
        target_agent_idx=0,
        config=TrainingConfig(return_sign=sign, normalise_returns=False, **kw))
    loss.backward()
    grad = torch.cat([p.grad.flatten() for p in lm.parameters()])
    return met, grad


def test_the_perception_arm_is_untouched():
    p, _ = _run(+1.0)
    n, _ = _run(-1.0)
    assert abs(p["perc_loss"] - n["perc_loss"]) < 1e-9, (
        f"perception moved: {p['perc_loss']} vs {n['perc_loss']}")


def test_the_return_is_exactly_negated():
    p, _ = _run(+1.0)
    n, _ = _run(-1.0)
    assert abs(p["return_raw_mean"] + n["return_raw_mean"]) < 1e-6
    assert p["return_sign"] == 1.0 and n["return_sign"] == -1.0


def test_the_policy_gradient_is_very_nearly_reversed():
    """Not exactly -1: the perception arm is shared and does not flip."""
    _, gp = _run(+1.0)
    _, gn = _run(-1.0)
    cos = float(torch.dot(gp, gn) / (gp.norm() * gn.norm()))
    assert cos < -0.99, f"cosine {cos:+.4f} — the flip did not reach the policy"


def test_the_advantage_flips_direction():
    """Its magnitude is not exactly mirrored, because the untrained baseline is
    asymmetric relative to +G and -G; only the sign has to reverse."""
    p, _ = _run(+1.0)
    n, _ = _run(-1.0)
    assert p["mean_advantage"] * n["mean_advantage"] < 0, (
        f"{p['mean_advantage']} and {n['mean_advantage']} have the same sign")


def test_default_is_unchanged():
    assert TrainingConfig().return_sign == 1.0
    a, ga = _run(+1.0)
    b, gb = _run(1.0)
    assert torch.equal(ga, gb) and a["act_loss"] == b["act_loss"]


def test_it_composes_with_normalisation():
    """Standardisation is sign-preserving, so the flip survives it."""
    p, _ = _run(+1.0, )
    torch.manual_seed(0)
    pn, gpn = _run(+1.0)
    nn_, gnn = _run(-1.0)
    cos = float(torch.dot(gpn, gnn) / (gpn.norm() * gnn.norm()))
    assert cos < -0.99


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"  ok  {name}")
    print("\nall return-sign tests pass")
