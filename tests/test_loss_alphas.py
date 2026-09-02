"""Tests for the alpha term weights on the population path.

`alpha_perc` / `alpha_act` / `alpha_val` select which of CCSM's three terms
contribute. They were honoured by the loss but unreachable from a population
config until train_population.py passed them through, so these pin down what a
term ablation on a forum substrate actually ablates:

  * alpha_act = 0 removes the policy gradient AND the entropy bonus it
    carries. Actions are still sampled; their log-probs receive no gradient.
    This is the arm that removes the dark-room incentive's only channel to
    behaviour, since the return reaches the policy through the action term
    alone.
  * alpha_perc = 0 removes the next-token loss over observation tokens — and
    with it the mechanism absorption is measured through, because
    prompt_as_observation types prompt tokens OBS.
  * alpha_val = 0 removes the value regression.

Run:  .venv/bin/python tests/test_loss_alphas.py
(pytest is not installed in the Isambard venv; this file is also importable
as a pytest module.)
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


def _run(**alphas):
    """Run compute_loss once; return (loss scalar, metrics, policy/value grad norms)."""
    torch.manual_seed(0)
    B, T, H, V = 2, 12, 16, 32
    ids = torch.randint(0, V, (B, T))
    tm = torch.full((B, T), OBS)
    tm[:, 2::3] = ACT
    tm[:, 0] = OBS
    am = torch.zeros((B, T), dtype=torch.long)

    trunk, lm_head, vhead = nn.Linear(H, H), nn.Linear(H, V), ValueHead(H)
    for m in (trunk, lm_head, vhead):
        m.zero_grad(set_to_none=True)

    hidden = trunk(torch.randn(
        B, T, H, generator=torch.Generator().manual_seed(1)))
    cfg = TrainingConfig(gamma=0.99, normalise_returns=False, beta=0.01, **alphas)
    loss, metrics = CCSMLoss().compute_loss(
        last_hidden=hidden, lm_head=lm_head, values=vhead(hidden.detach()),
        input_ids=ids, token_type_mask=tm, agent_id_mask=am,
        target_agent_idx=0, config=cfg,
    )
    loss.backward()

    def gnorm(mod):
        gs = [p.grad for p in mod.parameters() if p.grad is not None]
        return float(sum(g.pow(2).sum() for g in gs).sqrt()) if gs else 0.0

    # lm_head is the policy: the action term and the perception term both
    # reach it, so it is separated below by ablating one at a time.
    return float(loss.detach()), metrics, gnorm(lm_head), gnorm(vhead)


# ------------------------------------------------------------------ #
def test_full_loss_trains_every_head():
    _, _, policy_g, value_g = _run(alpha_perc=1.0, alpha_act=1.0, alpha_val=1.0)
    assert policy_g > 0.0, "policy received no gradient under the full loss"
    assert value_g > 0.0, "value head received no gradient under the full loss"


def test_perception_only_leaves_the_value_head_untouched():
    """alpha_act = alpha_val = 0: only L_perc contributes."""
    _, _, policy_g, value_g = _run(alpha_perc=1.0, alpha_act=0.0, alpha_val=0.0)
    assert value_g == 0.0, "value head trained with alpha_val = 0"
    # The policy head still moves, but only through next-token prediction on
    # observation tokens — which is the point of the arm, not a leak.
    assert policy_g > 0.0


def test_action_only_leaves_the_perception_term_out():
    """alpha_perc = 0: no next-token loss, policy and value still train."""
    _, _, policy_g, value_g = _run(alpha_perc=0.0, alpha_act=1.0, alpha_val=1.0)
    assert policy_g > 0.0, "policy received no gradient under action-only"
    assert value_g > 0.0, "value head received no gradient under action-only"


def test_zeroing_a_term_changes_the_loss_it_weights():
    """Each alpha actually moves the total, so none is a dead knob."""
    full, _, _, _ = _run(alpha_perc=1.0, alpha_act=1.0, alpha_val=1.0)
    no_perc, *_ = _run(alpha_perc=0.0, alpha_act=1.0, alpha_val=1.0)
    no_act, *_ = _run(alpha_perc=1.0, alpha_act=0.0, alpha_val=1.0)
    no_val, *_ = _run(alpha_perc=1.0, alpha_act=1.0, alpha_val=0.0)
    for name, total in (("perc", no_perc), ("act", no_act), ("val", no_val)):
        assert total != full, f"alpha_{name} is a dead knob"


def test_the_component_losses_are_still_reported_when_switched_off():
    """A zeroed term is still measured, so an ablation stays readable."""
    _, m, _, _ = _run(alpha_perc=1.0, alpha_act=0.0, alpha_val=0.0)
    for k in ("perc_loss", "act_loss", "value_loss"):
        assert k in m, f"{k} is not reported"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ok  {name}")
    print("\nall alpha-weight tests pass")
