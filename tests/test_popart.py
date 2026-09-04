"""Tests for PopArt on the value head.

PopArt exists for the problem this project has: the return runs to ~250 at
gamma=0.99 and ~2300 at gamma=1, a plain MSE head over that range is badly
conditioned, and standardising the TARGET instead throws away the level the
baseline is meant to absorb. PopArt normalises the target's conditioning while
POP-rescaling the output layer so the denormalised value is preserved.

What has to hold:
  1. POP is exact — moving the moments must not move the denormalised output.
  2. The head becomes trainable on raw returns.
  3. The advantage stays in the return's units.
  4. Off by default, and then the head is a plain linear layer as before.

Run:  .venv/bin/python tests/test_popart.py
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


def test_pop_preserves_outputs_precisely():
    """The invariant is measured against SIGMA, the scale the value lives on.

    Not against max|output|, which at initialisation is ~0.4 while sigma is
    ~500, and would make float32 rounding look like a real drift. In float64
    the formula is exact to 1e-16 relative to sigma; float32 gives ~5e-7,
    which is machine epsilon.
    """
    torch.manual_seed(0)
    h = ValueHead(8, popart=True, popart_beta=0.3)
    x = torch.randn(4, 3, 8)
    for mean, sq in ((2300.0, 2300.0**2 + 500.0**2),
                     (900.0, 900.0**2 + 300.0**2),
                     (12.0, 12.0**2 + 4.0**2)):
        before = h.denormalise(h(x)).detach().clone()
        h.popart_update(mean, sq)
        after = h.denormalise(h(x)).detach()
        drift = float((after - before).abs().max()) / float(h.popart_sigma)
        assert drift < 1e-5, f"POP moved the output by {drift:.2e} of sigma"


def test_pop_is_exact_in_double():
    """Separates the formula from the arithmetic."""
    torch.manual_seed(0)
    h = ValueHead(8, popart=True, popart_beta=0.3).to(torch.float64)
    x = torch.randn(4, 3, 8, dtype=torch.float64)
    before = h.denormalise(h(x)).detach().clone()
    h.popart_update(2300.0, 2300.0**2 + 500.0**2)
    after = h.denormalise(h(x)).detach()
    drift = float((after - before).abs().max()) / float(h.popart_sigma)
    assert drift < 1e-12, f"the POP formula itself is wrong: {drift:.2e}"


def test_moments_track_the_target():
    h = ValueHead(4, popart=True, popart_beta=0.5)
    h.popart_update(2000.0, 2000.0**2 + 400.0**2)
    assert abs(float(h.popart_mu) - 2000.0) < 1.0, float(h.popart_mu)
    assert abs(float(h.popart_sigma) - 400.0) < 5.0, float(h.popart_sigma)


def _train(popart, steps=4):
    torch.manual_seed(0)
    B, T, H, V = 2, 24, 16, 32
    ids = torch.randint(0, V, (B, T))
    tm = torch.full((B, T), OBS); tm[:, 2::3] = ACT; tm[:, 0] = OBS
    am = torch.zeros((B, T), dtype=torch.long)
    trunk, lm = nn.Linear(H, H), nn.Linear(H, V)
    vh = ValueHead(H, popart=popart, popart_beta=0.3)
    opt = torch.optim.Adam(vh.parameters(), lr=1e-2)
    cfg = TrainingConfig(gamma=1.0, normalise_returns=False)
    losses = []
    for i in range(steps):
        hid = trunk(torch.randn(B, T, H, generator=torch.Generator().manual_seed(i)))
        loss, m = CCSMLoss().compute_loss(
            last_hidden=hid, lm_head=lm, values=vh(hid.detach()),
            input_ids=ids, token_type_mask=tm, agent_id_mask=am,
            target_agent_idx=0, config=cfg,
            value_head=vh if popart else None)
        opt.zero_grad(); loss.backward(); opt.step()
        if popart:
            vh.popart_update(m["popart/batch_mean"], m["popart/batch_sq_mean"])
        losses.append(m["value_loss"])
    return losses


def test_it_makes_raw_returns_trainable():
    off, on = _train(False), _train(True)
    assert on[-1] < off[-1] / 10, (
        f"popart value_loss {on[-1]:.1f} vs {off[-1]:.1f} without — "
        "expected an order of magnitude")
    assert on[-1] < on[0] / 10, f"popart did not converge: {on}"


def test_off_by_default_is_a_plain_head():
    h = ValueHead(4)
    assert h.popart is False
    x = torch.randn(2, 3, 4)
    assert torch.equal(h.denormalise(h(x)), h(x))
    assert torch.equal(h.normalise_target(x), x)
    assert h.popart_update(1.0, 2.0) == {}
    assert TrainingConfig().popart is False


def test_pre_popart_checkpoints_still_load():
    """Regression: adding the four running-moment buffers made every earlier
    checkpoint unloadable, which breaks resume as well as analysis over old
    checkpoints. It killed job 6312471 and would have killed three queued
    restricted-pairing resumes."""
    old_sd = {"linear.weight": torch.randn(1, 8), "linear.bias": torch.zeros(1)}
    for popart in (False, True):
        h = ValueHead(8, popart=popart)
        h.load_state_dict(dict(old_sd))              # must not raise
        assert torch.equal(h.linear.weight, old_sd["linear.weight"])
        assert float(h.popart_ready) == 0.0, "should seed on the first update"


def test_a_genuinely_missing_key_still_raises():
    """The tolerance is for the PopArt buffers only."""
    h = ValueHead(8)
    try:
        h.load_state_dict({"linear.weight": torch.randn(1, 8)})   # no bias
    except RuntimeError as e:
        assert "linear.bias" in str(e)
    else:
        raise AssertionError("a real missing key was silently tolerated")


def test_popart_state_round_trips():
    a = ValueHead(8, popart=True)
    a.popart_update(2300.0, 2300.0 ** 2 + 500.0 ** 2)
    b = ValueHead(8, popart=True)
    b.load_state_dict(a.state_dict())
    assert torch.equal(a.popart_sigma, b.popart_sigma)
    assert torch.equal(a.popart_mu, b.popart_mu)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"  ok  {name}")
    print("\nall PopArt tests pass")
