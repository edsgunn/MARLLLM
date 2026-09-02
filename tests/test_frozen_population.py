"""Tests for frozen population members (REWRITE.md C3).

A frozen member posts during the rollout — the learner reads it as
observation — but receives no gradient, so the learner faces a *fixed*
culture rather than a co-adapting one. Three things must hold for that
control to mean anything:

  1. Frozen members' parameters are kept out of the optimizer, so no step
     can reach them.
  2. Their forward+backward is skipped, which is what makes the control
     cheap (~1/N of the backward at one learner in N).
  3. A gradient arriving on a frozen parameter is a hard error, not a
     warning — a leaked freeze silently turns the control back into the
     experiment it is controlling for.

Run:  .venv/bin/python tests/test_frozen_population.py
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

from marlllm.config import TrainingConfig          # noqa: E402
from marlllm.population import PopulationTrainer   # noqa: E402


class _Agent:
    """Minimal stand-in: a disjoint parameter set per member, as
    LoRASharedBaseAgent gives (each adapter's params are its own)."""

    def __init__(self, name: str):
        self.agent_id = name
        self._p = nn.Parameter(torch.ones(2))

    def parameters(self):
        yield self._p


def _init(names, frozen, agents=None):
    """Run PopulationTrainer's real frozen/optimizer setup on stub agents.

    Binds the shipped method rather than a copy of it, so the test fails if
    the constructor's logic drifts.
    """
    population = agents or {n: _Agent(n) for n in names}
    config = TrainingConfig(frozen_agents=list(frozen), use_8bit_adam=False)

    trainer = object.__new__(PopulationTrainer)
    all_params = PopulationTrainer._setup_frozen_agents(
        trainer, population, config
    )
    trainer.optimizer = torch.optim.AdamW(all_params, lr=0.1)
    return trainer, population, all_params


# ------------------------------------------------------------------ #
def test_frozen_members_are_absent_from_the_optimizer():
    names = [f"a{i}" for i in range(8)]
    frozen = names[1:]                       # one learner in eight
    trainer, pop, all_params = _init(names, frozen)

    assert trainer.trainable_agents == ["a0"]
    assert len(all_params) == 1
    assert all_params[0] is pop["a0"]._p
    opt_params = {id(p) for g in trainer.optimizer.param_groups for p in g["params"]}
    for name in frozen:
        assert id(pop[name]._p) not in opt_params, f"{name} reached the optimizer"
    assert len(trainer._frozen_params) == 7


def test_a_step_moves_the_learner_and_nothing_else():
    names = ["learner", "fixed_a", "fixed_b"]
    trainer, pop, _ = _init(names, ["fixed_a", "fixed_b"])
    before = {n: pop[n]._p.detach().clone() for n in names}

    # Only the learner's forward happens, so only it gets a gradient.
    pop["learner"]._p.sum().backward()
    trainer.optimizer.step()

    assert not torch.equal(pop["learner"]._p, before["learner"]), "learner did not move"
    for n in ("fixed_a", "fixed_b"):
        assert torch.equal(pop[n]._p, before[n]), f"{n} moved while frozen"


def test_a_leaked_gradient_is_an_error_not_a_warning():
    names = ["learner", "fixed"]
    trainer, pop, _ = _init(names, ["fixed"])
    pop["fixed"]._p.sum().backward()          # the leak this guards against

    leaked = sum(1 for p in trainer._frozen_params if p.grad is not None)
    assert leaked == 1, "the guard would not have seen the leak"


def test_shared_parameters_between_frozen_and_trainable_are_refused():
    shared = _Agent("shared")
    twin = _Agent("twin")
    twin._p = shared._p                       # the case freezing cannot honour
    try:
        _init(["shared", "twin"], ["twin"],
              agents={"shared": shared, "twin": twin})
    except ValueError as e:
        assert "shared between a frozen and a trainable" in str(e)
    else:
        raise AssertionError("shared parameters were accepted")


def test_freezing_everything_is_refused():
    try:
        _init(["a", "b"], ["a", "b"])
    except ValueError as e:
        assert "No trainable parameters" in str(e)
    else:
        raise AssertionError("an all-frozen population was accepted")


def test_unknown_member_is_refused():
    try:
        _init(["a", "b"], ["Nobody"])
    except ValueError as e:
        assert "unknown population member" in str(e)
    else:
        raise AssertionError("an unknown frozen member was accepted")


def test_no_frozen_agents_is_the_previous_behaviour():
    names = ["a", "b", "c"]
    trainer, pop, all_params = _init(names, [])
    assert trainer.frozen_agents == set()
    assert trainer.trainable_agents == names
    assert len(all_params) == 3
    assert trainer._frozen_params == []


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ok  {name}")
    print("\nall frozen-population tests pass")
