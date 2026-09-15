"""Tests for the learning-progress return — the H(p_env) cancellation.

The surrogate action return is the stated objective plus the entropy of the
environment:

    E[-ln p_theta] = H(p_env) + KL(p_env || p_theta)

and that entropy term is a REINFORCE reward for making the world simpler. It is
common to any cross-entropy against p_env, so it cancels exactly in a
difference of two of them. Replacing the return's per-observation summand with
``M_s - M_s^-`` therefore removes it — for ANY second reference, however bad.

What has to hold is mostly about what does *not* move:

  1. The perception arm is untouched. It still minimises the agent's own
     observation surprise; only the return's summand changes.
  2. The return is exactly the surrogate minus the reference's own return.
  3. A reference equal to the current model gives an identically zero return —
     the degenerate plug-in case of surprise_minimisation_derivation.md 5.5.4,
     and the reason the reference is double-buffered so it can never be the
     current weights.
  4. Default off leaves everything byte-identical.
  5. Enabling it without a reference is a hard error, not a silent fallback to
     the surrogate.

Run:  .venv/bin/python tests/test_learning_progress.py
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

os.environ.setdefault("HF_HOME", "/lus/lfs1aip2/projects/a5l/egunn/hf_cache")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from marlllm.agent import LearningProgressReference, ValueHead   # noqa: E402
from marlllm.config import TrainingConfig                        # noqa: E402
from marlllm.loss import CCSMLoss                                # noqa: E402
from marlllm.types import TokenType                              # noqa: E402

OBS, ACT = int(TokenType.OBS), int(TokenType.ACT)
B, T, H, V = 2, 12, 16, 32


def _fixture(seed: int = 0):
    """Deterministic (hidden, reference hidden, ids, masks, modules)."""
    torch.manual_seed(seed)
    ids = torch.randint(0, V, (B, T))
    tm = torch.full((B, T), OBS)
    tm[:, 2::3] = ACT
    tm[:, 0] = OBS
    am = torch.zeros((B, T), dtype=torch.long)
    trunk, lm, vh = nn.Linear(H, H), nn.Linear(H, V), ValueHead(H)
    x = torch.randn(B, T, H, generator=torch.Generator().manual_seed(1))
    # A deliberately different second policy: a fixed random perturbation of
    # the same trunk, standing in for "the adapter a few iterations ago".
    ref_hidden = trunk(
        x + 0.5 * torch.randn(B, T, H, generator=torch.Generator().manual_seed(2))
    ).detach()
    return ids, tm, am, trunk, lm, vh, x, ref_hidden


def _run(*, learning_progress, ref_hidden=None, seed=0, **cfg_kw):
    ids, tm, am, trunk, lm, vh, x, default_ref = _fixture(seed)
    for m in (trunk, lm, vh):
        m.zero_grad(set_to_none=True)
    hidden = trunk(x)
    if learning_progress and ref_hidden is None:
        ref_hidden = default_ref
    config = TrainingConfig(
        learning_progress_return=learning_progress,
        normalise_returns=cfg_kw.pop("normalise_returns", False),
        **cfg_kw,
    )
    loss, met = CCSMLoss().compute_loss(
        last_hidden=hidden, lm_head=lm, values=vh(hidden.detach()),
        input_ids=ids, token_type_mask=tm, agent_id_mask=am,
        target_agent_idx=0, config=config,
        last_hidden_lp=ref_hidden if learning_progress else None,
    )
    loss.backward()
    grad = torch.cat([p.grad.flatten() for p in lm.parameters()])
    return met, grad


# ------------------------------------------------------------------ #
# The return                                                           #
# ------------------------------------------------------------------ #

def test_the_perception_arm_is_untouched():
    plain, _ = _run(learning_progress=False)
    lp, _ = _run(learning_progress=True)
    assert abs(plain["perc_loss"] - lp["perc_loss"]) < 1e-9, (
        f"perception moved: {plain['perc_loss']} vs {lp['perc_loss']}")
    assert abs(plain["mean_surprise"] - lp["mean_surprise"]) < 1e-9


def test_the_surrogate_is_reported_alongside_and_matches_the_plain_run():
    plain, _ = _run(learning_progress=False)
    lp, _ = _run(learning_progress=True)
    assert abs(lp["lp/return_surrogate_mean"] - plain["return_raw_mean"]) < 1e-4, (
        f"{lp['lp/return_surrogate_mean']} vs {plain['return_raw_mean']}")


def test_the_return_is_the_surrogate_minus_the_reference_return():
    """G^lp = G^own - G^ref, position by position, so it holds in the mean."""
    ids, tm, am, trunk, lm, vh, x, ref_hidden = _fixture()
    lp, _ = _run(learning_progress=True)
    # The reference's own return: run the loss with the reference hidden states
    # in the training slot and read its surrogate return.
    for m in (trunk, lm, vh):
        m.zero_grad(set_to_none=True)
    _, ref_run = CCSMLoss().compute_loss(
        last_hidden=ref_hidden.clone().requires_grad_(True), lm_head=lm,
        values=vh(ref_hidden), input_ids=ids, token_type_mask=tm,
        agent_id_mask=am, target_agent_idx=0,
        config=TrainingConfig(normalise_returns=False),
    )
    expected = lp["lp/return_surrogate_mean"] - ref_run["return_raw_mean"]
    assert abs(lp["return_raw_mean"] - expected) < 1e-3, (
        f"{lp['return_raw_mean']} != {expected}")


def test_a_reference_equal_to_the_current_model_gives_a_zero_return():
    """The plug-in case of 5.5.4. This is why the buffer is double-slotted."""
    ids, tm, am, trunk, lm, vh, x, _ = _fixture()
    hidden = trunk(x)
    _, met = CCSMLoss().compute_loss(
        last_hidden=hidden, lm_head=lm, values=vh(hidden.detach()),
        input_ids=ids, token_type_mask=tm, agent_id_mask=am,
        target_agent_idx=0,
        config=TrainingConfig(learning_progress_return=True,
                              normalise_returns=False),
        last_hidden_lp=hidden.detach(),
    )
    assert abs(met["return_raw_mean"]) < 1e-6, met["return_raw_mean"]
    assert abs(met["return_raw_abs_max"]) < 1e-5, met["return_raw_abs_max"]
    assert abs(met["lp/progress_per_token"]) < 1e-6


def test_progress_is_the_reference_surprise_minus_our_own():
    """The reported progress is an identity on the two surprise means, which
    is the cheapest available check that the right tensor is being
    subtracted."""
    lp, _ = _run(learning_progress=True)
    assert abs(lp["lp/progress_per_token"]
               - (lp["lp/ref_surprise_mean"] - lp["mean_surprise"])) < 1e-5


def test_progress_is_positive_when_the_reference_is_worse():
    """A model fitted to this sequence beats a uniform reference, so the
    per-token progress is positive and the return — which is paid for
    improvement — goes negative."""
    ids, tm, am, trunk, lm, vh, x, _ = _fixture()
    hidden = trunk(x).detach()
    # Fit the head to the actual tokens so "better than uniform" is true by
    # construction rather than by luck of the initialisation: a random head is
    # routinely WORSE than uniform, which is what makes the untrained version
    # of this test meaningless.
    opt = torch.optim.Adam(lm.parameters(), lr=0.1)
    for _ in range(200):
        opt.zero_grad()
        nll = torch.nn.functional.cross_entropy(
            lm(hidden[:, :-1, :]).reshape(-1, V), ids[:, 1:].reshape(-1))
        nll.backward()
        opt.step()
    # Zero hidden states through a bias-free head give a uniform distribution:
    # surprise ln V at every token.
    lm_uniform = nn.Linear(H, V, bias=False)
    with torch.no_grad():
        lm_uniform.weight.zero_()
    _, met = CCSMLoss().compute_loss(
        last_hidden=hidden.clone().requires_grad_(True), lm_head=lm,
        values=vh(hidden), input_ids=ids, token_type_mask=tm,
        agent_id_mask=am, target_agent_idx=0,
        config=TrainingConfig(learning_progress_return=True,
                              normalise_returns=False),
        last_hidden_lp=torch.zeros_like(hidden), lm_head_lp=lm_uniform,
    )
    assert met["lp/ref_surprise_mean"] > met["mean_surprise"]
    assert met["lp/progress_per_token"] > 0.0
    # Position 0 carries no shifted prediction, so its progress is exactly 0
    # under both policies and cannot be counted as positive.
    assert met["lp/progress_positive_frac"] >= 0.8
    # Paying for improvement means the return is NEGATIVE where the agent is
    # ahead of its reference, so the advantage -(G - V) points up.
    assert met["return_raw_mean"] < 0.0


def test_the_correction_reaches_the_policy_gradient():
    _, plain = _run(learning_progress=False)
    _, lp = _run(learning_progress=True)
    assert not torch.allclose(plain, lp), "the corrected return changed nothing"


def test_default_is_off_and_byte_identical():
    assert TrainingConfig().learning_progress_return is False
    assert TrainingConfig().learning_progress_reference == "lagged"
    a, ga = _run(learning_progress=False)
    b, gb = _run(learning_progress=False)
    assert torch.equal(ga, gb)
    assert "lp/in_use" not in a, "a legacy run must not gain a stray lp/* key"


def test_it_composes_with_normalisation_and_with_the_sign_flip():
    met, _ = _run(learning_progress=True, normalise_returns=True)
    assert met["lp/in_use"] == 1.0
    flipped, _ = _run(learning_progress=True, return_sign=-1.0)
    straight, _ = _run(learning_progress=True)
    assert abs(flipped["return_raw_mean"] + straight["return_raw_mean"]) < 1e-5


def test_a_missing_reference_is_a_hard_error():
    ids, tm, am, trunk, lm, vh, x, _ = _fixture()
    hidden = trunk(x)
    try:
        CCSMLoss().compute_loss(
            last_hidden=hidden, lm_head=lm, values=vh(hidden.detach()),
            input_ids=ids, token_type_mask=tm, agent_id_mask=am,
            target_agent_idx=0,
            config=TrainingConfig(learning_progress_return=True),
            last_hidden_lp=None,
        )
    except RuntimeError as e:
        assert "silently fall back" in str(e)
    else:
        raise AssertionError(
            "a run that cannot compute its declared objective must stop")


# ------------------------------------------------------------------ #
# The two-slot buffer                                                  #
# ------------------------------------------------------------------ #

def test_the_reference_adapter_name_cannot_collide_with_a_live_one():
    """PEFT selects an adapter's weights with a plain substring test:

        {k: v for k, v in sd.items() if "lora_" in k and adapter_name in k}

    so a reference adapter named ``<live>__lp_ref`` is swept into every
    ``save_pretrained(selected_adapters=[live])``. That call runs on the vLLM
    sync path after each optimiser step, which would have handed the rollout
    engine a file containing two policies. The names must stay disjoint.
    """
    names = ["agent_0", "Priya Shah", "Henrietta 'Henri' Voss", "default",
             "Mhairi Buchanan"]
    refs = [LearningProgressReference._lp_reference_adapter_name(n)
            for n in names]
    for live, ref in zip(names, refs):
        assert live not in ref, f"{live!r} is a substring of {ref!r}"
        assert ref not in live
    assert len(set(refs)) == len(refs), "two live adapters share a reference name"
    # Deterministic, so a checkpoint written by one process is readable by the
    # next.
    assert refs == [LearningProgressReference._lp_reference_adapter_name(n)
                    for n in names]


# ------------------------------------------------------------------ #
# The chunked path                                                     #
# ------------------------------------------------------------------ #

class _FakeChunkedAgent:
    """Minimal stand-in for compute_loss_chunked's ``agent`` argument: it
    slices pre-computed hidden states instead of running a backbone."""

    def __init__(self, hidden, hidden_lp, lm_module, value_head):
        self._hidden = hidden
        self._hidden_lp = hidden_lp
        self._lm = lm_module
        self._value_head = value_head
        self._keep_ref_model = False
        self._backbone = nn.Sequential()
        self._backbone.register_parameter(
            "dummy", nn.Parameter(torch.zeros(1, dtype=hidden.dtype)))
        self.device = hidden.device

    def _slices(self, tensor, chunk_size):
        T = tensor.shape[1]
        for s in range(0, T, chunk_size):
            e = min(s + chunk_size, T)
            yield s, e, tensor[:, s:e, :]

    def evaluate_hidden_chunked(self, ids, attn, chunk_size):
        for s, e, h in self._slices(self._hidden, chunk_size):
            yield s, e, h, self._value_head(h.detach())

    def evaluate_hidden_ref_chunked(self, ids, attn, chunk_size):
        for s in range(0, self._hidden.shape[1], chunk_size):
            yield s, min(s + chunk_size, self._hidden.shape[1]), None

    def evaluate_hidden_lp_chunked(self, ids, attn, chunk_size):
        yield from self._slices(self._hidden_lp, chunk_size)

    def lm_head(self, hidden):
        return self._lm(hidden)

    def train_mode(self):
        pass


def test_the_chunked_path_agrees_with_the_single_shot_path():
    ids, tm, am, trunk, lm, vh, x, ref_hidden = _fixture()
    cfg = dict(learning_progress_return=True, normalise_returns=False,
               gamma=0.9, beta=0.05, kl_coef=0.0)

    hidden_a = trunk(x)
    _, met_a = CCSMLoss().compute_loss(
        last_hidden=hidden_a, lm_head=lm, values=vh(hidden_a.detach()),
        input_ids=ids, token_type_mask=tm, agent_id_mask=am,
        target_agent_idx=0, config=TrainingConfig(**cfg),
        last_hidden_lp=ref_hidden,
    )

    hidden_b = trunk(x)
    agent = _FakeChunkedAgent(hidden_b, ref_hidden, lm, vh)
    _, met_b = CCSMLoss().compute_loss_chunked(
        agent=agent, input_ids=ids, attention_mask=torch.ones_like(ids),
        token_type_mask=tm, agent_id_mask=am, target_agent_idx=0,
        config=TrainingConfig(seq_chunk_size=T, **cfg), backward_scale=1.0,
    )
    for key in ("lp/progress_per_token", "lp/return_mean",
                "lp/return_surrogate_mean", "mean_return", "perc_loss"):
        assert abs(met_a[key] - met_b[key]) < 1e-3, (
            f"{key}: {met_a[key]} vs {met_b[key]}")


def test_the_chunked_path_also_refuses_to_fall_back():
    ids, tm, am, trunk, lm, vh, x, _ = _fixture()
    hidden = trunk(x)
    agent = _FakeChunkedAgent(hidden, None, lm, vh)
    # 'base' with no frozen reference in memory yields (s, e, None) chunks.
    try:
        CCSMLoss().compute_loss_chunked(
            agent=agent, input_ids=ids, attention_mask=torch.ones_like(ids),
            token_type_mask=tm, agent_id_mask=am, target_agent_idx=0,
            config=TrainingConfig(learning_progress_return=True,
                                  learning_progress_reference="base",
                                  seq_chunk_size=T),
            backward_scale=1.0,
        )
    except RuntimeError as e:
        assert "silently fall back" in str(e)
    else:
        raise AssertionError("the chunked path fell back to the surrogate")


class _StubAgent(LearningProgressReference):
    """Exercises the rotation bookkeeping without a backbone.

    ``weights`` is a single integer standing in for the whole adapter: after
    the optimiser step of iteration ``i`` it is ``i``, so the weights in force
    during iteration ``i``'s forward pass are ``i - 1``. Both slots start
    seeded at 0, exactly as ``enable_learning_progress_reference`` leaves them.
    """

    def __init__(self):
        self.weights = 0
        self.reference = 0
        self._lp_adapter_name = "stub__lp_ref"
        self._lp_pending = {"w": 0}
        self._lp_reference_iteration = 0
        self._lp_pending_iteration = 0

    def _lp_live_adapter_name(self):
        return "stub"

    def _lp_snapshot_live(self):
        return {"w": self.weights}

    def _lp_write_reference(self, state):
        self.reference = state["w"]


def _drive(agent, lag: int, iterations: int):
    """Run the trainer's rotation schedule against the stub.

    Mirrors PopulationTrainer._rotate_learning_progress: rotate at the START
    of iterations L+1, 2L+1, ..., stamping the snapshot with ``iteration - 1``
    because that is the optimiser step the current weights came from.
    """
    trace = []
    for iteration in range(1, iterations + 1):
        if iteration > 1 and (iteration - 1) % lag == 0:
            agent.rotate_learning_progress_reference(iteration - 1)
        current = iteration - 1            # weights in force this iteration
        trace.append((iteration, current, agent.reference,
                      agent.learning_progress_reference_age(iteration)))
        agent.weights = iteration          # optimiser step
    return trace


def test_the_reference_is_never_the_current_weights():
    """Rotation installs the SECOND-most-recent snapshot, so even on the
    iteration immediately after a rotation the reference is a different model
    and the return cannot collapse to the identically-zero plug-in case."""
    trace = _drive(_StubAgent(), lag=25, iterations=200)
    # Iteration 1 is the one honest exception: no step has been taken, so the
    # current weights ARE the initial weights and there is no progress to
    # measure. The return is zero for that single iteration and non-zero from
    # then on.
    assert all(ref != current for it, current, ref, _ in trace if it > 1), (
        "the reference coincided with the current weights")
    assert trace[0][1] == trace[0][2] == 0
    by_iter = {it: ref for it, _, ref, _ in trace}
    # At iteration 26 the reference is the snapshot from iteration 0, not 25.
    assert by_iter[26] == 0
    # After the second rotation it is the snapshot from 25, not 50.
    assert by_iter[51] == 25


def test_the_effective_lag_runs_from_lag_plus_one_to_twice_lag():
    trace = _drive(_StubAgent(), lag=25, iterations=200)
    # Skip the warm start, where both slots still hold the initial weights.
    ages = [age for it, _, _, age in trace if it > 50]
    assert min(ages) == 26 and max(ages) == 50, (min(ages), max(ages))


def test_the_age_metric_tracks_the_reference():
    agent = _StubAgent()
    agent.weights = 40
    agent.rotate_learning_progress_reference(25)
    assert agent.learning_progress_reference_age(40) == 40   # reference is init
    agent.rotate_learning_progress_reference(50)
    assert agent.learning_progress_reference_age(51) == 26   # reference is 25


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"  ok  {name}")
    print("\nall learning-progress tests pass")
