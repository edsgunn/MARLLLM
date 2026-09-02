"""Tests for recovering ln p_env from the generating agent's log-probs.

Covers the three things that must hold for the environment-likelihood
experiment to mean anything (environment_likelihood_derivation.md):

  1. No-tag mode delivers the generation byte for byte, so there is no latent.
  2. Provenance splices the generator's EXACT token ids into the observer's
     context, so a log-prob attaches per token with no re-encoding.
  3. The corrected return is G_t + sum gamma^(s-t) E_s, and the legacy path is
     untouched when the term is absent.

Run:  .venv/bin/python tests/test_env_likelihood.py
(pytest is not installed in the Isambard venv; this file is also importable
as a pytest module.)
"""
from __future__ import annotations

import os
import sys

import torch

os.environ.setdefault("HF_HOME", "/lus/lfs1aip2/projects/a5l/egunn/hf_cache")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from marlllm.config import TrainingConfig                       # noqa: E402
from marlllm.loss import (                                      # noqa: E402
    _compute_returns, _env_returns, _env_likelihood_terms, _spearman,
)
from marlllm.population import PopulationTrainer                # noqa: E402
from marlllm.types import EpisodeStep, RolloutBatch, TokenType, Trajectory  # noqa: E402

OBS, ACT = int(TokenType.OBS), int(TokenType.ACT)


def _tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct", local_files_only=True)


def _env(tok, **kw):
    from envs.forum.env import ForumEnv
    names = ["Priya", "Tom", "Hana"]
    base = dict(agent_names=names, agent_personas={n: f"You are {n}." for n in names},
                tokenizer=tok, total_token_budget=384, post_token_budget=384,
                max_posts=6, seed=0)
    base.update(kw)
    return ForumEnv(**base), names


# ------------------------------------------------------------------ #
def test_no_tag_mode_delivers_the_generation_verbatim():
    tok = _tokenizer()
    env, names = _env(tok, post_open_tag="", post_close_tag="")
    env.reset()
    raw = "Types matter.  A list is mutable; a tuple is not.\n\nSo pop() is absent."
    env.step(tok.encode(raw, add_special_tokens=False))
    assert env._thread[-1]["text"] == raw, "no-tag delivery must not alter the text"


def test_no_tag_mode_refuses_the_python_tool():
    tok = _tokenizer()
    try:
        _env(tok, post_open_tag="", post_close_tag="", python_tool_enabled=True)
    except ValueError:
        return
    raise AssertionError("a code block has nowhere to hide in no-tag mode")


def test_framing_segments_reproduce_the_string_exactly():
    """The segmented and string forms of the framing must not drift."""
    tok = _tokenizer()
    env, _ = _env(tok)
    cases = [
        [("Priya", "hello")], [("Priya", "")],
        [("Priya", "a"), ("Tom", "b")],
        [("Priya", "a"), ("Tom", ""), ("Hana", "c")],
        [("Priya", "multi\nline\n\npost", [1, 2], 0)],
    ]
    for c in cases:
        joined = "".join(seg[2] for seg in env._frame_post_segments(c))
        assert joined == env._frame_posts(c)


def test_provenance_splices_the_generators_exact_tokens():
    tok = _tokenizer()
    env, names = _env(tok, post_open_tag="", post_close_tag="", provenance=True)
    env.reset()
    speaker = env.agent_selection
    raw = "A list is mutable; a tuple is not, so pop() is absent."
    ids = [int(t) for t in tok.encode(raw, add_special_tokens=False)]
    lps = [-(i + 1) / 100.0 for i in range(len(ids))]      # distinctive per token
    idx = env._post_count
    env.step(ids)

    observer = next(a for a in names if a != speaker)
    obs_ids = env.observe(observer)

    class _Stub:
        _log_env_likelihood = True
        _env_lp_misalignments = 0
        _env_log_probs_for_obs = PopulationTrainer._env_log_probs_for_obs

    env_lps, known = _Stub()._env_log_probs_for_obs(
        env, observer, len(obs_ids), {idx: lps})

    assert len(env_lps) == len(known) == len(obs_ids)
    measured = [i for i, k in enumerate(known) if k and env_lps[i] != 0.0]
    lo, hi = min(measured), max(measured)
    assert obs_ids[lo:hi + 1] == ids, "spliced ids are not the generator's"
    assert [round(x, 6) for x in env_lps[lo:hi + 1]] == [round(x, 6) for x in lps]
    assert tok.decode(obs_ids[lo:hi + 1], skip_special_tokens=True) == raw
    # Scaffolding is deterministic: ln p_env = 0, and known to be so.
    outside = [i for i in range(len(obs_ids)) if not lo <= i <= hi]
    assert all(env_lps[i] == 0.0 and known[i] for i in outside)
    assert env._provenance_misses == 0


def test_legacy_env_leaves_the_term_dropped():
    tok = _tokenizer()
    env, names = _env(tok, post_token_budget=128)          # tags on, provenance off
    env.reset()
    speaker = env.agent_selection
    env.step(tok.encode("<post>hi</post>", add_special_tokens=False))
    observer = next(a for a in names if a != speaker)
    obs_ids = env.observe(observer)

    class _Stub:
        _log_env_likelihood = True
        _env_lp_misalignments = 0
        _env_log_probs_for_obs = PopulationTrainer._env_log_probs_for_obs

    lps, known = _Stub()._env_log_probs_for_obs(env, observer, len(obs_ids), {})
    assert lps is None and known is None


# ------------------------------------------------------------------ #
def test_corrected_return_is_the_surrogate_plus_the_env_term():
    tm = torch.tensor([[ACT, OBS, ACT, OBS, OBS]])
    sur = torch.tensor([[0.0, 2.0, 0.0, 3.0, 1.0]])
    env = torch.tensor([[0.0, -0.5, 0.0, -1.0, -0.25]])
    G = _compute_returns(sur, tm, 1.0)
    ER = _env_returns(env, tm, 1.0, None, sur.dtype)
    assert abs(G[0, 0].item() - 6.0) < 1e-6 and abs(G[0, 2].item() - 4.0) < 1e-6
    assert abs(ER[0, 0].item() + 1.75) < 1e-6 and abs(ER[0, 2].item() + 1.25) < 1e-6

    act, obs = tm == ACT, tm == OBS
    off = TrainingConfig(gamma=1.0, use_env_likelihood_return=False)
    on = TrainingConfig(gamma=1.0, use_env_likelihood_return=True)
    r_off, _, m_off = _env_likelihood_terms(G, tm, act, obs, off, None, env, None)
    r_on, _, _ = _env_likelihood_terms(G, tm, act, obs, on, None, env, None)
    assert torch.allclose(r_off, G), "flag off must not change the return"
    assert torch.allclose(r_on, G + ER), "flag on must add the env return"
    assert "envlp/return_env_mean" in m_off, "the term is measured either way"


def test_absent_env_term_is_a_no_op():
    tm = torch.tensor([[ACT, OBS, ACT, OBS]])
    sur = torch.tensor([[0.0, 1.0, 0.0, 2.0]])
    G = _compute_returns(sur, tm, 1.0)
    cfg = TrainingConfig(gamma=1.0)
    r, er, m = _env_likelihood_terms(G, tm, tm == ACT, tm == OBS, cfg, None, None, None)
    assert torch.allclose(r, G) and er is None and m == {}


def test_rollout_batch_carries_the_term_and_stays_backward_compatible():
    def traj(with_env: bool):
        steps = [
            EpisodeStep("a", [1, 2], TokenType.OBS, [], {},
                        env_log_probs=[-0.5, -1.5] if with_env else None,
                        env_lp_known=[True, False] if with_env else None),
            EpisodeStep("a", [3, 4], TokenType.ACT, [-0.1, -0.2], {}),
        ]
        return Trajectory(steps=steps, agent_ids_present=["a"])

    for builder in (RolloutBatch.from_trajectories, RolloutBatch.pack_trajectories):
        b = builder([traj(True), traj(True)], {"a": 0}, 0)
        assert b.env_log_probs is not None and b.env_lp_known is not None
        assert b.env_log_probs.shape == b.input_ids.shape
        assert b.to("cpu").env_log_probs is not None            # .to() carries it
        legacy = builder([traj(False)], {"a": 0}, 0)
        assert legacy.env_log_probs is None and legacy.env_lp_known is None


def test_compute_loss_integrates_the_env_term_and_the_probe():
    """The real compute_loss path, on synthetic tensors.

    Exercises what the unit tests above do not: slicing the new batch tensors
    into the loss, the metrics it emits, that the flag actually changes the
    advantages, and that the probe head's gradient reaches the probe and
    nothing else.
    """
    import torch.nn as nn
    from marlllm.loss import CCSMLoss
    from marlllm.agent import ValueHead

    torch.manual_seed(0)
    B, T, H, V = 2, 12, 16, 32
    ids = torch.randint(0, V, (B, T))
    tm = torch.full((B, T), OBS)
    tm[:, 2::3] = ACT                      # a few action positions
    tm[:, 0] = OBS                         # position 0 is never ACT
    am = torch.zeros((B, T), dtype=torch.long)
    trunk = nn.Linear(H, H)
    lm_head = nn.Linear(H, V)
    vhead, phead = ValueHead(H), ValueHead(H)

    env_lp = torch.zeros(B, T)
    env_lp[tm == OBS] = -torch.rand((tm == OBS).sum()) * 2.0   # ln p_env <= 0
    known = torch.ones(B, T, dtype=torch.bool)

    def run(use_corrected, with_probe):
        for m in (trunk, lm_head, vhead, phead):
            m.zero_grad(set_to_none=True)
        hidden = trunk(torch.randn(B, T, H, generator=torch.Generator().manual_seed(1)))
        cfg = TrainingConfig(gamma=0.99, normalise_returns=False,
                             use_env_likelihood_return=use_corrected)
        loss, metrics = CCSMLoss().compute_loss(
            last_hidden=hidden, lm_head=lm_head, values=vhead(hidden.detach()),
            input_ids=ids, token_type_mask=tm, agent_id_mask=am,
            target_agent_idx=0, config=cfg,
            env_log_probs=env_lp, env_lp_known=known,
            bias_values=(phead(hidden.detach()) if with_probe else None),
        )
        loss.backward()
        return metrics, hidden

    m_off, _ = run(False, True)
    m_on, _ = run(True, True)

    for k in ("envlp/return_surrogate_mean", "envlp/return_env_mean",
              "envlp/return_corrected_mean", "envlp/env_share_of_return",
              "envlp/return_rank_corr", "envlp/obs_tokens_measured_frac",
              "envlp/probe_loss", "envlp/bias_corr_adv_resid"):
        assert k in m_off, f"missing metric {k}"
    assert m_off["envlp/obs_tokens_measured_frac"] == 1.0
    assert m_off["envlp/corrected_return_in_use"] == 0.0
    assert m_on["envlp/corrected_return_in_use"] == 1.0
    # The env term is negative, so the corrected return is strictly smaller.
    assert m_off["envlp/return_env_mean"] < 0
    assert (m_off["envlp/return_corrected_mean"]
            < m_off["envlp/return_surrogate_mean"])
    # Mean advantage must differ once the term is in the return.
    assert abs(m_on["mean_advantage"] - m_off["mean_advantage"]) > 1e-9

    # The probe trains itself and leaks nowhere.
    for m in (trunk, lm_head, vhead, phead):
        m.zero_grad(set_to_none=True)
    hidden = trunk(torch.randn(B, T, H))
    cfg = TrainingConfig(gamma=0.99, alpha_perc=0.0, alpha_act=0.0, alpha_val=0.0,
                         normalise_returns=False)
    loss, _ = CCSMLoss().compute_loss(
        last_hidden=hidden, lm_head=lm_head, values=vhead(hidden.detach()),
        input_ids=ids, token_type_mask=tm, agent_id_mask=am, target_agent_idx=0,
        config=cfg, env_log_probs=env_lp, env_lp_known=known,
        bias_values=phead(hidden.detach()))
    loss.backward()
    assert phead.linear.weight.grad is not None, "probe did not train"
    # The other terms are weighted to zero, so their paths may still exist with
    # an all-zero gradient. What must hold is that the probe contributes none.
    for mod, name in ((trunk, "trunk"), (vhead, "value head")):
        w = mod.weight if name == "trunk" else mod.linear.weight
        assert w.grad is None or torch.count_nonzero(w.grad) == 0, \
            f"probe leaked into the {name}"

    # Legacy call: no env arguments at all.
    for m in (trunk, lm_head, vhead):
        m.zero_grad(set_to_none=True)
    hidden = trunk(torch.randn(B, T, H))
    _, m_legacy = CCSMLoss().compute_loss(
        last_hidden=hidden, lm_head=lm_head, values=vhead(hidden.detach()),
        input_ids=ids, token_type_mask=tm, agent_id_mask=am, target_agent_idx=0,
        config=TrainingConfig(gamma=0.99))
    assert not any(k.startswith("envlp/") for k in m_legacy), \
        "legacy call must emit no environment-likelihood metrics"


def test_spearman():
    a = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert abs(_spearman(a, a) - 1.0) < 1e-5
    assert abs(_spearman(a, -a) + 1.0) < 1e-5
    assert _spearman(torch.tensor([1.0]), torch.tensor([1.0])) == 0.0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception as e:
            failed += 1
            print(f"  FAIL  {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
