"""Tests for per-episode roster subsets and the fixed partner graph.

Forum environments have always bound every population member into every
episode, so partner count was pinned to N-1 and could not be varied
independently of population size. `pairing_strategy` looks like the knob for
this and is not — it is only read by pair-mode envs ({agent_0, agent_1}), and
is a no-op on every forum run in this project.

`partner_graph_degree` k partitions the population into disjoint groups of k+1
and runs each episode with one group. What has to hold:

  1. Default behaviour is unchanged — the whole roster, every episode.
  2. Groups are a partition: every member in exactly one, all the same size.
  3. Groups are cycled, not sampled, so per-agent gradient volume stays even.
  4. The partition is seed-independent, so a seed replicate varies training
     noise without also varying who talks to whom.
  5. Bad degrees are refused at construction, not at episode 4000.

Run:  .venv/bin/python tests/test_partner_graph.py
(pytest is not installed in the Isambard venv; this file is also importable
as a pytest module.)
"""
from __future__ import annotations

import os
import sys
from collections import Counter

os.environ.setdefault("HF_HOME", "/lus/lfs1aip2/projects/a5l/egunn/hf_cache")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from marlllm.config import TrainingConfig          # noqa: E402
from marlllm.population import PopulationTrainer   # noqa: E402

ROSTER16 = [f"agent{i:02d}" for i in range(16)]


def _trainer(names, degree=None):
    """Run the real partner-graph setup on a stub population."""
    t = object.__new__(PopulationTrainer)
    PopulationTrainer._setup_partner_graph(
        t, {n: object() for n in names},
        TrainingConfig(partner_graph_degree=degree),
    )
    return t


# ------------------------------------------------------------------ #
def test_unconfigured_is_the_previous_behaviour():
    t = _trainer(ROSTER16)
    assert t.partner_groups is None and t._group_cycle is None
    # The roster handed back is the one passed in, unchanged and not copied
    # into some subset.
    assert PopulationTrainer._episode_roster(t, ROSTER16) == ROSTER16


def test_groups_partition_the_population():
    t = _trainer(ROSTER16, degree=3)
    assert len(t.partner_groups) == 4
    assert all(len(g) == 4 for g in t.partner_groups)
    flat = [n for g in t.partner_groups for n in g]
    assert sorted(flat) == sorted(ROSTER16), "not a partition"
    assert len(set(flat)) == len(flat), "a member appears twice"


def test_every_agent_has_exactly_k_partners_and_always_the_same_k():
    t = _trainer(ROSTER16, degree=3)
    partners = {}
    for _ in range(64):
        roster = PopulationTrainer._episode_roster(t, ROSTER16)
        assert len(roster) == 4
        for a in roster:
            partners.setdefault(a, set()).update(set(roster) - {a})
    assert len(partners) == 16, "not every agent appeared"
    for a, ps in partners.items():
        assert len(ps) == 3, f"{a} saw {len(ps)} partners, expected 3"


def test_groups_are_cycled_so_volume_stays_even():
    """16 episodes over 4 groups must be 4 episodes each, not 7/4/3/2."""
    t = _trainer(ROSTER16, degree=3)
    seen = Counter()
    for _ in range(16):
        seen[tuple(PopulationTrainer._episode_roster(t, ROSTER16))] += 1
    assert len(seen) == 4 and set(seen.values()) == {4}, dict(seen)


def test_partition_does_not_depend_on_the_seed():
    a = _trainer(ROSTER16, degree=3).partner_groups
    b = _trainer(ROSTER16, degree=3).partner_groups
    assert a == b, "the partition moved between constructions"


def test_a_group_that_misses_an_env_roster_is_skipped():
    """Environments may declare different rosters; an episode still needs two."""
    t = _trainer(ROSTER16, degree=3)
    small_env = ROSTER16[:4]                       # only group 0 overlaps
    for _ in range(8):
        roster = PopulationTrainer._episode_roster(t, small_env)
        assert set(roster) <= set(small_env) and len(roster) >= 2


def test_incompatible_rosters_are_refused_loudly():
    t = _trainer(ROSTER16, degree=3)
    try:
        PopulationTrainer._episode_roster(t, ["agent00", "agent04"])  # 1 per group
    except ValueError as e:
        assert "incompatible" in str(e).lower()
    else:
        raise AssertionError("an unsatisfiable roster was accepted")


def test_a_degree_that_does_not_divide_the_population_is_refused():
    for degree, why in ((4, "groups of 5 do not divide 16"),
                        (0, "degree below 1"),
                        (16, "group larger than the population")):
        try:
            _trainer(ROSTER16, degree=degree)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted {degree}: {why}")


# ------------------------------------------------------------------ #
# End-to-end against a real ForumEnv                                   #
# ------------------------------------------------------------------ #
_TOK = None


def _tokenizer():
    """Load once — these tests build many envs and the load dominates."""
    global _TOK
    if _TOK is None:
        from transformers import AutoTokenizer
        _TOK = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-7B-Instruct", local_files_only=True)
    return _TOK


def _forum(names):
    from envs.forum.env import ForumEnv
    tok = _tokenizer()
    return ForumEnv(
        agent_names=names, agent_personas={n: f"You are {n}." for n in names},
        tokenizer=tok, total_token_budget=384, post_token_budget=128,
        max_posts=8, seed=0)


def test_forum_env_runs_a_subset_and_reports_it():
    names = [f"A{i}" for i in range(8)]
    env = _forum(names)
    subset = ["A1", "A3", "A5", "A7"]
    env.reset(seed=0, options={"order_seed": 0, "active_agents": subset})

    assert set(env.agents) == set(subset), env.agents
    assert list(env.possible_agents) == names, "possible_agents was mutated"
    # Turn order is then shuffled from order_seed, exactly as it is for a full
    # roster — the subset is *selected* in possible_agents order so that which
    # agents take part cannot depend on how the group list was built, but the
    # order they speak in is the env's business as usual.
    again = _forum(names)
    again.reset(seed=0, options={"order_seed": 0, "active_agents": subset[::-1]})
    assert again.agents == env.agents, "turn order depended on the subset's order"
    assert set(env.episode_trace()["agents"]) == set(subset)
    # Only the agents present take a turn.
    for _ in range(8):
        if not env.agents:
            break
        assert env.agent_selection in set(subset)
        env.step(env._tok.encode("<post>hi</post>", add_special_tokens=False))
    assert set(env.episode_trace()["agents"]) == set(subset)


def test_no_active_agents_is_byte_identical_to_before():
    """The 45 jobs queued when this landed must be unaffected."""
    names = [f"A{i}" for i in range(8)]
    a, b = _forum(names), _forum(names)
    a.reset(seed=7, options={"order_seed": 3})
    b.reset(seed=7, options={"order_seed": 3, "active_agents": None})
    assert a.agents == b.agents, "active_agents=None changed the turn order"
    assert set(a.agents) == set(names) and len(a.agents) == 8
    # And the same thread comes out of both.
    for env in (a, b):
        for _ in range(8):
            if not env.agents:
                break
            env.step(env._tok.encode("<post>x</post>", add_special_tokens=False))
    assert a.episode_trace()["thread"] == b.episode_trace()["thread"]
    assert a.episode_trace()["agents"] == b.episode_trace()["agents"] == names


def test_a_subset_naming_an_unknown_agent_is_refused():
    env = _forum([f"A{i}" for i in range(4)])
    try:
        env.reset(seed=0, options={"active_agents": ["A0", "Nobody"]})
    except ValueError as e:
        assert "not in possible_agents" in str(e)
    else:
        raise AssertionError("an unknown active agent was accepted")


def test_a_subset_of_one_is_refused():
    env = _forum([f"A{i}" for i in range(4)])
    try:
        env.reset(seed=0, options={"active_agents": ["A0"]})
    except ValueError as e:
        assert "at least 2" in str(e)
    else:
        raise AssertionError("a single-agent episode was accepted")


# ------------------------------------------------------------------ #
# The trainer's own binding, against real ForumEnv specs               #
# ------------------------------------------------------------------ #
def _spec(env, name="forum"):
    from marlllm.population import EnvironmentSpec
    return EnvironmentSpec(env=env, name=name)


def _bind(names, degree, n_episodes):
    """Run the shipped binding method over n_episodes of one forum spec."""
    t = _trainer(names, degree=degree)
    t.pairing_strategy = "random_no_self"
    t.population = {n: object() for n in names}
    specs = [_spec(_forum(names)) for _ in range(n_episodes)]
    return t, specs, PopulationTrainer._bind_episode_roles(t, specs)


def test_binding_restricts_roles_to_the_episode_roster():
    names = [f"A{i}" for i in range(8)]
    _, _, (pairings, role_map, active) = _bind(names, degree=3, n_episodes=4)
    for pr, rm, ac in zip(pairings, role_map, active):
        assert len(pr) == 4 and len(rm) == 4 and ac is not None
        assert set(pr) == set(rm) == set(ac), "the three views disagree"
        assert all(k == v for k, v in rm.items()), "named mode must bind 1:1"


def test_binding_leaves_the_default_path_alone():
    """What every queued job does: no graph, whole roster, active_rosters None."""
    names = [f"A{i}" for i in range(8)]
    _, _, (pairings, role_map, active) = _bind(names, degree=None, n_episodes=3)
    for pr, rm, ac in zip(pairings, role_map, active):
        assert ac is None, "a subset was passed to an unconfigured run"
        assert tuple(pr) == tuple(names) and set(rm) == set(names)


def test_the_bound_roster_survives_deepcopy_and_reset():
    """The integration the unit tests cannot reach: binding -> copy -> reset."""
    import copy
    names = [f"A{i}" for i in range(8)]
    _, specs, (pairings, _, active) = _bind(names, degree=3, n_episodes=4)
    for spec, pr, ac in zip(specs, pairings, active):
        env = copy.deepcopy(spec.env)
        env._tok = spec.env._tok
        env.reset(seed=0, options={"order_seed": 0, "active_agents": ac})
        assert set(env.agents) == set(pr), "env roster != bound pairing"
        assert set(env.episode_trace()["agents"]) == set(pr)
        assert list(env.possible_agents) == names, "template roster was mutated"


def test_trajectory_filter_sees_only_the_episode_members():
    """_build_agent_trajectories selects on `pop_name in ep[2]`, so a subset
    episode must not build trajectories for members who never spoke."""
    names = [f"A{i}" for i in range(8)]
    _, _, (pairings, _, _) = _bind(names, degree=3, n_episodes=8)
    for name in names:
        mine = [pr for pr in pairings if name in pr]
        assert 0 < len(mine) < len(pairings), f"{name} in {len(mine)}/8 episodes"
        for pr in mine:
            assert len(pr) == 4


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ok  {name}")
    print("\nall partner-graph tests pass")
