"""Per-agent plot rows must be labelled with the agent that produced them.

THE BUG THIS PINS DOWN

`load_metrics` stacks `<agent>/<metric>` into `agents/<metric>`, skipping
agents that lack the metric — but it used to keep the FULL roster in
`agent_names`. Every panel then labelled row i with `agent_names[i]`.

That is harmless when every agent emits every metric, and wrong the moment
one does not. Under partial freezing (REWRITE.md C3) only the learner has
loss metrics while all members have generation metrics, so `agents/perc_loss`
had a single row — the learner's — which was drawn and legended as whichever
agent sorted first alphabetically. In `c3_frozen_pop_learner_priya_seed1` the
learner is Priya Shah and her curves were labelled "Beatrice Okafor".

Two further consequences fixed at the same time:

  * `n_agents` came from the row count of `agents/mean_return`, so a frozen
    run's legend showed ONE agent for an eight-member population.
  * Colours were assigned by row index within each panel, so an agent changed
    colour between a panel where all eight appear and one where only the
    learner does.

Run:  .venv/bin/python tests/test_plot_agent_labels.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "scripts"))

import matplotlib                                     # noqa: E402
matplotlib.use("Agg")
from plot_results import load_metrics, plot_experiment  # noqa: E402

AGENTS = ["Beatrice Okafor", "Hana Yilmaz", "Priya Shah", "Tom Whitaker"]
LEARNER = "Priya Shah"          # deliberately NOT first alphabetically


def _frozen_run(tmp: Path) -> Path:
    """A run where only the learner has loss metrics, as under freezing."""
    d = tmp / "frozen"
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps({
        "environments": [{"name": "forum",
                          "post_token_budget": 128,
                          "total_token_budget": 384}],
        "frozen_agents": [a for a in AGENTS if a != LEARNER],
    }))
    rows = []
    for it in range(1, 31):
        r = {"iteration": it, "total_loss": 3.0}
        # loss metrics: learner only
        r[f"{LEARNER}/perc_loss"] = 2.0
        r[f"{LEARNER}/act_loss"] = 0.1
        r[f"{LEARNER}/value_loss"] = 1.0
        r[f"{LEARNER}/kl"] = 0.01 * it
        r[f"{LEARNER}/entropy"] = 0.8
        r[f"{LEARNER}/mean_return"] = -0.2
        # generation metrics: every member
        for i, a in enumerate(AGENTS):
            r[f"{a}/mean_post_tokens"] = 50.0 + 10 * i
            r[f"{a}/mean_action_tokens"] = 70.0 + 10 * i
            r[f"{a}/mean_thinking_tokens"] = 20.0
            r[f"{a}/silent_turn_rate"] = 0.0
        r[f"{LEARNER}/entropy_action_total"] = 0.8 * (70.0 + 10 * AGENTS.index(LEARNER))
        # non-agent namespaces that must NOT become legend entries
        r["per_agent/fb_s_max"] = 1.0
        r["pop/n_frozen"] = 3.0
        r["util/x"] = 1.0
        r["async/x"] = 1.0
        rows.append(r)
    (d / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    return d


def test_row_names_match_the_contributing_agents():
    with tempfile.TemporaryDirectory() as tmp:
        data = load_metrics(_frozen_run(Path(tmp)))
        # the learner's loss row must be named the LEARNER, not the first agent
        assert data["agents/perc_loss"].shape[0] == 1
        assert data["agent_names/perc_loss"] == [LEARNER], \
            data["agent_names/perc_loss"]
        assert data["agent_names/perc_loss"][0] != AGENTS[0], \
            "regression: learner mislabelled as the alphabetically-first agent"
        # generation metrics cover everyone
        assert data["agents/mean_post_tokens"].shape[0] == len(AGENTS)
        assert data["agent_names/mean_post_tokens"] == sorted(AGENTS)
        print("PASS row names match contributing agents")


def test_roster_excludes_non_agent_namespaces():
    with tempfile.TemporaryDirectory() as tmp:
        data = load_metrics(_frozen_run(Path(tmp)))
        assert set(data["agent_names"]) == set(AGENTS), data["agent_names"]
        for junk in ("per_agent", "pop", "util", "async"):
            assert junk not in data["agent_names"], junk
        print("PASS roster excludes non-agent namespaces")


def test_stacked_rows_carry_the_right_values():
    """Row order must follow the recorded names, not the roster."""
    with tempfile.TemporaryDirectory() as tmp:
        data = load_metrics(_frozen_run(Path(tmp)))
        names = data["agent_names/mean_post_tokens"]
        arr = data["agents/mean_post_tokens"]
        for i, name in enumerate(names):
            expected = 50.0 + 10 * AGENTS.index(name)
            assert arr[i][0] == expected, (name, arr[i][0], expected)
        print("PASS stacked rows carry the right values")


def test_figure_renders_and_legends_every_member():
    with tempfile.TemporaryDirectory() as tmp:
        d = _frozen_run(Path(tmp))
        out = plot_experiment(d)
        assert out is not None and out.exists() and out.stat().st_size > 0
        # the legend is built from the full roster, not from the row count of
        # any single metric — a frozen run showed one name for four members
        data = load_metrics(d)
        assert len(data["agent_names"]) == len(AGENTS)
        print("PASS figure renders and legends every member")


if __name__ == "__main__":
    test_row_names_match_the_contributing_agents()
    test_roster_excludes_non_agent_namespaces()
    test_stacked_rows_carry_the_right_values()
    test_figure_renders_and_legends_every_member()
    print("\nall plot-label tests passed")
