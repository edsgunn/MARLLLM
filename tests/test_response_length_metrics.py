"""Per-agent response length metrics.

`_compute_generation_metrics` emits two views of the same token accounting:

  * ``gen/<env>/…``  — the population-wide mean, which is what existed before.
  * ``<agent>/…``    — per agent, which is the only view that can tell a
                       population where ONE member went quiet from one where
                       everybody did.

That distinction is not cosmetic. Under partial freezing most members cannot
move at all, so the population mean is dominated by agents that are pinned by
construction; the frozen-majority reading in
`runs/cultural_emergence/c2_population_size/NOTES.md` was withdrawn once for
exactly this reason.

What must hold:

  1. Public length (`mean_post_tokens`) counts only what lands inside the post
     tags; total length (`mean_action_tokens`) counts everything generated.
  2. A per-agent mean is over that agent's OWN turns, not the population's.
  3. A silent turn is counted, and pulls the public mean down without changing
     the length of the posts that do exist.
  4. The keys land in the same `<agent>/<metric>` namespace as the loss
     metrics, so the plotter discovers them and the regime classifier ignores
     them.

Run:  .venv/bin/python tests/test_response_length_metrics.py
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("HF_HOME", "/lus/lfs1aip2/projects/a5l/egunn/hf_cache")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from marlllm.population import PopulationTrainer   # noqa: E402
from marlllm.regime_tracking import _parse_agent_metrics  # noqa: E402
from marlllm.types import EpisodeStep, TokenType   # noqa: E402


class _Tok:
    """Character-level stand-in: one token per character, so token counts in
    the assertions below are just string lengths."""

    def decode_action(self, ids):
        return "".join(chr(i) for i in ids)

    def encode_observation(self, text):
        return [ord(c) for c in text]


class _Env:
    _post_open_tag = "<post>"
    _post_close_tag = "</post>"
    _post_token_budget = 128
    _total_token_budget = 384


class _Spec:
    def __init__(self):
        self.name = "forum"
        self.env = _Env()


def _enc(text):
    return [ord(c) for c in text]


def _trainer():
    t = object.__new__(PopulationTrainer)
    t.tokeniser = _Tok()
    t.environments = [_Spec()]
    return t


def _episode(turns):
    """turns: list of (agent_id, full action text). Returns one _RawEpisode."""
    history = [
        EpisodeStep(agent_id=a, token_ids=_enc(txt),
                    token_type=TokenType.ACT, log_probs=[])
        for a, txt in turns
    ]
    return (history, {}, (), {}, "forum", {}, None)


def test_public_vs_total_length():
    """Public counts the post body; total counts the whole generation."""
    # 20 chars of thinking, a 9-char post body inside 13 chars of tags.
    text = "think think think<post>some post</post>"
    m = _trainer()._compute_generation_metrics([_episode([("Ada", text)])])
    assert m["Ada/mean_post_tokens"] == len("some post"), m["Ada/mean_post_tokens"]
    assert m["Ada/mean_action_tokens"] == len(text), m["Ada/mean_action_tokens"]
    # thinking is the remainder, so the two panels decompose the third
    assert (m["Ada/mean_thinking_tokens"]
            == m["Ada/mean_action_tokens"] - m["Ada/mean_post_tokens"])
    print("PASS public vs total length")


def test_per_agent_means_are_not_population_means():
    """One long agent and one short one must not be averaged together."""
    long_txt  = "<post>" + "x" * 100 + "</post>"
    short_txt = "<post>" + "y" * 4 + "</post>"
    m = _trainer()._compute_generation_metrics([
        _episode([("Long", long_txt), ("Short", short_txt)])
    ])
    assert m["Long/mean_post_tokens"] == 100
    assert m["Short/mean_post_tokens"] == 4
    # the population view averages them; the per-agent view must not
    assert m["gen/forum/mean_post_tokens"] == 52
    print("PASS per-agent means are per agent")


def test_silence_is_separated_from_brevity():
    """A silent turn drags the public mean down while posts stay long."""
    post = "<post>" + "z" * 50 + "</post>"
    silent = "thinking only, no post block"
    m = _trainer()._compute_generation_metrics([
        _episode([("Ada", post), ("Ada", silent)])
    ])
    assert m["Ada/silent_turn_rate"] == 0.5, m["Ada/silent_turn_rate"]
    assert m["Ada/mean_post_tokens"] == 25.0            # 50 over two turns
    assert m["Ada/mean_post_tokens_when_posting"] == 50.0  # unchanged
    print("PASS silence separated from brevity")


def test_multiple_turns_average_over_own_turns_only():
    a = "<post>" + "a" * 10 + "</post>"
    b = "<post>" + "b" * 30 + "</post>"
    m = _trainer()._compute_generation_metrics([
        _episode([("Ada", a), ("Bob", b), ("Ada", b)])
    ])
    assert m["Ada/mean_post_tokens"] == 20.0   # (10 + 30) / 2
    assert m["Bob/mean_post_tokens"] == 30.0
    print("PASS averaging is over each agent's own turns")


def test_obs_steps_are_not_counted():
    """Only ACT tokens are the agent's response; observations are not."""
    hist = [
        EpisodeStep("Ada", _enc("<post>" + "q" * 8 + "</post>"), TokenType.ACT, []),
        EpisodeStep("Ada", _enc("x" * 500), TokenType.OBS, []),
    ]
    m = _trainer()._compute_generation_metrics([(hist, {}, (), {}, "forum", {}, None)])
    assert m["Ada/mean_post_tokens"] == 8
    assert m["Ada/mean_action_tokens"] == len("<post>" + "q" * 8 + "</post>")
    print("PASS observation steps excluded")


def test_keys_are_inert_for_the_regime_classifier():
    """New per-agent keys must not make the classifier think it has an agent."""
    m = _trainer()._compute_generation_metrics([
        _episode([("Ada", "<post>hello</post>")])
    ])
    assert _parse_agent_metrics(m) == {}, _parse_agent_metrics(m)
    # and it still works when the loss fields ARE present
    m.update({"Ada/kl": 0.5, "Ada/perc_loss": 2.0, "Ada/entropy": 0.8})
    parsed = _parse_agent_metrics(m)
    assert set(parsed) == {"Ada"}
    assert set(parsed["Ada"]) == {"kl", "perc_loss", "entropy"}
    print("PASS keys inert for the regime classifier")


def test_untagged_env_treats_everything_as_public():
    """Substrates without post tags have no private region to separate."""
    t = _trainer()
    t.environments[0].env = type("E", (), {})()   # no tag attributes
    m = t._compute_generation_metrics([_episode([("Ada", "plain response")])])
    assert m["Ada/mean_post_tokens"] == m["Ada/mean_action_tokens"]
    assert m["Ada/mean_thinking_tokens"] == 0
    print("PASS untagged substrate: public == total")


if __name__ == "__main__":
    test_public_vs_total_length()
    test_per_agent_means_are_not_population_means()
    test_silence_is_separated_from_brevity()
    test_multiple_turns_average_over_own_turns_only()
    test_obs_steps_are_not_counted()
    test_keys_are_inert_for_the_regime_classifier()
    test_untagged_env_treats_everything_as_public()
    print("\nall response-length metric tests passed")
