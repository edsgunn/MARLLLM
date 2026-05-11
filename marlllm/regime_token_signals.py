"""Per-agent token-level diagnostic signals derived from rollout traces.

These are the Tier-3 signals from the regime-tracking integration plan —
features that don't come from existing metrics (KL/perception loss/entropy)
but from looking at the actual tokens the model emitted.

- **tail_loop_rate**: fraction of ACT generations whose tail after the last
  `</post>` (or equivalent close-tag) contains a short n-gram that repeats
  three or more times consecutively. This is the smoking gun for the
  ``</post>\\n</python>`` style policy collapse — the visible post text reads
  fine but the model has locked into a closing-tag loop afterwards.

- **non_ascii_rate**: fraction of generated tokens that decode to text
  containing characters outside the printable ASCII range. For
  English-language substrates a sustained non-zero rate is the
  commitment-overshoot pattern (Chinese characters injected into Kai
  Dempsey's late-iter posts, etc.).

- **top_token_frac**: fraction of generated tokens that equal the single
  most-frequent token for that agent at this iteration. Higher = the
  distribution is concentrated on one token = collapse signal that
  pre-empts the entropy threshold by ~10 iters.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterable

from marlllm.types import EpisodeStep, TokenType


def _has_consecutive_repeats(token_ids: list[int], min_run: int = 3,
                             max_ngram: int = 3) -> bool:
    """Return True if `token_ids` contains any n-gram (length 1..max_ngram)
    that repeats `min_run` or more times consecutively. O(N·max_ngram)."""
    n = len(token_ids)
    if n < min_run * 1:
        return False
    for ng in range(1, max_ngram + 1):
        if n < ng * min_run:
            continue
        i = 0
        while i <= n - ng * min_run:
            block = token_ids[i:i + ng]
            ok = True
            for k in range(1, min_run):
                if token_ids[i + k * ng:i + (k + 1) * ng] != block:
                    ok = False
                    break
            if ok:
                return True
            i += 1
    return False


def _is_non_ascii(text: str) -> bool:
    return any(ord(c) > 127 for c in text)


def compute_per_agent_token_signals(
    raw_episodes: Iterable[tuple],
    tokeniser,
    post_close_tag: str | None = "</post>",
) -> dict[str, float]:
    """Per-agent rollout-derived signals.

    `raw_episodes` follows the population.py `_RawEpisode` tuple layout:
        (history, ep_info, pairing, prompt_ids, env_name, ctx, env_trace)
    where each step in `history` is a `EpisodeStep`.

    Returns a flat dict keyed `<agent_id>/<metric>`. Returns empty dict if
    no ACT steps are seen.
    """
    n_acts: dict[str, int] = defaultdict(int)
    n_tail_loops: dict[str, int] = defaultdict(int)
    total_tokens: dict[str, int] = defaultdict(int)
    non_ascii_tokens: dict[str, int] = defaultdict(int)
    token_counts: dict[str, Counter] = defaultdict(Counter)

    for raw in raw_episodes:
        if not raw:
            continue
        history = raw[0]
        for step in history:
            if not isinstance(step, EpisodeStep):
                continue
            if step.token_type != TokenType.ACT:
                continue
            agent = step.agent_id
            ids = list(step.token_ids)
            if not ids:
                continue
            n_acts[agent] += 1
            total_tokens[agent] += len(ids)
            token_counts[agent].update(ids)

            text = ""
            try:
                text = tokeniser.decode_action(ids)
            except Exception:
                # Tokenisers without decode_action — degrade gracefully.
                pass
            if text and _is_non_ascii(text):
                # Cheap upper bound: count tokens whose decoded text is non-ASCII.
                for tid in ids:
                    try:
                        if _is_non_ascii(tokeniser.decode_action([tid])):
                            non_ascii_tokens[agent] += 1
                    except Exception:
                        break

            # Tail-loop detection — look at tokens after the last close tag.
            tail_ids = ids
            if post_close_tag and text:
                idx = text.rfind(post_close_tag)
                if idx >= 0:
                    tail_text = text[idx + len(post_close_tag):]
                    if not tail_text.strip():
                        # Nothing after — fine, no loop.
                        continue
                    try:
                        tail_ids = tokeniser.encode_observation(tail_text)
                    except Exception:
                        tail_ids = ids
            if _has_consecutive_repeats(tail_ids, min_run=3, max_ngram=3):
                n_tail_loops[agent] += 1

    out: dict[str, float] = {}
    for agent, n_a in n_acts.items():
        out[f"{agent}/tail_loop_rate"] = n_tail_loops.get(agent, 0) / n_a
        total = total_tokens.get(agent, 0) or 1
        out[f"{agent}/non_ascii_rate"] = non_ascii_tokens.get(agent, 0) / total
        counts = token_counts.get(agent)
        if counts:
            top_count = counts.most_common(1)[0][1]
            out[f"{agent}/top_token_frac"] = top_count / total
        else:
            out[f"{agent}/top_token_frac"] = 0.0
    return out
