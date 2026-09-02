"""
Append-only forum environment for CCSM training.

Designed to mirror DealOrNoDealEnv's observation shape so an instruct-tuned
model handles it as conversational chat (which it does naturally) rather
than as an open-ended posting task (which it doesn't).

Observation sequencing
----------------------
Each agent's chat log grows naturally one user/assistant pair at a time:

    Turn 1   first agent obs → ctx_initial                   acts → post A
    Turn 2   second agent obs → ctx_initial + "A wrote: …"   acts → post B
    Turn 3   first agent obs → "B wrote: …"                  acts → post A2
    Turn 4   second agent obs → "A2 wrote: …"                acts → post B2
    ...

Three properties enforced by this design:

  1. **Strict monotonicity at the per-agent context level.**  The trainer
     uses ``obs_is_full_context = False`` (the default).  Each ``observe``
     call returns ONLY the new content since this agent last acted.  The
     trainer wraps each delivery as a fresh ``<|im_start|>user…<|im_end|>``
     turn and appends it; nothing is ever rewritten.

  2. **Conversational framing.**  Each delivery is rendered as direct
     speech (``"<Speaker> wrote:\\n<text>"``) — no meta-instructions like
     "Compose your next post".  This matches the chat structure the
     instruct model was RLHF'd on, so it keeps responding instead of
     interpreting one round as task-completion.

  3. **Speaker-role correctness.**  Own posts stay tagged ``assistant``,
     others' posts stay tagged ``user``.  CCSM perception/action loss
     tagging works unchanged.

Persona handling
----------------
Personas are exposed via the ``default_character_prompts`` property and
the trainer plumbs them in as the system message at episode start
(via ``EnvironmentSpec.character_prompts`` → ``formatter.wrap_prompt``).
The persona text is NEVER repeated in observations.

Memory tool use (NOT implemented)
---------------------------------
A future ``recall`` tool would be a learned action emitted in the
agent's generated tokens.  The env intercepts the tool call, executes a
retrieval, and **appends** the result as a new ``user`` delivery in the
agent's chat log.  Retrieval becomes an action within the stream, never
a preprocessor over it.
"""
from __future__ import annotations

import ast
import json
import multiprocessing as _mp
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

from pettingzoo import AECEnv
from transformers import PreTrainedTokenizerBase


# ── pybot fast path ──────────────────────────────────────────────────────────
# A forkserver-backed worker drops per-call cost from a cold subprocess
# launch (~80–200 ms on ARM Grace) to a fork from a pre-warmed helper
# (~5–15 ms).  The helper is created lazily on first call and shared by
# all ``ForumEnv`` instances in the process; each request still spawns
# a fresh child for hard isolation (no state leakage between calls).

_PYEXEC_CTX: Any = None


def _get_pyexec_ctx() -> Any:
    global _PYEXEC_CTX
    if _PYEXEC_CTX is None:
        _PYEXEC_CTX = _mp.get_context("forkserver")
    return _PYEXEC_CTX


_PYBOT_RLIMITS = {
    # Address space cap. Matplotlib + numpy can be hungry; 1.5 GB is comfortable.
    "AS": 1_500 * 1024 * 1024,
    # Max single-file write. Big enough for plots, small enough that
    # accidental "write 100M zeros" stops fast.
    "FSIZE": 64 * 1024 * 1024,
    # Cap forked children so a fork-bomb can't escape the timeout.
    "NPROC": 64,
}


def _apply_pybot_sandbox(cwd: str, env: dict[str, str]) -> None:
    """Chdir, replace env, and apply rlimits. Called inside the child."""
    os.chdir(cwd)
    os.environ.clear()
    os.environ.update(env)
    try:
        import resource

        for name, lim in _PYBOT_RLIMITS.items():
            r = getattr(resource, f"RLIMIT_{name}", None)
            if r is None:
                continue
            try:
                soft, hard = resource.getrlimit(r)
                new_hard = lim if hard == resource.RLIM_INFINITY else min(lim, hard)
                resource.setrlimit(r, (min(lim, new_hard), new_hard))
            except (ValueError, OSError):
                pass
    except ImportError:
        pass


def _pyexec_worker(code: str, conn: Any, cwd: str, env: dict[str, str]) -> None:
    """Run ``code`` with stdout/stderr captured; send (out, err, rc) back."""
    import contextlib
    import io
    import traceback

    _apply_pybot_sandbox(cwd, env)

    out_buf, err_buf = io.StringIO(), io.StringIO()
    rc = 0
    try:
        with contextlib.redirect_stdout(out_buf), contextlib.redirect_stderr(err_buf):
            exec(compile(code, "<pybot>", "exec"), {"__name__": "__main__"})
    except SystemExit as e:
        rc = e.code if isinstance(e.code, int) else 1
    except BaseException:
        traceback.print_exc(file=err_buf)
        rc = 1
    try:
        conn.send((out_buf.getvalue(), err_buf.getvalue(), rc))
    finally:
        conn.close()


def _pybot_minimal_env(scratch: str) -> dict[str, str]:
    """Minimal env for pybot children: enough to import numpy/matplotlib/etc."""
    parent = os.environ
    env = {
        "PATH": parent.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "HOME": scratch,
        "TMPDIR": scratch,
        "PWD": scratch,
        "LANG": parent.get("LANG", "C.UTF-8"),
        "LC_ALL": parent.get("LC_ALL", "C.UTF-8"),
        # Headless matplotlib so plt.savefig works without a display.
        "MPLBACKEND": "Agg",
        # Keep the user's Python environment importable.
        "PYTHONPATH": parent.get("PYTHONPATH", ""),
    }
    for k in ("LD_LIBRARY_PATH", "LD_PRELOAD", "VIRTUAL_ENV", "CONDA_PREFIX"):
        if k in parent:
            env[k] = parent[k]
    return env


class ForumEnv(AECEnv):
    """Append-only multi-agent forum environment with conversational obs.

    Parameters
    ----------
    agent_names:
        Ordered list of agent slot names.  Used as ``possible_agents``.
    agent_personas:
        Map ``name → persona text``.  Exposed via
        ``default_character_prompts`` for the trainer to install as each
        agent's system prompt.  The env itself does NOT render personas
        into observations.
    tokenizer:
        HuggingFace tokenizer for encoding observations.
    forum_description:
        Brief description of the forum (used in the initial ctx).
    initial_invitation:
        Single-sentence call-to-action for the first observation each
        agent receives.  Default invites them to post on a new thread.
    action_token_budget:
        Tokens per turn; read by the trainer.
    max_posts:
        Total posts in the episode (across all agents).
    post_order:
        ``"round_robin"`` or ``"random"``.
    seed:
        RNG seed for ``random`` post ordering.
    reward_fn:
        ``(final_state, agent_name) → float``.  Defaults to zero.
    """

    metadata = {"render_modes": [], "name": "forum_v1"}

    # Trainer uses the standard incremental-append path; obs is delta-only.
    obs_is_full_context: bool = False

    def __init__(
        self,
        agent_names: list[str],
        agent_personas: dict[str, str],
        tokenizer: PreTrainedTokenizerBase,
        forum_description: str = (
            "An online forum where members post and reply to threads."
        ),
        initial_invitation: str = (
            "A new thread has just opened on the forum. The discussion is active."
        ),
        action_token_budget: int | None = None,
        max_posts: int = 12,
        post_order: str = "round_robin",
        seed: int | None = None,
        reward_fn: Callable[[Any, str], float] | None = None,
        post_length_note: str | None = None,
        post_open_tag: str = "<post>",
        post_close_tag: str = "</post>",
        provenance: bool = False,
        post_token_budget: int | None = None,
        total_token_budget: int | None = None,
        python_tool_enabled: bool = False,
        python_open_tag: str = "<python>",
        python_close_tag: str = "</python>",
        python_bot_name: str = "pybot",
        python_timeout_seconds: float = 5.0,
        python_output_max_chars: int = 2000,
        python_use_forkserver: bool = False,
        pybot_scratch_root: str | None = None,
        pybot_archive_dir: str | None = None,
    ) -> None:
        super().__init__()

        self.possible_agents = list(agent_names)
        self._episode_agents = list(agent_names)
        self._personas = dict(agent_personas)
        self._tok = tokenizer
        self._forum_description = forum_description
        self._initial_invitation = initial_invitation
        # Budget model:
        #   * ``post_token_budget``  — instructed cap on the public post
        #     (the part that lands in every OTHER agent's context).  The
        #     forum cares about this number; the model is told it explicitly.
        #   * ``total_token_budget`` — hard cap on tokens generated this
        #     turn (thinking + post combined).  This is what the trainer
        #     enforces via ``action_token_budget``.  When thinking is
        #     enabled the model is shown this number and learns to balance
        #     how many of the total tokens go to private reasoning vs the
        #     public post — the only constraint is post ≤ post_budget and
        #     total ≤ total_budget.
        #   * ``action_token_budget`` — legacy single-number alias.  When
        #     supplied (and the new fields are not), it is taken as the
        #     total cap, with the post cap defaulting to the same value.
        self._post_token_budget = (
            int(post_token_budget) if post_token_budget is not None
            else (int(action_token_budget) if action_token_budget is not None else 128)
        )
        if total_token_budget is not None:
            total = int(total_token_budget)
        elif action_token_budget is not None:
            total = int(action_token_budget)
        else:
            total = self._post_token_budget
        if total < self._post_token_budget:
            raise ValueError(
                f"total_token_budget ({total}) must be >= post_token_budget "
                f"({self._post_token_budget})."
            )
        self._total_token_budget = total
        self.action_token_budget = total
        self._max_posts = int(max_posts)
        if post_order not in ("round_robin", "random"):
            raise ValueError(
                f"post_order must be 'round_robin' or 'random', got {post_order!r}"
            )
        self._post_order = post_order
        self._seed = seed
        self._rng = random.Random(seed)
        self._reward_fn = reward_fn if reward_fn is not None else (lambda _s, _a: 0.0)

        # Tag-extraction format.  The env treats every generation as a
        # structured stream of three regions:
        #
        #   * ``<post>...</post>``      — the public utterance posted to the
        #                                  thread (extracted, joined, sent to
        #                                  every other agent).
        #   * ``<python>...</python>``  — code submitted to the pybot tool
        #                                  (extracted, executed, surfaced as
        #                                  a separate pybot post).  Only
        #                                  active when ``python_tool_enabled``.
        #   * everything else           — private thinking; never shown to
        #                                  other agents and never recorded
        #                                  on the canonical thread.
        #
        # The agent's own context keeps the raw action tokens (the trainer's
        # append-only path is unchanged); only the public output is filtered.
        self._post_open_tag = post_open_tag
        self._post_close_tag = post_close_tag
        # No-tag mode. With the post tags blank there is no private channel:
        # the whole generation is the public post, so nothing an agent emits
        # is discarded and p_env has no latent to marginalise (see
        # environment_likelihood_derivation.md, section 4.1a). The python tool
        # is unavailable in this mode — its blocks would have nowhere to hide.
        self._no_tag_mode = not (post_open_tag and post_close_tag)
        if self._no_tag_mode and python_tool_enabled:
            raise ValueError(
                "python_tool_enabled requires post tags: in no-tag mode the "
                "whole generation is the post, so a <python> block cannot be "
                "separated from it.")
        # Carry each delivered post's generating-agent token ids through to
        # observers, so the trainer can attach ln p_env per token without
        # re-encoding. Off by default: it changes how observations are
        # tokenised (segment-wise rather than as one string) and must not
        # alter runs that predate it.
        self._provenance = bool(provenance)
        # agent -> segments of its most recent observe(), each
        #   ("scaffold", None, ids) or ("post", speaker, ids, post_index)
        self._last_obs_segments: dict[str, list[tuple]] = {}
        # post_index -> {speaker, token_ids, src_positions}
        self._post_token_provenance: dict[int, dict] = {}
        # posts whose token ids could not be recovered exactly
        self._provenance_misses = 0

        # Optional ``pybot`` python-execution tool.  When enabled, any code
        # the agent puts inside ``<python>...</python>`` tags is stripped
        # from the public post (like thinking tokens) and executed in a
        # subprocess; the result is appended to the thread as a separate
        # post by ``python_bot_name`` (default "pybot") that contains the
        # original code plus the captured stdout/stderr, attributed to the
        # submitter.  The bot post is delivered to every agent — including
        # the submitter — so they can see the result on their next turn.
        # pybot posts are tool outputs and do NOT count against
        # ``max_posts`` (only agent turns do).
        self._python_tool_enabled = bool(python_tool_enabled)
        self._python_open_tag = python_open_tag
        self._python_close_tag = python_close_tag
        self._python_bot_name = python_bot_name
        self._python_timeout_seconds = float(python_timeout_seconds)
        self._python_output_max_chars = int(python_output_max_chars)
        self._python_use_forkserver = bool(python_use_forkserver)

        # Per-call scratch dir for pybot. Defaults to a fresh dir under TMPDIR.
        # Without this, code like ``plt.savefig("foo.png")`` lands wherever the
        # trainer was launched from (usually the repo root).
        self._pybot_scratch_root = (
            Path(pybot_scratch_root) if pybot_scratch_root is not None
            else Path(tempfile.gettempdir()) / f"pybot_scratch_{os.getpid()}"
        )
        self._pybot_scratch_root.mkdir(parents=True, exist_ok=True)
        # Optional archive: when set, any pybot call that produced files has
        # its scratch dir moved here, with ``code.py`` and ``meta.json``
        # alongside, keyed by ``ep{N}/post{P}_{speaker}_call{C}``.
        self._pybot_archive_dir = (
            Path(pybot_archive_dir) if pybot_archive_dir is not None else None
        )
        if self._pybot_archive_dir is not None:
            self._pybot_archive_dir.mkdir(parents=True, exist_ok=True)
        self._pybot_run_id = f"{os.getpid()}_{int(time.time())}"
        self._episode_idx = -1
        self._pybot_call_idx = 0

        # Optional sentence appended to the initial ctx so the model is told
        # up-front roughly how long a post is allowed to be — small instruct
        # models otherwise frequently exceed the per-turn token budget and
        # get cut off mid-sentence.  ``None`` → derive from action_token_budget.
        if post_length_note is None:
            post_words = max(1, int(self._post_token_budget * 0.7))
            post_length_note = (
                f"Posts on this forum are capped at about {self._post_token_budget} "
                f"tokens (roughly {post_words} words). Anything longer is cut off "
                f"mid-sentence, so keep each post within that budget — finish your "
                f"thought before you run out of room."
            )
        self._post_length_note = post_length_note

        # PettingZoo AEC state — populated in reset()
        self.agents: list[str] = []
        self.agent_selection: str = ""
        # Buffer of (speaker, text) tuples accumulated since this agent last acted.
        self._pending_posts: dict[str, list[tuple[str, str]]] = {}
        # Whether each agent has already received its initial ctx.
        self._initial_delivered: dict[str, bool] = {}
        self._post_count: int = 0
        self._thread: list[dict] = []
        self._cumulative_rewards: dict[str, float] = {}
        self._terminations: dict[str, bool] = {}
        self._truncations: dict[str, bool] = {}
        self._infos: dict[str, dict] = {}
        self._final_state: Any = None
        self._next_idx: int = 0
        # Per-episode tool-use counters. ``python_*`` outcomes are mutually
        # exclusive per call: every call lands in exactly one of success /
        # runtime_error / timeout / launch_error. ``python_unclosed`` counts
        # tagged blocks that were dropped (cut off mid-tag) and so never ran.
        self._tool_counts: dict[str, int] = {}
        self._tool_counts_per_agent: dict[str, dict[str, int]] = {}

    # ------------------------------------------------------------------ #
    # Trainer plumbing                                                    #
    # ------------------------------------------------------------------ #

    @property
    def default_character_prompts(self) -> dict[str, str]:
        """Personas the trainer should install as each agent's system prompt."""
        return dict(self._personas)

    # ------------------------------------------------------------------ #
    # Deep-copy support                                                   #
    # ------------------------------------------------------------------ #

    def __deepcopy__(self, memo: dict) -> "ForumEnv":
        new = ForumEnv(
            agent_names=list(self.possible_agents),
            agent_personas=dict(self._personas),
            tokenizer=self._tok,
            forum_description=self._forum_description,
            initial_invitation=self._initial_invitation,
            action_token_budget=self.action_token_budget,
            max_posts=self._max_posts,
            post_order=self._post_order,
            seed=self._seed,
            reward_fn=self._reward_fn,
            post_length_note=self._post_length_note,
            post_open_tag=self._post_open_tag,
            post_close_tag=self._post_close_tag,
            provenance=self._provenance,
            post_token_budget=self._post_token_budget,
            total_token_budget=self._total_token_budget,
            python_tool_enabled=self._python_tool_enabled,
            python_open_tag=self._python_open_tag,
            python_close_tag=self._python_close_tag,
            python_bot_name=self._python_bot_name,
            python_timeout_seconds=self._python_timeout_seconds,
            python_output_max_chars=self._python_output_max_chars,
            python_use_forkserver=self._python_use_forkserver,
            pybot_scratch_root=str(self._pybot_scratch_root),
            pybot_archive_dir=(
                str(self._pybot_archive_dir) if self._pybot_archive_dir else None
            ),
        )
        memo[id(self)] = new
        return new

    # ------------------------------------------------------------------ #
    # PettingZoo AEC interface                                            #
    # ------------------------------------------------------------------ #

    def reset(self, seed: int | None = None, options: dict | None = None) -> None:
        if seed is not None:
            self._rng = random.Random(seed)

        # Per-episode roster. ``possible_agents`` is every agent that could ever
        # appear; ``agents`` is the set actually in this episode. Passing
        # ``active_agents`` runs the thread with a subset, which is how partner
        # count is varied independently of population size — see
        # PopulationTrainer's partner_graph_degree. Absent means the full
        # roster, so the default path is unchanged.
        active = (options or {}).get("active_agents")
        if active:
            unknown = [a for a in active if a not in self.possible_agents]
            if unknown:
                raise ValueError(
                    f"active_agents {unknown} are not in possible_agents "
                    f"{list(self.possible_agents)}"
                )
            if len(active) < 2:
                raise ValueError(
                    f"active_agents needs at least 2 agents, got {list(active)}"
                )
            # Keep possible_agents' order so turn order does not depend on the
            # order the subset happened to be built in.
            self.agents = [a for a in self.possible_agents if a in set(active)]
        else:
            self.agents = list(self.possible_agents)
        # Who took part in THIS episode, captured here for two reasons:
        # ``agents`` is drained as members terminate, so it cannot answer that
        # once the thread is over and the trace is read after the fact; and
        # taking it *before* the turn-order shuffle below keeps it in
        # possible_agents order, which is byte-for-byte what the trace and the
        # termination loop recorded when they read possible_agents directly.
        self._episode_agents = list(self.agents)
        if self._post_order == "round_robin":
            # Shuffle the round-robin permutation so the same agent doesn't
            # always open. The shuffle is keyed off ``order_seed`` (passed via
            # options), separate from the main ``seed``, so the trainer can
            # share one order across all parallel episodes in an iteration —
            # keeping turn schedules aligned so vLLM can batch generations.
            order_seed = (options or {}).get("order_seed")
            order_rng = random.Random(order_seed) if order_seed is not None else self._rng
            order_rng.shuffle(self.agents)
        self._pending_posts = {a: [] for a in self.agents}
        self._last_obs_segments = {a: [] for a in self.agents}
        self._post_token_provenance = {}
        self._provenance_misses = 0
        self._initial_delivered = {a: False for a in self.agents}
        self._post_count = 0
        self._thread = []
        self._episode_idx += 1
        self._pybot_call_idx = 0
        self._tool_counts = {
            "python_calls": 0,
            "python_success": 0,
            "python_runtime_error": 0,
            "python_timeout": 0,
            "python_launch_error": 0,
            "python_unclosed": 0,
            # Per-turn post-tag outcomes (one bucket per agent turn).
            # Mutually exclusive: every turn lands in exactly one of
            # post_success / post_missing / post_unclosed / post_empty.
            # ``post_blocks`` counts the total number of closed <post>
            # blocks emitted across the episode (so multi-block turns
            # show up).
            "post_turns": 0,
            "post_success": 0,
            "post_missing": 0,
            "post_unclosed": 0,
            "post_empty": 0,
            "post_blocks": 0,
        }
        self._tool_counts_per_agent = {
            a: dict.fromkeys(self._tool_counts, 0) for a in self.agents
        }
        # Per-episode diagnostic event log: a chronological list of
        # noteworthy things that happened (parser anomalies, pybot errors,
        # truncations, etc.) with enough context to debug a broken run.
        # Surfaced in episode_trace() so the trace viewer can render them.
        self._events: list[dict] = []
        self._cumulative_rewards = {a: 0.0 for a in self.agents}
        self._terminations = {a: False for a in self.agents}
        self._truncations = {a: False for a in self.agents}
        self._infos = {a: {} for a in self.agents}
        self._final_state = None

        if self._post_order == "round_robin":
            self._next_idx = 0
            self.agent_selection = self.agents[self._next_idx]
        else:
            self.agent_selection = self._rng.choice(self.agents)

    def observe(self, agent: str) -> list[int]:
        """Return token ids for the new content this agent has not yet seen.

        First-time observers receive the initial ctx (forum description +
        invitation + any posts that landed before their first turn).
        Subsequent observers receive only the partner posts that have
        accumulated since their last act.
        """
        if agent not in self._pending_posts:
            return []

        body_parts: list[str] = []
        if not self._initial_delivered[agent]:
            body_parts.append(self._render_initial_ctx())
            self._initial_delivered[agent] = True

        posts = self._pending_posts[agent]
        self._pending_posts[agent] = []
        if posts:
            body_parts.append(self._frame_posts(posts))

        if not body_parts:
            return []
        if self._provenance:
            return self._observe_with_provenance(agent, body_parts, posts)
        text = "\n\n".join(body_parts)
        ids = self._tok.encode(text, add_special_tokens=False)
        return [int(t) for t in ids]

    def _observe_with_provenance(self, agent, body_parts, posts) -> list[int]:
        """Tokenise the observation segment-wise so every token's origin is known.

        Scaffolding is encoded here (deterministic: ln p_env = 0). Another
        agent's post is spliced in as the exact ids that agent sampled, never
        re-encoded, so the observer's tokens ARE the generator's tokens and its
        stored log-prob attaches to each one without alignment.

        The cost is a segment-wise rather than canonical tokenisation: BPE will
        not merge across a segment boundary. That is the point — merging is
        precisely what would mix a scaffold token with a post token.

        ``self._last_obs_segments[agent]`` records, per segment,
        ``(kind, speaker, n_tokens, post_index)`` for the trainer to consume.
        """
        enc = lambda t: [int(x) for x in self._tok.encode(t, add_special_tokens=False)]
        segs: list[tuple] = []
        # body_parts were joined with "\n\n"; rebuild that structure exactly.
        n_before_posts = len(body_parts) - (1 if posts else 0)
        for n in range(n_before_posts):
            if n:
                segs.append(("scaffold", None, enc("\n\n"), None))
            segs.append(("scaffold", None, enc(body_parts[n]), None))
        if posts:
            if n_before_posts:
                segs.append(("scaffold", None, enc("\n\n"), None))
            for kind, sp, txt, idx, ids in self._frame_post_segments(posts):
                if kind == "post" and ids:
                    segs.append(("post", sp, [int(x) for x in ids], idx))
                elif kind == "post":
                    # No recoverable ids (e.g. a tool post, or a text/token
                    # mismatch). Encode it, and mark it unmeasured rather than
                    # asserting ln p_env = 0 for something that was sampled.
                    segs.append(("post_unknown", sp, enc(txt), idx))
                else:
                    segs.append(("scaffold", None, enc(txt), None))
        self._last_obs_segments[agent] = [
            (k, sp, len(i), idx) for k, sp, i, idx in segs
        ]
        out: list[int] = []
        for _k, _sp, i, _idx in segs:
            out.extend(i)
        return out

    def observation_segments(self, agent: str) -> list[tuple]:
        """``(kind, speaker, n_tokens, post_index)`` for the agent's last
        observation. ``kind`` is "scaffold" (ln p_env = 0), "post" (the
        generating agent's log-probs apply) or "post_unknown"."""
        return list(self._last_obs_segments.get(agent, []))

    def step(self, action: Any) -> None:
        speaker = self.agent_selection
        if self._terminations.get(speaker) or self._truncations.get(speaker):
            self._was_dead_step(action)
            return

        raw = self._decode_action(action)

        # Tag extraction: pull <post> and <python> blocks out of the
        # generation; everything between them is private thinking and
        # is discarded for the public stream.
        text, code_blocks, post_unclosed, unclosed, n_post_blocks = (
            self._extract_tagged(raw)
        )
        self._record_tool_event(speaker, "post_turns")
        for _ in range(n_post_blocks):
            self._record_tool_event(speaker, "post_blocks")
        if post_unclosed:
            self._record_tool_event(speaker, "post_unclosed")
            self._log_event(
                kind="post_unclosed",
                severity="warn",
                speaker=speaker,
                message=(
                    f"{speaker} emitted an unclosed <post> block "
                    f"(likely cut off by the post token budget)."
                ),
                raw_tail=raw[-200:],
            )
        elif n_post_blocks == 0:
            self._record_tool_event(speaker, "post_missing")
            self._log_event(
                kind="post_missing",
                severity="warn",
                speaker=speaker,
                message=(
                    f"{speaker} produced no <post> block — turn was "
                    f"all thinking / no public output."
                ),
                raw_len=len(raw),
            )
        elif not text:
            self._record_tool_event(speaker, "post_empty")
            self._log_event(
                kind="post_empty",
                severity="warn",
                speaker=speaker,
                message=f"{speaker} emitted an empty <post> block.",
            )
        else:
            self._record_tool_event(speaker, "post_success")

        # Record on canonical thread.
        self._thread.append({
            "post_index": self._post_count,
            "speaker": speaker,
            "text": text,
        })
        self._post_count += 1

        # Provenance: keep the generating agent's token ids for this post, and
        # the positions they occupied in its action sequence, so the trainer
        # can attach that agent's own log-prob to each token when the post
        # lands in someone else's context (environment_likelihood_derivation.md).
        post_idx = self._post_count - 1
        content_ids = None
        if self._provenance and text:
            content_ids, src_pos = self._content_token_ids(action, text)
            if content_ids is not None:
                self._post_token_provenance[post_idx] = {
                    "speaker": speaker,
                    "token_ids": content_ids,
                    "src_positions": src_pos,
                }
            else:
                self._provenance_misses += 1

        # Deliver this post to every OTHER agent's pending buffer.
        for other in self.agents:
            if other != speaker:
                self._pending_posts[other].append(
                    (speaker, text, content_ids, post_idx))

        # Run any python blocks the speaker submitted and emit a pybot
        # post per block.  pybot posts are tool outputs, not turns, so
        # they do not advance ``_post_count`` / ``max_posts``.
        submitter_post_idx = self._post_count - 1
        for code in code_blocks:
            output, status = self._run_python(code, speaker, submitter_post_idx)
            self._record_tool_event(speaker, "python_calls")
            self._record_tool_event(speaker, f"python_{status}")
            if status != "success":
                self._log_event(
                    kind=f"python_{status}",
                    severity="error" if status in ("launch_error",) else "warn",
                    speaker=speaker,
                    message=(
                        f"{speaker} pybot call #{self._pybot_call_idx} "
                        f"finished with status={status}."
                    ),
                    code=code[:500],
                    output=output[:500],
                )
            bot_text = self._format_pybot_post(speaker, code, output, status)
            self._thread.append({
                "post_index": None,
                "speaker": self._python_bot_name,
                "text": bot_text,
                "kind": "tool_output",
                "submitter": speaker,
                "status": status,
            })
            for a in self.agents:
                self._pending_posts[a].append((self._python_bot_name, bot_text))

        # Surface an unclosed ``<python>`` tag as a visible pybot failure
        # rather than silently dropping the trailing code.  Without this,
        # an agent who ran out of token budget mid-block sees no reply
        # and has no signal that the tool didn't fire.
        if unclosed:
            self._record_tool_event(speaker, "python_unclosed")
            self._log_event(
                kind="python_unclosed",
                severity="warn",
                speaker=speaker,
                message=(
                    f"{speaker} emitted an unclosed "
                    f"{self._python_open_tag} block — code dropped."
                ),
            )
            err_text = (
                f"{self._python_bot_name}: {speaker}'s post contained an "
                f"unclosed {self._python_open_tag} block — no code was "
                f"executed (likely cut off by the post token budget)."
            )
            self._thread.append({
                "post_index": None,
                "speaker": self._python_bot_name,
                "text": err_text,
                "kind": "tool_output",
                "submitter": speaker,
                "status": "unclosed",
            })
            for a in self.agents:
                self._pending_posts[a].append((self._python_bot_name, err_text))

        # Termination check.
        if self._post_count >= self._max_posts:
            self._final_state = {"thread": list(self._thread)}
            tool_info = self._build_tool_info()
            for a in self._episode_agents:
                self._terminations[a] = True
                self._cumulative_rewards[a] = self._reward_fn(self._final_state, a)
                self._infos[a] = {**self._infos.get(a, {}), **tool_info}
            return

        # Advance.
        self.agent_selection = self._select_next_speaker(prev=speaker)

    # ------------------------------------------------------------------ #
    # Internals                                                           #
    # ------------------------------------------------------------------ #

    def _content_token_ids(self, action, text: str):
        """Ids within ``action`` that decode to exactly ``text``, and their
        positions. Returns ``(None, None)`` if that cannot be established, so
        the caller can fall back to re-encoding and mark those tokens as
        having no measured ``ln p_env`` rather than guessing.

        ``_decode_action`` drops special tokens, so the delivered text is the
        decode of the non-special subsequence; this recovers that subsequence
        and verifies the round trip before returning it.
        """
        try:
            ids = [int(t) for t in action]
        except (TypeError, ValueError):
            return None, None
        specials = set(getattr(self._tok, "all_special_ids", []) or [])
        keep = [(n, t) for n, t in enumerate(ids) if t not in specials]
        cand = [t for _, t in keep]
        if not cand:
            return None, None
        try:
            if self._tok.decode(cand, skip_special_tokens=True) != text:
                return None, None
        except Exception:
            return None, None
        return cand, [n for n, _ in keep]

    def _decode_action(self, action: Any) -> str:
        if action is None:
            self._log_event(
                kind="action_none",
                severity="warn",
                speaker=self.agent_selection,
                message="Agent submitted None as its action.",
            )
            return ""
        try:
            ids = list(action)
        except TypeError:
            return str(action)
        if not ids:
            self._log_event(
                kind="action_empty",
                severity="warn",
                speaker=self.agent_selection,
                message="Agent submitted an empty token list.",
            )
            return ""
        return self._tok.decode(ids, skip_special_tokens=True)

    # Native-thinking container. The model wraps private reasoning in
    # <think>...</think> when enable_thinking is on. Any <post>/<python>
    # text *inside* a think block is just thinking content — it must not
    # trigger a post or pybot call. We hardcode the literal tags here
    # because they're a Qwen3-template invariant; if a future model uses
    # a different reasoning tag, surface this as a constructor arg.
    _THINK_OPEN  = "<think>"
    _THINK_CLOSE = "</think>"

    def _extract_tagged(
        self, text: str
    ) -> tuple[str, list[str], bool, bool, int]:
        """Partition ``text`` into post / python / thinking regions.

        Only **outermost** tags are functional. The parser walks the
        string left-to-right at depth 0 and, at each step, picks the
        earliest open tag among ``<think>``, ``<post>``, ``<python>``.
        Whichever it finds, the parser jumps to that tag's matching close
        (or end-of-string if unclosed) without rescanning the interior
        for nested tags. So a ``<post>`` written inside a ``<think>``
        block is just text and does not accidentally start a post; a
        ``<python>`` inside a ``<post>`` is part of the post's content;
        a stray nested ``<post>`` inside an already-open ``<post>`` is
        absorbed by the first ``</post>`` (existing behaviour).

        Returns ``(public_post, code_blocks, post_unclosed, python_unclosed,
        n_post_blocks)``:

        * ``public_post`` — closed ``<post>`` block contents joined with
          a blank line, plus any trailing partial-post content if the
          last open ``<post>`` was unclosed (best-effort surface so a
          turn that ran out of budget mid-post still shows what it
          managed to write).
        * ``code_blocks`` — closed ``<python>`` blocks, in order.
          Unclosed python blocks are dropped (we will not run partial
          code) but counted via ``python_unclosed=True``.
        """
        if self._no_tag_mode:
            # Everything the agent generated is the post. One "closed" block,
            # nothing unclosed, no code. Keeps every downstream counter
            # (post_success / post_missing / post_blocks) meaningful.
            body = text.strip()
            return body, [], False, False, (1 if body else 0)
        po, pc = self._post_open_tag, self._post_close_tag
        yo, yc = self._python_open_tag, self._python_close_tag
        to, tc = self._THINK_OPEN, self._THINK_CLOSE
        py_active = self._python_tool_enabled
        posts: list[str] = []
        codes: list[str] = []
        post_unclosed = False
        python_unclosed = False
        n_closed_posts = 0
        i = 0
        while i < len(text):
            jp = text.find(po, i) if po else -1
            jy = text.find(yo, i) if (yo and py_active) else -1
            jt = text.find(to, i) if to else -1
            # Pick the earliest open tag at depth 0. -1 means "not found"
            # — sort by (found?, position) so missing tags never win.
            candidates = [
                (jp, "post"),
                (jy, "python"),
                (jt, "think"),
            ]
            candidates = [(pos, kind) for pos, kind in candidates if pos >= 0]
            if not candidates:
                break
            pos, kind = min(candidates, key=lambda pk: pk[0])
            if kind == "post":
                k = text.find(pc, pos + len(po))
                if k < 0:
                    post_unclosed = True
                    posts.append(text[pos + len(po):])
                    break
                posts.append(text[pos + len(po):k])
                n_closed_posts += 1
                i = k + len(pc)
            elif kind == "python":
                k = text.find(yc, pos + len(yo))
                if k < 0:
                    python_unclosed = True
                    break
                codes.append(text[pos + len(yo):k])
                i = k + len(yc)
            else:  # think — skip the whole region without rescanning
                k = text.find(tc, pos + len(to))
                if k < 0:
                    # Truncated mid-think (budget cut off, or model never
                    # closed). Discard the tail; do not look for nested
                    # post/python inside it.
                    break
                i = k + len(tc)
        public = "\n\n".join(p.strip() for p in posts).strip()
        return public, codes, post_unclosed, python_unclosed, n_closed_posts

    @staticmethod
    def _autoprint_rewrite(code: str) -> str:
        """If the last top-level statement is a bare expression, wrap it
        so its value is printed (Jupyter / REPL convention).

        Leaves the code unchanged if it doesn't parse, has no body,
        ends in a non-expression statement, or the trailing expression
        is a docstring / None constant / existing ``print(...)`` call.
        Adds no output for ``None``-valued expressions, matching the
        Jupyter displayhook.
        """
        try:
            tree = ast.parse(code, mode="exec")
        except SyntaxError:
            return code  # let the subprocess surface the SyntaxError
        if not tree.body:
            return code
        last = tree.body[-1]
        if not isinstance(last, ast.Expr):
            return code
        val = last.value
        # Skip trailing docstrings / bare None / plain string literals.
        if isinstance(val, ast.Constant) and (
            val.value is None or isinstance(val.value, (str, bytes))
        ):
            return code
        # Skip if it is already ``print(...)``.
        if (
            isinstance(val, ast.Call)
            and isinstance(val.func, ast.Name)
            and val.func.id == "print"
        ):
            return code
        # Replace ``<expr>`` with ``__pybot_r__ = <expr>; if __pybot_r__ is
        # not None: print(repr(__pybot_r__))``.
        assign = ast.Assign(
            targets=[ast.Name(id="__pybot_r__", ctx=ast.Store())],
            value=val,
        )
        guarded_print = ast.parse(
            "if __pybot_r__ is not None:\n    print(repr(__pybot_r__))"
        ).body[0]
        new_body = list(tree.body[:-1]) + [assign, guarded_print]
        new_tree = ast.Module(body=new_body, type_ignores=[])
        ast.fix_missing_locations(new_tree)
        try:
            return ast.unparse(new_tree)
        except Exception:  # noqa: BLE001
            return code

    def _run_python(
        self,
        code: str,
        submitter: str = "?",
        submitter_post_idx: int = -1,
    ) -> tuple[str, str]:
        """Execute one code block in isolation and return formatted output.

        Uses a forkserver-backed worker (fast: ~5–15 ms launch) when
        ``python_use_forkserver`` is set; falls back to a cold
        ``subprocess.run`` of ``python -I -c`` otherwise.  Both paths
        are sandboxed in a fresh process (no state leakage between
        calls) and honour ``python_timeout_seconds``.

        Each call runs in a fresh per-call scratch dir under
        ``pybot_scratch_root``; the dir is deleted afterwards unless
        ``pybot_archive_dir`` is set, in which case any non-empty scratch
        is moved into the archive alongside ``code.py`` and ``meta.json``.
        """
        run_code = self._autoprint_rewrite(code)
        call_idx = self._pybot_call_idx
        self._pybot_call_idx += 1
        scratch = Path(tempfile.mkdtemp(
            prefix=f"ep{self._episode_idx}_post{submitter_post_idx}_call{call_idx}_",
            dir=str(self._pybot_scratch_root),
        ))
        env = _pybot_minimal_env(str(scratch))
        try:
            if self._python_use_forkserver:
                stdout, stderr, rc = self._run_python_forkserver(
                    run_code, str(scratch), env
                )
            else:
                stdout, stderr, rc = self._run_python_subprocess(
                    run_code, str(scratch), env
                )
        finally:
            self._archive_or_clean_scratch(
                scratch, code, submitter, submitter_post_idx, call_idx
            )

        if rc == "timeout":
            return (
                f"[pybot: timed out after {self._python_timeout_seconds:g}s]",
                "timeout",
            )
        if rc == "error":
            return f"[pybot: execution error: {stderr}]", "launch_error"

        parts: list[str] = []
        if stdout:
            parts.append(stdout.rstrip("\n"))
        if stderr:
            parts.append("[stderr]\n" + stderr.rstrip("\n"))
        if isinstance(rc, int) and rc != 0 and not stderr:
            parts.append(f"[exit code {rc}]")
        result = "\n".join(parts) if parts else "(no output)"
        if len(result) > self._python_output_max_chars:
            result = result[: self._python_output_max_chars] + "\n[... truncated]"
        status = "success" if (isinstance(rc, int) and rc == 0) else "runtime_error"
        return result, status

    def _run_python_forkserver(
        self, code: str, cwd: str, env: dict[str, str]
    ) -> tuple[str, str, Any]:
        ctx = _get_pyexec_ctx()
        parent_conn, child_conn = ctx.Pipe(duplex=False)
        try:
            proc = ctx.Process(
                target=_pyexec_worker, args=(code, child_conn, cwd, env)
            )
            proc.start()
            child_conn.close()  # parent only reads
            proc.join(self._python_timeout_seconds)
            if proc.is_alive():
                proc.terminate()
                proc.join(0.5)
                if proc.is_alive():
                    proc.kill()
                    proc.join(0.5)
                return "", "", "timeout"
            stdout, stderr, rc = "", "", -1
            try:
                if parent_conn.poll(0.5):
                    stdout, stderr, rc = parent_conn.recv()
            except (EOFError, OSError):
                pass
            return stdout, stderr, rc
        except Exception as e:  # noqa: BLE001
            return "", str(e), "error"
        finally:
            try:
                parent_conn.close()
            except Exception:  # noqa: BLE001
                pass

    def _run_python_subprocess(
        self, code: str, cwd: str, env: dict[str, str]
    ) -> tuple[str, str, Any]:
        def _preexec() -> None:  # runs in the child after fork, before exec
            try:
                import resource

                for name, lim in _PYBOT_RLIMITS.items():
                    r = getattr(resource, f"RLIMIT_{name}", None)
                    if r is None:
                        continue
                    try:
                        soft, hard = resource.getrlimit(r)
                        new_hard = (
                            lim if hard == resource.RLIM_INFINITY else min(lim, hard)
                        )
                        resource.setrlimit(r, (min(lim, new_hard), new_hard))
                    except (ValueError, OSError):
                        pass
            except ImportError:
                pass

        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-c", code],
                capture_output=True,
                text=True,
                timeout=self._python_timeout_seconds,
                cwd=cwd,
                env=env,
                preexec_fn=_preexec,
            )
        except subprocess.TimeoutExpired:
            return "", "", "timeout"
        except Exception as e:  # noqa: BLE001
            return "", str(e), "error"
        return proc.stdout, proc.stderr, proc.returncode

    def _archive_or_clean_scratch(
        self,
        scratch: Path,
        code: str,
        submitter: str,
        submitter_post_idx: int,
        call_idx: int,
    ) -> None:
        try:
            entries = list(scratch.iterdir())
        except OSError:
            entries = []

        if not entries or self._pybot_archive_dir is None:
            shutil.rmtree(scratch, ignore_errors=True)
            return

        safe_speaker = "".join(
            c if c.isalnum() or c in "-_" else "_" for c in submitter
        )
        dest = self._pybot_archive_dir / (
            f"{self._pybot_run_id}/ep{self._episode_idx}/"
            f"post{submitter_post_idx}_{safe_speaker}_call{call_idx}"
        )
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.move(str(scratch), str(dest))
        except OSError:
            shutil.rmtree(scratch, ignore_errors=True)
            return

        try:
            (dest / "code.py").write_text(code)
            meta = {
                "run_id": self._pybot_run_id,
                "pid": os.getpid(),
                "wall_time": time.time(),
                "episode_idx": self._episode_idx,
                "submitter": submitter,
                "submitter_post_idx": submitter_post_idx,
                "call_idx": call_idx,
            }
            (dest / "meta.json").write_text(json.dumps(meta, indent=2))
        except OSError:
            pass

    def _format_pybot_post(
        self, submitter: str, code: str, output: str, status: str
    ) -> str:
        status_label = {
            "success": "success",
            "runtime_error": "failed (runtime error)",
            "timeout": "failed (timeout)",
            "launch_error": "failed (launch error)",
        }.get(status, status)
        return (
            f"Running code submitted by {submitter} [{status_label}]:\n"
            f"{self._python_open_tag}\n{code}\n{self._python_close_tag}\n"
            f"Output:\n{output}"
        )

    def _record_tool_event(self, speaker: str, key: str) -> None:
        if key in self._tool_counts:
            self._tool_counts[key] += 1
        per_agent = self._tool_counts_per_agent.setdefault(
            speaker, dict.fromkeys(self._tool_counts, 0)
        )
        per_agent[key] = per_agent.get(key, 0) + 1

    def _log_event(
        self,
        kind: str,
        severity: str,
        message: str,
        speaker: str | None = None,
        **extra: Any,
    ) -> None:
        ev = {
            "post_index": self._post_count,
            "speaker": speaker,
            "kind": kind,
            "severity": severity,
            "message": message,
        }
        if extra:
            ev.update(extra)
        self._events.append(ev)

    def _render_initial_ctx(self) -> str:
        """First-observation framing each agent sees once."""
        parts = [self._forum_description, self._initial_invitation]
        if self._post_length_note:
            parts.append(self._post_length_note)
        post_words = max(1, int(self._post_token_budget * 0.7))
        total_words = max(1, int(self._total_token_budget * 0.7))
        po, pc = self._post_open_tag, self._post_close_tag
        yo, yc = self._python_open_tag, self._python_close_tag
        tool_line = ""
        if self._python_tool_enabled:
            tool_line = (
                f" You can also call the python tool by writing code "
                f"inside {yo}...{yc}; the code runs and the result is "
                f"shared with everyone as a separate post by "
                f"'{self._python_bot_name}'. Each {yo} block runs in a "
                f"fresh interpreter — nothing carries over between calls, "
                f"so re-do your imports and re-define any functions or "
                f"variables you need every time. "
            )
        if self._no_tag_mode:
            # No private channel: the whole turn is the post. Nothing is
            # discarded, so the prompt must not promise a place to think.
            parts.append(
                f"Each turn, write your reply. Everything you write is the "
                f"post other forum members see — there is no private space "
                f"and nothing is hidden. Keep it under about "
                f"{self._total_token_budget} tokens (~{total_words} words); "
                f"anything past that is cut off mid-sentence."
            )
        else:
            parts.append(
                f"Each turn, write your reply inside {po}...{pc} tags — "
                f"that is the text other forum members see.{tool_line}"
                f"Anything outside {po} and {yo} tags is private "
                f"reasoning: it stays in your own notes and is never shown "
                f"to anyone else. If it helps, jot a quick note to yourself "
                f"before the {po} — work through a case, check a step, or "
                f"think about what someone said — then write the post. "
                f"Two limits: the {po} content itself must stay under about "
                f"{self._post_token_budget} tokens (~{post_words} words), "
                f"and the total of thinking + post + tool calls combined "
                f"must stay under about {self._total_token_budget} tokens "
                f"(~{total_words} words) — anything past the total is cut "
                f"off mid-sentence. Always close {pc} before you stop writing."
            )
        return "\n\n".join(p for p in parts if p)

    def _frame_post_segments(self, posts: list[tuple]) -> list[tuple]:
        """Segment the framed partner posts into scaffolding and post bodies.

        Returns ``[(kind, speaker, text, post_index, token_ids), ...]`` where
        ``kind`` is ``"scaffold"`` — deterministic framing this environment
        wrote, so ``ln p_env = 0`` — or ``"post"``, a token sequence another
        agent sampled, whose ``ln p_env`` is that agent's own log-prob.

        ``"".join(seg_text)`` reproduces :meth:`_frame_posts` exactly. The
        string form is derived from this one so the two cannot drift.
        """
        def unpack(q):
            sp, txt = q[0], q[1]
            ids = q[2] if len(q) > 2 else None
            idx = q[3] if len(q) > 3 else None
            return sp, txt, ids, idx

        segs: list[tuple] = []
        if len(posts) == 1:
            sp, txt, ids, idx = unpack(posts[0])
            if txt:
                segs.append(("scaffold", None, f"{sp} wrote:\n", None, None))
                segs.append(("post", sp, txt, idx, ids))
            else:
                segs.append(("scaffold", None, f"{sp} wrote nothing.", None, None))
            return segs
        segs.append(("scaffold", None, "Several people posted on the forum:\n\n", None, None))
        for n, q in enumerate(posts):
            sp, txt, ids, idx = unpack(q)
            if n:
                segs.append(("scaffold", None, "\n\n", None, None))
            if txt:
                segs.append(("scaffold", None, f"{sp} wrote:\n", None, None))
                segs.append(("post", sp, txt, idx, ids))
            else:
                segs.append(("scaffold", None, f"{sp} wrote nothing.", None, None))
        return segs

    def _frame_posts(self, posts: list[tuple]) -> str:
        """Render new partner posts as direct-speech narrative.

        Derived from :meth:`_frame_post_segments` so the segmented and string
        forms cannot disagree.
        """
        return "".join(seg[2] for seg in self._frame_post_segments(posts))
    def _select_next_speaker(self, prev: str) -> str:
        if self._post_order == "round_robin":
            self._next_idx = (self._next_idx + 1) % len(self.agents)
            return self.agents[self._next_idx]
        choices = [a for a in self.agents if a != prev] or list(self.agents)
        return self._rng.choice(choices)

    def _was_dead_step(self, action: Any) -> None:
        if action is not None:
            raise ValueError("Only None is valid for a terminated/truncated agent.")
        agent = self.agent_selection
        if agent in self.agents:
            self.agents.remove(agent)
        self._cumulative_rewards[agent] = 0.0
        if self.agents:
            self.agent_selection = self.agents[0]

    # ------------------------------------------------------------------ #
    # Trace                                                               #
    # ------------------------------------------------------------------ #

    def episode_trace(self) -> dict:
        return {
            "thread": list(self._thread),
            "agents": list(self._episode_agents),
            "post_order": self._post_order,
            "max_posts": self._max_posts,
            "completed_posts": self._post_count,
            "tool_counts": dict(self._tool_counts),
            "tool_counts_per_agent": {
                a: dict(c) for a, c in self._tool_counts_per_agent.items()
            },
            "events": list(self._events),
        }

    def _build_tool_info(self) -> dict:
        """Tool-use counters destined for ``infos`` (and thus ``metrics.jsonl``).

        Episode-totals are emitted with a ``tool/`` prefix so the trainer's
        metrics aggregator can forward them by key match without knowing the
        specific names.
        """
        out: dict[str, int] = {}
        for k, v in self._tool_counts.items():
            out[f"tool/{k}"] = int(v)
        return out

    # ------------------------------------------------------------------ #
    # PettingZoo required properties                                      #
    # ------------------------------------------------------------------ #

    @property
    def terminations(self) -> dict[str, bool]:
        return dict(self._terminations)

    @property
    def truncations(self) -> dict[str, bool]:
        return dict(self._truncations)

    @property
    def rewards(self) -> dict[str, float]:
        return dict(self._cumulative_rewards)

    @property
    def infos(self) -> dict[str, dict]:
        return dict(self._infos)

    def observation_space(self, agent: str) -> None:
        return None

    def action_space(self, agent: str) -> None:
        return None

    def render(self) -> None:
        pass

    def close(self) -> None:
        pass


