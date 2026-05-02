"""
Append-only forum environment for CCSM training.

Honours the embedded/embodied-agent invariant
---------------------------------------------
Each agent's context is a strictly monotonic chat-message log:

    [system: <persona>]
    [user:   <forum preamble + initial topic>]
    [assistant: <my post>]                        ← own posts as assistant
    [user:   <other speakers' posts so far>]      ← others' posts as user
    [assistant: <my next post>]
    [user:   ...]
    ...

Two invariants:
  1. Each turn appends exactly one new message somewhere in some agent's
     log.  Nothing is ever rewritten or removed.
  2. Speaker-role correctness — own posts stay tagged ``assistant``,
     others' posts stay tagged ``user``.  CCSM's perception/action loss
     tagging works unchanged.

No retrieval, no windowing, no GM-LLM, no summarisation.  The forum
"thread" lives implicitly inside each agent's monotonic message log.

Memory tool use (NOT implemented here)
---------------------------------------
A future ``recall`` tool would be a learned action: the agent emits a
structured tool-call within its generated tokens, the env intercepts
it, executes a retrieval, and **appends** the retrieved text as a new
``user`` message in the agent's log.  Retrieval becomes an action
within the stream, never a preprocessor over the stream.

PettingZoo AEC contract
-----------------------
Standard AEC.  ``observe(agent)`` returns the agent's full chat-template-
tokenised message log + generation prompt as a list of token ids.  Sets
``obs_is_full_context = True`` so the trainer treats the observation as
the entire context window.
"""
from __future__ import annotations

import random
from typing import Any, Callable

from pettingzoo import AECEnv
from transformers import PreTrainedTokenizerBase


# ---------------------------------------------------------------------------
# ForumEnv
# ---------------------------------------------------------------------------


class ForumEnv(AECEnv):
    """Append-only multi-agent forum environment.

    Parameters
    ----------
    agent_names:
        Ordered list of agent slot names.  These appear as posters in the
        thread and key the per-agent message logs.
    agent_personas:
        Map ``name → persona text``.  Inserted as the ``system`` message
        in each agent's log at reset.  Unmentioned agents fall back to a
        minimal placeholder persona.
    tokenizer:
        HuggingFace tokenizer used for chat-template assembly + decoding
        actions.
    forum_preamble:
        Initial ``user`` message inserted at reset, framing the thread
        topic.  This is identical for every agent.
    action_token_budget:
        Tokens per turn.  Read by the Trainer — does not constrain the
        env directly.
    max_posts:
        Total number of posts in the episode (across all agents).
        Episode terminates when this is reached.
    post_order:
        ``"round_robin"`` (cycle agents in given order) or ``"random"``
        (uniform sample without immediate self-repeat).
    seed:
        RNG seed for ``random`` post ordering.
    reward_fn:
        ``(final_state, agent_name) → float``.  Defaults to zero (CCSM
        relies on the perception loss, not environment reward).
    """

    metadata = {"render_modes": [], "name": "forum_v1"}

    obs_is_full_context: bool = True

    def __init__(
        self,
        agent_names: list[str],
        agent_personas: dict[str, str],
        tokenizer: PreTrainedTokenizerBase,
        forum_preamble: str = (
            "A new thread has just opened on the forum.\n"
            "Topic: General discussion. Anyone may start.\n"
            "Compose your next post."
        ),
        action_token_budget: int = 128,
        max_posts: int = 12,
        post_order: str = "round_robin",
        seed: int | None = None,
        reward_fn: Callable[[Any, str], float] | None = None,
    ) -> None:
        super().__init__()

        self.possible_agents = list(agent_names)
        self._personas = dict(agent_personas)
        self._tok = tokenizer
        self._forum_preamble = forum_preamble
        self.action_token_budget = action_token_budget
        self._max_posts = int(max_posts)
        if post_order not in ("round_robin", "random"):
            raise ValueError(f"post_order must be 'round_robin' or 'random', got {post_order!r}")
        self._post_order = post_order
        self._seed = seed
        self._rng = random.Random(seed)
        self._reward_fn = reward_fn if reward_fn is not None else (lambda _s, _a: 0.0)

        # PettingZoo AEC state (populated in reset)
        self.agents: list[str] = []
        self.agent_selection: str = ""
        self._messages: dict[str, list[dict]] = {}
        # Buffer of "speaker: text" strings to flush into the next observer's log.
        self._pending_obs: dict[str, list[str]] = {}
        self._post_count: int = 0
        self._thread: list[dict] = []   # canonical thread record for env_trace
        self._cumulative_rewards: dict[str, float] = {}
        self._terminations: dict[str, bool] = {}
        self._truncations: dict[str, bool] = {}
        self._infos: dict[str, dict] = {}
        self._final_state: Any = None

    # ------------------------------------------------------------------ #
    # Deep-copy support                                                   #
    # ------------------------------------------------------------------ #

    def __deepcopy__(self, memo: dict) -> "ForumEnv":
        """Config-only copy; runtime state is rebuilt in reset()."""
        new = ForumEnv(
            agent_names=list(self.possible_agents),
            agent_personas=dict(self._personas),
            tokenizer=self._tok,
            forum_preamble=self._forum_preamble,
            action_token_budget=self.action_token_budget,
            max_posts=self._max_posts,
            post_order=self._post_order,
            seed=self._seed,
            reward_fn=self._reward_fn,
        )
        memo[id(self)] = new
        return new

    # ------------------------------------------------------------------ #
    # PettingZoo AEC interface                                            #
    # ------------------------------------------------------------------ #

    def reset(self, seed: int | None = None, options: dict | None = None) -> None:
        if seed is not None:
            self._rng = random.Random(seed)

        self.agents = list(self.possible_agents)
        self._messages = {
            name: [
                {"role": "system", "content": self._personas.get(
                    name,
                    f"You are {name}, a member of an online forum. "
                    "Write your next post.",
                )},
                {"role": "user", "content": self._forum_preamble},
            ]
            for name in self.agents
        }
        self._pending_obs = {a: [] for a in self.agents}
        self._post_count = 0
        self._thread = []
        self._cumulative_rewards = {a: 0.0 for a in self.agents}
        self._terminations = {a: False for a in self.agents}
        self._truncations = {a: False for a in self.agents}
        self._infos = {a: {} for a in self.agents}
        self._final_state = None

        # First poster
        if self._post_order == "round_robin":
            self._next_idx = 0
            self.agent_selection = self.agents[self._next_idx]
        else:  # random
            self.agent_selection = self._rng.choice(self.agents)

    def observe(self, agent: str) -> list[int]:
        """Tokenised chat-template render of *agent*'s full monotonic log.

        Renders to a string via ``apply_chat_template(tokenize=False)`` and
        then encodes — bypasses transformers-version inconsistencies where
        ``tokenize=True`` may return a ``BatchEncoding`` (dict) rather than
        a flat ``list[int]``.
        """
        msgs = self._messages.get(agent)
        if not msgs:
            return []
        try:
            text = self._tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
            )
        except Exception:
            # Fallback: flat string concat for tokenizers without a chat template.
            text = "\n".join(f"[{m['role']}] {m['content']}" for m in msgs)
        # Chat template already includes special tokens — don't double-add them.
        ids = self._tok.encode(text, add_special_tokens=False)
        return [int(t) for t in ids]

    def step(self, action: Any) -> None:
        speaker = self.agent_selection

        if self._terminations.get(speaker) or self._truncations.get(speaker):
            self._was_dead_step(action)
            return

        # Decode the speaker's post
        if action is None or len(action) == 0:
            text = ""
        else:
            text = self._tok.decode(list(action), skip_special_tokens=True)

        # 1. Append the post as 'assistant' in the speaker's own log.
        self._messages[speaker].append({"role": "assistant", "content": text})

        # 2. Buffer the post as a 'user' observation for every other agent.
        for other in self.agents:
            if other != speaker:
                self._pending_obs[other].append(f"{speaker}: {text}")

        # 3. Record on the canonical thread (for env_trace).
        self._thread.append({
            "post_index": self._post_count,
            "speaker": speaker,
            "text": text,
        })
        self._post_count += 1

        # 4. Termination check.
        if self._post_count >= self._max_posts:
            self._final_state = {"thread": list(self._thread)}
            for a in self.possible_agents:
                self._terminations[a] = True
                self._cumulative_rewards[a] = self._reward_fn(self._final_state, a)
            return

        # 5. Advance to next speaker; flush their pending observations.
        next_agent = self._select_next_speaker(prev=speaker)
        self._flush_pending(next_agent)
        self.agent_selection = next_agent

    # ------------------------------------------------------------------ #
    # Internals                                                           #
    # ------------------------------------------------------------------ #

    def _select_next_speaker(self, prev: str) -> str:
        if self._post_order == "round_robin":
            self._next_idx = (self._next_idx + 1) % len(self.agents)
            return self.agents[self._next_idx]
        # random, no immediate repeat
        choices = [a for a in self.agents if a != prev] or list(self.agents)
        return self._rng.choice(choices)

    def _flush_pending(self, agent: str) -> None:
        """Bundle all pending observations into one user message in agent's log."""
        pending = self._pending_obs.get(agent, [])
        if not pending:
            return
        bundled = "\n\n".join(pending)
        # Frame the bundle as a forum-thread update.
        body = (
            "New posts on the thread:\n\n"
            f"{bundled}\n\n"
            "Compose your next post."
        )
        self._messages[agent].append({"role": "user", "content": body})
        self._pending_obs[agent] = []

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
        """Return a structured trace of the canonical thread + agent metadata."""
        return {
            "thread": list(self._thread),
            "agents": list(self.possible_agents),
            "post_order": self._post_order,
            "max_posts": self._max_posts,
            "completed_posts": self._post_count,
        }

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


# ---------------------------------------------------------------------------
# Convenience constructor for the Robotic Athanor scenario
# ---------------------------------------------------------------------------


def make_robotic_athanor_forum(
    *,
    tokenizer: PreTrainedTokenizerBase,
    character_set: str = "canonical_4",
    max_posts: int = 12,
    post_order: str = "round_robin",
    action_token_budget: int = 128,
    seed: int | None = None,
    forum_preamble: str | None = None,
) -> ForumEnv:
    """Build a ForumEnv configured with the Robotic Athanor character roster."""
    from envs.robotic_athanor_personas import (
        FORUM_DESCRIPTION, get_character_set, get_personas,
    )

    agent_names = get_character_set(character_set)
    personas = get_personas(character_set)
    if forum_preamble is None:
        forum_preamble = (
            f"{FORUM_DESCRIPTION}\n\n"
            "A new thread has just opened on the Alchemical Theory section.\n"
            "Anyone may start. Compose your post."
        )
    return ForumEnv(
        agent_names=agent_names,
        agent_personas=personas,
        tokenizer=tokenizer,
        forum_preamble=forum_preamble,
        action_token_budget=action_token_budget,
        max_posts=max_posts,
        post_order=post_order,
        seed=seed,
    )
