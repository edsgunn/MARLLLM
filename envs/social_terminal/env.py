"""
SocialTerminalEnv — a place-based multi-agent text substrate (PettingZoo AEC).

Design
------
Agents inhabit a directed graph of *places*. Each agent is in exactly one
place at all times; location is the subscription mechanism. The agent
perceives only its current place and interacts with co-present agents
through tools. Every place has at least one fixture object (whiteboard,
noticeboard, etc.) that agents can read and, for editable ones, write to.
Objects are bolted to their place — nothing is portable.

Per-turn protocol
-----------------
* The trainer samples a string from the policy.
* The env scans the string for tool tags — anything between recognised
  tool tags fires; anything outside them is treated as thinking and
  ignored. Tags inside a ``<think>...</think>`` block are explicitly
  *not* fired: a reasoning model rehearsing "I could <read> the board"
  there has not committed to the action.
* Multiple tool tags in one generation are *stacked*: the env executes
  them in document order — but only *productive* verbs (say/emote/
  whisper/write) stack freely. The *information-returning* verbs
  (``<go>``, ``<read>``, ``<where>``, ``<look>``) END THE TURN: each
  queues perception (a new room, an object's contents, a re-render) that
  the actor only receives on its next ``observe``, so any tag appearing
  after a successful one is dropped with an error queued to the actor —
  the policy could not have conditioned on information it has not yet
  seen. For the same reason the env exposes the close tags of read/where/
  look as generation stop-strings (``generation_stop_strings``) so the
  sampler can halt the turn the moment the actor commits to waiting.
* The env mutates world state, emits events at the appropriate place(s),
  and queues directed messages (read results, whispered content, error
  feedback) into per-agent inboxes.
* On the next ``observe`` for each agent, that agent receives, in order:
  the contents of its private inbox, then either a full room render
  (set on entry / first turn) or the room-event delta since the agent
  last looked at this place.

σ convention
------------
The env never authors agent generations. Every byte it emits via
``observe`` is σ=1 for the receiving agent — environment-authored
perception. Everything an agent's policy generated is σ=0 for that
agent. The framework handles tagging; the env only delivers strings.

Tool tag grammar
----------------
Tags fire wherever they appear. Bodies and attributes are stripped of
surrounding whitespace.

  <say>message</say>                       — speak to everyone here
  <whisper to="Agent Name">message</whisper>
                                            — private to one co-present
                                              agent; bystanders see that
                                              it happened
  <emote>action</emote>                    — non-verbal signalled action
  <go>exit-name</go>                       — move through an exit
                                              (exclusive: ends the turn)
  <look/> or <look></look>                 — request a full re-render
                                              of the current place
  <where/>                                 — list exits and people here
  <read>object-name</read>                 — read a fixture's contents
  <write to="object">content</write>       — write to an editable fixture
  <wait/>                                  — explicit no-op

Unclosed tags (cut off by the token budget), tags with missing required
attributes, or tags with empty bodies where one is needed become no-ops
with a specific error message queued back to the actor — the raw
generation is never silently discarded.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from pettingzoo import AECEnv
from transformers import PreTrainedTokenizerBase

from envs.social_terminal.scenario import Place, PlaceObject, Scenario


# ──────────────────────────────────────────────────────────────────────
# World events
# ──────────────────────────────────────────────────────────────────────

@dataclass
class Event:
    """A thing that happened at a place, visible to co-present agents."""
    seq: int
    place: str
    kind: str          # 'arrived','left','say','emote','whisper_signal','write'
    actor: str
    rendered: str      # human-readable line shown to co-present observers
    payload: dict = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────
# Tool-tag parsing
# ──────────────────────────────────────────────────────────────────────

# All recognised tool-tag names. Tags fire wherever they appear in the
# generation, *except* inside a ``<think>...</think>`` block — see
# ``parse_tool_tags`` — and anything outside any recognised tag is treated
# as thinking and ignored by the env.
_TAG_NAMES = ("say", "whisper", "emote", "look", "where", "go", "read",
              "write", "wait", "help")

# Verbs whose effect is to return information the actor cannot have seen
# when it wrote the generation: ``<go>`` reveals a new room next turn;
# ``<read>``/``<where>``/``<look>`` queue room/object content or a re-render
# that only arrives on the next ``observe``. All of them END THE TURN: any
# tag appearing after a *successful* one in the same generation is dropped,
# because the policy could not have conditioned on information it has not
# yet perceived. (Productive verbs — say/emote/whisper/write — return nothing
# to the actor, so they remain freely stackable.)
#
# Every turn-ending verb is also a generation stop-point: once the actor
# commits to one it is waiting on perception, so there is nothing useful
# left to sample this turn. ``<go>`` very much included — without it the
# policy keeps generating actions for (and confabulating the contents of)
# the room it just walked into but has not yet seen.
_TURN_ENDING = {"go", "read", "where", "look"}

# Argless info verbs are usually written self-closing (``<where/>``,
# ``<look/>``); verbs that take an argument carry it in the body and so end
# at the close tag (``<read>x</read>``, ``<go>x</go>``). We can't form a
# reliable stop-string for the bare-arg variants (``<read x>``, ``<go x>``),
# which have no terminator — the env-level turn-ending in ``step`` is the
# backstop for those.
_ARGLESS_INFO_VERBS = {"where", "look"}

# Pattern matching a ``<think>...</think>`` block, including the
# truncated-open case (``<think>...`` with no close, cut off by the token
# budget) via ``\Z``. Tags inside such a block are the model reasoning aloud
# about what it *might* do, not committing to it, so they must not fire.
_THINK_BLOCK_RE = re.compile(
    r"<think\b[^>]*>.*?(?:</think\s*>|\Z)",
    re.IGNORECASE | re.DOTALL,
)


def _thinking_spans(text: str) -> list[tuple[int, int]]:
    """Byte spans covered by ``<think>...</think>`` blocks (open..close)."""
    return [(m.start(), m.end()) for m in _THINK_BLOCK_RE.finditer(text)]

# Pattern matches any recognised open tag, capturing everything between
# the tag name and the closing ``>`` as a loose ``rest`` string. We
# decompose ``rest`` afterwards into attributes, bare-argument content,
# and the self-closing slash, so the parser accepts all of:
#
#     <go>library</go>                       (canonical body form)
#     <go library>                           (bare arg in open tag)
#     <go library/>                          (bare arg, self-closing)
#     <whisper to="Hana">hi</whisper>        (canonical to= form)
#     <whisper Hana>hi</whisper>             (bare arg in open tag)
#
# The principle: the model has been instruction-tuned on several
# tool-call conventions and will mix them. Be liberal in what we
# accept; only reject when the intent is genuinely ambiguous.
_TAG_OPEN_RE = re.compile(
    r"<(?P<name>" + "|".join(_TAG_NAMES) + r")"
    r"(?P<rest>[^>]*)>",
    re.IGNORECASE,
)
_ATTR_RE = re.compile(r"([a-zA-Z_-]+)\s*=\s*\"([^\"]*)\"")


@dataclass
class ParsedCommand:
    verb: str
    target: str = ""    # exit / object / whisper recipient / "" otherwise
    content: str = ""   # message body / written content
    ok: bool = True
    error: str = ""     # populated when ok=False
    span: tuple[int, int] = (0, 0)  # byte offsets in the generation


def _decompose_rest(rest: str) -> tuple[dict[str, str], str, bool]:
    """Split the chunk between ``<NAME`` and ``>`` into its parts.

    Returns ``(attrs, bare_arg, is_self_closed)``:

    * ``attrs``   — dict of ``key="value"`` pairs.
    * ``bare_arg`` — any non-attribute, non-whitespace content (the
                     thing the model jammed into the open tag).
    * ``is_self_closed`` — True if ``rest`` ended with ``/``.
    """
    s = rest
    is_self_closed = s.rstrip().endswith("/")
    if is_self_closed:
        s = s.rstrip()[:-1]
    # Extract key="value" pairs, leaving everything else in ``leftover``.
    attrs: dict[str, str] = {}
    leftover_parts: list[str] = []
    last_end = 0
    for m in _ATTR_RE.finditer(s):
        leftover_parts.append(s[last_end:m.start()])
        attrs[m.group(1).lower()] = m.group(2)
        last_end = m.end()
    leftover_parts.append(s[last_end:])
    bare_arg = " ".join(p.strip() for p in leftover_parts if p.strip()).strip()
    return attrs, bare_arg, is_self_closed


def parse_tool_tags(text: str) -> list[ParsedCommand]:
    """Scan ``text`` for tool tags in document order.

    Returns a list of :class:`ParsedCommand` in the order they appeared.
    Tags whose intent can't be recovered (missing target on whisper,
    empty body where content is required, etc.) come back with
    ``ok=False`` and a specific error message; the caller queues the
    error back to the actor.

    Parsing rules:

    * Outer match only. Once we open a tag, we look for the *next*
      matching close tag without re-scanning the interior — so a
      ``<say>`` inside ``<say>...</say>`` is absorbed by the first
      ``</say>`` (same convention as the forum env's <post> parser).
    * Unclosed tag: append one error ParsedCommand, then stop. An
      unclosed tag normally means the generation was truncated, so
      anything after it is past the budget anyway.
    * Liberal argument resolution. For each verb the argument may
      appear as a ``to="..."`` attribute, a bare token in the open
      tag (``<go library>``), or the body (``<go>library</go>``).
      Whichever is present wins; if more than one is present, the
      attribute beats the bare arg beats the body.
    * Tags inside a ``<think>...</think>`` block do not fire. Reasoning
      models rehearse actions there ("I could <read> the board…") without
      committing to them; only tags the model emits *outside* its thinking
      are real. The block is skipped wholesale, truncated-open included.
    """
    think_spans = _thinking_spans(text)

    def _in_think(idx: int) -> int | None:
        """If ``idx`` falls inside a thinking block, return that block's end
        offset (so the scan can jump past it); otherwise None."""
        for start, end in think_spans:
            if start <= idx < end:
                return end
        return None

    cmds: list[ParsedCommand] = []
    pos = 0
    while pos < len(text):
        m = _TAG_OPEN_RE.search(text, pos)
        if not m:
            break
        # An open tag inside a thinking block is rehearsal, not action:
        # jump past the whole block and keep scanning after it.
        think_end = _in_think(m.start())
        if think_end is not None:
            pos = think_end
            continue
        name = m.group("name").lower()
        rest = m.group("rest") or ""
        open_start = m.start()
        open_end = m.end()
        attrs, bare_arg, is_self_closed = _decompose_rest(rest)

        if is_self_closed:
            body = ""
            span_end = open_end
            pos = open_end
        else:
            close_tag = f"</{name}>"
            lower = text.lower()
            k = lower.find(close_tag, open_end)
            if k < 0:
                # No matching close tag. Two sub-cases:
                # (a) the open tag already carries the argument the verb
                #     needs (``<go library>``, ``<say hello>``) — the
                #     model intended an implicit self-close. Treat as
                #     such, with empty body.
                # (b) genuinely truncated mid-tag — no bare arg, no
                #     attrs, no body — surface as an unclosed error.
                if bare_arg or attrs:
                    body = ""
                    span_end = open_end
                    pos = open_end
                else:
                    cmds.append(ParsedCommand(
                        verb=name, ok=False,
                        error=(f"<{name}> was not closed before the end "
                               f"of the turn (likely cut off by the token "
                               f"budget). No action was taken."),
                        span=(open_start, len(text)),
                    ))
                    break
            else:
                body = text[open_end:k].strip()
                span_end = k + len(close_tag)
                pos = span_end

        # Resolve target/content per verb. Priority for single-arg verbs:
        # to= attribute > bare arg > body. Resolve into ParsedCommand.
        target = ""
        content = ""

        if name in ("say", "emote"):
            # Just need content. Prefer body; fall back to bare arg.
            content = body if body else bare_arg
            if not content:
                cmds.append(ParsedCommand(
                    verb=name, ok=False,
                    error=f"<{name}> had no content.",
                    span=(open_start, span_end),
                ))
                continue

        elif name in ("go", "read"):
            # Single target. Prefer to= attr, then bare arg, then body.
            target = attrs.get("to", "").strip() or bare_arg or body
            if not target:
                cmds.append(ParsedCommand(
                    verb=name, ok=False,
                    error=(f"<{name}> needs an argument — write it as "
                           f"<{name}>name</{name}> or <{name} name>."),
                    span=(open_start, span_end),
                ))
                continue

        elif name in ("whisper", "write"):
            # Target + content. Target: to= attr > bare arg. Content: body.
            target = attrs.get("to", "").strip() or bare_arg
            content = body
            if not target:
                cmds.append(ParsedCommand(
                    verb=name, ok=False,
                    error=(f"<{name}> needs a target — write it as "
                           f"<{name} to=\"...\">...</{name}>."),
                    span=(open_start, span_end),
                ))
                continue
            if not content:
                cmds.append(ParsedCommand(
                    verb=name, ok=False,
                    error=f"<{name}> had no content.",
                    span=(open_start, span_end),
                ))
                continue

        # else: look / where / wait / help — no args, no body. Anything
        # bare_arg or body that came along is ignored silently.

        cmds.append(ParsedCommand(
            verb=name, target=target, content=content,
            span=(open_start, span_end),
        ))
    return cmds


# ──────────────────────────────────────────────────────────────────────
# Env
# ──────────────────────────────────────────────────────────────────────

class SocialTerminalEnv(AECEnv):
    """Place-based multi-agent text environment.

    Parameters
    ----------
    scenario:
        Loaded :class:`Scenario` defining the place graph, agents, and
        personas.
    tokenizer:
        HuggingFace tokenizer for encoding observation strings.
    max_turns:
        Total turns across all agents before the episode terminates.
    action_token_budget:
        Tokens per turn (total, thinking + <act>).
    act_token_budget:
        Cap on the ``<act>`` content the agent is *told* to stay under.
        Defaults to ``action_token_budget``.
    whisper_visible:
        If True (default), co-present bystanders see ``"X whispered
        something to Y"`` events (content stays private). If False,
        whispers are entirely invisible to bystanders.
    seed:
        RNG seed for turn order.
    reward_fn:
        ``(final_state, agent_name) → float``. Defaults to zero.
    stop_at_info_actions:
        If True (default), the close tags of the information-returning
        verbs (``</read>``, ``</where>``, ``</look>``) are exposed via
        ``generation_stop_strings`` so the sampler halts the turn the
        moment one is emitted, instead of letting the model continue (and
        typically *confabulate* the result it hasn't received yet). The
        env-level turn-ending behaviour applies regardless; this flag only
        governs the sampler-side optimisation, which the trainer disables
        for native-thinking models (a ``</read>`` inside ``<think>`` would
        otherwise truncate the chain of thought).
    """

    metadata = {"render_modes": [], "name": "social_terminal_v1"}

    obs_is_full_context: bool = False

    def __init__(
        self,
        scenario: Scenario,
        tokenizer: PreTrainedTokenizerBase,
        max_turns: int = 24,
        action_token_budget: int = 384,
        act_token_budget: int | None = None,
        whisper_visible: bool = True,
        seed: int | None = None,
        reward_fn: Callable[[Any, str], float] | None = None,
        stop_at_info_actions: bool = True,
    ) -> None:
        super().__init__()
        self._scenario = scenario
        self._tok = tokenizer
        self._stop_at_info_actions = bool(stop_at_info_actions)
        self._max_turns = int(max_turns)
        self.action_token_budget = int(action_token_budget)
        self._act_token_budget = int(act_token_budget or action_token_budget)
        self._whisper_visible = bool(whisper_visible)
        self._seed = seed
        self._rng = random.Random(seed)
        self._reward_fn = reward_fn if reward_fn is not None else (lambda _s, _a: 0.0)

        self.possible_agents = scenario.agent_names()
        self._agents_by_name = {a.name: a for a in scenario.agents}

        # Per-episode mutable state — populated in reset().
        self._places: dict[str, Place] = {}
        self._agent_place: dict[str, str] = {}
        self._place_events: dict[str, list[Event]] = {}
        self._event_counter: int = 0
        self._agent_last_seq: dict[str, dict[str, int]] = {}
        self._agent_inbox: dict[str, list[str]] = {}
        self._agent_needs_full_render: dict[str, bool] = {}
        self._initial_delivered: dict[str, bool] = {}
        self._turns: list[dict] = []
        self._turn_count: int = 0
        self._next_idx: int = 0
        self._events_log: list[dict] = []  # diagnostic event log

        # PettingZoo AEC required state.
        self.agents: list[str] = []
        self.agent_selection: str = ""
        self._cumulative_rewards: dict[str, float] = {}
        self._terminations: dict[str, bool] = {}
        self._truncations: dict[str, bool] = {}
        self._infos: dict[str, dict] = {}
        self._final_state: Any = None

    # ── trainer plumbing ─────────────────────────────────────────────

    @property
    def default_character_prompts(self) -> dict[str, str]:
        return dict(self._scenario.personas)

    @property
    def generation_stop_strings(self) -> list[str]:
        """Literal strings whose emission should halt generation this turn.

        Covers every information-returning verb (``_TURN_ENDING``): once the
        model writes ``</read>`` or ``</go>`` it is committing to wait for
        perception it can't see until next turn, so there is nothing useful
        left to sample — and anything it *did* sample would be confabulated
        (e.g. the contents of the room it just walked into). The trainer
        passes these to the sampler with ``include_stop_str_in_output=True``
        so the env still receives the close tag to parse.

        For the argless verbs we also stop on the self-closing form
        (``<look/>``, ``<look />``), which is how they're usually written.

        Best-effort prevention only: it catches the close-tag and
        self-closing forms but not the bare-arg variants the parser also
        accepts (``<read board>``, ``<go library>`` have no terminator). The
        env-level turn-ending in ``step`` is the authoritative backstop for
        everything the stop-strings miss. Empty when disabled.
        """
        if not self._stop_at_info_actions:
            return []
        stops: list[str] = [f"</{v}>" for v in sorted(_TURN_ENDING)]
        for v in sorted(_ARGLESS_INFO_VERBS):
            stops.append(f"<{v}/>")
            stops.append(f"<{v} />")
        return stops

    # ── PettingZoo required surface ──────────────────────────────────

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

    # ── reset / step / observe ───────────────────────────────────────

    def reset(self, seed: int | None = None, options: dict | None = None) -> None:
        if seed is not None:
            self._rng = random.Random(seed)

        # Deep-copy the place graph so per-episode edits to objects don't
        # bleed across episodes.
        self._places = {
            pid: Place(
                place_id=p.place_id,
                title=p.title,
                description=p.description,
                exits=dict(p.exits),
                objects={
                    oid: PlaceObject(
                        obj_id=o.obj_id, title=o.title,
                        description=o.description, content=o.content,
                        editable=o.editable, append_only=o.append_only,
                    )
                    for oid, o in p.objects.items()
                },
            )
            for pid, p in self._scenario.places.items()
        }
        self._agent_place = {a.name: a.start for a in self._scenario.agents}
        self._place_events = {pid: [] for pid in self._places}
        self._event_counter = 0
        self._agent_last_seq = {
            a: {pid: 0 for pid in self._places} for a in self.possible_agents
        }
        self._agent_inbox = {a: [] for a in self.possible_agents}
        self._agent_needs_full_render = {a: True for a in self.possible_agents}
        self._initial_delivered = {a: False for a in self.possible_agents}
        self._turns = []
        self._turn_count = 0
        self._events_log = []

        self.agents = list(self.possible_agents)
        order_seed = (options or {}).get("order_seed")
        order_rng = random.Random(order_seed) if order_seed is not None else self._rng
        order_rng.shuffle(self.agents)
        self._next_idx = 0
        self.agent_selection = self.agents[0]

        self._cumulative_rewards = {a: 0.0 for a in self.agents}
        self._terminations = {a: False for a in self.agents}
        self._truncations = {a: False for a in self.agents}
        self._infos = {a: {} for a in self.agents}
        self._final_state = None

    def observe(self, agent: str) -> list[int]:
        if agent not in self._agent_place:
            return []

        parts: list[str] = []

        if not self._initial_delivered[agent]:
            parts.append(self._render_intro())
            self._initial_delivered[agent] = True

        # Drain private inbox (read results, whispered content, error
        # feedback). These are σ=1 for the actor — env-authored.
        inbox = self._agent_inbox[agent]
        self._agent_inbox[agent] = []
        for msg in inbox:
            parts.append(msg)

        place = self._agent_place[agent]
        events = self._place_events[place]
        if self._agent_needs_full_render[agent]:
            parts.append(self._render_place(place, viewer=agent))
            self._agent_last_seq[agent][place] = len(events)
            self._agent_needs_full_render[agent] = False
        else:
            seen = self._agent_last_seq[agent].get(place, 0)
            # Filter out events the viewer themselves caused — those
            # tokens were already in their σ=0 generation; re-delivering
            # them as σ=1 perception is redundant and corrupts the split.
            new = [e for e in events[seen:] if e.actor != agent]
            if new:
                parts.append(self._render_events(new))
            self._agent_last_seq[agent][place] = len(events)

        if not parts:
            return []
        text = "\n\n".join(parts)
        ids = self._tok.encode(text, add_special_tokens=False)
        return [int(t) for t in ids]

    def step(self, action: Any) -> None:
        speaker = self.agent_selection
        if self._terminations.get(speaker) or self._truncations.get(speaker):
            self._was_dead_step(action)
            return

        raw = self._decode_action(action)
        commands = parse_tool_tags(raw)
        turn_record: dict[str, Any] = {
            "turn": self._turn_count,
            "speaker": speaker,
            "place": self._agent_place[speaker],
            "raw": raw,
            "n_tags_found": len(commands),
            "commands": [],
        }

        if not commands:
            self._agent_inbox[speaker].append(
                "(no tool tags found in your last turn — no action was taken. "
                "Use tags like <say>...</say>, <whisper to=\"Name\">...</whisper>, "
                "<go>...</go> to act in the world.)"
            )
            turn_record["parse_status"] = "no_tags"
        else:
            # Walk in document order. ``<go>`` is exclusive: once a
            # successful go fires, any later tag in the same generation
            # is dropped with a specific error queued to the actor.
            turn_ended_by: str | None = None
            for cmd in commands:
                cmd_record: dict[str, Any] = {
                    "verb": cmd.verb, "target": cmd.target,
                    "content": cmd.content, "span": list(cmd.span),
                }
                if turn_ended_by:
                    cmd_record["dropped"] = (
                        f"earlier <{turn_ended_by}> ended the turn"
                    )
                    self._agent_inbox[speaker].append(
                        f"(<{cmd.verb}> was dropped: an earlier "
                        f"<{turn_ended_by}> in the same turn ended it. "
                        f"<{turn_ended_by}> returns something you only see "
                        f"next turn, so you can't act on it yet — split this "
                        f"across turns.)"
                    )
                    turn_record["commands"].append(cmd_record)
                    continue
                if not cmd.ok:
                    self._agent_inbox[speaker].append(
                        f"(tag rejected: {cmd.error})"
                    )
                    cmd_record["parse_status"] = "parse_error"
                    cmd_record["parse_error"] = cmd.error
                    turn_record["commands"].append(cmd_record)
                    continue
                effect = self._dispatch(speaker, cmd)
                cmd_record["parse_status"] = effect.get("parse_status", "ok")
                cmd_record["effect"] = effect
                turn_record["commands"].append(cmd_record)
                if (cmd.verb in _TURN_ENDING
                        and effect.get("parse_status", "ok") == "ok"):
                    turn_ended_by = cmd.verb

        self._turns.append(turn_record)
        self._turn_count += 1

        if self._turn_count >= self._max_turns:
            self._final_state = {"turns": list(self._turns)}
            for a in self.possible_agents:
                self._terminations[a] = True
                self._cumulative_rewards[a] = self._reward_fn(self._final_state, a)
            return

        self._next_idx = (self._next_idx + 1) % len(self.agents)
        self.agent_selection = self.agents[self._next_idx]

    # ── command dispatch ─────────────────────────────────────────────

    def _dispatch(self, actor: str, cmd: ParsedCommand) -> dict:
        """Execute one command. Returns an effect dict for the per-turn
        record; ``effect["parse_status"]`` is ``"ok"`` on success or a
        specific failure code (``no_such_exit``, ``no_such_target`` …).
        Side-effects (room events, inbox messages) happen here.
        """
        place = self._agent_place[actor]
        v = cmd.verb

        if v == "look":
            self._agent_needs_full_render[actor] = True
            return {"kind": "look", "parse_status": "ok"}

        if v == "wait":
            return {"kind": "wait", "parse_status": "ok"}

        if v == "help":
            self._agent_inbox[actor].append(self._help_text())
            return {"kind": "help", "parse_status": "ok"}

        if v == "where":
            self._agent_inbox[actor].append(self._where_text(place, actor))
            return {"kind": "where", "parse_status": "ok"}

        if v == "go":
            return self._do_go(actor, cmd.target)

        if v == "say":
            self._emit_event(
                place=place, kind="say", actor=actor,
                rendered=f"{actor} says: {cmd.content}",
                payload={"text": cmd.content},
            )
            return {"kind": "say", "text": cmd.content, "parse_status": "ok"}

        if v == "emote":
            self._emit_event(
                place=place, kind="emote", actor=actor,
                rendered=f"* {actor} {cmd.content}",
                payload={"text": cmd.content},
            )
            return {"kind": "emote", "text": cmd.content, "parse_status": "ok"}

        if v == "whisper":
            return self._do_whisper(actor, cmd.target, cmd.content)

        if v == "read":
            return self._do_read(actor, cmd.target)

        if v == "write":
            return self._do_write(actor, cmd.target, cmd.content)

        # Defensive — parse_tool_tags shouldn't produce any other verb.
        self._agent_inbox[actor].append(f"(unhandled verb: {v})")
        return {"kind": v, "parse_status": "unhandled_verb"}

    def _do_go(self, actor: str, exit_label: str) -> dict:
        old_place = self._agent_place[actor]
        place = self._places[old_place]
        dest = self._match_key(exit_label, place.exits.keys())
        if dest is None:
            self._agent_inbox[actor].append(
                f"(no exit named {exit_label!r}. Exits here: "
                f"{', '.join(sorted(place.exits)) or '(none)'}.)"
            )
            return {"kind": "go", "parse_status": "no_such_exit"}
        new_place = place.exits[dest]
        self._emit_event(
            place=old_place, kind="left", actor=actor,
            rendered=f"{actor} left towards {dest}.",
            payload={"exit": dest, "to": new_place},
        )
        self._agent_place[actor] = new_place
        arrived_via = self._reverse_exit(new_place, old_place) or "elsewhere"
        self._emit_event(
            place=new_place, kind="arrived", actor=actor,
            rendered=f"{actor} arrived from {arrived_via}.",
            payload={"from": old_place, "from_exit": arrived_via},
        )
        # Actor will full-render the destination next turn; do not
        # redeliver the arrival event to them.
        self._agent_needs_full_render[actor] = True
        self._agent_last_seq[actor][new_place] = len(self._place_events[new_place])
        return {"kind": "go", "from": old_place, "to": new_place,
                "parse_status": "ok"}

    def _do_whisper(self, actor: str, target_raw: str, content: str) -> dict:
        place = self._agent_place[actor]
        co_present = [a for a in self.possible_agents
                      if a != actor and self._agent_place[a] == place]
        target = self._match_agent(target_raw, co_present)
        if target is None:
            self._agent_inbox[actor].append(
                f"(no one named {target_raw!r} is here to whisper to. "
                f"Here with you: {', '.join(co_present) or '(no one)'}.)"
            )
            return {"kind": "whisper", "parse_status": "no_such_target"}
        self._agent_inbox[target].append(f"{actor} whispers to you: {content}")
        if self._whisper_visible:
            self._emit_event(
                place=place, kind="whisper_signal", actor=actor,
                rendered=f"{actor} whispers something to {target}.",
                payload={"target": target},
            )
        return {"kind": "whisper", "target": target,
                "content_len": len(content), "parse_status": "ok"}

    def _do_read(self, actor: str, obj_raw: str) -> dict:
        place = self._agent_place[actor]
        objects = self._places[place].objects
        oid = self._match_key(obj_raw, objects.keys())
        if oid is None:
            self._agent_inbox[actor].append(
                f"(no object named {obj_raw!r} here. Objects here: "
                f"{', '.join(sorted(objects)) or '(none)'}.)"
            )
            return {"kind": "read", "parse_status": "no_such_object"}
        obj = objects[oid]
        body = obj.content.rstrip() or "(blank)"
        self._agent_inbox[actor].append(
            f"You read the {obj.title}:\n{body}"
        )
        return {"kind": "read", "object": oid, "parse_status": "ok"}

    def _do_write(self, actor: str, obj_raw: str, content: str) -> dict:
        place = self._agent_place[actor]
        objects = self._places[place].objects
        oid = self._match_key(obj_raw, objects.keys())
        if oid is None:
            self._agent_inbox[actor].append(
                f"(no object named {obj_raw!r} here. Objects here: "
                f"{', '.join(sorted(objects)) or '(none)'}.)"
            )
            return {"kind": "write", "parse_status": "no_such_object"}
        obj = objects[oid]
        if not obj.editable:
            self._agent_inbox[actor].append(
                f"(the {obj.title} is not something you can write on.)"
            )
            return {"kind": "write", "parse_status": "not_editable"}
        if obj.append_only:
            sep = "\n" if obj.content else ""
            obj.content = f"{obj.content}{sep}[{actor}] {content}"
            kind = "appended to"
        else:
            obj.content = content
            kind = "rewrote"
        self._emit_event(
            place=place, kind="write", actor=actor,
            rendered=f"{actor} {kind} the {obj.title}: {content}",
            payload={"object": oid, "content": content,
                     "append": obj.append_only},
        )
        return {"kind": "write", "object": oid,
                "append": obj.append_only, "parse_status": "ok"}

    # ── world helpers ────────────────────────────────────────────────

    def _emit_event(self, place: str, kind: str, actor: str,
                    rendered: str, payload: dict | None = None) -> None:
        ev = Event(
            seq=self._event_counter, place=place, kind=kind, actor=actor,
            rendered=rendered, payload=payload or {},
        )
        self._event_counter += 1
        self._place_events[place].append(ev)

    def _reverse_exit(self, place: str, src: str) -> str | None:
        """Return the exit-label in ``place`` that leads back to ``src``."""
        for label, dest in self._places[place].exits.items():
            if dest == src:
                return label
        return None

    @staticmethod
    def _match_key(query: str, keys) -> str | None:
        if not query:
            return None
        q = query.strip().lower().rstrip(".!?,")
        keys = list(keys)
        # Exact (case-insensitive).
        for k in keys:
            if k.lower() == q:
                return k
        # Prefix.
        for k in keys:
            if k.lower().startswith(q):
                return k
        # Loose: drop spaces/underscores and compare.
        qc = q.replace(" ", "").replace("_", "").replace("-", "")
        for k in keys:
            kc = k.lower().replace(" ", "").replace("_", "").replace("-", "")
            if kc == qc:
                return k
        return None

    def _match_agent(self, query: str, candidates: list[str]) -> str | None:
        """Match against full names, then short/first-name forms."""
        if not query:
            return None
        q = query.strip().lower().rstrip(".!?,:")
        for c in candidates:
            if c.lower() == q:
                return c
        # First name.
        for c in candidates:
            first = c.split()[0].lower()
            if first == q:
                return c
        # Configured short name.
        for c in candidates:
            short = self._agents_by_name.get(c)
            if short and short.short.lower() == q:
                return c
        # Prefix on full name.
        for c in candidates:
            if c.lower().startswith(q):
                return c
        return None

    # ── rendering ────────────────────────────────────────────────────

    def _render_intro(self) -> str:
        s = self._scenario
        return (
            f"{s.description}\n\n"
            f"You only see and hear what's in the room you're in. To find "
            f"out who or what is elsewhere, go there and look.\n\n"
            f"You act by writing tool tags in your reply. Text outside any "
            f"tag is private — think things through there if it helps. You "
            f"can use multiple tags per turn; they fire in the order you "
            f"wrote them.\n\n"
            f"Tags:\n"
            f"{self._help_text()}\n\n"
            f"Token budget per turn: ~{self.action_token_budget}."
        )

    def _help_text(self) -> str:
        return (
            "  <say>message</say>                       — speak to everyone here\n"
            "  <whisper to=\"Name\">message</whisper>     — private to one person; others see it happened\n"
            "  <emote>action</emote>                    — non-verbal action visible here\n"
            "  <go>exit-name</go>                       — move to a connected room (any tag written after this in the same reply is dropped — you haven't seen the new room yet)\n"
            "  <look/>                                  — re-render the current room\n"
            "  <where/>                                 — list exits and people here\n"
            "  <read>object-name</read>                 — read a fixture's contents\n"
            "  <write to=\"object\">content</write>       — write to an editable fixture\n"
            "  <wait/>                                  — no-op\n"
            "  <help/>                                  — re-deliver this list"
        )

    def _where_text(self, place_id: str, viewer: str) -> str:
        p = self._places[place_id]
        others = [a for a in self.possible_agents
                  if a != viewer and self._agent_place[a] == place_id]
        exits = ", ".join(f"{label} → {self._places[dest].title}"
                          for label, dest in sorted(p.exits.items())) or "(none)"
        people = ", ".join(others) or "(no one else here)"
        return (
            f"You are in {p.title}.\n"
            f"Exits: {exits}\n"
            f"Also here: {people}"
        )

    def _render_place(self, place_id: str, viewer: str) -> str:
        p = self._places[place_id]
        others = [a for a in self.possible_agents
                  if a != viewer and self._agent_place[a] == place_id]
        exit_lines = [f"  - {label} → {self._places[dest].title}"
                      for label, dest in sorted(p.exits.items())]
        if not exit_lines:
            exit_lines = ["  (no exits)"]
        if p.objects:
            obj_lines = [f"  - {o.title} ({oid}): {o.description}"
                         for oid, o in p.objects.items()]
        else:
            obj_lines = ["  (none)"]
        people_line = (
            f"Also here: {', '.join(others)}." if others
            else "You are alone here."
        )
        return (
            f"=== {p.title} ===\n"
            f"{p.description}\n\n"
            f"{people_line}\n\n"
            f"Exits:\n" + "\n".join(exit_lines) + "\n\n"
            f"Things here:\n" + "\n".join(obj_lines)
        )

    def _render_events(self, events: list[Event]) -> str:
        if len(events) == 1:
            return events[0].rendered
        return "Since you last looked:\n" + "\n".join(
            f"  · {e.rendered}" for e in events
        )

    # ── action decode ────────────────────────────────────────────────

    def _decode_action(self, action: Any) -> str:
        if action is None:
            return ""
        try:
            ids = list(action)
        except TypeError:
            return str(action)
        if not ids:
            return ""
        return self._tok.decode(ids, skip_special_tokens=True)

    # ── PettingZoo dead-step ─────────────────────────────────────────

    def _was_dead_step(self, action: Any) -> None:
        if action is not None:
            raise ValueError("Only None is valid for a terminated/truncated agent.")
        agent = self.agent_selection
        if agent in self.agents:
            self.agents.remove(agent)
        self._cumulative_rewards[agent] = 0.0
        if self.agents:
            self.agent_selection = self.agents[0]

    # ── trace ────────────────────────────────────────────────────────

    def episode_trace(self) -> dict:
        return {
            "scenario": self._scenario.name,
            "agents": list(self.possible_agents),
            "turns": list(self._turns),
            "completed_turns": self._turn_count,
            "max_turns": self._max_turns,
            "final_object_state": {
                pid: {oid: o.content for oid, o in p.objects.items()}
                for pid, p in self._places.items()
            },
            "final_agent_locations": dict(self._agent_place),
        }
