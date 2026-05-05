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
import multiprocessing as _mp
import random
import subprocess
import sys
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


def _pyexec_worker(code: str, conn: Any) -> None:
    """Run ``code`` with stdout/stderr captured; send (out, err, rc) back."""
    import contextlib
    import io
    import traceback

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
        thinking_enabled: bool = False,
        thinking_open_tag: str = "<think>",
        thinking_close_tag: str = "</think>",
        post_token_budget: int | None = None,
        total_token_budget: int | None = None,
        python_tool_enabled: bool = False,
        python_open_tag: str = "<python>",
        python_close_tag: str = "</python>",
        python_bot_name: str = "pybot",
        python_timeout_seconds: float = 5.0,
        python_output_max_chars: int = 2000,
        python_use_forkserver: bool = False,
    ) -> None:
        super().__init__()

        self.possible_agents = list(agent_names)
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

        # Optional "thinking tokens" toggle.  When enabled the model is told
        # it may write private reasoning between ``<think>`` and ``</think>``
        # before its final post.  The agent's own context keeps the raw
        # action tokens (the trainer's append-only path is unchanged), but
        # the env strips the tagged regions before recording the post on the
        # canonical thread or delivering it to other agents.
        self._thinking_enabled = bool(thinking_enabled)
        self._thinking_open_tag = thinking_open_tag
        self._thinking_close_tag = thinking_close_tag

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
            thinking_enabled=self._thinking_enabled,
            thinking_open_tag=self._thinking_open_tag,
            thinking_close_tag=self._thinking_close_tag,
            post_token_budget=self._post_token_budget,
            total_token_budget=self._total_token_budget,
            python_tool_enabled=self._python_tool_enabled,
            python_open_tag=self._python_open_tag,
            python_close_tag=self._python_close_tag,
            python_bot_name=self._python_bot_name,
            python_timeout_seconds=self._python_timeout_seconds,
            python_output_max_chars=self._python_output_max_chars,
            python_use_forkserver=self._python_use_forkserver,
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
        self._initial_delivered = {a: False for a in self.agents}
        self._post_count = 0
        self._thread = []
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
        text = "\n\n".join(body_parts)
        ids = self._tok.encode(text, add_special_tokens=False)
        return [int(t) for t in ids]

    def step(self, action: Any) -> None:
        speaker = self.agent_selection
        if self._terminations.get(speaker) or self._truncations.get(speaker):
            self._was_dead_step(action)
            return

        text = self._decode_action(action)

        # Strip ``<python>`` blocks from the public post (the raw
        # assistant tokens, including the tags, remain in the speaker's
        # own context — same convention as thinking tokens).  The
        # extracted code blocks are executed below and surfaced via
        # separate ``pybot`` posts.
        if self._python_tool_enabled:
            text, code_blocks, unclosed = self._extract_python(text)
        else:
            code_blocks = []
            unclosed = False

        # Record on canonical thread.
        self._thread.append({
            "post_index": self._post_count,
            "speaker": speaker,
            "text": text,
        })
        self._post_count += 1

        # Deliver this post to every OTHER agent's pending buffer.
        for other in self.agents:
            if other != speaker:
                self._pending_posts[other].append((speaker, text))

        # Run any python blocks the speaker submitted and emit a pybot
        # post per block.  pybot posts are tool outputs, not turns, so
        # they do not advance ``_post_count`` / ``max_posts``.
        for code in code_blocks:
            output = self._run_python(code)
            bot_text = self._format_pybot_post(speaker, code, output)
            self._thread.append({
                "post_index": None,
                "speaker": self._python_bot_name,
                "text": bot_text,
                "kind": "tool_output",
                "submitter": speaker,
            })
            for a in self.agents:
                self._pending_posts[a].append((self._python_bot_name, bot_text))

        # Surface an unclosed ``<python>`` tag as a visible pybot failure
        # rather than silently dropping the trailing code.  Without this,
        # an agent who ran out of token budget mid-block sees no reply
        # and has no signal that the tool didn't fire.
        if unclosed:
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
            })
            for a in self.agents:
                self._pending_posts[a].append((self._python_bot_name, err_text))

        # Termination check.
        if self._post_count >= self._max_posts:
            self._final_state = {"thread": list(self._thread)}
            for a in self.possible_agents:
                self._terminations[a] = True
                self._cumulative_rewards[a] = self._reward_fn(self._final_state, a)
            return

        # Advance.
        self.agent_selection = self._select_next_speaker(prev=speaker)

    # ------------------------------------------------------------------ #
    # Internals                                                           #
    # ------------------------------------------------------------------ #

    def _decode_action(self, action: Any) -> str:
        if action is None:
            return ""
        try:
            ids = list(action)
        except TypeError:
            text = str(action)
        else:
            if not ids:
                return ""
            text = self._tok.decode(ids, skip_special_tokens=True)
        if self._thinking_enabled:
            text = self._strip_thinking(text)
        return text

    def _strip_thinking(self, text: str) -> str:
        """Remove ``<think>...</think>`` regions from a generated post.

        Only the public post is forwarded to the canonical thread and to
        other agents.  Unclosed thinking blocks (e.g. when the agent ran
        out of token budget mid-reasoning) are dropped from the close tag
        on, which is the right behaviour: the partial reasoning shouldn't
        leak into the public thread.
        """
        o, c = self._thinking_open_tag, self._thinking_close_tag
        out: list[str] = []
        i = 0
        while i < len(text):
            j = text.find(o, i)
            if j < 0:
                out.append(text[i:])
                break
            out.append(text[i:j])
            k = text.find(c, j + len(o))
            if k < 0:
                break  # unclosed — drop the rest
            i = k + len(c)
        return "".join(out).strip()

    def _extract_python(self, text: str) -> tuple[str, list[str], bool]:
        """Pull ``<python>...</python>`` blocks out of a post.

        Returns ``(public_text, code_blocks, unclosed)``.  The tag pair
        is stripped from the post (along with its contents) so that
        other agents never see the raw code in the speaker's post —
        they see it only via the pybot reply that follows.  An open
        tag without a matching close (typically: agent ran out of
        token budget mid-block) sets ``unclosed=True`` so the caller
        can surface a visible failure rather than silently dropping.
        """
        o, c = self._python_open_tag, self._python_close_tag
        out: list[str] = []
        blocks: list[str] = []
        unclosed = False
        i = 0
        while i < len(text):
            j = text.find(o, i)
            if j < 0:
                out.append(text[i:])
                break
            out.append(text[i:j])
            k = text.find(c, j + len(o))
            if k < 0:
                unclosed = True
                break  # drop the trailing partial code; do not execute
            blocks.append(text[j + len(o):k])
            i = k + len(c)
        return "".join(out).strip(), blocks, unclosed

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

    def _run_python(self, code: str) -> str:
        """Execute one code block in isolation and return formatted output.

        Uses a forkserver-backed worker (fast: ~5–15 ms launch) when
        ``python_use_forkserver`` is set; falls back to a cold
        ``subprocess.run`` of ``python -I -c`` otherwise.  Both paths
        are sandboxed in a fresh process (no state leakage between
        calls) and honour ``python_timeout_seconds``.
        """
        run_code = self._autoprint_rewrite(code)
        if self._python_use_forkserver:
            stdout, stderr, rc = self._run_python_forkserver(run_code)
        else:
            stdout, stderr, rc = self._run_python_subprocess(run_code)

        if rc == "timeout":
            return f"[pybot: timed out after {self._python_timeout_seconds:g}s]"
        if rc == "error":
            return f"[pybot: execution error: {stderr}]"

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
        return result

    def _run_python_forkserver(self, code: str) -> tuple[str, str, Any]:
        ctx = _get_pyexec_ctx()
        parent_conn, child_conn = ctx.Pipe(duplex=False)
        try:
            proc = ctx.Process(target=_pyexec_worker, args=(code, child_conn))
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

    def _run_python_subprocess(self, code: str) -> tuple[str, str, Any]:
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-c", code],
                capture_output=True,
                text=True,
                timeout=self._python_timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            return "", "", "timeout"
        except Exception as e:  # noqa: BLE001
            return "", str(e), "error"
        return proc.stdout, proc.stderr, proc.returncode

    def _format_pybot_post(self, submitter: str, code: str, output: str) -> str:
        return (
            f"Running code submitted by {submitter}:\n"
            f"{self._python_open_tag}\n{code}\n{self._python_close_tag}\n"
            f"Output:\n{output}"
        )

    def _render_initial_ctx(self) -> str:
        """First-observation framing each agent sees once."""
        parts = [self._forum_description, self._initial_invitation]
        if self._post_length_note:
            parts.append(self._post_length_note)
        if self._thinking_enabled:
            post_words = max(1, int(self._post_token_budget * 0.7))
            total_words = max(1, int(self._total_token_budget * 0.7))
            parts.append(
                f"Before each post you may write private reasoning between "
                f"{self._thinking_open_tag} and {self._thinking_close_tag}. "
                f"Anything inside those tags stays in your own private notes "
                f"and is not shown to other forum members; only the text "
                f"outside the tags is posted. Two limits: the public post "
                f"itself must stay under about {self._post_token_budget} "
                f"tokens (~{post_words} words), and the total of thinking + "
                f"post combined must stay under about "
                f"{self._total_token_budget} tokens (~{total_words} words) "
                f"— anything past the total is cut off mid-sentence. Within "
                f"those two limits you can spend as much or as little as "
                f"you like on private reasoning. Always close "
                f"{self._thinking_close_tag} before you start writing the "
                f"post."
            )
        return "\n\n".join(p for p in parts if p)

    def _frame_posts(self, posts: list[tuple[str, str]]) -> str:
        """Render new partner posts as direct-speech narrative.

        - Single post:   ``"<Speaker> wrote:\\n<text>"``
        - Multiple:      ``"Several people posted:\\n\\n<framed1>\\n\\n<framed2>"``

        Always direct-speech form — never wrapped in meta-instructions like
        "Compose your next post".  The chat-template wrapping (added by the
        trainer's formatter) provides all the structural framing the model
        needs to know it should respond.
        """
        if len(posts) == 1:
            sp, txt = posts[0]
            return f"{sp} wrote:\n{txt}" if txt else f"{sp} wrote nothing."
        framed = []
        for sp, txt in posts:
            framed.append(f"{sp} wrote:\n{txt}" if txt else f"{sp} wrote nothing.")
        return "Several people posted on the forum:\n\n" + "\n\n".join(framed)

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


