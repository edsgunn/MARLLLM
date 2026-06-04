"""
Scripted (no-LLM) smoke test for SocialTerminalEnv.

Drives the env from a hand-written sequence of command strings and
prints, for each turn:

  * the rendered observation delivered to the actor (σ=1 for them)
  * the actor's full generation (σ=0 for them)
  * the parsed command and its effect

In addition, before the first turn we render every agent's initial
observation side-by-side, so a reader can eyeball the fog-of-war
property: at episode start each agent sees only their starting room
and only the people in it.

Run::

    python -m envs.social_terminal.smoke_test
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Make the repo root importable when run as ``python smoke_test.py`` too.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from envs.social_terminal.env import SocialTerminalEnv
from envs.social_terminal.scenario import load_scenario


class _ByteTokenizer:
    """Round-trip tokenizer for offline smoke testing.

    The env only uses ``encode(text, add_special_tokens=False)`` and
    ``decode(ids, skip_special_tokens=True)``. We encode UTF-8 bytes so
    each token id round-trips a byte — dependency-free.
    """

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        try:
            return bytes(int(i) for i in ids).decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            return ""


def _print_block(title: str, body: str) -> None:
    bar = "─" * 72
    print(f"\n{bar}\n{title}\n{bar}\n{body}")


def _fog_of_war_check(env: SocialTerminalEnv, tok: _ByteTokenizer) -> None:
    """Render every agent's first observation without consuming the env.

    We do this by temporarily reaching into the env's state to clone the
    ``_initial_delivered`` flag, calling ``observe`` for each agent in
    turn (which would normally only be called for the currently-active
    agent), and resetting the flag afterwards so the real first-turn
    observe still emits the intro. The diagnostic is read-only.
    """
    saved_initial = dict(env._initial_delivered)  # noqa: SLF001
    saved_last_seq = {a: dict(v) for a, v in env._agent_last_seq.items()}  # noqa: SLF001
    saved_needs_full = dict(env._agent_needs_full_render)  # noqa: SLF001
    saved_inbox = {a: list(v) for a, v in env._agent_inbox.items()}  # noqa: SLF001

    print("\n" + "═" * 72)
    print("FOG-OF-WAR CHECK — each agent's view at episode start")
    print("═" * 72)
    for agent in env.possible_agents:
        ids = env.observe(agent)
        text = tok.decode(ids)
        _print_block(f"{agent}'s initial view (σ=1)", text)

    # Restore so the actual first turn still gets the intro etc.
    env._initial_delivered = saved_initial  # noqa: SLF001
    env._agent_last_seq = saved_last_seq  # noqa: SLF001
    env._agent_needs_full_render = saved_needs_full  # noqa: SLF001
    env._agent_inbox = saved_inbox  # noqa: SLF001


def main() -> None:
    scenario = load_scenario("ashbourne_school")
    tok = _ByteTokenizer()
    env = SocialTerminalEnv(
        scenario=scenario,
        tokenizer=tok,
        max_turns=16,
        action_token_budget=384,
        seed=0,
    )
    env.reset(seed=0, options={"order_seed": 0})

    print(f"Turn order (round-robin): {env.agents}")
    _fog_of_war_check(env, tok)

    # Build the script *after* we know the order. The first 8 turns are
    # round 1; turns 9–16 are round 2. We index by the agent's name so
    # the script is robust to whatever shuffle the seed produced.
    #
    # The script exercises the new tag grammar end-to-end:
    #   * single tool tags (<say>, <write>, <go>, <read>, <whisper>)
    #   * thinking outside any tag (free prose) — must be ignored
    #   * stacking multiple tags in one turn (<read> + <say> on Priya's R2)
    #   * <go>-exclusivity: an extra tag after <go> in the same turn is
    #     dropped with an error queued to the actor (Olu's R1)
    by_name: dict[str, list[str]] = {
        # Round 1
        "Priya Shah": [
            # In maths_classroom. Free prose thinking, then a single tag.
            "Let me grab the sheet first so I can see Problem 1 properly.\n"
            "<read>problem_sheet</read>",
            # Round 2: STACKED — say a conjecture *and* commit it to the board.
            "<say>I think P1 is just Fibonacci — subsets of {1..n} with no two consecutive count = F_{n+2}.</say>\n"
            "<write to=\"whiteboard\">P1 conjecture: a(n) = F_{n+2}. Proof by recurrence a(n) = a(n-1) + a(n-2).</write>",
        ],
        "Tom Whitaker": [
            "<think>i want to see priya's framing before i commit to anything.</think>\n"
            "<say>Priya — what does the sheet say for P1? I didn't grab one.</say>",
            "<say>Hold on — does the recurrence work at n=1? Vacuous case so a(1) = 2 ({} and {1}). F_3 = 2. OK, fine.</say>",
        ],
        "Hana Yilmaz": [
            "<say>I'll do n=1,2,3 on paper and tell you the counts in a minute.</say>",
            "<say>n=1 gives 2, n=2 gives 3, n=3 gives 5. Fibonacci, as Priya said.</say>",
        ],
        "Olu Adeyemi": [
            # In common_room. Move to maths — but also (wrongly) try to <say>
            # *after* the <go>. The post-<go> tag must be dropped with an
            # error queued back to Olu next turn.
            "<go>maths classroom</go>\n"
            "<say>Hi everyone — I'm here to chip in on P1.</say>",
            "<say>I bet P1 and P3 are the same problem — line vs cycle. The cycle case is the Lucas numbers.</say>",
        ],
        "Beatrice Okafor": [
            "<write to=\"chalkboard\">Hint for P1: try a(n) = a(n-1) + a(n-2). Two cases on whether n is in the subset.</write>",
            "<go>library</go>",
        ],
        "Sam Pritchard": [
            # Whisper — content goes to Beatrice's inbox only; bystanders
            # see a content-less event.
            "<whisper to=\"Beatrice Okafor\">honestly — is P3 just P1 on a cycle? feels like a trick.</whisper>",
            # Round 2: alone in common_room now. Use <look/> self-closing.
            "<look/>",
        ],
        "Imogen Carter": [
            "<write to=\"library_noticeboard\">I'll do a clean writeup of P1 once someone's settled on the closed form — ping me.</write>",
            "<go>maths classroom</go>",
        ],
        "Marcus Webb": [
            "<read>reference_shelf</read>",
            "<say>Sorry — is 'generating function' the kind of thing I should already know, or OK to ask?</say>",
        ],
    }

    # Flatten into the turn order.
    turn_order = list(env.agents)
    SCRIPT: list[str] = []
    for round_idx in range(2):
        for agent in turn_order:
            SCRIPT.append(by_name[agent][round_idx])

    for turn_idx, generation in enumerate(SCRIPT):
        agent = env.agent_selection
        obs_ids = env.observe(agent)
        obs_text = tok.decode(obs_ids) if obs_ids else "(no new content)"
        _print_block(
            f"TURN {turn_idx + 1} — observation delivered to {agent} (σ=1)",
            obs_text,
        )
        _print_block(
            f"TURN {turn_idx + 1} — {agent}'s generation (σ=0)",
            generation,
        )
        act_ids = tok.encode(generation)
        env.step(act_ids)
        rec = env._turns[-1]  # noqa: SLF001 — diagnostic peek
        eff = {k: v for k, v in rec.items()
               if k in ("parse_status", "n_tags_found", "commands")}
        _print_block(
            f"TURN {turn_idx + 1} — per-turn record",
            json.dumps(eff, indent=2),
        )

    print("\n" + "═" * 72)
    print("FINAL STATE")
    print("═" * 72)
    trace = env.episode_trace()
    print("Final agent locations:")
    for a, p in trace["final_agent_locations"].items():
        print(f"  {a:<20} → {p}")
    print("\nFinal object contents:")
    for pid, objs in trace["final_object_state"].items():
        print(f"  [{pid}]")
        for oid, content in objs.items():
            body = content.replace("\n", "\n      ")
            print(f"    {oid}:\n      {body or '(blank)'}")


if __name__ == "__main__":
    main()
