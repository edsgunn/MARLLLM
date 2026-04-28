"""
Smoke test for CCSMTerminalEnv — runs without a real model or GPU.

Uses a tiny stub tokenizer so we can exercise the full env loop without
loading a multi-GB checkpoint. Validates:

  1. Stage 0: obs arrives as token IDs, action budget is 0, episode
     terminates on budget exhaustion without any shell subprocess.

  2. Stage 1 (N=1): reset gives ls output, step with a cat command gives
     file content, gate blocks an unsupported command.

  3. Stage 3.5 (N=2): two agents share scratch. Agent 0 writes a file;
     Agent 1 reads it back and gets the content — demonstrating that other
     agents' actions are visible as normal observations with no special routing.

Run from the project root:
    python -m envs.terminal.smoke_test

Or directly:
    python envs/terminal/smoke_test.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Stub tokenizer — encodes text as UTF-8 bytes, decodes symmetrically.
# Sufficient for testing env flow; not a real language model tokenizer.
# ---------------------------------------------------------------------------

class _StubTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return list(text.encode("utf-8", errors="replace"))

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return bytes(b for b in ids if 0 <= b < 256).decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Corpus fixture — a small directory of text files for testing
# ---------------------------------------------------------------------------

def _make_corpus(root: Path) -> None:
    """Create a minimal corpus: two topic dirs, three files each."""
    topics = {
        "physics": {
            "relativity.txt":     "Special relativity: E = mc^2.\n" * 40,
            "quantum.txt":        "Quantum mechanics: wave-particle duality.\n" * 40,
            "thermodynamics.txt": "Entropy always increases in an isolated system.\n" * 40,
        },
        "biology": {
            "evolution.txt":   "Natural selection acts on heritable variation.\n" * 40,
            "genetics.txt":    "DNA encodes genetic information in codons.\n" * 40,
            "ecology.txt":     "Ecosystems are interconnected webs of organisms.\n" * 40,
        },
    }
    for topic, files in topics.items():
        (root / topic).mkdir(parents=True, exist_ok=True)
        for name, content in files.items():
            (root / topic / name).write_text(content)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_stage_0(corpus: Path, work: Path) -> None:
    print("\n=== Stage 0: pure observation ===")
    from envs.terminal import CCSMTerminalEnv

    tok = _StubTokenizer()
    env = CCSMTerminalEnv(
        tokenizer=tok,
        stage="0",
        corpus_path=corpus,
        token_budget=512,
        files_per_episode=6,
        seed=0,
        work_dir=work / "stage0",
    )

    assert env.action_token_budget == 0, "Stage 0 must have action_token_budget=0"

    env.reset(seed=0)
    assert env.agents, "agents list must be non-empty after reset"
    assert env.agent_selection == "agent_0"

    steps = 0
    while env.agents:
        agent = env.agent_selection
        obs = env.observe(agent)
        assert isinstance(obs, list) and all(isinstance(t, int) for t in obs), \
            "observation must be list[int]"
        assert len(obs) > 0, "Stage 0 observation must be non-empty"
        # Stage 0: action is ignored — pass empty list
        env.step([])
        if env.terminations.get(agent):
            env.step(None)
        steps += 1
        if steps > 100:
            raise RuntimeError("Stage 0 did not terminate within 100 steps")

    print(f"  OK: episode completed in {steps} steps, all obs were list[int]")
    env.close()


def test_stage_1(corpus: Path, work: Path) -> None:
    print("\n=== Stage 1: ls + cat ===")
    from envs.terminal import CCSMTerminalEnv

    tok = _StubTokenizer()
    env = CCSMTerminalEnv(
        tokenizer=tok,
        stage="1",
        corpus_path=corpus,
        token_budget=8192,
        files_per_episode=6,
        action_token_budget=64,
        seed=1,
        work_dir=work / "stage1",
    )

    env.reset(seed=1)
    agent = env.agent_selection

    # Initial observation should be ls output (file names)
    init_obs = env.observe(agent)
    init_text = tok.decode(init_obs)
    print(f"  Initial ls output: {repr(init_text)[:120]}")
    assert len(init_obs) > 0, "Initial observation must be non-empty"

    # Step 1: ls (allowed at Stage 1)
    ls_action = tok.encode("ls")
    env.step(ls_action)
    ls_obs = env.observe(agent)
    ls_text = tok.decode(ls_obs)
    print(f"  ls response: {repr(ls_text)[:120]}")
    assert "$ " in ls_text, "Shell prompt '$ ' must be in ls output"

    # Step 2: try to use grep (not allowed at Stage 1)
    grep_action = tok.encode("grep -r quantum .")
    env.step(grep_action)
    grep_obs = tok.decode(env.observe(agent))
    print(f"  grep (blocked) response: {repr(grep_obs)[:120]}")
    assert "[gate]" in grep_obs, "Gate must block 'grep' at Stage 1"

    # Step 3: cat a file if one exists
    # Parse the ls output to find a file
    file_to_cat = None
    for part in ls_text.replace("$ ", "").split():
        candidate = Path(env._shells[agent].corpus_dir) / part
        if candidate.is_file():
            file_to_cat = part
            break
        # Check in subdirs
        subdirs = list((env._shells[agent].corpus_dir / part).iterdir()) \
            if (env._shells[agent].corpus_dir / part).is_dir() else []
        if subdirs:
            rel = subdirs[0].relative_to(env._shells[agent].corpus_dir)
            file_to_cat = str(rel)
            break

    if file_to_cat:
        cat_action = tok.encode(f"cat {file_to_cat}")
        env.step(cat_action)
        cat_obs = tok.decode(env.observe(agent))
        print(f"  cat '{file_to_cat}' response: {repr(cat_obs)[:120]}")
        assert len(cat_obs) > 4, "cat should return non-trivial content"

    print("  OK: gate blocks non-Stage-1 commands, allowed commands work")
    env.close()


def test_stage_3_5_shared_scratch(corpus: Path, work: Path) -> None:
    print("\n=== Stage 3.5: N=2, shared scratch, embodiment test ===")
    from envs.terminal import CCSMTerminalEnv

    tok = _StubTokenizer()
    env = CCSMTerminalEnv(
        tokenizer=tok,
        stage="3.5",
        corpus_path=corpus,
        n_agents=2,
        token_budget=32768,
        files_per_episode=6,
        action_token_budget=128,
        seed=2,
        work_dir=work / "stage35",
    )

    env.reset(seed=2)

    # Agents alternate: agent_0 then agent_1.
    # We'll do one step for each and then test the core claim.

    assert "agent_0" in env.agents
    assert "agent_1" in env.agents

    # --- Agent 0's turn: write a message to shared scratch ---
    assert env.agent_selection == "agent_0"
    msg = "hello from agent_0"
    write_cmd = f'echo "{msg}" > "$SCRATCH/comms.txt"'
    env.step(tok.encode(write_cmd))
    write_obs = tok.decode(env.observe("agent_0"))
    print(f"  agent_0 write response: {repr(write_obs)[:80]}")
    # echo with redirect produces no stdout; just a prompt
    assert "$ " in write_obs, "Prompt must follow write command"

    # --- Agent 1's turn: read the file agent_0 wrote ---
    assert env.agent_selection == "agent_1"
    read_cmd = 'cat "$SCRATCH/comms.txt"'
    env.step(tok.encode(read_cmd))
    read_obs = tok.decode(env.observe("agent_1"))
    print(f"  agent_1 cat response: {repr(read_obs)[:120]}")

    # THE CORE ASSERTION: agent_1 sees agent_0's text as a normal observation.
    # No special routing. No provenance. Just file content from the shell.
    assert msg in read_obs, (
        f"Agent 1 must observe Agent 0's message via shared filesystem.\n"
        f"Expected {msg!r} in observation, got: {read_obs!r}"
    )

    print("  OK: Agent 1 observed Agent 0's filesystem write as a normal observation.")
    print("      No special inter-agent routing. The filesystem IS the channel.")
    env.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    with tempfile.TemporaryDirectory(prefix="ccsm_smoke_") as tmpdir:
        corpus = Path(tmpdir) / "corpus"
        work   = Path(tmpdir) / "work"
        _make_corpus(corpus)

        try:
            test_stage_0(corpus, work)
            test_stage_1(corpus, work)
            test_stage_3_5_shared_scratch(corpus, work)
        except Exception as e:
            print(f"\nFAIL: {e}")
            raise

    print("\n=== All smoke tests passed ===")


if __name__ == "__main__":
    # Allow running as a script from the project root
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    main()
