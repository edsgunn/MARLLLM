"""
Stage definitions for CCSMTerminalEnv.

A stage is defined by exactly three things:
  1. The command set enabled (what reaches the shell)
  2. Whether the corpus view is shared across agents
  3. Whether the scratch directory is shared across agents

Nothing else changes between stages — same model, same loss, same training
loop, same PettingZoo AEC interface. The N=1 single-agent case is stage "0"
through "3" with default_n_agents=1; the multi-agent case is "3.5"+ with
default_n_agents≥2. The environment code path is identical either way.

Note on `less`, `vi`, `nano`: these are interactive commands that require a
pseudo-terminal and would block the sentinel-based output reader. They are
excluded from all stages and intercepted by the gate with a clear message.
The agent can always use `cat`, `head`, or `tail` to read files.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class StageConfig:
    """
    Immutable specification for one training stage.

    corpus_shared:
        True → all agents are symlinked into the same staging directory and
        see the same file selection for the episode. Appropriate at Stage 3.5+
        where agents are meant to work with the same reference material.
        False → each agent gets an independently rotated corpus view.

    scratch_shared:
        True → all agents' shells use the same writable scratch directory.
        This is the ONLY mechanism by which agents observe each other. There
        are no special inter-agent channels: Agent A writes a file, Agent B
        reads it with `cat`. From B's perspective it's a normal filesystem
        observation — σ_t=1, just like any other shell output.
        False → each agent has private scratch; no inter-agent visibility.
    """
    name: str
    allowed_commands: frozenset
    corpus_shared: bool
    scratch_shared: bool
    default_n_agents: int
    description: str


# ---------------------------------------------------------------------------
# Stage 0 — Pure observation (~0% action tokens)
# ---------------------------------------------------------------------------
# No commands. The agent cannot act. Corpus text is streamed directly into
# the observation channel. All tokens are σ_t=1. Training loss should be
# indistinguishable from continued pretraining on the same corpus.
#
# Purpose: verify that the σ_t routing, perception loss, and REINFORCE path
# are wired correctly before action tokens exist. If loss curves diverge from
# a plain pretraining baseline, there is a framework bug.

STAGE_0 = StageConfig(
    name="0",
    allowed_commands=frozenset(),
    corpus_shared=False,
    scratch_shared=False,
    default_n_agents=1,
    description=(
        "Pure observation. No shell. Corpus text streamed as observations. "
        "All tokens σ_t=1. action_token_budget is set to 0 automatically."
    ),
)

# ---------------------------------------------------------------------------
# Stage 1 — Choose-your-own-adventure (~1–3% action tokens)
# ---------------------------------------------------------------------------
# `ls` is the only branching action; `cat` deterministically streams a file.
# File and directory names are informative so the action distribution is
# conditional on observed structure: the prompt-to-action chain is short
# and well-defined.
#
# Purpose: introduce action tokens at minimum density and validate that
# REINFORCE gradients at sparse action positions are stable.

STAGE_1 = StageConfig(
    name="1",
    allowed_commands=frozenset({"ls", "cat"}),
    corpus_shared=False,
    scratch_shared=False,
    default_n_agents=1,
    description=(
        "ls + cat only. ~1–3% action tokens. "
        "Validates sparse REINFORCE stability."
    ),
)

# ---------------------------------------------------------------------------
# Stage 2 — Read-only navigation (~5–15% action tokens)
# ---------------------------------------------------------------------------
# Full read-only toolset. Corpus should be large enough that exhaustive
# `cat`-ing is infeasible and structured enough that `grep`/`find` are useful.
#
# Purpose: first stage where character prompts should produce measurably
# different command sequences. Flat prompt-sensitivity here means the
# behavioural specification mechanism is broken.

STAGE_2 = StageConfig(
    name="2",
    allowed_commands=frozenset({
        "ls", "cat", "head", "tail", "wc", "grep", "find", "pwd", "cd",
    }),
    corpus_shared=False,
    scratch_shared=False,
    default_n_agents=1,
    description=(
        "Read-only navigation. ~5–15% action tokens. "
        "First test of prompt-sensitivity."
    ),
)

# ---------------------------------------------------------------------------
# Stage 3 — Write access in scratch (~15–35% action tokens)
# ---------------------------------------------------------------------------
# Agents can create, modify, and delete files within their private scratch
# directory. The corpus is still read-only. `rm` is guarded by the shell
# setup to reject paths outside $SCRATCH (belt-and-suspenders alongside gate).
#
# Purpose: first stage where actions have persistent state effects within
# an episode. Self-referential KL dynamics begin here. Watch for the
# self-consistent-but-useless attractor (agent repeatedly touches the same
# file — detectable as flat filesystem entropy over the episode).

STAGE_3 = StageConfig(
    name="3",
    allowed_commands=frozenset({
        "ls", "cat", "head", "tail", "wc", "grep", "find", "pwd", "cd",
        "echo", "mkdir", "touch", "mv", "cp", "rm",
        "sed", "awk", "tee", "python3",
    }),
    corpus_shared=False,
    scratch_shared=False,
    default_n_agents=1,
    description=(
        "Write access in scratch. ~15–35% action tokens. "
        "rm/mv/cp gated to $SCRATCH. python3 available (no network/pip)."
    ),
)

# ---------------------------------------------------------------------------
# Stage 3.5 — Multi-agent with shared scratch (~25–45% action tokens)
# ---------------------------------------------------------------------------
# The load-bearing experimental stage for the framework's core claim.
#
# scratch_shared=True is the only change from Stage 3. The shared scratch is
# the sole inter-agent communication channel. Agent A's `echo ... > notes.txt`
# is a filesystem write; Agent B's `cat notes.txt` is a normal shell read.
# From B's perspective, A's text arrives as σ_t=1 observation tokens —
# indistinguishable from any other file content. This falls out correctly
# by construction: the env returns whatever the shell produces, and the
# training loop labels all env output σ_t=1.
#
# corpus_shared=True means all agents see the same file selection. This is
# appropriate because coordination tasks require a shared reference corpus.
#
# IMPORTANT: the σ_t provenance subtlety from the spec is a non-issue in
# this implementation because we don't track provenance. Agent A's tokens
# that appear in Agent B's `cat` output are simply part of the shell's
# stdout, returned as a list[int] observation — always σ_t=1 for B.
# No cross-agent token labelling is needed.

STAGE_3_5 = StageConfig(
    name="3.5",
    allowed_commands=frozenset({
        "ls", "cat", "head", "tail", "wc", "grep", "find", "pwd", "cd",
        "echo", "mkdir", "touch", "mv", "cp", "rm",
        "sed", "awk", "tee", "python3",
    }),
    corpus_shared=True,
    scratch_shared=True,
    default_n_agents=2,
    description=(
        "Multi-agent. Shared scratch is the only inter-agent channel. "
        "Other agents are indistinguishable from the environment. "
        "~25–45% action tokens."
    ),
)

# ---------------------------------------------------------------------------
# Stage 4 — Open computer use (~35%+ action tokens)
# ---------------------------------------------------------------------------
# Network access added. Run N=1 ablations to isolate multi-agent contribution.

STAGE_4 = StageConfig(
    name="4",
    allowed_commands=frozenset({
        "ls", "cat", "head", "tail", "wc", "grep", "find", "pwd", "cd",
        "echo", "mkdir", "touch", "mv", "cp", "rm",
        "sed", "awk", "tee", "python3",
        "curl", "wget",
    }),
    corpus_shared=True,
    scratch_shared=True,
    default_n_agents=2,
    description=(
        "Open computer use with network access. ~35%+ action tokens. "
        "Run N=1 ablation alongside N>1 to separate contributions."
    ),
)

STAGES: dict[str, StageConfig] = {
    "0":   STAGE_0,
    "1":   STAGE_1,
    "2":   STAGE_2,
    "3":   STAGE_3,
    "3.5": STAGE_3_5,
    "4":   STAGE_4,
}
