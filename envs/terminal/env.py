"""
CCSMTerminalEnv: PettingZoo AEC environment for terminal-based CCSM training.

Core philosophical commitment — embodiment and embeddedness
-----------------------------------------------------------
Each agent sees the world exactly as if it were in a single-agent
environment. Other agents are not special: they are parts of the
environment, observable only through the effects they leave on the shared
filesystem. There is no labelled "other agent" channel, no explicit message
passing, no routing logic that says "this token came from agent B." The agent
experiences other agents the same way it experiences weather: indirectly,
through observations of a shared world it did not fully control.

Implementation consequence: the environment does not track token provenance.
Agent A writes to $SHARED/notes.txt. Agent B runs `cat $SHARED/notes.txt`.
Agent B's observation is the file's content — whatever the shell produced,
returned as a list[int] with σ_t=1. The training loop marks it as an
observation token exactly as it would mark any other environment output.
No cross-agent labelling, no provenance map, no special cases. The
correctness of σ_t assignment follows from the architecture, not from
careful bookkeeping.

This is why the environment has no message bus, no explicit communication
primitive, and no `shared_state` dict. The shared scratch directory IS the
communication medium. Its contents are observable through normal shell
commands (ls, cat, grep, find). This is not a limitation — it is the design.

σ_t assignment
--------------
All tokens returned by observe() are σ_t=1 (observation).
All tokens the model generated for the action are σ_t=0 (action).
The training loop owns σ_t=0 labelling; this env never touches it.

Rewards
-------
Always 0.0. CCSM computes intrinsic return inside the training loop from
observation surprise. The rewards dict is kept for PettingZoo compatibility
so standard wrappers (vectorisation, logging, recording) work without
modification.

AEC vs Parallel
---------------
AEC is used (not ParallelEnv) because:
  - Terminal interaction is sequential at the command level. One command
    runs, produces output, then the next command runs. Parallel step
    semantics would require either serialising commands internally (hiding
    the sequencing from the interface) or truly concurrent shell writes
    (fragile and hard to attribute for debugging).
  - In the multi-agent case, agents naturally take turns. Agent B's
    commands see the filesystem effects of all previous agents' commands,
    which is the desired behaviour for emergent coordination.
  - Consistent with the existing codebase (CountingEnv, DealOrNoDealEnv).

The N=1 single-agent case is the default for Stages 0–3. The N>1 case is
the default for Stage 3.5+. The code path is identical.

Observation format
------------------
Observations are list[int] (tokenizer token IDs), matching the convention
in CountingEnv and DealOrNoDealEnv. The tokenizer is injected at construction
and used to encode shell output and decode action token IDs back to text.

The observation at each step is:
  <shell_output><newline>$<space>

The trailing "$ " is the shell prompt. It tells the model that the
environment is ready for the next command and serves as a step boundary
token — something the model will learn to predict as it learns the
environment's structure.

Stage 0 observations are raw corpus text chunks (no shell, no prompt).
"""

from __future__ import annotations

import random
import shutil
import tempfile
from pathlib import Path
from typing import Any

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from transformers import PreTrainedTokenizerBase

from .corpus import CorpusManager
from .shell import AgentShell
from .stages import STAGES, StageConfig

# Approximate chars per token. Used for Stage-0 chunk sizing only;
# exact token counts come from the tokenizer after encoding.
_CHARS_PER_TOKEN = 4

# Stage-0 chunk size in tokens (approximate). One "step" delivers this
# many tokens of corpus text. Roughly matches a moderate-length paragraph.
_STAGE0_CHUNK_TOKENS = 256


class CCSMTerminalEnv(AECEnv):
    """
    Terminal environment for CCSM training.

    Parameters
    ----------
    tokenizer:
        HuggingFace tokenizer. Encodes shell output → list[int] observations;
        decodes action list[int] → command string sent to the shell.
    stage:
        One of "0", "1", "2", "3", "3.5", "4".
    corpus_path:
        Path to the corpus directory on disk. Must contain at least one
        readable text file somewhere in its subtree.
    n_agents:
        Number of agents. Defaults to the stage's recommended count.
        N=1 at Stage 3.5 is a valid ablation (single agent with shared
        scratch, which is just Stage 3 with a different scratch location).
    character_prompts:
        Pool of character prompts to sample from at episode start.
        One is drawn per agent per episode and placed in infos["character_prompt"].
        The training loop is responsible for prepending it to the agent's
        context as σ_t=1 tokens. If None, agents receive no character prompt.
    token_budget:
        Approximate maximum observation tokens per agent per episode.
        Computed from encoded observation lengths; episode terminates when
        cumulative tokens exceed this. Tune per stage: lower budgets
        (2048–4096) for early stages, higher (8192–16384) for Stage 3+.
    action_token_budget:
        Maximum tokens the agent may generate per step (the command length
        limit). Read by the Trainer. Stage 0 overrides this to 0.
        128 tokens is enough for most shell commands including long greps.
    files_per_episode:
        Number of corpus files exposed per episode per agent (or shared,
        for Stage 3.5+). Balance between variety and manageability.
    overlap_fraction:
        Fraction of corpus files carried over between episodes.
        See corpus.py for tuning guidance.
    command_timeout:
        Seconds before a shell command is killed and [timeout:...] emitted.
        30s handles most legitimate commands; lower if agents run `yes`.
    max_output_chars:
        Hard cap on shell output characters per step before truncation.
    seed:
        RNG seed for reproducible corpus rotation and prompt sampling.
    work_dir:
        Parent directory for staging and scratch subdirs. In Slurm, set
        this to $SCRATCH or a node-local fast filesystem (e.g. /tmp or
        $LOCAL_SCRATCH) for low-latency episode resets. If None, a
        temporary directory is created and cleaned up on close().
    """

    metadata = {"render_modes": [], "name": "ccsm_terminal_v0"}

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        stage: str,
        corpus_path: str | Path,
        n_agents: int | None = None,
        character_prompts: list[str] | None = None,
        token_budget: int = 4096,
        action_token_budget: int = 128,
        files_per_episode: int = 50,
        overlap_fraction: float = 0.3,
        command_timeout: int = 30,
        max_output_chars: int = 8192,
        seed: int | None = None,
        work_dir: str | Path | None = None,
    ) -> None:
        super().__init__()

        if stage not in STAGES:
            raise ValueError(f"Unknown stage {stage!r}. Valid: {sorted(STAGES)}")

        self._cfg: StageConfig = STAGES[stage]
        self._tok = tokenizer
        self._corpus_path = Path(corpus_path).resolve()
        self._n_agents = n_agents if n_agents is not None else self._cfg.default_n_agents
        self._character_prompts = character_prompts or []
        self._token_budget = token_budget
        self._files_per_episode = files_per_episode
        self._overlap = overlap_fraction
        self._command_timeout = command_timeout
        self._max_output_chars = max_output_chars
        self._rng = random.Random(seed)

        # Stage 0: agent produces no action tokens (pure observation).
        # The Trainer reads this attribute to know how many tokens to generate.
        self.action_token_budget: int = 0 if stage == "0" else action_token_budget

        # Work directory. In Slurm point this at a fast local filesystem.
        self._owned_work_dir = work_dir is None
        self._work_dir = (
            Path(tempfile.mkdtemp(prefix="ccsm_env_"))
            if work_dir is None
            else Path(work_dir)
        )
        self._work_dir.mkdir(parents=True, exist_ok=True)

        # Stable agent IDs, consistent across episodes.
        self.possible_agents: list[str] = [f"agent_{i}" for i in range(self._n_agents)]

        # AEC state — all populated in reset().
        self.agents: list[str] = []
        self._selector: agent_selector.AgentSelector | None = None
        self.agent_selection: str = ""

        self._shells: dict[str, AgentShell] = {}
        self._corpus_mgrs: dict[str, CorpusManager] = {}
        self._shared_scratch: Path | None = None

        # Per-agent episode accounting.
        self._pending_obs: dict[str, list[int]] = {}
        self._tokens_used: dict[str, int] = {}
        self._cumulative_rewards: dict[str, float] = {}
        self._terminations: dict[str, bool] = {}
        self._truncations: dict[str, bool] = {}
        self._infos: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # PettingZoo AEC API
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None, options: dict | None = None) -> None:
        """
        Start a new episode for all agents.

        Corpus files are rotated (fresh symlinked staging dirs built).
        Shells are restarted with clean working directories and scratch.
        Initial observation for each agent is the output of `ls` (Stage 1+)
        or the first chunk of corpus text (Stage 0).

        Character prompts (if any) are placed in infos["character_prompt"]
        and NOT in the observation token stream. The training loop prepends
        them to the agent's context as σ_t=1 tokens — they are observations
        in the CCSM sense (text that specifies the agent's character), but
        they are not shell output and do not need to go through the env step.
        """
        if seed is not None:
            self._rng = random.Random(seed)

        for shell in self._shells.values():
            shell.close()
        self._shells.clear()

        self.agents = list(self.possible_agents)
        self._selector = agent_selector.AgentSelector(self.agents)
        self.agent_selection = self._selector.next()

        # Shared scratch for Stage 3.5+. This single directory is mounted
        # into all agents' shells as their $HOME and $SCRATCH. When Agent A
        # writes a file here, Agent B can read it with `cat` — their only
        # window into each other's actions.
        if self._cfg.scratch_shared:
            self._shared_scratch = self._work_dir / "shared_scratch"
            self._shared_scratch.mkdir(parents=True, exist_ok=True)
        else:
            self._shared_scratch = None

        # Shared corpus: one CorpusManager, one staging dir, all agents start
        # in the same directory. Per-agent corpus: independent managers and
        # independently rotated staging dirs.
        shared_corpus_mgr: CorpusManager | None = None
        if self._cfg.corpus_shared:
            shared_corpus_mgr = CorpusManager(
                corpus_root=self._corpus_path,
                staging_dir=self._work_dir / "shared_staging",
                files_per_episode=self._files_per_episode,
                overlap_fraction=self._overlap,
                rng=random.Random(self._rng.randint(0, 2**31)),
            )
            shared_corpus_mgr.rotate()

        for agent_id in self.agents:
            # Corpus assignment
            if self._cfg.corpus_shared and shared_corpus_mgr is not None:
                corpus_mgr = shared_corpus_mgr
            else:
                corpus_mgr = CorpusManager(
                    corpus_root=self._corpus_path,
                    staging_dir=self._work_dir / f"{agent_id}_staging",
                    files_per_episode=self._files_per_episode,
                    overlap_fraction=self._overlap,
                    rng=random.Random(self._rng.randint(0, 2**31)),
                )
                corpus_mgr.rotate()
            self._corpus_mgrs[agent_id] = corpus_mgr

            # Scratch assignment
            if self._cfg.scratch_shared and self._shared_scratch is not None:
                scratch_dir = self._shared_scratch
            else:
                scratch_dir = self._work_dir / f"{agent_id}_scratch"
                scratch_dir.mkdir(parents=True, exist_ok=True)

            # Shell setup (skipped for Stage 0 — no commands, no subprocess overhead)
            if self._cfg.allowed_commands:
                shell = AgentShell(
                    agent_id=agent_id,
                    corpus_dir=corpus_mgr.staging_dir,
                    scratch_dir=scratch_dir,
                    allowed_commands=self._cfg.allowed_commands,
                    stage_name=self._cfg.name,
                    timeout_seconds=self._command_timeout,
                    max_output_chars=self._max_output_chars,
                )
                shell.start()
                self._shells[agent_id] = shell
                # Initial observation: show the agent what's in its corpus dir.
                init_text = shell.step("ls")
            else:
                # Stage 0: stream first chunk of corpus text directly.
                init_text = corpus_mgr.read_chunk(_STAGE0_CHUNK_TOKENS * _CHARS_PER_TOKEN)

            char_prompt = (
                self._rng.choice(self._character_prompts)
                if self._character_prompts
                else ""
            )

            obs_tokens = self._enc(init_text)
            self._pending_obs[agent_id] = obs_tokens
            self._tokens_used[agent_id] = len(obs_tokens)
            self._cumulative_rewards[agent_id] = 0.0
            self._terminations[agent_id] = False
            self._truncations[agent_id] = False
            self._infos[agent_id] = {
                "stage": self._cfg.name,
                "character_prompt": char_prompt,
                "tokens_used": len(obs_tokens),
                "token_budget": self._token_budget,
            }

    def observe(self, agent: str) -> list[int]:
        return list(self._pending_obs.get(agent, []))

    def step(self, action: Any) -> None:
        """
        Advance one agent by one command.

        Parameters
        ----------
        action:
            list[int] of token IDs produced by the agent, or None if the
            agent is already terminated (PettingZoo dead-step convention).

        The action token IDs are decoded to a command string and sent to
        the agent's shell. The shell's stdout (observations, σ_t=1) is
        encoded back to token IDs and stored as the agent's next observation.

        Multi-agent note: if scratch_shared is True, the filesystem effects
        of this agent's command are immediately visible to all other agents'
        shells. The next agent to run `ls $SCRATCH` or `cat $SCRATCH/...`
        will see those effects as normal filesystem observations. No routing,
        no labelling, no special handling — exactly as if a file had appeared
        from any other source.
        """
        agent = self.agent_selection

        if self._terminations[agent] or self._truncations[agent]:
            self._was_dead_step(action)
            return

        if self._cfg.allowed_commands:
            command = self._decode_action(action)
            obs_text = self._shells[agent].step(command)
        else:
            # Stage 0: action is ignored; emit the next corpus text chunk.
            obs_text = self._corpus_mgrs[agent].read_chunk(
                _STAGE0_CHUNK_TOKENS * _CHARS_PER_TOKEN
            )

        obs_tokens = self._enc(obs_text)
        self._pending_obs[agent] = obs_tokens
        self._tokens_used[agent] += len(obs_tokens)
        self._infos[agent]["tokens_used"] = self._tokens_used[agent]
        self._cumulative_rewards[agent] = 0.0

        if self._tokens_used[agent] >= self._token_budget:
            self._terminations[agent] = True

        self.agent_selection = self._selector.next()

    # ------------------------------------------------------------------
    # PettingZoo required properties
    # ------------------------------------------------------------------

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

    def observation_space(self, agent: str):
        return None  # token IDs; no gym space needed

    def action_space(self, agent: str):
        return None  # token IDs; no gym space needed

    def render(self) -> None:
        pass

    def close(self) -> None:
        for shell in self._shells.values():
            shell.close()
        self._shells.clear()
        if self._owned_work_dir and self._work_dir.exists():
            shutil.rmtree(self._work_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _enc(self, text: str) -> list[int]:
        return self._tok.encode(text, add_special_tokens=False)

    def _decode_action(self, action: Any) -> str:
        if action is None:
            return ""
        ids = list(action)
        if not ids:
            return ""
        return self._tok.decode(ids, skip_special_tokens=True)

    def _was_dead_step(self, action: Any) -> None:
        """
        Handle step() for an already-terminated agent (PettingZoo convention).

        The Trainer calls observe() to get the terminal observation, then
        calls step(None) to signal that it has consumed the terminal state.
        This removes the agent from self.agents and advances the selector.
        """
        if action is not None:
            raise ValueError("Only None is valid for a terminated agent.")
        agent = self.agent_selection
        self.agents.remove(agent)
        self._cumulative_rewards[agent] = 0.0
        if self.agents:
            self.agent_selection = self._selector.next()
