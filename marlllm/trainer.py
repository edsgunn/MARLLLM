"""
Trainer: orchestrates the CCSM training loop (spec §3).

Episode Collection → Loss Computation → Parameter Update, repeated
for config.num_iterations iterations.

Key design notes:
- env.last() is used (not env.observe()) so terminal observations are
  always collected before the Trainer calls env.step(None) on dead agents.
- The character prompt is prepended to every trajectory as PAD-typed tokens
  so the training forward pass (evaluate()) sees the same prefix context
  as the rollout (act()), making log-probs consistent.
- Logging writes to both stdout and a file; metrics are emitted as JSONL
  (one record per log interval) for easy downstream analysis.
- Checkpoints are saved every config.checkpoint_every iterations and at the
  end of training, enabling SLURM job restarts without re-running from scratch.
"""
from __future__ import annotations

import copy
import dataclasses
import json
import logging
import os
import time
from pathlib import Path

import torch
from torch.optim import AdamW

from marlllm.agent import Agent
from marlllm.config import TrainingConfig
from marlllm.loss import Loss
from marlllm.store import TrajectoryStore
from marlllm.tokeniser import Tokeniser
from marlllm.types import EpisodeStep, RolloutBatch, TokenType


class Trainer:
    """
    Drives the CCSM training loop against a PettingZoo AEC environment.

    The environment must expose the standard PettingZoo AEC interface plus:
        env.action_token_budget: int  — tokens sampled per turn
    """

    def __init__(
        self,
        agents: dict[str, Agent],
        env,                    # PettingZoo AECEnv
        loss: Loss,
        tokeniser: Tokeniser,
        store: TrajectoryStore,
        config: TrainingConfig,
    ) -> None:
        self.agents = agents
        self.env = env
        self.loss = loss
        self.tokeniser = tokeniser
        self.store = store
        self.config = config

        self.device = torch.device(config.device)
        self.agent_index = {aid: i for i, aid in enumerate(agents)}

        # Deduplicate by parameter identity so shared-weight agents (where both
        # dict entries point to the same IndependentAgent) don't double-count.
        seen_ids: set[int] = set()
        all_params: list = []
        for a in agents.values():
            for p in a.parameters():
                if id(p) not in seen_ids:
                    seen_ids.add(id(p))
                    all_params.append(p)
        self.optimizer = AdamW(all_params, lr=config.lr)

        self._start_time = time.time()
        self._rng_counter = config.seed  # advances per episode for distinct env scenarios
        self._setup_output_dir()
        self._logger = self._setup_logging()
        self._metrics_path = Path(config.output_dir) / "metrics.jsonl"

    # ------------------------------------------------------------------ #
    # Main training loop                                                   #
    # ------------------------------------------------------------------ #

    def train(self, start_iteration: int = 1) -> None:
        self._logger.info(
            "Starting training: iterations=%d, episodes_per_iter=%d, device=%s",
            self.config.num_iterations, self.config.episodes_per_iter, self.config.device,
        )
        self._logger.info("Output directory: %s", self.config.output_dir)

        for iteration in range(start_iteration, self.config.num_iterations + 1):
            # 1. Collect episodes — all N episodes run simultaneously so every
            # agent turn is a single batched forward pass instead of N serial ones.
            # Each episode yields per-agent trajectory views (other agents' steps
            # are masked as PAD in each view) so every agent trains only on the
            # tokens it actually observed during rollout.
            episode_results: list[str] = []
            episode_correct_counts: list[int] = []
            trace_traj = None
            trace_info: dict = {}
            per_agent_trajs: dict[str, list] = {aid: [] for aid in self.agents}

            for agent_traj_dict, ep_info in self._collect_episodes_batched(
                self.config.episodes_per_iter
            ):
                for aid, traj in agent_traj_dict.items():
                    if aid == "_combined":
                        continue
                    per_agent_trajs[aid].append(traj)
                if trace_traj is None:
                    # Use the combined-view trajectory (stored under the first
                    # agent key) for the environment-overview trace only.
                    trace_traj = agent_traj_dict.get("_combined") or next(
                        iter(agent_traj_dict.values())
                    )
                    trace_info = ep_info
                if "result" in ep_info:
                    episode_results.append(ep_info["result"])
                if "correct_count" in ep_info:
                    episode_correct_counts.append(ep_info["correct_count"])

            # 2. Per-agent forward pass + losses with gradient accumulation.
            # Each agent trains on its own N trajectories (the episodes it
            # participated in, from its own observation perspective).  Both
            # agents' backward passes accumulate into the shared optimizer
            # before the step, matching the previous gradient magnitude.
            pad_id = self._pad_token_id()
            self.optimizer.zero_grad()
            total_loss_scalar = 0.0
            all_metrics: dict[str, float] = {}

            for agent_id, agent in self.agents.items():
                trajs = per_agent_trajs[agent_id]
                if not trajs:
                    continue

                batch = RolloutBatch.from_trajectories(
                    trajs, self.agent_index, pad_id
                )

                K = batch.input_ids.shape[0]
                grad_accum = max(1, min(self.config.grad_accum_steps, K))
                micro_size = max(1, (K + grad_accum - 1) // grad_accum)
                micro_starts = list(range(0, K, micro_size))
                num_micros = len(micro_starts)

                agent.train_mode()
                dev = agent.device
                agent_idx = self.agent_index[agent_id]

                for micro_start in micro_starts:
                    micro_end = min(micro_start + micro_size, K)

                    mb_input_ids      = batch.input_ids[micro_start:micro_end].to(dev)
                    mb_attention_mask = batch.attention_mask[micro_start:micro_end].to(dev)
                    mb_token_type     = batch.token_type_mask[micro_start:micro_end].to(dev)
                    mb_agent_id       = batch.agent_id_mask[micro_start:micro_end].to(dev)

                    actual_len = int(mb_attention_mask.sum(dim=1).max())
                    if actual_len < mb_input_ids.shape[1]:
                        mb_input_ids      = mb_input_ids[:, :actual_len]
                        mb_attention_mask = mb_attention_mask[:, :actual_len]
                        mb_token_type     = mb_token_type[:, :actual_len]
                        mb_agent_id       = mb_agent_id[:, :actual_len]

                    logits, values = agent.evaluate(mb_input_ids, mb_attention_mask)
                    ref_logits = agent.evaluate_ref(mb_input_ids, mb_attention_mask)

                    loss_val, metrics = self.loss.compute_loss(
                        logits=logits,
                        values=values,
                        input_ids=mb_input_ids,
                        token_type_mask=mb_token_type,
                        agent_id_mask=mb_agent_id,
                        target_agent_idx=agent_idx,
                        config=self.config,
                        ref_logits=ref_logits,
                    )

                    (loss_val / num_micros).backward()
                    total_loss_scalar += loss_val.item() / num_micros

                    for k, v in metrics.items():
                        key = f"{agent_id}/{k}"
                        all_metrics[key] = all_metrics.get(key, 0.0) + v / num_micros

            # 3. Gradient update
            self.optimizer.step()

            # 4. Clear on-policy store (kept for API compatibility)
            self.store.clear()

            # 6. Log and checkpoint
            all_metrics["total_loss"] = total_loss_scalar
            all_metrics["iteration"] = iteration
            all_metrics["wall_time"] = time.time() - self._start_time

            # Episode outcome stats
            n_eps = len(episode_results)
            if n_eps > 0:
                all_metrics["success_rate"] = episode_results.count("success") / n_eps
                all_metrics["wrong_rate"] = episode_results.count("wrong") / n_eps
            if episode_correct_counts:
                all_metrics["mean_correct"] = sum(episode_correct_counts) / len(episode_correct_counts)

            if iteration % self.config.log_every == 0:
                self._log_metrics(iteration, all_metrics)
                if trace_traj is not None:
                    self._write_trace(iteration, trace_traj, trace_info)

            self._write_metrics_jsonl(all_metrics)

            if iteration % self.config.checkpoint_every == 0:
                self.save_checkpoint(iteration)

        self.save_checkpoint(self.config.num_iterations, tag="final")
        self._logger.info("Training complete.")

    # ------------------------------------------------------------------ #
    # Episode collection                                                   #
    # ------------------------------------------------------------------ #

    def _collect_episode(self) -> tuple:
        """
        Run one episode via the PettingZoo AEC API and return a Trajectory.

        The character prompt is prepended as a PAD-typed step so that the
        training forward pass (evaluate()) conditions on the same context as
        the rollout (act()). PAD-typed tokens are excluded from all losses.

        env.last() is used to retrieve each observation, which guarantees the
        terminal observation (error token, success echo) is collected before
        env.step(None) removes the agent from self.agents.
        """
        episode_history: list[EpisodeStep] = []
        token_count = 0
        episode_info: dict = {}

        self.env.reset()

        # Per-agent contexts start with the character prompt.
        # These are used for act() calls only; they are not stored directly
        # in the trajectory — instead the prompt is prepended as PAD tokens below.
        contexts: dict[str, list[int]] = {}
        prompt_ids_per_agent: dict[str, list[int]] = {}
        for agent_id in self.agents:
            prompt_text = self.config.character_prompts.get(agent_id, "")
            pids = self.tokeniser.encode_prompt(prompt_text)
            prompt_ids_per_agent[agent_id] = pids
            contexts[agent_id] = list(pids)

        for agent_id in self.env.agent_iter():
            # env.last() returns (obs, reward, terminated, truncated, info)
            # for the current agent_selection. Always call before checking
            # termination so we don't miss the terminal observation.
            obs, _rew, term, trunc, _info = self.env.last()

            obs_ids = self.tokeniser.encode_observation(obs)
            if obs_ids:
                obs_step = EpisodeStep(
                    agent_id=agent_id,
                    token_ids=obs_ids,
                    token_type=TokenType.OBS,
                    log_probs=[],
                    info={},
                )
                episode_history.append(obs_step)
                contexts[agent_id].extend(obs_ids)
                token_count += len(obs_ids)

            if term or trunc:
                self.env.step(None)
                continue

            if token_count >= self.config.max_episode_tokens:
                self.env.step(None)
                continue

            if agent_id in self.agents:
                agent = self.agents[agent_id]
                agent.eval_mode()
                n_tokens = getattr(self.env, "action_token_budget", 1)

                with torch.no_grad():
                    act_ids, act_lps = agent.act(
                        context_token_ids=contexts[agent_id],
                        n_tokens=n_tokens,
                        temperature=self.config.temperature,
                    )

                act_step = EpisodeStep(
                    agent_id=agent_id,
                    token_ids=act_ids,
                    token_type=TokenType.ACT,
                    log_probs=act_lps,
                    info={},
                )
                episode_history.append(act_step)
                contexts[agent_id].extend(act_ids)
                token_count += len(act_ids)

                self.env.step(act_ids)
            else:
                self.env.step(None)

        # Capture final episode info (result, correct_count) from env
        for aid in self.env.possible_agents:
            if aid in self.env.infos:
                episode_info = self.env.infos[aid]
                break

        agent_traj_dict = self._build_agent_views(episode_history, prompt_ids_per_agent)
        return agent_traj_dict, episode_info

    # ------------------------------------------------------------------ #
    # Trajectory utilities                                                 #
    # ------------------------------------------------------------------ #

    def _build_agent_views(
        self,
        history: list[EpisodeStep],
        prompt_ids: dict[str, list[int]],
    ) -> dict[str, "Trajectory"]:
        """
        Build one training trajectory per agent from a shared episode history.

        Each agent's view contains ONLY the tokens that agent actually had in
        its context window during rollout:

          [PROMPT][OBS_1][ACT_1][OBS_2][ACT_2] ...

        where every OBS and ACT step belongs to this agent.  Steps belonging
        to other agents are excluded entirely — not remapped to PAD — so the
        transformer never attends to tokens the agent did not observe.  This
        matches the rollout context exactly: ``contexts[agent_id]`` during
        collection is built from the same set of steps.

        The PettingZoo AEC contract guarantees that ``env.last()`` delivers
        each observation only to the currently-selected agent.  Other agents'
        utterances reach this agent only after being processed by the
        environment into a new OBS step addressed to this agent.

        The dictionary also contains a ``"_combined"`` key holding a
        trajectory with all steps in their original types, used only for
        writing environment-overview traces (not for training).
        """
        agent_ids = list(self.agents.keys())
        result: dict[str, "Trajectory"] = {}

        for aid in agent_ids:
            steps: list[EpisodeStep] = []
            pids = prompt_ids.get(aid, [])
            if pids:
                steps.append(EpisodeStep(
                    agent_id=aid,
                    token_ids=pids,
                    token_type=TokenType.PAD,
                    log_probs=[],
                    info={},
                ))
            for step in history:
                # Include only steps that belong to this agent.
                # Other agents' OBS and ACT are not in this agent's context.
                if step.agent_id == aid:
                    steps.append(step)
            result[aid] = self.tokeniser.build_trajectory(
                episode_history=steps,
                agent_ids_present=agent_ids,
            )

        # Combined view for traces: all steps with original types.
        primary = agent_ids[0]
        combined_steps: list[EpisodeStep] = []
        pids = prompt_ids.get(primary, [])
        if pids:
            combined_steps.append(EpisodeStep(
                agent_id=primary,
                token_ids=pids,
                token_type=TokenType.PAD,
                log_probs=[],
                info={},
            ))
        combined_steps.extend(history)
        result["_combined"] = self.tokeniser.build_trajectory(
            episode_history=combined_steps,
            agent_ids_present=agent_ids,
        )

        return result

    # ------------------------------------------------------------------ #
    # Batched episode collection                                           #
    # ------------------------------------------------------------------ #

    def _collect_episodes_batched(self, n: int) -> list[tuple]:
        """
        Run n episodes simultaneously, batching all per-agent turns into a
        single (n, T) forward pass instead of n serial (1, T) calls.

        At each "tick" we:
          1. Inspect every active env's current agent and observation.
          2. Group envs by which agent should act.
          3. Call agent.act_batch() once per agent with all their contexts —
             one GPU call instead of n GPU calls.
          4. Step each env with its result.

        This collapses ~(n × dialogue_turns × token_budget) serial forward
        passes into ~(dialogue_turns × token_budget) batched ones, yielding
        much higher GPU arithmetic intensity.
        """
        # --- Create n independent env copies ---
        # Deep-copy so each env has its own Python state, but share the
        # tokenizer object (read-only, no need to duplicate vocab tables).
        envs: list = []
        for k in range(n):
            env_k = copy.deepcopy(self.env)
            if hasattr(env_k, "_tok"):
                env_k._tok = self.env._tok  # re-share tokenizer reference
            # Give each env a unique-but-reproducible seed so they generate
            # distinct scenarios rather than n identical copies.
            env_k.reset(seed=self._rng_counter)
            self._rng_counter += 1
            envs.append(env_k)

        # Per-env bookkeeping
        histories:     list[list[EpisodeStep]]      = [[] for _ in range(n)]
        contexts:      list[dict[str, list[int]]]   = [{} for _ in range(n)]
        prompt_ids:    list[dict[str, list[int]]]   = [{} for _ in range(n)]
        ep_infos:      list[dict]                   = [{} for _ in range(n)]
        token_counts:  list[int]                    = [0] * n
        active:        list[bool]                   = [True] * n

        for k in range(n):
            for agent_id in self.agents:
                pt = self.config.character_prompts.get(agent_id, "")
                pids = self.tokeniser.encode_prompt(pt)
                prompt_ids[k][agent_id] = pids
                contexts[k][agent_id] = list(pids)

        # --- Step all envs until every one is done ---
        while any(active):
            # Groups for this tick
            act_groups:  dict[str, list[int]] = {}  # agent_id -> [env indices]
            null_indices: list[int] = []

            for k in range(n):
                if not active[k]:
                    continue
                env = envs[k]
                if not env.agents:
                    active[k] = False
                    continue

                agent_id = env.agent_selection
                obs, _rew, term, trunc, _info = env.last()

                # Append observation to this episode's history
                obs_ids = self.tokeniser.encode_observation(obs)
                if obs_ids:
                    histories[k].append(EpisodeStep(
                        agent_id=agent_id,
                        token_ids=obs_ids,
                        token_type=TokenType.OBS,
                        log_probs=[],
                        info={},
                    ))
                    contexts[k][agent_id].extend(obs_ids)
                    token_counts[k] += len(obs_ids)

                if term or trunc:
                    # Capture final info before stepping with None
                    for aid in env.possible_agents:
                        if aid in env.infos:
                            ep_infos[k] = env.infos[aid]
                            break
                    null_indices.append(k)
                elif token_counts[k] >= self.config.max_episode_tokens:
                    null_indices.append(k)
                elif agent_id in self.agents:
                    act_groups.setdefault(agent_id, []).append(k)
                else:
                    null_indices.append(k)

            # Execute null steps
            for k in null_indices:
                envs[k].step(None)
                if not envs[k].agents:
                    active[k] = False

            # Execute batched act() — one GPU call per agent instead of one per env
            for agent_id, env_indices in act_groups.items():
                agent = self.agents[agent_id]
                agent.eval_mode()
                n_tokens = getattr(self.env, "action_token_budget", 1)

                batch_contexts = [contexts[k][agent_id] for k in env_indices]
                with torch.no_grad():
                    batch_ids, batch_lps = agent.act_batch(
                        contexts=batch_contexts,
                        n_tokens=n_tokens,
                        temperature=self.config.temperature,
                    )

                for j, k in enumerate(env_indices):
                    act_ids = batch_ids[j]
                    act_lps = batch_lps[j]
                    histories[k].append(EpisodeStep(
                        agent_id=agent_id,
                        token_ids=act_ids,
                        token_type=TokenType.ACT,
                        log_probs=act_lps,
                        info={},
                    ))
                    contexts[k][agent_id].extend(act_ids)
                    token_counts[k] += len(act_ids)
                    envs[k].step(act_ids)
                    if not envs[k].agents:
                        active[k] = False

        # --- Build per-agent trajectory views ---
        # Each agent's view contains only the steps it actually observed:
        # - OBS steps addressed to this agent: kept as OBS (contribute to L_perc)
        # - ACT steps by this agent: kept as ACT (contribute to L_act)
        # - All other agents' OBS and ACT steps: remapped to PAD (excluded from
        #   loss and return computation — this agent never saw them directly)
        # This correctly reflects the PettingZoo AEC guarantee that env.last()
        # delivers each observation only to the currently-selected agent.
        results = []
        for k in range(n):
            agent_traj_dict = self._build_agent_views(histories[k], prompt_ids[k])
            results.append((agent_traj_dict, ep_infos[k]))
        return results

    # ------------------------------------------------------------------ #
    # Checkpointing                                                        #
    # ------------------------------------------------------------------ #

    def save_checkpoint(self, iteration: int, tag: str | None = None) -> None:
        """
        Save model weights, value head weights, optimizer state, and RNG state.

        Writes two files:
          checkpoints/iter_{iteration:06d}.pt  — named checkpoint
          checkpoints/latest.pt                — always the most recent
        """
        ckpt_dir = Path(self.config.output_dir) / "checkpoints"
        agent_states = {
            aid: {
                "backbone": agent._backbone.state_dict(),
                "value_head": agent._value_head.state_dict(),
            }
            for aid, agent in self.agents.items()
        }
        payload = {
            "iteration": iteration,
            "agent_states": agent_states,
            "optimizer_state": self.optimizer.state_dict(),
            "rng_state": torch.get_rng_state(),
            "config": dataclasses.asdict(self.config),
        }
        fname = f"iter_{iteration:06d}.pt" if tag is None else f"{tag}.pt"
        path = ckpt_dir / fname
        torch.save(payload, path)
        # Overwrite latest symlink (use copy on platforms without symlinks)
        latest = ckpt_dir / "latest.pt"
        if latest.exists() or latest.is_symlink():
            latest.unlink()
        try:
            latest.symlink_to(fname)
        except (OSError, NotImplementedError):
            torch.save(payload, latest)

        self._logger.info("Checkpoint saved: %s", path)

    def load_checkpoint(self, path: str) -> int:
        """
        Restore model, optimizer, and RNG state from a checkpoint.
        Returns the iteration number stored in the checkpoint.
        """
        payload = torch.load(path, map_location=self.device)
        for aid, agent in self.agents.items():
            states = payload["agent_states"][aid]
            agent._backbone.load_state_dict(states["backbone"])
            agent._value_head.load_state_dict(states["value_head"])
        self.optimizer.load_state_dict(payload["optimizer_state"])
        torch.set_rng_state(payload["rng_state"].cpu())
        iteration = payload["iteration"]
        self._logger.info("Checkpoint loaded from %s (iteration %d)", path, iteration)
        return iteration

    # ------------------------------------------------------------------ #
    # Logging helpers                                                      #
    # ------------------------------------------------------------------ #

    def _setup_output_dir(self) -> None:
        out = Path(self.config.output_dir)
        (out / "checkpoints").mkdir(parents=True, exist_ok=True)
        cfg_path = out / "config.json"
        if not cfg_path.exists():
            with open(cfg_path, "w") as f:
                json.dump(dataclasses.asdict(self.config), f, indent=2)

    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger(f"marlllm.{id(self)}")
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

        fmt = logging.Formatter(
            "%(asctime)s %(levelname)-8s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # File handler: full debug log, always flushed (important for SLURM)
        fh = logging.FileHandler(Path(self.config.output_dir) / "train.log", mode="a")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        fh.terminator = "\n"

        # Console handler: INFO only
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(fmt)

        logger.addHandler(fh)
        logger.addHandler(ch)
        return logger

    def _log_metrics(self, iteration: int, metrics: dict) -> None:
        elapsed = metrics.get("wall_time", 0.0)
        parts = [f"iter {iteration:5d}/{self.config.num_iterations} | {elapsed:7.1f}s"]
        for key in [
            "total_loss",
            "agent_0/mean_surprise",
            "agent_0/mean_return",
            "agent_0/mean_advantage",
            "agent_0/entropy",
            "agent_0/perc_loss",
            "agent_0/act_loss",
            "agent_0/value_loss",
            "agent_0/kl",
            "success_rate",
            "mean_correct",
        ]:
            if key in metrics:
                short = key.split("/")[-1]
                parts.append(f"{short} {metrics[key]:.4f}")
        self._logger.info(" | ".join(parts))

    def _write_trace(self, iteration: int, traj, ep_info: dict) -> None:
        """
        Decode and save the first episode of this iteration as a human-readable
        trace file under <output_dir>/traces/iter_XXXXXX.txt.

        The trace has two sections:
          1. ENVIRONMENT OVERVIEW — chronological event log showing which agent
             sent what to whom, including action→observation routing arrows.
          2. PER-AGENT CONTEXT — each agent's full context window with broadcast
             observations annotated.
        """
        from marlllm.trace_utils import format_trace

        tok = list(self.agents.values())[0].tokenizer
        traces_dir = Path(self.config.output_dir) / "traces"
        traces_dir.mkdir(exist_ok=True)
        path = traces_dir / f"iter_{iteration:06d}.txt"

        text = format_trace(
            iteration=iteration,
            traj=traj,
            ep_info=ep_info,
            tokenizer=tok,
            character_prompts=self.config.character_prompts,
        )
        with open(path, "w") as f:
            f.write(text)

    def _write_metrics_jsonl(self, metrics: dict) -> None:
        with open(self._metrics_path, "a") as f:
            f.write(json.dumps(metrics) + "\n")

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _pad_token_id(self) -> int:
        primary = list(self.agents.values())[0]
        tok = getattr(primary, "tokenizer", None)
        if tok is not None:
            pid = tok.pad_token_id
            if pid is not None:
                return pid
            return tok.eos_token_id
        return 0
