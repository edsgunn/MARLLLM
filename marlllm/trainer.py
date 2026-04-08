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

        all_params = list(p for a in agents.values() for p in a.parameters())
        self.optimizer = AdamW(all_params, lr=config.lr)

        self._start_time = time.time()
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
            # 1. Collect episodes
            episode_results: list[str] = []
            episode_correct_counts: list[int] = []
            trace_traj = None
            trace_info: dict = {}
            for _ in range(self.config.episodes_per_iter):
                traj, ep_info = self._collect_episode()
                self.store.store(traj)
                if trace_traj is None:
                    trace_traj = traj
                    trace_info = ep_info
                if "result" in ep_info:
                    episode_results.append(ep_info["result"])
                if "correct_count" in ep_info:
                    episode_correct_counts.append(ep_info["correct_count"])

            # 2. Build batch
            trajectories = self.store.sample(batch_size=self.config.episodes_per_iter)
            if not trajectories:
                self._logger.warning("iter %d: empty trajectory store, skipping update", iteration)
                continue

            pad_id = self._pad_token_id()
            batch = RolloutBatch.from_trajectories(
                trajectories, self.agent_index, pad_id
            ).to(self.device)

            # 3. Forward pass + losses for each agent
            _model_dtype = next(next(iter(self.agents.values()))._backbone.parameters()).dtype
            total_loss = torch.tensor(0.0, device=self.device, dtype=_model_dtype)
            all_metrics: dict[str, float] = {}

            for agent_id, agent in self.agents.items():
                agent.train_mode()
                logits, values = agent.evaluate(batch.input_ids, batch.attention_mask)
                ref_logits = agent.evaluate_ref(batch.input_ids, batch.attention_mask)
                agent_idx = self.agent_index[agent_id]

                loss_val, metrics = self.loss.compute_loss(
                    logits=logits,
                    values=values,
                    input_ids=batch.input_ids,
                    token_type_mask=batch.token_type_mask,
                    agent_id_mask=batch.agent_id_mask,
                    target_agent_idx=agent_idx,
                    config=self.config,
                    ref_logits=ref_logits,
                )
                total_loss = total_loss + loss_val
                all_metrics.update({f"{agent_id}/{k}": v for k, v in metrics.items()})

            # 4. Gradient update
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            # 5. Clear on-policy store
            self.store.clear()

            # 6. Log and checkpoint
            all_metrics["total_loss"] = total_loss.item()
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
                for ctx_id in self.agents:
                    contexts[ctx_id].extend(obs_ids)
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

        # Prepend the primary agent's prompt as PAD-typed tokens.
        # This ensures the training forward pass conditions on the same prefix
        # as the rollout, keeping log-probs consistent across rollout and update.
        primary = list(self.agents.keys())[0]
        prompt_ids = prompt_ids_per_agent[primary]
        if prompt_ids:
            prompt_step = EpisodeStep(
                agent_id=primary,
                token_ids=prompt_ids,
                token_type=TokenType.PAD,
                log_probs=[],
                info={},
            )
            episode_history = [prompt_step] + episode_history

        traj = self.tokeniser.build_trajectory(
            episode_history=episode_history,
            agent_ids_present=list(self.agents.keys()),
        )
        return traj, episode_info

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

        Each line shows the token type, the decoded text, and the raw token ID(s),
        so you can see exactly what the agent produced and what the env returned.
        """
        tok = list(self.agents.values())[0].tokenizer
        traces_dir = Path(self.config.output_dir) / "traces"
        traces_dir.mkdir(exist_ok=True)
        path = traces_dir / f"iter_{iteration:06d}.txt"

        lines: list[str] = [
            f"=== Iteration {iteration} — episode trace ===",
            f"result:        {ep_info.get('result', 'unknown')}",
            f"correct_count: {ep_info.get('correct_count', '?')}",
            f"episode_length:{ep_info.get('episode_length', '?')}",
            "",
        ]

        type_labels = {0: "PROMPT", 1: "OBS   ", 2: "ACT   "}
        for step in traj.steps:
            label = type_labels.get(int(step.token_type), "??????")
            decoded = tok.decode(step.token_ids, skip_special_tokens=False)
            ids_str = " ".join(str(t) for t in step.token_ids)
            lines.append(f"[{label}] {decoded!r:30s}  ids=[{ids_str}]")

        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")

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
