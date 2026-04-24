"""
Population-based CCSM training.

A population of named agents (e.g. Marcus, Sophia, Viktor, Chen) is trained
by randomly pairing them for each episode.  Each agent owns a private character
prompt and either an independent model or a LoRA adapter on a shared backbone.

Key differences from the two-agent Trainer
-------------------------------------------
- `population` dict maps character name → Agent (instead of env-role → Agent).
- Episode histories tag steps with the *character name*, not the env role.
- The training update iterates over population members: for each member it
  builds a RolloutBatch from the episodes they participated in, prepends their
  own prompt as the PAD context, and computes loss masked to their token positions.
  This gives every agent a forward pass conditioned on their own character prompt.
- A single shared optimizer covers all agents' parameters, updated once per
  iteration after all per-agent backward passes have accumulated gradients.

Pairing strategies
------------------
- ``random_no_self``  : random pair, different characters (default)
- ``random_with_self``: random pair, self-play allowed
- ``round_robin``     : cycle through all ordered (A,B) pairs deterministically
"""
from __future__ import annotations

import copy
import dataclasses
import itertools
import json
import logging
import random
import time
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW

from marlllm.agent import Agent
from marlllm.config import TrainingConfig
from marlllm.loss import Loss
from marlllm.store import TrajectoryStore
from marlllm.tokeniser import Tokeniser
from marlllm.trace_utils import format_trace
from marlllm.types import EpisodeStep, RolloutBatch, TokenType, Trajectory


# ── Raw episode record returned by _collect_episodes_batched ─────────────────
# (history, ep_info, pairing, prompt_ids_per_name)
_RawEpisode = tuple[
    list[EpisodeStep],   # steps tagged with population-member names
    dict,                # ep_info from env
    tuple[str, str],     # (name_for_agent_0_role, name_for_agent_1_role)
    dict[str, list[int]] # prompt token IDs keyed by population-member name
]


class PopulationTrainer:
    """
    Trains a population of agents via random pairings in a two-player env.

    Parameters
    ----------
    population:
        Mapping of character name → Agent.  The order of keys defines the
        integer indices used in agent_id_mask tensors.
    env:
        A PettingZoo AEC environment with exactly two possible agents
        (``agent_0`` and ``agent_1``).
    loss, tokeniser, store, config:
        Same roles as in the standard Trainer.
    pairing_strategy:
        How to pair agents each episode.  One of
        ``"random_no_self"`` | ``"random_with_self"`` | ``"round_robin"``.
    """

    def __init__(
        self,
        population: dict[str, Agent],
        env,
        loss: Loss,
        tokeniser: Tokeniser,
        store: TrajectoryStore,
        config: TrainingConfig,
        pairing_strategy: str = "random_no_self",
    ) -> None:
        if len(population) < 1:
            raise ValueError("population must have at least one member")
        env_agents = getattr(env, "possible_agents", [])
        if len(env_agents) != 2:
            raise ValueError(
                f"PopulationTrainer requires an env with exactly 2 possible_agents, "
                f"got {env_agents}"
            )

        self.population = population
        self.env = env
        self.loss = loss
        self.tokeniser = tokeniser
        self.config = config
        self.pairing_strategy = pairing_strategy

        self.device = torch.device(config.device)
        # Integer index for every population member — used in agent_id_mask
        self.agent_index: dict[str, int] = {
            name: i for i, name in enumerate(population)
        }

        # Single optimizer over all unique parameters across the population.
        # For LoRA shared base this covers all adapters + value heads (tiny).
        # For independent models it covers all separate model weights.
        seen_ids: set[int] = set()
        all_params: list = []
        for agent in population.values():
            for p in agent.parameters():
                if id(p) not in seen_ids:
                    seen_ids.add(id(p))
                    all_params.append(p)
        self.optimizer = AdamW(all_params, lr=config.lr)

        self._start_time = time.time()
        self._rng_counter = config.seed
        self._rng = random.Random(config.seed)

        # Round-robin cycle state: shuffle all ordered (A,B) pairs once, then repeat
        names = list(population.keys())
        if pairing_strategy == "round_robin":
            all_pairs = [(a, b) for a in names for b in names if a != b]
            self._rng.shuffle(all_pairs)
            self._rr_cycle = itertools.cycle(all_pairs)
        else:
            self._rr_cycle = None

        self._setup_output_dir()
        self._logger = self._setup_logging()
        self._metrics_path = Path(config.output_dir) / "metrics.jsonl"

    # ------------------------------------------------------------------ #
    # Main training loop                                                   #
    # ------------------------------------------------------------------ #

    def train(self, start_iteration: int = 1) -> None:
        pop_names = list(self.population.keys())
        self._logger.info(
            "Population training: %d members %s | pairing=%s | iters=%d",
            len(pop_names), pop_names, self.pairing_strategy,
            self.config.num_iterations,
        )
        self._logger.info("Output directory: %s", self.config.output_dir)

        for iteration in range(start_iteration, self.config.num_iterations + 1):

            # ── 1. Collect episodes ───────────────────────────────────────
            raw_episodes: list[_RawEpisode] = self._collect_episodes_batched(
                self.config.episodes_per_iter
            )

            # ── 2. Per-agent update ───────────────────────────────────────
            # For each population member:
            #   a) gather the episodes they participated in
            #   b) build RolloutBatch conditioned on their own prompt
            #   c) compute + accumulate loss
            # Then a single optimizer.step() after all agents have contributed.

            self.optimizer.zero_grad()
            total_loss_scalar = 0.0
            all_metrics: dict[str, float] = {}
            pad_id = self._pad_token_id()

            for pop_name, agent in self.population.items():
                agent_episodes = [
                    ep for ep in raw_episodes if pop_name in ep[2]
                ]
                if not agent_episodes:
                    continue

                trajectories = self._build_agent_trajectories(
                    agent_episodes, pop_name
                )
                if not trajectories:
                    continue

                batch = RolloutBatch.from_trajectories(
                    trajectories, self.agent_index, pad_id
                )

                # Gradient accumulation within this agent's mini-batch
                K = batch.input_ids.shape[0]
                grad_accum = max(1, min(self.config.grad_accum_steps, K))
                micro_size = max(1, (K + grad_accum - 1) // grad_accum)
                micro_starts = list(range(0, K, micro_size))
                num_micros = len(micro_starts)

                agent.train_mode()
                dev = agent.device

                for micro_start in micro_starts:
                    micro_end = min(micro_start + micro_size, K)

                    mb_input_ids = batch.input_ids[micro_start:micro_end].to(dev)
                    mb_attn     = batch.attention_mask[micro_start:micro_end].to(dev)
                    mb_types    = batch.token_type_mask[micro_start:micro_end].to(dev)
                    mb_agents   = batch.agent_id_mask[micro_start:micro_end].to(dev)

                    # Trim to actual length within this micro-batch
                    actual_len = int(mb_attn.sum(dim=1).max())
                    if actual_len < mb_input_ids.shape[1]:
                        mb_input_ids = mb_input_ids[:, :actual_len]
                        mb_attn      = mb_attn[:, :actual_len]
                        mb_types     = mb_types[:, :actual_len]
                        mb_agents    = mb_agents[:, :actual_len]

                    logits, values = agent.evaluate(mb_input_ids, mb_attn)
                    ref_logits = agent.evaluate_ref(mb_input_ids, mb_attn)
                    agent_idx = self.agent_index[pop_name]

                    loss_val, metrics = self.loss.compute_loss(
                        logits=logits,
                        values=values,
                        input_ids=mb_input_ids,
                        token_type_mask=mb_types,
                        agent_id_mask=mb_agents,
                        target_agent_idx=agent_idx,
                        config=self.config,
                        ref_logits=ref_logits,
                    )

                    # Scale so the sum over micros equals the full per-agent loss,
                    # and then divide by population size so the total gradient
                    # magnitude is independent of population size.
                    scale = 1.0 / (num_micros * len(self.population))
                    (loss_val * scale).backward()
                    total_loss_scalar += loss_val.item() / num_micros

                    for k, v in metrics.items():
                        key = f"{pop_name}/{k}"
                        all_metrics[key] = all_metrics.get(key, 0.0) + v / num_micros

            # ── 3. Optimizer step ─────────────────────────────────────────
            self.optimizer.step()

            # ── 4. Episode outcome stats ──────────────────────────────────
            episode_results = [ep[1].get("result", "") for ep in raw_episodes]
            n_eps = len(episode_results)
            all_metrics["total_loss"]  = total_loss_scalar
            all_metrics["iteration"]   = iteration
            all_metrics["wall_time"]   = time.time() - self._start_time
            all_metrics["n_episodes"]  = n_eps
            if n_eps:
                all_metrics["success_rate"] = episode_results.count("success") / n_eps

            # Pairing coverage: how many unique pairings appeared this iter
            pairings = [ep[2] for ep in raw_episodes]
            unique_pairings = len(set(pairings))
            all_metrics["unique_pairings"] = unique_pairings

            # ── 5. Log and checkpoint ─────────────────────────────────────
            if iteration % self.config.log_every == 0:
                self._log_metrics(iteration, all_metrics)
                # Write a trace for one episode that had both agents present
                for raw_ep in raw_episodes:
                    hist, ep_info, pairing, pids = raw_ep
                    if hist:
                        traj = self._make_trace_trajectory(hist, pairing, pids)
                        self._write_trace(iteration, traj, ep_info, pairing)
                        break

            self._write_metrics_jsonl(all_metrics)

            if iteration % self.config.checkpoint_every == 0:
                self.save_checkpoint(iteration)

        self.save_checkpoint(self.config.num_iterations, tag="final")
        self._logger.info("Training complete.")

    # ------------------------------------------------------------------ #
    # Episode collection                                                   #
    # ------------------------------------------------------------------ #

    def _sample_pairs(self, n: int) -> list[tuple[str, str]]:
        """Generate n (name_for_agent0, name_for_agent1) pairings."""
        names = list(self.population.keys())
        pairs: list[tuple[str, str]] = []

        if self.pairing_strategy == "round_robin":
            assert self._rr_cycle is not None
            for _ in range(n):
                pairs.append(next(self._rr_cycle))

        elif self.pairing_strategy == "random_with_self":
            for _ in range(n):
                a = self._rng.choice(names)
                b = self._rng.choice(names)
                pairs.append((a, b))

        else:  # random_no_self (default)
            if len(names) < 2:
                # Only one agent → forced self-play
                for _ in range(n):
                    pairs.append((names[0], names[0]))
            else:
                for _ in range(n):
                    a, b = self._rng.sample(names, 2)
                    pairs.append((a, b))

        return pairs

    def _collect_episodes_batched(self, n: int) -> list[_RawEpisode]:
        """
        Run n episodes in parallel, each with a sampled pairing.

        Steps are tagged with population-member names as agent_id (not the
        env roles agent_0/agent_1).  Observations are broadcast to both
        population members' context windows (matching Trainer behaviour).
        """
        pairings = self._sample_pairs(n)

        # env_role_to_name[k] maps "agent_0"/"agent_1" → population name for ep k
        env_role_to_name: list[dict[str, str]] = [
            {"agent_0": p[0], "agent_1": p[1]} for p in pairings
        ]

        # Create n deep-copied envs
        envs: list = []
        for k in range(n):
            env_k = copy.deepcopy(self.env)
            if hasattr(env_k, "_tok"):
                env_k._tok = self.env._tok
            env_k.reset(seed=self._rng_counter)
            self._rng_counter += 1
            envs.append(env_k)

        # Per-episode bookkeeping
        histories:    list[list[EpisodeStep]]      = [[] for _ in range(n)]
        # contexts: env_role → token_ids (used during act() for context construction)
        contexts:     list[dict[str, list[int]]]   = [{} for _ in range(n)]
        prompt_ids:   list[dict[str, list[int]]]   = [{} for _ in range(n)]
        ep_infos:     list[dict]                   = [{} for _ in range(n)]
        token_counts: list[int]                    = [0] * n
        active:       list[bool]                   = [True] * n

        # Initialise contexts with character prompts for each population member
        for k in range(n):
            for env_role, pop_name in env_role_to_name[k].items():
                pt = self.config.character_prompts.get(pop_name, "")
                pids = self.tokeniser.encode_prompt(pt)
                prompt_ids[k][pop_name] = pids
                contexts[k][env_role] = list(pids)

        n_tokens = getattr(self.env, "action_token_budget", 1)

        while any(active):
            # Groups: population_name → list of env indices needing that agent to act
            act_groups: dict[str, list[int]] = {}
            null_indices: list[int] = []

            for k in range(n):
                if not active[k]:
                    continue
                env = envs[k]
                if not env.agents:
                    active[k] = False
                    continue

                env_role = env.agent_selection
                pop_name = env_role_to_name[k][env_role]
                obs, _rew, term, trunc, _info = env.last()

                obs_ids = self.tokeniser.encode_observation(obs)
                if obs_ids:
                    # OBS step tagged with the population member whose turn it is
                    histories[k].append(EpisodeStep(
                        agent_id=pop_name,
                        token_ids=obs_ids,
                        token_type=TokenType.OBS,
                        log_probs=[],
                        info={},
                    ))
                    # Broadcast to both env roles' contexts (Trainer convention)
                    for role_ctx in contexts[k]:
                        contexts[k][role_ctx].extend(obs_ids)
                    token_counts[k] += len(obs_ids)

                if term or trunc:
                    for aid in env.possible_agents:
                        if aid in env.infos:
                            ep_infos[k] = env.infos[aid]
                            break
                    null_indices.append(k)
                elif token_counts[k] >= self.config.max_episode_tokens:
                    null_indices.append(k)
                elif pop_name in self.population:
                    act_groups.setdefault(pop_name, []).append(k)
                else:
                    null_indices.append(k)

            for k in null_indices:
                envs[k].step(None)
                if not envs[k].agents:
                    active[k] = False

            # Batched act: one GPU call per population member per tick
            for pop_name, env_indices in act_groups.items():
                agent = self.population[pop_name]
                agent.eval_mode()

                # Each env may have this pop_name in a different role
                batch_contexts = [
                    contexts[k][envs[k].agent_selection]
                    for k in env_indices
                ]
                with torch.no_grad():
                    batch_ids, batch_lps = agent.act_batch(
                        contexts=batch_contexts,
                        n_tokens=n_tokens,
                        temperature=self.config.temperature,
                    )

                for j, k in enumerate(env_indices):
                    act_ids = batch_ids[j]
                    act_lps = batch_lps[j]
                    env_role = envs[k].agent_selection

                    histories[k].append(EpisodeStep(
                        agent_id=pop_name,   # character name, not env role
                        token_ids=act_ids,
                        token_type=TokenType.ACT,
                        log_probs=act_lps,
                        info={},
                    ))
                    contexts[k][env_role].extend(act_ids)
                    token_counts[k] += len(act_ids)
                    envs[k].step(act_ids)
                    if not envs[k].agents:
                        active[k] = False

        return [
            (histories[k], ep_infos[k], pairings[k], prompt_ids[k])
            for k in range(n)
        ]

    # ------------------------------------------------------------------ #
    # Trajectory building for training                                    #
    # ------------------------------------------------------------------ #

    def _build_agent_trajectories(
        self,
        agent_episodes: list[_RawEpisode],
        pop_name: str,
    ) -> list[Trajectory]:
        """
        Build one Trajectory per episode from the perspective of `pop_name`.

        Each trajectory:
        - Is prefixed with pop_name's own character prompt as PAD tokens,
          so the training forward pass conditions on the same context the
          agent saw during rollout.
        - Contains all OBS steps from the episode (broadcast by Trainer
          convention) plus all steps belonging to any agent (both ACT and
          OBS are labelled with their originating population member).
        - Loss computation downstream will mask to positions where
          agent_id_mask == self.agent_index[pop_name].
        """
        trajectories: list[Trajectory] = []
        for history, _ep_info, _pairing, pids in agent_episodes:
            padded: list[EpisodeStep] = []
            prompt = pids.get(pop_name, [])
            if prompt:
                padded.append(EpisodeStep(
                    agent_id=pop_name,
                    token_ids=prompt,
                    token_type=TokenType.PAD,
                    log_probs=[],
                    info={},
                ))
            padded.extend(history)
            traj = self.tokeniser.build_trajectory(
                episode_history=padded,
                agent_ids_present=list(self.population.keys()),
            )
            trajectories.append(traj)
        return trajectories

    def _make_trace_trajectory(
        self,
        history: list[EpisodeStep],
        pairing: tuple[str, str],
        pids: dict[str, list[int]],
    ) -> Trajectory:
        """Build a trace trajectory prefixed with the first agent's prompt."""
        name_0 = pairing[0]
        padded = []
        prompt = pids.get(name_0, [])
        if prompt:
            padded.append(EpisodeStep(
                agent_id=name_0,
                token_ids=prompt,
                token_type=TokenType.PAD,
                log_probs=[],
                info={},
            ))
        padded.extend(history)
        return self.tokeniser.build_trajectory(
            episode_history=padded,
            agent_ids_present=list(self.population.keys()),
        )

    # ------------------------------------------------------------------ #
    # Checkpointing                                                        #
    # ------------------------------------------------------------------ #

    def save_checkpoint(self, iteration: int, tag: str | None = None) -> None:
        ckpt_dir = Path(self.config.output_dir) / "checkpoints"
        agent_states = {
            name: {
                "backbone": agent._backbone.state_dict(),
                "value_head": agent._value_head.state_dict(),
            }
            for name, agent in self.population.items()
        }
        payload = {
            "iteration": iteration,
            "agent_states": agent_states,
            "optimizer_state": self.optimizer.state_dict(),
            "rng_state": torch.get_rng_state(),
            "config": dataclasses.asdict(self.config),
            "population_names": list(self.population.keys()),
        }
        fname = f"iter_{iteration:06d}.pt" if tag is None else f"{tag}.pt"
        path = ckpt_dir / fname
        torch.save(payload, path)
        latest = ckpt_dir / "latest.pt"
        if latest.exists() or latest.is_symlink():
            latest.unlink()
        try:
            latest.symlink_to(fname)
        except (OSError, NotImplementedError):
            torch.save(payload, latest)
        self._logger.info("Checkpoint saved: %s", path)

    def load_checkpoint(self, path: str) -> int:
        payload = torch.load(path, map_location=self.device)
        agent_states = payload.get("agent_states", {})
        loaded_ids: set[int] = set()
        for name, agent in self.population.items():
            if id(agent) in loaded_ids:
                continue
            if name in agent_states:
                states = agent_states[name]
                agent._backbone.load_state_dict(states["backbone"])
                agent._value_head.load_state_dict(states["value_head"])
                loaded_ids.add(id(agent))
        self.optimizer.load_state_dict(payload["optimizer_state"])
        torch.set_rng_state(payload["rng_state"].cpu())
        iteration = payload["iteration"]
        self._logger.info(
            "Checkpoint loaded from %s (iteration %d)", path, iteration
        )
        return iteration

    # ------------------------------------------------------------------ #
    # Logging helpers                                                      #
    # ------------------------------------------------------------------ #

    def _write_trace(
        self,
        iteration: int,
        traj: Trajectory,
        ep_info: dict,
        pairing: tuple[str, str],
    ) -> None:
        tok = list(self.population.values())[0].tokenizer
        traces_dir = Path(self.config.output_dir) / "traces"
        traces_dir.mkdir(exist_ok=True)
        path = traces_dir / f"iter_{iteration:06d}.txt"
        ep_info_ext = dict(ep_info)
        ep_info_ext["pairing"] = f"{pairing[0]} vs {pairing[1]}"
        text = format_trace(
            iteration=iteration,
            traj=traj,
            ep_info=ep_info_ext,
            tokenizer=tok,
            character_prompts=self.config.character_prompts,
        )
        with open(path, "w") as f:
            f.write(text)

    def _write_metrics_jsonl(self, metrics: dict) -> None:
        with open(self._metrics_path, "a") as f:
            f.write(json.dumps(metrics) + "\n")

    def _log_metrics(self, iteration: int, metrics: dict) -> None:
        elapsed = metrics.get("wall_time", 0.0)
        parts = [
            f"iter {iteration:5d}/{self.config.num_iterations} | {elapsed:7.1f}s",
            f"loss {metrics.get('total_loss', 0.0):.4f}",
            f"success {metrics.get('success_rate', 0.0):.3f}",
            f"pairs {int(metrics.get('unique_pairings', 0))}",
        ]
        for name in self.population:
            ret = metrics.get(f"{name}/mean_return")
            if ret is not None:
                parts.append(f"{name}/ret {ret:.4f}")
        self._logger.info(" | ".join(parts))

    def _pad_token_id(self) -> int:
        primary = list(self.population.values())[0]
        tok = getattr(primary, "tokenizer", None)
        if tok is not None:
            pid = tok.pad_token_id
            if pid is not None:
                return pid
            return tok.eos_token_id
        return 0

    def _setup_output_dir(self) -> None:
        out = Path(self.config.output_dir)
        (out / "checkpoints").mkdir(parents=True, exist_ok=True)
        cfg_path = out / "config.json"
        if not cfg_path.exists():
            with open(cfg_path, "w") as f:
                import dataclasses as _dc
                json.dump(_dc.asdict(self.config), f, indent=2)

    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger(f"marlllm.population.{id(self)}")
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        fmt = logging.Formatter(
            "%(asctime)s %(levelname)-8s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        fh = logging.FileHandler(
            Path(self.config.output_dir) / "train.log", mode="a"
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(ch)
        return logger
