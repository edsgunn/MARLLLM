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
from marlllm.dialogue import chat_eos_token_ids, verify_special_tokens
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

        # Frozen agents: participate in rollouts but receive no gradient
        # updates. Used for asymmetric (focal vs fixed partner) training —
        # see TrainingConfig.frozen_agents and Phase A §2.4.
        self.frozen_agents: set[str] = set(config.frozen_agents or [])
        for aid in self.frozen_agents:
            if aid not in agents:
                raise ValueError(
                    f"frozen_agents references unknown agent_id {aid!r}; "
                    f"known agents: {list(agents)}"
                )

        # Deduplicate by parameter identity so shared-weight agents (where both
        # dict entries point to the same IndependentAgent) don't double-count.
        # Frozen agents contribute no parameters to the optimizer.
        seen_ids: set[int] = set()
        all_params: list = []
        for aid, a in agents.items():
            if aid in self.frozen_agents:
                continue
            for p in a.parameters():
                if id(p) not in seen_ids:
                    seen_ids.add(id(p))
                    all_params.append(p)
        if not all_params:
            raise ValueError(
                "No trainable parameters: every agent is in frozen_agents."
            )
        self.optimizer = AdamW(all_params, lr=config.lr)

        # Translate perception_agents (string IDs) to int indices once.
        self._perception_source_indices: list[int] | None = None
        if config.perception_agents:
            unknown = [a for a in config.perception_agents if a not in self.agent_index]
            if unknown:
                raise ValueError(
                    f"perception_agents references unknown agent_ids {unknown!r}; "
                    f"known: {list(self.agent_index)}"
                )
            self._perception_source_indices = [
                self.agent_index[a] for a in config.perception_agents
            ]

        # Chat-template EOS ids for this run. Computed once; passed to every
        # act() / act_batch() call so generation halts cleanly on <|im_end|>
        # / <|endoftext|> instead of running to budget and dragging garbage
        # into the next turn's context.
        any_tok = next(iter(agents.values())).tokenizer
        verify_special_tokens(any_tok)
        self._eos_token_ids = chat_eos_token_ids(any_tok)

        self._start_time = time.time()
        self._rng_counter = config.seed  # advances per episode for distinct env scenarios
        self._setup_output_dir()
        self._logger = self._setup_logging()
        self._metrics_path = Path(config.output_dir) / "metrics.jsonl"

        # Live regime tracker. Pure-Python; no GPU work in here. See
        # marlllm/regime_tracking.py for the calibration rationale.
        if getattr(self.config, "regime_tracking_enabled", True):
            from marlllm.regime_tracking import RegimeTracker, RegimeThresholds
            self._regime_tracker = RegimeTracker(
                thresholds=RegimeThresholds(
                    kl_threshold=self.config.regime_kl_threshold,
                    perc_coherent=self.config.regime_perc_coherent,
                    perc_degen=self.config.regime_perc_degen,
                    ent_low=self.config.regime_ent_low,
                    ent_high=self.config.regime_ent_high,
                ),
                window=self.config.regime_slope_window,
            )
            (Path(self.config.output_dir) / "regime").mkdir(exist_ok=True)
        else:
            self._regime_tracker = None

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
            trace_episodes: list[tuple] = []  # (ctx_snapshot, ep_info, env_trace) for trace saving
            per_agent_trajs: dict[str, list] = {aid: [] for aid in self.agents}
            tool_totals: dict[str, float] = {}

            for agent_traj_dict, ep_info, ctx_snapshot, env_trace_k in self._collect_episodes_batched(
                self.config.episodes_per_iter
            ):
                for aid, traj in agent_traj_dict.items():
                    per_agent_trajs[aid].append(traj)
                if len(trace_episodes) < self.config.num_checkpoint_traces:
                    trace_episodes.append((ctx_snapshot, ep_info, env_trace_k))
                if "result" in ep_info:
                    episode_results.append(ep_info["result"])
                if "correct_count" in ep_info:
                    episode_correct_counts.append(ep_info["correct_count"])
                for k, v in ep_info.items():
                    if isinstance(k, str) and k.startswith("tool/"):
                        tool_totals[k] = tool_totals.get(k, 0.0) + float(v)

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
                if agent_id in self.frozen_agents:
                    continue
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

                    last_hidden, values = agent.evaluate_hidden(mb_input_ids, mb_attention_mask)
                    last_hidden_ref = agent.evaluate_hidden_ref(mb_input_ids, mb_attention_mask)

                    loss_val, metrics = self.loss.compute_loss(
                        last_hidden=last_hidden,
                        lm_head=agent.lm_head,
                        values=values,
                        input_ids=mb_input_ids,
                        token_type_mask=mb_token_type,
                        agent_id_mask=mb_agent_id,
                        target_agent_idx=agent_idx,
                        config=self.config,
                        last_hidden_ref=last_hidden_ref,
                        lm_head_ref=agent.lm_head_ref if last_hidden_ref is not None else None,
                        perception_source_indices=self._perception_source_indices,
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
            if tool_totals:
                denom = max(1, n_eps)
                for k, total in tool_totals.items():
                    all_metrics[f"{k}_total"] = total
                    all_metrics[f"{k}_per_ep"] = total / denom

            # Optional Tier-3 token-level signals from the rollouts.
            if (self._regime_tracker is not None
                    and getattr(self.config, "regime_token_signals", False)
                    and trace_episodes):
                try:
                    from marlllm.regime_token_signals import (
                        compute_per_agent_token_signals,
                    )
                    tok = list(self.agents.values())[0].tokenizer
                    # The trace_episodes list is the same _RawEpisode tuple
                    # shape population.py uses; pass it straight through.
                    sigs = compute_per_agent_token_signals(trace_episodes, tok)
                    all_metrics.update(sigs)
                except Exception as exc:  # pragma: no cover - defensive
                    self._logger.warning("token-signal computation failed: %s", exc)

            # Regime classification — must run BEFORE writing the jsonl so the
            # regime/* fields land in the same record.
            if self._regime_tracker is not None:
                additions, alerts = self._regime_tracker.update(
                    iteration, all_metrics
                )
                all_metrics.update(additions)
                for a in alerts:
                    self._logger.warning("regime: %s", a)
                # Pre-collapse checkpoint copy.
                if (additions.get("regime/pre_collapse_trigger")
                        and getattr(self.config, "regime_pre_collapse_checkpoint", True)):
                    from marlllm.regime_tracking import copy_pre_collapse_checkpoint
                    dst = copy_pre_collapse_checkpoint(
                        Path(self.config.output_dir) / "checkpoints",
                        iteration,
                        prev_iteration=iteration - 1,
                    )
                    if dst is not None:
                        self._logger.warning(
                            "regime: saved pre-collapse snapshot to %s", dst
                        )

            if iteration % self.config.log_every == 0:
                self._log_metrics(iteration, all_metrics)
                if self._regime_tracker is not None:
                    self._logger.info(
                        self._regime_tracker.summary_line(iteration, all_metrics)
                    )
                if trace_episodes:
                    ctx_snapshot0, info0, env_trace0 = trace_episodes[0]
                    self._write_trace(iteration, ctx_snapshot0, info0, env_trace0)

            self._write_metrics_jsonl(all_metrics)

            # Live regime plots — periodic, cheap.
            if (self._regime_tracker is not None
                    and iteration % self.config.regime_viz_every == 0):
                regime_dir = Path(self.config.output_dir) / "regime"
                try:
                    self._regime_tracker.render_population_plot(
                        regime_dir / "regime_trajectory.png"
                    )
                    self._regime_tracker.render_agent_strip(
                        regime_dir / "agent_regime_strip.png"
                    )
                except Exception as exc:  # pragma: no cover
                    self._logger.warning("regime viz failed: %s", exc)

            if iteration % self.config.checkpoint_every == 0:
                self.save_checkpoint(iteration)
                self._write_checkpoint_traces(iteration, trace_episodes)

        self.save_checkpoint(self.config.num_iterations, tag="final")
        self._write_checkpoint_traces(self.config.num_iterations, trace_episodes)
        if self._regime_tracker is not None:
            regime_dir = Path(self.config.output_dir) / "regime"
            try:
                self._regime_tracker.render_population_plot(
                    regime_dir / "regime_trajectory.png"
                )
                self._regime_tracker.render_agent_strip(
                    regime_dir / "agent_regime_strip.png"
                )
                self._regime_tracker.write_summary_md(
                    regime_dir / "regime_summary.md"
                )
            except Exception as exc:  # pragma: no cover
                self._logger.warning("end-of-run regime artefacts failed: %s", exc)
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
            formatter = self.agents[agent_id].context_formatter
            contexts[agent_id] = formatter.wrap_prompt(pids)

        for agent_id in self.env.agent_iter():
            # env.last() returns (obs, reward, terminated, truncated, info)
            # for the current agent_selection. Always call before checking
            # termination so we don't miss the terminal observation.
            obs, _rew, term, trunc, info = self.env.last()

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
                formatter = self.agents[agent_id].context_formatter
                contexts[agent_id].extend(formatter.wrap_observation(obs_ids))
                token_count += len(obs_ids)

            if term or trunc:
                self.env.step(None)
                continue

            # Respect the env's must_act flag: if the env signals that the agent
            # must act this turn (e.g. a commit phase), never skip it due to the
            # token budget — otherwise the agent submits an empty action.
            if (token_count >= self.config.max_episode_tokens
                    and not info.get("must_act", False)):
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
                        eos_token_ids=self._eos_token_ids,
                    )

                act_step = EpisodeStep(
                    agent_id=agent_id,
                    token_ids=act_ids,
                    token_type=TokenType.ACT,
                    log_probs=act_lps,
                    info={},
                )
                episode_history.append(act_step)
                formatter = self.agents[agent_id].context_formatter
                contexts[agent_id].extend(formatter.wrap_action(act_ids))
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

        Each agent's trajectory contains only that agent's steps; no combined
        view is needed since traces now use the raw context window token IDs
        captured during rollout.
        """
        agent_ids = list(self.agents.keys())
        result: dict[str, "Trajectory"] = {}

        for aid in agent_ids:
            formatter = self.agents[aid].context_formatter
            steps: list[EpisodeStep] = []
            pids = prompt_ids.get(aid, [])
            if pids:
                prompt_type = (
                    TokenType.OBS if self.config.prompt_as_observation else TokenType.PAD
                )
                steps.append(EpisodeStep(
                    agent_id=aid,
                    token_ids=formatter.wrap_prompt(pids),
                    token_type=prompt_type,
                    log_probs=[],
                    info={},
                ))
            for step in history:
                # Include only steps that belong to this agent, with role
                # markers applied so the training forward pass sees the same
                # formatted context as the rollout act() calls did.
                if step.agent_id == aid:
                    if step.token_type == TokenType.OBS:
                        fids = formatter.wrap_observation(step.token_ids)
                        lps = step.log_probs
                    elif step.token_type == TokenType.ACT:
                        fids = formatter.wrap_action(step.token_ids)
                        # wrap_action may append closure tokens (<|im_end|>\n);
                        # pad log_probs with 0.0 so lengths stay aligned.
                        lps = step.log_probs + [0.0] * (len(fids) - len(step.token_ids))
                    else:
                        fids = step.token_ids
                        lps = step.log_probs
                    steps.append(EpisodeStep(aid, fids, step.token_type, lps, step.info))
            result[aid] = self.tokeniser.build_trajectory(
                episode_history=steps,
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

        _full_ctx = getattr(self.env, 'obs_is_full_context', False)
        for k in range(n):
            for agent_id, agent in self.agents.items():
                if _full_ctx:
                    # Context is provided wholesale by each obs; no prompt init needed.
                    prompt_ids[k][agent_id] = []
                    contexts[k][agent_id] = []
                else:
                    pt = self.config.character_prompts.get(agent_id, "")
                    pids = self.tokeniser.encode_prompt(pt)
                    prompt_ids[k][agent_id] = pids
                    contexts[k][agent_id] = agent.context_formatter.wrap_prompt(pids)
                if hasattr(agent, "reset_history"):
                    agent.reset_history(slot=k)

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
                obs, _rew, term, trunc, info = env.last()

                # Append observation to this episode's history (raw ids) and
                # the formatted version to the context buffer for act().
                obs_ids = self.tokeniser.encode_observation(obs)
                if obs_ids:
                    histories[k].append(EpisodeStep(
                        agent_id=agent_id,
                        token_ids=obs_ids,
                        token_type=TokenType.OBS,
                        log_probs=[],
                        info={},
                    ))
                    if _full_ctx:
                        contexts[k][agent_id] = list(obs_ids)
                    else:
                        formatter = self.agents[agent_id].context_formatter
                        contexts[k][agent_id].extend(formatter.wrap_observation(obs_ids))
                    token_counts[k] += len(obs_ids)
                    if hasattr(self.agents[agent_id], "note_observation"):
                        try:
                            obs_text = self.tokeniser.decode_action(obs_ids) \
                                if hasattr(self.tokeniser, "decode_action") \
                                else self.agents[agent_id].tokenizer.decode(
                                    obs_ids, skip_special_tokens=True)
                        except Exception:
                            obs_text = ""
                        self.agents[agent_id].note_observation(obs_text, slot=k)

                if term or trunc:
                    # Capture final info before stepping with None
                    for aid in env.possible_agents:
                        if aid in env.infos:
                            ep_infos[k] = env.infos[aid]
                            break
                    null_indices.append(k)
                elif (token_counts[k] >= self.config.max_episode_tokens
                      and not info.get("must_act", False)):
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
                        eos_token_ids=self._eos_token_ids,
                    )

                formatter = self.agents[agent_id].context_formatter
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
                    contexts[k][agent_id].extend(formatter.wrap_action(act_ids))
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
        from marlllm.trace_utils import get_env_trace
        context_snapshots = [
            {aid: list(contexts[k][aid]) for aid in self.agents}
            for k in range(n)
        ]
        env_traces_list = [get_env_trace(envs[k]) for k in range(n)]
        results = []
        for k in range(n):
            agent_traj_dict = self._build_agent_views(histories[k], prompt_ids[k])
            results.append((agent_traj_dict, ep_infos[k], context_snapshots[k], env_traces_list[k]))
        return results

    # ------------------------------------------------------------------ #
    # Checkpointing                                                        #
    # ------------------------------------------------------------------ #

    def save_checkpoint(self, iteration: int, tag: str | None = None) -> None:
        """Save per-agent checkpoint directory.

        Layout::
            checkpoints/iter_NNNNNN/
                meta.pt
                <agent_id>.pt   (one per unique agent object)
            checkpoints/latest -> iter_NNNNNN
        """
        from marlllm.checkpoint_utils import save_population_checkpoint

        params_agents = {
            aid: agent for aid, agent in self.agents.items()
            if hasattr(agent, "_backbone") and hasattr(agent, "_value_head")
        }
        ckpt_dir = save_population_checkpoint(
            population=params_agents,
            iteration=iteration,
            optimizer=self.optimizer,
            config_dict=dataclasses.asdict(self.config),
            output_dir=Path(self.config.output_dir),
            tag=tag,
        )
        self._logger.info("Checkpoint saved: %s", ckpt_dir)

    def load_checkpoint(self, path: str) -> int:
        """Restore from a per-agent checkpoint dir or legacy .pt file."""
        from marlllm.checkpoint_utils import load_population_checkpoint

        p = Path(path)
        if p.is_dir():
            params_agents = {
                aid: agent for aid, agent in self.agents.items()
                if hasattr(agent, "_backbone") and hasattr(agent, "_value_head")
            }
            iteration = load_population_checkpoint(
                population=params_agents,
                optimizer=self.optimizer,
                ckpt_dir=p,
                device=self.device,
            )
            self._logger.info("Checkpoint loaded from %s (iteration %d)", p, iteration)
            return iteration

        # Legacy single-file format
        payload = torch.load(p, map_location=self.device)
        for aid, agent in self.agents.items():
            if aid not in payload["agent_states"]:
                continue
            states = payload["agent_states"][aid]
            agent._backbone.load_state_dict(states["backbone"])
            agent._value_head.load_state_dict(states["value_head"])
        self.optimizer.load_state_dict(payload["optimizer_state"])
        torch.set_rng_state(payload["rng_state"].cpu())
        iteration = payload["iteration"]
        self._logger.info("Checkpoint loaded from %s (iteration %d)", p, iteration)
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

    def _write_trace(self, iteration: int, ctx_snapshot: dict, ep_info: dict, env_trace) -> None:
        from marlllm.trace_utils import make_episode_record, write_records_json, write_records_txt

        tok = list(self.agents.values())[0].tokenizer
        traces_dir = Path(self.config.output_dir) / "traces"
        traces_dir.mkdir(exist_ok=True)

        record = make_episode_record(
            episode_idx=0,
            agent_context_tokens=ctx_snapshot,
            tokenizer=tok,
            env_trace=env_trace or ep_info or None,
        )
        write_records_json([record], traces_dir / f"iter_{iteration:06d}.json")
        write_records_txt([record], traces_dir / f"iter_{iteration:06d}.txt")

    def _write_checkpoint_traces(
        self, iteration: int, trace_episodes: list[tuple]
    ) -> None:
        """Write multiple episode traces alongside a checkpoint."""
        if not trace_episodes:
            return
        from marlllm.trace_utils import make_episode_record, write_records_json, write_records_txt

        tok = list(self.agents.values())[0].tokenizer
        ckpt_traces_dir = (
            Path(self.config.output_dir) / "traces" / f"ckpt_{iteration:06d}"
        )
        ckpt_traces_dir.mkdir(parents=True, exist_ok=True)

        records = [
            make_episode_record(
                episode_idx=ep_idx,
                agent_context_tokens=ctx_snapshot,
                tokenizer=tok,
                env_trace=env_trace or ep_info or None,
            )
            for ep_idx, (ctx_snapshot, ep_info, env_trace) in enumerate(trace_episodes)
        ]
        write_records_json(records, ckpt_traces_dir / "traces.json")
        write_records_txt(records, ckpt_traces_dir / "traces.txt")

        self._logger.info(
            "Checkpoint traces saved: %s (%d episodes)", ckpt_traces_dir, len(records)
        )

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
