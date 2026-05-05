"""
Population-based CCSM training with multi-environment support.

A population of named agents (e.g. Marcus, Sophia, Viktor, Chen) is trained
by randomly pairing them for each episode.  Each agent owns a private character
prompt and either an independent model or a LoRA adapter on a shared backbone.

Each iteration can draw episodes from a *mixture of environments*, sampled by
weight.  Every environment can supply its own character prompt overrides (a
single string, or a list of variants to sample from per episode), enabling
diversity of experience analogous to training on varied text corpora.

Diversity sources
-----------------
1. **Population pairing** — who plays whom each episode.
2. **Pairing strategy** — random_no_self / random_with_self / round_robin.
3. **Role shuffle** — who goes first (per-env or global).
4. **LoRA adapters** — each character has its own adapter on a shared backbone.
5. **Multi-environment** — episodes drawn from different env types / configs.
6. **Prompt variants** — each env can specify multiple prompt strings per
   character; one is sampled per episode, so the same character can present
   differently depending on context.

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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW

try:
    import psutil  # type: ignore
    _PSUTIL_PROC = psutil.Process()
except Exception:  # pragma: no cover - optional dep
    psutil = None
    _PSUTIL_PROC = None


def _resource_metrics() -> dict[str, float]:
    """Snapshot of GPU/CPU memory and related counters for the current process."""
    out: dict[str, float] = {}
    GB = 1024 ** 3

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            prefix = f"mem/gpu{i}"
            out[f"{prefix}/alloc_gb"]      = torch.cuda.memory_allocated(i) / GB
            out[f"{prefix}/reserved_gb"]   = torch.cuda.memory_reserved(i) / GB
            out[f"{prefix}/peak_alloc_gb"] = torch.cuda.max_memory_allocated(i) / GB
            out[f"{prefix}/peak_reserved_gb"] = torch.cuda.max_memory_reserved(i) / GB
            try:
                free_b, total_b = torch.cuda.mem_get_info(i)
                out[f"{prefix}/free_gb"]  = free_b / GB
                out[f"{prefix}/total_gb"] = total_b / GB
                out[f"{prefix}/used_frac"] = 1.0 - free_b / total_b
            except Exception:
                pass

    if _PSUTIL_PROC is not None:
        try:
            mi = _PSUTIL_PROC.memory_info()
            out["mem/cpu_rss_gb"] = mi.rss / GB
            out["mem/cpu_vms_gb"] = mi.vms / GB
        except Exception:
            pass
        try:
            out["cpu/percent"] = _PSUTIL_PROC.cpu_percent(interval=None)
            out["cpu/num_threads"] = float(_PSUTIL_PROC.num_threads())
        except Exception:
            pass

    return out

from marlllm.agent import Agent
from marlllm.config import TrainingConfig
from marlllm.dialogue import chat_eos_token_ids, verify_special_tokens
from marlllm.loss import Loss
from marlllm.store import TrajectoryStore
from marlllm.tokeniser import Tokeniser
from marlllm.trace_utils import get_env_trace
from marlllm.types import EpisodeStep, RolloutBatch, TokenType, Trajectory


# ── EnvironmentSpec ───────────────────────────────────────────────────────────

@dataclass
class EnvironmentSpec:
    """
    One environment configuration for population training.

    Parameters
    ----------
    env:
        A PettingZoo AEC environment template.  The trainer deep-copies this
        object before each episode; the template itself is never reset/stepped.
    name:
        Short identifier used in logs and trace files.
    weight:
        Relative sampling probability.  Normalised across all specs so only
        the ratio matters.
    character_prompts:
        Per-character prompt overrides for this environment.  A value can be:
        - ``str`` — a single fixed prompt.
        - ``list[str]`` — a list of variants; one is sampled per episode.
        Characters not listed here fall back to the population's global
        ``config.character_prompts`` (which may also be a list of variants).
    """
    env: Any
    name: str = "env"
    weight: float = 1.0
    character_prompts: dict[str, str | list[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Normalise all prompt values to list[str] for uniform sampling.
        normalised: dict[str, list[str]] = {}
        for name, val in self.character_prompts.items():
            normalised[name] = [val] if isinstance(val, str) else list(val)
        self.character_prompts = normalised  # type: ignore[assignment]


# ── Raw episode record returned by _collect_episodes_batched ─────────────────
# (history, ep_info, pairing, prompt_ids_per_name, env_name,
#  context_snapshot, env_trace)
_RawEpisode = tuple[
    list[EpisodeStep],    # steps tagged with population-member names
    dict,                 # ep_info from env
    tuple[str, ...],      # population names participating in this episode
    dict[str, list[int]], # prompt token IDs keyed by population-member name
    str,                  # env_name (for logging / traces)
    dict[str, list[int]], # context snapshot: env_role -> flat token IDs
    dict | None,          # env_trace from env.episode_trace() or None
]


class PopulationTrainer:
    """
    Trains a population of agents via random pairings across one or more envs.

    Parameters
    ----------
    population:
        Mapping of character name → Agent.  The order of keys defines the
        integer indices used in agent_id_mask tensors.
    environments:
        One or more ``EnvironmentSpec`` objects.  Each iteration samples
        episodes from the weighted mixture defined by their ``weight`` fields.
        Pass a single PettingZoo env (not wrapped in EnvironmentSpec) for
        backward compatibility — it will be wrapped automatically.
    loss, tokeniser, store, config:
        Same roles as in the standard Trainer.
    pairing_strategy:
        How to pair agents each episode.  One of
        ``"random_no_self"`` | ``"random_with_self"`` | ``"round_robin"``.
    """

    def __init__(
        self,
        population: dict[str, Agent],
        environments: list[EnvironmentSpec] | Any,  # Any = single bare env
        loss: Loss,
        tokeniser: Tokeniser,
        store: TrajectoryStore,
        config: TrainingConfig,
        pairing_strategy: str = "random_no_self",
        sampling_engine: Any = None,
        sampling_engine_peft_model: Any = None,
        ddp_rank: int = 0,
        ddp_world_size: int = 1,
    ) -> None:
        if len(population) < 1:
            raise ValueError("population must have at least one member")

        # Backward compat: accept a single bare env or a list of EnvironmentSpec.
        if not isinstance(environments, list):
            environments = [EnvironmentSpec(env=environments, name="env")]
        for i, spec in enumerate(environments):
            if not isinstance(spec, EnvironmentSpec):
                environments[i] = EnvironmentSpec(env=spec, name=f"env_{i}")

        # Two role-mapping modes are supported:
        #  - "pair": env exposes generic roles {"agent_0", "agent_1"} and we
        #    sample a 2-tuple of population members to bind each episode.
        #  - "named": env's possible_agents are population character names
        #    (Concordia case); each role is bound 1:1 to its same-named member.
        for spec in environments:
            env_agents = list(getattr(spec.env, "possible_agents", []))
            if len(env_agents) < 2:
                raise ValueError(
                    f"EnvironmentSpec '{spec.name}': requires an env with at least "
                    f"2 possible_agents, got {env_agents}"
                )
            if len(env_agents) == 2 and set(env_agents) == {"agent_0", "agent_1"}:
                continue  # pair-mode env
            missing = [a for a in env_agents if a not in population]
            if missing:
                raise ValueError(
                    f"EnvironmentSpec '{spec.name}': env roles {missing} are not "
                    f"present in population {list(population)}"
                )

        self.population = population
        self.environments = environments
        self.loss = loss
        self.tokeniser = tokeniser
        self.config = config
        self.pairing_strategy = pairing_strategy
        # Optional vLLM sampling engine (rollout-only). When set, _collect_episodes_batched
        # routes generation through it instead of agent.act_batch, and the trainer
        # calls sync_all_adapters() after each optimizer.step().
        self.sampling_engine = sampling_engine
        self._sampling_engine_peft_model = sampling_engine_peft_model

        self.device = torch.device(config.device)
        self.agent_index: dict[str, int] = {
            name: i for i, name in enumerate(population)
        }

        # Pre-compute normalised environment sampling weights.
        total_w = sum(s.weight for s in environments)
        self._env_weights = [s.weight / total_w for s in environments]

        # Normalise global character_prompts in config to list[str] for
        # uniform sampling alongside per-env variants.
        self._global_prompt_variants: dict[str, list[str]] = {}
        for name, val in config.character_prompts.items():
            self._global_prompt_variants[name] = (
                [val] if isinstance(val, str) else list(val)
            )

        # Single optimizer over all unique parameters across the population.
        seen_ids: set[int] = set()
        all_params: list = []
        for agent in population.values():
            for p in agent.parameters():
                if id(p) not in seen_ids:
                    seen_ids.add(id(p))
                    all_params.append(p)
        if getattr(config, "use_8bit_adam", False):
            import bitsandbytes as bnb
            self.optimizer = bnb.optim.AdamW8bit(all_params, lr=config.lr)
        else:
            self.optimizer = AdamW(all_params, lr=config.lr)

        # Data-parallel state. When world_size > 1 each rank runs an independent
        # train() loop on its own slice of episodes; gradients are all-reduced
        # before every optimizer.step() so all ranks converge on the same
        # optimizer state. Only rank 0 writes checkpoints/traces/snapshots/metrics.
        self.ddp_rank = ddp_rank
        self.ddp_world_size = ddp_world_size
        self.ddp = ddp_world_size > 1
        self._ddp_params = all_params  # cached for the gradient all-reduce

        # Chat-template EOS ids: stop generation cleanly on <|im_end|> /
        # <|endoftext|> (or the equivalent for the active model family) so
        # actions don't drag garbage past the end-of-turn marker.
        any_tok = next(iter(population.values())).tokenizer
        verify_special_tokens(any_tok)
        self._eos_token_ids = chat_eos_token_ids(any_tok)

        self._start_time = time.time()
        if _PSUTIL_PROC is not None:
            try:
                _PSUTIL_PROC.cpu_percent(interval=None)  # prime the meter
            except Exception:
                pass
        self._rollout_stats: dict[str, float] = {}
        # Each rank gets its own slice of the seed space so envs/pairings
        # diverge across ranks (we want different episodes per rank, not
        # duplicates).
        rank_seed_offset = ddp_rank * 1_000_003  # prime, avoids accidental aliasing
        self._rng_counter = config.seed + rank_seed_offset
        self._rng = random.Random(config.seed + rank_seed_offset)

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

        # Variance-decomposition eval state (lazy: contexts built on first use).
        self._var_decomp_contexts: list | None = None
        self._var_decomp_metrics_path = (
            Path(config.output_dir) / "variance_decomposition.jsonl"
        )

    # ------------------------------------------------------------------ #
    # Main training loop                                                   #
    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    # DDP helpers                                                          #
    # ------------------------------------------------------------------ #

    def _ddp_all_reduce_grads(self) -> None:
        """Average .grad across ranks for every trainable parameter.

        Pads ranks that didn't compute a grad for some parameter (e.g. an
        adapter whose agent didn't appear in any of this rank's episodes)
        with a zero tensor of the right shape so all ranks participate in
        every collective. Without this, NCCL hangs because of the missing
        op on those ranks.
        """
        import torch.distributed as dist
        for p in self._ddp_params:
            if p.grad is None:
                # This rank has no gradient for p — fill with zeros so the
                # all-reduce still happens (and contributes 0).
                p.grad = torch.zeros_like(p)
            dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)

    def _ddp_reduce_metrics(self, metrics: dict[str, float]) -> dict[str, float]:
        """Average scalar metrics across ranks. Sums episode counts."""
        import torch.distributed as dist
        if not self.ddp or not metrics:
            return metrics
        # Sums for things that should aggregate, averages for everything else.
        SUM_KEYS_PREFIX = ("rollout/gen_tokens", "rollout/gen_calls", "n_episodes",
                            "unique_pairings", "env_episodes/")
        keys = sorted(metrics.keys())
        # Pack into one tensor per reduction op for efficiency.
        def _is_sum_key(k: str) -> bool:
            if any(k.startswith(p) for p in SUM_KEYS_PREFIX):
                return True
            # Tool-use raw counts sum across ranks; per-episode rates average.
            return k.startswith("tool/") and k.endswith("_total")
        sum_keys = [k for k in keys if _is_sum_key(k)]
        avg_keys = [k for k in keys if k not in sum_keys and isinstance(metrics[k], (int, float))]
        if sum_keys:
            t = torch.tensor([float(metrics[k]) for k in sum_keys], device="cuda")
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            for k, v in zip(sum_keys, t.cpu().tolist()):
                metrics[k] = v
        if avg_keys:
            t = torch.tensor([float(metrics[k]) for k in avg_keys], device="cuda")
            dist.all_reduce(t, op=dist.ReduceOp.AVG)
            for k, v in zip(avg_keys, t.cpu().tolist()):
                metrics[k] = v
        return metrics

    @property
    def is_main_rank(self) -> bool:
        return self.ddp_rank == 0

    def _compute_generation_metrics(
        self, raw_episodes: list["_RawEpisode"]
    ) -> dict[str, float]:
        """Per-env aggregates of action / thinking / post token usage.

        Lets us see how the model spends its token budget: average post
        length, average thinking length, fraction of the action spent in
        thinking, and how often the post or total budget gets clipped.
        """
        env_by_name = {spec.name: spec.env for spec in self.environments}
        agg: dict[str, dict[str, float]] = {}
        for raw in raw_episodes:
            history, _info, _pairing, _pids, env_name, _ctx, _trace = raw
            env_t = env_by_name.get(env_name)
            thinking_enabled = bool(getattr(env_t, "_thinking_enabled", False))
            open_tag = getattr(env_t, "_thinking_open_tag", "<think>")
            close_tag = getattr(env_t, "_thinking_close_tag", "</think>")
            post_budget = int(getattr(env_t, "_post_token_budget", 0) or 0)
            total_budget = int(getattr(env_t, "_total_token_budget", 0) or 0)
            a = agg.setdefault(env_name, {
                "n_actions": 0, "sum_action_tokens": 0,
                "sum_thinking_tokens": 0, "sum_post_tokens": 0,
                "n_thinking_present": 0, "n_total_truncated": 0,
                "n_post_truncated": 0, "n_unclosed_thinking": 0,
                "n_episodes": 0, "sum_episode_act_tokens": 0,
                "sum_episode_total_tokens": 0, "sum_posts_per_episode": 0,
                "max_action_tokens": 0, "max_episode_total_tokens": 0,
            })
            ep_act_tokens = 0
            ep_total_tokens = 0
            ep_posts = 0
            for step in history:
                ep_total_tokens += len(step.token_ids)
                if step.token_type != TokenType.ACT:
                    continue
                ids = list(step.token_ids)
                n_act = len(ids)
                ep_act_tokens += n_act
                ep_posts += 1
                text = self.tokeniser.decode_action(ids)
                thinking_tokens = 0
                unclosed = False
                if thinking_enabled and open_tag in text:
                    j = text.find(open_tag)
                    k_close = text.find(close_tag, j + len(open_tag))
                    if k_close < 0:
                        unclosed = True
                        thinking_text = text[j:]
                        post_text = text[:j]
                    else:
                        thinking_text = text[j:k_close + len(close_tag)]
                        post_text = text[:j] + text[k_close + len(close_tag):]
                    thinking_tokens = (
                        len(self.tokeniser.encode_observation(thinking_text))
                        if thinking_text else 0
                    )
                    post_tokens = (
                        len(self.tokeniser.encode_observation(post_text.strip()))
                        if post_text.strip() else 0
                    )
                else:
                    post_tokens = n_act
                a["n_actions"] += 1
                a["sum_action_tokens"] += n_act
                a["sum_thinking_tokens"] += thinking_tokens
                a["sum_post_tokens"] += post_tokens
                if n_act > a["max_action_tokens"]:
                    a["max_action_tokens"] = n_act
                if thinking_enabled and thinking_tokens > 0:
                    a["n_thinking_present"] += 1
                if unclosed:
                    a["n_unclosed_thinking"] += 1
                if total_budget and n_act >= total_budget:
                    a["n_total_truncated"] += 1
                if post_budget and post_tokens > post_budget:
                    a["n_post_truncated"] += 1
            a["n_episodes"] += 1
            a["sum_episode_act_tokens"] += ep_act_tokens
            a["sum_episode_total_tokens"] += ep_total_tokens
            a["sum_posts_per_episode"] += ep_posts
            if ep_total_tokens > a["max_episode_total_tokens"]:
                a["max_episode_total_tokens"] = ep_total_tokens

        out: dict[str, float] = {}
        for env_name, a in agg.items():
            n_act = a["n_actions"] or 1
            n_eps = a["n_episodes"] or 1
            p = f"gen/{env_name}"
            out[f"{p}/mean_action_tokens"]   = a["sum_action_tokens"] / n_act
            out[f"{p}/mean_thinking_tokens"] = a["sum_thinking_tokens"] / n_act
            out[f"{p}/mean_post_tokens"]     = a["sum_post_tokens"] / n_act
            out[f"{p}/thinking_frac"] = (
                a["sum_thinking_tokens"] / a["sum_action_tokens"]
                if a["sum_action_tokens"] > 0 else 0.0
            )
            out[f"{p}/thinking_present_rate"] = a["n_thinking_present"] / n_act
            out[f"{p}/total_truncation_rate"] = a["n_total_truncated"] / n_act
            out[f"{p}/post_truncation_rate"]  = a["n_post_truncated"] / n_act
            out[f"{p}/unclosed_thinking_rate"] = a["n_unclosed_thinking"] / n_act
            out[f"{p}/mean_episode_act_tokens"]   = a["sum_episode_act_tokens"] / n_eps
            out[f"{p}/mean_episode_total_tokens"] = a["sum_episode_total_tokens"] / n_eps
            out[f"{p}/mean_posts_per_episode"] = a["sum_posts_per_episode"] / n_eps
            out[f"{p}/max_action_tokens"] = float(a["max_action_tokens"])
            out[f"{p}/max_episode_total_tokens"] = float(a["max_episode_total_tokens"])
        return out

    def train(self, start_iteration: int = 1) -> None:
        pop_names = list(self.population.keys())
        env_names = [s.name for s in self.environments]
        if self.ddp_rank == 0:
            self._logger.info(
                "Population training: %d members %s | envs=%s | pairing=%s | iters=%d | ddp_world_size=%d",
                len(pop_names), pop_names, env_names, self.pairing_strategy,
                self.config.num_iterations, self.ddp_world_size,
            )
            self._logger.info("Output directory: %s", self.config.output_dir)

        # When running data-parallel, each rank only collects 1/world_size of the
        # configured episodes per iter. The aggregate work is the same as a
        # single-rank run with episodes_per_iter, but split across ranks.
        local_episodes_per_iter = max(
            1, self.config.episodes_per_iter // self.ddp_world_size
        )

        # ── Baseline rollout at iter 0 (before any training) ──────────────
        # Collect a batch of episodes from the untrained policy and dump
        # traces so we have a "what does the model do out of the box" record
        # to compare later checkpoints against. Only on a fresh run.
        if start_iteration == 1:
            baseline_episodes: list[_RawEpisode] = self._collect_episodes_batched(
                local_episodes_per_iter
            )
            if self.is_main_rank:
                self._write_checkpoint_traces(0, baseline_episodes)
            if self.ddp:
                import torch.distributed as dist
                dist.barrier()

        for iteration in range(start_iteration, self.config.num_iterations + 1):

            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    torch.cuda.reset_peak_memory_stats(i)

            # ── 1. Collect episodes ───────────────────────────────────────
            _t_phase = time.perf_counter()
            raw_episodes: list[_RawEpisode] = self._collect_episodes_batched(
                local_episodes_per_iter
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            rollout_s = time.perf_counter() - _t_phase

            # ── 2. Per-agent update ───────────────────────────────────────
            _t_phase = time.perf_counter()
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

                    actual_len = int(mb_attn.sum(dim=1).max())
                    if actual_len < mb_input_ids.shape[1]:
                        mb_input_ids = mb_input_ids[:, :actual_len]
                        mb_attn      = mb_attn[:, :actual_len]
                        mb_types     = mb_types[:, :actual_len]
                        mb_agents    = mb_agents[:, :actual_len]

                    last_hidden, values = agent.evaluate_hidden(mb_input_ids, mb_attn)
                    last_hidden_ref = agent.evaluate_hidden_ref(mb_input_ids, mb_attn)
                    agent_idx = self.agent_index[pop_name]

                    loss_val, metrics = self.loss.compute_loss(
                        last_hidden=last_hidden,
                        lm_head=agent.lm_head,
                        values=values,
                        input_ids=mb_input_ids,
                        token_type_mask=mb_types,
                        agent_id_mask=mb_agents,
                        target_agent_idx=agent_idx,
                        config=self.config,
                        last_hidden_ref=last_hidden_ref,
                        lm_head_ref=agent.lm_head_ref if last_hidden_ref is not None else None,
                    )

                    scale = 1.0 / (num_micros * len(self.population))
                    (loss_val * scale).backward()
                    total_loss_scalar += loss_val.item() / num_micros

                    for k, v in metrics.items():
                        key = f"{pop_name}/{k}"
                        all_metrics[key] = all_metrics.get(key, 0.0) + v / num_micros

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            loss_s = time.perf_counter() - _t_phase

            # ── 3. Optimizer step ─────────────────────────────────────────
            _t_phase = time.perf_counter()
            if self.ddp:
                # Average gradients across ranks before stepping. We don't wrap
                # in DistributedDataParallel because PEFT swaps the wrapped
                # forward (adapter switching) and torch.compile + DDP +
                # find_unused_parameters has rough edges. Manual all-reduce on
                # the .grad buffers is equivalent, simpler, and easy to reason
                # about: every rank applies the same averaged gradient and
                # therefore stays bit-identical in optimizer state.
                self._ddp_all_reduce_grads()
            self.optimizer.step()
            if self.sampling_engine is not None and self._sampling_engine_peft_model is not None:
                # Push fresh LoRA weights into vLLM so the next rollout
                # samples from the just-updated policy.
                self.sampling_engine.sync_all_adapters(self._sampling_engine_peft_model)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                # Release the caching allocator's reserved-but-unallocated blocks
                # so subsequent steps (DDP metric reduce, next iter's rollout
                # forward) have room. Without this, fragmentation builds across
                # many small backward calls and tiny allocations later OOM.
                torch.cuda.empty_cache()
            optim_s = time.perf_counter() - _t_phase

            # ── 4. Metrics ────────────────────────────────────────────────
            episode_results = [ep[1].get("result", "") for ep in raw_episodes]
            n_eps = len(episode_results)
            all_metrics["total_loss"]  = total_loss_scalar
            all_metrics["iteration"]   = iteration
            all_metrics["wall_time"]   = time.time() - self._start_time
            all_metrics["time/rollout_s"] = rollout_s
            all_metrics["time/loss_s"]    = loss_s
            all_metrics["time/optim_s"]   = optim_s
            iter_s = rollout_s + loss_s + optim_s
            all_metrics["time/iter_s"]    = iter_s
            # Phase fraction of total iter wall time (different from rollout/gen_frac
            # which is the fraction of the rollout *phase* spent in actual generation).
            all_metrics["time/rollout_frac"] = (rollout_s / iter_s) if iter_s > 0 else 0.0
            all_metrics["time/loss_frac"]    = (loss_s / iter_s) if iter_s > 0 else 0.0
            gen_time = self._rollout_stats.get("gen_time_s", 0.0)
            gen_tokens = self._rollout_stats.get("gen_tokens", 0)
            all_metrics["time/gen_s"]       = gen_time
            all_metrics["time/env_step_s"]  = self._rollout_stats.get("env_step_time_s", 0.0)
            all_metrics["rollout/gen_tokens"] = gen_tokens
            gen_calls = self._rollout_stats.get("gen_calls", 0)
            all_metrics["rollout/gen_calls"]  = gen_calls
            all_metrics["rollout/tokens_per_s"] = (gen_tokens / gen_time) if gen_time > 0 else 0.0
            all_metrics["rollout/gen_frac"]     = (gen_time / rollout_s) if rollout_s > 0 else 0.0
            gen_batch_sum = self._rollout_stats.get("gen_batch_sum", 0)
            all_metrics["rollout/mean_batch"] = (gen_batch_sum / gen_calls) if gen_calls > 0 else 0.0
            all_metrics["rollout/max_batch"]  = self._rollout_stats.get("gen_batch_max", 0)

            all_metrics.update(_resource_metrics())
            all_metrics["n_episodes"]  = n_eps
            if n_eps:
                all_metrics["success_rate"] = episode_results.count("success") / n_eps

            tool_totals: dict[str, float] = {}
            for ep in raw_episodes:
                ep_info = ep[1] or {}
                for k, v in ep_info.items():
                    if isinstance(k, str) and k.startswith("tool/"):
                        tool_totals[k] = tool_totals.get(k, 0.0) + float(v)
            if tool_totals:
                denom = max(1, n_eps)
                for k, total in tool_totals.items():
                    all_metrics[f"{k}_total"] = total
                    all_metrics[f"{k}_per_ep"] = total / denom

            pairings = [ep[2] for ep in raw_episodes]
            all_metrics["unique_pairings"] = len(set(pairings))

            # Per-env episode counts
            env_counts: dict[str, int] = {}
            for ep in raw_episodes:
                env_name = ep[4]
                env_counts[env_name] = env_counts.get(env_name, 0) + 1
            for env_name, count in env_counts.items():
                all_metrics[f"env_episodes/{env_name}"] = count

            all_metrics.update(self._compute_generation_metrics(raw_episodes))

            # Reduce metrics across ranks before logging — sum-style for counts,
            # average for losses/timings — so the JSONL row is identical to a
            # single-rank run with the same total episodes_per_iter.
            if self.ddp:
                all_metrics = self._ddp_reduce_metrics(all_metrics)

            # ── 4b. Variance-decomposition diagnostic (cooperative DDP) ──
            # Run after the optimizer step so we measure the *current*
            # post-update policy. All ranks participate so DDP collectives
            # (the per-iter barrier below, and next iter's gradient
            # all-reduce) stay in lockstep — running this on rank 0 alone
            # caused the 10-minute NCCL watchdog to trip at the next barrier.
            if self._should_run_var_decomp(iteration):
                self._run_var_decomp(iteration, all_metrics)

            # ── 5. Log and checkpoint (rank 0 only) ──────────────────────
            if self.is_main_rank:
                if iteration % self.config.log_every == 0:
                    self._log_metrics(iteration, all_metrics)
                    for raw_ep in raw_episodes:
                        hist, ep_info, pairing, pids, env_name, ctx_snapshot, env_trace_k = raw_ep
                        if hist:
                            self._write_trace(iteration, ctx_snapshot, ep_info, pairing, env_name, env_trace_k)
                            break

                self._write_metrics_jsonl(all_metrics)

                if iteration % self.config.log_every == 0:
                    self._auto_plot_training_curves()

                if iteration % self.config.checkpoint_every == 0:
                    self.save_checkpoint(iteration)
                    self._write_checkpoint_traces(iteration, raw_episodes)
                    self._write_behavioral_snapshot(iteration)
                    # Snapshot eval allocates large activation buffers and leaves
                    # the caching allocator fragmented; release before next iter.
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            # All ranks barrier here so non-main ranks don't race ahead while
            # rank 0 is doing slow snapshot/checkpoint I/O. Without this, rank 0
            # falls behind and gradient all-reduce next iter would block.
            if self.ddp:
                import torch.distributed as dist
                dist.barrier()

        if self.is_main_rank:
            self.save_checkpoint(self.config.num_iterations, tag="final")
            self._write_checkpoint_traces(self.config.num_iterations, raw_episodes)
            self._write_behavioral_snapshot(self.config.num_iterations)
            self._auto_plot_training_curves()
            self._logger.info("Training complete.")
        if self.ddp:
            import torch.distributed as dist
            dist.barrier()

    # ------------------------------------------------------------------ #
    # Prompt sampling                                                       #
    # ------------------------------------------------------------------ #

    def _sample_prompt(self, pop_name: str, env_spec: EnvironmentSpec) -> str:
        """
        Sample a prompt string for `pop_name` in the context of `env_spec`.

        Priority:
          1. env_spec.character_prompts[pop_name]  (list → sample uniformly)
          2. self._global_prompt_variants[pop_name] (list → sample uniformly)
          3. Empty string fallback.
        """
        env_variants = env_spec.character_prompts.get(pop_name)
        if env_variants:
            return self._rng.choice(env_variants)
        global_variants = self._global_prompt_variants.get(pop_name)
        if global_variants:
            return self._rng.choice(global_variants)
        return ""

    # ------------------------------------------------------------------ #
    # Episode collection                                                   #
    # ------------------------------------------------------------------ #

    def _sample_pairs(self, n: int) -> list[tuple[str, str]]:
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
                for _ in range(n):
                    pairs.append((names[0], names[0]))
            else:
                for _ in range(n):
                    a, b = self._rng.sample(names, 2)
                    pairs.append((a, b))

        return pairs

    def _sample_env_specs(self, n: int) -> list[EnvironmentSpec]:
        """Sample n EnvironmentSpec objects according to their weights."""
        return self._rng.choices(self.environments, weights=self._env_weights, k=n)

    def _collect_episodes_batched(self, n: int) -> list[_RawEpisode]:
        """
        Run n episodes in parallel.  For each episode:
          - an environment is sampled from the weighted mixture
          - a character pairing is sampled
          - prompts are sampled (potentially from per-env variants)

        Steps are tagged with population-member names as agent_id (not env roles).
        Observations are broadcast to both population members' context windows.
        """
        self._rollout_stats = {
            "gen_time_s": 0.0,
            "gen_tokens": 0,
            "gen_calls": 0,
            "gen_batch_sum": 0,
            "gen_batch_max": 0,
            "env_step_time_s": 0.0,
        }

        env_specs  = self._sample_env_specs(n)

        # Build per-episode role→name mapping based on each env's possible_agents.
        # Pair-mode envs ({"agent_0","agent_1"}) sample a 2-tuple of pop members;
        # named-role envs (e.g. Concordia) bind each role to its same-named member.
        pair_count = sum(
            1 for spec in env_specs
            if set(getattr(spec.env, "possible_agents", [])) == {"agent_0", "agent_1"}
        )
        sampled_pairs = self._sample_pairs(pair_count) if pair_count else []
        pair_iter = iter(sampled_pairs)

        pairings: list[tuple[str, ...]] = []
        env_role_to_name: list[dict[str, str]] = []
        for spec in env_specs:
            roles = list(spec.env.possible_agents)
            if set(roles) == {"agent_0", "agent_1"}:
                p = next(pair_iter)
                env_role_to_name.append({"agent_0": p[0], "agent_1": p[1]})
                pairings.append(p)
            else:
                env_role_to_name.append({r: r for r in roles})
                pairings.append(tuple(roles))

        # Deep-copy one env per episode from the appropriate spec template.
        # Use a single shared ``order_seed`` across all parallel episodes so
        # round-robin turn schedules stay aligned — otherwise per-episode
        # turn-order shuffling desynchronizes which agent acts at each step
        # and collapses vLLM batching from N→1 (3× more gen calls, ~2× lower
        # tokens/s). The order varies across rollout calls (one shuffle per
        # iteration), which is enough to avoid a fixed opener.
        order_seed = self._rng_counter
        self._rng_counter += 1
        envs: list = []
        for k in range(n):
            spec = env_specs[k]
            env_k = copy.deepcopy(spec.env)
            if hasattr(env_k, "_tok"):
                env_k._tok = spec.env._tok
            env_k.reset(seed=self._rng_counter, options={"order_seed": order_seed})
            self._rng_counter += 1
            envs.append(env_k)

        # Per-episode bookkeeping
        histories:    list[list[EpisodeStep]]      = [[] for _ in range(n)]
        contexts:     list[dict[str, list[int]]]   = [{} for _ in range(n)]
        prompt_ids:   list[dict[str, list[int]]]   = [{} for _ in range(n)]
        ep_infos:     list[dict]                   = [{} for _ in range(n)]
        token_counts: list[int]                    = [0] * n
        active:       list[bool]                   = [True] * n

        # Initialise contexts: sample one prompt per character per episode.
        # The sampled prompt (possibly env-specific and/or one of many variants)
        # is stored in prompt_ids so _build_agent_trajectories can prepend the
        # exact context the agent actually saw during rollout.
        _full_ctx = False
        if self.environments:
            _full_ctx = getattr(self.environments[0].env, 'obs_is_full_context', False)

        for k in range(n):
            spec = env_specs[k]
            for env_role, pop_name in env_role_to_name[k].items():
                if _full_ctx:
                    if pop_name not in prompt_ids[k]:
                        prompt_ids[k][pop_name] = []
                    contexts[k][env_role] = []
                else:
                    formatter = self.population[pop_name].context_formatter
                    if pop_name not in prompt_ids[k]:
                        prompt_text = self._sample_prompt(pop_name, spec)
                        pids = self.tokeniser.encode_prompt(prompt_text)
                        prompt_ids[k][pop_name] = pids
                        contexts[k][env_role] = formatter.wrap_prompt(pids)
                    else:
                        # Self-play: both roles map to the same character — reuse.
                        contexts[k][env_role] = formatter.wrap_prompt(prompt_ids[k][pop_name])

        # Per-episode token budget from the env (may differ across env types).
        n_tokens_per_ep: list[int] = [
            getattr(envs[k], "action_token_budget", 1) for k in range(n)
        ]

        while any(active):
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
                obs, _rew, term, trunc, info = env.last()

                obs_ids = self.tokeniser.encode_observation(obs)
                if obs_ids:
                    histories[k].append(EpisodeStep(
                        agent_id=pop_name,
                        token_ids=obs_ids,
                        token_type=TokenType.OBS,
                        log_probs=[],
                        info={},
                    ))
                    if _full_ctx:
                        contexts[k][env_role] = list(obs_ids)
                    else:
                        formatter = self.population[pop_name].context_formatter
                        contexts[k][env_role].extend(formatter.wrap_observation(obs_ids))
                    token_counts[k] += len(obs_ids)

                if term or trunc:
                    for aid in env.possible_agents:
                        if aid in env.infos:
                            ep_infos[k] = env.infos[aid]
                            break
                    null_indices.append(k)
                elif (token_counts[k] >= self.config.max_episode_tokens
                      and not info.get("must_act", False)):
                    null_indices.append(k)
                elif pop_name in self.population:
                    act_groups.setdefault(pop_name, []).append(k)
                else:
                    null_indices.append(k)

            _t_env = time.perf_counter()
            for k in null_indices:
                envs[k].step(None)
                if not envs[k].agents:
                    active[k] = False
            self._rollout_stats["env_step_time_s"] += time.perf_counter() - _t_env

            for pop_name, env_indices in act_groups.items():
                agent = self.population[pop_name]
                agent.eval_mode()

                batch_contexts = [
                    contexts[k][envs[k].agent_selection]
                    for k in env_indices
                ]
                # Token budget may differ per episode — use the max for batching,
                # which is safe since outputs are trimmed to actual length.
                n_tokens = max(n_tokens_per_ep[k] for k in env_indices)
                _t_gen = time.perf_counter()
                if self.sampling_engine is not None:
                    batch_ids, batch_lps = self.sampling_engine.generate(
                        contexts=batch_contexts,
                        n_tokens=n_tokens,
                        temperature=self.config.temperature,
                        eos_token_ids=self._eos_token_ids,
                        adapter_name=pop_name,
                    )
                else:
                    with torch.no_grad():
                        batch_ids, batch_lps = agent.act_batch(
                            contexts=batch_contexts,
                            n_tokens=n_tokens,
                            temperature=self.config.temperature,
                            eos_token_ids=self._eos_token_ids,
                        )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                self._rollout_stats["gen_time_s"] += time.perf_counter() - _t_gen
                self._rollout_stats["gen_tokens"] += sum(len(ids) for ids in batch_ids)
                self._rollout_stats["gen_calls"] += 1
                _b = len(batch_contexts)
                self._rollout_stats["gen_batch_sum"] += _b
                if _b > self._rollout_stats["gen_batch_max"]:
                    self._rollout_stats["gen_batch_max"] = _b

                formatter = self.population[pop_name].context_formatter
                for j, k in enumerate(env_indices):
                    act_ids = batch_ids[j]
                    act_lps = batch_lps[j]
                    # Trim to this episode's budget if we over-generated.
                    budget = n_tokens_per_ep[k]
                    if len(act_ids) > budget:
                        act_ids = act_ids[:budget]
                        act_lps = act_lps[:budget]

                    env_role = envs[k].agent_selection

                    histories[k].append(EpisodeStep(
                        agent_id=pop_name,
                        token_ids=act_ids,
                        token_type=TokenType.ACT,
                        log_probs=act_lps,
                        info={},
                    ))
                    contexts[k][env_role].extend(formatter.wrap_action(act_ids))
                    token_counts[k] += len(act_ids)
                    _t_env = time.perf_counter()
                    envs[k].step(act_ids)
                    self._rollout_stats["env_step_time_s"] += time.perf_counter() - _t_env
                    if not envs[k].agents:
                        active[k] = False

        context_snapshots = [
            {env_role: list(contexts[k][env_role]) for env_role in contexts[k]}
            for k in range(n)
        ]
        env_traces_list = [get_env_trace(envs[k]) for k in range(n)]
        return [
            (histories[k], ep_infos[k], pairings[k], prompt_ids[k], env_specs[k].name,
             context_snapshots[k], env_traces_list[k])
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

        Each trajectory is prefixed with the exact prompt tokens that were
        used for this character in this episode (which may vary across episodes
        when prompt variants are enabled), ensuring the training forward pass
        is conditioned on the same context the agent saw during rollout.
        """
        formatter = self.population[pop_name].context_formatter
        trajectories: list[Trajectory] = []
        for history, _ep_info, _pairing, pids, _env_name, _ctx_snapshot, _env_trace in agent_episodes:
            steps: list[EpisodeStep] = []
            prompt = pids.get(pop_name, [])
            if prompt:
                prompt_type = (
                    TokenType.OBS if self.config.prompt_as_observation else TokenType.PAD
                )
                steps.append(EpisodeStep(
                    agent_id=pop_name,
                    token_ids=formatter.wrap_prompt(prompt),
                    token_type=prompt_type,
                    log_probs=[],
                    info={},
                ))
            for step in history:
                if step.agent_id == pop_name:
                    if step.token_type == TokenType.OBS:
                        fids = formatter.wrap_observation(step.token_ids)
                        lps = step.log_probs
                    elif step.token_type == TokenType.ACT:
                        fids = formatter.wrap_action(step.token_ids)
                        lps = step.log_probs + [0.0] * (len(fids) - len(step.token_ids))
                    else:
                        fids = step.token_ids
                        lps = step.log_probs
                    steps.append(EpisodeStep(pop_name, fids, step.token_type, lps, step.info))
            traj = self.tokeniser.build_trajectory(
                episode_history=steps,
                agent_ids_present=list(self.population.keys()),
            )
            trajectories.append(traj)
        return trajectories

    # ------------------------------------------------------------------ #
    # Checkpointing                                                        #
    # ------------------------------------------------------------------ #

    def save_checkpoint(self, iteration: int, tag: str | None = None) -> None:
        from marlllm.checkpoint_utils import save_population_checkpoint

        ckpt_dir = save_population_checkpoint(
            population=self.population,
            iteration=iteration,
            optimizer=self.optimizer,
            config_dict=dataclasses.asdict(self.config),
            output_dir=Path(self.config.output_dir),
            extra_meta={"env_names": [s.name for s in self.environments]},
            tag=tag,
        )
        self._logger.info("Checkpoint saved: %s", ckpt_dir)

    def load_checkpoint(self, path: str) -> int:
        """Load a checkpoint from a directory (new format) or .pt (legacy)."""
        from marlllm.checkpoint_utils import load_population_checkpoint

        p = Path(path)
        if p.is_dir():
            iteration = load_population_checkpoint(
                population=self.population,
                optimizer=self.optimizer,
                ckpt_dir=p,
                device=self.device,
            )
            self._logger.info(
                "Checkpoint loaded from %s (iteration %d)", p, iteration
            )
            return iteration

        # Legacy single-file .pt format
        payload = torch.load(p, map_location=self.device)
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
            "Checkpoint loaded from %s (iteration %d)", p, iteration
        )
        return iteration

    # ------------------------------------------------------------------ #
    # Logging helpers                                                      #
    # ------------------------------------------------------------------ #

    def _write_trace(
        self,
        iteration: int,
        ctx_snapshot: dict,
        ep_info: dict,
        pairing: tuple[str, str],
        env_name: str,
        env_trace=None,
    ) -> None:
        from marlllm.trace_utils import make_episode_record, write_records_json, write_records_txt

        tok = list(self.population.values())[0].tokenizer
        traces_dir = Path(self.config.output_dir) / "traces"
        traces_dir.mkdir(exist_ok=True)

        combined_env_trace = dict(ep_info) if ep_info else {}
        combined_env_trace["pairing"] = f"{pairing[0]} vs {pairing[1]}"
        combined_env_trace["env"] = env_name
        if env_trace:
            combined_env_trace.update(env_trace)

        record = make_episode_record(
            episode_idx=0,
            agent_context_tokens=ctx_snapshot,
            tokenizer=tok,
            env_trace=combined_env_trace,
        )
        write_records_json([record], traces_dir / f"iter_{iteration:06d}.json")
        write_records_txt([record], traces_dir / f"iter_{iteration:06d}.txt")

    def _write_checkpoint_traces(
        self,
        iteration: int,
        raw_episodes: list,
    ) -> None:
        """Write up to ``num_checkpoint_traces`` episode records alongside checkpoints."""
        from marlllm.trace_utils import make_episode_record, write_records_json, write_records_txt

        n_target = getattr(self.config, "num_checkpoint_traces", 0) or 0
        if n_target <= 0 or not raw_episodes:
            return

        # Pick the first n_target non-empty episodes
        chosen: list = []
        for raw_ep in raw_episodes:
            hist, ep_info, pairing, _pids, env_name, ctx_snapshot, env_trace_k = raw_ep
            if hist:
                chosen.append(raw_ep)
            if len(chosen) >= n_target:
                break
        if not chosen:
            return

        tok = list(self.population.values())[0].tokenizer
        ckpt_traces_dir = (
            Path(self.config.output_dir) / "traces" / f"ckpt_{iteration:06d}"
        )
        ckpt_traces_dir.mkdir(parents=True, exist_ok=True)

        records = []
        for idx, raw_ep in enumerate(chosen):
            _hist, ep_info, pairing, _pids, env_name, ctx_snapshot, env_trace_k = raw_ep
            combined_env_trace = dict(ep_info) if ep_info else {}
            combined_env_trace["pairing"] = f"{pairing[0]} vs {pairing[1]}"
            combined_env_trace["env"] = env_name
            if env_trace_k:
                combined_env_trace.update(env_trace_k)
            records.append(make_episode_record(
                episode_idx=idx,
                agent_context_tokens=ctx_snapshot,
                tokenizer=tok,
                env_trace=combined_env_trace,
            ))

        write_records_json(records, ckpt_traces_dir / "traces.json")
        write_records_txt(records, ckpt_traces_dir / "traces.txt")
        self._logger.info(
            "Checkpoint traces saved: %s (%d episodes)",
            ckpt_traces_dir, len(records),
        )

    def _write_behavioral_snapshot(self, iteration: int) -> None:
        """Sample each agent on the held-out eval set and write to disk."""
        path = getattr(self.config, "snapshot_eval_path", None)
        if not path:
            return
        try:
            from marlllm.snapshot_eval import (
                load_eval_contexts, collect_snapshot, write_snapshot,
            )
        except ImportError as e:
            self._logger.warning("snapshot_eval import failed: %s", e)
            return
        try:
            eval_data = load_eval_contexts(path)
        except Exception as e:
            self._logger.warning("Failed to load snapshot_eval_path %s: %s", path, e)
            return

        tok = list(self.population.values())[0].tokenizer
        try:
            from marlllm.dialogue import chat_eos_token_ids
            eos_ids = chat_eos_token_ids(tok)
        except Exception:
            eos_ids = None

        snap = collect_snapshot(
            population=self.population,
            eval_data=eval_data,
            tokenizer=tok,
            samples_per_context=self.config.snapshot_samples_per_context,
            max_new_tokens=self.config.snapshot_max_new_tokens,
            temperature=self.config.temperature,
            eos_token_ids=eos_ids,
            sampling_engine=self.sampling_engine,
        )
        out = write_snapshot(
            output_dir=Path(self.config.output_dir),
            iteration=iteration,
            snapshot=snap,
            eval_set_name=Path(path).name,
            temperature=self.config.temperature,
            samples_per_context=self.config.snapshot_samples_per_context,
        )
        self._logger.info("Behavioural snapshot saved: %s", out)

    # ------------------------------------------------------------------ #
    # Variance-decomposition diagnostic                                    #
    # ------------------------------------------------------------------ #

    def _should_run_var_decomp(self, iteration: int) -> bool:
        if not getattr(self.config, "var_decomp_enabled", False):
            return False
        switch = int(self.config.var_decomp_switch_iter)
        cadence = (
            self.config.var_decomp_n_eval_early if iteration <= switch
            else self.config.var_decomp_n_eval_late
        )
        cadence = max(1, int(cadence))
        return iteration % cadence == 0

    def _var_decomp_contexts_path(self) -> Path:
        cfg_path = getattr(self.config, "var_decomp_eval_contexts_path", None)
        if cfg_path:
            return Path(cfg_path)
        run_name = Path(self.config.output_dir).name or "run"
        return Path("data/eval_contexts") / f"{run_name}.var_decomp.jsonl"

    def _ensure_var_decomp_contexts(self) -> list:
        """Load held-out contexts from disk, harvesting cooperatively if missing.

        Under DDP every rank must call this in lockstep — the cooperative
        harvest path uses ``all_gather_object`` so the file is built from
        each rank's slice of the work. Rank 0 alone is too slow on 7B
        (the initial 32-episode harvest exceeds the NCCL watchdog timeout).
        """
        if self._var_decomp_contexts is not None:
            return self._var_decomp_contexts

        from marlllm.variance_decomposition import (
            EvalContext, build_eval_contexts, load_eval_contexts, write_eval_contexts,
        )
        path = self._var_decomp_contexts_path()

        if path.exists():
            if self.is_main_rank:
                self._logger.info("Loading variance-decomp contexts from %s", path)
            self._var_decomp_contexts = load_eval_contexts(path)
            return self._var_decomp_contexts

        # Cooperative harvest: split n_contexts across ranks. Each rank uses
        # a disjoint seed so contexts don't duplicate. Then gather + (rank 0)
        # write + everyone caches the same final list.
        target_total = int(self.config.var_decomp_n_contexts)
        world = max(1, self.ddp_world_size)
        n_per_rank = (target_total + world - 1) // world

        if self.is_main_rank:
            self._logger.info(
                "Building %d variance-decomp contexts cooperatively across %d ranks "
                "(%d per rank, saving to %s)",
                target_total, world, n_per_rank, path,
            )

        local = build_eval_contexts(
            population=self.population,
            environments=self.environments,
            tokeniser=self.tokeniser,
            config=self.config,
            n_contexts=n_per_rank,
            # Disjoint seeds across ranks so harvested contexts differ.
            rng_seed=self.config.seed + 7919 * self.ddp_rank,
            eos_token_ids=self._eos_token_ids,
        )
        local_json = [c.to_json() for c in local]

        if self.ddp:
            import torch.distributed as dist
            gathered: list = [None] * self.ddp_world_size
            dist.all_gather_object(gathered, local_json)
            all_json = [d for sub in gathered for d in (sub or [])]
        else:
            all_json = list(local_json)

        # Truncate to exactly target_total so the resulting set size is
        # deterministic regardless of world_size.
        all_json = all_json[:target_total]
        contexts = [EvalContext.from_json(d) for d in all_json]

        if self.is_main_rank:
            write_eval_contexts(contexts, path)
        if self.ddp:
            import torch.distributed as dist
            dist.barrier()  # wait for rank 0 to finish writing before anyone proceeds

        self._var_decomp_contexts = contexts
        return contexts

    def _run_var_decomp(self, iteration: int, metrics_inout: dict) -> None:
        """Run the variance-decomposition eval cooperatively across DDP ranks.

        Each rank evaluates its slice of contexts (``contexts[rank::world]``).
        Per-context arrays are gathered to rank 0 via ``all_gather_object``;
        rank 0 then aggregates and writes to ``variance_decomposition.jsonl``
        and folds scalar metrics into the iteration's metrics dict.
        """
        from marlllm.variance_decomposition import (
            variance_decomposition, aggregate_decomposition,
        )

        contexts = self._ensure_var_decomp_contexts()
        if not contexts:
            return

        # Slice across ranks. Order is preserved so per-context output stays
        # interpretable when we re-merge on rank 0.
        my_contexts = contexts[self.ddp_rank :: max(1, self.ddp_world_size)]

        t0 = time.time()
        try:
            local_results = variance_decomposition(
                population=self.population,
                environments=self.environments,
                eval_contexts=my_contexts,
                tokeniser=self.tokeniser,
                config=self.config,
                K=self.config.var_decomp_K,
                M=self.config.var_decomp_M,
                max_continuation_steps=self.config.var_decomp_max_continuation_steps,
                eos_token_ids=self._eos_token_ids,
                pad_token_id=self._pad_token_id(),
            )
        except Exception as e:
            self._logger.warning(
                "Variance-decomposition eval failed on rank %d: %s",
                self.ddp_rank, e,
            )
            local_results = {}

        # Strip wall_time from local results before gathering — we'll
        # recompute the wall-clock aggregate (max across ranks) on rank 0.
        for v in local_results.values():
            v.pop("wall_time_s", None)

        if self.ddp:
            import torch.distributed as dist
            gathered: list = [None] * self.ddp_world_size
            dist.all_gather_object(gathered, local_results)
        else:
            gathered = [local_results]

        elapsed = time.time() - t0

        if not self.is_main_rank:
            return

        # Merge per-rank results: each rank may have processed different
        # focal_pop_names. Re-aggregate from the merged per-context arrays.
        merged: dict[str, dict[str, list]] = {}
        for rank_results in gathered:
            if not rank_results:
                continue
            for pop_name, agg in rank_results.items():
                pc = agg.get("per_context", {})
                m = merged.setdefault(pop_name, {
                    "context_id": [], "signal": [], "noise": [],
                    "mean_return": [], "n_rollouts": [],
                })
                m["context_id"].extend(pc.get("context_id", []))
                m["signal"].extend(pc.get("signal", []))
                m["noise"].extend(pc.get("noise", []))
                m["mean_return"].extend(pc.get("mean_return", []))
                m["n_rollouts"].extend(pc.get("n_rollouts", []))

        if not merged:
            return

        results: dict[str, dict] = {}
        for pop_name, m in merged.items():
            agg = aggregate_decomposition(m["signal"], m["noise"])
            agg["per_context"] = m
            results[pop_name] = agg

        # Fold scalar metrics into the iteration's metrics dict.
        for pop_name, agg in results.items():
            for key in ("signal", "noise", "snr", "log_signal", "log_noise", "log_snr"):
                metrics_inout[f"eval/{pop_name}/{key}"] = float(agg[key])
            metrics_inout[f"eval/{pop_name}/n_contexts"] = int(agg["n_contexts"])
        metrics_inout["eval/var_decomp/wall_time_s"] = elapsed

        # Per-context detail to its own JSONL for plot_variance_decomposition.py.
        record = {"iteration": iteration, "wall_time_s": elapsed, "agents": {}}
        for pop_name, agg in results.items():
            record["agents"][pop_name] = {
                "signal": agg["signal"],
                "noise": agg["noise"],
                "snr": agg["snr"],
                "log_signal": agg["log_signal"],
                "log_noise": agg["log_noise"],
                "log_snr": agg["log_snr"],
                "n_contexts": agg["n_contexts"],
                "per_context": agg["per_context"],
            }
        with open(self._var_decomp_metrics_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        self._logger.info(
            "var_decomp iter=%d %s | %.1fs",
            iteration,
            " ".join(
                f"{n}/snr={results[n]['snr']:.3g}" for n in sorted(results)
            ),
            elapsed,
        )

    def _write_metrics_jsonl(self, metrics: dict) -> None:
        with open(self._metrics_path, "a") as f:
            f.write(json.dumps(metrics) + "\n")

    def _auto_plot_training_curves(self) -> None:
        """Regenerate training_curves.png in the experiment directory.

        Runs only on rank 0 (callers already gate on ``is_main_rank``). Wrapped
        in try/except so a plotting failure never kills training. Plotting cost
        scales with the number of logged iterations and runs at checkpoint
        cadence only, so the overhead is negligible compared to a checkpoint.
        """
        try:
            import sys
            scripts_dir = str(Path(__file__).resolve().parent.parent / "scripts")
            if scripts_dir not in sys.path:
                sys.path.insert(0, scripts_dir)
            from plot_results import plot_experiment  # type: ignore
            plot_experiment(Path(self.config.output_dir))
        except Exception as e:
            self._logger.warning(f"Auto-plot failed: {e}")

    def _log_metrics(self, iteration: int, metrics: dict) -> None:
        elapsed = metrics.get("wall_time", 0.0)
        parts = [
            f"iter {iteration:5d}/{self.config.num_iterations} | {elapsed:7.1f}s",
            f"loss {metrics.get('total_loss', 0.0):.4f}",
            f"success {metrics.get('success_rate', 0.0):.3f}",
            f"pairs {int(metrics.get('unique_pairings', 0))}",
        ]
        env_names = [s.name for s in self.environments]
        if len(env_names) > 1:
            counts = [
                f"{name}:{int(metrics.get(f'env_episodes/{name}', 0))}"
                for name in env_names
            ]
            parts.append("envs[" + " ".join(counts) + "]")
        for name in self.population:
            ret = metrics.get(f"{name}/mean_return")
            if ret is not None:
                parts.append(f"{name}/ret {ret:.4f}")
        for name in [s.name for s in self.environments]:
            mp = metrics.get(f"gen/{name}/mean_post_tokens")
            if mp is None:
                continue
            mt  = metrics.get(f"gen/{name}/mean_thinking_tokens", 0.0)
            ma  = metrics.get(f"gen/{name}/mean_action_tokens", 0.0)
            me  = metrics.get(f"gen/{name}/mean_episode_total_tokens", 0.0)
            tr  = metrics.get(f"gen/{name}/total_truncation_rate", 0.0)
            pr  = metrics.get(f"gen/{name}/post_truncation_rate", 0.0)
            parts.append(
                f"{name}/tok act{ma:.0f} think{mt:.0f} post{mp:.0f} "
                f"ep{me:.0f} trunc(t{tr:.2f}/p{pr:.2f})"
            )
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
