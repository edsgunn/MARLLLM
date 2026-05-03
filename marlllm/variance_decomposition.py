"""
Variance-decomposition diagnostic for the surprise-minimisation training loop.

For a fixed held-out context x_<t the total return variance decomposes as

    Var[G_t | x_<t] = Var_a[ E[G_t | a, x_<t] ]   <- SIGNAL (action-conditional)
                    + E_a[ Var[G_t | a, x_<t] ]   <- NOISE  (within-action)

This module estimates both terms by sampling, for each held-out context,
K candidate actions and M environment continuations per (context, action),
then computing G_t under the agent's *own* current model.

Pipeline (Option A — env replay):
  1. ``build_eval_contexts`` runs a few normal rollouts, harvests states at
     random ticks where the focal agent is about to act, and saves the
     prefix of action tokens leading to that state. Saved as JSONL.
  2. ``variance_decomposition`` loads each context, deep-copies the env,
     reset(seed)s, replays the prefix verbatim, then runs K*M parallel
     continuations from there. Surprise/returns are computed via the
     same helpers used by the training loss so the measurement is
     self-consistent with what REINFORCE actually optimises.

Test seam: ``compute_decomposition_from_returns`` is the pure-numerical
estimator (no rollouts) and is exercised directly by the unit tests with
synthetic G_t arrays — it isolates the (M-1)/(K-1) Bessel-correct
variance formula from any model/env machinery.
"""
from __future__ import annotations

import copy
import json
import logging
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from marlllm.loss import _compute_obs_surprises, _compute_returns
from marlllm.types import EpisodeStep, RolloutBatch, TokenType, Trajectory

_LOG = logging.getLogger(__name__)

# Floor for log-scale metrics. Signal/noise are non-negative variances —
# clamp before taking logs so a deterministic-policy degenerate case
# doesn't produce -inf.
_EPS = 1e-12


# --------------------------------------------------------------------------- #
# Eval-context dataclass + persistence                                         #
# --------------------------------------------------------------------------- #


@dataclass
class EvalContext:
    """A frozen mid-episode state that we can deterministically replay.

    To reproduce the state: deep-copy ``environments[env_name].env``,
    ``env.reset(seed=env_seed)``, then ``env.step(action_tokens)`` for each
    entry in ``prefix_actions`` in order. After replay, the env's
    ``agent_selection`` will be ``focal_env_role`` and the focal agent's
    context window will contain everything observed up to that point.
    """
    context_id: str
    env_name: str
    env_seed: int
    role_to_name: dict[str, str]    # env_role -> population member name
    focal_env_role: str             # role about to act after replay
    focal_pop_name: str             # population member at focal role
    prompts: dict[str, str]         # pop_name -> chosen prompt text (per-episode variant)
    prefix_actions: list[dict]      # ordered: [{"env_role": str, "tokens": [int, ...]}]

    def to_json(self) -> dict:
        return {
            "context_id": self.context_id,
            "env_name": self.env_name,
            "env_seed": self.env_seed,
            "role_to_name": self.role_to_name,
            "focal_env_role": self.focal_env_role,
            "focal_pop_name": self.focal_pop_name,
            "prompts": self.prompts,
            "prefix_actions": self.prefix_actions,
        }

    @staticmethod
    def from_json(d: dict) -> "EvalContext":
        return EvalContext(
            context_id=d["context_id"],
            env_name=d["env_name"],
            env_seed=int(d["env_seed"]),
            role_to_name=dict(d["role_to_name"]),
            focal_env_role=d["focal_env_role"],
            focal_pop_name=d["focal_pop_name"],
            prompts=dict(d["prompts"]),
            prefix_actions=[
                {"env_role": p["env_role"], "tokens": [int(t) for t in p["tokens"]]}
                for p in d["prefix_actions"]
            ],
        )


def write_eval_contexts(contexts: list[EvalContext], path: Path | str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        for c in contexts:
            f.write(json.dumps(c.to_json()) + "\n")


def load_eval_contexts(path: Path | str) -> list[EvalContext]:
    out: list[EvalContext] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(EvalContext.from_json(json.loads(line)))
    return out


# --------------------------------------------------------------------------- #
# Pure variance estimator (test seam)                                          #
# --------------------------------------------------------------------------- #


def compute_decomposition_from_returns(
    returns_KM: torch.Tensor | list[list[float]],
) -> tuple[float, float]:
    """Compute (signal, noise) from a K x M tensor of returns at fixed context.

    signal = unbiased Var over K of E[G | a_k] (means across M)
    noise  = mean over K of unbiased Var over M of G | a_k

    Bessel corrections (M-1) and (K-1) are applied per the spec because
    K and M are small. Degenerate cases (K<2 or M<2) return 0.0 for the
    affected term — a single sample carries no variance information.
    """
    if not isinstance(returns_KM, torch.Tensor):
        returns_KM = torch.tensor(returns_KM, dtype=torch.float64)
    else:
        returns_KM = returns_KM.to(dtype=torch.float64)
    if returns_KM.ndim != 2:
        raise ValueError(f"returns_KM must be 2D [K, M]; got shape {tuple(returns_KM.shape)}")

    K, M = returns_KM.shape
    if M >= 2:
        var_per_action = returns_KM.var(dim=1, unbiased=True)        # (K,)
        noise = float(var_per_action.mean().item())
    else:
        noise = 0.0
    if K >= 2:
        mean_per_action = returns_KM.mean(dim=1)                      # (K,)
        signal = float(mean_per_action.var(unbiased=True).item())
    else:
        signal = 0.0
    return signal, noise


def aggregate_decomposition(
    per_context_signal: list[float],
    per_context_noise: list[float],
) -> dict[str, float]:
    """Aggregate per-context signal/noise into the metrics emitted to the logger."""
    if not per_context_signal:
        return {
            "signal": 0.0, "noise": 0.0, "snr": 0.0,
            "log_signal": math.log(_EPS), "log_noise": math.log(_EPS),
            "log_snr": 0.0, "n_contexts": 0,
        }
    signal = sum(per_context_signal) / len(per_context_signal)
    noise = sum(per_context_noise) / len(per_context_noise)
    snr = signal / max(noise, _EPS)
    return {
        "signal": signal,
        "noise": noise,
        "snr": snr,
        "log_signal": math.log(max(signal, _EPS)),
        "log_noise": math.log(max(noise, _EPS)),
        "log_snr": math.log(max(snr, _EPS)),
        "n_contexts": len(per_context_signal),
    }


# --------------------------------------------------------------------------- #
# Replay helpers (mirror PopulationTrainer's per-tick bookkeeping)             #
# --------------------------------------------------------------------------- #


def _init_env_contexts(
    env,
    role_to_name: dict[str, str],
    prompts: dict[str, str],
    population: dict[str, Any],
    tokeniser,
) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
    """Build per-role context windows from prompts (matches PopulationTrainer)."""
    full_ctx = bool(getattr(env, "obs_is_full_context", False))
    contexts: dict[str, list[int]] = {}
    prompt_ids: dict[str, list[int]] = {}
    for env_role, pop_name in role_to_name.items():
        if full_ctx:
            prompt_ids.setdefault(pop_name, [])
            contexts[env_role] = []
            continue
        formatter = population[pop_name].context_formatter
        if pop_name not in prompt_ids:
            pids = tokeniser.encode_prompt(prompts.get(pop_name, ""))
            prompt_ids[pop_name] = pids
            contexts[env_role] = list(formatter.wrap_prompt(pids))
        else:
            contexts[env_role] = list(formatter.wrap_prompt(prompt_ids[pop_name]))
    return contexts, prompt_ids


def _ingest_obs(
    env,
    env_role: str,
    pop_name: str,
    contexts: dict[str, list[int]],
    history: list[EpisodeStep],
    population: dict[str, Any],
    tokeniser,
) -> None:
    """Read env.last() for the agent that's about to act and update bookkeeping.

    Mirrors the OBS-handling block in PopulationTrainer._collect_episodes_batched.
    """
    full_ctx = bool(getattr(env, "obs_is_full_context", False))
    obs, _r, _term, _trunc, _info = env.last()
    obs_ids = tokeniser.encode_observation(obs)
    if not obs_ids:
        return
    history.append(EpisodeStep(
        agent_id=pop_name,
        token_ids=list(obs_ids),
        token_type=TokenType.OBS,
        log_probs=[],
        info={},
    ))
    if full_ctx:
        contexts[env_role] = list(obs_ids)
    else:
        formatter = population[pop_name].context_formatter
        contexts[env_role].extend(formatter.wrap_observation(obs_ids))


def _record_action(
    env_role: str,
    pop_name: str,
    act_ids: list[int],
    contexts: dict[str, list[int]],
    history: list[EpisodeStep],
    population: dict[str, Any],
    log_probs: list[float] | None = None,
) -> None:
    """Append an action step to history and extend the role context."""
    formatter = population[pop_name].context_formatter
    history.append(EpisodeStep(
        agent_id=pop_name,
        token_ids=list(act_ids),
        token_type=TokenType.ACT,
        log_probs=list(log_probs) if log_probs is not None else [0.0] * len(act_ids),
        info={},
    ))
    contexts[env_role].extend(formatter.wrap_action(act_ids))


def _replay_to_focal(
    ec: EvalContext,
    env_template,
    population: dict[str, Any],
    tokeniser,
) -> tuple[Any, dict[str, list[int]], dict[str, list[int]], list[EpisodeStep]]:
    """Replay an EvalContext deterministically.

    Returns: ``(env, contexts, prompt_ids, history)`` where ``env.agent_selection``
    is ``ec.focal_env_role`` and ``contexts[focal_env_role]`` is the focal
    agent's full input window at the moment of decision.
    """
    env = copy.deepcopy(env_template)
    if hasattr(env, "_tok"):
        env._tok = env_template._tok  # avoid duplicating tokenizer state
    env.reset(seed=ec.env_seed)
    contexts, prompt_ids = _init_env_contexts(
        env, ec.role_to_name, ec.prompts, population, tokeniser,
    )
    history: list[EpisodeStep] = []
    for prefix in ec.prefix_actions:
        env_role = env.agent_selection
        if env_role != prefix["env_role"]:
            raise RuntimeError(
                f"Replay drift in context {ec.context_id!r}: env at role "
                f"{env_role!r}, prefix expects {prefix['env_role']!r}"
            )
        pop_name = ec.role_to_name[env_role]
        _ingest_obs(env, env_role, pop_name, contexts, history, population, tokeniser)
        act_ids = list(prefix["tokens"])
        _record_action(env_role, pop_name, act_ids, contexts, history, population)
        env.step(act_ids)

    # Now we should be at the focal's turn. Ingest the focal's pre-action obs
    # so the trajectory captures every OBS the focal saw before deciding.
    if not env.agents or env.agent_selection != ec.focal_env_role:
        raise RuntimeError(
            f"Replay of {ec.context_id!r} did not land at focal role "
            f"{ec.focal_env_role!r} (env at {env.agent_selection!r}, "
            f"agents={env.agents})"
        )
    _ingest_obs(
        env, ec.focal_env_role, ec.focal_pop_name,
        contexts, history, population, tokeniser,
    )
    return env, contexts, prompt_ids, history


# --------------------------------------------------------------------------- #
# Harvest: build a held-out context set from the current population            #
# --------------------------------------------------------------------------- #


def build_eval_contexts(
    *,
    population: dict[str, Any],
    environments: list,            # list[EnvironmentSpec] from PopulationTrainer
    tokeniser,
    config,
    n_contexts: int = 32,
    rng_seed: int = 0,
    eos_token_ids: list[int] | None = None,
) -> list[EvalContext]:
    """Run rollouts with the *current* population, snapshot mid-episode states.

    The returned EvalContexts are independent of the population that
    produced them (the prefix is stored as raw action tokens), so they
    remain valid as agents train. We call this once at iteration 0 and
    reuse the saved file forever.

    Strategy: roll one episode at a time to completion, recording every
    action. After the episode, pick a uniformly-random tick where the
    focal role acts — that becomes one context. Repeat until we have
    ``n_contexts``.
    """
    rng = random.Random(rng_seed)
    pop_names = list(population.keys())
    if len(pop_names) < 1:
        raise ValueError("Empty population")

    # Normalise environment-spec list (allows passing bare envs through).
    from marlllm.population import EnvironmentSpec  # local import to avoid cycle
    specs: list[EnvironmentSpec] = []
    for s in environments:
        if isinstance(s, EnvironmentSpec):
            specs.append(s)
        else:
            specs.append(EnvironmentSpec(env=s, name="env"))
    env_weights = [s.weight for s in specs]

    contexts_out: list[EvalContext] = []
    seed_counter = config.seed + 10_000  # disjoint from training seed stream
    safety_budget = max(8 * n_contexts, 64)  # bail rather than spin if every episode is empty
    attempts = 0

    while len(contexts_out) < n_contexts and attempts < safety_budget:
        attempts += 1
        spec = rng.choices(specs, weights=env_weights, k=1)[0]
        env_template = spec.env
        roles = list(getattr(env_template, "possible_agents", []))

        if set(roles) == {"agent_0", "agent_1"}:
            if len(pop_names) >= 2:
                a, b = rng.sample(pop_names, 2)
            else:
                a = b = pop_names[0]
            role_to_name = {"agent_0": a, "agent_1": b}
        else:
            role_to_name = {r: r for r in roles}
            missing = [r for r in roles if r not in population]
            if missing:
                raise ValueError(
                    f"EvalContext harvest: env {spec.name!r} has roles {missing} "
                    f"not in population {pop_names}"
                )

        prompts: dict[str, str] = {}
        for env_role, pop_name in role_to_name.items():
            if pop_name in prompts:
                continue
            variants = spec.character_prompts.get(pop_name)
            if variants:
                prompts[pop_name] = rng.choice(variants)
            else:
                # Mirrors PopulationTrainer's fallback path.
                gv = config.character_prompts.get(pop_name, "")
                if isinstance(gv, list):
                    prompts[pop_name] = rng.choice(gv) if gv else ""
                else:
                    prompts[pop_name] = gv or ""

        env_seed = seed_counter
        seed_counter += 1

        # Run one episode end-to-end, recording every action and every tick's
        # current role. After completion, sampling a random "focal-acts-next"
        # tick gives us a context whose prefix is everything before it.
        env = copy.deepcopy(env_template)
        if hasattr(env, "_tok"):
            env._tok = env_template._tok
        env.reset(seed=env_seed)
        contexts_role, _prompt_ids = _init_env_contexts(
            env, role_to_name, prompts, population, tokeniser,
        )
        token_count = 0
        # Record (env_role, pop_name, act_tokens) in chronological order.
        action_log: list[dict] = []

        while env.agents:
            env_role = env.agent_selection
            pop_name = role_to_name[env_role]
            obs, _r, term, trunc, info = env.last()
            obs_ids = tokeniser.encode_observation(obs)
            if obs_ids:
                full_ctx = bool(getattr(env, "obs_is_full_context", False))
                if full_ctx:
                    contexts_role[env_role] = list(obs_ids)
                else:
                    formatter = population[pop_name].context_formatter
                    contexts_role[env_role].extend(formatter.wrap_observation(obs_ids))
                token_count += len(obs_ids)

            if term or trunc:
                env.step(None)
                continue
            if (token_count >= config.max_episode_tokens
                    and not info.get("must_act", False)):
                env.step(None)
                continue
            if pop_name not in population:
                env.step(None)
                continue

            agent = population[pop_name]
            agent.eval_mode()
            n_tokens = getattr(env, "action_token_budget", 1)
            with torch.no_grad():
                batch_ids, _batch_lps = agent.act_batch(
                    contexts=[contexts_role[env_role]],
                    n_tokens=n_tokens,
                    temperature=config.temperature,
                    eos_token_ids=eos_token_ids,
                )
            act_ids = list(batch_ids[0])
            action_log.append(
                {"env_role": env_role, "pop_name": pop_name, "tokens": act_ids}
            )
            formatter = population[pop_name].context_formatter
            contexts_role[env_role].extend(formatter.wrap_action(act_ids))
            token_count += len(act_ids)
            env.step(act_ids)

        if len(action_log) < 2:
            continue  # not enough room to leave a non-trivial prefix

        # Pick a focal tick uniformly at random over interior ticks
        # (skip tick 0 — empty prefix is fine but uninteresting; skip the
        # final tick — we want at least one continuation step possible).
        # If no interior tick exists, fall back to any tick >= 1.
        candidate_ticks = list(range(1, len(action_log) - 1)) or [0]
        chosen = rng.choice(candidate_ticks)
        chosen_step = action_log[chosen]
        prefix = [
            {"env_role": s["env_role"], "tokens": s["tokens"]}
            for s in action_log[:chosen]
        ]
        ec = EvalContext(
            context_id=f"{spec.name}_seed{env_seed}_t{chosen}",
            env_name=spec.name,
            env_seed=env_seed,
            role_to_name=role_to_name,
            focal_env_role=chosen_step["env_role"],
            focal_pop_name=chosen_step["pop_name"],
            prompts=prompts,
            prefix_actions=prefix,
        )
        contexts_out.append(ec)

    if len(contexts_out) < n_contexts:
        _LOG.warning(
            "build_eval_contexts produced only %d/%d contexts after %d attempts",
            len(contexts_out), n_contexts, attempts,
        )
    return contexts_out


# --------------------------------------------------------------------------- #
# Continuation rollout: K * M parallel envs from a replayed prefix             #
# --------------------------------------------------------------------------- #


def _continue_until_done(
    envs: list,
    role_to_name: dict[str, str],
    contexts_per_env: list[dict[str, list[int]]],
    histories_per_env: list[list[EpisodeStep]],
    active: list[bool],
    population: dict[str, Any],
    tokeniser,
    config,
    eos_token_ids: list[int] | None,
    max_steps: int,
) -> None:
    """Drive a population of partially-played envs in lockstep until done.

    Mirrors PopulationTrainer._collect_episodes_batched's per-tick loop, but
    starts from caller-supplied env state and per-env contexts. Mutates
    ``contexts_per_env`` and ``histories_per_env`` in place.
    """
    n = len(envs)
    steps_taken = [0] * n

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
            if max_steps > 0 and steps_taken[k] >= max_steps:
                active[k] = False
                continue

            env_role = env.agent_selection
            pop_name = role_to_name[env_role]
            obs, _r, term, trunc, info = env.last()
            full_ctx = bool(getattr(env, "obs_is_full_context", False))
            obs_ids = tokeniser.encode_observation(obs)
            if obs_ids:
                histories_per_env[k].append(EpisodeStep(
                    agent_id=pop_name,
                    token_ids=list(obs_ids),
                    token_type=TokenType.OBS,
                    log_probs=[],
                    info={},
                ))
                if full_ctx:
                    contexts_per_env[k][env_role] = list(obs_ids)
                else:
                    formatter = population[pop_name].context_formatter
                    contexts_per_env[k][env_role].extend(
                        formatter.wrap_observation(obs_ids)
                    )

            if term or trunc:
                null_indices.append(k)
            elif pop_name not in population:
                null_indices.append(k)
            else:
                act_groups.setdefault(pop_name, []).append(k)

        for k in null_indices:
            envs[k].step(None)
            steps_taken[k] += 1
            if not envs[k].agents:
                active[k] = False

        for pop_name, env_indices in act_groups.items():
            agent = population[pop_name]
            agent.eval_mode()
            batch_ctx = [
                contexts_per_env[k][envs[k].agent_selection] for k in env_indices
            ]
            n_tokens = max(
                getattr(envs[k], "action_token_budget", 1) for k in env_indices
            )
            with torch.no_grad():
                batch_ids, batch_lps = agent.act_batch(
                    contexts=batch_ctx,
                    n_tokens=n_tokens,
                    temperature=config.temperature,
                    eos_token_ids=eos_token_ids,
                )
            for j, k in enumerate(env_indices):
                act_ids = list(batch_ids[j])
                lps = list(batch_lps[j]) if batch_lps and j < len(batch_lps) else None
                env_role = envs[k].agent_selection
                _record_action(
                    env_role, pop_name, act_ids,
                    contexts_per_env[k], histories_per_env[k],
                    population, log_probs=lps,
                )
                envs[k].step(act_ids)
                steps_taken[k] += 1
                if not envs[k].agents:
                    active[k] = False


# --------------------------------------------------------------------------- #
# Trajectory builder + return-at-position extraction                           #
# --------------------------------------------------------------------------- #


def _build_focal_trajectory(
    history: list[EpisodeStep],
    focal_pop_name: str,
    population: dict[str, Any],
    prompts: dict[str, str],
    tokeniser,
    config,
) -> tuple[Trajectory, int]:
    """Build a focal-perspective Trajectory and return ``(traj, target_act_pos)``.

    ``target_act_pos`` is the index of the first focal ACT step *after* the
    replayed prefix — i.e. the action whose return G_t we measure. The
    caller has guaranteed (by prefix construction) that exactly one
    "first focal action after prefix" exists in ``history``.
    """
    formatter = population[focal_pop_name].context_formatter
    steps: list[EpisodeStep] = []

    # Prepend prompt as in PopulationTrainer._build_agent_trajectories.
    pids = tokeniser.encode_prompt(prompts.get(focal_pop_name, ""))
    if pids:
        prompt_type = (
            TokenType.OBS if config.prompt_as_observation else TokenType.PAD
        )
        steps.append(EpisodeStep(
            agent_id=focal_pop_name,
            token_ids=list(formatter.wrap_prompt(pids)),
            token_type=prompt_type,
            log_probs=[],
            info={},
        ))

    target_act_pos = -1
    for step in history:
        if step.agent_id != focal_pop_name:
            continue
        if step.token_type == TokenType.OBS:
            fids = formatter.wrap_observation(step.token_ids)
            lps = step.log_probs
        elif step.token_type == TokenType.ACT:
            fids = formatter.wrap_action(step.token_ids)
            lps = list(step.log_probs) + [0.0] * (len(fids) - len(step.token_ids))
        else:
            fids = step.token_ids
            lps = step.log_probs
        new_step = EpisodeStep(
            agent_id=focal_pop_name,
            token_ids=list(fids),
            token_type=step.token_type,
            log_probs=lps,
            info={},
        )
        # Mark the FIRST focal ACT step we encounter — the focal_first_act
        # we're measuring G_t at.
        if target_act_pos < 0 and step.token_type == TokenType.ACT:
            target_act_pos = len(steps)
        steps.append(new_step)

    traj = tokeniser.build_trajectory(
        episode_history=steps,
        agent_ids_present=list(population.keys()),
    )
    return traj, target_act_pos


def _g_t_at_first_act(
    traj: Trajectory,
    target_act_step_idx: int,
    agent,
    agent_index: dict[str, int],
    pad_id: int,
    config,
) -> float:
    """Run agent.evaluate() on the trajectory and return G_t at the target act step.

    Surprises and returns are computed with the same helpers used by the
    training loss (``_compute_obs_surprises`` + ``_compute_returns``) so
    the diagnostic measures the exact quantity REINFORCE is estimating.
    """
    if target_act_step_idx < 0:
        return 0.0

    batch = RolloutBatch.from_trajectories([traj], agent_index, pad_id)
    dev = agent.device
    input_ids = batch.input_ids.to(dev)
    attn = batch.attention_mask.to(dev)
    types = batch.token_type_mask.to(dev)

    actual_len = int(attn.sum(dim=1).max())
    if actual_len < input_ids.shape[1]:
        input_ids = input_ids[:, :actual_len]
        attn = attn[:, :actual_len]
        types = types[:, :actual_len]

    agent.eval_mode()
    with torch.no_grad():
        logits, _values = agent.evaluate(input_ids, attn)
        surprises = _compute_obs_surprises(logits, input_ids, types)
        returns = _compute_returns(surprises, types, config.gamma)

    # Find the token-position of the target ACT step's first token.
    pos = 0
    for i, step in enumerate(traj.steps):
        if i == target_act_step_idx:
            break
        pos += len(step.token_ids)
    if pos >= returns.shape[1]:
        return 0.0
    return float(returns[0, pos].item())


# --------------------------------------------------------------------------- #
# Top-level entry: variance_decomposition over all contexts                    #
# --------------------------------------------------------------------------- #


def variance_decomposition(
    *,
    population: dict[str, Any],
    environments: list,            # list[EnvironmentSpec]
    eval_contexts: list[EvalContext],
    tokeniser,
    config,
    K: int = 8,
    M: int = 4,
    max_continuation_steps: int = 0,
    eos_token_ids: list[int] | None = None,
    pad_token_id: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Estimate signal/noise decomposition for each population member.

    Returns a dict ``{pop_name: metrics}`` where ``metrics`` carries the
    aggregated signal/noise/SNR scalars plus per-context arrays for
    diagnostic plotting.

    Cost is O(n_contexts * K * M * env_step_cost). The K * M continuations
    for a single context are batched into one ``act_batch`` call per turn
    so the GPU sees full-width batches.
    """
    if not eval_contexts:
        return {}

    # Index environments by name for replay.
    from marlllm.population import EnvironmentSpec
    env_by_name: dict[str, EnvironmentSpec] = {}
    for s in environments:
        if isinstance(s, EnvironmentSpec):
            env_by_name[s.name] = s
        else:
            env_by_name.setdefault("env", EnvironmentSpec(env=s, name="env"))

    agent_index = {name: i for i, name in enumerate(population.keys())}

    if pad_token_id is None:
        primary = next(iter(population.values()))
        tok = getattr(primary, "tokenizer", None)
        pad_token_id = (
            tok.pad_token_id if tok is not None and tok.pad_token_id is not None
            else (tok.eos_token_id if tok is not None else 0)
        )

    # Bucket contexts by focal pop_name so we can report per-agent stats.
    per_agent: dict[str, dict[str, list]] = {}
    for ec in eval_contexts:
        per_agent.setdefault(ec.focal_pop_name, {
            "context_id": [], "signal": [], "noise": [],
            "mean_return": [], "n_rollouts": [],
        })

    t0 = time.time()
    for ec in eval_contexts:
        spec = env_by_name.get(ec.env_name)
        if spec is None:
            _LOG.warning("EvalContext %s: unknown env_name %r — skipping",
                         ec.context_id, ec.env_name)
            continue

        # 1) Replay deterministically.
        try:
            replayed_env, ctx_after_replay, _pids, history_after_replay = _replay_to_focal(
                ec, spec.env, population, tokeniser,
            )
        except RuntimeError as e:
            _LOG.warning("EvalContext %s: replay failed: %s", ec.context_id, e)
            continue

        focal = population[ec.focal_pop_name]
        focal.eval_mode()
        focal_ctx_at_decision = list(ctx_after_replay[ec.focal_env_role])

        # 2) Sample K focal actions from the same context (one batched call).
        n_tokens = getattr(replayed_env, "action_token_budget", 1)
        with torch.no_grad():
            k_action_ids, _k_lps = focal.act_batch(
                contexts=[focal_ctx_at_decision] * K,
                n_tokens=n_tokens,
                temperature=config.temperature,
                eos_token_ids=eos_token_ids,
            )

        # 3) For each of the K actions, run M continuations in lockstep.
        # We batch all K * M envs together so act_batch sees the widest batch.
        n_branches = K * M
        envs = []
        contexts_per_env: list[dict[str, list[int]]] = []
        histories_per_env: list[list[EpisodeStep]] = []
        for k in range(K):
            for _m in range(M):
                env_km = copy.deepcopy(replayed_env)
                if hasattr(env_km, "_tok"):
                    env_km._tok = replayed_env._tok
                envs.append(env_km)
                contexts_per_env.append({
                    role: list(ids) for role, ids in ctx_after_replay.items()
                })
                histories_per_env.append([
                    EpisodeStep(
                        agent_id=h.agent_id,
                        token_ids=list(h.token_ids),
                        token_type=h.token_type,
                        log_probs=list(h.log_probs),
                        info=dict(h.info),
                    )
                    for h in history_after_replay
                ])

        # 3a) Force the first focal action per branch.
        for idx in range(n_branches):
            k = idx // M
            act_ids = list(k_action_ids[k])
            env_role = envs[idx].agent_selection
            assert env_role == ec.focal_env_role, (
                f"branch {idx}: expected focal role {ec.focal_env_role!r}, "
                f"got {env_role!r}"
            )
            _record_action(
                env_role, ec.focal_pop_name, act_ids,
                contexts_per_env[idx], histories_per_env[idx],
                population,
            )
            envs[idx].step(act_ids)

        # 3b) Continue rolling out partner + downstream focal turns.
        active = [bool(envs[i].agents) for i in range(n_branches)]
        _continue_until_done(
            envs=envs,
            role_to_name=ec.role_to_name,
            contexts_per_env=contexts_per_env,
            histories_per_env=histories_per_env,
            active=active,
            population=population,
            tokeniser=tokeniser,
            config=config,
            eos_token_ids=eos_token_ids,
            max_steps=max_continuation_steps,
        )

        # 4) Compute G_t at each branch's first-focal-action position.
        returns_KM = torch.zeros(K, M, dtype=torch.float64)
        for idx in range(n_branches):
            traj, target_pos = _build_focal_trajectory(
                histories_per_env[idx],
                focal_pop_name=ec.focal_pop_name,
                population=population,
                prompts=ec.prompts,
                tokeniser=tokeniser,
                config=config,
            )
            g_t = _g_t_at_first_act(
                traj, target_pos, focal, agent_index, pad_token_id, config,
            )
            returns_KM[idx // M, idx % M] = g_t

        signal_c, noise_c = compute_decomposition_from_returns(returns_KM)
        bucket = per_agent[ec.focal_pop_name]
        bucket["context_id"].append(ec.context_id)
        bucket["signal"].append(signal_c)
        bucket["noise"].append(noise_c)
        bucket["mean_return"].append(float(returns_KM.mean().item()))
        bucket["n_rollouts"].append(n_branches)

    # 5) Aggregate per agent.
    out: dict[str, dict[str, Any]] = {}
    for pop_name, data in per_agent.items():
        agg = aggregate_decomposition(data["signal"], data["noise"])
        agg["per_context"] = {
            "context_id": data["context_id"],
            "signal": data["signal"],
            "noise": data["noise"],
            "mean_return": data["mean_return"],
            "n_rollouts": data["n_rollouts"],
        }
        agg["wall_time_s"] = time.time() - t0
        out[pop_name] = agg
    return out


__all__ = [
    "EvalContext",
    "write_eval_contexts",
    "load_eval_contexts",
    "compute_decomposition_from_returns",
    "aggregate_decomposition",
    "build_eval_contexts",
    "variance_decomposition",
]
