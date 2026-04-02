"""
Core data structures for MARLLLM.

The TokenType mask is the central routing mechanism: all loss functions select
positions using `mask == OBS` or `mask == ACT`. The two meaningful values are
OBS and ACT. PAD (0) is a technical sentinel used only for right-padding in
RolloutBatch — it is excluded from all losses because loss functions check for
OBS or ACT explicitly.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

import torch


class TokenType(IntEnum):
    PAD = 0  # padding sentinel — excluded from all losses; not a semantic token type
    OBS = 1  # environment/other-agent observation tokens — L_perc (NTP)
    ACT = 2  # this agent's action tokens — L_act + L_val (REINFORCE + value)


@dataclass
class EpisodeStep:
    """A single turn in an episode: one agent acting or one env observation."""
    agent_id: str
    token_ids: list[int]
    token_type: TokenType
    log_probs: list[float]  # sampled log-probs for ACT tokens; [] for OBS
    info: dict = field(default_factory=dict)


@dataclass
class Trajectory:
    """
    A complete episode record. Steps are in chronological order.
    Contains only OBS and ACT steps — character prompts are the agent's concern
    and are not tracked in the trajectory.
    """
    steps: list[EpisodeStep]
    agent_ids_present: list[str]

    def token_ids(self) -> list[int]:
        ids = []
        for step in self.steps:
            ids.extend(step.token_ids)
        return ids

    def type_mask(self) -> list[TokenType]:
        mask = []
        for step in self.steps:
            mask.extend([step.token_type] * len(step.token_ids))
        return mask

    def agent_mask(self) -> list[str]:
        """agent_id per token."""
        mask = []
        for step in self.steps:
            mask.extend([step.agent_id] * len(step.token_ids))
        return mask

    def act_log_probs_flat(self) -> list[float]:
        """Flat list of stored log-probs aligned to token positions; 0.0 at non-ACT."""
        result = []
        for step in self.steps:
            if step.token_type == TokenType.ACT:
                result.extend(step.log_probs)
            else:
                result.extend([0.0] * len(step.token_ids))
        return result


@dataclass
class RolloutBatch:
    """
    Padded, tensorised batch of K trajectories for the training forward pass.
    Trajectories are right-padded to T_max. Padding positions use pad_token_id
    in input_ids and TokenType.PROMPT (0) in token_type_mask.
    """
    input_ids: torch.Tensor          # (K, T_max) long
    attention_mask: torch.Tensor     # (K, T_max) long
    token_type_mask: torch.Tensor    # (K, T_max) long  — TokenType ints
    agent_id_mask: torch.Tensor      # (K, T_max) long  — agent index; -1 for PROMPT/pad
    act_log_probs_old: torch.Tensor  # (K, T_max) float — 0 at non-ACT positions

    def to(self, device: str | torch.device) -> RolloutBatch:
        return RolloutBatch(
            input_ids=self.input_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            token_type_mask=self.token_type_mask.to(device),
            agent_id_mask=self.agent_id_mask.to(device),
            act_log_probs_old=self.act_log_probs_old.to(device),
        )

    @staticmethod
    def from_trajectories(
        trajectories: list[Trajectory],
        agent_index: dict[str, int],
        pad_token_id: int,
    ) -> RolloutBatch:
        """
        Collate variable-length Trajectory objects into a right-padded batch.

        Args:
            trajectories: List of K Trajectory objects.
            agent_index: Maps agent_id -> integer index for tensor encoding.
            pad_token_id: Token ID for right-padding (usually eos_token_id).
        """
        seqs = [t.token_ids() for t in trajectories]
        type_masks = [t.type_mask() for t in trajectories]
        agent_masks = [t.agent_mask() for t in trajectories]
        lp_lists = [t.act_log_probs_flat() for t in trajectories]

        T_max = max(len(s) for s in seqs)
        K = len(trajectories)

        input_ids = torch.full((K, T_max), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros(K, T_max, dtype=torch.long)
        token_type_mask = torch.zeros(K, T_max, dtype=torch.long)  # PROMPT=0 as pad
        agent_id_mask = torch.full((K, T_max), -1, dtype=torch.long)
        act_log_probs_old = torch.zeros(K, T_max, dtype=torch.float)

        for i, (seq, tm, am, lp) in enumerate(zip(seqs, type_masks, agent_masks, lp_lists)):
            L = len(seq)
            input_ids[i, :L] = torch.tensor(seq, dtype=torch.long)
            attention_mask[i, :L] = 1
            token_type_mask[i, :L] = torch.tensor([int(t) for t in tm], dtype=torch.long)
            agent_id_mask[i, :L] = torch.tensor(
                [agent_index.get(a, -1) for a in am], dtype=torch.long
            )
            act_log_probs_old[i, :L] = torch.tensor(lp, dtype=torch.float)

        return RolloutBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_mask=token_type_mask,
            agent_id_mask=agent_id_mask,
            act_log_probs_old=act_log_probs_old,
        )
