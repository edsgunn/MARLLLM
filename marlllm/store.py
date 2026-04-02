"""
TrajectoryStore ABC and OnPolicyStore implementation.

OnPolicyStore is a simple list buffer suitable for REINFORCE and PPO.
It collects trajectories from each episode, serves them for the update step,
then clears. No replay, no prioritisation.
"""
from __future__ import annotations

import uuid
from abc import ABC, abstractmethod

from marlllm.types import Trajectory


class TrajectoryStore(ABC):
    @abstractmethod
    def store(self, trajectory: Trajectory) -> str:
        """Store a completed trajectory. Returns a trajectory_id string."""
        ...

    @abstractmethod
    def sample(self, batch_size: int, agent_id: str | None = None) -> list[Trajectory]:
        """
        Return up to batch_size trajectories, optionally filtered by agent_id.
        In an on-policy store this returns all stored trajectories (ignoring
        batch_size if fewer are available) since on-policy methods must use all
        collected data before the update.
        """
        ...

    @abstractmethod
    def clear(self) -> None:
        """Flush all stored trajectories (call after each policy update)."""
        ...

    @abstractmethod
    def __len__(self) -> int: ...


class OnPolicyStore(TrajectoryStore):
    """
    Simple in-memory buffer for on-policy training.
    Thread-safety is not a concern for the single-process MVP.
    """

    def __init__(self) -> None:
        self._store: dict[str, Trajectory] = {}

    def store(self, trajectory: Trajectory) -> str:
        tid = str(uuid.uuid4())
        self._store[tid] = trajectory
        return tid

    def sample(self, batch_size: int, agent_id: str | None = None) -> list[Trajectory]:
        trajs = list(self._store.values())
        if agent_id is not None:
            trajs = [t for t in trajs if agent_id in t.agent_ids_present]
        return trajs[:batch_size]

    def clear(self) -> None:
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)
