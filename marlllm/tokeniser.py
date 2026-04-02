"""
Tokeniser ABC and TextTokeniser implementation.

The Tokeniser is the modality boundary: it converts raw environment
observations/actions to/from token IDs and assembles the flat token
sequence (with TokenType mask) that the model and loss functions consume.

Trajectories contain only OBS and ACT steps. Character prompts are the
agent's concern and are not tracked in the trajectory.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from transformers import PreTrainedTokenizerBase

from marlllm.types import EpisodeStep, Trajectory, TokenType


class Tokeniser(ABC):
    @abstractmethod
    def encode_observation(self, observation: Any) -> list[int]:
        """Convert a raw environment observation to token IDs."""
        ...

    @abstractmethod
    def decode_action(self, token_ids: list[int]) -> Any:
        """Convert model-generated token IDs to an environment action."""
        ...

    @abstractmethod
    def encode_prompt(self, prompt: str) -> list[int]:
        """
        Tokenise a character prompt string.
        Used by the Trainer to build per-agent context windows for act() calls.
        Not stored in the trajectory.
        """
        ...

    @abstractmethod
    def build_trajectory(
        self,
        episode_history: list[EpisodeStep],
        agent_ids_present: list[str],
    ) -> Trajectory:
        """
        Assemble a Trajectory from an episode's OBS/ACT steps.

        Steps are taken in chronological order. No prompt prefix is added —
        character prompts are the agent's concern and are not tracked here.
        The agent_mask in the resulting RolloutBatch routes each ACT position
        to the correct agent's loss computation.
        """
        ...


class TextTokeniser(Tokeniser):
    """
    Tokeniser for text-only environments.

    Wraps the HuggingFace tokenizer from the agent's model. Observations and
    actions are decoded to/from UTF-8 strings.
    """

    def __init__(self, hf_tokenizer: PreTrainedTokenizerBase) -> None:
        self._tok = hf_tokenizer

    def encode_observation(self, observation: Any) -> list[int]:
        if isinstance(observation, str):
            text = observation
        elif isinstance(observation, (bytes, bytearray)):
            text = observation.decode("utf-8")
        elif isinstance(observation, list):
            # Already token IDs (e.g. from CountingEnv)
            return [int(t) for t in observation]
        elif observation is None or (hasattr(observation, '__len__') and len(observation) == 0):
            return []
        else:
            text = str(observation)
        return self._tok.encode(text, add_special_tokens=False)

    def decode_action(self, token_ids: list[int]) -> Any:
        return self._tok.decode(token_ids, skip_special_tokens=True)

    def encode_prompt(self, prompt: str) -> list[int]:
        return self._tok.encode(prompt, add_special_tokens=False)

    def build_trajectory(
        self,
        episode_history: list[EpisodeStep],
        agent_ids_present: list[str],
    ) -> Trajectory:
        return Trajectory(
            steps=list(episode_history),
            agent_ids_present=agent_ids_present,
        )
