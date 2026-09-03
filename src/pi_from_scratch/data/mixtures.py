"""Typed heterogeneous batches and reproducible mixture schedules."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch
from torch import Tensor


class SampleKind(str, Enum):
    ROBOT_ACTION = "robot_action"
    SEMANTIC = "semantic"


@dataclass(frozen=True)
class RobotActionBatch:
    """A robot batch that supports discrete and continuous action objectives."""

    observation: Tensor
    actions: Tensor
    action_tokens: Tensor
    source: str = "robot"

    def __post_init__(self) -> None:
        if self.observation.ndim != 2 or not self.observation.is_floating_point():
            raise ValueError("observation must be float with shape [batch, observation_dim]")
        if self.actions.ndim != 3 or not self.actions.is_floating_point():
            raise ValueError("actions must be float with shape [batch, horizon, action_dim]")
        if self.actions.shape[0] != self.observation.shape[0]:
            raise ValueError("observation and actions must have the same batch size")
        if self.action_tokens.shape != (self.observation.shape[0],):
            raise ValueError("action_tokens must have shape [batch]")
        if self.action_tokens.dtype != torch.long:
            raise TypeError("action_tokens must use torch.long")
        if not self.source:
            raise ValueError("robot source must be named")

    @property
    def kind(self) -> SampleKind:
        return SampleKind.ROBOT_ACTION


@dataclass(frozen=True)
class SemanticBatch:
    """A non-action batch such as subtask prediction or a web VLM task."""

    observation: Tensor
    labels: Tensor
    source: str = "semantic"

    def __post_init__(self) -> None:
        if self.observation.ndim != 2 or not self.observation.is_floating_point():
            raise ValueError("observation must be float with shape [batch, observation_dim]")
        if self.labels.shape != (self.observation.shape[0],):
            raise ValueError("labels must have shape [batch]")
        if self.labels.dtype != torch.long:
            raise TypeError("labels must use torch.long")
        if not self.source:
            raise ValueError("semantic source must be named")

    @property
    def kind(self) -> SampleKind:
        return SampleKind.SEMANTIC


MixedBatch = RobotActionBatch | SemanticBatch


@dataclass(frozen=True)
class MixtureSchedule:
    """A pre-sampled task schedule whose realized ratios can be audited."""

    kinds: tuple[SampleKind, ...]
    requested_robot_probability: float

    @classmethod
    def draw(
        cls,
        num_steps: int,
        *,
        robot_probability: float,
        seed: int,
    ) -> MixtureSchedule:
        if num_steps <= 0:
            raise ValueError("num_steps must be positive")
        if not 0.0 <= robot_probability <= 1.0:
            raise ValueError("robot_probability must lie in [0, 1]")
        generator = torch.Generator().manual_seed(seed)
        draws = torch.rand(num_steps, generator=generator)
        kinds = tuple(
            SampleKind.ROBOT_ACTION if value < robot_probability else SampleKind.SEMANTIC
            for value in draws.tolist()
        )
        return cls(kinds=kinds, requested_robot_probability=robot_probability)

    def counts(self) -> dict[str, int]:
        robot = sum(kind is SampleKind.ROBOT_ACTION for kind in self.kinds)
        return {
            SampleKind.ROBOT_ACTION.value: robot,
            SampleKind.SEMANTIC.value: len(self.kinds) - robot,
        }

    def realized_ratios(self) -> dict[str, float]:
        counts = self.counts()
        total = len(self.kinds)
        return {name: count / total for name, count in counts.items()}
