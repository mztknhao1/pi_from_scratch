"""Typed experience records and small RECAP-style return utilities."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch
from torch import Tensor


class ExperienceSource(str, Enum):
    DEMONSTRATION = "demonstration"
    AUTONOMOUS = "autonomous"
    INTERVENTION = "intervention"


@dataclass(frozen=True)
class ExperienceEpisode:
    """One trajectory with provenance, rewards, and intervention ownership.

    Observations include the terminal observation, hence their leading length
    is one greater than the action and reward lengths.
    """

    task: str
    observations: Tensor
    actions: Tensor
    rewards: Tensor
    source: ExperienceSource
    success: bool
    intervention_mask: Tensor

    def __post_init__(self) -> None:
        if not self.task.strip():
            raise ValueError("task must contain text")
        if self.observations.ndim != 2 or not self.observations.is_floating_point():
            raise ValueError("observations must be float with shape [steps + 1, observation_dim]")
        if self.actions.ndim != 2 or not self.actions.is_floating_point():
            raise ValueError("actions must be float with shape [steps, action_dim]")
        if self.rewards.ndim != 1 or not self.rewards.is_floating_point():
            raise ValueError("rewards must be float with shape [steps]")
        steps = self.actions.shape[0]
        if self.observations.shape[0] != steps + 1 or self.rewards.shape[0] != steps:
            raise ValueError("episode time dimensions do not align")
        if self.intervention_mask.shape != (steps,) or self.intervention_mask.dtype != torch.bool:
            raise ValueError("intervention_mask must be bool with shape [steps]")
        if not torch.isfinite(self.observations).all().item():
            raise ValueError("observations must be finite")
        if not torch.isfinite(self.actions).all().item():
            raise ValueError("actions must be finite")
        if not torch.isfinite(self.rewards).all().item():
            raise ValueError("rewards must be finite")
        if self.source is ExperienceSource.INTERVENTION and not self.intervention_mask.any().item():
            raise ValueError("an intervention episode must identify at least one corrected action")


def sparse_completion_rewards(
    steps: int,
    *,
    success: bool,
    failure_penalty: float = 20.0,
) -> Tensor:
    """Create the sparse step-cost reward used in the RECAP paper."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    if failure_penalty <= 0:
        raise ValueError("failure_penalty must be positive")
    rewards = torch.full((steps,), -1.0)
    rewards[-1] = 0.0 if success else -float(failure_penalty)
    return rewards


def returns_to_go(rewards: Tensor) -> Tensor:
    """Undiscounted empirical return from each step to episode end."""
    if rewards.ndim != 1 or not rewards.is_floating_point():
        raise ValueError("rewards must be float with shape [steps]")
    if rewards.numel() == 0 or not torch.isfinite(rewards).all().item():
        raise ValueError("rewards must be non-empty and finite")
    return torch.flip(torch.cumsum(torch.flip(rewards, dims=(0,)), dim=0), dims=(0,))


def n_step_advantages(rewards: Tensor, values: Tensor, *, n_step: int) -> Tensor:
    """Compute A_t = sum(r_t:t+n) + V_{t+n} - V_t without discounting."""
    if rewards.ndim != 1 or values.ndim != 1:
        raise ValueError("rewards and values must be one-dimensional")
    if not rewards.is_floating_point() or not values.is_floating_point():
        raise ValueError("rewards and values must be floating point")
    if values.shape[0] != rewards.shape[0] + 1:
        raise ValueError("values must include one bootstrap value after the final action")
    if n_step <= 0:
        raise ValueError("n_step must be positive")
    if not torch.isfinite(rewards).all().item() or not torch.isfinite(values).all().item():
        raise ValueError("rewards and values must be finite")

    advantages = torch.empty_like(rewards)
    steps = rewards.shape[0]
    for start in range(steps):
        end = min(start + n_step, steps)
        advantages[start] = rewards[start:end].sum() + values[end] - values[start]
    return advantages


def task_advantage_threshold(advantages: Tensor, *, quantile: float = 0.3) -> float:
    if advantages.ndim != 1 or not advantages.is_floating_point():
        raise ValueError("advantages must be float with shape [samples]")
    if advantages.numel() == 0 or not torch.isfinite(advantages).all().item():
        raise ValueError("advantages must be non-empty and finite")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must lie in [0, 1]")
    return float(torch.quantile(advantages, quantile).item())


def improvement_indicators(
    advantages: Tensor,
    *,
    threshold: float,
    intervention_mask: Tensor | None = None,
) -> Tensor:
    """Binarize advantage and force human corrections to positive."""
    if advantages.ndim != 1 or not advantages.is_floating_point():
        raise ValueError("advantages must be float with shape [samples]")
    if not torch.isfinite(advantages).all().item():
        raise ValueError("advantages must be finite")
    indicators = advantages > threshold
    if intervention_mask is not None:
        if intervention_mask.shape != indicators.shape or intervention_mask.dtype != torch.bool:
            raise ValueError("intervention_mask must be bool with the same shape as advantages")
        indicators = indicators | intervention_mask
    return indicators
