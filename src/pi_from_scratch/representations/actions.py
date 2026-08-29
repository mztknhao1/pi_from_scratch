"""Reversible action representations and train-only standardization.

The two operations in this module deliberately stay separate:

1. representation transforms change the physical meaning of a value;
2. normalization only changes its numerical scale.
"""

import hashlib
import json
import math
from dataclasses import dataclass

import torch
from torch import Tensor

from pi_from_scratch.contracts import ActionChunk


def _check_action_and_state(actions: Tensor, current_state: Tensor) -> None:
    if actions.ndim < 2:
        raise ValueError("actions must have shape [..., horizon, action_dim]")
    if current_state.shape != actions.shape[:-2] + actions.shape[-1:]:
        raise ValueError("current_state must have shape [..., action_dim]")
    if not actions.is_floating_point() or not current_state.is_floating_point():
        raise TypeError("actions and current_state must be floating point")


def _dimension_mask(mask: Tensor | None, action_dim: int, device: torch.device) -> Tensor:
    if mask is None:
        return torch.ones(action_dim, dtype=torch.bool, device=device)
    mask = torch.as_tensor(mask, dtype=torch.bool, device=device)
    if mask.shape != (action_dim,):
        raise ValueError("dimension_mask must have shape [action_dim]")
    return mask


@dataclass(frozen=True)
class CurrentStateDeltaTransform:
    """Express selected absolute targets relative to the current state.

    Every future target uses the same observation-time state as its reference:
    ``delta[h] = target[h] - current_state``. This is not a temporal difference
    between adjacent actions.
    """

    dimension_mask: Tensor | None = None

    def forward(self, absolute_actions: Tensor, current_state: Tensor) -> Tensor:
        _check_action_and_state(absolute_actions, current_state)
        mask = _dimension_mask(
            self.dimension_mask, absolute_actions.shape[-1], absolute_actions.device
        )
        relative = absolute_actions - current_state.unsqueeze(-2)
        return torch.where(mask, relative, absolute_actions)

    def inverse(self, represented_actions: Tensor, current_state: Tensor) -> Tensor:
        _check_action_and_state(represented_actions, current_state)
        mask = _dimension_mask(
            self.dimension_mask, represented_actions.shape[-1], represented_actions.device
        )
        absolute = represented_actions + current_state.unsqueeze(-2)
        return torch.where(mask, absolute, represented_actions)


@dataclass(frozen=True)
class FiniteDifferenceVelocityTransform:
    """Convert selected absolute targets to per-second finite differences."""

    fps: float
    dimension_mask: Tensor | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("fps must be finite and positive")

    def forward(self, absolute_actions: Tensor, current_state: Tensor) -> Tensor:
        _check_action_and_state(absolute_actions, current_state)
        mask = _dimension_mask(
            self.dimension_mask, absolute_actions.shape[-1], absolute_actions.device
        )
        previous = torch.cat((current_state.unsqueeze(-2), absolute_actions[..., :-1, :]), dim=-2)
        velocity = (absolute_actions - previous) * self.fps
        return torch.where(mask, velocity, absolute_actions)

    def inverse(self, represented_actions: Tensor, current_state: Tensor) -> Tensor:
        _check_action_and_state(represented_actions, current_state)
        mask = _dimension_mask(
            self.dimension_mask, represented_actions.shape[-1], represented_actions.device
        )
        integrated = current_state.unsqueeze(-2) + torch.cumsum(
            represented_actions / self.fps, dim=-2
        )
        return torch.where(mask, integrated, represented_actions)


@dataclass(frozen=True)
class NormalizationStats:
    """Per-dimension statistics fitted exclusively on training episodes."""

    mean: Tensor
    std: Tensor
    count: int
    train_episode_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.mean.ndim != 1 or self.std.shape != self.mean.shape:
            raise ValueError("mean and std must have shape [action_dim]")
        if not self.mean.is_floating_point() or not self.std.is_floating_point():
            raise TypeError("mean and std must be floating point")
        if self.count < 2:
            raise ValueError("at least two valid training actions are required")
        if not self.train_episode_ids or len(set(self.train_episode_ids)) != len(
            self.train_episode_ids
        ):
            raise ValueError("train_episode_ids must be non-empty and unique")
        if not torch.isfinite(self.mean).all().item() or not torch.isfinite(self.std).all().item():
            raise ValueError("normalization statistics must be finite")
        if torch.any(self.std < 0).item():
            raise ValueError("standard deviations must be non-negative")

    @property
    def artifact_id(self) -> str:
        payload = {
            "mean": self.mean.detach().cpu().tolist(),
            "std": self.std.detach().cpu().tolist(),
            "count": self.count,
            "train_episode_ids": self.train_episode_ids,
        }
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]
        return f"action-zscore-{digest}"


class RunningActionStats:
    """Accumulate masked statistics without retaining the complete dataset."""

    def __init__(self) -> None:
        self._count = 0
        self._sum: Tensor | None = None
        self._sum_of_squares: Tensor | None = None

    def update(self, values: Tensor, valid_mask: Tensor | None = None) -> None:
        if values.ndim < 2 or not values.is_floating_point():
            raise ValueError("values must be floating point with shape [..., action_dim]")
        if valid_mask is None:
            valid_mask = torch.ones(values.shape[:-1], dtype=torch.bool, device=values.device)
        if valid_mask.shape != values.shape[:-1] or valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask must be bool with shape values.shape[:-1]")
        selected = values[valid_mask].detach().to(dtype=torch.float64, device="cpu")
        if selected.numel() == 0:
            return
        if not torch.isfinite(selected).all().item():
            raise ValueError("statistics inputs must be finite")
        batch_sum = selected.sum(dim=0)
        batch_sum_of_squares = selected.square().sum(dim=0)
        if self._sum is None:
            self._sum = torch.zeros_like(batch_sum)
            self._sum_of_squares = torch.zeros_like(batch_sum_of_squares)
        if self._sum.shape != batch_sum.shape:
            raise ValueError("all action batches must use the same action dimension")
        self._count += selected.shape[0]
        self._sum += batch_sum
        self._sum_of_squares += batch_sum_of_squares

    def finalize(self, *, train_episode_ids: tuple[int, ...]) -> NormalizationStats:
        if self._sum is None or self._sum_of_squares is None or self._count < 2:
            raise ValueError("at least two valid actions are required before finalize")
        mean = self._sum / self._count
        variance = (self._sum_of_squares / self._count - mean.square()).clamp_min(0.0)
        return NormalizationStats(
            mean=mean.float(),
            std=variance.sqrt().float(),
            count=self._count,
            train_episode_ids=train_episode_ids,
        )


@dataclass(frozen=True)
class ActionNormalizer:
    """Apply and invert per-dimension z-score normalization."""

    stats: NormalizationStats
    epsilon: float = 1e-6

    def __post_init__(self) -> None:
        if not math.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("epsilon must be finite and positive")

    def normalize_values(self, values: Tensor) -> Tensor:
        mean, scale = self._parameters_like(values)
        return (values - mean) / scale

    def denormalize_values(self, values: Tensor) -> Tensor:
        mean, scale = self._parameters_like(values)
        return values * scale + mean

    def normalize_chunk(self, chunk: ActionChunk) -> ActionChunk:
        if chunk.normalized:
            raise ValueError("action chunk is already normalized")
        return ActionChunk(
            values=self.normalize_values(chunk.values),
            valid_mask=chunk.valid_mask,
            timestamps_s=chunk.timestamps_s,
            spec=chunk.spec,
            normalized=True,
            normalization_id=self.stats.artifact_id,
        )

    def denormalize_chunk(self, chunk: ActionChunk) -> ActionChunk:
        if not chunk.normalized:
            raise ValueError("action chunk is not normalized")
        if chunk.normalization_id != self.stats.artifact_id:
            raise ValueError("action chunk was produced with a different normalization artifact")
        return ActionChunk(
            values=self.denormalize_values(chunk.values),
            valid_mask=chunk.valid_mask,
            timestamps_s=chunk.timestamps_s,
            spec=chunk.spec,
        )

    def _parameters_like(self, values: Tensor) -> tuple[Tensor, Tensor]:
        if values.shape[-1] != self.stats.mean.shape[0]:
            raise ValueError("values action dimension does not match normalization statistics")
        mean = self.stats.mean.to(device=values.device, dtype=values.dtype)
        std = self.stats.std.to(device=values.device, dtype=values.dtype)
        return mean, std.clamp_min(self.epsilon)
