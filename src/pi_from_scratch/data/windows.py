"""Episode-safe temporal windows used by lesson 2 and later data adapters."""

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class EpisodeSplit:
    """Disjoint episode ids for training and validation."""

    train: tuple[int, ...]
    validation: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.train or not self.validation:
            raise ValueError("both train and validation splits must contain an episode")
        if set(self.train) & set(self.validation):
            raise ValueError("train and validation episodes must be disjoint")


@dataclass(frozen=True)
class FutureActionWindow:
    """A fixed-size target window cut from exactly one episode."""

    values: Tensor
    valid_mask: Tensor
    timestamps_s: Tensor
    source_indices: Tensor

    def __post_init__(self) -> None:
        if self.values.ndim != 2:
            raise ValueError("values must have shape [horizon, action_dim]")
        horizon = self.values.shape[0]
        if self.valid_mask.shape != (horizon,) or self.valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask must be bool with shape [horizon]")
        if self.timestamps_s.shape != (horizon,) or not self.timestamps_s.is_floating_point():
            raise ValueError("timestamps_s must be float with shape [horizon]")
        if self.source_indices.shape != (horizon,) or self.source_indices.dtype != torch.long:
            raise ValueError("source_indices must be int64 with shape [horizon]")
        if horizon > 1 and torch.any(self.timestamps_s[1:] <= self.timestamps_s[:-1]).item():
            raise ValueError("timestamps_s must be strictly increasing")
        if horizon > 1 and torch.any((~self.valid_mask[:-1]) & self.valid_mask[1:]).item():
            raise ValueError("valid_mask must be a contiguous valid prefix")
        if not self.valid_mask[0].item():
            raise ValueError("the anchor action must be valid")


def split_episode_ids(
    episode_ids: Sequence[int], *, validation_fraction: float = 0.2, seed: int = 7
) -> EpisodeSplit:
    """Split complete episodes so adjacent frames never leak across subsets."""
    ids = [int(episode_id) for episode_id in episode_ids]
    if len(ids) != len(set(ids)):
        raise ValueError("episode_ids must be unique")
    if len(ids) < 2:
        raise ValueError("at least two episodes are required for a split")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one")

    shuffled = sorted(ids)
    random.Random(seed).shuffle(shuffled)
    num_validation = min(len(shuffled) - 1, max(1, math.ceil(len(shuffled) * validation_fraction)))
    validation = tuple(sorted(shuffled[:num_validation]))
    train = tuple(sorted(shuffled[num_validation:]))
    return EpisodeSplit(train=train, validation=validation)


def build_future_action_window(
    episode_actions: Tensor, *, anchor_index: int, horizon: int, fps: float
) -> FutureActionWindow:
    """Build ``[a_t, ..., a_(t+H-1)]`` without crossing an episode boundary.

    Missing tail actions repeat the last action so the tensor stays fixed-size. The
    repeated values are explicitly marked invalid and must not contribute to loss.
    """
    if episode_actions.ndim != 2 or episode_actions.shape[0] == 0:
        raise ValueError("episode_actions must have shape [episode_length, action_dim]")
    if not 0 <= anchor_index < episode_actions.shape[0]:
        raise IndexError("anchor_index must point inside the episode")
    if horizon < 1:
        raise ValueError("horizon must be positive")
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")

    offsets = torch.arange(horizon, device=episode_actions.device)
    requested_indices = anchor_index + offsets
    valid_mask = requested_indices < episode_actions.shape[0]
    source_indices = requested_indices.clamp_max(episode_actions.shape[0] - 1).long()
    values = episode_actions.index_select(0, source_indices)
    timestamps_s = requested_indices.to(torch.float32) / fps
    return FutureActionWindow(
        values=values,
        valid_mask=valid_mask,
        timestamps_s=timestamps_s,
        source_indices=source_indices,
    )
