"""A tiny policy for demonstrating advantage-conditioned regression."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class TinyAdvantagePolicy(nn.Module):
    """Predict continuous actions with positive, negative, or absent condition.

    Indicator values use -1 for omitted, 0 for negative, and 1 for positive.
    """

    def __init__(self, *, observation_dim: int = 1, action_dim: int = 1, width: int = 24):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.network = nn.Sequential(
            nn.Linear(observation_dim + 2, width),
            nn.Tanh(),
            nn.Linear(width, action_dim),
        )

    def forward(self, observation: Tensor, indicator: Tensor) -> Tensor:
        if observation.ndim != 2 or observation.shape[1] != self.observation_dim:
            raise ValueError("observation must have shape [batch, observation_dim]")
        if indicator.shape != (observation.shape[0],) or indicator.dtype != torch.long:
            raise ValueError("indicator must be int64 with shape [batch]")
        if not torch.all((indicator >= -1) & (indicator <= 1)).item():
            raise ValueError("indicator values must be -1, 0, or 1")
        encoded = torch.stack((indicator == 0, indicator == 1), dim=-1).to(observation.dtype)
        return self.network(torch.cat((observation, encoded), dim=-1))
