"""A tiny model for teaching π0.5-style heterogeneous objectives."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class TinyPi05(nn.Module):
    """Shared semantic backbone plus discrete and continuous action heads.

    The continuous expert can consume a detached backbone representation. This
    is the small teaching analogue of knowledge insulation; it does not
    reproduce the blockwise attention implementation used by the full model.
    """

    def __init__(
        self,
        *,
        observation_dim: int = 6,
        width: int = 24,
        semantic_classes: int = 3,
        action_token_classes: int = 4,
        action_dim: int = 2,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.backbone = nn.Sequential(
            nn.Linear(observation_dim, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
        )
        self.semantic_head = nn.Linear(width, semantic_classes)
        self.discrete_action_head = nn.Linear(width, action_token_classes)
        self.action_input = nn.Linear(action_dim, width)
        self.time_input = nn.Linear(1, width)
        self.action_expert = nn.Sequential(
            nn.Linear(width * 3, width),
            nn.SiLU(),
            nn.Linear(width, action_dim),
        )

    def encode(self, observation: Tensor) -> Tensor:
        return self.backbone(observation)

    def semantic_logits(self, observation: Tensor) -> Tensor:
        return self.semantic_head(self.encode(observation))

    def discrete_action_logits(self, observation: Tensor) -> Tensor:
        return self.discrete_action_head(self.encode(observation))

    def predict_velocity(
        self,
        observation: Tensor,
        noisy_actions: Tensor,
        time: Tensor,
        *,
        insulate_backbone: bool,
    ) -> Tensor:
        if noisy_actions.ndim != 3 or noisy_actions.shape[-1] != self.action_dim:
            raise ValueError("noisy_actions must have shape [batch, horizon, action_dim]")
        if time.shape != (noisy_actions.shape[0],):
            raise ValueError("time must have shape [batch]")
        condition = self.encode(observation)
        if insulate_backbone:
            condition = condition.detach()
        horizon = noisy_actions.shape[1]
        condition = condition[:, None].expand(-1, horizon, -1)
        action = self.action_input(noisy_actions)
        time_token = self.time_input(time[:, None])[:, None].expand(-1, horizon, -1)
        return self.action_expert(torch.cat((condition, action, time_token), dim=-1))


def parameter_grad_norm(module: nn.Module) -> float:
    squared = torch.zeros(())
    for parameter in module.parameters():
        if parameter.grad is not None:
            squared = squared + parameter.grad.detach().square().sum().cpu()
    return float(squared.sqrt().item())
