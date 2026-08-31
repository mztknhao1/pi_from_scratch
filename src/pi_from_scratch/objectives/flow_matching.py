"""Conditional flow-matching targets for continuous action chunks."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class FlowMatchingBatch:
    """One supervised flow point for each action chunk in a batch."""

    noisy_actions: Tensor
    time: Tensor
    target_velocity: Tensor
    noise: Tensor


def linear_flow_path(actions: Tensor, noise: Tensor, time: Tensor) -> FlowMatchingBatch:
    """Interpolate data at ``t=0`` to noise at ``t=1`` using openpi's convention."""
    if actions.ndim != 3 or not actions.is_floating_point():
        raise ValueError("actions must be floating point with shape [batch, horizon, action_dim]")
    if noise.shape != actions.shape or noise.dtype != actions.dtype or noise.device != actions.device:
        raise ValueError("noise must match actions in shape, dtype, and device")
    if time.shape != (actions.shape[0],) or not time.is_floating_point():
        raise ValueError("time must be floating point with shape [batch]")
    if time.device != actions.device:
        raise ValueError("time and actions must be on the same device")
    if not bool(torch.all((time >= 0.0) & (time <= 1.0))):
        raise ValueError("time values must lie in [0, 1]")

    time_view = time[:, None, None].to(actions.dtype)
    noisy_actions = (1.0 - time_view) * actions + time_view * noise
    target_velocity = noise - actions
    return FlowMatchingBatch(
        noisy_actions=noisy_actions,
        time=time.to(actions.dtype),
        target_velocity=target_velocity,
        noise=noise,
    )


def sample_flow_batch(
    actions: Tensor,
    *,
    noise: Tensor | None = None,
    time: Tensor | None = None,
) -> FlowMatchingBatch:
    """Sample noise and time, then construct an openpi-style flow training target."""
    if actions.ndim != 3 or not actions.is_floating_point():
        raise ValueError("actions must be floating point with shape [batch, horizon, action_dim]")
    if noise is None:
        noise = torch.randn_like(actions)
    if time is None:
        beta = torch.distributions.Beta(
            torch.tensor(1.5, device=actions.device),
            torch.tensor(1.0, device=actions.device),
        )
        time = beta.sample((actions.shape[0],)).to(actions.dtype) * 0.999 + 0.001
    return linear_flow_path(actions, noise, time)


def masked_flow_matching_loss(
    predicted_velocity: Tensor,
    target_velocity: Tensor,
    valid_mask: Tensor,
) -> Tensor:
    """Average vector-field MSE over valid action timesteps."""
    if predicted_velocity.shape != target_velocity.shape or predicted_velocity.ndim != 3:
        raise ValueError(
            "predicted_velocity and target_velocity must share shape "
            "[batch, horizon, action_dim]"
        )
    if valid_mask.shape != predicted_velocity.shape[:2] or valid_mask.dtype != torch.bool:
        raise ValueError("valid_mask must be bool with shape [batch, horizon]")
    per_step = F.mse_loss(predicted_velocity, target_velocity, reduction="none").mean(dim=-1)
    weights = valid_mask.to(per_step.dtype)
    return (per_step * weights).sum() / weights.sum().clamp_min(1.0)
