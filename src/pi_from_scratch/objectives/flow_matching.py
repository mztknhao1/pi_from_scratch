"""Conditional flow-matching targets for continuous action chunks."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

FLOW_TIME_CONVENTION = "paper_tau_noise_0_action_1_v1"


@dataclass(frozen=True)
class FlowMatchingBatch:
    """One supervised flow point for each action chunk in a batch."""

    noisy_actions: Tensor
    time: Tensor
    target_velocity: Tensor
    noise: Tensor


@dataclass(frozen=True)
class TrainingRTCFlowBatch:
    """Flow point with a clean committed prefix and a noisy supervised postfix."""

    noisy_actions: Tensor
    token_time: Tensor
    target_velocity: Tensor
    loss_mask: Tensor
    noise: Tensor


def linear_flow_path(actions: Tensor, noise: Tensor, time: Tensor) -> FlowMatchingBatch:
    """Interpolate noise at ``tau=0`` to action data at ``tau=1`` as in the π₀ paper."""
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
    noisy_actions = (1.0 - time_view) * noise + time_view * actions
    target_velocity = actions - noise
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
    """Sample noise and paper-convention flow time, then construct a training target."""
    if actions.ndim != 3 or not actions.is_floating_point():
        raise ValueError("actions must be floating point with shape [batch, horizon, action_dim]")
    if noise is None:
        noise = torch.randn_like(actions)
    if time is None:
        beta = torch.distributions.Beta(
            torch.tensor(1.5, device=actions.device),
            torch.tensor(1.0, device=actions.device),
        )
        # π₀ samples a shifted Beta distribution that emphasizes low flow times.
        # openpi stores the complementary noise-time variable; converting it to
        # paper time gives tau = 1 - (0.999 * beta + 0.001).
        time = (1.0 - beta.sample((actions.shape[0],))).to(actions.dtype) * 0.999
    return linear_flow_path(actions, noise, time)


def training_rtc_flow_batch(
    actions: Tensor,
    prefix_lengths: Tensor,
    *,
    noise: Tensor,
    time: Tensor,
) -> TrainingRTCFlowBatch:
    """Construct training-time RTC inputs with tau=0 noise and tau=1 action data."""
    if actions.ndim != 3 or not actions.is_floating_point():
        raise ValueError("actions must be floating point with shape [batch, horizon, action_dim]")
    if noise.shape != actions.shape or noise.dtype != actions.dtype or noise.device != actions.device:
        raise ValueError("noise must match actions in shape, dtype, and device")
    if prefix_lengths.shape != (actions.shape[0],) or prefix_lengths.dtype != torch.long:
        raise ValueError("prefix_lengths must be int64 with shape [batch]")
    if prefix_lengths.device != actions.device:
        raise ValueError("prefix_lengths and actions must be on the same device")
    if torch.any(prefix_lengths < 0).item() or torch.any(prefix_lengths >= actions.shape[1]).item():
        raise ValueError("each prefix length must satisfy 0 <= d < horizon")
    if time.shape != (actions.shape[0],) or not time.is_floating_point():
        raise ValueError("time must be floating point with shape [batch]")
    if time.device != actions.device:
        raise ValueError("time and actions must be on the same device")
    if torch.any((time < 0.0) | (time > 1.0)).item():
        raise ValueError("time must lie in [0, 1]")

    positions = torch.arange(actions.shape[1], device=actions.device)[None]
    prefix_mask = positions < prefix_lengths[:, None]
    token_time = time[:, None].expand(-1, actions.shape[1]).clone()
    token_time[prefix_mask] = 1.0
    noisy_actions = (
        (1.0 - token_time[:, :, None]) * noise
        + token_time[:, :, None] * actions
    )
    return TrainingRTCFlowBatch(
        noisy_actions=noisy_actions,
        token_time=token_time,
        target_velocity=actions - noise,
        loss_mask=~prefix_mask,
        noise=noise,
    )


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
