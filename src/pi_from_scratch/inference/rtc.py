"""Real-Time Chunking guidance in the repository's t=1 noise -> t=0 data convention."""

import math
from collections.abc import Callable
from typing import Literal

import torch
from torch import Tensor

RTCSchedule = Literal["hard", "linear", "exponential"]


def rtc_prefix_weights(
    horizon: int,
    *,
    delay_steps: int,
    execution_horizon: int,
    schedule: RTCSchedule = "exponential",
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Build Eq. 5 prefix weights: frozen prefix, decayed overlap, fresh suffix."""
    if horizon < 1:
        raise ValueError("horizon must be positive")
    if delay_steps < 0 or delay_steps > execution_horizon:
        raise ValueError("delay_steps must satisfy 0 <= delay_steps <= execution_horizon")
    if execution_horizon < 1 or execution_horizon > horizon - delay_steps:
        raise ValueError("execution_horizon must satisfy 1 <= s <= horizon - delay_steps")
    if schedule not in ("hard", "linear", "exponential"):
        raise ValueError(f"unknown RTC schedule: {schedule}")

    weights = torch.zeros(horizon, device=device, dtype=dtype)
    weights[:delay_steps] = 1.0
    if schedule == "hard":
        return weights

    overlap_end = horizon - execution_horizon
    denominator = overlap_end - delay_steps + 1
    for index in range(delay_steps, overlap_end):
        coefficient = (overlap_end - index) / denominator
        if schedule == "linear":
            weights[index] = coefficient
        else:
            weights[index] = coefficient * math.expm1(coefficient) / math.expm1(1.0)
    return weights


def _guidance_weight(time: Tensor, max_guidance_weight: float) -> Tensor:
    """Convert the paper's tau=0->1 coefficient to openpi's t=1->0 convention."""
    if max_guidance_weight <= 0 or not math.isfinite(max_guidance_weight):
        raise ValueError("max_guidance_weight must be finite and positive")
    tau = 1.0 - time
    coefficient = (time.square() + tau.square()) / (time * tau)
    coefficient = torch.nan_to_num(
        coefficient,
        nan=0.0,
        posinf=max_guidance_weight,
        neginf=0.0,
    )
    return coefficient.clamp(min=0.0, max=max_guidance_weight)


def rtc_guided_velocity(
    velocity_fn: Callable[[Tensor, Tensor], Tensor],
    noisy_actions: Tensor,
    time: Tensor,
    previous_actions: Tensor,
    weights: Tensor,
    *,
    max_guidance_weight: float,
) -> Tensor:
    """Apply the RTC pseudoinverse-guidance vector-Jacobian correction."""
    if noisy_actions.ndim != 3 or not noisy_actions.is_floating_point():
        raise ValueError("noisy_actions must be floating point with shape [batch, horizon, dim]")
    if previous_actions.shape != noisy_actions.shape:
        raise ValueError("previous_actions must match noisy_actions")
    if weights.shape != noisy_actions.shape[1:2]:
        raise ValueError("weights must have shape [horizon]")
    if time.shape != (noisy_actions.shape[0],):
        raise ValueError("time must have shape [batch]")

    with torch.enable_grad():
        differentiable_actions = noisy_actions.detach().requires_grad_(True)
        base_velocity = velocity_fn(differentiable_actions, time)
        if base_velocity.shape != noisy_actions.shape:
            raise ValueError("velocity_fn must preserve action shape")
        predicted_data = differentiable_actions - time[:, None, None] * base_velocity
        weighted_error = (previous_actions - predicted_data) * weights[None, :, None]
        correction = torch.autograd.grad(
            predicted_data,
            differentiable_actions,
            grad_outputs=weighted_error.detach(),
            retain_graph=False,
        )[0]
        guidance = _guidance_weight(time, max_guidance_weight)[:, None, None]
        guided_velocity = base_velocity - guidance * correction
    return guided_velocity.detach()


def rtc_flow_sample(
    velocity_fn: Callable[[Tensor, Tensor], Tensor],
    shape: tuple[int, int, int],
    *,
    previous_actions: Tensor,
    delay_steps: int,
    execution_horizon: int,
    num_steps: int,
    device: torch.device,
    noise: Tensor,
    schedule: RTCSchedule = "exponential",
    max_guidance_weight: float = 5.0,
) -> Tensor:
    """Sample a chunk with RTC soft-prefix guidance at every Euler step."""
    if len(shape) != 3 or any(size < 1 for size in shape):
        raise ValueError("shape must be positive [batch, horizon, action_dim]")
    if num_steps < 1:
        raise ValueError("num_steps must be positive")
    if noise.shape != shape:
        raise ValueError("noise must match the requested sample shape")
    if previous_actions.ndim != 3:
        raise ValueError("previous_actions must have shape [batch, overlap, action_dim]")
    if previous_actions.shape[0] != shape[0] or previous_actions.shape[2] != shape[2]:
        raise ValueError("previous_actions batch and action dimensions must match shape")
    if previous_actions.shape[1] > shape[1]:
        raise ValueError("previous_actions cannot be longer than the sample horizon")

    weights = rtc_prefix_weights(
        shape[1],
        delay_steps=delay_steps,
        execution_horizon=execution_horizon,
        schedule=schedule,
        device=device,
        dtype=noise.dtype,
    )
    overlap = previous_actions.shape[1]
    if overlap < shape[1]:
        weights[overlap:] = 0.0
    padded_previous = torch.zeros(shape, device=device, dtype=noise.dtype)
    padded_previous[:, :overlap] = previous_actions.to(device=device, dtype=noise.dtype)

    actions = noise.to(device=device).clone()
    dt = -1.0 / num_steps
    for step in range(num_steps):
        time = torch.full(
            (shape[0],),
            1.0 - step / num_steps,
            device=device,
            dtype=actions.dtype,
        )
        velocity = rtc_guided_velocity(
            velocity_fn,
            actions,
            time,
            padded_previous,
            weights,
            max_guidance_weight=max_guidance_weight,
        )
        actions = actions + dt * velocity
    return actions.detach()
