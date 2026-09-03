"""Numerical samplers for flow-based action policies."""

from collections.abc import Callable
from typing import Literal

import torch
from torch import Tensor

FlowSolver = Literal["euler", "heun"]


def model_evaluations(solver: FlowSolver, num_steps: int) -> int:
    """Return the number of velocity-network calls made by a solver."""
    if num_steps < 1:
        raise ValueError("num_steps must be positive")
    if solver == "euler":
        return num_steps
    if solver == "heun":
        return 2 * num_steps
    raise ValueError(f"unknown flow solver: {solver}")


@torch.no_grad()
def flow_sample(
    velocity_fn: Callable[[Tensor, Tensor], Tensor],
    shape: tuple[int, int, int],
    *,
    device: torch.device,
    num_steps: int = 10,
    noise: Tensor | None = None,
    solver: FlowSolver = "euler",
) -> Tensor:
    """Integrate from noise at ``tau=0`` to an action sample at ``tau=1``."""
    model_evaluations(solver, num_steps)
    if len(shape) != 3 or any(size < 1 for size in shape):
        raise ValueError("shape must contain positive [batch, horizon, action_dim] sizes")
    if noise is not None and (noise.shape != shape or noise.device != device):
        raise ValueError("noise must match shape and device")
    if noise is not None and not noise.is_floating_point():
        raise TypeError("noise must be floating point")

    x_tau = torch.randn(shape, device=device) if noise is None else noise.clone()
    d_tau = 1.0 / num_steps
    for step in range(num_steps):
        tau_value = step * d_tau
        time = torch.full((shape[0],), tau_value, device=device, dtype=x_tau.dtype)
        velocity = velocity_fn(x_tau, time)
        if velocity.shape != x_tau.shape:
            raise ValueError("velocity_fn must return the same shape as its action input")
        if solver == "euler":
            x_tau = x_tau + d_tau * velocity
            continue

        proposal = x_tau + d_tau * velocity
        next_time_value = min(1.0, (step + 1) * d_tau)
        next_time = torch.full(
            (shape[0],),
            next_time_value,
            device=device,
            dtype=x_tau.dtype,
        )
        next_velocity = velocity_fn(proposal, next_time)
        if next_velocity.shape != x_tau.shape:
            raise ValueError("velocity_fn must return the same shape as its action input")
        x_tau = x_tau + 0.5 * d_tau * (velocity + next_velocity)
    return x_tau


@torch.no_grad()
def euler_sample(
    velocity_fn: Callable[[Tensor, Tensor], Tensor],
    shape: tuple[int, int, int],
    *,
    device: torch.device,
    num_steps: int = 10,
    noise: Tensor | None = None,
) -> Tensor:
    """Backward-compatible Euler wrapper around :func:`flow_sample`."""
    return flow_sample(
        velocity_fn,
        shape,
        device=device,
        num_steps=num_steps,
        noise=noise,
        solver="euler",
    )


@torch.no_grad()
def heun_sample(
    velocity_fn: Callable[[Tensor, Tensor], Tensor],
    shape: tuple[int, int, int],
    *,
    device: torch.device,
    num_steps: int = 10,
    noise: Tensor | None = None,
) -> Tensor:
    """Second-order predictor-corrector integration from ``tau=0`` to ``tau=1``."""
    return flow_sample(
        velocity_fn,
        shape,
        device=device,
        num_steps=num_steps,
        noise=noise,
        solver="heun",
    )
