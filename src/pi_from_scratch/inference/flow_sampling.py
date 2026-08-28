"""Numerical samplers for flow-based action policies."""

from collections.abc import Callable

import torch
from torch import Tensor


@torch.no_grad()
def euler_sample(
    velocity_fn: Callable[[Tensor, Tensor], Tensor],
    shape: tuple[int, int, int],
    *,
    device: torch.device,
    num_steps: int = 10,
    noise: Tensor | None = None,
) -> Tensor:
    """Integrate from noise at t=1 to an action sample at t=0."""
    if num_steps < 1:
        raise ValueError("num_steps must be positive")
    x_t = torch.randn(shape, device=device) if noise is None else noise.clone()
    dt = -1.0 / num_steps
    for step in range(num_steps):
        time = torch.full((shape[0],), 1.0 + step * dt, device=device)
        x_t = x_t + dt * velocity_fn(x_t, time)
    return x_t
