from collections.abc import Callable

import torch
from torch import Tensor


def sample_flow_batch(actions: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Create noisy actions and target velocities using openpi's time convention."""
    batch_size = actions.shape[0]
    noise = torch.randn_like(actions)
    beta = torch.distributions.Beta(
        torch.tensor(1.5, device=actions.device), torch.tensor(1.0, device=actions.device)
    )
    time = beta.sample((batch_size,)).to(actions.dtype) * 0.999 + 0.001
    time_view = time[:, None, None]
    noisy_actions = time_view * noise + (1.0 - time_view) * actions
    target_velocity = noise - actions
    return noisy_actions, time, target_velocity


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
