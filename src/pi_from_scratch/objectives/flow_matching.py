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
