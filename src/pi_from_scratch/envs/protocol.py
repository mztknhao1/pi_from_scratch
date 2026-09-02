"""Model-agnostic environment boundary used by closed-loop runtimes."""

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch
from torch import Tensor

from pi_from_scratch.contracts import ActionSpec, ObservationBatch


@dataclass(frozen=True)
class EnvTransition:
    """Result of applying one physical-space action to an environment."""

    observation: ObservationBatch
    reward: float
    terminated: bool
    truncated: bool
    success: bool

    def __post_init__(self) -> None:
        if self.observation.batch_size != 1:
            raise ValueError("closed-loop teaching environments must return batch size one")
        if not math.isfinite(self.reward):
            raise ValueError("reward must be finite")
        if self.success and not self.terminated:
            raise ValueError("a successful transition must terminate the episode")


@runtime_checkable
class ClosedLoopEnv(Protocol):
    """The only environment methods visible to a policy runtime."""

    fps: float
    action_spec: ActionSpec

    def reset(self, *, seed: int) -> ObservationBatch: ...

    def step(self, action: Tensor) -> EnvTransition: ...

    def close(self) -> None: ...


def validate_physical_action(action: Tensor, spec: ActionSpec) -> Tensor:
    """Validate one denormalized action and return a float32 CPU copy."""
    action = torch.as_tensor(action, dtype=torch.float32).detach().cpu()
    if action.shape != (spec.dim,):
        raise ValueError(f"action must have shape [{spec.dim}]")
    if not torch.isfinite(action).all().item():
        raise ValueError("action must be finite")
    minimum = torch.tensor(spec.minimum, dtype=action.dtype)
    maximum = torch.tensor(spec.maximum, dtype=action.dtype)
    if torch.any(action < minimum).item() or torch.any(action > maximum).item():
        raise ValueError("action is outside ActionSpec bounds")
    return action
