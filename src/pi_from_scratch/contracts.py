"""Typed boundaries shared by data, policies, runtimes, and evaluators.

The contracts intentionally stay model-agnostic: token ids, flow timesteps, and
hidden states belong inside a policy implementation rather than at this boundary.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

import torch
from torch import Tensor


class ActionRepresentation(str, Enum):
    ABSOLUTE = "absolute"
    DELTA = "delta"
    VELOCITY = "velocity"


@dataclass(frozen=True)
class ActionSpec:
    """Meaning of every action dimension before normalization."""

    dim: int
    space: str
    representation: ActionRepresentation
    frame: str
    units: tuple[str, ...]
    minimum: tuple[float, ...]
    maximum: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.dim < 1:
            raise ValueError("action dim must be positive")
        for name, values in (
            ("units", self.units),
            ("minimum", self.minimum),
            ("maximum", self.maximum),
        ):
            if len(values) != self.dim:
                raise ValueError(f"{name} must contain one entry per action dimension")
        for low, high in zip(self.minimum, self.maximum, strict=True):
            if not (math.isfinite(low) and math.isfinite(high) and low < high):
                raise ValueError("each action bound must be finite and minimum < maximum")
        if not self.space or not self.frame or not all(self.units):
            raise ValueError("action space, frame, and units must be named explicitly")


@dataclass(frozen=True)
class ObservationBatch:
    """Model-agnostic observations at policy invocation time.

    Images use one named tensor per camera so camera order never carries meaning.
    The timestamp uses simulator/robot time, not the host's monotonic wall clock.
    """

    images: Mapping[str, Tensor]
    image_masks: Mapping[str, Tensor]
    state: Tensor
    state_mask: Tensor
    prompts: tuple[str, ...]
    timestamp_s: Tensor

    def __post_init__(self) -> None:
        if self.state.ndim != 2:
            raise ValueError("state must have shape [batch, state_dim]")
        if not self.state.is_floating_point():
            raise TypeError("state must be floating point")
        batch_size = self.state.shape[0]
        if self.state_mask.shape != self.state.shape or self.state_mask.dtype != torch.bool:
            raise ValueError("state_mask must be bool with the same shape as state")
        if len(self.prompts) != batch_size or not all(
            isinstance(item, str) for item in self.prompts
        ):
            raise ValueError("prompts must contain one string per batch element")
        if self.timestamp_s.shape != (batch_size,) or not self.timestamp_s.is_floating_point():
            raise ValueError("timestamp_s must be float with shape [batch]")
        if not torch.isfinite(self.timestamp_s).all().item():
            raise ValueError("timestamp_s must be finite")
        if not self.images:
            raise ValueError("at least one named image is required")
        if set(self.images) != set(self.image_masks):
            raise ValueError("images and image_masks must have identical camera names")
        for camera_name, image in self.images.items():
            if image.ndim != 4 or image.shape[0] != batch_size:
                raise ValueError(
                    f"image {camera_name!r} must have shape [batch, channel, height, width]"
                )
            mask = self.image_masks[camera_name]
            if mask.shape != (batch_size,) or mask.dtype != torch.bool:
                raise ValueError(f"image mask {camera_name!r} must be bool with shape [batch]")

    @property
    def batch_size(self) -> int:
        return self.state.shape[0]


@dataclass(frozen=True)
class ActionChunk:
    """A timestamped future action sequence."""

    values: Tensor
    valid_mask: Tensor
    timestamps_s: Tensor
    spec: ActionSpec
    normalized: bool = False
    normalization_id: str | None = None

    def __post_init__(self) -> None:
        if self.values.ndim != 3:
            raise ValueError("action values must have shape [batch, horizon, action_dim]")
        if not self.values.is_floating_point():
            raise TypeError("action values must be floating point")
        if self.values.shape[-1] != self.spec.dim:
            raise ValueError("action tensor dimension does not match ActionSpec")
        expected_window_shape = self.values.shape[:2]
        if self.valid_mask.shape != expected_window_shape or self.valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask must be bool with shape [batch, horizon]")
        if (
            self.timestamps_s.shape != expected_window_shape
            or not self.timestamps_s.is_floating_point()
        ):
            raise ValueError("timestamps_s must be float with shape [batch, horizon]")
        if (
            not torch.isfinite(self.values).all().item()
            or not torch.isfinite(self.timestamps_s).all().item()
        ):
            raise ValueError("action values and timestamps must be finite")
        if self.horizon > 1:
            if torch.any((~self.valid_mask[:, :-1]) & self.valid_mask[:, 1:]).item():
                raise ValueError("valid_mask must be a contiguous valid prefix")
            consecutive_valid = self.valid_mask[:, :-1] & self.valid_mask[:, 1:]
            timestamp_delta = self.timestamps_s[:, 1:] - self.timestamps_s[:, :-1]
            if torch.any(consecutive_valid & (timestamp_delta <= 0)).item():
                raise ValueError("valid action timestamps must be strictly increasing")
        if self.normalized and not self.normalization_id:
            raise ValueError("normalized actions must identify their normalization artifact")
        if not self.normalized and self.normalization_id is not None:
            raise ValueError("raw actions must not carry a normalization_id")

    @property
    def batch_size(self) -> int:
        return self.values.shape[0]

    @property
    def horizon(self) -> int:
        return self.values.shape[1]


@dataclass(frozen=True)
class PolicyOutput:
    """What a runtime receives from any policy implementation."""

    action_chunk: ActionChunk
    source_observation_timestamp_s: Tensor
    generated_at_monotonic_s: float
    inference_latency_s: float

    def __post_init__(self) -> None:
        if self.action_chunk.normalized:
            raise ValueError(
                "a policy must inverse-transform actions before returning PolicyOutput"
            )
        if self.source_observation_timestamp_s.shape != (self.action_chunk.batch_size,):
            raise ValueError("source observation timestamps must have shape [batch]")
        if not self.source_observation_timestamp_s.is_floating_point():
            raise TypeError("source observation timestamps must be floating point")
        if not torch.isfinite(self.source_observation_timestamp_s).all().item():
            raise ValueError("source observation timestamps must be finite")
        if not math.isfinite(self.generated_at_monotonic_s):
            raise ValueError("generated_at_monotonic_s must be finite")
        if not math.isfinite(self.inference_latency_s) or self.inference_latency_s < 0:
            raise ValueError("inference_latency_s must be finite and non-negative")


@dataclass(frozen=True)
class EpisodeResult:
    """Small common result; large action traces are optional artifacts."""

    success: bool
    total_reward: float
    num_steps: int
    inference_latencies_s: tuple[float, ...]
    deadline_misses: int
    executed_actions: Tensor | None = None
    chunk_boundary_steps: tuple[int, ...] = ()
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.total_reward):
            raise ValueError("total_reward must be finite")
        if self.num_steps < 0 or self.deadline_misses < 0:
            raise ValueError("step and deadline counts must be non-negative")
        if any((not math.isfinite(value) or value < 0) for value in self.inference_latencies_s):
            raise ValueError("inference latencies must be finite and non-negative")
        if self.executed_actions is not None and (
            self.executed_actions.ndim != 2 or self.executed_actions.shape[0] != self.num_steps
        ):
            raise ValueError("executed_actions must have shape [num_steps, action_dim]")
