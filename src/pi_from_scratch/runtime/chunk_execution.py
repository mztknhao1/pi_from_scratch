"""Synchronous action-chunk execution used in lesson 4.

This module consumes timestamped, physical-space action chunks. Model sampling,
inverse normalization, representation inversion, and rate conversion happen before
this boundary.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from pi_from_scratch.contracts import ActionChunk, ActionSpec


@dataclass(frozen=True)
class ChunkTiming:
    """Timing quantities implied by a uniformly sampled chunk."""

    horizon: int
    fps: float
    timestamp_span_s: float
    coverage_duration_s: float


def chunk_timing(horizon: int, fps: float) -> ChunkTiming:
    """Return timestamp span and control coverage for ``horizon`` commands."""
    if horizon < 1:
        raise ValueError("horizon must be positive")
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    return ChunkTiming(
        horizon=horizon,
        fps=fps,
        timestamp_span_s=(horizon - 1) / fps,
        coverage_duration_s=horizon / fps,
    )


@dataclass(frozen=True)
class ActionTrace:
    """Actions selected for execution from one or more replanned chunks."""

    values: Tensor
    timestamps_s: Tensor
    source_chunk_indices: Tensor
    chunk_boundary_steps: tuple[int, ...]
    spec: ActionSpec

    def __post_init__(self) -> None:
        if self.values.ndim != 2 or not self.values.is_floating_point():
            raise ValueError("values must be floating point with shape [steps, action_dim]")
        num_steps = self.values.shape[0]
        if num_steps < 1 or self.values.shape[1] != self.spec.dim:
            raise ValueError("trace must contain actions matching ActionSpec")
        if self.timestamps_s.shape != (num_steps,) or not self.timestamps_s.is_floating_point():
            raise ValueError("timestamps_s must be float with shape [steps]")
        if self.source_chunk_indices.shape != (num_steps,) or self.source_chunk_indices.dtype != torch.long:
            raise ValueError("source_chunk_indices must be int64 with shape [steps]")
        if not torch.isfinite(self.values).all().item() or not torch.isfinite(self.timestamps_s).all().item():
            raise ValueError("trace values and timestamps must be finite")
        if num_steps > 1 and torch.any(self.timestamps_s[1:] <= self.timestamps_s[:-1]).item():
            raise ValueError("executed timestamps must be strictly increasing")
        if tuple(sorted(set(self.chunk_boundary_steps))) != self.chunk_boundary_steps:
            raise ValueError("chunk_boundary_steps must be sorted and unique")
        if any(step <= 0 or step >= num_steps for step in self.chunk_boundary_steps):
            raise ValueError("each chunk boundary must lie between two executed actions")


def stitch_chunk_prefixes(
    chunks: Sequence[ActionChunk], *, execution_horizon: int
) -> ActionTrace:
    """Execute at most ``E`` valid actions from each chunk, then replan.

    This is the blocking synchronous baseline: inference is assumed to finish before
    each segment starts, so the replanning interval equals the number of actions
    selected from the previous chunk.
    """
    if not chunks:
        raise ValueError("at least one action chunk is required")
    if execution_horizon < 1:
        raise ValueError("execution_horizon must be positive")

    spec = chunks[0].spec
    prediction_horizon = chunks[0].horizon
    if execution_horizon > prediction_horizon:
        raise ValueError("execution_horizon must not exceed the prediction horizon")
    values: list[Tensor] = []
    timestamps: list[Tensor] = []
    source_indices: list[Tensor] = []
    boundaries: list[int] = []
    total_steps = 0

    for chunk_index, chunk in enumerate(chunks):
        if chunk.batch_size != 1:
            raise ValueError("the synchronous teaching executor accepts batch size one")
        if chunk.horizon != prediction_horizon:
            raise ValueError("all replanned chunks must use the same prediction horizon")
        if chunk.normalized:
            raise ValueError("denormalize and inverse-transform chunks before execution")
        if chunk.spec != spec:
            raise ValueError("all chunks in one trace must use the same ActionSpec")
        valid_count = int(chunk.valid_mask[0].sum().item())
        selected_count = min(execution_horizon, valid_count)
        if selected_count == 0:
            continue
        if total_steps > 0:
            boundaries.append(total_steps)
        values.append(chunk.values[0, :selected_count])
        timestamps.append(chunk.timestamps_s[0, :selected_count])
        source_indices.append(
            torch.full(
                (selected_count,),
                chunk_index,
                dtype=torch.long,
                device=chunk.values.device,
            )
        )
        total_steps += selected_count

    if not values:
        raise ValueError("chunks contain no valid actions")
    return ActionTrace(
        values=torch.cat(values, dim=0),
        timestamps_s=torch.cat(timestamps, dim=0),
        source_chunk_indices=torch.cat(source_indices, dim=0),
        chunk_boundary_steps=tuple(boundaries),
        spec=spec,
    )


def boundary_action_jumps(trace: ActionTrace) -> Tensor:
    """L2 action-command changes where a newly replanned chunk starts."""
    if not trace.chunk_boundary_steps:
        return trace.values.new_empty((0,))
    boundary_steps = torch.tensor(
        trace.chunk_boundary_steps, dtype=torch.long, device=trace.values.device
    )
    changes = trace.values.index_select(0, boundary_steps) - trace.values.index_select(
        0, boundary_steps - 1
    )
    return torch.linalg.vector_norm(changes, dim=-1)
