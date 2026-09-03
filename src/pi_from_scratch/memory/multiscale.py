"""Small, typed memory components inspired by Multi-Scale Embodied Memory."""

from __future__ import annotations

import math
from collections import OrderedDict, deque
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class VisualMemoryFrame:
    timestamp_s: float
    features: Tensor
    visible_mask: Tensor

    def __post_init__(self) -> None:
        if not math.isfinite(self.timestamp_s):
            raise ValueError("timestamp_s must be finite")
        if self.features.ndim != 1 or not self.features.is_floating_point():
            raise ValueError("features must be float with shape [feature_dim]")
        if self.visible_mask.shape != self.features.shape or self.visible_mask.dtype != torch.bool:
            raise ValueError("visible_mask must be bool with shape [feature_dim]")
        if not torch.isfinite(self.features).all().item():
            raise ValueError("features must be finite")


@dataclass(frozen=True)
class VisualMemoryEncoding:
    values: Tensor
    valid_mask: Tensor


class ShortTermVideoMemory:
    """A bounded dense frame window with a transparent toy encoder.

    The paper uses a learned video encoder with spatial and causal-temporal
    attention. This teaching implementation keeps raw feature frames and uses
    the latest visible value per feature, making the memory contract easy to
    inspect before replacing the encoder.
    """

    def __init__(self, capacity: int, feature_dim: int):
        if capacity <= 0 or feature_dim <= 0:
            raise ValueError("capacity and feature_dim must be positive")
        self.capacity = capacity
        self.feature_dim = feature_dim
        self._frames: deque[VisualMemoryFrame] = deque(maxlen=capacity)

    def append(self, frame: VisualMemoryFrame) -> None:
        if frame.features.shape != (self.feature_dim,):
            raise ValueError("frame feature dimension does not match memory")
        if self._frames and frame.timestamp_s <= self._frames[-1].timestamp_s:
            raise ValueError("visual memory timestamps must be strictly increasing")
        self._frames.append(
            VisualMemoryFrame(
                timestamp_s=float(frame.timestamp_s),
                features=frame.features.detach().clone(),
                visible_mask=frame.visible_mask.detach().clone(),
            )
        )

    def encode(self) -> VisualMemoryEncoding:
        values = torch.zeros(self.feature_dim)
        valid = torch.zeros(self.feature_dim, dtype=torch.bool)
        for frame in reversed(self._frames):
            take = frame.visible_mask & ~valid
            values[take] = frame.features[take]
            valid |= frame.visible_mask
            if valid.all().item():
                break
        return VisualMemoryEncoding(values=values, valid_mask=valid)

    @property
    def frames(self) -> tuple[VisualMemoryFrame, ...]:
        return tuple(self._frames)


class LongTermTextMemory:
    """A structured stand-in for a model-generated compressed text summary."""

    def __init__(self, max_entries: int = 16):
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        self.max_entries = max_entries
        self._completed: OrderedDict[str, int] = OrderedDict()
        self._failed: OrderedDict[str, int] = OrderedDict()

    @staticmethod
    def _update_counts(table: OrderedDict[str, int], subtask: str, limit: int) -> None:
        table[subtask] = table.get(subtask, 0) + 1
        table.move_to_end(subtask)
        while len(table) > limit:
            table.popitem(last=False)

    def update(self, subtask: str, *, success: bool) -> None:
        cleaned = " ".join(subtask.split())
        if not cleaned:
            raise ValueError("subtask must contain text")
        table = self._completed if success else self._failed
        self._update_counts(table, cleaned, self.max_entries)

    def completed(self, subtask: str) -> bool:
        return subtask in self._completed

    def summary(self) -> str:
        def render(table: OrderedDict[str, int]) -> str:
            return ", ".join(
                f"{name} ×{count}" if count > 1 else name for name, count in table.items()
            )

        sections = []
        if self._completed:
            sections.append(f"completed: {render(self._completed)}")
        if self._failed:
            sections.append(f"failed: {render(self._failed)}")
        return "; ".join(sections) if sections else "memory: empty"


@dataclass(frozen=True)
class MemoryContext:
    short_term_features: Tensor
    short_term_mask: Tensor
    long_term_summary: str


class MultiScaleMemory:
    """Own the two memory stores and expose one typed policy context."""

    def __init__(self, *, video_capacity: int, feature_dim: int, text_entries: int = 16):
        self.short_term = ShortTermVideoMemory(video_capacity, feature_dim)
        self.long_term = LongTermTextMemory(text_entries)

    def context(self) -> MemoryContext:
        visual = self.short_term.encode()
        return MemoryContext(
            short_term_features=visual.values,
            short_term_mask=visual.valid_mask,
            long_term_summary=self.long_term.summary(),
        )
