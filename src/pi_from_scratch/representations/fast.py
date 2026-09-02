"""A small, inspectable FAST-like action tokenizer.

The real FAST tokenizer serializes quantized DCT coefficients as characters and
learns a byte-level BPE vocabulary. This teaching implementation applies BPE
directly to signed integer coefficients so the mathematical pipeline stays visible.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from itertools import pairwise

import torch
from torch import Tensor


def dct_matrix(length: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Return the orthonormal DCT-II matrix with shape ``[frequency, time]``."""
    if length <= 0:
        raise ValueError("length must be positive")
    frequency = torch.arange(length, device=device, dtype=dtype)[:, None]
    time = torch.arange(length, device=device, dtype=dtype)[None, :]
    basis = torch.cos(math.pi / length * (time + 0.5) * frequency)
    basis[0] *= math.sqrt(1.0 / length)
    if length > 1:
        basis[1:] *= math.sqrt(2.0 / length)
    return basis


def dct_actions(actions: Tensor) -> Tensor:
    """Apply an orthonormal DCT over the horizon axis of ``[B, H, A]`` actions."""
    if actions.ndim != 3:
        raise ValueError(f"actions must have shape [B, H, A], got {tuple(actions.shape)}")
    basis = dct_matrix(actions.shape[1], device=actions.device, dtype=actions.dtype)
    return torch.einsum("kh,bha->bka", basis, actions)


def idct_actions(coefficients: Tensor) -> Tensor:
    """Invert :func:`dct_actions` for orthonormal coefficients."""
    if coefficients.ndim != 3:
        raise ValueError(
            "coefficients must have shape [B, H, A], "
            f"got {tuple(coefficients.shape)}"
        )
    basis = dct_matrix(
        coefficients.shape[1], device=coefficients.device, dtype=coefficients.dtype
    )
    return torch.einsum("kh,bka->bha", basis, coefficients)


@dataclass(frozen=True)
class FastQuantileStats:
    """Per-action-dimension train-split statistics used by FAST normalization."""
    low: Tensor
    high: Tensor
    lower_quantile: float = 0.01
    upper_quantile: float = 0.99

    @classmethod
    def fit(
        cls,
        train_actions: Tensor,
        *,
        lower_quantile: float = 0.01,
        upper_quantile: float = 0.99,
    ) -> FastQuantileStats:
        if train_actions.ndim < 2:
            raise ValueError("train_actions needs a final action dimension")
        if not 0.0 <= lower_quantile < upper_quantile <= 1.0:
            raise ValueError("quantiles must satisfy 0 <= low < high <= 1")
        flattened = train_actions.reshape(-1, train_actions.shape[-1]).float()
        low = torch.quantile(flattened, lower_quantile, dim=0)
        high = torch.quantile(flattened, upper_quantile, dim=0)
        if torch.any(high - low <= 1e-8):
            raise ValueError("every action dimension must have a non-zero quantile range")
        return cls(
            low=low,
            high=high,
            lower_quantile=lower_quantile,
            upper_quantile=upper_quantile,
        )

    def normalize(self, actions: Tensor) -> Tensor:
        self._check(actions)
        low = self.low.to(device=actions.device, dtype=actions.dtype)
        high = self.high.to(device=actions.device, dtype=actions.dtype)
        return (2.0 * (actions - low) / (high - low) - 1.0).clamp(-1.0, 1.0)

    def denormalize(self, normalized: Tensor) -> Tensor:
        self._check(normalized)
        low = self.low.to(device=normalized.device, dtype=normalized.dtype)
        high = self.high.to(device=normalized.device, dtype=normalized.dtype)
        return (normalized + 1.0) * 0.5 * (high - low) + low

    def _check(self, actions: Tensor) -> None:
        if actions.shape[-1] != self.low.numel():
            raise ValueError(
                f"expected action dimension {self.low.numel()}, got {actions.shape[-1]}"
            )


@dataclass(frozen=True)
class MergeRule:
    left: int
    right: int
    merged: int


@dataclass(frozen=True)
class IntegerBPE:
    """A deterministic BPE codec over a bounded alphabet of signed integers."""
    minimum_value: int
    maximum_value: int
    rules: tuple[MergeRule, ...]
    expansions: tuple[tuple[int, ...], ...]

    @classmethod
    def fit(
        cls,
        sequences: list[list[int]],
        *,
        minimum_value: int,
        maximum_value: int,
        vocab_size: int,
        min_frequency: int = 2,
    ) -> IntegerBPE:
        if minimum_value > maximum_value:
            raise ValueError("minimum_value must not exceed maximum_value")
        base_size = maximum_value - minimum_value + 1
        if vocab_size < base_size:
            raise ValueError(f"vocab_size must be at least the base alphabet size {base_size}")
        if min_frequency < 2:
            raise ValueError("min_frequency must be at least 2")

        tokenized = [
            [cls._base_id(value, minimum_value, maximum_value) for value in sequence]
            for sequence in sequences
        ]
        expansions: list[tuple[int, ...]] = [
            (value,) for value in range(minimum_value, maximum_value + 1)
        ]
        rules: list[MergeRule] = []

        while len(expansions) < vocab_size:
            pair_counts: Counter[tuple[int, int]] = Counter()
            for sequence in tokenized:
                pair_counts.update(pairwise(sequence))
            if not pair_counts:
                break
            best_pair, count = min(
                pair_counts.items(),
                key=lambda item: (-item[1], item[0][0], item[0][1]),
            )
            if count < min_frequency:
                break
            merged_id = len(expansions)
            left, right = best_pair
            rules.append(MergeRule(left=left, right=right, merged=merged_id))
            expansions.append(expansions[left] + expansions[right])
            tokenized = [
                cls._apply_merge(sequence, left, right, merged_id) for sequence in tokenized
            ]

        return cls(
            minimum_value=minimum_value,
            maximum_value=maximum_value,
            rules=tuple(rules),
            expansions=tuple(expansions),
        )

    @property
    def vocab_size(self) -> int:
        return len(self.expansions)

    def encode(self, values: list[int]) -> list[int]:
        tokens = [
            self._base_id(value, self.minimum_value, self.maximum_value) for value in values
        ]
        for rule in self.rules:
            tokens = self._apply_merge(tokens, rule.left, rule.right, rule.merged)
        return tokens

    def decode(self, tokens: list[int]) -> list[int]:
        values: list[int] = []
        for token in tokens:
            if token < 0 or token >= len(self.expansions):
                raise ValueError(f"unknown BPE token id {token}")
            values.extend(self.expansions[token])
        return values

    @staticmethod
    def _base_id(value: int, minimum_value: int, maximum_value: int) -> int:
        if value < minimum_value or value > maximum_value:
            raise ValueError(
                f"coefficient {value} lies outside [{minimum_value}, {maximum_value}]"
            )
        return value - minimum_value

    @staticmethod
    def _apply_merge(tokens: list[int], left: int, right: int, merged: int) -> list[int]:
        output: list[int] = []
        index = 0
        while index < len(tokens):
            if index + 1 < len(tokens) and tokens[index] == left and tokens[index + 1] == right:
                output.append(merged)
                index += 2
            else:
                output.append(tokens[index])
                index += 1
        return output


@dataclass(frozen=True)
class FastActionTokenizer:
    """Train-split-fitted FAST-like codec for fixed-shape action chunks."""
    horizon: int
    action_dim: int
    scale: float
    stats: FastQuantileStats
    bpe: IntegerBPE

    @classmethod
    def fit(
        cls,
        train_chunks: Tensor,
        *,
        scale: float = 10.0,
        vocab_size: int = 256,
        min_frequency: int = 2,
    ) -> FastActionTokenizer:
        if train_chunks.ndim != 3:
            raise ValueError("train_chunks must have shape [N, H, A]")
        if scale <= 0:
            raise ValueError("scale must be positive")
        horizon, action_dim = train_chunks.shape[1:]
        stats = FastQuantileStats.fit(train_chunks)
        normalized = stats.normalize(train_chunks.float())
        quantized = torch.round(dct_actions(normalized) * scale).to(torch.int64)
        sequences = quantized.reshape(quantized.shape[0], -1).tolist()

        # With an orthonormal transform, a signal in [-1, 1] has coefficients
        # bounded by sqrt(H). A complete alphabet also covers unseen validation values.
        coefficient_bound = math.ceil(scale * math.sqrt(horizon))
        bpe = IntegerBPE.fit(
            sequences,
            minimum_value=-coefficient_bound,
            maximum_value=coefficient_bound,
            vocab_size=vocab_size,
            min_frequency=min_frequency,
        )
        return cls(
            horizon=horizon,
            action_dim=action_dim,
            scale=scale,
            stats=stats,
            bpe=bpe,
        )

    @property
    def coefficient_count(self) -> int:
        return self.horizon * self.action_dim

    def encode(self, chunk: Tensor) -> list[int]:
        if chunk.shape != (self.horizon, self.action_dim):
            raise ValueError(
                f"expected chunk shape {(self.horizon, self.action_dim)}, got {tuple(chunk.shape)}"
            )
        normalized = self.stats.normalize(chunk.float()).unsqueeze(0)
        integers = torch.round(dct_actions(normalized) * self.scale).to(torch.int64)
        return self.bpe.encode(integers.reshape(-1).tolist())

    def encode_batch(self, chunks: Tensor) -> list[list[int]]:
        if chunks.ndim != 3 or chunks.shape[1:] != (self.horizon, self.action_dim):
            raise ValueError(
                f"expected chunks shape [N, {self.horizon}, {self.action_dim}], "
                f"got {tuple(chunks.shape)}"
            )
        return [self.encode(chunk) for chunk in chunks]

    def decode(self, tokens: list[int]) -> Tensor:
        integers = self.bpe.decode(tokens)
        if len(integers) != self.coefficient_count:
            raise ValueError(
                f"decoded {len(integers)} coefficients, expected {self.coefficient_count}"
            )
        coefficients = torch.tensor(integers, dtype=self.stats.low.dtype)
        coefficients = coefficients.reshape(1, self.horizon, self.action_dim) / self.scale
        normalized = idct_actions(coefficients)[0]
        return self.stats.denormalize(normalized)
