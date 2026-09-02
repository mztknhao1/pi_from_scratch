"""A reproducible compression experiment for the FAST lesson."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import Tensor

from pi_from_scratch.representations.fast import FastActionTokenizer


def make_smooth_action_chunks(
    count: int,
    *,
    horizon: int,
    action_dim: int,
    seed: int,
) -> Tensor:
    """Create a deterministic collection of low-frequency action chunks."""
    if count <= 0 or horizon <= 1 or action_dim <= 0:
        raise ValueError("count/action_dim must be positive and horizon must exceed one")
    generator = torch.Generator().manual_seed(seed)
    time = torch.linspace(0.0, 1.0, horizon)
    chunks = []
    for _ in range(count):
        offset = torch.randn(action_dim, generator=generator) * 0.35
        slope = torch.randn(action_dim, generator=generator) * 0.55
        amplitude = 0.05 + torch.rand(action_dim, generator=generator) * 0.3
        phase = torch.rand(action_dim, generator=generator) * (2.0 * torch.pi)
        frequency = torch.randint(1, 3, (action_dim,), generator=generator).float()
        chunk = offset + time[:, None] * slope
        chunk += amplitude * torch.sin(
            2.0 * torch.pi * time[:, None] * frequency + phase
        )
        chunks.append(chunk)
    return torch.stack(chunks)


def run_fast_experiment(
    *,
    train_count: int = 400,
    validation_count: int = 100,
    horizon: int = 20,
    action_dim: int = 7,
    scale: float = 10.0,
    vocab_size: int = 256,
    seed: int = 11,
) -> tuple[dict[str, object], Tensor, Tensor]:
    """Fit on train chunks and evaluate compression/reconstruction on validation."""
    chunks = make_smooth_action_chunks(
        train_count + validation_count,
        horizon=horizon,
        action_dim=action_dim,
        seed=seed,
    )
    train_chunks = chunks[:train_count]
    validation_chunks = chunks[train_count:]
    tokenizer = FastActionTokenizer.fit(
        train_chunks,
        scale=scale,
        vocab_size=vocab_size,
    )

    token_sequences = tokenizer.encode_batch(validation_chunks)
    reconstructions = torch.stack(
        [tokenizer.decode(tokens) for tokens in token_sequences]
    )
    token_lengths = torch.tensor([len(tokens) for tokens in token_sequences], dtype=torch.float32)
    coefficient_count = tokenizer.coefficient_count

    normalized = tokenizer.stats.normalize(validation_chunks)
    scalar_quantized = torch.round(normalized * scale) / scale
    scalar_reconstruction = tokenizer.stats.denormalize(scalar_quantized)
    scalar_mae = torch.mean(torch.abs(scalar_reconstruction - validation_chunks)).item()
    fast_mae = torch.mean(torch.abs(reconstructions - validation_chunks)).item()
    mean_tokens = token_lengths.mean().item()

    metrics: dict[str, object] = {
        "split": {
            "train_chunks": train_count,
            "validation_chunks": validation_count,
            "normalization_source": "train only",
        },
        "shape": {"horizon": horizon, "action_dim": action_dim},
        "tokenizer": {
            "scale": scale,
            "requested_vocab_size": vocab_size,
            "learned_vocab_size": tokenizer.bpe.vocab_size,
            "merge_rules": len(tokenizer.bpe.rules),
            "flatten_order": "frequency-major, then action dimension",
        },
        "scalar_quantization": {
            "tokens_per_chunk": coefficient_count,
            "validation_mae": scalar_mae,
        },
        "dct_without_bpe": {
            "symbols_per_chunk": coefficient_count,
            "validation_mae": fast_mae,
        },
        "fast_like_bpe": {
            "mean_tokens_per_chunk": mean_tokens,
            "min_tokens_per_chunk": int(token_lengths.min().item()),
            "max_tokens_per_chunk": int(token_lengths.max().item()),
            "compression_ratio": coefficient_count / mean_tokens,
            "validation_mae": fast_mae,
            "relative_quadratic_attention_work": (mean_tokens / coefficient_count) ** 2,
        },
    }
    return metrics, validation_chunks[0], reconstructions[0]


def write_fast_metrics(path: Path, metrics: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_fast_comparison_svg(
    path: Path,
    metrics: dict[str, object],
    original: Tensor,
    reconstructed: Tensor,
) -> None:
    """Write a dependency-free bar-and-trajectory figure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    coefficient_count = int(metrics["scalar_quantization"]["tokens_per_chunk"])  # type: ignore[index]
    mean_tokens = float(metrics["fast_like_bpe"]["mean_tokens_per_chunk"])  # type: ignore[index]
    fast_mae = float(metrics["fast_like_bpe"]["validation_mae"])  # type: ignore[index]
    compression = float(metrics["fast_like_bpe"]["compression_ratio"])  # type: ignore[index]

    width, height = 980, 500
    chart_top, chart_height = 100, 270
    max_bar = coefficient_count
    bar_width = 120
    raw_height = chart_height
    fast_height = chart_height * mean_tokens / max_bar

    values = torch.cat([original[:, 0], reconstructed[:, 0]])
    value_min, value_max = float(values.min()), float(values.max())
    span = max(value_max - value_min, 1e-6)

    def points(series: Tensor) -> str:
        result = []
        for index, value in enumerate(series.tolist()):
            x = 520 + index / (len(series) - 1) * 400
            y = 370 - (value - value_min) / span * 240
            result.append(f"{x:.1f},{y:.1f}")
        return " ".join(result)

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#f7f4ed"/>
<text x="50" y="48" font-family="sans-serif" font-size="26" font-weight="700" fill="#17212b">FAST-like：让平滑动作变成更短的 token 序列</text>
<text x="80" y="80" font-family="sans-serif" font-size="15" fill="#4b5563">每个 action chunk 的序列长度</text>
<line x1="60" y1="370" x2="440" y2="370" stroke="#9ca3af"/>
<rect x="110" y="{chart_top + chart_height - raw_height:.1f}" width="{bar_width}" height="{raw_height:.1f}" rx="8" fill="#dd6b55"/>
<rect x="290" y="{chart_top + chart_height - fast_height:.1f}" width="{bar_width}" height="{fast_height:.1f}" rx="8" fill="#287c78"/>
<text x="170" y="92" text-anchor="middle" font-family="sans-serif" font-size="18" font-weight="700">{coefficient_count}</text>
<text x="350" y="{chart_top + chart_height - fast_height - 10:.1f}" text-anchor="middle" font-family="sans-serif" font-size="18" font-weight="700">{mean_tokens:.1f}</text>
<text x="170" y="400" text-anchor="middle" font-family="sans-serif" font-size="15">逐标量 / DCT 系数</text>
<text x="350" y="400" text-anchor="middle" font-family="sans-serif" font-size="15">DCT + BPE</text>
<text x="260" y="442" text-anchor="middle" font-family="sans-serif" font-size="17" font-weight="700" fill="#287c78">压缩 {compression:.2f}×</text>
<text x="520" y="80" font-family="sans-serif" font-size="15" fill="#4b5563">第 0 个动作维度：原始与解码结果</text>
<line x1="520" y1="370" x2="920" y2="370" stroke="#9ca3af"/>
<polyline points="{points(original[:, 0])}" fill="none" stroke="#dd6b55" stroke-width="4"/>
<polyline points="{points(reconstructed[:, 0])}" fill="none" stroke="#287c78" stroke-width="3" stroke-dasharray="8 5"/>
<circle cx="545" cy="420" r="6" fill="#dd6b55"/><text x="558" y="425" font-family="sans-serif" font-size="14">原始动作</text>
<line x1="680" y1="420" x2="700" y2="420" stroke="#287c78" stroke-width="3" stroke-dasharray="7 4"/><text x="710" y="425" font-family="sans-serif" font-size="14">FAST-like 解码</text>
<text x="720" y="462" text-anchor="middle" font-family="sans-serif" font-size="15">验证集 MAE = {fast_mae:.4f}</text>
</svg>
"""
    path.write_text(svg, encoding="utf-8")
