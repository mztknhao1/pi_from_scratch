"""Dependency-free 2D action-trajectory visualization."""

import math
from itertools import pairwise
from pathlib import Path

import torch
from torch import Tensor


def write_loss_curve_svg(
    steps: list[int],
    train_loss: list[float],
    validation_loss: list[float],
    path: Path,
) -> None:
    """Write fixed-bank train and validation losses on a logarithmic y-axis."""
    if not steps or len(steps) != len(train_loss) or len(steps) != len(validation_loss):
        raise ValueError("steps, train_loss, and validation_loss must have the same non-zero length")
    if any(step < 0 for step in steps) or any(b <= a for a, b in pairwise(steps)):
        raise ValueError("steps must be non-negative and strictly increasing")
    losses = train_loss + validation_loss
    if any(not math.isfinite(value) or value < 0.0 for value in losses):
        raise ValueError("loss values must be finite and non-negative")

    width, height = 720.0, 420.0
    left, right, top, bottom = 76.0, 28.0, 72.0, 56.0
    x_span = max(1, steps[-1] - steps[0])
    log_losses = [math.log10(max(value, 1e-12)) for value in losses]
    lower, upper = min(log_losses), max(log_losses)
    y_span = max(1e-6, upper - lower)

    def points(values: list[float]) -> str:
        coordinates = []
        for step, value in zip(steps, values, strict=True):
            x = left + (step - steps[0]) / x_span * (width - left - right)
            y = top + (upper - math.log10(max(value, 1e-12))) / y_span * (
                height - top - bottom
            )
            coordinates.append(f"{x:.2f},{y:.2f}")
        return " ".join(coordinates)

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}" viewBox="0 0 {width:.0f} {height:.0f}">
  <rect width="100%" height="100%" fill="#fbfbfd" />
  <text x="28" y="34" font-family="sans-serif" font-size="22" fill="#202124">Fixed flow-bank loss</text>
  <line x1="28" y1="56" x2="52" y2="56" stroke="#2563eb" stroke-width="4" />
  <text x="60" y="62" font-family="sans-serif" font-size="15" fill="#374151">train</text>
  <line x1="120" y1="56" x2="144" y2="56" stroke="#f97316" stroke-width="4" />
  <text x="152" y="62" font-family="sans-serif" font-size="15" fill="#374151">validation</text>
  <line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#9ca3af" />
  <line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#9ca3af" />
  <text x="{left - 62}" y="{top + 6}" font-family="monospace" font-size="13" fill="#4b5563">{10**upper:.2g}</text>
  <text x="{left - 62}" y="{height - bottom + 5}" font-family="monospace" font-size="13" fill="#4b5563">{10**lower:.2g}</text>
  <text x="{left}" y="{height - 20}" font-family="monospace" font-size="13" fill="#4b5563">step {steps[0]}</text>
  <text x="{width - right - 72}" y="{height - 20}" font-family="monospace" font-size="13" fill="#4b5563">step {steps[-1]}</text>
  <polyline points="{points(train_loss)}" fill="none" stroke="#2563eb" stroke-width="4" stroke-linecap="round" stroke-linejoin="round" />
  <polyline points="{points(validation_loss)}" fill="none" stroke="#f97316" stroke-width="4" stroke-linecap="round" stroke-linejoin="round" />
</svg>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")


def write_trajectory_svg(target: Tensor, prediction: Tensor, path: Path) -> None:
    """Write target and predicted ``[horizon, 2]`` trajectories to an SVG file."""
    if target.ndim != 2 or target.shape[1] != 2 or prediction.shape != target.shape:
        raise ValueError("target and prediction must share shape [horizon, 2]")
    if target.shape[0] < 1:
        raise ValueError("trajectory must contain at least one point")
    values = torch.cat((target, prediction), dim=0).detach().float().cpu()
    if not torch.isfinite(values).all().item():
        raise ValueError("trajectory values must be finite")

    width, height, padding = 720.0, 480.0, 64.0
    lower = values.amin(dim=0)
    upper = values.amax(dim=0)
    span = (upper - lower).clamp_min(1e-6)
    usable = torch.tensor([width - 2 * padding, height - 2 * padding])
    scale = torch.min(usable / span).item()

    def points(value: Tensor) -> list[tuple[float, float]]:
        xy = (value.detach().float().cpu() - lower) * scale
        x = xy[:, 0] + padding
        y = height - padding - xy[:, 1]
        return list(zip(x.tolist(), y.tolist(), strict=True))

    target_points = points(target)
    prediction_points = points(prediction)

    def polyline(value: list[tuple[float, float]]) -> str:
        return " ".join(f"{x:.2f},{y:.2f}" for x, y in value)

    def circles(value: list[tuple[float, float]], color: str) -> str:
        return "\n".join(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{color}" />'
            for x, y in value
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}" viewBox="0 0 {width:.0f} {height:.0f}">
  <rect width="100%" height="100%" fill="#fbfbfd" />
  <text x="32" y="36" font-family="sans-serif" font-size="22" fill="#202124">TinyPi0 action chunk</text>
  <line x1="32" y1="64" x2="58" y2="64" stroke="#2563eb" stroke-width="4" />
  <text x="66" y="70" font-family="sans-serif" font-size="16" fill="#374151">target</text>
  <line x1="142" y1="64" x2="168" y2="64" stroke="#f97316" stroke-width="4" />
  <text x="176" y="70" font-family="sans-serif" font-size="16" fill="#374151">prediction</text>
  <polyline points="{polyline(target_points)}" fill="none" stroke="#2563eb" stroke-width="4" stroke-linecap="round" stroke-linejoin="round" />
  <polyline points="{polyline(prediction_points)}" fill="none" stroke="#f97316" stroke-width="4" stroke-linecap="round" stroke-linejoin="round" />
  {circles(target_points, "#2563eb")}
  {circles(prediction_points, "#f97316")}
</svg>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")
