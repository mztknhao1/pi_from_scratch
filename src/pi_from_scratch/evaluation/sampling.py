"""Sampling-step, error, and latency sweeps for flow policies."""

import math
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from html import escape
from pathlib import Path

import torch
from torch import Tensor

from pi_from_scratch.inference import FlowSolver, model_evaluations


@dataclass(frozen=True)
class SamplingSweepPoint:
    solver: FlowSolver
    steps: int
    model_evaluations: int
    action_mae: float
    median_latency_ms: float


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _masked_action_mae(prediction: Tensor, target: Tensor, valid_mask: Tensor) -> float:
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("prediction and target must share shape [batch, horizon, action_dim]")
    if valid_mask.shape != target.shape[:2] or valid_mask.dtype != torch.bool:
        raise ValueError("valid_mask must be bool with shape [batch, horizon]")
    weights = valid_mask.unsqueeze(-1).to(target.dtype)
    return (
        (prediction - target).abs().mul(weights).sum()
        / weights.sum().clamp_min(1.0)
        / target.shape[-1]
    ).item()


@torch.no_grad()
def run_sampling_sweep(
    sample_fn: Callable[[FlowSolver, int], Tensor],
    target: Tensor,
    valid_mask: Tensor,
    *,
    solvers: Sequence[FlowSolver],
    steps: Sequence[int],
    warmup: int = 1,
    repeats: int = 10,
) -> list[SamplingSweepPoint]:
    """Compare solvers while the caller keeps checkpoint, observation, and noise fixed."""
    if warmup < 0 or repeats < 1:
        raise ValueError("warmup must be non-negative and repeats must be positive")
    if not solvers or not steps or any(value < 1 for value in steps):
        raise ValueError("solvers and positive step counts are required")

    points: list[SamplingSweepPoint] = []
    for solver in solvers:
        for num_steps in steps:
            prediction: Tensor | None = None
            for _ in range(warmup):
                prediction = sample_fn(solver, num_steps)
            latencies = []
            for _ in range(repeats):
                _synchronize(target.device)
                start = time.perf_counter()
                prediction = sample_fn(solver, num_steps)
                _synchronize(target.device)
                latencies.append((time.perf_counter() - start) * 1_000.0)
            assert prediction is not None
            points.append(
                SamplingSweepPoint(
                    solver=solver,
                    steps=num_steps,
                    model_evaluations=model_evaluations(solver, num_steps),
                    action_mae=_masked_action_mae(prediction, target, valid_mask),
                    median_latency_ms=statistics.median(latencies),
                )
            )
    return points


def sampling_points_to_json(points: Sequence[SamplingSweepPoint]) -> list[dict[str, object]]:
    return [asdict(point) for point in points]


def write_sampling_sweep_svg(
    points: Sequence[SamplingSweepPoint],
    path: Path,
    *,
    title: str,
) -> None:
    """Draw action error and median latency against solver steps."""
    if not points:
        raise ValueError("at least one sampling sweep point is required")
    if any(
        point.steps < 1
        or point.action_mae < 0.0
        or point.median_latency_ms <= 0.0
        or not math.isfinite(point.action_mae)
        or not math.isfinite(point.median_latency_ms)
        for point in points
    ):
        raise ValueError("sampling sweep points contain invalid values")

    width, height = 980.0, 440.0
    panel_width, panel_height = 390.0, 270.0
    panel_top = 100.0
    panel_lefts = (76.0, 566.0)
    colors = {"euler": "#2563eb", "heun": "#f97316"}
    all_steps = [point.steps for point in points]
    min_x, max_x = min(all_steps), max(all_steps)
    x_low, x_high = math.log2(min_x), math.log2(max_x)
    x_span = max(1e-6, x_high - x_low)

    def x_coordinate(step: int, left: float) -> float:
        return left + (math.log2(step) - x_low) / x_span * panel_width

    def panel_series(metric: str, left: float) -> tuple[str, list[str]]:
        values = [max(float(getattr(point, metric)), 1e-12) for point in points]
        logs = [math.log10(value) for value in values]
        lower, upper = min(logs), max(logs)
        span = max(1e-6, upper - lower)

        def y_coordinate(value: float) -> float:
            return panel_top + (upper - math.log10(max(value, 1e-12))) / span * panel_height

        shapes = []
        for solver in ("euler", "heun"):
            selected = sorted((point for point in points if point.solver == solver), key=lambda p: p.steps)
            if not selected:
                continue
            coordinates = " ".join(
                f"{x_coordinate(point.steps, left):.2f},{y_coordinate(float(getattr(point, metric))):.2f}"
                for point in selected
            )
            shapes.append(
                f'<polyline points="{coordinates}" fill="none" stroke="{colors[solver]}" '
                'stroke-width="4" stroke-linecap="round" stroke-linejoin="round" />'
            )
            shapes.extend(
                f'<circle cx="{x_coordinate(point.steps, left):.2f}" '
                f'cy="{y_coordinate(float(getattr(point, metric))):.2f}" r="4" '
                f'fill="{colors[solver]}" />'
                for point in selected
            )
        labels = [
            (
                f'<text x="{left - 64:.0f}" y="{panel_top + 5:.0f}" '
                f'font-family="monospace" font-size="12" '
                f'fill="#4b5563">{10**upper:.2g}</text>'
            ),
            (
                f'<text x="{left - 64:.0f}" y="{panel_top + panel_height + 5:.0f}" '
                f'font-family="monospace" font-size="12" '
                f'fill="#4b5563">{10**lower:.2g}</text>'
            ),
        ]
        return "\n".join(shapes), labels

    error_shapes, error_labels = panel_series("action_mae", panel_lefts[0])
    latency_shapes, latency_labels = panel_series("median_latency_ms", panel_lefts[1])
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}" viewBox="0 0 {width:.0f} {height:.0f}">
  <rect width="100%" height="100%" fill="#fbfbfd" />
  <text x="28" y="34" font-family="sans-serif" font-size="22" fill="#202124">{escape(title)}</text>
  <line x1="28" y1="60" x2="54" y2="60" stroke="#2563eb" stroke-width="4" />
  <text x="62" y="66" font-family="sans-serif" font-size="15" fill="#374151">Euler (1 NFE/step)</text>
  <line x1="220" y1="60" x2="246" y2="60" stroke="#f97316" stroke-width="4" />
  <text x="254" y="66" font-family="sans-serif" font-size="15" fill="#374151">Heun (2 NFE/step)</text>
  <text x="{panel_lefts[0]}" y="90" font-family="sans-serif" font-size="17" fill="#202124">action MAE (log)</text>
  <text x="{panel_lefts[1]}" y="90" font-family="sans-serif" font-size="17" fill="#202124">median latency ms (log)</text>
  <line x1="{panel_lefts[0]}" y1="{panel_top}" x2="{panel_lefts[0]}" y2="{panel_top + panel_height}" stroke="#9ca3af" />
  <line x1="{panel_lefts[0]}" y1="{panel_top + panel_height}" x2="{panel_lefts[0] + panel_width}" y2="{panel_top + panel_height}" stroke="#9ca3af" />
  <line x1="{panel_lefts[1]}" y1="{panel_top}" x2="{panel_lefts[1]}" y2="{panel_top + panel_height}" stroke="#9ca3af" />
  <line x1="{panel_lefts[1]}" y1="{panel_top + panel_height}" x2="{panel_lefts[1] + panel_width}" y2="{panel_top + panel_height}" stroke="#9ca3af" />
  {''.join(error_labels)}
  {''.join(latency_labels)}
  {error_shapes}
  {latency_shapes}
  <text x="{panel_lefts[0]}" y="405" font-family="monospace" font-size="12" fill="#4b5563">steps {min_x}</text>
  <text x="{panel_lefts[0] + panel_width - 62}" y="405" font-family="monospace" font-size="12" fill="#4b5563">{max_x}</text>
  <text x="{panel_lefts[1]}" y="405" font-family="monospace" font-size="12" fill="#4b5563">steps {min_x}</text>
  <text x="{panel_lefts[1] + panel_width - 62}" y="405" font-family="monospace" font-size="12" fill="#4b5563">{max_x}</text>
</svg>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")
