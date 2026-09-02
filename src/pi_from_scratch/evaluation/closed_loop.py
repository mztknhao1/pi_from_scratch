"""Small dependency-free artifacts for closed-loop lesson results."""

import json
from pathlib import Path

import torch

from pi_from_scratch.runtime import ClosedLoopRun


def write_closed_loop_summary(path: Path, run: ClosedLoopRun, *, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    latencies = torch.tensor(run.episode.inference_latencies_s)
    summary = {
        "seed": seed,
        "success": run.episode.success,
        "num_steps": run.episode.num_steps,
        "num_replans": len(run.episode.inference_latencies_s),
        "total_reward": run.episode.total_reward,
        "refill_deadline_misses": run.episode.deadline_misses,
        "median_inference_latency_ms": float(latencies.median().item() * 1000.0),
        "chunk_boundary_steps": list(run.episode.chunk_boundary_steps),
        "failure_reason": run.episode.failure_reason,
    }
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_point_trajectory_svg(path: Path, run: ClosedLoopRun) -> None:
    """Draw agent states, action targets and the final goal for PointReachEnv."""
    if run.trace.states.shape[1] < 4 or run.trace.actions.shape[1] < 2:
        raise ValueError("point trajectory plot requires 2-D actions and [agent, target] state")
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height, margin = 720, 620, 70

    def project(point: torch.Tensor) -> tuple[float, float]:
        x = margin + (float(point[0]) + 1.0) * (width - 2 * margin) / 2.0
        y = height - margin - (float(point[1]) + 1.0) * (height - 2 * margin) / 2.0
        return x, y

    state_points = [project(point[:2]) for point in run.trace.states]
    action_points = [project(point[:2]) for point in run.trace.actions]
    goal = project(run.trace.states[0, 2:4])
    state_polyline = " ".join(f"{x:.2f},{y:.2f}" for x, y in state_points)
    action_circles = "\n".join(
        f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3" fill="#f2b84b" opacity="0.75" />'
        for x, y in action_points
    )
    boundary_circles = "\n".join(
        f'<circle cx="{state_points[index][0]:.2f}" cy="{state_points[index][1]:.2f}" '
        'r="7" fill="none" stroke="#a855f7" stroke-width="3" />'
        for index in run.trace.chunk_boundary_steps
    )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#101521" />
<text x="{margin}" y="38" fill="#f8fafc" font-family="sans-serif" font-size="23">Closed-loop trajectory</text>
<text x="{margin}" y="60" fill="#94a3b8" font-family="sans-serif" font-size="14">blue: executed state · yellow: policy command · purple: replan boundary</text>
<rect x="{margin}" y="{margin}" width="{width - 2 * margin}" height="{height - 2 * margin}" fill="#172033" stroke="#526078" />
<line x1="{width / 2}" y1="{margin}" x2="{width / 2}" y2="{height - margin}" stroke="#334155" />
<line x1="{margin}" y1="{height / 2}" x2="{width - margin}" y2="{height / 2}" stroke="#334155" />
<circle cx="{goal[0]:.2f}" cy="{goal[1]:.2f}" r="14" fill="#ef4444" opacity="0.9" />
{action_circles}
<polyline points="{state_polyline}" fill="none" stroke="#4ca5ff" stroke-width="5" stroke-linejoin="round" />
{boundary_circles}
<circle cx="{state_points[0][0]:.2f}" cy="{state_points[0][1]:.2f}" r="8" fill="#22c55e" />
<circle cx="{state_points[-1][0]:.2f}" cy="{state_points[-1][1]:.2f}" r="8" fill="#f8fafc" />
</svg>
"""
    path.write_text(svg, encoding="utf-8")
