"""Artifacts for the deterministic RTC latency comparison."""

import json
from pathlib import Path

from pi_from_scratch.runtime import LatencyRuntimeTrace


def rtc_metrics(trace: LatencyRuntimeTrace) -> dict[str, float | int | str]:
    return {
        "method": trace.method,
        "num_actions": trace.actions.shape[0],
        "num_boundaries": len(trace.boundary_steps),
        "wall_time_s": trace.wall_time_s,
        "throughput_hz": trace.throughput_hz,
        "mean_boundary_jump": float(trace.boundary_jumps.mean().item()),
        "mean_boundary_jerk": float(trace.boundary_jerks.mean().item()),
        "mean_boundary_observation_age_s": float(
            trace.boundary_observation_ages_s.mean().item()
        ),
        "deadline_misses": trace.deadline_misses,
    }


def write_rtc_metrics(path: Path, traces: list[LatencyRuntimeTrace], *, config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"config": config, "results": [rtc_metrics(trace) for trace in traces]}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_rtc_comparison_svg(path: Path, traces: list[LatencyRuntimeTrace]) -> None:
    """Plot the command trajectory of all runtime strategies plus headline bars."""
    if len(traces) != 3:
        raise ValueError("RTC comparison plot expects three runtime traces")
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1080, 720
    colors = {"blocking": "#60a5fa", "naive_async": "#f97316", "rtc": "#22c55e"}
    labels = {"blocking": "Blocking", "naive_async": "Naive async", "rtc": "RTC"}

    def project(x: float, y: float, left: float, top: float) -> tuple[float, float]:
        return left + (x + 1.1) / 2.2 * 250, top + 230 - (y + 0.9) / 1.8 * 230

    trajectory_parts = []
    for panel, trace in enumerate(traces):
        left = 55 + panel * 330
        top = 105
        points = " ".join(
            f"{project(float(point[0]), float(point[1]), left, top)[0]:.1f},"
            f"{project(float(point[0]), float(point[1]), left, top)[1]:.1f}"
            for point in trace.actions
        )
        boundary_marks = "".join(
            f'<circle cx="{project(float(trace.actions[i, 0]), float(trace.actions[i, 1]), left, top)[0]:.1f}" '
            f'cy="{project(float(trace.actions[i, 0]), float(trace.actions[i, 1]), left, top)[1]:.1f}" '
            'r="5" fill="none" stroke="#e879f9" stroke-width="2" />'
            for i in trace.boundary_steps
        )
        trajectory_parts.append(
            f'<rect x="{left}" y="{top}" width="250" height="230" fill="#172033" stroke="#475569" />'
            f'<text x="{left}" y="{top - 16}" fill="#e2e8f0" font-size="18" font-family="sans-serif">'
            f'{labels[trace.method]}</text>'
            f'<polyline points="{points}" fill="none" stroke="{colors[trace.method]}" '
            'stroke-width="4" stroke-linejoin="round" />'
            f'{boundary_marks}'
        )

    max_jump = max(float(trace.boundary_jumps.mean()) for trace in traces) * 1.2
    max_jerk = max(float(trace.boundary_jerks.mean()) for trace in traces) * 1.2
    bars = []
    for index, trace in enumerate(traces):
        x = 120 + index * 300
        jump_h = float(trace.boundary_jumps.mean()) / max_jump * 150
        jerk_h = float(trace.boundary_jerks.mean()) / max_jerk * 150
        bars.append(
            f'<rect x="{x}" y="{655 - jump_h:.1f}" width="70" height="{jump_h:.1f}" fill="{colors[trace.method]}" />'
            f'<rect x="{x + 85}" y="{655 - jerk_h:.1f}" width="70" height="{jerk_h:.1f}" fill="{colors[trace.method]}" opacity="0.55" />'
            f'<text x="{x}" y="682" fill="#cbd5e1" font-size="14" font-family="sans-serif">{labels[trace.method]}</text>'
        )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#0f1420" />
<text x="55" y="42" fill="#f8fafc" font-size="27" font-family="sans-serif">Runtime latency comparison</text>
<text x="55" y="69" fill="#94a3b8" font-size="15" font-family="sans-serif">purple circles mark chunk handoff; alternate plans choose opposite arcs</text>
{''.join(trajectory_parts)}
<text x="55" y="410" fill="#f8fafc" font-size="21" font-family="sans-serif">Boundary metrics (normalized bar height)</text>
<rect x="55" y="450" width="970" height="220" fill="#172033" stroke="#475569" />
<text x="780" y="475" fill="#cbd5e1" font-size="14" font-family="sans-serif">solid: action jump · translucent: position-command jerk</text>
{''.join(bars)}
</svg>
"""
    path.write_text(svg, encoding="utf-8")
