"""Deterministic discrete-event comparison of blocking, naive async, and RTC."""

import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from pi_from_scratch.inference import rtc_flow_sample

RuntimeMethod = Literal["blocking", "naive_async", "rtc"]


@dataclass(frozen=True)
class LatencyRuntimeTrace:
    method: RuntimeMethod
    actions: Tensor
    boundary_steps: tuple[int, ...]
    boundary_jumps: Tensor
    boundary_jerks: Tensor
    boundary_observation_ages_s: Tensor
    wall_time_s: float
    throughput_hz: float
    deadline_misses: int


def _arc_chunk(start: Tensor, target: Tensor, *, sign: float, horizon: int) -> Tensor:
    progress = torch.arange(1, horizon + 1, dtype=start.dtype) / horizon
    straight = start[None] + progress[:, None] * (target - start)[None]
    straight[:, 1] += 0.55 * torch.sin(math.pi * progress) * sign
    return straight


def _rtc_condition_chunk(
    candidate: Tensor,
    previous: Tensor,
    *,
    delay_steps: int,
    execution_horizon: int,
    num_denoising_steps: int,
    seed: int,
) -> Tensor:
    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn(candidate.shape, generator=generator)[None]

    def analytic_velocity(_actions: Tensor, _time: Tensor) -> Tensor:
        return candidate[None] - noise

    return rtc_flow_sample(
        analytic_velocity,
        tuple(noise.shape),
        previous_actions=previous[None],
        delay_steps=delay_steps,
        execution_horizon=execution_horizon,
        num_steps=num_denoising_steps,
        device=torch.device("cpu"),
        noise=noise,
        schedule="exponential",
        max_guidance_weight=5.0,
    )[0]


def _boundary_metrics(
    actions: Tensor, boundary_steps: tuple[int, ...], fps: float
) -> tuple[Tensor, Tensor]:
    jumps = []
    jerks = []
    for index in boundary_steps:
        jumps.append(torch.linalg.vector_norm(actions[index] - actions[index - 1]))
        if index >= 3:
            third_difference = (
                actions[index]
                - 3 * actions[index - 1]
                + 3 * actions[index - 2]
                - actions[index - 3]
            )
            jerks.append(torch.linalg.vector_norm(third_difference) * fps**3)
    return (
        torch.stack(jumps) if jumps else torch.empty(0),
        torch.stack(jerks) if jerks else torch.empty(0),
    )


def simulate_latency_runtime(
    method: RuntimeMethod,
    *,
    horizon: int = 16,
    execution_horizon: int = 6,
    delay_steps: int = 3,
    num_replans: int = 5,
    fps: float = 10.0,
    num_denoising_steps: int = 20,
    seed: int = 7,
) -> LatencyRuntimeTrace:
    """Run a transparent multimodal-plan switch experiment on a control-step clock."""
    if method not in ("blocking", "naive_async", "rtc"):
        raise ValueError(f"unknown runtime method: {method}")
    if horizon < 2 or execution_horizon < 1 or num_replans < 1 or fps <= 0:
        raise ValueError("horizon, execution horizon, replans, and fps must be positive")
    if delay_steps < 0 or delay_steps > execution_horizon:
        raise ValueError("delay_steps must satisfy 0 <= d <= execution_horizon")
    if execution_horizon > horizon - delay_steps:
        raise ValueError("RTC requires execution_horizon <= horizon - delay_steps")

    target = torch.tensor([1.0, 0.0])
    current = _arc_chunk(torch.tensor([-1.0, 0.0]), target, sign=1.0, horizon=horizon)
    current_index = execution_horizon
    executed = [action.clone() for action in current[:execution_horizon]]
    boundaries: list[int] = []
    boundary_ages: list[float] = []
    wall_time_s = execution_horizon / fps
    deadline_misses = 0

    for replan in range(num_replans):
        observation_position = executed[-1]
        sign = -1.0 if replan % 2 == 0 else 1.0
        candidate = _arc_chunk(observation_position, target, sign=sign, horizon=horizon)
        latency_s = delay_steps / fps

        if method == "blocking":
            wall_time_s += latency_s
            boundaries.append(len(executed))
            boundary_ages.append(latency_s)
            executed.extend(action.clone() for action in candidate[:execution_horizon])
            current = candidate
            current_index = execution_horizon
        else:
            available = current.shape[0] - current_index
            if available < delay_steps:
                deadline_misses += 1
                break
            previous = current[current_index:]
            generated = candidate
            if method == "rtc":
                generated = _rtc_condition_chunk(
                    candidate,
                    previous,
                    delay_steps=delay_steps,
                    execution_horizon=execution_horizon,
                    num_denoising_steps=num_denoising_steps,
                    seed=seed + replan,
                )
            executed.extend(
                action.clone() for action in current[current_index : current_index + delay_steps]
            )
            boundaries.append(len(executed))
            boundary_ages.append(latency_s)
            executed.extend(action.clone() for action in generated[delay_steps:execution_horizon])
            current = generated
            current_index = execution_horizon
        wall_time_s += execution_horizon / fps

    actions = torch.stack(executed)
    boundary_tuple = tuple(boundaries)
    jumps, jerks = _boundary_metrics(actions, boundary_tuple, fps)
    return LatencyRuntimeTrace(
        method=method,
        actions=actions,
        boundary_steps=boundary_tuple,
        boundary_jumps=jumps,
        boundary_jerks=jerks,
        boundary_observation_ages_s=torch.tensor(boundary_ages),
        wall_time_s=wall_time_s,
        throughput_hz=actions.shape[0] / wall_time_s,
        deadline_misses=deadline_misses,
    )
