"""Synchronous observe-plan-execute-reobserve loop used in lesson 9."""

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from pi_from_scratch.contracts import EpisodeResult, ObservationBatch
from pi_from_scratch.envs import ClosedLoopEnv
from pi_from_scratch.policies import Policy


@dataclass(frozen=True)
class ClosedLoopTrace:
    """Enough timing and state provenance to reconstruct one rollout."""

    states: Tensor
    actions: Tensor
    rewards: Tensor
    action_timestamps_s: Tensor
    source_observation_timestamps_s: Tensor
    plan_indices: Tensor
    chunk_offsets: Tensor
    chunk_boundary_steps: tuple[int, ...]

    def __post_init__(self) -> None:
        num_steps = self.actions.shape[0]
        if self.states.ndim != 2 or self.states.shape[0] != num_steps + 1:
            raise ValueError("states must have shape [num_steps + 1, state_dim]")
        if self.actions.ndim != 2:
            raise ValueError("actions must have shape [num_steps, action_dim]")
        for name, value in (
            ("rewards", self.rewards),
            ("action_timestamps_s", self.action_timestamps_s),
            ("source_observation_timestamps_s", self.source_observation_timestamps_s),
            ("plan_indices", self.plan_indices),
            ("chunk_offsets", self.chunk_offsets),
        ):
            if value.shape != (num_steps,):
                raise ValueError(f"{name} must have shape [num_steps]")
        if self.plan_indices.dtype != torch.long or self.chunk_offsets.dtype != torch.long:
            raise TypeError("plan indices and chunk offsets must be int64")
        if num_steps and not torch.isfinite(self.states).all().item():
            raise ValueError("trace states must be finite")
        if num_steps > 1 and torch.any(
            self.action_timestamps_s[1:] <= self.action_timestamps_s[:-1]
        ).item():
            raise ValueError("executed action timestamps must be strictly increasing")


@dataclass(frozen=True)
class ClosedLoopRun:
    episode: EpisodeResult
    trace: ClosedLoopTrace


def _validate_policy_output(
    observation: ObservationBatch,
    output: object,
    env: ClosedLoopEnv,
    *,
    execution_horizon: int,
) -> None:
    action_chunk = output.action_chunk  # type: ignore[attr-defined]
    if action_chunk.batch_size != 1:
        raise ValueError("closed-loop runner accepts batch size one")
    if action_chunk.spec != env.action_spec:
        raise ValueError("policy ActionSpec does not match the environment")
    if action_chunk.normalized:
        raise ValueError("policy must return denormalized physical-space actions")
    if execution_horizon > action_chunk.horizon:
        raise ValueError("execution_horizon must not exceed the policy horizon")
    if not torch.allclose(
        output.source_observation_timestamp_s,  # type: ignore[attr-defined]
        observation.timestamp_s,
        atol=1e-6,
        rtol=0.0,
    ):
        raise ValueError("policy output refers to a different observation timestamp")

    valid_count = int(action_chunk.valid_mask[0].sum().item())
    if valid_count < 1:
        raise ValueError("policy returned no valid actions")
    expected = observation.timestamp_s[0] + torch.arange(
        valid_count,
        device=action_chunk.timestamps_s.device,
        dtype=action_chunk.timestamps_s.dtype,
    ) / env.fps
    if not torch.allclose(
        action_chunk.timestamps_s[0, :valid_count], expected, atol=1e-5, rtol=0.0
    ):
        raise ValueError("policy action timestamps do not match the environment control grid")


def run_synchronous_episode(
    env: ClosedLoopEnv,
    policy: Policy,
    *,
    execution_horizon: int,
    max_steps: int,
    seed: int,
) -> ClosedLoopRun:
    """Run a blocking chunk policy while keeping simulator and wall time separate.

    The simulator advances only in ``env.step``. Inference latency is recorded and
    compared with the time covered by one executed prefix; this diagnostic becomes
    the asynchronous buffer refill deadline in lesson 10.
    """
    if execution_horizon < 1 or max_steps < 1:
        raise ValueError("execution_horizon and max_steps must be positive")
    if not math.isfinite(env.fps) or env.fps <= 0:
        raise ValueError("environment fps must be finite and positive")

    observation = env.reset(seed=seed)
    if observation.batch_size != 1:
        raise ValueError("closed-loop runner accepts batch size one")
    states = [observation.state[0].detach().cpu()]
    actions: list[Tensor] = []
    rewards: list[float] = []
    action_timestamps: list[float] = []
    source_timestamps: list[float] = []
    plan_indices: list[int] = []
    chunk_offsets: list[int] = []
    boundaries: list[int] = []
    latencies: list[float] = []
    refill_deadline_misses = 0
    success = False
    finished = False
    plan_index = 0

    while len(actions) < max_steps and not finished:
        output = policy.predict_chunk(observation)
        _validate_policy_output(
            observation,
            output,
            env,
            execution_horizon=execution_horizon,
        )
        latencies.append(output.inference_latency_s)
        valid_count = int(output.action_chunk.valid_mask[0].sum().item())
        execute_count = min(execution_horizon, valid_count, max_steps - len(actions))
        refill_budget_s = execution_horizon / env.fps
        if output.inference_latency_s > refill_budget_s:
            refill_deadline_misses += 1
        if actions:
            boundaries.append(len(actions))

        for chunk_offset in range(execute_count):
            action = output.action_chunk.values[0, chunk_offset].detach().cpu()
            transition = env.step(action)
            actions.append(action)
            rewards.append(transition.reward)
            action_timestamps.append(
                float(output.action_chunk.timestamps_s[0, chunk_offset].item())
            )
            source_timestamps.append(
                float(output.source_observation_timestamp_s[0].item())
            )
            plan_indices.append(plan_index)
            chunk_offsets.append(chunk_offset)
            states.append(transition.observation.state[0].detach().cpu())
            observation = transition.observation
            success = transition.success
            finished = transition.terminated or transition.truncated
            if finished:
                break
        plan_index += 1

    if not actions:
        raise RuntimeError("episode produced no environment steps")
    action_tensor = torch.stack(actions)
    trace = ClosedLoopTrace(
        states=torch.stack(states),
        actions=action_tensor,
        rewards=torch.tensor(rewards, dtype=torch.float32),
        action_timestamps_s=torch.tensor(action_timestamps, dtype=torch.float32),
        source_observation_timestamps_s=torch.tensor(source_timestamps, dtype=torch.float32),
        plan_indices=torch.tensor(plan_indices, dtype=torch.long),
        chunk_offsets=torch.tensor(chunk_offsets, dtype=torch.long),
        chunk_boundary_steps=tuple(boundaries),
    )
    failure_reason = None if success else "episode ended before the task succeeded"
    episode = EpisodeResult(
        success=success,
        total_reward=float(trace.rewards.sum().item()),
        num_steps=action_tensor.shape[0],
        inference_latencies_s=tuple(latencies),
        deadline_misses=refill_deadline_misses,
        executed_actions=action_tensor,
        chunk_boundary_steps=tuple(boundaries),
        failure_reason=failure_reason,
    )
    return ClosedLoopRun(episode=episode, trace=trace)
