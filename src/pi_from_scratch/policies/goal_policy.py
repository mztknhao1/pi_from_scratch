"""A transparent policy for verifying closed-loop runtime behavior."""

import time

import torch

from pi_from_scratch.contracts import ActionChunk, ActionSpec, ObservationBatch, PolicyOutput


class PointGoalPolicy:
    """Plan a straight absolute-position chunk from state ``[agent, target]``."""

    def __init__(
        self,
        action_spec: ActionSpec,
        *,
        horizon: int,
        fps: float,
        motion_per_step: float,
        simulated_latency_s: float | None = None,
    ) -> None:
        if (
            horizon < 1
            or fps <= 0
            or motion_per_step <= 0
            or (simulated_latency_s is not None and simulated_latency_s < 0)
        ):
            raise ValueError("policy horizon, fps, motion and latency must be valid")
        if action_spec.dim != 2:
            raise ValueError("PointGoalPolicy requires a two-dimensional action")
        self.action_spec = action_spec
        self.horizon = horizon
        self.fps = fps
        self.motion_per_step = motion_per_step
        self.simulated_latency_s = simulated_latency_s

    def predict_chunk(self, observation: ObservationBatch) -> PolicyOutput:
        started_at = time.monotonic()
        if observation.state.shape[1] < 4:
            raise ValueError("PointGoalPolicy expects state [agent_x, agent_y, target_x, target_y]")
        agent = observation.state[:, :2]
        target = observation.state[:, 2:4]
        displacement = target - agent
        distance = torch.linalg.vector_norm(displacement, dim=-1, keepdim=True)
        direction = displacement / distance.clamp_min(1e-8)
        step_numbers = torch.arange(
            1,
            self.horizon + 1,
            device=agent.device,
            dtype=agent.dtype,
        )[None, :, None]
        travel = (step_numbers * self.motion_per_step).minimum(distance[:, None])
        actions = agent[:, None] + direction[:, None] * travel
        minimum = torch.tensor(self.action_spec.minimum, device=agent.device, dtype=agent.dtype)
        maximum = torch.tensor(self.action_spec.maximum, device=agent.device, dtype=agent.dtype)
        actions = actions.clamp(minimum, maximum)
        offsets = torch.arange(
            self.horizon,
            device=agent.device,
            dtype=observation.timestamp_s.dtype,
        ) / self.fps
        finished_at = time.monotonic()
        latency_s = (
            finished_at - started_at
            if self.simulated_latency_s is None
            else self.simulated_latency_s
        )
        return PolicyOutput(
            action_chunk=ActionChunk(
                values=actions,
                valid_mask=torch.ones(
                    observation.batch_size,
                    self.horizon,
                    dtype=torch.bool,
                    device=agent.device,
                ),
                timestamps_s=observation.timestamp_s[:, None] + offsets[None],
                spec=self.action_spec,
            ),
            source_observation_timestamp_s=observation.timestamp_s.clone(),
            generated_at_monotonic_s=finished_at,
            inference_latency_s=latency_s,
        )
