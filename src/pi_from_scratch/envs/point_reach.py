"""A dependency-free 2-D environment for testing the complete control loop."""

import math

import torch
from torch import Tensor

from pi_from_scratch.contracts import ActionRepresentation, ActionSpec, ObservationBatch
from pi_from_scratch.envs.protocol import EnvTransition, validate_physical_action


class PointReachEnv:
    """Move a point toward a target using absolute 2-D position commands."""

    def __init__(
        self,
        *,
        fps: float = 10.0,
        max_steps: int = 80,
        max_motion_per_step: float = 0.12,
        success_radius: float = 0.04,
        image_size: int = 32,
    ) -> None:
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be finite and positive")
        if max_steps < 1 or max_motion_per_step <= 0 or success_radius <= 0:
            raise ValueError("step and distance settings must be positive")
        if image_size < 8:
            raise ValueError("image_size must be at least 8")
        self.fps = fps
        self.max_steps = max_steps
        self.max_motion_per_step = max_motion_per_step
        self.success_radius = success_radius
        self.image_size = image_size
        self.action_spec = ActionSpec(
            dim=2,
            space="planar_position",
            representation=ActionRepresentation.ABSOLUTE,
            frame="world",
            units=("normalized_position", "normalized_position"),
            minimum=(-1.0, -1.0),
            maximum=(1.0, 1.0),
        )
        self._agent = torch.zeros(2)
        self._target = torch.zeros(2)
        self._step = 0
        self._finished = False

    def reset(self, *, seed: int) -> ObservationBatch:
        generator = torch.Generator().manual_seed(seed)
        self._agent = torch.rand(2, generator=generator) * 0.5 - 0.9
        self._target = torch.rand(2, generator=generator) * 0.5 + 0.4
        self._target[1] *= -1.0
        self._step = 0
        self._finished = False
        return self._observation()

    def step(self, action: Tensor) -> EnvTransition:
        if self._finished:
            raise RuntimeError("reset the environment before stepping a finished episode")
        command = validate_physical_action(action, self.action_spec)
        displacement = command - self._agent
        distance_to_command = torch.linalg.vector_norm(displacement)
        if distance_to_command > self.max_motion_per_step:
            displacement = displacement * (self.max_motion_per_step / distance_to_command)
        self._agent = self._agent + displacement
        self._step += 1

        distance_to_goal = float(torch.linalg.vector_norm(self._target - self._agent))
        success = distance_to_goal <= self.success_radius
        truncated = self._step >= self.max_steps and not success
        self._finished = success or truncated
        reward = 1.0 - distance_to_goal / (2.0 * math.sqrt(2.0))
        return EnvTransition(
            observation=self._observation(),
            reward=reward,
            terminated=success,
            truncated=truncated,
            success=success,
        )

    def close(self) -> None:
        self._finished = True

    def _observation(self) -> ObservationBatch:
        state = torch.cat((self._agent, self._target))[None]
        return ObservationBatch(
            images={"front": self._render()[None]},
            image_masks={"front": torch.ones(1, dtype=torch.bool)},
            state=state,
            state_mask=torch.ones_like(state, dtype=torch.bool),
            prompts=("move the blue point to the red target",),
            timestamp_s=torch.tensor([self._step / self.fps], dtype=torch.float32),
        )

    def _render(self) -> Tensor:
        coordinates = torch.linspace(-1.0, 1.0, self.image_size)
        yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
        agent_mask = (xx - self._agent[0]).square() + (yy - self._agent[1]).square() < 0.04**2
        target_mask = (xx - self._target[0]).square() + (yy - self._target[1]).square() < 0.08**2
        image = torch.full((3, self.image_size, self.image_size), 0.08)
        image[0, target_mask] = 0.95
        image[1, target_mask] = 0.18
        image[2, target_mask] = 0.12
        image[0, agent_mask] = 0.10
        image[1, agent_mask] = 0.45
        image[2, agent_mask] = 1.00
        return image
