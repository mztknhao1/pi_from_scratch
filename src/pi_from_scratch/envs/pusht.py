"""Optional adapter for the official ``gym-pusht`` environment."""

from typing import Any

import torch
from torch import Tensor

from pi_from_scratch.contracts import ActionRepresentation, ActionSpec, ObservationBatch
from pi_from_scratch.envs.protocol import EnvTransition, validate_physical_action


class GymPushTEnv:
    """Expose gym-pusht through the same boundary as the dependency-free toy env."""

    def __init__(self, *, fps: float = 10.0, max_steps: int = 300) -> None:
        try:
            import gym_pusht  # noqa: F401
            import gymnasium as gym
        except ImportError as exc:
            raise ImportError(
                "PushT simulation dependencies are missing. Use Python 3.11/3.12 and run "
                "pip install -e '.[sim]'"
            ) from exc
        self.fps = fps
        self.action_spec = ActionSpec(
            dim=2,
            space="pusht_planar_target",
            representation=ActionRepresentation.ABSOLUTE,
            frame="pusht_workspace",
            units=("workspace_pixel", "workspace_pixel"),
            minimum=(0.0, 0.0),
            maximum=(512.0, 512.0),
        )
        self._env = gym.make(
            "gym_pusht/PushT-v0",
            obs_type="pixels_agent_pos",
            render_mode="rgb_array",
            max_episode_steps=max_steps,
        )
        self._step = 0

    def reset(self, *, seed: int) -> ObservationBatch:
        raw, _ = self._env.reset(seed=seed)
        self._step = 0
        return self._observation(raw)

    def step(self, action: Tensor) -> EnvTransition:
        command = validate_physical_action(action, self.action_spec).numpy()
        raw, reward, terminated, truncated, info = self._env.step(command)
        self._step += 1
        success = bool(info.get("is_success", float(reward) >= 0.95))
        terminated = bool(terminated) or success
        return EnvTransition(
            observation=self._observation(raw),
            reward=float(reward),
            terminated=terminated,
            truncated=bool(truncated),
            success=success,
        )

    def close(self) -> None:
        self._env.close()

    def _observation(self, raw: dict[str, Any]) -> ObservationBatch:
        image = torch.as_tensor(raw["pixels"])
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError("gym-pusht pixels must have shape [height, width, 3]")
        image = image.permute(2, 0, 1).float() / 255.0
        state = torch.as_tensor(raw["agent_pos"], dtype=torch.float32)[None]
        return ObservationBatch(
            images={"front": image[None]},
            image_masks={"front": torch.ones(1, dtype=torch.bool)},
            state=state,
            state_mask=torch.ones_like(state, dtype=torch.bool),
            prompts=("push the T-shaped block to the target",),
            timestamp_s=torch.tensor([self._step / self.fps], dtype=torch.float32),
        )
