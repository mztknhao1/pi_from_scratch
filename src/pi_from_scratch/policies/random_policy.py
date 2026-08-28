import time

import torch

from pi_from_scratch.contracts import ActionChunk, ActionSpec, ObservationBatch, PolicyOutput


class RandomPolicy:
    """A no-learning policy used to verify the lesson-1 system contract."""

    def __init__(self, action_spec: ActionSpec, horizon: int, fps: float, seed: int = 0):
        if horizon < 1:
            raise ValueError("horizon must be positive")
        if fps <= 0:
            raise ValueError("fps must be positive")
        self.action_spec = action_spec
        self.horizon = horizon
        self.fps = fps
        self.generator = torch.Generator().manual_seed(seed)

    def predict_chunk(self, observation: ObservationBatch) -> PolicyOutput:
        started_at = time.monotonic()
        shape = (observation.batch_size, self.horizon, self.action_spec.dim)
        unit_actions = torch.rand(shape, generator=self.generator, dtype=torch.float32)
        minimum = torch.tensor(self.action_spec.minimum, dtype=torch.float32)[None, None]
        maximum = torch.tensor(self.action_spec.maximum, dtype=torch.float32)[None, None]
        actions = (minimum + unit_actions * (maximum - minimum)).to(observation.state.device)
        offsets = (
            torch.arange(
                self.horizon,
                dtype=observation.timestamp_s.dtype,
                device=observation.timestamp_s.device,
            )
            / self.fps
        )
        timestamps = observation.timestamp_s[:, None] + offsets[None, :]
        valid_mask = torch.ones(
            observation.batch_size,
            self.horizon,
            dtype=torch.bool,
            device=observation.state.device,
        )
        finished_at = time.monotonic()
        return PolicyOutput(
            action_chunk=ActionChunk(
                values=actions,
                valid_mask=valid_mask,
                timestamps_s=timestamps,
                spec=self.action_spec,
            ),
            source_observation_timestamp_s=observation.timestamp_s.clone(),
            generated_at_monotonic_s=finished_at,
            inference_latency_s=finished_at - started_at,
        )
