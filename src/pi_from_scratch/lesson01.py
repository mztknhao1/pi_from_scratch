"""Executable contract probe for lesson 1."""

import json

import torch

from pi_from_scratch.contracts import (
    ActionRepresentation,
    ActionSpec,
    EpisodeResult,
    ObservationBatch,
)
from pi_from_scratch.policies.random_policy import RandomPolicy


def make_observation(state: torch.Tensor, step: int, fps: float) -> ObservationBatch:
    batch_size = state.shape[0]
    return ObservationBatch(
        images={"front": torch.zeros(batch_size, 3, 32, 32)},
        image_masks={"front": torch.ones(batch_size, dtype=torch.bool)},
        state=state,
        state_mask=torch.ones_like(state, dtype=torch.bool),
        prompts=tuple("move the point" for _ in range(batch_size)),
        timestamp_s=torch.full((batch_size,), step / fps),
    )


def run_contract_probe(num_replans: int = 3) -> EpisodeResult:
    """Run observe -> predict -> consume once, without claiming task performance."""
    fps = 10.0
    spec = ActionSpec(
        dim=2,
        space="toy_planar_position",
        representation=ActionRepresentation.ABSOLUTE,
        frame="world",
        units=("normalized_position", "normalized_position"),
        minimum=(-1.0, -1.0),
        maximum=(1.0, 1.0),
    )
    policy = RandomPolicy(spec, horizon=4, fps=fps, seed=7)
    state = torch.zeros(1, 2)
    executed_actions = []
    latencies = []
    for step in range(num_replans):
        observation = make_observation(state, step=step, fps=fps)
        output = policy.predict_chunk(observation)
        action = output.action_chunk.values[:, 0]
        state = action
        executed_actions.append(action.squeeze(0))
        latencies.append(output.inference_latency_s)
    return EpisodeResult(
        success=False,
        total_reward=0.0,
        num_steps=num_replans,
        inference_latencies_s=tuple(latencies),
        deadline_misses=0,
        executed_actions=torch.stack(executed_actions),
        chunk_boundary_steps=tuple(range(num_replans)),
        failure_reason="contract probe has no task objective",
    )


def main() -> None:
    result = run_contract_probe()
    summary = {
        "contract": "ok",
        "num_replans": result.num_steps,
        "executed_actions_shape": list(result.executed_actions.shape),
        "chunk_boundary_steps": list(result.chunk_boundary_steps),
        "note": result.failure_reason,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
