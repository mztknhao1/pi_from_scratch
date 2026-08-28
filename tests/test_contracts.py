import pytest
import torch

from pi_from_scratch.cli.lesson01 import make_observation, run_contract_probe
from pi_from_scratch.contracts import (
    ActionChunk,
    ActionRepresentation,
    ActionSpec,
    ObservationBatch,
)
from pi_from_scratch.policies.protocol import Policy
from pi_from_scratch.policies.random_policy import RandomPolicy


def make_action_spec() -> ActionSpec:
    return ActionSpec(
        dim=2,
        space="toy_planar_position",
        representation=ActionRepresentation.ABSOLUTE,
        frame="world",
        units=("normalized_position", "normalized_position"),
        minimum=(-1.0, -1.0),
        maximum=(1.0, 1.0),
    )


def test_observation_rejects_camera_batch_mismatch() -> None:
    with pytest.raises(ValueError, match="image 'front'"):
        ObservationBatch(
            images={"front": torch.zeros(2, 3, 16, 16)},
            image_masks={"front": torch.ones(1, dtype=torch.bool)},
            state=torch.zeros(1, 2),
            state_mask=torch.ones(1, 2, dtype=torch.bool),
            prompts=("move",),
            timestamp_s=torch.zeros(1),
        )


def test_action_chunk_rejects_non_increasing_valid_timestamps() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        ActionChunk(
            values=torch.zeros(1, 3, 2),
            valid_mask=torch.ones(1, 3, dtype=torch.bool),
            timestamps_s=torch.tensor([[0.0, 0.1, 0.1]]),
            spec=make_action_spec(),
        )


def test_action_chunk_rejects_valid_steps_after_padding() -> None:
    with pytest.raises(ValueError, match="contiguous valid prefix"):
        ActionChunk(
            values=torch.zeros(1, 3, 2),
            valid_mask=torch.tensor([[True, False, True]]),
            timestamps_s=torch.tensor([[0.0, 0.1, 0.2]]),
            spec=make_action_spec(),
        )


def test_random_policy_satisfies_policy_contract() -> None:
    policy = RandomPolicy(make_action_spec(), horizon=4, fps=10.0, seed=1)
    assert isinstance(policy, Policy)
    observation = make_observation(torch.zeros(2, 2), step=2, fps=10.0)
    output = policy.predict_chunk(observation)
    assert output.action_chunk.values.shape == (2, 4, 2)
    assert output.action_chunk.valid_mask.all()
    torch.testing.assert_close(
        output.action_chunk.timestamps_s[0], torch.tensor([0.2, 0.3, 0.4, 0.5])
    )
    torch.testing.assert_close(output.source_observation_timestamp_s, observation.timestamp_s)


def test_contract_probe_completes_three_replans() -> None:
    result = run_contract_probe(num_replans=3)
    assert result.num_steps == 3
    assert result.executed_actions.shape == (3, 2)
    assert result.chunk_boundary_steps == (0, 1, 2)
