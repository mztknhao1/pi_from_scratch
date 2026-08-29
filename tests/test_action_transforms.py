import pytest
import torch

from pi_from_scratch.contracts import (
    ActionChunk,
    ActionRepresentation,
    ActionSpec,
)
from pi_from_scratch.representations import (
    ActionNormalizer,
    CurrentStateDeltaTransform,
    FiniteDifferenceVelocityTransform,
    RunningActionStats,
)


def test_current_state_delta_is_not_adjacent_action_difference() -> None:
    current_state = torch.tensor([[10.0, 20.0]])
    targets = torch.tensor([[[11.0, 22.0], [13.0, 25.0], [16.0, 29.0]]])
    transform = CurrentStateDeltaTransform()

    represented = transform.forward(targets, current_state)

    torch.testing.assert_close(
        represented, torch.tensor([[[1.0, 2.0], [3.0, 5.0], [6.0, 9.0]]])
    )
    torch.testing.assert_close(transform.inverse(represented, current_state), targets)


def test_dimension_mask_can_leave_gripper_absolute() -> None:
    current_state = torch.tensor([[1.0, 2.0, 0.25]])
    targets = torch.tensor([[[1.5, 3.0, 0.8], [2.0, 4.0, 0.2]]])
    transform = CurrentStateDeltaTransform(torch.tensor([True, True, False]))

    represented = transform.forward(targets, current_state)

    torch.testing.assert_close(
        represented, torch.tensor([[[0.5, 1.0, 0.8], [1.0, 2.0, 0.2]]])
    )
    torch.testing.assert_close(transform.inverse(represented, current_state), targets)


def test_velocity_transform_uses_fps_and_round_trips() -> None:
    current_state = torch.tensor([[0.0]])
    targets = torch.tensor([[[0.1], [0.3], [0.6]]])
    transform = FiniteDifferenceVelocityTransform(fps=10.0)

    velocity = transform.forward(targets, current_state)

    torch.testing.assert_close(velocity, torch.tensor([[[1.0], [2.0], [3.0]]]))
    torch.testing.assert_close(transform.inverse(velocity, current_state), targets)


def test_statistics_ignore_padded_actions() -> None:
    accumulator = RunningActionStats()
    values = torch.tensor([[[1.0], [3.0], [1000.0]]])
    accumulator.update(values, torch.tensor([[True, True, False]]))
    stats = accumulator.finalize(train_episode_ids=(0,))

    torch.testing.assert_close(stats.mean, torch.tensor([2.0]))
    torch.testing.assert_close(stats.std, torch.tensor([1.0]))
    assert stats.count == 2


def test_normalizer_round_trips_chunk_and_checks_artifact() -> None:
    accumulator = RunningActionStats()
    accumulator.update(torch.tensor([[1.0, 10.0], [3.0, 14.0]]))
    stats = accumulator.finalize(train_episode_ids=(0, 2))
    normalizer = ActionNormalizer(stats)
    spec = ActionSpec(
        dim=2,
        space="toy_joint_position",
        representation=ActionRepresentation.ABSOLUTE,
        frame="robot_base",
        units=("rad", "rad"),
        minimum=(-10.0, -10.0),
        maximum=(10.0, 20.0),
    )
    raw = ActionChunk(
        values=torch.tensor([[[1.0, 10.0], [3.0, 14.0]]]),
        valid_mask=torch.ones(1, 2, dtype=torch.bool),
        timestamps_s=torch.tensor([[0.0, 0.1]]),
        spec=spec,
    )

    normalized = normalizer.normalize_chunk(raw)
    restored = normalizer.denormalize_chunk(normalized)

    torch.testing.assert_close(normalized.values, torch.tensor([[[-1.0, -1.0], [1.0, 1.0]]]))
    torch.testing.assert_close(restored.values, raw.values)
    assert normalized.normalization_id == stats.artifact_id
    assert not restored.normalized


def test_statistics_require_explicit_training_episode_provenance() -> None:
    accumulator = RunningActionStats()
    accumulator.update(torch.tensor([[0.0], [1.0]]))

    with pytest.raises(ValueError, match="train_episode_ids"):
        accumulator.finalize(train_episode_ids=())
