import math

import pytest
import torch

from pi_from_scratch.cli.lesson05 import run_tiny_overfit
from pi_from_scratch.inference import euler_sample
from pi_from_scratch.objectives import (
    linear_flow_path,
    masked_flow_matching_loss,
    sample_flow_batch,
)


def test_linear_flow_path_has_expected_endpoints_and_velocity() -> None:
    actions = torch.tensor([[[1.0, -1.0], [2.0, -2.0]]])
    noise = torch.tensor([[[5.0, 3.0], [6.0, 2.0]]])

    at_data = linear_flow_path(actions, noise, torch.tensor([0.0]))
    at_noise = linear_flow_path(actions, noise, torch.tensor([1.0]))
    at_quarter = linear_flow_path(actions, noise, torch.tensor([0.25]))

    torch.testing.assert_close(at_data.noisy_actions, actions)
    torch.testing.assert_close(at_noise.noisy_actions, noise)
    torch.testing.assert_close(at_quarter.noisy_actions, 0.75 * actions + 0.25 * noise)
    torch.testing.assert_close(at_quarter.target_velocity, noise - actions)


def test_sampling_direction_recovers_data_with_an_oracle_velocity() -> None:
    actions = torch.tensor([[[0.25, -0.5], [1.0, 0.75]]])
    noise = torch.tensor([[[2.0, 1.0], [-1.0, 3.0]]])

    def oracle_velocity(x_t: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        return (x_t - actions) / time[:, None, None]

    sampled = euler_sample(
        oracle_velocity,
        actions.shape,
        device=actions.device,
        num_steps=4,
        noise=noise,
    )

    torch.testing.assert_close(sampled, actions)


def test_reversing_velocity_moves_away_from_the_data() -> None:
    actions = torch.zeros(1, 2, 1)
    noise = torch.ones_like(actions)

    def reversed_velocity(x_t: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        return -(x_t - actions) / time[:, None, None]

    sampled = euler_sample(
        reversed_velocity,
        actions.shape,
        device=actions.device,
        num_steps=4,
        noise=noise,
    )

    assert torch.mean(torch.abs(sampled - actions)) > torch.mean(torch.abs(noise - actions))


def test_masked_flow_loss_ignores_padded_timesteps() -> None:
    predicted = torch.tensor([[[1.0], [100.0], [100.0]]])
    target = torch.zeros_like(predicted)
    valid_mask = torch.tensor([[True, False, False]])

    loss = masked_flow_matching_loss(predicted, target, valid_mask)

    assert loss.item() == pytest.approx(1.0)


def test_sample_flow_batch_contract_and_time_range() -> None:
    actions = torch.zeros(8, 4, 2)

    batch = sample_flow_batch(actions)

    assert batch.noisy_actions.shape == actions.shape
    assert batch.target_velocity.shape == actions.shape
    assert batch.noise.shape == actions.shape
    assert batch.time.shape == (8,)
    assert torch.all((batch.time >= 0.001) & (batch.time <= 1.0))


def test_linear_flow_path_rejects_invalid_time_shape_and_range() -> None:
    actions = torch.zeros(2, 4, 2)
    noise = torch.ones_like(actions)

    with pytest.raises(ValueError, match="shape"):
        linear_flow_path(actions, noise, torch.tensor([[0.5], [0.5]]))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        linear_flow_path(actions, noise, torch.tensor([0.5, 1.1]))


def test_fixed_flow_bank_can_be_overfit() -> None:
    result = run_tiny_overfit(steps=500, seed=7)

    assert result.final_loss < 0.01
    assert result.final_loss < result.initial_loss * 0.01
    assert math.isfinite(result.sampled_action_mae)
