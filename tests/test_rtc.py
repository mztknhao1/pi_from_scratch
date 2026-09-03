import pytest
import torch

from pi_from_scratch.config import ModelConfig
from pi_from_scratch.data import SyntheticPiDataset
from pi_from_scratch.inference import flow_sample, rtc_flow_sample, rtc_prefix_weights
from pi_from_scratch.models import TinyPi0
from pi_from_scratch.objectives import training_rtc_flow_batch
from pi_from_scratch.runtime import simulate_latency_runtime


def test_rtc_exponential_weights_have_frozen_decay_and_fresh_regions() -> None:
    weights = rtc_prefix_weights(
        16,
        delay_steps=3,
        execution_horizon=6,
        schedule="exponential",
    )

    torch.testing.assert_close(weights[:3], torch.ones(3))
    assert torch.all(weights[3:9] > weights[4:10])
    torch.testing.assert_close(weights[10:], torch.zeros(6))


def test_hard_mask_only_constrains_guaranteed_delay_prefix() -> None:
    weights = rtc_prefix_weights(16, delay_steps=3, execution_horizon=6, schedule="hard")

    torch.testing.assert_close(weights[:3], torch.ones(3))
    torch.testing.assert_close(weights[3:], torch.zeros(13))


def test_rtc_guidance_reduces_weighted_previous_chunk_error() -> None:
    shape = (1, 12, 2)
    generator = torch.Generator().manual_seed(7)
    noise = torch.randn(shape, generator=generator)
    target = torch.zeros(shape)
    previous = torch.ones(1, 7, 2) * 0.7

    def velocity(_actions: torch.Tensor, _time: torch.Tensor) -> torch.Tensor:
        return target - noise

    base = flow_sample(
        velocity,
        shape,
        device=torch.device("cpu"),
        num_steps=20,
        noise=noise,
    )
    guided = rtc_flow_sample(
        velocity,
        shape,
        previous_actions=previous,
        delay_steps=2,
        execution_horizon=5,
        num_steps=20,
        device=torch.device("cpu"),
        noise=noise,
    )
    padded = torch.zeros_like(base)
    padded[:, : previous.shape[1]] = previous
    weights = rtc_prefix_weights(12, delay_steps=2, execution_horizon=5)
    base_error = ((base - padded).abs() * weights[None, :, None]).sum()
    guided_error = ((guided - padded).abs() * weights[None, :, None]).sum()

    assert guided_error < base_error * 0.25
    assert not torch.allclose(guided[:, -5:], padded[:, -5:])


def test_rtc_reduces_chunk_handoff_jump_without_blocking() -> None:
    blocking = simulate_latency_runtime("blocking")
    naive = simulate_latency_runtime("naive_async")
    rtc = simulate_latency_runtime("rtc")

    assert rtc.throughput_hz == pytest.approx(naive.throughput_hz)
    assert rtc.throughput_hz > blocking.throughput_hz
    assert rtc.boundary_jumps.mean() < naive.boundary_jumps.mean() * 0.2
    assert rtc.boundary_jerks.mean() < naive.boundary_jerks.mean()
    torch.testing.assert_close(
        rtc.boundary_observation_ages_s,
        naive.boundary_observation_ages_s,
    )


def test_rtc_rejects_an_infeasible_execution_horizon() -> None:
    with pytest.raises(ValueError, match="horizon - delay"):
        simulate_latency_runtime("rtc", horizon=8, execution_horizon=6, delay_steps=3)


def test_training_time_rtc_keeps_prefix_clean_and_masks_its_loss() -> None:
    actions = torch.randn(2, 6, 3)
    noise = torch.randn_like(actions)
    batch = training_rtc_flow_batch(
        actions,
        torch.tensor([2, 4]),
        noise=noise,
        time=torch.tensor([0.7, 0.5]),
    )

    torch.testing.assert_close(batch.noisy_actions[0, :2], actions[0, :2])
    torch.testing.assert_close(batch.noisy_actions[1, :4], actions[1, :4])
    torch.testing.assert_close(batch.token_time[0, :2], torch.ones(2))
    assert not batch.loss_mask[0, :2].any()
    assert batch.loss_mask[0, 2:].all()


def test_tiny_pi0_training_time_rtc_loss_backpropagates() -> None:
    config = ModelConfig(
        image_size=32,
        action_horizon=4,
        width=32,
        num_layers=1,
        num_heads=4,
    )
    dataset = SyntheticPiDataset(config, length=2)
    batch = {key: torch.stack([dataset[0][key], dataset[1][key]]) for key in dataset[0]}
    model = TinyPi0(config)
    loss = model.training_rtc_loss(
        batch,
        prefix_lengths=torch.tensor([1, 2]),
        noise=torch.randn_like(batch["actions"]),
        time=torch.tensor([0.6, 0.8]),
    )

    loss.backward()
    assert torch.isfinite(loss)
