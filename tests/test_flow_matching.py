import torch

from pi_from_scratch.inference import euler_sample
from pi_from_scratch.objectives import sample_flow_batch


def test_flow_batch_shapes() -> None:
    actions = torch.randn(4, 8, 2)
    noisy, time, target = sample_flow_batch(actions)
    assert noisy.shape == actions.shape
    assert target.shape == actions.shape
    assert time.shape == (4,)
    assert torch.all((time > 0) & (time <= 1))


def test_euler_sample_constant_velocity() -> None:
    noise = torch.ones(2, 3, 1)
    result = euler_sample(
        lambda actions, time: torch.ones_like(actions),
        noise.shape,
        device=torch.device("cpu"),
        num_steps=10,
        noise=noise,
    )
    torch.testing.assert_close(result, torch.zeros_like(result), atol=1e-6, rtol=0)
