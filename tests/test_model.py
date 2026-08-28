import torch

from pi_from_scratch.config import ModelConfig
from pi_from_scratch.data import SyntheticPiDataset
from pi_from_scratch.models import TinyPi0


def test_model_loss_and_sampling_shapes() -> None:
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
    loss = model.loss(batch)
    loss.backward()
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    actions = model.sample_actions(batch, num_steps=2)
    assert actions.shape == (2, config.action_horizon, config.action_dim)
