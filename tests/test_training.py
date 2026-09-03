import json

import pytest
import torch
from torch.utils.data import DataLoader

from pi_from_scratch.config import DataConfig, ModelConfig, TrainConfig
from pi_from_scratch.data import SyntheticPiDataset, create_dataset_splits
from pi_from_scratch.models import TinyPi0
from pi_from_scratch.objectives import FLOW_TIME_CONVENTION
from pi_from_scratch.training import evaluate_flow_loss, load_tiny_checkpoint, train_experiment


def tiny_training_config(output_dir: str, *, steps: int = 60) -> TrainConfig:
    return TrainConfig(
        model=ModelConfig(
            image_size=32,
            action_horizon=4,
            width=32,
            num_layers=1,
            num_heads=4,
        ),
        data=DataConfig(
            validation_fraction=0.25,
            synthetic_num_episodes=4,
            synthetic_episode_length=3,
        ),
        batch_size=6,
        learning_rate=3e-3,
        weight_decay=0.0,
        steps=steps,
        log_every=steps,
        eval_every=20,
        save_every=steps,
        max_eval_batches=2,
        sampling_steps=4,
        overfit_samples=6,
        seed=7,
        output_dir=output_dir,
    )


def test_synthetic_split_keeps_complete_episodes_disjoint() -> None:
    config = tiny_training_config("unused")
    splits = create_dataset_splits(config.data, config.model, seed=config.seed)

    assert set(splits.episode_ids.train).isdisjoint(splits.episode_ids.validation)
    assert len(splits.train) == 9
    assert len(splits.validation) == 3


def test_fixed_flow_evaluation_repeats_exactly() -> None:
    config = tiny_training_config("unused")
    loader = DataLoader(SyntheticPiDataset(config.model, length=4), batch_size=2)
    model = TinyPi0(config.model)

    first = evaluate_flow_loss(model, loader, device=torch.device("cpu"), seed=17, max_batches=2)
    second = evaluate_flow_loss(model, loader, device=torch.device("cpu"), seed=17, max_batches=2)

    assert first == second


def test_fixed_bank_overfit_saves_reproducible_artifacts(tmp_path) -> None:
    config = tiny_training_config(str(tmp_path))
    result = train_experiment(config, "cpu", progress=False)

    assert result.final_train_loss < result.initial_train_loss * 0.1
    assert torch.isfinite(torch.tensor(result.final_validation_loss))
    assert result.validation_action_mae >= 0.0
    assert result.checkpoint_path.exists()
    assert result.metrics_path.exists()
    assert result.loss_curve_path.exists()
    assert result.trajectory_path.exists()
    assert "validation" in result.loss_curve_path.read_text()
    assert "target" in result.trajectory_path.read_text()
    assert "prediction" in result.trajectory_path.read_text()

    split = json.loads((tmp_path / "split.json").read_text())
    normalization = json.loads((tmp_path / "normalization.json").read_text())
    assert normalization["train_episode_ids"] == split["train"]
    assert set(split["train"]).isdisjoint(split["validation"])

    checkpoint = torch.load(result.checkpoint_path, weights_only=True)
    assert checkpoint["step"] == config.steps
    assert checkpoint["flow_time_convention"] == FLOW_TIME_CONVENTION
    assert checkpoint["normalization"]["train_episode_ids"] == tuple(split["train"])
    assert checkpoint["metrics"][0]["step"] == 0
    assert checkpoint["metrics"][-1]["step"] == config.steps

    loaded = load_tiny_checkpoint(result.checkpoint_path, device=torch.device("cpu"))
    assert loaded.step == config.steps
    assert loaded.normalizer.stats.artifact_id == normalization["artifact_id"]
    assert loaded.splits.episode_ids.train == tuple(split["train"])


def test_checkpoint_rejects_an_incompatible_flow_time_convention(tmp_path) -> None:
    config = tiny_training_config(str(tmp_path), steps=1)
    result = train_experiment(config, "cpu", progress=False)
    checkpoint = torch.load(result.checkpoint_path, weights_only=True)
    checkpoint["flow_time_convention"] = "openpi_t_noise_1_action_0"
    incompatible_path = tmp_path / "incompatible.pt"
    torch.save(checkpoint, incompatible_path)

    with pytest.raises(ValueError, match="flow-time convention"):
        load_tiny_checkpoint(incompatible_path, device=torch.device("cpu"))
