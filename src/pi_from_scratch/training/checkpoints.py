"""Load the complete teaching checkpoint needed for offline inference."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from pi_from_scratch.config import DataConfig, ModelConfig, TrainConfig
from pi_from_scratch.data import DatasetSplits, create_dataset_splits
from pi_from_scratch.models import TinyPi0
from pi_from_scratch.objectives import FLOW_TIME_CONVENTION
from pi_from_scratch.representations import ActionNormalizer, NormalizationStats


@dataclass(frozen=True)
class LoadedTinyCheckpoint:
    step: int
    config: TrainConfig
    model: TinyPi0
    normalizer: ActionNormalizer
    splits: DatasetSplits


def _train_config(value: dict[str, Any]) -> TrainConfig:
    fields = dict(value)
    fields["model"] = ModelConfig(**fields["model"])
    fields["data"] = DataConfig(**fields["data"])
    return TrainConfig(**fields)


def load_tiny_checkpoint(path: Path, *, device: torch.device) -> LoadedTinyCheckpoint:
    """Restore model, config, split, and normalization with consistency checks."""
    payload = torch.load(path, map_location=device, weights_only=True)
    required = {
        "flow_time_convention",
        "step",
        "model",
        "config",
        "split",
        "normalization",
    }
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"checkpoint is missing required keys: {sorted(missing)}")
    if payload["flow_time_convention"] != FLOW_TIME_CONVENTION:
        raise ValueError(
            "checkpoint flow-time convention is incompatible: expected "
            f"{FLOW_TIME_CONVENTION!r}, got {payload['flow_time_convention']!r}"
        )

    config = _train_config(payload["config"])
    split_payload = payload["split"]
    splits = create_dataset_splits(config.data, config.model, seed=config.seed)
    if tuple(split_payload["train"]) != splits.episode_ids.train or tuple(
        split_payload["validation"]
    ) != splits.episode_ids.validation:
        raise ValueError("checkpoint split does not match the resolved dataset split")

    normalization = payload["normalization"]
    stats = NormalizationStats(
        mean=normalization["mean"].float().cpu(),
        std=normalization["std"].float().cpu(),
        count=int(normalization["count"]),
        train_episode_ids=tuple(int(value) for value in normalization["train_episode_ids"]),
    )
    if normalization.get("artifact_id") != stats.artifact_id:
        raise ValueError("checkpoint normalization artifact id is inconsistent")

    model = TinyPi0(config.model).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return LoadedTinyCheckpoint(
        step=int(payload["step"]),
        config=config,
        model=model,
        normalizer=ActionNormalizer(stats),
        splits=splits,
    )
