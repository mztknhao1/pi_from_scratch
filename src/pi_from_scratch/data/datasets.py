from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import Dataset, Subset

from pi_from_scratch.config import DataConfig, ModelConfig
from pi_from_scratch.data.windows import EpisodeSplit, split_episode_ids
from pi_from_scratch.representations import HashTokenizer


@dataclass(frozen=True)
class DatasetSplits:
    """Episode-disjoint datasets plus the ids used to create them."""

    train: Dataset[dict[str, Tensor]]
    validation: Dataset[dict[str, Tensor]]
    episode_ids: EpisodeSplit


class SyntheticPiDataset(Dataset[dict[str, Tensor]]):
    """Deterministic fake samples for testing the complete training path."""

    def __init__(self, model: ModelConfig, length: int = 1024):
        if model.state_dim < model.action_dim:
            raise ValueError("synthetic data requires state_dim >= action_dim")
        self.model = model
        self.length = length
        self.tokenizer = HashTokenizer(model.vocab_size, model.max_text_tokens)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        generator = torch.Generator().manual_seed(index)
        image = torch.rand(3, self.model.image_size, self.model.image_size, generator=generator)
        state = torch.rand(self.model.state_dim, generator=generator) * 2.0 - 1.0
        phase = torch.linspace(0.0, 1.0, self.model.action_horizon)
        base = state[: self.model.action_dim]
        actions = base[None].repeat(self.model.action_horizon, 1)
        actions = actions + 0.1 * phase[:, None]
        text_ids, text_mask = self.tokenizer.encode("move toward the target")
        return {
            "image": image,
            "state": state,
            "text_ids": text_ids,
            "text_mask": text_mask,
            "actions": actions,
            "action_mask": torch.ones(self.model.action_horizon, dtype=torch.bool),
        }


class LeRobotPiDataset(Dataset[dict[str, Tensor]]):
    """Thin adapter from LeRobot's schema to this project's training tensors."""

    def __init__(
        self,
        data: DataConfig,
        model: ModelConfig,
        *,
        episodes: tuple[int, ...] | None = None,
    ):
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
        except ImportError as exc:
            raise ImportError(
                "LeRobot dataset dependencies are missing. Run: pip install -e '.[lerobot]'"
            ) from exc

        self.data = data
        self.model = model
        self.tokenizer = HashTokenizer(model.vocab_size, model.max_text_tokens)
        repo_id = data.dataset
        metadata = LeRobotDatasetMetadata(repo_id, revision=data.dataset_revision)
        self.dataset = LeRobotDataset(
            repo_id,
            episodes=None if episodes is None else list(episodes),
            delta_timestamps={
                data.action_key: [step / metadata.fps for step in range(model.action_horizon)]
            },
            revision=data.dataset_revision,
            video_backend=data.video_backend,
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        raw: Mapping[str, Any] = self.dataset[index]
        image = torch.as_tensor(raw[self.data.image_key])
        if image.ndim == 3 and image.shape[-1] == 3:
            image = image.permute(2, 0, 1)
        state = torch.as_tensor(raw[self.data.state_key], dtype=torch.float32)
        actions = torch.as_tensor(raw[self.data.action_key], dtype=torch.float32)
        if actions.ndim == 1:
            actions = actions[None]
        padding_key = f"{self.data.action_key}_is_pad"
        action_padding = torch.as_tensor(
            raw.get(padding_key, torch.zeros(actions.shape[0], dtype=torch.bool)),
            dtype=torch.bool,
        )
        if action_padding.shape != actions.shape[:1]:
            raise ValueError(f"{padding_key} must have shape [horizon]")
        if action_padding[0].item() or torch.any(action_padding[:-1] & ~action_padding[1:]).item():
            raise ValueError(f"{padding_key} must describe a non-empty valid prefix")
        prompt = str(raw.get("task", self.data.default_prompt))
        text_ids, text_mask = self.tokenizer.encode(prompt)
        return {
            "image": image,
            "state": state,
            "text_ids": text_ids,
            "text_mask": text_mask,
            "actions": actions,
            "action_mask": ~action_padding,
        }


def create_dataset(data: DataConfig, model: ModelConfig) -> Dataset[dict[str, Tensor]]:
    if data.dataset == "synthetic":
        return SyntheticPiDataset(model)
    return LeRobotPiDataset(data, model)


def create_dataset_splits(data: DataConfig, model: ModelConfig, *, seed: int) -> DatasetSplits:
    """Create train/validation datasets by splitting complete episodes."""
    if data.dataset == "synthetic":
        if data.synthetic_num_episodes < 2 or data.synthetic_episode_length < 1:
            raise ValueError("synthetic data requires at least two non-empty episodes")
        split = split_episode_ids(
            tuple(range(data.synthetic_num_episodes)),
            validation_fraction=data.validation_fraction,
            seed=seed,
        )
        dataset = SyntheticPiDataset(
            model,
            length=data.synthetic_num_episodes * data.synthetic_episode_length,
        )

        def indices_for(selected: tuple[int, ...]) -> list[int]:
            return [
                episode_id * data.synthetic_episode_length + frame
                for episode_id in selected
                for frame in range(data.synthetic_episode_length)
            ]

        return DatasetSplits(
            train=Subset(dataset, indices_for(split.train)),
            validation=Subset(dataset, indices_for(split.validation)),
            episode_ids=split,
        )

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    except ImportError as exc:
        raise ImportError(
            "LeRobot dataset dependencies are missing. Run: pip install -e '.[lerobot]'"
        ) from exc

    metadata = LeRobotDatasetMetadata(data.dataset, revision=data.dataset_revision)
    split = split_episode_ids(
        tuple(range(metadata.total_episodes)),
        validation_fraction=data.validation_fraction,
        seed=seed,
    )
    return DatasetSplits(
        train=LeRobotPiDataset(data, model, episodes=split.train),
        validation=LeRobotPiDataset(data, model, episodes=split.validation),
        episode_ids=split,
    )
