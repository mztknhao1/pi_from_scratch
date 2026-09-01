"""One small, diagnosable training path used by lesson 7 and ``pi-train``."""

import json
import platform
import random
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import trange

from pi_from_scratch.config import TrainConfig
from pi_from_scratch.data import DatasetSplits, create_dataset_splits
from pi_from_scratch.evaluation import write_loss_curve_svg, write_trajectory_svg
from pi_from_scratch.models import TinyPi0
from pi_from_scratch.representations import ActionNormalizer, NormalizationStats, RunningActionStats

Batch = dict[str, Tensor]


@dataclass(frozen=True)
class MetricPoint:
    """Comparable metrics evaluated on fixed flow points."""

    step: int
    train_flow_loss: float
    validation_flow_loss: float
    optimization_loss: float | None
    gradient_norm: float | None


@dataclass(frozen=True)
class TrainingResult:
    """Paths and headline metrics produced by one completed run."""

    checkpoint_path: Path
    metrics_path: Path
    loss_curve_path: Path
    trajectory_path: Path
    initial_train_loss: float
    final_train_loss: float
    initial_validation_loss: float
    final_validation_loss: float
    validation_action_mae: float


class NormalizedActionDataset(Dataset[Batch]):
    """Apply one train-fitted action normalizer without changing the source dataset."""

    def __init__(self, dataset: Dataset[Batch], normalizer: ActionNormalizer):
        self.dataset = dataset
        self.normalizer = normalizer

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Batch:
        sample = dict(self.dataset[index])
        sample["actions"] = self.normalizer.normalize_values(sample["actions"].float())
        return sample


def move_to_device(batch: Batch, device: torch.device) -> Batch:
    return {key: value.to(device) for key, value in batch.items()}


def fit_action_normalizer(
    dataset: Dataset[Batch], *, train_episode_ids: tuple[int, ...]
) -> ActionNormalizer:
    """Fit per-dimension statistics using valid actions from training episodes only."""
    accumulator = RunningActionStats()
    for sample in dataset:
        accumulator.update(sample["actions"].float(), sample["action_mask"])
    return ActionNormalizer(accumulator.finalize(train_episode_ids=train_episode_ids))


def _fixed_flow_inputs(actions: Tensor, generator: torch.Generator) -> tuple[Tensor, Tensor]:
    noise = torch.randn(actions.shape, generator=generator, dtype=torch.float32)
    uniform = torch.rand((actions.shape[0],), generator=generator, dtype=torch.float32)
    time = uniform.pow(1.0 / 1.5) * 0.999 + 0.001
    return noise.to(device=actions.device, dtype=actions.dtype), time.to(
        device=actions.device,
        dtype=actions.dtype,
    )


@torch.no_grad()
def evaluate_flow_loss(
    model: TinyPi0,
    loader: DataLoader[Batch],
    *,
    device: torch.device,
    seed: int,
    max_batches: int,
) -> float:
    """Evaluate the same noise/time bank on every call."""
    if max_batches < 1:
        raise ValueError("max_batches must be positive")
    was_training = model.training
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    weighted_loss = 0.0
    valid_steps = 0
    for batch_index, cpu_batch in enumerate(loader):
        if batch_index >= max_batches:
            break
        batch = move_to_device(cpu_batch, device)
        noise, time = _fixed_flow_inputs(batch["actions"], generator)
        loss = model.loss(batch, noise=noise, time=time)
        weight = int(batch["action_mask"].sum().item())
        weighted_loss += loss.item() * weight
        valid_steps += weight
    model.train(was_training)
    if valid_steps == 0:
        raise ValueError("evaluation loader produced no valid action steps")
    return weighted_loss / valid_steps


def _normalization_json(stats: NormalizationStats) -> dict[str, Any]:
    return {
        "artifact_id": stats.artifact_id,
        "mean": stats.mean.tolist(),
        "std": stats.std.tolist(),
        "count": stats.count,
        "train_episode_ids": list(stats.train_episode_ids),
    }


def _git_commit(repo_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _save_checkpoint(
    path: Path,
    *,
    model: TinyPi0,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: TrainConfig,
    splits: DatasetSplits,
    normalizer: ActionNormalizer,
    metrics: list[MetricPoint],
) -> None:
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": asdict(config),
            "split": asdict(splits.episode_ids),
            "normalization": {
                "mean": normalizer.stats.mean,
                "std": normalizer.stats.std,
                "count": normalizer.stats.count,
                "train_episode_ids": normalizer.stats.train_episode_ids,
                "artifact_id": normalizer.stats.artifact_id,
            },
            "metrics": [asdict(point) for point in metrics],
        },
        path,
    )


def _validate_config(config: TrainConfig) -> None:
    if config.steps < 1 or config.batch_size < 1:
        raise ValueError("steps and batch_size must be positive")
    if config.log_every < 1 or config.eval_every < 1 or config.save_every < 1:
        raise ValueError("log_every, eval_every, and save_every must be positive")
    if config.sampling_steps < 1 or config.max_eval_batches < 1:
        raise ValueError("sampling_steps and max_eval_batches must be positive")
    if config.overfit_samples is not None and config.overfit_samples < 1:
        raise ValueError("overfit_samples must be positive when provided")


def _make_loaders(
    config: TrainConfig,
    splits: DatasetSplits,
    normalizer: ActionNormalizer,
) -> tuple[DataLoader[Batch], DataLoader[Batch], DataLoader[Batch]]:
    train_dataset: Dataset[Batch] = NormalizedActionDataset(splits.train, normalizer)
    validation_dataset: Dataset[Batch] = NormalizedActionDataset(splits.validation, normalizer)
    if config.overfit_samples is not None:
        if config.overfit_samples > len(train_dataset):
            raise ValueError("overfit_samples exceeds the number of training samples")
        train_dataset = Subset(train_dataset, range(config.overfit_samples))

    generator = torch.Generator().manual_seed(config.seed)
    train_batch_size = len(train_dataset) if config.overfit_samples is not None else config.batch_size
    train_loader = DataLoader(
        train_dataset,
        batch_size=min(train_batch_size, len(train_dataset)),
        shuffle=config.overfit_samples is None,
        generator=generator,
        num_workers=config.data.num_workers,
    )
    train_eval_loader = DataLoader(
        train_dataset,
        batch_size=min(config.batch_size, len(train_dataset)),
        shuffle=False,
        num_workers=config.data.num_workers,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=min(config.batch_size, len(validation_dataset)),
        shuffle=False,
        num_workers=config.data.num_workers,
    )
    return train_loader, train_eval_loader, validation_loader


def _next_batch(iterator: Any, loader: DataLoader[Batch]) -> tuple[Batch, Any]:
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def train_experiment(
    config: TrainConfig,
    device_name: str,
    *,
    progress: bool = True,
) -> TrainingResult:
    """Run one reproducible tiny-policy experiment and save its provenance."""
    _validate_config(config)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device(device_name)

    splits = create_dataset_splits(config.data, config.model, seed=config.seed)
    normalizer = fit_action_normalizer(
        splits.train,
        train_episode_ids=splits.episode_ids.train,
    )
    train_loader, train_eval_loader, validation_loader = _make_loaders(
        config,
        splits,
        normalizer,
    )
    model = TinyPi0(config.model).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    overfit_batch: Batch | None = None
    overfit_noise: Tensor | None = None
    overfit_time: Tensor | None = None
    if config.overfit_samples is not None:
        overfit_batch = move_to_device(next(iter(train_loader)), device)
        overfit_generator = torch.Generator().manual_seed(config.seed + 101)
        overfit_noise, overfit_time = _fixed_flow_inputs(
            overfit_batch["actions"],
            overfit_generator,
        )

    def evaluate(step: int, optimization_loss: float | None, gradient_norm: float | None) -> None:
        metrics.append(
            MetricPoint(
                step=step,
                train_flow_loss=evaluate_flow_loss(
                    model,
                    train_eval_loader,
                    device=device,
                    seed=config.seed + 101,
                    max_batches=config.max_eval_batches,
                ),
                validation_flow_loss=evaluate_flow_loss(
                    model,
                    validation_loader,
                    device=device,
                    seed=config.seed + 202,
                    max_batches=config.max_eval_batches,
                ),
                optimization_loss=optimization_loss,
                gradient_norm=gradient_norm,
            )
        )

    metrics: list[MetricPoint] = []
    evaluate(0, None, None)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    iterator = iter(train_loader)
    last_loss = 0.0
    last_gradient_norm = 0.0

    model.train()
    progress_bar = trange(1, config.steps + 1, desc="training", disable=not progress)
    for step in progress_bar:
        if overfit_batch is not None:
            batch = overfit_batch
            loss = model.loss(batch, noise=overfit_noise, time=overfit_time)
        else:
            cpu_batch, iterator = _next_batch(iterator, train_loader)
            batch = move_to_device(cpu_batch, device)
            loss = model.loss(batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        last_loss = loss.item()
        last_gradient_norm = float(gradient_norm)

        if step == 1 or step % config.log_every == 0:
            progress_bar.set_postfix(loss=f"{last_loss:.4f}")
        if step % config.eval_every == 0 or step == config.steps:
            evaluate(step, last_loss, last_gradient_norm)
        if step % config.save_every == 0 and step != config.steps:
            _save_checkpoint(
                output_dir / f"checkpoint_{step:06d}.pt",
                model=model,
                optimizer=optimizer,
                step=step,
                config=config,
                splits=splits,
                normalizer=normalizer,
                metrics=metrics,
            )

    validation_batch = move_to_device(next(iter(validation_loader)), device)
    sample_generator = torch.Generator().manual_seed(config.seed + 303)
    sample_noise = torch.randn(
        validation_batch["actions"].shape,
        generator=sample_generator,
        dtype=torch.float32,
    ).to(device)
    model.eval()
    predicted_normalized = model.sample_actions(
        validation_batch,
        num_steps=config.sampling_steps,
        noise=sample_noise,
    )
    target_raw = normalizer.denormalize_values(validation_batch["actions"])
    prediction_raw = normalizer.denormalize_values(predicted_normalized)
    action_weights = validation_batch["action_mask"].unsqueeze(-1).to(prediction_raw.dtype)
    validation_action_mae = (
        (prediction_raw - target_raw).abs().mul(action_weights).sum()
        / action_weights.sum().clamp_min(1.0)
        / prediction_raw.shape[-1]
    ).item()

    trajectory_path = output_dir / "validation_trajectory.svg"
    first_valid_horizon = int(validation_batch["action_mask"][0].sum().item())
    write_trajectory_svg(
        target_raw[0, :first_valid_horizon],
        prediction_raw[0, :first_valid_horizon],
        trajectory_path,
    )
    loss_curve_path = output_dir / "loss_curve.svg"
    write_loss_curve_svg(
        [point.step for point in metrics],
        [point.train_flow_loss for point in metrics],
        [point.validation_flow_loss for point in metrics],
        loss_curve_path,
    )
    checkpoint_path = output_dir / f"checkpoint_{config.steps:06d}.pt"
    _save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        step=config.steps,
        config=config,
        splits=splits,
        normalizer=normalizer,
        metrics=metrics,
    )

    repo_root = Path(__file__).resolve().parents[3]
    _write_json(output_dir / "resolved_config.json", asdict(config))
    _write_json(
        output_dir / "run_metadata.json",
        {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "git_commit": _git_commit(repo_root),
            "python": sys.version,
            "torch": torch.__version__,
            "platform": platform.platform(),
            "device": str(device),
            "seed": config.seed,
        },
    )
    _write_json(output_dir / "split.json", asdict(splits.episode_ids))
    _write_json(output_dir / "normalization.json", _normalization_json(normalizer.stats))
    metrics_path = output_dir / "metrics.json"
    _write_json(
        metrics_path,
        {
            "flow": [asdict(point) for point in metrics],
            "validation_action_mae": validation_action_mae,
        },
    )

    return TrainingResult(
        checkpoint_path=checkpoint_path,
        metrics_path=metrics_path,
        loss_curve_path=loss_curve_path,
        trajectory_path=trajectory_path,
        initial_train_loss=metrics[0].train_flow_loss,
        final_train_loss=metrics[-1].train_flow_loss,
        initial_validation_loss=metrics[0].validation_flow_loss,
        final_validation_loss=metrics[-1].validation_flow_loss,
        validation_action_mae=validation_action_mae,
    )
