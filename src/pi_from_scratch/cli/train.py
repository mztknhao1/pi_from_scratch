"""Command-line entry point for the reusable tiny-policy trainer."""

import argparse
from dataclasses import replace
from pathlib import Path

import torch

from pi_from_scratch.config import DataConfig, ModelConfig, TrainConfig
from pi_from_scratch.training import TrainingResult, train_experiment


def train(config: TrainConfig, device_name: str) -> Path:
    """Backward-compatible wrapper returning the final checkpoint path."""
    return train_experiment(config, device_name).checkpoint_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the tiny π0 teaching model")
    parser.add_argument("--dataset", default="synthetic")
    parser.add_argument("--dataset-revision", default="v3.0")
    parser.add_argument("--steps", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--overfit-samples", type=int)
    parser.add_argument("--sampling-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default="outputs/debug")
    return parser.parse_args()


def _print_result(result: TrainingResult) -> None:
    print("training diagnostics")
    print(f"  fixed train flow loss:      {result.initial_train_loss:.6f} -> {result.final_train_loss:.6f}")
    print(
        "  fixed validation flow loss: "
        f"{result.initial_validation_loss:.6f} -> {result.final_validation_loss:.6f}"
    )
    print(f"  validation action MAE:      {result.validation_action_mae:.6f}")
    print(f"  checkpoint:                 {result.checkpoint_path}")
    print(f"  metrics:                    {result.metrics_path}")
    print(f"  loss curve:                 {result.loss_curve_path}")
    print(f"  trajectory:                 {result.trajectory_path}")


def main() -> None:
    args = parse_args()
    model = ModelConfig()
    if args.dataset == "lerobot/pusht":
        model = replace(model, state_dim=2, action_dim=2)
    config = TrainConfig(
        model=model,
        data=DataConfig(dataset=args.dataset, dataset_revision=args.dataset_revision),
        steps=args.steps,
        batch_size=args.batch_size,
        eval_every=args.eval_every,
        overfit_samples=args.overfit_samples,
        sampling_steps=args.sampling_steps,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    _print_result(train_experiment(config, args.device))


if __name__ == "__main__":
    main()
