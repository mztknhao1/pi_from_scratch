"""Runnable V2 overfit benchmark for lesson 7."""

import argparse

import torch

from pi_from_scratch.config import DataConfig, ModelConfig, TrainConfig
from pi_from_scratch.training import train_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overfit a TinyPi0 flow bank and save diagnostics")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="outputs/lesson07")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    config = TrainConfig(
        model=ModelConfig(
            image_size=32,
            action_horizon=4,
            width=32,
            num_layers=1,
            num_heads=4,
        ),
        data=DataConfig(
            dataset="synthetic",
            validation_fraction=0.2,
            synthetic_num_episodes=5,
            synthetic_episode_length=4,
        ),
        batch_size=8,
        learning_rate=3e-3,
        weight_decay=0.0,
        steps=args.steps,
        log_every=25,
        eval_every=50,
        save_every=args.steps,
        max_eval_batches=4,
        sampling_steps=20,
        overfit_samples=8,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    result = train_experiment(config, args.device)
    print("\nTinyPi0 fixed-bank overfit")
    print(f"  train flow loss:      {result.initial_train_loss:.6f} -> {result.final_train_loss:.6f}")
    print(
        "  validation flow loss: "
        f"{result.initial_validation_loss:.6f} -> {result.final_validation_loss:.6f}"
    )
    print(f"  validation action MAE: {result.validation_action_mae:.6f}")
    print(f"  checkpoint:            {result.checkpoint_path}")
    print(f"  metrics:               {result.metrics_path}")
    print(f"  loss curve:            {result.loss_curve_path}")
    print(f"  trajectory:            {result.trajectory_path}")


if __name__ == "__main__":
    main()
