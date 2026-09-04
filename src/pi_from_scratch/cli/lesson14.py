"""Run the controlled RECAP policy-extraction experiment."""

import argparse
from pathlib import Path

from pi_from_scratch.evaluation.recap import (
    run_recap_experiment,
    write_recap_comparison_svg,
    write_recap_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the RECAP mechanism experiment")
    parser.add_argument("--samples", type=int, default=96)
    parser.add_argument("--train-steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=14)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/lesson14"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = run_recap_experiment(
        samples=args.samples,
        train_steps=args.train_steps,
        seed=args.seed,
    )
    metrics_path = args.output_dir / "metrics.json"
    figure_path = args.output_dir / "comparison.svg"
    write_recap_metrics(metrics_path, metrics)
    write_recap_comparison_svg(figure_path, metrics)
    errors = metrics["high_quality_action_mae"]
    print("RECAP controlled policy-extraction check")
    print(f"  mixed behavior cloning MAE: {errors['mixed_behavior_cloning']:.4f}")
    print(f"  advantage-conditioned MAE: {errors['advantage_conditioned']:.4f}")
    print(f"  metrics: {metrics_path}")
    print(f"  figure:  {figure_path}")


if __name__ == "__main__":
    main()
