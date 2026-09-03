"""Run the controlled multi-scale memory experiment."""

import argparse
from pathlib import Path

from pi_from_scratch.evaluation.memory import (
    run_memory_experiment,
    write_memory_comparison_svg,
    write_memory_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the MEM mechanism experiment")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/lesson13"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = run_memory_experiment(episodes=args.episodes, seed=args.seed)
    metrics_path = args.output_dir / "metrics.json"
    figure_path = args.output_dir / "comparison.svg"
    write_memory_metrics(metrics_path, metrics)
    write_memory_comparison_svg(figure_path, metrics)
    print("MEM controlled mechanism check")
    for name, result in metrics["methods"].items():
        print(
            f"  {name:12s} occlusion={result['occlusion_accuracy']:.3f} "
            f"long_horizon={result['long_horizon_progress']:.3f}"
        )
    print(f"  metrics: {metrics_path}")
    print(f"  figure:  {figure_path}")


if __name__ == "__main__":
    main()
