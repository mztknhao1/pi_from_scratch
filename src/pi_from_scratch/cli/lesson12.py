"""Run the π0.5 heterogeneous-objective mechanism check."""

import argparse
from pathlib import Path

from pi_from_scratch.evaluation.pi05 import (
    run_pi05_mixture_experiment,
    write_pi05_metrics,
    write_pi05_routing_svg,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the π0.5 mixture and routing demo")
    parser.add_argument("--mixture-steps", type=int, default=2000)
    parser.add_argument("--robot-probability", type=float, default=0.35)
    parser.add_argument("--finetune-steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/lesson12"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = run_pi05_mixture_experiment(
        mixture_steps=args.mixture_steps,
        robot_probability=args.robot_probability,
        finetune_steps=args.finetune_steps,
        seed=args.seed,
    )
    metrics_path = args.output_dir / "metrics.json"
    figure_path = args.output_dir / "routing.svg"
    write_pi05_metrics(metrics_path, metrics)
    write_pi05_routing_svg(figure_path, metrics)
    gradients = metrics["backbone_gradient_norm"]
    drift = metrics["flow_only_finetune"]
    print("π0.5 heterogeneous objective routing")
    print(f"  naive flow -> backbone grad: {gradients['continuous_flow_without_insulation']:.6f}")
    print(f"  insulated flow -> backbone:  {gradients['continuous_flow_with_insulation']:.6f}")
    print(
        f"  naive backbone drift:        {drift['without_insulation']['backbone_parameter_drift']:.6f}"
    )
    print(
        f"  insulated backbone drift:    {drift['with_insulation']['backbone_parameter_drift']:.6f}"
    )
    print(f"  metrics:                     {metrics_path}")
    print(f"  figure:                      {figure_path}")


if __name__ == "__main__":
    main()
