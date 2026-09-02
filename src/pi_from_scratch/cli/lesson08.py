"""Solver, step-count, error, and latency sweeps for lesson 8."""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import torch
from torch import Tensor

from pi_from_scratch.evaluation import (
    SamplingSweepPoint,
    run_sampling_sweep,
    sampling_points_to_json,
    write_sampling_sweep_svg,
    write_trajectory_svg,
)
from pi_from_scratch.inference import FlowSolver, flow_sample
from pi_from_scratch.training import load_tiny_checkpoint


def parse_steps(value: str) -> tuple[int, ...]:
    try:
        steps = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("steps must be comma-separated integers") from exc
    if not steps or any(step < 1 for step in steps) or len(set(steps)) != len(steps):
        raise argparse.ArgumentTypeError("steps must be unique positive integers")
    return tuple(sorted(steps))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep flow solvers with fixed initial noise")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--steps", type=parse_steps, default=parse_steps("1,2,4,8,16,32"))
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/lesson08"))
    return parser.parse_args()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _print_points(title: str, points: Sequence[SamplingSweepPoint]) -> None:
    print(f"\n{title}")
    print("solver | steps | NFE | action MAE | median latency ms")
    print("-------+-------+-----+------------+------------------")
    for point in points:
        print(
            f"{point.solver:>6} | {point.steps:>5} | {point.model_evaluations:>3} | "
            f"{point.action_mae:>10.6f} | {point.median_latency_ms:>17.4f}"
        )


def run_oracle_sweep(
    *,
    steps: tuple[int, ...],
    repeats: int,
    seed: int,
    output_dir: Path,
) -> None:
    generator = torch.Generator().manual_seed(seed)
    target = torch.tensor(
        [[[0.0, 0.0], [0.3, 0.5], [0.8, 0.7], [1.0, 1.0]]],
        dtype=torch.float32,
    )
    noise = torch.randn(target.shape, generator=generator)
    delta = noise - target
    valid_mask = torch.ones(target.shape[:2], dtype=torch.bool)

    def velocity_fn(_x_t: Tensor, time: Tensor) -> Tensor:
        return 2.0 * time[:, None, None] * delta

    def sample(solver: FlowSolver, num_steps: int) -> Tensor:
        return flow_sample(
            velocity_fn,
            target.shape,
            device=target.device,
            num_steps=num_steps,
            noise=noise,
            solver=solver,
        )

    points = run_sampling_sweep(
        sample,
        target,
        valid_mask,
        solvers=("euler", "heun"),
        steps=steps,
        warmup=2,
        repeats=repeats,
    )
    _write_json(output_dir / "oracle_sampling_metrics.json", sampling_points_to_json(points))
    write_sampling_sweep_svg(
        points,
        output_dir / "oracle_sampling_sweep.svg",
        title="Quadratic path: solver error and latency",
    )
    _print_points("quadratic-path analytic oracle", points)


def run_checkpoint_sweep(
    checkpoint: Path,
    *,
    steps: tuple[int, ...],
    repeats: int,
    seed: int,
    device: torch.device,
    output_dir: Path,
) -> None:
    loaded = load_tiny_checkpoint(checkpoint, device=device)
    raw_sample = dict(loaded.splits.validation[0])
    raw_target = raw_sample["actions"].float()[None].to(device)
    batch = {key: value[None].to(device) for key, value in raw_sample.items()}
    batch["actions"] = loaded.normalizer.normalize_values(raw_target)
    valid_mask = batch["action_mask"]
    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn(batch["actions"].shape, generator=generator).to(device)

    def sample(solver: FlowSolver, num_steps: int) -> Tensor:
        normalized = loaded.model.sample_actions(
            batch,
            num_steps=num_steps,
            noise=noise,
            solver=solver,
        )
        return loaded.normalizer.denormalize_values(normalized)

    points = run_sampling_sweep(
        sample,
        raw_target,
        valid_mask,
        solvers=("euler", "heun"),
        steps=steps,
        warmup=1,
        repeats=repeats,
    )
    _write_json(
        output_dir / "checkpoint_sampling_metrics.json",
        {
            "checkpoint": str(checkpoint),
            "checkpoint_step": loaded.step,
            "normalization_id": loaded.normalizer.stats.artifact_id,
            "noise_seed": seed,
            "points": sampling_points_to_json(points),
        },
    )
    write_sampling_sweep_svg(
        points,
        output_dir / "checkpoint_sampling_sweep.svg",
        title="TinyPi0 checkpoint: fixed-noise sampling sweep",
    )
    largest_step = max(steps)
    for solver in ("euler", "heun"):
        prediction = sample(solver, largest_step)
        valid_horizon = int(valid_mask[0].sum().item())
        write_trajectory_svg(
            raw_target[0, :valid_horizon],
            prediction[0, :valid_horizon],
            output_dir / f"checkpoint_{solver}_{largest_step}_trajectory.svg",
        )
    _print_points(f"TinyPi0 checkpoint {checkpoint}", points)


def main() -> None:
    args = parse_args()
    if args.repeats < 1:
        raise ValueError("repeats must be positive")
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_oracle_sweep(
        steps=args.steps,
        repeats=args.repeats,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    if args.checkpoint is not None:
        run_checkpoint_sweep(
            args.checkpoint,
            steps=args.steps,
            repeats=args.repeats,
            seed=args.seed,
            device=device,
            output_dir=args.output_dir,
        )
    else:
        print("\ncheckpoint sweep skipped; pass --checkpoint from lesson 7 to enable it")


if __name__ == "__main__":
    main()
