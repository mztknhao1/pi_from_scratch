"""Run the first complete observation-policy-execution closed loop."""

import argparse
from pathlib import Path

from pi_from_scratch.envs import GymPushTEnv, PointReachEnv
from pi_from_scratch.evaluation import write_closed_loop_summary, write_point_trajectory_svg
from pi_from_scratch.policies import PointGoalPolicy, RandomPolicy
from pi_from_scratch.runtime import run_synchronous_episode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a synchronous action-chunk rollout")
    parser.add_argument("--env", choices=("point", "pusht"), default="point")
    parser.add_argument("--policy", choices=("goal", "random"), default="goal")
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--execution-horizon", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/lesson09"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.env == "point":
        env = PointReachEnv(max_steps=args.max_steps)
    else:
        env = GymPushTEnv(max_steps=args.max_steps)

    if args.policy == "goal":
        if args.env != "point":
            raise ValueError("the transparent goal policy only understands PointReach state")
        policy = PointGoalPolicy(
            env.action_spec,
            horizon=args.horizon,
            fps=env.fps,
            motion_per_step=env.max_motion_per_step,
        )
    else:
        policy = RandomPolicy(
            env.action_spec,
            horizon=args.horizon,
            fps=env.fps,
            seed=args.seed,
        )

    try:
        run = run_synchronous_episode(
            env,
            policy,
            execution_horizon=args.execution_horizon,
            max_steps=args.max_steps,
            seed=args.seed,
        )
    finally:
        env.close()

    summary_path = args.output_dir / "summary.json"
    write_closed_loop_summary(summary_path, run, seed=args.seed)
    print("Synchronous closed-loop rollout")
    print(f"  environment:            {args.env}")
    print(f"  policy:                 {args.policy}")
    print(f"  success:                {run.episode.success}")
    print(f"  control steps:          {run.episode.num_steps}")
    print(f"  replans:                {len(run.episode.inference_latencies_s)}")
    print(f"  refill deadline misses: {run.episode.deadline_misses}")
    print(f"  summary:                {summary_path}")
    if args.env == "point":
        trajectory_path = args.output_dir / "trajectory.svg"
        write_point_trajectory_svg(trajectory_path, run)
        print(f"  trajectory:             {trajectory_path}")


if __name__ == "__main__":
    main()
