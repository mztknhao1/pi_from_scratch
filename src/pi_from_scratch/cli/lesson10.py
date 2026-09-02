"""Compare blocking, unconstrained asynchronous execution, and RTC."""

import argparse
from pathlib import Path

from pi_from_scratch.evaluation import write_rtc_comparison_svg, write_rtc_metrics
from pi_from_scratch.runtime import simulate_latency_runtime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the deterministic RTC latency experiment")
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--execution-horizon", type=int, default=6)
    parser.add_argument("--delay-steps", type=int, default=3)
    parser.add_argument("--num-replans", type=int, default=5)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--denoising-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/lesson10"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    kwargs = {
        "horizon": args.horizon,
        "execution_horizon": args.execution_horizon,
        "delay_steps": args.delay_steps,
        "num_replans": args.num_replans,
        "fps": args.fps,
        "num_denoising_steps": args.denoising_steps,
        "seed": args.seed,
    }
    traces = [
        simulate_latency_runtime(method, **kwargs)
        for method in ("blocking", "naive_async", "rtc")
    ]
    metrics_path = args.output_dir / "metrics.json"
    plot_path = args.output_dir / "comparison.svg"
    write_rtc_metrics(metrics_path, traces, config=kwargs)
    write_rtc_comparison_svg(plot_path, traces)

    print("Real-Time Chunking comparison")
    for trace in traces:
        print(
            f"  {trace.method:11s} "
            f"throughput={trace.throughput_hz:5.2f}Hz "
            f"jump={trace.boundary_jumps.mean().item():.4f} "
            f"jerk={trace.boundary_jerks.mean().item():.2f} "
            f"misses={trace.deadline_misses}"
        )
    print(f"  metrics: {metrics_path}")
    print(f"  plot:    {plot_path}")


if __name__ == "__main__":
    main()
