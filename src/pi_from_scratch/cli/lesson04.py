"""Runnable probe for lesson 4: predict H actions and execute E actions."""

import argparse

import torch

from pi_from_scratch.contracts import ActionChunk, ActionRepresentation, ActionSpec
from pi_from_scratch.runtime import boundary_action_jumps, chunk_timing, stitch_chunk_prefixes


def _make_chunk(
    *, chunk_index: int, horizon: int, execution_horizon: int, fps: float, offset: float
) -> ActionChunk:
    start_step = chunk_index * execution_horizon
    global_steps = start_step + torch.arange(horizon, dtype=torch.float32)
    values = torch.stack((global_steps, 0.5 * global_steps), dim=-1)
    if chunk_index > 0:
        direction = 1.0 if chunk_index % 2 else -1.0
        values[:, 1] += direction * offset
    timestamps_s = global_steps / fps
    spec = ActionSpec(
        dim=2,
        space="toy_planar_position",
        representation=ActionRepresentation.ABSOLUTE,
        frame="world",
        units=("m", "m"),
        minimum=(-1000.0, -1000.0),
        maximum=(1000.0, 1000.0),
    )
    return ActionChunk(
        values=values[None],
        valid_mask=torch.ones(1, horizon, dtype=torch.bool),
        timestamps_s=timestamps_s[None],
        spec=spec,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect action-chunk execution timing")
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--execution-horizon", type=int, default=2)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--replans", type=int, default=3)
    parser.add_argument("--boundary-offset", type=float, default=0.4)
    parser.add_argument(
        "--inference-latency-ms",
        type=float,
        default=0.0,
        help="Blocking inference latency used only for the wall-clock timing comparison",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 1 <= args.execution_horizon <= args.horizon:
        raise ValueError("execution_horizon must satisfy 1 <= E <= H")
    if args.replans < 1:
        raise ValueError("replans must be positive")
    if args.inference_latency_ms < 0:
        raise ValueError("inference_latency_ms must be non-negative")

    prediction = chunk_timing(args.horizon, args.fps)
    execution = chunk_timing(args.execution_horizon, args.fps)
    inference_latency_s = args.inference_latency_ms / 1000.0
    blocking_cycle_s = inference_latency_s + execution.coverage_duration_s
    chunks = [
        _make_chunk(
            chunk_index=index,
            horizon=args.horizon,
            execution_horizon=args.execution_horizon,
            fps=args.fps,
            offset=args.boundary_offset,
        )
        for index in range(args.replans)
    ]
    trace = stitch_chunk_prefixes(chunks, execution_horizon=args.execution_horizon)
    jumps = boundary_action_jumps(trace)

    print("chunk timing")
    print(f"  prediction horizon H:       {args.horizon} actions")
    print(f"  timestamp span:             {prediction.timestamp_span_s:.3f} s")
    print(f"  prediction coverage:        {prediction.coverage_duration_s:.3f} s")
    print(f"  execution horizon E:        {args.execution_horizon} actions")
    print(f"  time executed per replan:   {execution.coverage_duration_s:.3f} s")
    print(f"  nominal replanning rate:    {args.fps / args.execution_horizon:>8.3f} Hz")
    print(f"  blocking inference latency: {inference_latency_s:>8.3f} s")
    print(f"  blocking wall-clock period: {blocking_cycle_s:>8.3f} s")
    print(f"  blocking replanning rate:   {1.0 / blocking_cycle_s:>8.3f} Hz")
    print(f"  discarded tail per chunk:   {args.horizon - args.execution_horizon} actions")

    print("\nexecuted trace")
    print("step | time  | source chunk | boundary | action")
    print("-----+-------+--------------+----------+----------------")
    boundary_set = set(trace.chunk_boundary_steps)
    for step, (timestamp, chunk_index, action) in enumerate(
        zip(trace.timestamps_s, trace.source_chunk_indices, trace.values, strict=True)
    ):
        action_text = "[" + ", ".join(f"{value:.2f}" for value in action.tolist()) + "]"
        print(
            f"{step:>4} | {timestamp.item():>4.1f}s | {chunk_index.item():>12} | "
            f"{('new chunk' if step in boundary_set else ''):>8} | {action_text}"
        )

    print("\nboundary action jumps")
    if jumps.numel():
        print("  " + ", ".join(f"{value:.3f}" for value in jumps.tolist()))
        print(f"  mean: {jumps.mean().item():.3f}")
    else:
        print("  no boundary: only one chunk was executed")
    print("\nThe jump is measured in physical action space at the commands the controller receives.")


if __name__ == "__main__":
    main()
