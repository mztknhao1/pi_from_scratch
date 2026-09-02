"""Fit and inspect the teaching FAST-like action tokenizer."""

import argparse
from pathlib import Path

from pi_from_scratch.evaluation import (
    run_fast_experiment,
    write_fast_comparison_svg,
    write_fast_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the FAST-like compression experiment")
    parser.add_argument("--train-chunks", type=int, default=400)
    parser.add_argument("--validation-chunks", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--action-dim", type=int, default=7)
    parser.add_argument("--scale", type=float, default=10.0)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/lesson11"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics, original, reconstructed = run_fast_experiment(
        train_count=args.train_chunks,
        validation_count=args.validation_chunks,
        horizon=args.horizon,
        action_dim=args.action_dim,
        scale=args.scale,
        vocab_size=args.vocab_size,
        seed=args.seed,
    )
    metrics_path = args.output_dir / "metrics.json"
    figure_path = args.output_dir / "comparison.svg"
    write_fast_metrics(metrics_path, metrics)
    write_fast_comparison_svg(figure_path, metrics, original, reconstructed)

    scalar = metrics["scalar_quantization"]
    fast = metrics["fast_like_bpe"]
    print("FAST-like action tokenizer")
    print(f"  raw scalar tokens/chunk: {scalar['tokens_per_chunk']}")
    print(f"  BPE mean tokens/chunk:   {fast['mean_tokens_per_chunk']:.2f}")
    print(f"  compression ratio:       {fast['compression_ratio']:.2f}x")
    print(f"  reconstruction MAE:      {fast['validation_mae']:.5f}")
    print(f"  metrics:                 {metrics_path}")
    print(f"  figure:                  {figure_path}")


if __name__ == "__main__":
    main()
