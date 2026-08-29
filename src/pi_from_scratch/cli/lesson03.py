"""Runnable probes for lesson 3: representation and normalization."""

import torch
from torch import Tensor

from pi_from_scratch.representations import (
    ActionNormalizer,
    CurrentStateDeltaTransform,
    FiniteDifferenceVelocityTransform,
    RunningActionStats,
)


def _row(name: str, values: Tensor) -> None:
    rounded = [[round(float(value), 3) for value in step] for step in values.tolist()]
    print(f"{name:<27} {rounded}")


def _fit(values: Tensor, episode_ids: tuple[int, ...]) -> ActionNormalizer:
    accumulator = RunningActionStats()
    accumulator.update(values)
    return ActionNormalizer(accumulator.finalize(train_episode_ids=episode_ids))


def main() -> None:
    current_state = torch.tensor([10.0, 20.0])
    absolute_targets = torch.tensor([[11.0, 22.0], [13.0, 25.0], [16.0, 29.0]])
    delta_transform = CurrentStateDeltaTransform()
    velocity_transform = FiniteDifferenceVelocityTransform(fps=10.0)

    delta_targets = delta_transform.forward(absolute_targets, current_state)
    velocity_targets = velocity_transform.forward(absolute_targets, current_state)

    print("representation: the same motion, three different numerical targets")
    _row("absolute position", absolute_targets)
    _row("delta from current state", delta_targets)
    _row("finite-difference / second", velocity_targets)
    print("delta round-trip max error:   ", end="")
    print(f"{(delta_transform.inverse(delta_targets, current_state) - absolute_targets).abs().max():.2e}")
    print("velocity round-trip max error:", end=" ")
    print(
        f"{(velocity_transform.inverse(velocity_targets, current_state) - absolute_targets).abs().max():.2e}"
    )

    train_actions = torch.tensor(
        [[0.0, 10.0], [2.0, 12.0], [4.0, 14.0], [6.0, 16.0]]
    )
    validation_actions = torch.tensor([[100.0, 200.0], [102.0, 202.0]])
    train_normalizer = _fit(train_actions, episode_ids=(0, 1))
    leaky_normalizer = _fit(
        torch.cat((train_actions, validation_actions)), episode_ids=(0, 1, 9)
    )

    print("\nnormalization: validation must not influence the coordinate system used for training")
    _row("train mean", train_normalizer.stats.mean[None])
    _row("train std", train_normalizer.stats.std[None])
    _row("leaky train+val mean", leaky_normalizer.stats.mean[None])
    _row("leaky train+val std", leaky_normalizer.stats.std[None])
    normalized_train = train_normalizer.normalize_values(train_actions)
    restored_train = train_normalizer.denormalize_values(normalized_train)
    _row("normalized train actions", normalized_train)
    print(f"artifact id:                  {train_normalizer.stats.artifact_id}")
    print(
        "normalization round-trip max error: "
        f"{(restored_train - train_actions).abs().max():.2e}"
    )
    print("\nOnly episode ids (0, 1) belong in this artifact; episode 9 is validation.")


if __name__ == "__main__":
    main()
