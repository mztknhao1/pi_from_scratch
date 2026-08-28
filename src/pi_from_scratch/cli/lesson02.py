"""Runnable probes for lesson 2: episode splits and future action windows."""

import argparse
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

from pi_from_scratch.data.windows import (
    FutureActionWindow,
    build_future_action_window,
    split_episode_ids,
)


def _format_action(action: Tensor) -> str:
    return "[" + ", ".join(f"{value:.2f}" for value in action.tolist()) + "]"


def print_window(window: FutureActionWindow, *, title: str) -> None:
    print(f"\n{title}")
    print("slot | source frame | timestamp | valid | action")
    print("-----+--------------+-----------+-------+----------------")
    for slot in range(window.values.shape[0]):
        print(
            f"{slot:>4} | {window.source_indices[slot].item():>12} | "
            f"{window.timestamps_s[slot].item():>8.2f}s | "
            f"{window.valid_mask[slot].item()!s:>5} | "
            f"{_format_action(window.values[slot])}"
        )


def run_toy_probe(horizon: int, validation_fraction: float) -> None:
    episodes = {
        0: torch.tensor([[0.0, 0.0], [1.0, 0.5], [2.0, 1.0], [3.0, 1.5]]),
        1: torch.tensor([[100.0, 100.0], [101.0, 100.5], [102.0, 101.0]]),
    }
    split = split_episode_ids(range(10), validation_fraction=validation_fraction, seed=7)
    print("episode split")
    print(f"  train:      {split.train}")
    print(f"  validation: {split.validation}")
    print("  overlap:    0 episodes")

    first = build_future_action_window(episodes[0], anchor_index=0, horizon=horizon, fps=10.0)
    boundary = build_future_action_window(episodes[0], anchor_index=2, horizon=horizon, fps=10.0)
    batch_values = torch.stack((first.values, boundary.values))
    batch_mask = torch.stack((first.valid_mask, boundary.valid_mask))
    print("\nbatch")
    print(f"  actions shape: {tuple(batch_values.shape)}")
    print(f"  mask shape:    {tuple(batch_mask.shape)}")
    print_window(boundary, title="sample 1: an anchor near the end of episode 0")
    print("\nNotice: padded rows repeat episode 0's final action; episode 1 never leaks into the window.")


def _import_lerobot() -> tuple[type[Any], type[Any]]:
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    except ImportError as exc:
        raise ImportError(
            "LeRobot dataset dependencies are missing. Run: pip install -e '.[lerobot]'"
        ) from exc
    return LeRobotDataset, LeRobotDatasetMetadata


def _scalar(value: Any) -> int | float:
    tensor = torch.as_tensor(value)
    return tensor.item()


def run_lerobot_probe(
    repo_id: str, *, horizon: int, index: int, validation_fraction: float
) -> None:
    LeRobotDataset, LeRobotDatasetMetadata = _import_lerobot()
    metadata = LeRobotDatasetMetadata(repo_id)
    delta_timestamps = {"action": [step / metadata.fps for step in range(horizon)]}
    dataset = LeRobotDataset(repo_id, delta_timestamps=delta_timestamps, video_backend="pyav")
    resolved_index = index if index >= 0 else len(dataset) + index
    if not 0 <= resolved_index < len(dataset):
        raise IndexError(f"index {index} is outside a dataset with {len(dataset)} frames")

    raw: Mapping[str, Any] = dataset[resolved_index]
    actions = torch.as_tensor(raw["action"], dtype=torch.float32)
    if actions.ndim == 1:
        actions = actions[None]
    padding = torch.as_tensor(
        raw.get("action_is_pad", torch.zeros(actions.shape[0], dtype=torch.bool)),
        dtype=torch.bool,
    )
    base_timestamp = float(_scalar(raw["timestamp"]))
    timestamps = base_timestamp + torch.arange(actions.shape[0], dtype=torch.float32) / metadata.fps
    frame_index = int(_scalar(raw["frame_index"]))
    valid_count = int((~padding).sum().item())
    requested_indices = frame_index + torch.arange(actions.shape[0], dtype=torch.long)
    source_indices = requested_indices.clamp_max(frame_index + valid_count - 1)
    window = FutureActionWindow(
        values=actions,
        valid_mask=~padding,
        timestamps_s=timestamps,
        source_indices=source_indices,
    )
    split = split_episode_ids(
        range(metadata.total_episodes), validation_fraction=validation_fraction, seed=7
    )

    print(f"dataset:       {repo_id}")
    print(f"fps:           {metadata.fps}")
    print(f"episodes:      {metadata.total_episodes}")
    print(f"frames:        {len(dataset)}")
    print(f"train/val:     {len(split.train)}/{len(split.validation)} episodes")
    print(f"sample index:  {resolved_index}")
    print(f"episode/frame: {_scalar(raw['episode_index'])}/{frame_index}")
    print_window(window, title="future action window returned by LeRobot")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect episode-safe VLA training windows")
    parser.add_argument("--dataset", default="toy")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--index", type=int, default=-2)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dataset == "toy":
        run_toy_probe(args.horizon, args.validation_fraction)
    else:
        run_lerobot_probe(
            args.dataset,
            horizon=args.horizon,
            index=args.index,
            validation_fraction=args.validation_fraction,
        )


if __name__ == "__main__":
    main()
