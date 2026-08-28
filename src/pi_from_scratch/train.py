import argparse
import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import trange

from pi_from_scratch.config import DataConfig, ModelConfig, TrainConfig
from pi_from_scratch.data import create_dataset
from pi_from_scratch.model import TinyPi0


def move_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def train(config: TrainConfig, device_name: str) -> Path:
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device(device_name)

    dataset = create_dataset(config.data, config.model)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        drop_last=True,
    )
    batches = iter(loader)
    model = TinyPi0(config.model).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    progress = trange(1, config.steps + 1, desc="training")
    for step in progress:
        try:
            batch = next(batches)
        except StopIteration:
            batches = iter(loader)
            batch = next(batches)
        batch = move_to_device(batch, device)
        loss = model.loss(batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if step == 1 or step % config.log_every == 0:
            progress.set_postfix(loss=f"{loss.item():.4f}")
        if step % config.save_every == 0 or step == config.steps:
            torch.save(
                {"step": step, "model": model.state_dict(), "config": config},
                output_dir / f"checkpoint_{step:06d}.pt",
            )
    return output_dir / f"checkpoint_{config.steps:06d}.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the tiny π0 teaching model")
    parser.add_argument("--dataset", default="synthetic")
    parser.add_argument("--steps", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default="outputs/debug")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = ModelConfig()
    if args.dataset == "lerobot/pusht":
        model = replace(model, state_dim=2, action_dim=2)
    config = TrainConfig(
        model=model,
        data=DataConfig(dataset=args.dataset),
        steps=args.steps,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
    )
    checkpoint = train(config, args.device)
    print(f"saved checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
