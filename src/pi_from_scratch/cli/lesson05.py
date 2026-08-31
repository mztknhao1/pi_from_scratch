"""Runnable probes for lesson 5: conditional flow-matching targets and direction."""

import argparse
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from pi_from_scratch.inference import euler_sample
from pi_from_scratch.objectives import (
    linear_flow_path,
    masked_flow_matching_loss,
)


class TinyConditionalVectorField(nn.Module):
    """A small MLP used only to prove that the flow objective can overfit."""

    def __init__(self, *, condition_dim: int, horizon: int, action_dim: int, width: int = 64):
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        flat_action_dim = horizon * action_dim
        self.net = nn.Sequential(
            nn.Linear(flat_action_dim + condition_dim + 1, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, flat_action_dim),
        )

    def forward(self, noisy_actions: Tensor, time: Tensor, condition: Tensor) -> Tensor:
        flat_actions = noisy_actions.flatten(start_dim=1)
        inputs = torch.cat((flat_actions, time[:, None], condition), dim=-1)
        return self.net(inputs).reshape(-1, self.horizon, self.action_dim)


@dataclass(frozen=True)
class TinyOverfitResult:
    initial_loss: float
    final_loss: float
    sampled_action_mae: float


def make_toy_conditioned_chunks(horizon: int = 6) -> tuple[Tensor, Tensor]:
    """Return four conditions with one deterministic action chunk per condition."""
    condition = torch.tensor(
        [
            [-1.0, -1.0],
            [-1.0, 1.0],
            [1.0, -1.0],
            [1.0, 1.0],
        ]
    )
    phase = torch.linspace(0.0, 1.0, horizon)[None, :]
    x = condition[:, :1] + 0.4 * phase * condition[:, 1:2]
    y = condition[:, 1:2] - 0.3 * phase * condition[:, :1]
    return condition, torch.stack((x, y), dim=-1)


def run_tiny_overfit(*, steps: int = 1_000, seed: int = 7) -> TinyOverfitResult:
    """Overfit a fixed bank built from four condition-to-action mappings on CPU."""
    if steps < 1:
        raise ValueError("steps must be positive")
    torch.manual_seed(seed)
    condition, actions = make_toy_conditioned_chunks()
    model = TinyConditionalVectorField(
        condition_dim=condition.shape[-1],
        horizon=actions.shape[1],
        action_dim=actions.shape[2],
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

    flow_points_per_condition = 16
    train_condition = condition.repeat_interleave(flow_points_per_condition, dim=0)
    train_actions = actions.repeat_interleave(flow_points_per_condition, dim=0)
    train_noise = torch.randn_like(train_actions)
    train_time = torch.linspace(0.05, 0.95, flow_points_per_condition).repeat(
        condition.shape[0]
    )
    train_flow = linear_flow_path(train_actions, train_noise, train_time)
    valid_mask = torch.ones(train_actions.shape[:2], dtype=torch.bool)

    def eval_loss() -> Tensor:
        prediction = model(
            train_flow.noisy_actions,
            train_flow.time,
            train_condition,
        )
        return masked_flow_matching_loss(
            prediction,
            train_flow.target_velocity,
            valid_mask,
        )

    with torch.no_grad():
        initial_loss = eval_loss().item()

    model.train()
    for _ in range(steps):
        prediction = model(
            train_flow.noisy_actions,
            train_flow.time,
            train_condition,
        )
        loss = masked_flow_matching_loss(
            prediction,
            train_flow.target_velocity,
            valid_mask,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        final_loss = eval_loss().item()
        sample_noise = torch.randn_like(actions)
        sampled_actions = euler_sample(
            lambda x_t, time: model(x_t, time, condition),
            actions.shape,
            device=actions.device,
            num_steps=20,
            noise=sample_noise,
        )
        sampled_action_mae = torch.mean(torch.abs(sampled_actions - actions)).item()

    return TinyOverfitResult(
        initial_loss=initial_loss,
        final_loss=final_loss,
        sampled_action_mae=sampled_action_mae,
    )


def _print_path_probe() -> None:
    actions = torch.tensor([[[2.0]]])
    noise = torch.tensor([[[-1.0]]])
    print("linear path: data at t=0, noise at t=1")
    print("time | x_t   | target velocity")
    print("-----+-------+----------------")
    for value in (0.0, 0.25, 0.5, 0.75, 1.0):
        flow = linear_flow_path(actions, noise, torch.tensor([value]))
        print(
            f"{value:>4.2f} | {flow.noisy_actions.item():>5.2f} | "
            f"{flow.target_velocity.item():>15.2f}"
        )


def _print_direction_probe() -> None:
    actions = torch.tensor([[[0.25, -0.5], [1.0, 0.75]]])
    noise = torch.tensor([[[2.0, 1.0], [-1.0, 3.0]]])

    def oracle_velocity(x_t: Tensor, time: Tensor) -> Tensor:
        return (x_t - actions) / time[:, None, None]

    correct = euler_sample(
        oracle_velocity,
        actions.shape,
        device=actions.device,
        num_steps=4,
        noise=noise,
    )
    reversed_result = euler_sample(
        lambda x_t, time: -oracle_velocity(x_t, time),
        actions.shape,
        device=actions.device,
        num_steps=4,
        noise=noise,
    )
    print("\noracle sampler direction")
    print(f"  start noise MAE:       {torch.mean(torch.abs(noise - actions)).item():.6f}")
    print(f"  correct direction MAE: {torch.mean(torch.abs(correct - actions)).item():.6f}")
    print(
        "  reversed velocity MAE: "
        f"{torch.mean(torch.abs(reversed_result - actions)).item():.6f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect conditional flow-matching targets")
    parser.add_argument("--steps", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _print_path_probe()
    _print_direction_probe()
    result = run_tiny_overfit(steps=args.steps, seed=args.seed)
    print("\nfour-sample conditional tiny overfit")
    print(f"  fixed evaluation loss before: {result.initial_loss:.6f}")
    print(f"  fixed evaluation loss after:  {result.final_loss:.6f}")
    print(f"  sampled action MAE:           {result.sampled_action_mae:.6f}")


if __name__ == "__main__":
    main()
