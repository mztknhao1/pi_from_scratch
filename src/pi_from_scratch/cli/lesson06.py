"""Runnable probes for lesson 6: π₀ prefix, action suffix, and attention flow."""

import torch
from torch import Tensor

from pi_from_scratch.config import ModelConfig
from pi_from_scratch.data import SyntheticPiDataset
from pi_from_scratch.models import TinyPi0, make_pi0_attention_layout


def _print_attention_mask() -> None:
    labels = ("I0", "I1", "L0", "S", "A0", "A1", "A2")
    layout = make_pi0_attention_layout(
        torch.ones(1, 3, dtype=torch.bool),
        torch.ones(1, 3, dtype=torch.bool),
    )
    print("attention mask: row=query, column=key, 1=visible")
    print("     " + " ".join(f"{label:>2}" for label in labels))
    for label, row in zip(labels, layout.attention_mask[0], strict=True):
        cells = " ".join(f"{int(value):>2}" for value in row)
        print(f"{label:>3}  {cells}")


def _predict(
    model: TinyPi0,
    batch: dict[str, Tensor],
    noisy_actions: Tensor,
    time: Tensor,
) -> tuple[Tensor, int, int]:
    prefix, prefix_mask = model.encode_prefix(
        batch["image"],
        batch["text_ids"],
        batch["text_mask"],
    )
    result = model.predict_velocity(
        noisy_actions,
        time,
        prefix,
        prefix_mask,
        batch["state"],
        batch["action_mask"],
        return_layout=True,
    )
    assert isinstance(result, tuple)
    velocity, layout = result
    return velocity, prefix.shape[1], layout.suffix_mask.shape[1]


def _print_condition_probe() -> None:
    torch.manual_seed(7)
    config = ModelConfig(
        image_size=32,
        action_horizon=4,
        width=32,
        num_layers=1,
        num_heads=4,
    )
    sample = SyntheticPiDataset(config, length=1)[0]
    batch = {key: value[None] for key, value in sample.items()}
    model = TinyPi0(config).eval()
    noisy_actions = torch.randn(1, config.action_horizon, config.action_dim)
    time = torch.tensor([0.6])

    baseline, prefix_len, suffix_len = _predict(model, batch, noisy_actions, time)

    image_batch = dict(batch)
    image_batch["image"] = 1.0 - batch["image"]
    image_velocity, _, _ = _predict(model, image_batch, noisy_actions, time)

    text_batch = dict(batch)
    text_batch["text_ids"] = batch["text_ids"].clone()
    text_batch["text_ids"][:, 0] = (text_batch["text_ids"][:, 0] + 1) % config.vocab_size
    text_velocity, _, _ = _predict(model, text_batch, noisy_actions, time)

    state_batch = dict(batch)
    state_batch["state"] = batch["state"] + 0.5
    state_velocity, _, _ = _predict(model, state_batch, noisy_actions, time)

    def mean_delta(other: Tensor) -> float:
        return torch.mean(torch.abs(baseline - other)).item()

    print("\ncondition path probe with fixed noisy action and flow time")
    print(f"  prefix tokens:       {prefix_len}")
    print(f"  suffix tokens:       {suffix_len} (1 state + {config.action_horizon} actions)")
    print(f"  change image |Δv|:   {mean_delta(image_velocity):.6f}")
    print(f"  change text  |Δv|:   {mean_delta(text_velocity):.6f}")
    print(f"  change state |Δv|:   {mean_delta(state_velocity):.6f}")


def main() -> None:
    _print_attention_mask()
    _print_condition_probe()


if __name__ == "__main__":
    main()
