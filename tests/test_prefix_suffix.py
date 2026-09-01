import torch

from pi_from_scratch.config import ModelConfig
from pi_from_scratch.data import SyntheticPiDataset
from pi_from_scratch.models import (
    TinyPi0,
    TwoExpertTransformerBlock,
    make_pi0_attention_layout,
)


def test_pi0_attention_layout_has_expected_information_flow() -> None:
    prefix_mask = torch.ones(1, 3, dtype=torch.bool)
    action_mask = torch.ones(1, 3, dtype=torch.bool)

    layout = make_pi0_attention_layout(prefix_mask, action_mask)

    expected = torch.tensor(
        [
            [1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
        ],
        dtype=torch.bool,
    )
    torch.testing.assert_close(layout.attention_mask[0], expected)


def test_pi0_attention_layout_excludes_padded_queries_and_keys() -> None:
    prefix_mask = torch.tensor([[True, True, False]])
    action_mask = torch.tensor([[True, False]])

    layout = make_pi0_attention_layout(prefix_mask, action_mask)

    invalid = ~layout.input_mask[0]
    assert not layout.attention_mask[0, invalid].any()
    assert not layout.attention_mask[0, :, invalid].any()


def test_prefix_output_cannot_depend_on_action_suffix() -> None:
    torch.manual_seed(0)
    block = TwoExpertTransformerBlock(width=16, num_heads=4).eval()
    prefix = torch.randn(2, 3, 16)
    suffix_a = torch.randn(2, 4, 16)
    suffix_b = torch.randn(2, 4, 16)
    layout = make_pi0_attention_layout(
        torch.ones(2, 3, dtype=torch.bool),
        torch.ones(2, 3, dtype=torch.bool),
    )

    prefix_a, _ = block(prefix, suffix_a, layout)
    prefix_b, _ = block(prefix, suffix_b, layout)

    torch.testing.assert_close(prefix_a, prefix_b)


def test_action_suffix_depends_on_observation_prefix() -> None:
    torch.manual_seed(0)
    block = TwoExpertTransformerBlock(width=16, num_heads=4).eval()
    prefix_a = torch.randn(2, 3, 16)
    prefix_b = prefix_a.clone()
    prefix_b[:, :, 0] += 1.0
    suffix = torch.randn(2, 4, 16)
    layout = make_pi0_attention_layout(
        torch.ones(2, 3, dtype=torch.bool),
        torch.ones(2, 3, dtype=torch.bool),
    )

    _, suffix_a = block(prefix_a, suffix, layout)
    _, suffix_b = block(prefix_b, suffix, layout)

    assert not torch.allclose(suffix_a, suffix_b)


def test_tiny_pi0_velocity_uses_image_text_and_state_conditions() -> None:
    torch.manual_seed(0)
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

    def predict(image: torch.Tensor, text_ids: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        prefix, prefix_mask = model.encode_prefix(
            image,
            text_ids,
            batch["text_mask"],
        )
        result = model.predict_velocity(
            noisy_actions,
            time,
            prefix,
            prefix_mask,
            state,
            batch["action_mask"],
        )
        assert isinstance(result, torch.Tensor)
        return result

    baseline = predict(batch["image"], batch["text_ids"], batch["state"])
    changed_image = predict(1.0 - batch["image"], batch["text_ids"], batch["state"])
    changed_text_ids = batch["text_ids"].clone()
    changed_text_ids[:, 0] = (changed_text_ids[:, 0] + 1) % config.vocab_size
    changed_text = predict(batch["image"], changed_text_ids, batch["state"])
    changed_state = predict(batch["image"], batch["text_ids"], batch["state"] + 0.5)

    assert not torch.allclose(baseline, changed_image)
    assert not torch.allclose(baseline, changed_text)
    assert not torch.allclose(baseline, changed_state)
