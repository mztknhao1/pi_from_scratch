import pytest
import torch

from pi_from_scratch.representations import (
    FastActionTokenizer,
    FastQuantileStats,
    IntegerBPE,
    dct_actions,
    idct_actions,
)


def smooth_chunks(count: int = 48, horizon: int = 20, action_dim: int = 4) -> torch.Tensor:
    generator = torch.Generator().manual_seed(17)
    time = torch.linspace(0.0, 1.0, horizon)
    chunks = []
    for _ in range(count):
        offset = torch.randn(action_dim, generator=generator) * 0.3
        slope = torch.randn(action_dim, generator=generator) * 0.5
        phase = torch.rand(action_dim, generator=generator) * 6.28
        amplitude = torch.rand(action_dim, generator=generator) * 0.25
        chunk = offset + time[:, None] * slope
        chunk += amplitude * torch.sin(2.0 * torch.pi * time[:, None] + phase)
        chunks.append(chunk)
    return torch.stack(chunks)


def test_orthonormal_dct_round_trip() -> None:
    actions = torch.randn(3, 16, 7)
    reconstructed = idct_actions(dct_actions(actions))
    torch.testing.assert_close(reconstructed, actions, atol=2e-5, rtol=2e-5)


def test_quantile_statistics_are_fitted_from_passed_training_data() -> None:
    train = torch.tensor([[[0.0, 10.0]], [[2.0, 14.0]]])
    stats = FastQuantileStats.fit(train, lower_quantile=0.0, upper_quantile=1.0)
    torch.testing.assert_close(stats.low, torch.tensor([0.0, 10.0]))
    torch.testing.assert_close(stats.high, torch.tensor([2.0, 14.0]))
    torch.testing.assert_close(stats.normalize(torch.tensor([[1.0, 12.0]])), torch.zeros(1, 2))


def test_integer_bpe_is_lossless_and_compresses_repeated_patterns() -> None:
    sequences = [[0, 0, 0, 0, 1, 1]] * 20
    codec = IntegerBPE.fit(
        sequences,
        minimum_value=-1,
        maximum_value=1,
        vocab_size=12,
    )
    tokens = codec.encode(sequences[0])
    assert codec.decode(tokens) == sequences[0]
    assert len(tokens) < len(sequences[0])


def test_fast_tokenizer_preserves_shape_with_bounded_error() -> None:
    chunks = smooth_chunks()
    tokenizer = FastActionTokenizer.fit(chunks[:36], scale=10.0, vocab_size=160)
    tokens = tokenizer.encode(chunks[36])
    reconstructed = tokenizer.decode(tokens)
    assert reconstructed.shape == chunks[36].shape
    assert torch.mean(torch.abs(reconstructed - chunks[36])).item() < 0.08
    assert len(tokens) < tokenizer.coefficient_count


def test_fast_tokenizer_rejects_wrong_decoded_length() -> None:
    chunks = smooth_chunks()
    tokenizer = FastActionTokenizer.fit(chunks, scale=10.0, vocab_size=160)
    with pytest.raises(ValueError, match="coefficients"):
        tokenizer.decode([])
