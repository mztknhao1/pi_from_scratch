import pytest
import torch

from pi_from_scratch.data_windows import build_future_action_window, split_episode_ids
from pi_from_scratch.model import masked_action_mse


def test_window_repeats_tail_but_marks_it_invalid() -> None:
    episode_actions = torch.tensor([[0.0], [1.0], [2.0]])
    window = build_future_action_window(
        episode_actions, anchor_index=1, horizon=4, fps=10.0
    )

    torch.testing.assert_close(window.values[:, 0], torch.tensor([1.0, 2.0, 2.0, 2.0]))
    assert window.valid_mask.tolist() == [True, True, False, False]
    assert window.source_indices.tolist() == [1, 2, 2, 2]
    torch.testing.assert_close(window.timestamps_s, torch.tensor([0.1, 0.2, 0.3, 0.4]))


def test_window_never_reads_another_episode() -> None:
    first_episode = torch.tensor([[0.0], [1.0]])
    second_episode = torch.tensor([[100.0], [101.0]])
    concatenated = torch.cat((first_episode, second_episode))

    safe = build_future_action_window(first_episode, anchor_index=1, horizon=3, fps=10.0)
    unsafe_global_slice = concatenated[1:4]

    assert safe.values[:, 0].tolist() == [1.0, 1.0, 1.0]
    assert unsafe_global_slice[:, 0].tolist() == [1.0, 100.0, 101.0]


def test_episode_split_is_disjoint_and_reproducible() -> None:
    first = split_episode_ids(range(10), validation_fraction=0.2, seed=7)
    second = split_episode_ids(range(10), validation_fraction=0.2, seed=7)

    assert first == second
    assert len(first.train) == 8
    assert len(first.validation) == 2
    assert not set(first.train) & set(first.validation)


def test_episode_split_rejects_duplicate_ids() -> None:
    with pytest.raises(ValueError, match="unique"):
        split_episode_ids([0, 0, 1])


def test_window_contract_rejects_non_prefix_mask() -> None:
    window = build_future_action_window(
        torch.tensor([[0.0], [1.0], [2.0]]), anchor_index=0, horizon=3, fps=10.0
    )
    with pytest.raises(ValueError, match="contiguous valid prefix"):
        type(window)(
            values=window.values,
            valid_mask=torch.tensor([True, False, True]),
            timestamps_s=window.timestamps_s,
            source_indices=window.source_indices,
        )


def test_masked_action_mse_ignores_padding() -> None:
    predicted = torch.tensor([[[0.0], [0.0], [100.0]]])
    target = torch.zeros_like(predicted)
    mask = torch.tensor([[True, True, False]])

    torch.testing.assert_close(masked_action_mse(predicted, target, mask), torch.tensor(0.0))
