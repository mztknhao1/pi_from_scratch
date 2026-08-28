import pytest
import torch

from pi_from_scratch.visualize_chunk import project_workspace_xy, select_timeline_slots


def test_project_workspace_corners_to_image() -> None:
    points = torch.tensor([[0.0, 0.0], [512.0, 512.0], [256.0, 128.0]])
    projected = project_workspace_xy(points, image_width=96, image_height=96)

    torch.testing.assert_close(
        projected,
        torch.tensor([[0.0, 0.0], [95.0, 95.0], [47.5, 23.75]]),
    )


def test_select_timeline_slots_includes_first_and_last() -> None:
    slots = select_timeline_slots(torch.ones(16, dtype=torch.bool), max_frames=6)

    assert slots == (0, 3, 6, 9, 12, 15)


def test_select_timeline_slots_ignores_padding() -> None:
    slots = select_timeline_slots(
        torch.tensor([True, True, True, False, False]), max_frames=6
    )

    assert slots == (0, 1, 2)


def test_select_timeline_slots_rejects_empty_chunk() -> None:
    with pytest.raises(ValueError, match="at least one valid"):
        select_timeline_slots(torch.zeros(3, dtype=torch.bool), max_frames=2)
