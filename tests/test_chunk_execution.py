import pytest
import torch

from pi_from_scratch.contracts import ActionChunk, ActionRepresentation, ActionSpec
from pi_from_scratch.runtime import boundary_action_jumps, chunk_timing, stitch_chunk_prefixes


def _spec() -> ActionSpec:
    return ActionSpec(
        dim=1,
        space="toy_position",
        representation=ActionRepresentation.ABSOLUTE,
        frame="world",
        units=("m",),
        minimum=(-100.0,),
        maximum=(100.0,),
    )


def _chunk(values: list[float], start_step: int, *, valid_count: int | None = None) -> ActionChunk:
    tensor = torch.tensor(values, dtype=torch.float32)[None, :, None]
    horizon = len(values)
    valid_count = horizon if valid_count is None else valid_count
    return ActionChunk(
        values=tensor,
        valid_mask=(torch.arange(horizon) < valid_count)[None],
        timestamps_s=(start_step + torch.arange(horizon, dtype=torch.float32))[None] / 10.0,
        spec=_spec(),
    )


def test_chunk_timing_distinguishes_span_and_coverage() -> None:
    timing = chunk_timing(16, 10.0)

    assert timing.timestamp_span_s == pytest.approx(1.5)
    assert timing.coverage_duration_s == pytest.approx(1.6)


def test_executor_uses_only_first_e_actions_from_each_chunk() -> None:
    trace = stitch_chunk_prefixes(
        [_chunk([0, 1, 2, 3], 0), _chunk([2.5, 3.5, 4.5, 5.5], 2)],
        execution_horizon=2,
    )

    assert trace.values[:, 0].tolist() == [0.0, 1.0, 2.5, 3.5]
    assert trace.timestamps_s.tolist() == pytest.approx([0.0, 0.1, 0.2, 0.3])
    assert trace.source_chunk_indices.tolist() == [0, 0, 1, 1]
    assert trace.chunk_boundary_steps == (2,)
    torch.testing.assert_close(boundary_action_jumps(trace), torch.tensor([1.5]))


def test_executor_respects_episode_tail_mask() -> None:
    trace = stitch_chunk_prefixes(
        [_chunk([0, 1, 1, 1], 0, valid_count=2)], execution_horizon=4
    )

    assert trace.values[:, 0].tolist() == [0.0, 1.0]
    assert trace.chunk_boundary_steps == ()


def test_executor_rejects_normalized_chunk() -> None:
    raw = _chunk([0, 1], 0)
    normalized = ActionChunk(
        values=raw.values,
        valid_mask=raw.valid_mask,
        timestamps_s=raw.timestamps_s,
        spec=raw.spec,
        normalized=True,
        normalization_id="test-stats",
    )

    with pytest.raises(ValueError, match="denormalize"):
        stitch_chunk_prefixes([normalized], execution_horizon=1)


def test_executor_requires_e_no_larger_than_h() -> None:
    with pytest.raises(ValueError, match="must not exceed"):
        stitch_chunk_prefixes([_chunk([0, 1], 0)], execution_horizon=3)


def test_executor_rejects_non_increasing_timestamps_across_replans() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        stitch_chunk_prefixes(
            [_chunk([0, 1], 0), _chunk([2, 3], 1)], execution_horizon=2
        )
