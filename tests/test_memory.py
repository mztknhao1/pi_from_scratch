import pytest
import torch

from pi_from_scratch.evaluation.memory import run_memory_experiment
from pi_from_scratch.memory import (
    LongTermTextMemory,
    MultiScaleMemory,
    ShortTermVideoMemory,
    VisualMemoryFrame,
)


def frame(timestamp: float, value: float, visible: bool) -> VisualMemoryFrame:
    return VisualMemoryFrame(
        timestamp_s=timestamp,
        features=torch.tensor([value]),
        visible_mask=torch.tensor([visible]),
    )


def test_short_term_memory_is_bounded_and_requires_monotonic_time() -> None:
    memory = ShortTermVideoMemory(capacity=2, feature_dim=1)
    memory.append(frame(0.0, -1.0, True))
    memory.append(frame(1.0, 0.0, False))
    memory.append(frame(2.0, 1.0, True))
    assert tuple(item.timestamp_s for item in memory.frames) == (1.0, 2.0)
    with pytest.raises(ValueError, match="strictly increasing"):
        memory.append(frame(2.0, 0.0, True))


def test_short_term_encoder_recovers_latest_visible_value_through_occlusion() -> None:
    memory = ShortTermVideoMemory(capacity=4, feature_dim=1)
    memory.append(frame(0.0, -1.0, True))
    memory.append(frame(1.0, 0.0, False))
    memory.append(frame(2.0, 0.0, False))
    encoded = memory.encode()
    assert encoded.valid_mask.tolist() == [True]
    assert encoded.values.tolist() == [-1.0]


def test_long_term_text_memory_tracks_success_and_failure_separately() -> None:
    memory = LongTermTextMemory()
    memory.update("add cheese", success=True)
    memory.update("add cheese", success=True)
    memory.update("close sandwich", success=False)
    assert memory.completed("add cheese")
    assert "add cheese ×2" in memory.summary()
    assert "failed: close sandwich" in memory.summary()


def test_multiscale_context_exposes_both_memory_modalities() -> None:
    memory = MultiScaleMemory(video_capacity=2, feature_dim=1)
    memory.short_term.append(frame(0.0, 1.0, True))
    memory.long_term.update("prepare pan", success=True)
    context = memory.context()
    assert context.short_term_mask.tolist() == [True]
    assert context.short_term_features.tolist() == [1.0]
    assert "prepare pan" in context.long_term_summary


def test_multiscale_memory_is_required_for_both_controlled_tasks() -> None:
    metrics = run_memory_experiment(episodes=100, seed=13)
    methods = metrics["methods"]
    assert methods["short_only"]["occlusion_accuracy"] == 1.0
    assert methods["short_only"]["long_horizon_progress"] < 1.0
    assert methods["long_only"]["occlusion_accuracy"] < 0.7
    assert methods["long_only"]["long_horizon_progress"] == 1.0
    assert methods["multi_scale"]["mean_mechanism_score"] == 1.0
