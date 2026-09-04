import pytest
import torch

from pi_from_scratch.data import (
    ExperienceEpisode,
    ExperienceSource,
    improvement_indicators,
    n_step_advantages,
    returns_to_go,
    sparse_completion_rewards,
    task_advantage_threshold,
)
from pi_from_scratch.evaluation.recap import run_recap_experiment


def test_experience_episode_requires_aligned_time_dimensions() -> None:
    with pytest.raises(ValueError, match="time dimensions"):
        ExperienceEpisode(
            task="place cup",
            observations=torch.zeros(3, 2),
            actions=torch.zeros(3, 1),
            rewards=torch.zeros(3),
            source=ExperienceSource.AUTONOMOUS,
            success=False,
            intervention_mask=torch.zeros(3, dtype=torch.bool),
        )


def test_sparse_rewards_and_returns_preserve_terminal_outcome() -> None:
    success = sparse_completion_rewards(4, success=True, failure_penalty=10.0)
    failure = sparse_completion_rewards(4, success=False, failure_penalty=10.0)
    assert success.tolist() == [-1.0, -1.0, -1.0, 0.0]
    assert returns_to_go(success).tolist() == [-3.0, -2.0, -1.0, 0.0]
    assert failure.tolist() == [-1.0, -1.0, -1.0, -10.0]


def test_n_step_advantage_and_intervention_override() -> None:
    rewards = torch.tensor([-1.0, -1.0, 0.0])
    values = torch.tensor([-2.5, -1.2, -0.1, 0.0])
    advantages = n_step_advantages(rewards, values, n_step=2)
    assert torch.allclose(advantages, torch.tensor([0.4, 0.2, 0.1]))
    threshold = task_advantage_threshold(advantages, quantile=0.5)
    indicators = improvement_indicators(
        advantages,
        threshold=threshold,
        intervention_mask=torch.tensor([False, False, True]),
    )
    assert indicators.tolist() == [True, False, True]


def test_advantage_conditioning_extracts_the_better_mode() -> None:
    metrics = run_recap_experiment(samples=64, train_steps=220, seed=14)
    errors = metrics["high_quality_action_mae"]
    assert errors["advantage_conditioned"] < 0.08
    assert errors["advantage_conditioned"] < errors["mixed_behavior_cloning"] * 0.2
    modes = metrics["conditioned_modes"]
    assert modes["mean_action_positive"] > modes["mean_action_negative"] + 1.5
