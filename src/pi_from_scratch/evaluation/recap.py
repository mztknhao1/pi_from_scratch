"""Controlled checks for RECAP-style advantage-conditioned policy extraction."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F

from pi_from_scratch.data.experience import (
    improvement_indicators,
    returns_to_go,
    sparse_completion_rewards,
)
from pi_from_scratch.models.tiny_recap import TinyAdvantagePolicy


def _fit_policy(
    observation: Tensor,
    indicator: Tensor,
    actions: Tensor,
    *,
    seed: int,
    steps: int,
) -> TinyAdvantagePolicy:
    torch.manual_seed(seed)
    model = TinyAdvantagePolicy()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-2)
    for _ in range(steps):
        prediction = model(observation, indicator)
        loss = F.mse_loss(prediction, actions)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model


def _make_mixed_experience(samples: int) -> tuple[Tensor, Tensor, Tensor, dict[str, int]]:
    if samples < 8:
        raise ValueError("samples must be at least 8")
    observations = torch.linspace(-1.0, 1.0, samples).unsqueeze(-1)
    good_actions = 0.6 * observations + 1.0
    bad_actions = 0.6 * observations - 1.0

    correction_count = samples // 4
    correction_observations = observations[:correction_count]
    correction_actions = good_actions[:correction_count]
    all_observations = torch.cat((observations, observations, correction_observations))
    all_actions = torch.cat((good_actions, bad_actions, correction_actions))

    advantages = torch.cat(
        (
            torch.ones(samples),
            -torch.ones(samples),
            -0.25 * torch.ones(correction_count),
        )
    )
    intervention_mask = torch.zeros_like(advantages, dtype=torch.bool)
    intervention_mask[-correction_count:] = True
    positive = improvement_indicators(
        advantages,
        threshold=0.0,
        intervention_mask=intervention_mask,
    )
    indicator = positive.to(dtype=torch.long)
    counts = {
        "demonstration": samples,
        "autonomous": samples,
        "intervention": correction_count,
    }
    return all_observations, all_actions, indicator, counts


def run_recap_experiment(
    *, samples: int = 96, train_steps: int = 300, seed: int = 14
) -> dict[str, object]:
    if train_steps <= 0:
        raise ValueError("train_steps must be positive")
    observation, actions, indicator, counts = _make_mixed_experience(samples)

    baseline = _fit_policy(
        observation,
        torch.full_like(indicator, -1),
        actions,
        seed=seed,
        steps=train_steps,
    )
    conditioned = _fit_policy(
        observation,
        indicator,
        actions,
        seed=seed,
        steps=train_steps,
    )

    evaluation_observation = torch.linspace(-0.95, 0.95, 41).unsqueeze(-1)
    target_good = 0.6 * evaluation_observation + 1.0
    target_bad = 0.6 * evaluation_observation - 1.0
    positive = torch.ones(evaluation_observation.shape[0], dtype=torch.long)
    negative = torch.zeros_like(positive)
    omitted = torch.full_like(positive, -1)

    with torch.no_grad():
        baseline_action = baseline(evaluation_observation, omitted)
        conditioned_good = conditioned(evaluation_observation, positive)
        conditioned_bad = conditioned(evaluation_observation, negative)
        conditioned_omitted = conditioned(evaluation_observation, omitted)

    success_rewards = sparse_completion_rewards(4, success=True, failure_penalty=10.0)
    failure_rewards = sparse_completion_rewards(4, success=False, failure_penalty=10.0)
    return {
        "experiment_scope": "controlled policy-extraction check, not online robot RL",
        "seed": seed,
        "train_steps": train_steps,
        "source_counts": counts,
        "reward_example": {
            "success_rewards": success_rewards.tolist(),
            "success_returns": returns_to_go(success_rewards).tolist(),
            "failure_rewards": failure_rewards.tolist(),
            "failure_returns": returns_to_go(failure_rewards).tolist(),
        },
        "high_quality_action_mae": {
            "mixed_behavior_cloning": float((baseline_action - target_good).abs().mean().item()),
            "advantage_conditioned": float((conditioned_good - target_good).abs().mean().item()),
        },
        "conditioned_modes": {
            "positive_action_mae": float((conditioned_good - target_good).abs().mean().item()),
            "negative_action_mae": float((conditioned_bad - target_bad).abs().mean().item()),
            "mean_action_positive": float(conditioned_good.mean().item()),
            "mean_action_negative": float(conditioned_bad.mean().item()),
            "mean_action_indicator_omitted": float(conditioned_omitted.mean().item()),
        },
    }


def write_recap_metrics(path: Path, metrics: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_recap_comparison_svg(path: Path, metrics: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    errors = metrics["high_quality_action_mae"]
    baseline = float(errors["mixed_behavior_cloning"])  # type: ignore[index]
    conditioned = float(errors["advantage_conditioned"])  # type: ignore[index]
    scale = 240.0 / max(baseline, conditioned, 1e-6)
    baseline_height = baseline * scale
    conditioned_height = conditioned * scale
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="900" height="470" viewBox="0 0 900 470">
<rect width="900" height="470" fill="#f7f4ed"/>
<text x="48" y="52" font-family="sans-serif" font-size="28" font-weight="700" fill="#17212b">RECAP：混合经验中保留行为，再选择更优模式</text>
<text x="48" y="83" font-family="sans-serif" font-size="15" fill="#4b5563">受控回归实验；纵轴为目标高质量动作的 MAE，越低越好</text>
<line x1="90" y1="380" x2="830" y2="380" stroke="#9ca3af"/>
<rect x="205" y="{380 - baseline_height:.1f}" width="150" height="{baseline_height:.1f}" rx="8" fill="#dd6b55"/>
<rect x="545" y="{380 - conditioned_height:.1f}" width="150" height="{conditioned_height:.1f}" rx="8" fill="#287c78"/>
<text x="280" y="{365 - baseline_height:.1f}" text-anchor="middle" font-family="sans-serif" font-size="18" font-weight="700">{baseline:.3f}</text>
<text x="620" y="{365 - conditioned_height:.1f}" text-anchor="middle" font-family="sans-serif" font-size="18" font-weight="700">{conditioned:.3f}</text>
<text x="280" y="415" text-anchor="middle" font-family="sans-serif" font-size="16">混合数据直接 BC</text>
<text x="620" y="415" text-anchor="middle" font-family="sans-serif" font-size="16">Advantage-conditioned</text>
</svg>
"""
    path.write_text(svg, encoding="utf-8")
