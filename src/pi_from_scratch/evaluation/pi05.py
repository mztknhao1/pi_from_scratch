"""Controlled mechanism checks for the π0.5 heterogeneous-training lesson."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import torch
from torch import Tensor

from pi_from_scratch.data.mixtures import MixtureSchedule, RobotActionBatch, SemanticBatch
from pi_from_scratch.models.tiny_pi05 import TinyPi05, parameter_grad_norm
from pi_from_scratch.objectives.mixed import (
    continuous_flow_objective,
    discrete_action_objective,
    semantic_objective,
)


def make_pi05_toy_batches(
    *,
    batch_size: int = 64,
    observation_dim: int = 6,
    horizon: int = 8,
    action_dim: int = 2,
    seed: int = 12,
) -> tuple[RobotActionBatch, SemanticBatch]:
    if batch_size <= 0 or horizon <= 0:
        raise ValueError("batch_size and horizon must be positive")
    generator = torch.Generator().manual_seed(seed)
    observation = torch.randn(batch_size, observation_dim, generator=generator)
    semantic_labels = torch.argmax(observation[:, :3], dim=-1)
    action_tokens = (observation[:, 0] > 0).long() + 2 * (observation[:, 1] > 0).long()
    phase = torch.linspace(0.0, 1.0, horizon)
    target_dimensions = []
    for index in range(action_dim):
        primary = observation[:, index % observation_dim]
        secondary = observation[:, (index + 2) % observation_dim]
        target_dimensions.append(primary + (0.4 / (index + 1)) * secondary)
    target = torch.stack(target_dimensions, dim=-1)[:, None]
    actions = target + 0.15 * phase[None, :, None]
    robot = RobotActionBatch(
        observation=observation,
        actions=actions,
        action_tokens=action_tokens,
        source="toy_robot",
    )
    semantic = SemanticBatch(
        observation=observation.roll(1, dims=0),
        labels=semantic_labels.roll(1, dims=0),
        source="toy_web_vlm",
    )
    return robot, semantic


def _backbone_gradient(model: TinyPi05, loss: Tensor) -> float:
    model.zero_grad(set_to_none=True)
    loss.backward()
    return parameter_grad_norm(model.backbone)


def _parameter_distance(before: TinyPi05, after: TinyPi05, name: str) -> float:
    first = dict(before.named_parameters())
    second = dict(after.named_parameters())
    squared = torch.zeros(())
    for key, value in first.items():
        if key.startswith(name):
            squared = squared + (value.detach().cpu() - second[key].detach().cpu()).square().sum()
    return float(squared.sqrt().item())


def _flow_finetune(
    initial: TinyPi05,
    robot: RobotActionBatch,
    *,
    insulate_backbone: bool,
    noise: Tensor,
    time: Tensor,
    steps: int,
) -> tuple[TinyPi05, float, float]:
    model = copy.deepcopy(initial)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-2)
    first_loss = 0.0
    last_loss = 0.0
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = continuous_flow_objective(
            model,
            robot,
            insulate_backbone=insulate_backbone,
            noise=noise,
            time=time,
        )
        if step == 0:
            first_loss = float(loss.item())
        loss.backward()
        optimizer.step()
        last_loss = float(loss.item())
    return model, first_loss, last_loss


def run_pi05_mixture_experiment(
    *,
    mixture_steps: int = 2000,
    robot_probability: float = 0.35,
    finetune_steps: int = 40,
    seed: int = 12,
) -> dict[str, object]:
    torch.manual_seed(seed)
    robot, semantic = make_pi05_toy_batches(seed=seed)
    model = TinyPi05()
    noise = torch.randn(robot.actions.shape, generator=torch.Generator().manual_seed(seed + 1))
    time = torch.full((robot.observation.shape[0],), 0.35)

    semantic_grad = _backbone_gradient(model, semantic_objective(model, semantic))
    discrete_grad = _backbone_gradient(model, discrete_action_objective(model, robot))
    naive_flow_grad = _backbone_gradient(
        model,
        continuous_flow_objective(
            model,
            robot,
            insulate_backbone=False,
            noise=noise,
            time=time,
        ),
    )
    insulated_flow_grad = _backbone_gradient(
        model,
        continuous_flow_objective(
            model,
            robot,
            insulate_backbone=True,
            noise=noise,
            time=time,
        ),
    )

    initial = copy.deepcopy(model)
    naive, naive_first, naive_last = _flow_finetune(
        initial,
        robot,
        insulate_backbone=False,
        noise=noise,
        time=time,
        steps=finetune_steps,
    )
    insulated, insulated_first, insulated_last = _flow_finetune(
        initial,
        robot,
        insulate_backbone=True,
        noise=noise,
        time=time,
        steps=finetune_steps,
    )
    with torch.no_grad():
        reference_logits = initial.semantic_logits(semantic.observation)
        naive_logit_drift = torch.mean(
            torch.abs(naive.semantic_logits(semantic.observation) - reference_logits)
        ).item()
        insulated_logit_drift = torch.mean(
            torch.abs(insulated.semantic_logits(semantic.observation) - reference_logits)
        ).item()

    schedule = MixtureSchedule.draw(
        mixture_steps,
        robot_probability=robot_probability,
        seed=seed,
    )
    return {
        "experiment_scope": "synthetic mechanism check, not paper-scale performance",
        "mixture": {
            "steps": mixture_steps,
            "requested_robot_probability": robot_probability,
            "counts": schedule.counts(),
            "realized_ratios": schedule.realized_ratios(),
        },
        "backbone_gradient_norm": {
            "semantic_ce": semantic_grad,
            "discrete_action_ce": discrete_grad,
            "continuous_flow_without_insulation": naive_flow_grad,
            "continuous_flow_with_insulation": insulated_flow_grad,
        },
        "flow_only_finetune": {
            "steps": finetune_steps,
            "without_insulation": {
                "first_loss": naive_first,
                "last_loss": naive_last,
                "backbone_parameter_drift": _parameter_distance(initial, naive, "backbone"),
                "semantic_logit_drift": naive_logit_drift,
            },
            "with_insulation": {
                "first_loss": insulated_first,
                "last_loss": insulated_last,
                "backbone_parameter_drift": _parameter_distance(initial, insulated, "backbone"),
                "semantic_logit_drift": insulated_logit_drift,
            },
        },
    }


def write_pi05_metrics(path: Path, metrics: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_pi05_routing_svg(path: Path, metrics: dict[str, object]) -> None:
    """Write a compact data-routing and gradient-flow diagram."""
    path.parent.mkdir(parents=True, exist_ok=True)
    gradients = metrics["backbone_gradient_norm"]
    naive = float(gradients["continuous_flow_without_insulation"])  # type: ignore[index]
    insulated = float(gradients["continuous_flow_with_insulation"])  # type: ignore[index]
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="520" viewBox="0 0 1100 520">
<rect width="1100" height="520" fill="#f7f4ed"/>
<text x="50" y="52" font-family="sans-serif" font-size="28" font-weight="700" fill="#17212b">π₀.₅ 风格异构训练：样本决定 objective，梯度决定知识去向</text>
<rect x="50" y="105" width="210" height="100" rx="14" fill="#dbeafe" stroke="#2563eb" stroke-width="2"/>
<text x="155" y="145" text-anchor="middle" font-family="sans-serif" font-size="20" font-weight="700">Semantic batch</text>
<text x="155" y="175" text-anchor="middle" font-family="sans-serif" font-size="15">image/text → label</text>
<rect x="50" y="310" width="210" height="110" rx="14" fill="#fee2e2" stroke="#dc2626" stroke-width="2"/>
<text x="155" y="350" text-anchor="middle" font-family="sans-serif" font-size="20" font-weight="700">Robot batch</text>
<text x="155" y="380" text-anchor="middle" font-family="sans-serif" font-size="15">observation → action</text>
<rect x="410" y="175" width="220" height="100" rx="14" fill="#fef3c7" stroke="#d97706" stroke-width="2"/>
<text x="520" y="218" text-anchor="middle" font-family="sans-serif" font-size="20" font-weight="700">VLM backbone</text>
<text x="520" y="247" text-anchor="middle" font-family="sans-serif" font-size="15">共享语义表征</text>
<rect x="790" y="80" width="250" height="80" rx="14" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
<text x="915" y="115" text-anchor="middle" font-family="sans-serif" font-size="18" font-weight="700">semantic / token head</text>
<text x="915" y="140" text-anchor="middle" font-family="sans-serif" font-size="14">CE 更新 backbone</text>
<rect x="790" y="315" width="250" height="90" rx="14" fill="#ede9fe" stroke="#7c3aed" stroke-width="2"/>
<text x="915" y="350" text-anchor="middle" font-family="sans-serif" font-size="18" font-weight="700">continuous action expert</text>
<text x="915" y="378" text-anchor="middle" font-family="sans-serif" font-size="14">flow matching，只更新 expert</text>
<path d="M260 155 L410 205" stroke="#2563eb" stroke-width="4" fill="none" marker-end="url(#arrow)"/>
<path d="M260 350 L410 245" stroke="#dc2626" stroke-width="4" fill="none" marker-end="url(#arrow)"/>
<path d="M630 205 L790 130" stroke="#16a34a" stroke-width="4" fill="none" marker-end="url(#arrow)"/>
<path d="M630 255 L735 330" stroke="#7c3aed" stroke-width="4" fill="none"/>
<line x1="735" y1="305" x2="735" y2="360" stroke="#dc2626" stroke-width="8"/>
<path d="M750 340 L790 350" stroke="#7c3aed" stroke-width="4" fill="none" marker-end="url(#arrow)"/>
<text x="690" y="302" text-anchor="middle" font-family="sans-serif" font-size="15" fill="#dc2626" font-weight="700">stop-gradient</text>
<text x="550" y="465" text-anchor="middle" font-family="sans-serif" font-size="16" fill="#374151">本次检查：flow→backbone grad {naive:.4f}；加入 insulation 后 {insulated:.4f}</text>
<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" fill="#374151"/></marker></defs>
</svg>
"""
    path.write_text(svg, encoding="utf-8")
