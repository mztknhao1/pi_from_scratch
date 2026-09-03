"""Objective routing for typed π0.5-style training samples."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor
from torch.nn import functional as F

from pi_from_scratch.data.mixtures import MixedBatch, RobotActionBatch, SemanticBatch
from pi_from_scratch.models.tiny_pi05 import TinyPi05
from pi_from_scratch.objectives.flow_matching import sample_flow_batch


@dataclass(frozen=True)
class RoutedLoss:
    total: Tensor
    terms: dict[str, Tensor]


def semantic_objective(model: TinyPi05, batch: SemanticBatch) -> Tensor:
    return F.cross_entropy(model.semantic_logits(batch.observation), batch.labels)


def discrete_action_objective(model: TinyPi05, batch: RobotActionBatch) -> Tensor:
    return F.cross_entropy(
        model.discrete_action_logits(batch.observation),
        batch.action_tokens,
    )


def continuous_flow_objective(
    model: TinyPi05,
    batch: RobotActionBatch,
    *,
    insulate_backbone: bool,
    noise: Tensor | None = None,
    time: Tensor | None = None,
) -> Tensor:
    flow = sample_flow_batch(batch.actions, noise=noise, time=time)
    predicted = model.predict_velocity(
        batch.observation,
        flow.noisy_actions,
        flow.time,
        insulate_backbone=insulate_backbone,
    )
    return F.mse_loss(predicted, flow.target_velocity)


def route_mixed_objective(
    model: TinyPi05,
    batch: MixedBatch,
    *,
    insulate_backbone: bool,
    discrete_action_weight: float = 1.0,
    flow_weight: float = 1.0,
    noise: Tensor | None = None,
    time: Tensor | None = None,
) -> RoutedLoss:
    """Apply only objectives for which the typed batch has valid labels."""
    if isinstance(batch, SemanticBatch):
        semantic = semantic_objective(model, batch)
        return RoutedLoss(total=semantic, terms={"semantic_ce": semantic})
    if isinstance(batch, RobotActionBatch):
        discrete = discrete_action_objective(model, batch)
        flow = continuous_flow_objective(
            model,
            batch,
            insulate_backbone=insulate_backbone,
            noise=noise,
            time=time,
        )
        return RoutedLoss(
            total=discrete_action_weight * discrete + flow_weight * flow,
            terms={"discrete_action_ce": discrete, "continuous_flow_mse": flow},
        )
    raise TypeError(f"unsupported mixed batch type: {type(batch)!r}")
