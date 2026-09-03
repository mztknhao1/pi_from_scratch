import torch

from pi_from_scratch.data.mixtures import MixtureSchedule, RobotActionBatch
from pi_from_scratch.evaluation.pi05 import make_pi05_toy_batches, run_pi05_mixture_experiment
from pi_from_scratch.models.tiny_pi05 import TinyPi05, parameter_grad_norm
from pi_from_scratch.objectives.mixed import continuous_flow_objective, route_mixed_objective


def test_typed_batches_reject_missing_batch_alignment() -> None:
    try:
        RobotActionBatch(
            observation=torch.zeros(2, 3),
            actions=torch.zeros(3, 4, 2),
            action_tokens=torch.zeros(2, dtype=torch.long),
        )
    except ValueError as exc:
        assert "batch size" in str(exc)
    else:
        raise AssertionError("misaligned robot batch must fail")


def test_mixture_schedule_is_seeded_and_reports_realized_ratio() -> None:
    first = MixtureSchedule.draw(1000, robot_probability=0.3, seed=5)
    second = MixtureSchedule.draw(1000, robot_probability=0.3, seed=5)
    assert first.kinds == second.kinds
    assert sum(first.counts().values()) == 1000
    assert abs(first.realized_ratios()["robot_action"] - 0.3) < 0.07


def test_objective_router_activates_only_compatible_terms() -> None:
    robot, semantic = make_pi05_toy_batches(batch_size=8)
    model = TinyPi05()
    semantic_loss = route_mixed_objective(model, semantic, insulate_backbone=True)
    robot_loss = route_mixed_objective(
        model,
        robot,
        insulate_backbone=True,
        noise=torch.zeros_like(robot.actions),
        time=torch.full((8,), 0.5),
    )
    assert set(semantic_loss.terms) == {"semantic_ce"}
    assert set(robot_loss.terms) == {"discrete_action_ce", "continuous_flow_mse"}


def test_knowledge_insulation_blocks_only_flow_gradient_to_backbone() -> None:
    robot, _ = make_pi05_toy_batches(batch_size=8)
    model = TinyPi05()
    loss = continuous_flow_objective(
        model,
        robot,
        insulate_backbone=True,
        noise=torch.zeros_like(robot.actions),
        time=torch.full((8,), 0.5),
    )
    loss.backward()
    assert parameter_grad_norm(model.backbone) == 0.0
    assert parameter_grad_norm(model.action_expert) > 0.0


def test_pi05_mechanism_experiment_preserves_backbone_under_insulation() -> None:
    metrics = run_pi05_mixture_experiment(mixture_steps=200, finetune_steps=8)
    gradients = metrics["backbone_gradient_norm"]
    drift = metrics["flow_only_finetune"]
    assert gradients["semantic_ce"] > 0
    assert gradients["discrete_action_ce"] > 0
    assert gradients["continuous_flow_without_insulation"] > 0
    assert gradients["continuous_flow_with_insulation"] == 0
    assert drift["without_insulation"]["backbone_parameter_drift"] > 0
    assert drift["with_insulation"]["backbone_parameter_drift"] == 0
