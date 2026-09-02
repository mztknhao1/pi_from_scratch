from itertools import pairwise

import pytest
import torch

from pi_from_scratch.evaluation import run_sampling_sweep
from pi_from_scratch.inference import FlowSolver, flow_sample, heun_sample, model_evaluations


def quadratic_path_problem() -> tuple[torch.Tensor, torch.Tensor]:
    target = torch.tensor([[[0.0, 0.5], [1.0, -0.5]]])
    noise = torch.tensor([[[2.0, -1.0], [-1.0, 3.0]]])
    return target, noise


def test_euler_error_decreases_with_steps_on_quadratic_path() -> None:
    target, noise = quadratic_path_problem()
    delta = noise - target

    def velocity(_x_t: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        return 2.0 * time[:, None, None] * delta

    errors = []
    for steps in (1, 2, 4, 8):
        sampled = flow_sample(
            velocity,
            target.shape,
            device=target.device,
            num_steps=steps,
            noise=noise,
            solver="euler",
        )
        errors.append(torch.mean(torch.abs(sampled - target)).item())

    assert all(later < earlier for earlier, later in pairwise(errors))
    assert errors[-1] == pytest.approx(errors[0] / 8.0)


def test_heun_exactly_integrates_time_linear_velocity() -> None:
    target, noise = quadratic_path_problem()
    delta = noise - target

    sampled = heun_sample(
        lambda _x_t, time: 2.0 * time[:, None, None] * delta,
        target.shape,
        device=target.device,
        num_steps=2,
        noise=noise,
    )

    torch.testing.assert_close(sampled, target)


def test_fixed_noise_sampling_and_sweep_are_reproducible() -> None:
    target, noise = quadratic_path_problem()
    valid_mask = torch.ones(target.shape[:2], dtype=torch.bool)
    delta = noise - target

    def sample(solver: FlowSolver, steps: int) -> torch.Tensor:
        return flow_sample(
            lambda _x_t, time: 2.0 * time[:, None, None] * delta,
            target.shape,
            device=target.device,
            num_steps=steps,
            noise=noise,
            solver=solver,
        )

    first = sample("euler", 4)
    second = sample("euler", 4)
    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)

    points = run_sampling_sweep(
        sample,
        target,
        valid_mask,
        solvers=("euler", "heun"),
        steps=(1, 2),
        warmup=0,
        repeats=1,
    )
    assert [(point.solver, point.steps, point.model_evaluations) for point in points] == [
        ("euler", 1, 1),
        ("euler", 2, 2),
        ("heun", 1, 2),
        ("heun", 2, 4),
    ]


def test_solver_contract_rejects_invalid_inputs() -> None:
    target, noise = quadratic_path_problem()

    assert model_evaluations("euler", 4) == 4
    assert model_evaluations("heun", 4) == 8
    with pytest.raises(ValueError, match="unknown"):
        model_evaluations("rk4", 4)
    with pytest.raises(ValueError, match="shape"):
        flow_sample(
            lambda x_t, _time: x_t,
            target.shape,
            device=target.device,
            noise=noise[:, :1],
        )
    with pytest.raises(TypeError, match="floating"):
        flow_sample(
            lambda x_t, _time: x_t,
            target.shape,
            device=target.device,
            noise=torch.ones(target.shape, dtype=torch.int64),
        )
