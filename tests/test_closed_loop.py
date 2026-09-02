import pytest
import torch

from pi_from_scratch.contracts import ActionChunk, ObservationBatch, PolicyOutput
from pi_from_scratch.envs import PointReachEnv
from pi_from_scratch.policies import PointGoalPolicy
from pi_from_scratch.runtime import run_synchronous_episode


def _policy(env: PointReachEnv, *, latency: float = 0.0) -> PointGoalPolicy:
    return PointGoalPolicy(
        env.action_spec,
        horizon=8,
        fps=env.fps,
        motion_per_step=env.max_motion_per_step,
        simulated_latency_s=latency,
    )


def test_synchronous_runner_reaches_goal_and_records_provenance() -> None:
    env = PointReachEnv(max_steps=40)
    run = run_synchronous_episode(
        env,
        _policy(env),
        execution_horizon=3,
        max_steps=40,
        seed=7,
    )

    assert run.episode.success
    assert run.episode.num_steps < 40
    assert run.trace.states.shape[0] == run.episode.num_steps + 1
    assert run.trace.actions.shape == (run.episode.num_steps, 2)
    assert torch.all(run.trace.source_observation_timestamps_s <= run.trace.action_timestamps_s)
    assert run.trace.chunk_offsets.tolist()[:3] == [0, 1, 2]
    assert run.trace.chunk_boundary_steps == run.episode.chunk_boundary_steps


def test_refill_deadline_is_measured_against_executed_prefix_duration() -> None:
    env = PointReachEnv(max_steps=40)
    run = run_synchronous_episode(
        env,
        _policy(env, latency=0.31),
        execution_horizon=3,
        max_steps=40,
        seed=7,
    )

    assert run.episode.deadline_misses == len(run.episode.inference_latencies_s)


class WrongTimestampPolicy:
    def __init__(self, wrapped: PointGoalPolicy) -> None:
        self.wrapped = wrapped

    def predict_chunk(self, observation: ObservationBatch) -> PolicyOutput:
        output = self.wrapped.predict_chunk(observation)
        chunk = output.action_chunk
        return PolicyOutput(
            action_chunk=ActionChunk(
                values=chunk.values,
                valid_mask=chunk.valid_mask,
                timestamps_s=chunk.timestamps_s + 0.05,
                spec=chunk.spec,
            ),
            source_observation_timestamp_s=output.source_observation_timestamp_s,
            generated_at_monotonic_s=output.generated_at_monotonic_s,
            inference_latency_s=output.inference_latency_s,
        )


def test_runner_rejects_policy_on_the_wrong_control_grid() -> None:
    env = PointReachEnv(max_steps=20)
    with pytest.raises(ValueError, match="control grid"):
        run_synchronous_episode(
            env,
            WrongTimestampPolicy(_policy(env)),
            execution_horizon=3,
            max_steps=20,
            seed=7,
        )
