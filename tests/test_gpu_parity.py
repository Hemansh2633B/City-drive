"""CPU-reference parity tests for the optional batched tensor backend."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ai.gpu_collision import forward_checkpoint_crossing, nearest_centerline_distance
from ai.gpu_training_env import GPUTrainingEnv
from ai.training_env import ActionMode, ObservationMode, RacingEnv, TrainingConfig
from core.config import load_config
from core.math import vec2


def make_reference(seed: int = 41) -> RacingEnv:
    config = TrainingConfig(action_mode=ActionMode.CONTINUOUS)
    config.curriculum["enabled"] = False
    config.track_complexity = 0.35
    config.max_episode_steps = 200
    env = RacingEnv(config, load_config())
    env.reset(seed=seed)
    return env


def test_gpu_request_never_silently_falls_back_to_cpu():
    if torch.cuda.is_available():
        pytest.skip("CUDA is present; unavailable-device check does not apply")
    reference = make_reference()
    try:
        with pytest.raises(RuntimeError, match="requested CUDA"):
            GPUTrainingEnv.from_env(reference, batch_size=2, device="cuda")
    finally:
        reference.close()


def test_reset_observation_and_one_step_match_cpu_reference():
    reference = make_reference(41)
    gpu_env = None
    try:
        gpu_env = GPUTrainingEnv.from_env(
            reference, batch_size=2, device="cpu", dtype=torch.float64
        )
        gpu_observation = gpu_env.reset()
        cpu_observation, _ = reference.reset(seed=41)
        np.testing.assert_allclose(
            gpu_observation[0].detach().cpu().numpy(),
            cpu_observation,
            atol=3e-4,
            rtol=3e-4,
        )

        action = np.array([0.0, 0.65, 0.0], dtype=np.float32)
        cpu_observation, cpu_reward, cpu_terminated, cpu_truncated, _ = reference.step(action)
        gpu_observation, gpu_reward, terminated, truncated, _ = gpu_env.step(
            torch.as_tensor(np.tile(action, (2, 1)), dtype=torch.float64)
        )
        assert bool(terminated[0]) == cpu_terminated
        assert bool(truncated[0]) == cpu_truncated
        assert float(gpu_reward[0]) == pytest.approx(cpu_reward, abs=5e-3, rel=5e-3)
        np.testing.assert_allclose(
            gpu_observation[0].detach().cpu().numpy(),
            cpu_observation,
            atol=3e-3,
            rtol=3e-3,
        )
        # Two identical actions and initial states evolve independently but
        # deterministically; neither slot is a shared mutable vehicle state.
        torch.testing.assert_close(gpu_observation[0], gpu_observation[1], atol=1e-9, rtol=1e-9)
    finally:
        if gpu_env is not None:
            gpu_env.close()
        reference.close()


def test_batched_centerline_distance_matches_reference_rule():
    points = torch.tensor([[0.0, 0.0], [5.0, 2.0], [11.0, -1.0]], dtype=torch.float64)
    starts = points[:-1]
    segments = points[1:] - points[:-1]
    lengths = segments.square().sum(dim=1)
    positions = torch.tensor([[2.0, 1.0], [5.0, 10.0]], dtype=torch.float64)
    distances = nearest_centerline_distance(
        positions, starts, segments, lengths, chunk_size=1
    )
    expected = []
    for position in positions.numpy():
        best = float("inf")
        for start, segment in zip(starts.numpy(), segments.numpy()):
            fraction = np.clip(np.dot(position - start, segment) / np.dot(segment, segment), 0.0, 1.0)
            closest = start + fraction * segment
            best = min(best, np.linalg.norm(position - closest))
        expected.append(best)
    np.testing.assert_allclose(distances.numpy(), expected, atol=1e-10, rtol=1e-10)


def test_checkpoint_crossing_masks_direction_and_gate_width():
    cp_pos = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
    cp_dir = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    cp_width = torch.tensor([4.0], dtype=torch.float64)
    target = torch.zeros(3, dtype=torch.long)
    previous = torch.tensor([[-1.0, 0.0], [1.0, 0.0], [-1.0, 3.0]], dtype=torch.float64)
    current = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [1.0, 3.0]], dtype=torch.float64)
    crossed = forward_checkpoint_crossing(
        previous, current, cp_pos, cp_dir, cp_width, target
    )
    assert crossed.tolist() == [True, False, False]


@pytest.mark.parametrize("mode", [ObservationMode.VECTOR, ObservationMode.SENSOR, ObservationMode.COMBINED])
def test_gpu_observation_modes_match_declared_shapes(mode):
    config = TrainingConfig(observation_mode=mode, action_mode=ActionMode.CONTINUOUS)
    config.curriculum["enabled"] = False
    reference = RacingEnv(config, load_config())
    gpu_env = None
    try:
        reference.reset(seed=83)
        gpu_env = GPUTrainingEnv.from_env(reference, batch_size=3, device="cpu")
        obs = gpu_env.reset()
        assert obs.shape == (3, reference.observation_space.shape[0])
        assert torch.isfinite(obs).all()
        assert torch.isfinite(obs[0]).all()
    finally:
        if gpu_env is not None:
            gpu_env.close()
        reference.close()


def test_partial_reset_and_terminated_slots_are_independent():
    reference = make_reference(72)
    gpu_env = None
    try:
        gpu_env = GPUTrainingEnv.from_env(reference, batch_size=3, device="cpu")
        gpu_env.reset()
        actions = torch.tensor([[0.0, 1.0, 0.0]] * 3)
        gpu_env.step(actions)
        frozen_position = gpu_env.position[1].clone()
        gpu_env.angle[1] = torch.pi
        frozen_angle = gpu_env.angle[1].clone()
        gpu_env.terminated[1] = True
        for _ in range(3):
            gpu_env.step(actions)
        torch.testing.assert_close(gpu_env.position[1], frozen_position)
        torch.testing.assert_close(gpu_env.angle[1], frozen_angle, atol=0.0, rtol=0.0)
        gpu_env.reset(mask=torch.tensor([False, True, False]))
        assert not bool(gpu_env.terminated[1])
        torch.testing.assert_close(gpu_env.position[1], gpu_env.spawn_position)
    finally:
        if gpu_env is not None:
            gpu_env.close()
        reference.close()


def test_multistep_acceleration_braking_and_steering_trajectory_matches_cpu():
    reference = make_reference(111)
    gpu_env = None
    try:
        reference.reset(seed=111)
        gpu_env = GPUTrainingEnv.from_env(
            reference, batch_size=1, device="cpu", dtype=torch.float64
        )
        gpu_env.set_track_from_env(reference)
        gpu_env.reset()

        def action_for_step(step):
            if step < 15:
                return np.array([0.0, 0.7, 0.0], dtype=np.float32)
            if step < 25:
                return np.array([0.0, 0.0, 0.65], dtype=np.float32)
            if step < 50:
                return np.array([0.35, 0.65, 0.0], dtype=np.float32)
            return np.array([-0.2, 0.45, 0.0], dtype=np.float32)

        for step in range(70):
            action = action_for_step(step)
            cpu_observation, cpu_reward, cpu_terminated, cpu_truncated, _ = reference.step(action)
            gpu_observation, gpu_reward, terminated, truncated, _ = gpu_env.step(
                torch.as_tensor(action[None, :], dtype=torch.float64)
            )

            np.testing.assert_allclose(
                reference.vehicle.state.position,
                gpu_env.position[0].detach().cpu().numpy(),
                atol=2e-2,
                rtol=2e-3,
            )
            assert float(gpu_env.angle[0]) == pytest.approx(
                reference.vehicle.state.angle, abs=2e-2, rel=2e-3
            )
            assert float(gpu_env.speed[0]) == pytest.approx(
                reference.vehicle.state.speed, abs=2e-2, rel=2e-3
            )
            assert float(gpu_reward[0]) == pytest.approx(cpu_reward, abs=1e-2, rel=2e-2)
            assert bool(terminated[0]) == cpu_terminated
            assert bool(truncated[0]) == cpu_truncated
            np.testing.assert_allclose(
                gpu_observation[0].detach().cpu().numpy(),
                cpu_observation,
                atol=2e-2,
                rtol=2e-2,
            )
            if cpu_terminated or cpu_truncated:
                break
    finally:
        if gpu_env is not None:
            gpu_env.close()
        reference.close()


def test_complete_short_episode_truncation_matches_cpu():
    config = TrainingConfig(action_mode=ActionMode.CONTINUOUS)
    config.curriculum["enabled"] = False
    config.max_episode_steps = 12
    config.track_complexity = 0.25
    reference = RacingEnv(config, load_config())
    gpu_env = None
    try:
        reference.reset(seed=234)
        gpu_env = GPUTrainingEnv.from_env(
            reference, batch_size=1, device="cpu", dtype=torch.float64
        )
        gpu_env.reset()
        action = np.array([0.0, 0.7, 0.0], dtype=np.float32)
        saw_done = False
        for _ in range(config.max_episode_steps):
            cpu_observation, cpu_reward, cpu_terminated, cpu_truncated, _ = reference.step(action)
            gpu_observation, gpu_reward, terminated, truncated, _ = gpu_env.step(
                torch.as_tensor(action[None, :], dtype=torch.float64)
            )
            assert bool(terminated[0]) == cpu_terminated
            assert bool(truncated[0]) == cpu_truncated
            assert float(gpu_reward[0]) == pytest.approx(cpu_reward, abs=1e-2, rel=1e-2)
            np.testing.assert_allclose(
                gpu_observation[0].detach().cpu().numpy(),
                cpu_observation,
                atol=1e-2,
                rtol=1e-2,
            )
            if cpu_terminated or cpu_truncated:
                saw_done = True
                break
        assert saw_done
        assert bool(gpu_env.truncated[0])
    finally:
        if gpu_env is not None:
            gpu_env.close()
        reference.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware is not available")
def test_cuda_multistep_physics_and_observations_match_cpu_reference():
    reference = make_reference(393)
    gpu_env = None
    try:
        reference.reset(seed=393)
        gpu_env = GPUTrainingEnv.from_env(
            reference, batch_size=1, device="cuda", dtype=torch.float64
        )
        gpu_env.reset()
        for step in range(25):
            action = (
                np.array([0.0, 0.7, 0.0], dtype=np.float32)
                if step < 12
                else np.array([0.25, 0.6, 0.0], dtype=np.float32)
            )
            cpu_observation, cpu_reward, cpu_terminated, cpu_truncated, _ = reference.step(action)
            gpu_observation, gpu_reward, terminated, truncated, _ = gpu_env.step(
                torch.as_tensor(action[None, :], dtype=torch.float64, device="cuda")
            )
            np.testing.assert_allclose(
                reference.vehicle.state.position,
                gpu_env.position[0].detach().cpu().numpy(),
                atol=2e-2,
                rtol=2e-3,
            )
            assert float(gpu_reward[0].detach().cpu()) == pytest.approx(
                cpu_reward, abs=1e-2, rel=1e-2
            )
            assert bool(terminated[0].detach().cpu()) == cpu_terminated
            assert bool(truncated[0].detach().cpu()) == cpu_truncated
            np.testing.assert_allclose(
                gpu_observation[0].detach().cpu().numpy(),
                cpu_observation,
                atol=2e-2,
                rtol=2e-2,
            )
            if cpu_terminated or cpu_truncated:
                break
    finally:
        if gpu_env is not None:
            gpu_env.close()
        reference.close()
