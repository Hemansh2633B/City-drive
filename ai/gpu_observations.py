"""GPU-resident observation assembly following RacingEnv feature ordering."""
from __future__ import annotations

import torch

from ai.training_env import ObservationMode, VECTOR_OBSERVATION_SIZE


@torch.no_grad()
def build_observations(env) -> torch.Tensor:
    """Build vector, sensor or combined observations on the active device.

    The reference training environment's PhysicsWorld has no static track bodies;
    its raycast therefore has no road geometry to hit. Sensor rays are all max
    range (normalized value 1), exactly as the current single-car setup returns.
    """
    mode = env.train_config.observation_mode
    if mode == ObservationMode.IMAGE:
        raise ValueError("GPU backend does not support image observations; use vector or sensor mode.")

    batch = env.batch_size
    device = env.device
    dtype = env.dtype
    forward = torch.stack((torch.cos(env.angle), torch.sin(env.angle)), dim=1)
    right = torch.stack((-forward[:, 1], forward[:, 0]), dim=1)

    next_checkpoint = torch.remainder(
        env.last_checkpoint + 1, max(env.checkpoint_positions.shape[0], 1)
    )
    if env.checkpoint_positions.shape[0]:
        to_checkpoint = env.checkpoint_positions[next_checkpoint] - env.position
    else:
        to_checkpoint = torch.zeros((batch, 2), device=device, dtype=dtype)
    checkpoint_distance = torch.linalg.vector_norm(to_checkpoint, dim=1)
    checkpoint_direction = to_checkpoint / checkpoint_distance.clamp_min(1e-12)[:, None]
    checkpoint_direction = torch.where(
        (checkpoint_distance > 0.0)[:, None],
        checkpoint_direction,
        torch.zeros_like(checkpoint_direction),
    )

    if env.racing_line_positions.shape[0]:
        line_last = env.racing_line_positions.shape[0] - 1
        progress = (env.checkpoints_crossed.to(dtype) / max(env.checkpoint_positions.shape[0], 1)).clamp(0.0, 1.0)
        line_index = progress * line_last
        index_a = torch.floor(line_index).to(torch.long).clamp(0, line_last)
        index_b = (index_a + 1).clamp(max=line_last)
        alpha = line_index - index_a.to(dtype)
        racing_position = (
            env.racing_line_positions[index_a] * (1.0 - alpha[:, None])
            + env.racing_line_positions[index_b] * alpha[:, None]
        )
        racing_direction = (
            env.racing_line_directions[index_a] * (1.0 - alpha[:, None])
            + env.racing_line_directions[index_b] * alpha[:, None]
        )
        racing_direction = racing_direction / torch.linalg.vector_norm(
            racing_direction, dim=1
        ).clamp_min(1e-12)[:, None]
        racing_speed = (
            env.racing_line_target_speeds[index_a] * (1.0 - alpha)
            + env.racing_line_target_speeds[index_b] * alpha
        )
        racing_curvature = (
            env.racing_line_curvatures[index_a] * (1.0 - alpha)
            + env.racing_line_curvatures[index_b] * alpha
        )
    else:
        racing_position = torch.zeros((batch, 2), device=device, dtype=dtype)
        racing_direction = torch.zeros_like(racing_position)
        racing_direction[:, 0] = 1.0
        racing_speed = torch.zeros(batch, device=device, dtype=dtype)
        racing_curvature = torch.zeros_like(racing_speed)

    to_racing_line = racing_position - env.position
    racing_distance = torch.linalg.vector_norm(to_racing_line, dim=1)
    racing_direction_to_car = to_racing_line / racing_distance.clamp_min(1e-12)[:, None]
    racing_direction_to_car = torch.where(
        (racing_distance > 0.0)[:, None],
        racing_direction_to_car,
        torch.zeros_like(racing_direction_to_car),
    )

    vector = torch.zeros(
        (batch, VECTOR_OBSERVATION_SIZE), device=device, dtype=dtype
    )
    vector[:, 0] = env.speed / 100.0
    vector[:, 1] = env.forward_speed / 100.0
    vector[:, 2] = env.lateral_speed / 50.0
    vector[:, 3] = env.angular_velocity / 10.0
    vector[:, 4] = env.slip_angle
    vector[:, 5] = env.is_drifting.to(dtype)
    vector[:, 6] = env.drift_intensity
    vector[:, 7] = env.on_road.to(dtype)
    vector[:, 8] = env.surface_grip
    vector[:, 9] = env.throttle
    vector[:, 10] = env.brake
    vector[:, 11] = env.steering
    vector[:, 12] = env.boost_active.to(dtype)
    vector[:, 13] = (env.boost_cooldown_timer <= 0.0).to(dtype)
    vector[:, 14] = checkpoint_distance / 200.0
    vector[:, 15] = (checkpoint_direction * forward).sum(dim=1)
    vector[:, 16] = (checkpoint_direction * right).sum(dim=1)
    vector[:, 17] = racing_distance / 50.0
    vector[:, 18] = (racing_direction_to_car * forward).sum(dim=1)
    vector[:, 19] = racing_speed / 100.0
    vector[:, 20] = racing_curvature * 50.0
    vector[:, 21] = env.laps_completed.to(dtype) / max(env.num_laps, 1)
    vector[:, 22] = env.checkpoints_crossed.to(dtype) / max(env.checkpoint_positions.shape[0], 1)
    vector[:, 23] = env.episode_steps.to(dtype) / max(int(env.train_config.max_episode_steps), 1)

    ray_count = int(env.train_config.sensor_config.get("num_rays", 16))
    sensor = torch.ones((batch, ray_count), device=device, dtype=dtype)
    sensor_state = torch.stack(
        (
            env.speed / 100.0,
            env.forward_speed / 100.0,
            env.lateral_speed / 50.0,
            env.angular_velocity / 10.0,
            env.slip_angle,
            env.on_road.to(dtype),
        ),
        dim=1,
    )
    sensor_obs = torch.cat((sensor, sensor_state), dim=1)
    if mode == ObservationMode.VECTOR:
        observation = vector
    elif mode == ObservationMode.SENSOR:
        observation = sensor_obs
    else:
        observation = torch.cat((vector, sensor_obs), dim=1)
    return torch.nan_to_num(observation, nan=0.0, posinf=1.0, neginf=-1.0)
