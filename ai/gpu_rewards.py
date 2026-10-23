"""Batched reward, checkpoint, lap and termination calculations."""
from __future__ import annotations

import torch

from ai.gpu_collision import forward_checkpoint_crossing


@torch.no_grad()
def calculate_rewards(env, active: torch.Tensor, previous_position: torch.Tensor) -> torch.Tensor:
    """Update per-slot episode counters and return rewards for one vector step.

    Reward terms and checkpoint crossing semantics are taken from
    RacingEnv._calculate_reward. Population members are independent episodes.
    """
    cfg = env.train_config
    weights = cfg.rewards
    dt = env.dt
    position = env.position
    delta = position - previous_position
    travelled = torch.linalg.vector_norm(delta, dim=1)
    reward = torch.where(
        env.forward_speed > 0.0,
        float(weights["progress_per_meter"]) * travelled,
        float(weights["backward_penalty"]) * travelled,
    )
    reward = reward + torch.where(
        env.forward_speed > 5.0,
        float(weights["speed_bonus_factor"]) * env.forward_speed,
        torch.zeros_like(reward),
    )

    count = env.checkpoint_positions.shape[0]
    if count:
        target = torch.remainder(env.last_checkpoint + 1, count)
        crossed = forward_checkpoint_crossing(
            previous_position,
            position,
            env.checkpoint_positions,
            env.checkpoint_directions,
            env.checkpoint_widths,
            target,
        ) & active
        old_crossed_count = env.checkpoints_crossed
        completed_lap = crossed & (target == 0) & (old_crossed_count > 0)
        reward = reward + torch.where(
            crossed,
            torch.where(
                completed_lap,
                float(weights["lap_bonus"]),
                float(weights["checkpoint_bonus"]),
            ),
            torch.zeros_like(reward),
        )
        env.last_checkpoint.copy_(torch.where(crossed, target, env.last_checkpoint))
        env.checkpoints_crossed.add_(crossed.to(torch.int64))
        env.laps_completed.add_(completed_lap.to(torch.int64))

        if env.lap_history.shape[1] > 0:
            lap_slot = old_crossed_count.new_tensor(0) + (
                env.laps_completed - completed_lap.to(torch.int64)
            )
            lap_slot = lap_slot.clamp(min=0, max=env.lap_history.shape[1] - 1)
            previous_lap = env.lap_history.gather(1, lap_slot[:, None]).squeeze(1)
            lap_value = torch.where(completed_lap, env.current_lap_time, previous_lap)
            env.lap_history.scatter_(1, lap_slot[:, None], lap_value[:, None])
        env.best_lap_time.copy_(
            torch.minimum(
                env.best_lap_time,
                torch.where(completed_lap, env.current_lap_time, env.best_lap_time),
            )
        )
        env.current_lap_time.copy_(
            torch.where(completed_lap, torch.zeros_like(env.current_lap_time), env.current_lap_time)
        )

    # Collision events are separate from the cumulative collision metric. The
    # focused reference environment has no static collision bodies or opponents,
    # so both are currently zero; keeping the counters separate avoids losing
    # events if collision geometry is added in a later compatible extension.
    reward = reward + float(weights["collision_penalty"]) * env.collision_events
    env.collision_events.copy_(
        torch.where(active, torch.zeros_like(env.collision_events), env.collision_events)
    )

    offroad = ~env.on_road
    next_offroad_time = torch.where(
        offroad, env.offroad_time + dt, torch.zeros_like(env.offroad_time)
    )
    next_total_offroad = env.total_offroad_time + offroad.to(env.position.dtype) * dt
    reward = reward + torch.where(
        offroad,
        float(weights["offroad_penalty_per_second"]) * dt,
        torch.zeros_like(reward),
    )
    env.offroad_time.copy_(torch.where(active, next_offroad_time, env.offroad_time))
    env.total_offroad_time.copy_(torch.where(active, next_total_offroad, env.total_offroad_time))

    reward = reward + torch.where(
        env.forward_speed < -1.0,
        float(weights["backward_penalty"]) * dt,
        torch.zeros_like(reward),
    )
    reward = reward + torch.where(
        env.is_drifting,
        float(weights["spin_penalty"]) * dt,
        torch.zeros_like(reward),
    )

    below_speed = env.speed < float(cfg.termination["min_speed_threshold"])
    next_idle = torch.where(
        below_speed, env.idle_time + dt, torch.zeros_like(env.idle_time)
    )
    reward = reward + torch.where(
        below_speed & (next_idle > 1.0),
        float(weights["idle_penalty"]) * dt,
        torch.zeros_like(reward),
    )
    env.idle_time.copy_(torch.where(active, next_idle, env.idle_time))

    if count:
        next_target = torch.remainder(env.last_checkpoint + 1, count)
        to_next = env.checkpoint_positions[next_target] - position
        forward = torch.stack((torch.cos(env.angle), torch.sin(env.angle)), dim=1)
        wrong_way = (to_next * forward).sum(dim=1) < -0.5
        next_wrong_way = torch.where(
            wrong_way, env.wrong_way_time + dt, torch.zeros_like(env.wrong_way_time)
        )
        reward = reward + torch.where(
            next_wrong_way > 1.0,
            float(weights["wrong_way_penalty"]) * dt,
            torch.zeros_like(reward),
        )
        env.wrong_way_time.copy_(torch.where(active, next_wrong_way, env.wrong_way_time))

    env.last_position.copy_(torch.where(active[:, None], position, env.last_position))
    reward = reward * active.to(reward.dtype)
    return reward


@torch.no_grad()
def update_termination(env, active: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute newly terminated/truncated flags using RacingEnv thresholds."""
    cfg = env.train_config
    next_step = env.episode_steps + 1
    next_time = env.elapsed_time + env.dt

    race_completed = env.laps_completed >= env.num_laps
    would_truncate = (
        (next_step >= int(cfg.max_episode_steps))
        | (next_time >= float(cfg.max_race_time))
    )
    other_termination = (
        (env.offroad_time > float(cfg.termination["max_offroad_time"]))
        | (env.vehicle_collision_count >= int(cfg.termination["max_collisions"]))
        | (env.idle_time > float(cfg.termination["max_idle_time"]))
        | (env.wrong_way_time > float(cfg.termination["max_wrong_way_time"]))
    )
    # Preserve RacingEnv check order: race completion, then truncation, then
    # off-road/collision/idle/wrong-way termination.
    terminated = race_completed | (~would_truncate & other_termination)
    truncated = ~race_completed & would_truncate
    terminated = terminated & active
    truncated = truncated & active
    return terminated, truncated
