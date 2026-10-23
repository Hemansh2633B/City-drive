"""GPU-native batched simulator for the focused RacingEnv training workflow.

This is an independent tensor backend, not a wrapper around Gymnasium. State
persists on the requested device between calls to step(). Track generation and
rendering remain CPU-side; the simulation hot path never calls RacingEnv.step().
"""
from __future__ import annotations

from typing import Any, Optional, Union
import time

import numpy as np
import torch

from ai.gpu_collision import nearest_centerline_distance
from ai.gpu_observations import build_observations
from ai.gpu_physics import VehicleParameters, calculate_vehicle_forces, integrate_rigid_body
from ai.gpu_rewards import calculate_rewards, update_termination
from ai.training_env import ActionMode, ObservationMode, TrainingConfig
from core.config import SimulationContext
from track.track import Track
from vehicles.vehicle import VehicleConfig


class GPUTrainingEnv:
    """Batched, tensor-resident counterpart to the single-car RacingEnv.

    Args:
        track: A pre-generated CPU Track. Convert geometry to tensors once.
        batch_size: Number of independent policies being simulated at once.
        device: Defaults to CUDA and raises if CUDA is unavailable. Passing
            device="cpu" is an explicit tensor-backend debug/parity mode; it is
            never selected automatically as a fallback.
        dtype: float32 is recommended for throughput; float64 is useful for
            parity debugging.

    Returned observations/rewards/flags are tensors on device with a leading
    batch dimension. This is intentionally a batched API rather than Gym's
    scalar single-environment API.
    """

    def __init__(
        self,
        track: Track,
        training_config: Optional[TrainingConfig] = None,
        vehicle_config: Optional[VehicleConfig] = None,
        *,
        dt: float = 1.0 / 60.0,
        physics_substeps: int = 4,
        batch_size: int = 1024,
        device: Union[str, torch.device] = "cuda",
        dtype: torch.dtype = torch.float32,
        segment_chunk_size: int = 64,
        profile_timing: bool = False,
    ) -> None:
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "GPUTrainingEnv requested CUDA, but torch.cuda.is_available() is False. "
                "Install a CUDA-enabled PyTorch build and verify the driver, or pass "
                "device='cpu' explicitly for debugging."
            )
        if self.device.type not in ("cuda", "cpu"):
            raise ValueError("device must be 'cuda' or 'cpu'")
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("dtype must be torch.float32 or torch.float64")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if segment_chunk_size < 1:
            raise ValueError("segment_chunk_size must be >= 1")

        self.dtype = dtype
        self.batch_size = int(batch_size)
        self.dt = float(dt)
        self.physics_substeps = int(physics_substeps)
        self.segment_chunk_size = int(segment_chunk_size)
        self.profile_timing = bool(profile_timing)
        self.last_step_timings: dict[str, float] = {}
        if self.physics_substeps < 1 or self.dt <= 0:
            raise ValueError("dt must be positive and physics_substeps >= 1")

        self.train_config = training_config or TrainingConfig()
        if self.train_config.observation_mode == ObservationMode.IMAGE:
            raise ValueError(
                "GPUTrainingEnv does not support image observations; use vector, sensor, or combined."
            )
        self.track: Track
        self.vehicle_config = vehicle_config or VehicleConfig()
        self.params = VehicleParameters.from_config(self.vehicle_config)
        self.num_laps = 3
        self._set_track(track)

        n, d, dev = self.batch_size, self.dtype, self.device

        # Rigid-body state; all per-agent tensors live on device for the rollout.
        self.position = torch.zeros((n, 2), dtype=d, device=dev)
        self.angle = torch.zeros(n, dtype=d, device=dev)
        self.velocity = torch.zeros((n, 2), dtype=d, device=dev)
        self.angular_velocity = torch.zeros(n, dtype=d, device=dev)
        self.force = torch.zeros((n, 2), dtype=d, device=dev)
        self.torque = torch.zeros(n, dtype=d, device=dev)

        # VehicleState values used by forces, rewards and observations.
        self.speed = torch.zeros(n, dtype=d, device=dev)
        self.forward_speed = torch.zeros(n, dtype=d, device=dev)
        self.lateral_speed = torch.zeros(n, dtype=d, device=dev)
        self.slip_angle = torch.zeros(n, dtype=d, device=dev)
        self.is_drifting = torch.zeros(n, dtype=torch.bool, device=dev)
        self.drift_intensity = torch.zeros(n, dtype=d, device=dev)
        self.on_road = torch.ones(n, dtype=torch.bool, device=dev)
        self.surface_grip = torch.full(
            (n,), self.surface_grip_value, dtype=d, device=dev
        )
        self.throttle = torch.zeros(n, dtype=d, device=dev)
        self.brake = torch.zeros(n, dtype=d, device=dev)
        self.steering = torch.zeros(n, dtype=d, device=dev)
        self.boost_active = torch.zeros(n, dtype=torch.bool, device=dev)
        self.boost_timer = torch.zeros(n, dtype=d, device=dev)
        self.boost_cooldown_timer = torch.zeros(n, dtype=d, device=dev)

        # Episode-level state and metrics.
        self.distance_traveled = torch.zeros(n, dtype=d, device=dev)
        self.max_speed_reached = torch.zeros(n, dtype=d, device=dev)
        self.vehicle_offroad_time = torch.zeros(n, dtype=d, device=dev)
        self.vehicle_collision_count = torch.zeros(n, dtype=torch.long, device=dev)
        self.collision_count = torch.zeros(n, dtype=torch.long, device=dev)
        self.collision_events = torch.zeros(n, dtype=torch.long, device=dev)
        self.episode_steps = torch.zeros(n, dtype=torch.long, device=dev)
        self.elapsed_time = torch.zeros(n, dtype=d, device=dev)
        self.episode_reward = torch.zeros(n, dtype=d, device=dev)
        self.last_checkpoint = torch.zeros(n, dtype=torch.long, device=dev)
        self.checkpoints_crossed = torch.zeros(n, dtype=torch.long, device=dev)
        self.laps_completed = torch.zeros(n, dtype=torch.long, device=dev)
        self.wrong_way_time = torch.zeros(n, dtype=d, device=dev)
        self.idle_time = torch.zeros(n, dtype=d, device=dev)
        self.offroad_time = torch.zeros(n, dtype=d, device=dev)
        self.total_offroad_time = torch.zeros(n, dtype=d, device=dev)
        self.last_position = torch.zeros((n, 2), dtype=d, device=dev)
        self.current_lap_time = torch.zeros(n, dtype=d, device=dev)
        self.best_lap_time = torch.full((n,), float("inf"), dtype=d, device=dev)
        self.lap_history = torch.zeros(
            (n, max(int(self.num_laps), 1)), dtype=d, device=dev
        )
        self.terminated = torch.zeros(n, dtype=torch.bool, device=dev)
        self.truncated = torch.zeros(n, dtype=torch.bool, device=dev)
        self.reset()

    @classmethod
    def from_env(
        cls,
        reference_env: Any,
        *,
        batch_size: int = 1024,
        device: Union[str, torch.device] = "cuda",
        dtype: torch.dtype = torch.float32,
        segment_chunk_size: int = 64,
        profile_timing: bool = False,
    ) -> "GPUTrainingEnv":
        """Build from an already-reset RacingEnv; no CPU stepping is performed."""
        if reference_env.track is None or reference_env.vehicle is None:
            raise RuntimeError("Call reference_env.reset(seed=...) before GPUTrainingEnv.from_env().")
        return cls(
            track=reference_env.track,
            training_config=reference_env.train_config,
            vehicle_config=reference_env.vehicle.config,
            dt=reference_env.sim_context.dt,
            physics_substeps=reference_env.sim_context.substeps,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
            segment_chunk_size=segment_chunk_size,
            profile_timing=profile_timing,
        )

    @property
    def active(self) -> torch.Tensor:
        """Mask of agents whose episodes have not finished."""
        return ~(self.terminated | self.truncated)

    def _to_tensor(self, values: Any) -> torch.Tensor:
        numpy_dtype = np.float64 if self.dtype == torch.float64 else np.float32
        array = np.asarray(values, dtype=numpy_dtype)
        return torch.as_tensor(array, dtype=self.dtype, device=self.device)

    def _set_track(self, track: Track) -> None:
        if not track.is_valid:
            raise ValueError("GPUTrainingEnv requires a valid generated track")
        if len(track.centerline) < 2:
            raise ValueError("track must contain at least two centerline points")
        self.track = track
        numpy_dtype = np.float64 if self.dtype == torch.float64 else np.float32
        centerline = self._to_tensor(np.asarray(track.centerline, dtype=numpy_dtype))
        self.centerline_starts = centerline[:-1].contiguous()
        self.centerline_segments = (centerline[1:] - centerline[:-1]).contiguous()
        self.centerline_segment_length_sq = (
            self.centerline_segments.square().sum(dim=1)
        )
        checkpoints = track.checkpoints
        self.checkpoint_positions = self._to_tensor(
            np.asarray([cp.position for cp in checkpoints], dtype=numpy_dtype)
            if checkpoints else np.zeros((0, 2), dtype=numpy_dtype)
        )
        self.checkpoint_directions = self._to_tensor(
            np.asarray([cp.direction for cp in checkpoints], dtype=numpy_dtype)
            if checkpoints else np.zeros((0, 2), dtype=numpy_dtype)
        )
        self.checkpoint_widths = self._to_tensor(
            np.asarray([cp.width for cp in checkpoints], dtype=numpy_dtype)
            if checkpoints else np.zeros((0,), dtype=numpy_dtype)
        )
        racing_line = track.racing_line
        self.racing_line_positions = self._to_tensor(
            np.asarray([p.position for p in racing_line], dtype=numpy_dtype)
            if racing_line else np.zeros((0, 2), dtype=numpy_dtype)
        )
        self.racing_line_directions = self._to_tensor(
            np.asarray([p.direction for p in racing_line], dtype=numpy_dtype)
            if racing_line else np.zeros((0, 2), dtype=numpy_dtype)
        )
        self.racing_line_target_speeds = self._to_tensor(
            np.asarray([p.target_speed for p in racing_line], dtype=numpy_dtype)
            if racing_line else np.zeros((0,), dtype=numpy_dtype)
        )
        self.racing_line_curvatures = self._to_tensor(
            np.asarray([p.curvature for p in racing_line], dtype=numpy_dtype)
            if racing_line else np.zeros((0,), dtype=numpy_dtype)
        )
        self.num_laps = int(track.num_laps)
        spawn_position, spawn_angle = track.get_spawn_position(0)
        self.spawn_position = self._to_tensor(spawn_position).reshape(2)
        self.spawn_angle = float(spawn_angle)
        self.surface_grip_value = float(
            self.vehicle_config.surface_grip_modifiers.get(track.theme, 1.0)
        )
        # Track geometry is immutable during an episode and is uploaded only
        # when a new track is selected, outside the step hot path.

    def set_track_from_env(self, reference_env: Any) -> None:
        """Replace immutable geometry from a reset CPU reference env.

        Intended for changing evaluation seeds/tracks between rollouts. Track
        generation and host-to-device upload happen here, never in step().
        """
        if reference_env.track is None:
            raise RuntimeError("reference_env must be reset before changing the GPU track")
        if reference_env.vehicle is not None:
            self.vehicle_config = reference_env.vehicle.config
            self.params = VehicleParameters.from_config(self.vehicle_config)
        self._set_track(reference_env.track)
        self.surface_grip.fill_(self.surface_grip_value)

    @torch.no_grad()
    def reset(self, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Reset every slot, or only selected slots, and return initial observations.

        A Boolean mask selects slots. Integer input is interpreted as indices.
        """
        if mask is None:
            selected = torch.ones(self.batch_size, dtype=torch.bool, device=self.device)
        else:
            raw = torch.as_tensor(mask, device=self.device)
            if raw.dtype == torch.bool:
                if raw.shape != (self.batch_size,):
                    raise ValueError("Boolean reset mask must have shape [batch_size]")
                selected = raw
            else:
                selected = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
                selected[raw.to(torch.long).reshape(-1)] = True

        self.position[selected] = self.spawn_position
        self.angle[selected] = self.spawn_angle
        self.velocity[selected] = 0.0
        self.angular_velocity[selected] = 0.0
        self.force[selected] = 0.0
        self.torque[selected] = 0.0
        self.speed[selected] = 0.0
        self.forward_speed[selected] = 0.0
        self.lateral_speed[selected] = 0.0
        self.slip_angle[selected] = 0.0
        self.is_drifting[selected] = False
        self.drift_intensity[selected] = 0.0
        self.on_road[selected] = True
        self.surface_grip[selected] = self.surface_grip_value
        self.throttle[selected] = 0.0
        self.brake[selected] = 0.0
        self.steering[selected] = 0.0
        self.boost_active[selected] = False
        self.boost_timer[selected] = 0.0
        self.boost_cooldown_timer[selected] = 0.0

        self.distance_traveled[selected] = 0.0
        self.max_speed_reached[selected] = 0.0
        self.vehicle_offroad_time[selected] = 0.0
        self.vehicle_collision_count[selected] = 0
        self.collision_count[selected] = 0
        self.collision_events[selected] = 0
        self.episode_steps[selected] = 0
        self.elapsed_time[selected] = 0.0
        self.episode_reward[selected] = 0.0
        self.last_checkpoint[selected] = self.checkpoint_positions.shape[0] - 1
        self.checkpoints_crossed[selected] = 0
        self.laps_completed[selected] = 0
        self.wrong_way_time[selected] = 0.0
        self.idle_time[selected] = 0.0
        self.offroad_time[selected] = 0.0
        self.total_offroad_time[selected] = 0.0
        self.last_position[selected] = self.spawn_position
        self.current_lap_time[selected] = 0.0
        self.best_lap_time[selected] = float("inf")
        self.lap_history[selected] = 0.0
        self.terminated[selected] = False
        self.truncated[selected] = False
        return build_observations(self)

    @torch.no_grad()
    def step(
        self, actions: Union[torch.Tensor, np.ndarray, list, tuple]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Advance every active slot using batched operations only.

        No per-agent Python loops and no .item(), .cpu(), or .numpy() calls are
        used in this hot path. Pass CUDA action tensors for a fully device-local
        policy/environment pipeline.
        """
        if self.profile_timing:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            self._profile_last = time.perf_counter()
            self.last_step_timings = {}

        active = self.active
        action = torch.as_tensor(actions, dtype=self.dtype, device=self.device)
        if self.train_config.action_mode == ActionMode.CONTINUOUS:
            if action.ndim == 1 and self.batch_size == 1:
                action = action.unsqueeze(0)
            if action.shape != (self.batch_size, 3):
                raise ValueError(
                    f"continuous actions must have shape [{self.batch_size}, 3], got {tuple(action.shape)}"
                )
            new_steering = action[:, 0].clamp(-1.0, 1.0)
            new_throttle = action[:, 1].clamp(0.0, 1.0)
            new_brake = action[:, 2].clamp(0.0, 1.0)
        else:
            action = action.reshape(-1).to(torch.long)
            if action.shape != (self.batch_size,):
                raise ValueError(f"discrete actions must have shape [{self.batch_size}]")
            # Unrecognized integer values map to coast, as in RacingEnv.step().
            new_steering = torch.where(
                action == 0, -torch.ones_like(self.steering),
                torch.where(action == 1, torch.ones_like(self.steering), torch.zeros_like(self.steering)),
            )
            new_throttle = (action == 2).to(self.dtype)
            new_brake = (action == 3).to(self.dtype)

        self.steering.copy_(torch.where(active, new_steering, self.steering))
        self.throttle.copy_(torch.where(active, new_throttle, self.throttle))
        self.brake.copy_(torch.where(active, new_brake, self.brake))
        previous_position = self.last_position.clone()

        integrate_rigid_body(
            self.position,
            self.angle,
            self.velocity,
            self.angular_velocity,
            self.force,
            self.torque,
            active,
            dt=self.dt,
            substeps=self.physics_substeps,
            inverse_mass=self.params.inverse_mass,
            inverse_inertia=self.params.inverse_inertia,
        )
        self._mark_profile("rigid_body_integration_seconds")

        # RacingEnv checks road membership after integrating, using nearest
        # centerline segment distance and 1.1 * half_road_width as threshold.
        road_distance = nearest_centerline_distance(
            self.position,
            self.centerline_starts,
            self.centerline_segments,
            self.centerline_segment_length_sq,
            chunk_size=self.segment_chunk_size,
        )
        calculated_on_road = road_distance <= (float(self.track.road_width) * 0.5 * 1.1)
        self.on_road.copy_(torch.where(active, calculated_on_road, self.on_road))
        self._mark_profile("road_boundary_seconds")

        forward = torch.stack((torch.cos(self.angle), torch.sin(self.angle)), dim=1)
        right = torch.stack((-forward[:, 1], forward[:, 0]), dim=1)
        new_speed = torch.linalg.vector_norm(self.velocity, dim=1)
        new_forward_speed = (self.velocity * forward).sum(dim=1)
        new_lateral_speed = (self.velocity * right).sum(dim=1)
        self.speed.copy_(torch.where(active, new_speed, self.speed))
        self.forward_speed.copy_(torch.where(active, new_forward_speed, self.forward_speed))
        self.lateral_speed.copy_(torch.where(active, new_lateral_speed, self.lateral_speed))
        self.distance_traveled.add_(self.speed * active.to(self.dtype) * self.dt)
        self.max_speed_reached.copy_(torch.maximum(self.max_speed_reached, self.speed))

        self.vehicle_offroad_time.add_(
            ((~self.on_road) & active).to(self.dtype) * self.dt
        )
        next_boost_timer = self.boost_timer - self.dt
        self.boost_timer.copy_(
            torch.where(active & self.boost_active, next_boost_timer, self.boost_timer)
        )
        self.boost_active.copy_(
            torch.where(
                active,
                self.boost_active & (self.boost_timer > 0.0),
                self.boost_active,
            )
        )
        next_cooldown = self.boost_cooldown_timer - self.dt
        self.boost_cooldown_timer.copy_(
            torch.where(active & (self.boost_cooldown_timer > 0.0), next_cooldown, self.boost_cooldown_timer)
        )

        new_force, new_torque = calculate_vehicle_forces(
            self.speed,
            self.forward_speed,
            self.angle,
            self.velocity,
            self.angular_velocity,
            self.throttle,
            self.brake,
            self.steering,
            self.boost_active,
            self.slip_angle,
            self.is_drifting,
            self.drift_intensity,
            self.surface_grip,
            active,
            self.params,
        )
        self.force.copy_(new_force)
        self.torque.copy_(new_torque)
        self._mark_profile("vehicle_state_and_force_seconds")

        calculated_slip = torch.where(
            self.forward_speed.abs() > 0.5,
            torch.atan2(self.lateral_speed, self.forward_speed.abs()),
            torch.zeros_like(self.slip_angle),
        )
        self.slip_angle.copy_(torch.where(active, calculated_slip, self.slip_angle))
        optimal_slip = self.params.optimal_slip_angle * self.surface_grip
        drifting_now = self.slip_angle.abs() > (optimal_slip * 1.5)
        new_drift_intensity = torch.where(
            drifting_now,
            (self.slip_angle.abs() / (optimal_slip * 3.0).clamp_min(1e-12)).clamp(max=1.0),
            self.drift_intensity * 0.95,
        )
        self.drift_intensity.copy_(torch.where(active, new_drift_intensity, self.drift_intensity))
        self.is_drifting.copy_(torch.where(active, drifting_now, self.is_drifting))

        rewards = calculate_rewards(self, active, previous_position)
        self.episode_reward.add_(rewards)
        self._mark_profile("reward_seconds")

        newly_terminated, newly_truncated = update_termination(self, active)
        self.terminated.logical_or_(newly_terminated)
        self.truncated.logical_or_(newly_truncated)

        # RacingEnv advances time/counters after termination checking and adds dt
        # to the current lap timer even on the step that crosses the finish line.
        active_f = active.to(self.dtype)
        self.episode_steps.add_(active.to(torch.long))
        self.elapsed_time.add_(active_f * self.dt)
        self.current_lap_time.add_(active_f * self.dt)
        self._mark_profile("termination_and_episode_management_seconds")
        observations = build_observations(self)
        self._mark_profile("observation_seconds")
        if self.profile_timing:
            self.last_step_timings["total_profiled_seconds"] = sum(self.last_step_timings.values())

        return (
            observations,
            rewards,
            newly_terminated,
            newly_truncated,
            self._info(),
        )

    def _mark_profile(self, name: str) -> None:
        """Record one opt-in profiled stage; synchronizes only in benchmark mode."""
        if not self.profile_timing:
            return
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        now = time.perf_counter()
        self.last_step_timings[name] = now - self._profile_last
        self._profile_last = now

    def _info(self) -> dict[str, torch.Tensor]:
        """Tensor-only metrics; converting to host values is left to callers."""
        return {
            "episode_reward": self.episode_reward,
            "episode_steps": self.episode_steps,
            "elapsed_time": self.elapsed_time,
            "laps_completed": self.laps_completed,
            "checkpoints_crossed": self.checkpoints_crossed,
            "collision_count": self.collision_count,
            "offroad_time": self.total_offroad_time,
            "distance_traveled": self.distance_traveled,
            "max_speed": self.max_speed_reached,
            "best_lap_time": self.best_lap_time,
            "active": self.active,
        }

    def close(self) -> None:
        """No external resources are owned by the tensor simulator."""
        return None
