"""Batched tensor vehicle physics mirroring the current Vehicle/RigidBody order.

The original environment first integrates accumulated forces in PhysicsWorld.step
and only afterwards computes the next force in Vehicle.step. This module keeps
that one-step force delay and semi-implicit Euler integration. It does not
pretend track boundaries are rigid collision bodies: the reference training
environment models road departure with its centerline-distance rule.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class VehicleParameters:
    top_speed: float
    acceleration: float
    brake_force: float
    reverse_acceleration: float
    steering_sensitivity: float
    steering_lock: float
    grip: float
    mass: float
    width: float
    length: float
    drag_coefficient: float
    tire_stiffness: float
    optimal_slip_angle: float
    peak_lateral_force: float
    drift_recovery: float
    boost_power: float
    has_boost: bool
    boost_duration: float
    boost_cooldown: float

    @classmethod
    def from_config(cls, config: Any) -> "VehicleParameters":
        return cls(
            top_speed=float(config.top_speed),
            acceleration=float(config.acceleration),
            brake_force=float(config.brake_force),
            reverse_acceleration=float(config.reverse_acceleration),
            steering_sensitivity=float(config.steering_sensitivity),
            steering_lock=float(config.steering_lock),
            grip=float(config.grip),
            mass=float(config.mass),
            width=float(config.width),
            length=float(config.length),
            drag_coefficient=float(config.drag_coefficient),
            tire_stiffness=float(config.tire_stiffness),
            optimal_slip_angle=float(config.optimal_slip_angle),
            peak_lateral_force=float(config.peak_lateral_force),
            drift_recovery=float(config.drift_recovery),
            boost_power=float(config.boost_power),
            has_boost=bool(config.has_boost),
            boost_duration=float(config.boost_duration),
            boost_cooldown=float(config.boost_cooldown),
        )

    @property
    def inverse_mass(self) -> float:
        return 1.0 / self.mass if self.mass > 0.0 else 0.0

    @property
    def inverse_inertia(self) -> float:
        inertia = self.mass * (self.width ** 2 + self.length ** 2) / 12.0
        return 1.0 / inertia if inertia > 0.0 else 0.0

    @property
    def wheelbase(self) -> float:
        return self.length * 0.85

    @property
    def track_width(self) -> float:
        return self.width * 0.9


@torch.no_grad()
def integrate_rigid_body(
    position: torch.Tensor,
    angle: torch.Tensor,
    velocity: torch.Tensor,
    angular_velocity: torch.Tensor,
    force: torch.Tensor,
    torque: torch.Tensor,
    active: torch.Tensor,
    *,
    dt: float,
    substeps: int,
    inverse_mass: float,
    inverse_inertia: float,
) -> None:
    """Integrate persistent batched rigid-body state in-place.

    Existing forces are cleared after the first substep, matching RigidBody's
    integrate() implementation. The active mask freezes completed episodes.
    """
    if substeps < 1:
        raise ValueError("substeps must be >= 1")
    sub_dt = float(dt) / int(substeps)
    active_f = active.to(dtype=position.dtype).unsqueeze(1)
    active_1d = active.to(dtype=position.dtype)
    for substep in range(substeps):
        velocity.add_(force * (inverse_mass * sub_dt) * active_f)
        position.add_(velocity * sub_dt * active_f)
        angular_velocity.add_(torque * (inverse_inertia * sub_dt) * active_1d)
        angle.add_(angular_velocity * sub_dt * active_1d)
        # Same normalized range as RigidBody.integrate(), without data-dependent
        # Python loops or host synchronization.
        angle.copy_(torch.remainder(angle + 3.141592653589793, 6.283185307179586) - 3.141592653589793)
        if substep == 0:
            force.zero_()
            torque.zero_()


@torch.no_grad()
def calculate_vehicle_forces(
    speed: torch.Tensor,
    forward_speed: torch.Tensor,
    angle: torch.Tensor,
    velocity: torch.Tensor,
    angular_velocity: torch.Tensor,
    throttle: torch.Tensor,
    brake: torch.Tensor,
    steering: torch.Tensor,
    boost_active: torch.Tensor,
    slip_angle: torch.Tensor,
    is_drifting: torch.Tensor,
    drift_intensity: torch.Tensor,
    surface_grip: torch.Tensor,
    active: torch.Tensor,
    params: VehicleParameters,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the same longitudinal, lateral, steering and drift torques."""
    dtype = velocity.dtype
    forward = torch.stack((torch.cos(angle), torch.sin(angle)), dim=1)
    right = torch.stack((-forward[:, 1], forward[:, 0]), dim=1)
    effective_grip = params.grip * surface_grip

    throttle_on = throttle > 0.01
    speed_ratio = speed / max(params.top_speed, 1e-12)
    torque_curve = (1.0 - speed_ratio * 0.5).clamp_min(0.3)
    boost_multiplier = torch.where(
        boost_active,
        torch.full_like(speed, params.boost_power),
        torch.ones_like(speed),
    )
    engine_force = (
        params.acceleration * params.mass * throttle * torque_curve * boost_multiplier
    )
    engine_force = torch.where(throttle_on, engine_force, torch.zeros_like(engine_force))

    raw_brake = params.brake_force * params.mass * brake
    brake_force = torch.where(
        forward_speed > 0.0, -raw_brake,
        torch.where(forward_speed < 0.0, raw_brake, raw_brake),
    )
    brake_force = torch.where(brake > 0.01, brake_force, torch.zeros_like(brake_force))

    reverse_force = -params.reverse_acceleration * params.mass * throttle
    reverse_force = torch.where(
        throttle_on & (forward_speed < 1.0),
        reverse_force,
        torch.zeros_like(reverse_force),
    )

    drag_force = (
        -0.5 * 1.225 * params.drag_coefficient
        * (params.width * params.length * 0.8) * speed.square()
    )
    drag_force = torch.where(speed > 0.0, -drag_force.abs(), torch.zeros_like(drag_force))
    rolling_resistance = (
        -params.mass * 9.81 * 0.015
        * torch.where(forward_speed >= 0.0, 1.0, -1.0)
    )
    rolling_resistance = torch.where(
        forward_speed.abs() < 0.5,
        torch.zeros_like(rolling_resistance),
        rolling_resistance,
    )
    total_longitudinal = engine_force + brake_force + reverse_force + drag_force + rolling_resistance
    force = forward * total_longitudinal[:, None]

    moving = forward_speed.abs() > 0.1
    optimal_slip = params.optimal_slip_angle * effective_grip
    safe_optimal = optimal_slip.clamp_min(1e-12)
    peak_force = params.peak_lateral_force * effective_grip * params.mass * 9.81
    absolute_slip = slip_angle.abs()
    linear_force = peak_force * (absolute_slip / safe_optimal)
    excess = (absolute_slip - optimal_slip).clamp_min(0.0)
    drop = (excess / (safe_optimal * 2.0)).clamp(0.0, 1.0)
    post_peak_force = peak_force * (1.0 - drop * 0.5)
    lateral_magnitude = torch.where(absolute_slip < optimal_slip, linear_force, post_peak_force)
    lateral_force = torch.where(slip_angle > 0.0, -lateral_magnitude, lateral_magnitude)
    lateral_force = torch.where(moving, lateral_force, torch.zeros_like(lateral_force))
    force = force + right * lateral_force[:, None]

    # Vehicle._calculate_forces computes this value but does not apply it.
    _weight_transfer = (
        (lateral_force / params.mass) * params.mass * 0.3 * params.track_width / 2.0
    )
    torque = lateral_force * (params.length * 0.1) * torch.where(
        slip_angle > 0.0, 1.0, -1.0
    )
    torque = torch.where(moving, torque, torch.zeros_like(torque))

    speed_factor = (1.0 - speed / max(params.top_speed, 1e-12)).clamp(0.3, 1.0)
    target_steer = steering * params.steering_lock * speed_factor
    max_steer_torque = params.mass * params.steering_sensitivity * effective_grip
    steer_torque = target_steer * max_steer_torque
    steer_damping = -angular_velocity * params.mass * 0.5
    torque = torque + steer_torque + steer_damping

    counter_steer = -torch.sign(slip_angle) * drift_intensity * 0.5
    recovery_torque = counter_steer * max_steer_torque * params.drift_recovery
    torque = torque + torch.where(is_drifting, recovery_torque, torch.zeros_like(recovery_torque))

    active_f = active.to(dtype=dtype)
    force = force * active_f[:, None]
    torque = torque * active_f
    return force, torque
