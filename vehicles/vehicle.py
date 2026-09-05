"""
Vehicle Physics

2D top-down vehicle physics with realistic but controllable handling.
Deterministic simulation with fixed timestep.
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass, field
from core.math import (
    Vec2, vec2, vec2_zero, vec2_add, vec2_sub, vec2_mul, vec2_dot,
    vec2_length, vec2_normalize, vec2_rotate, vec2_angle, vec2_perp,
    clamp, lerp
)
from core.physics import RigidBody, PhysicsMaterial
from core.random import DeterministicRNG


@dataclass
class VehicleConfig:
    """Vehicle configuration parameters."""
    # Performance
    top_speed: float = 45.0          # m/s
    acceleration: float = 12.0       # m/s^2
    brake_force: float = 18.0        # m/s^2
    reverse_acceleration: float = 5.0  # m/s^2
    
    # Handling
    steering_sensitivity: float = 3.5  # rad/s at max speed
    steering_lock: float = 0.6       # max steering angle (radians)
    grip: float = 1.0                # Lateral grip multiplier
    mass: float = 1200.0             # kg
    width: float = 2.0               # meters
    length: float = 4.5              # meters
    
    # Drift/slide
    drift_factor: float = 0.3        # How easily car slides
    drift_recovery: float = 2.0      # How fast drift recovers
    
    # Aerodynamics
    drag_coefficient: float = 0.32
    downforce: float = 0.0           # N at top speed
    
    # Tire model
    tire_stiffness: float = 1.0
    optimal_slip_angle: float = 0.08  # radians (~4.5 degrees)
    peak_lateral_force: float = 1.0  # Multiplier
    
    # Differential
    diff_lock: float = 0.0           # 0 = open, 1 = locked
    
    # Boost/nitro
    has_boost: bool = False
    boost_power: float = 1.5         # Multiplier
    boost_duration: float = 3.0      # seconds
    boost_cooldown: float = 10.0     # seconds
    
    # Surface modifiers (multipliers)
    surface_grip_modifiers: Dict[str, float] = field(default_factory=lambda: {
        "asphalt": 1.0,
        "desert": 0.85,
        "snow": 0.4,
        "grassland": 0.9,
        "neon": 1.1,
        "industrial": 1.0,
        "dirt": 0.7,
        "grass": 0.3,
        "sand": 0.25,
        "ice": 0.15
    })


@dataclass
class VehicleState:
    """Current vehicle state."""
    # Transform
    position: Vec2 = field(default_factory=vec2_zero)
    angle: float = 0.0
    
    # Velocity
    velocity: Vec2 = field(default_factory=vec2_zero)
    angular_velocity: float = 0.0
    
    # Local velocity (body frame)
    local_velocity: Vec2 = field(default_factory=vec2_zero)
    speed: float = 0.0
    forward_speed: float = 0.0
    lateral_speed: float = 0.0
    
    # Slip
    slip_angle: float = 0.0
    is_drifting: bool = False
    drift_intensity: float = 0.0
    
    # Surface
    surface_type: str = "asphalt"
    surface_grip: float = 1.0
    on_road: bool = True
    
    # Inputs
    throttle: float = 0.0
    brake: float = 0.0
    steering: float = 0.0
    boost_active: bool = False
    boost_timer: float = 0.0
    boost_cooldown_timer: float = 0.0
    
    # Physics body reference
    body_id: int = -1
    
    # Identity
    name: str = ""
    
    # Stats
    distance_traveled: float = 0.0
    max_speed_reached: float = 0.0
    collision_count: int = 0
    offroad_time: float = 0.0
    air_time: float = 0.0  # For future jumps


class Vehicle:
    """
    Vehicle with deterministic physics simulation.
    
    Features:
    - Pacejka-like tire model (simplified)
    - Weight transfer
    - Drift/slide dynamics
    - Surface-dependent grip
    - Boost/nitro system
    - Deterministic step function
    """
    
    def __init__(self, config: VehicleConfig, body: RigidBody, rng: DeterministicRNG):
        self.config = config
        self.body = body
        self.rng = rng
        self.state = VehicleState()
        self.state.body_id = body.id
        
        # Precompute some values
        self._max_steer_angle = config.steering_lock
        self._wheelbase = config.length * 0.85
        self._track_width = config.width * 0.9
        
        # For deterministic tire forces
        self._last_slip_angle = 0.0
    
    def reset(self, position: Vec2, angle: float) -> None:
        """Reset vehicle to initial state."""
        self.body.position = position.copy()
        self.body.angle = angle
        self.body.velocity = vec2_zero()
        self.body.angular_velocity = 0.0
        self.body.force = vec2_zero()
        self.body.torque = 0.0
        
        # Preserve name if it exists
        old_name = self.state.name if hasattr(self.state, 'name') else ""
        
        self.state = VehicleState()
        self.state.body_id = self.body.id
        self.state.position = position.copy()
        self.state.angle = angle
        self.state.name = old_name
        self._last_slip_angle = 0.0
    
    def set_input(self, throttle: float, brake: float, steering: float, boost: bool = False) -> None:
        """Set driver/AI inputs (clamped to valid ranges)."""
        self.state.throttle = clamp(throttle, 0.0, 1.0)
        self.state.brake = clamp(brake, 0.0, 1.0)
        self.state.steering = clamp(steering, -1.0, 1.0)
        
        # Handle boost
        if boost and self.config.has_boost and self.state.boost_cooldown_timer <= 0:
            self.state.boost_active = True
            self.state.boost_timer = self.config.boost_duration
            self.state.boost_cooldown_timer = self.config.boost_cooldown
    
    def step(self, dt: float, surface_type: str = "asphalt", on_road: bool = True) -> VehicleState:
        """
        Step vehicle physics by dt seconds.
        
        Returns updated state.
        """
        # Sync from physics body
        self.state.position = self.body.position.copy()
        self.state.angle = self.body.angle
        self.state.velocity = self.body.velocity.copy()
        self.state.angular_velocity = self.body.angular_velocity
        
        # Calculate local velocity
        forward = vec2(np.cos(self.body.angle), np.sin(self.body.angle))
        right = vec2(-forward[1], forward[0])
        
        self.state.forward_speed = vec2_dot(self.body.velocity, forward)
        self.state.lateral_speed = vec2_dot(self.body.velocity, right)
        self.state.local_velocity = vec2(self.state.forward_speed, self.state.lateral_speed)
        self.state.speed = vec2_length(self.body.velocity)
        
        # Update surface info
        self.state.surface_type = surface_type
        self.state.on_road = on_road
        self.state.surface_grip = self.config.surface_grip_modifiers.get(surface_type, 1.0)
        
        if not on_road:
            self.state.offroad_time += dt
        
        # Update stats
        self.state.distance_traveled += self.state.speed * dt
        self.state.max_speed_reached = max(self.state.max_speed_reached, self.state.speed)
        
        # Handle boost
        if self.state.boost_active:
            self.state.boost_timer -= dt
            if self.state.boost_timer <= 0:
                self.state.boost_active = False
        
        if self.state.boost_cooldown_timer > 0:
            self.state.boost_cooldown_timer -= dt
        
        # Calculate forces
        self._calculate_forces(dt)
        
        # Calculate slip angle
        if abs(self.state.forward_speed) > 0.5:
            self.state.slip_angle = math.atan2(self.state.lateral_speed, abs(self.state.forward_speed))
        else:
            self.state.slip_angle = 0.0
        
        # Drift detection
        optimal_slip = self.config.optimal_slip_angle * self.state.surface_grip
        if abs(self.state.slip_angle) > optimal_slip * 1.5:
            self.state.is_drifting = True
            self.state.drift_intensity = min(1.0, abs(self.state.slip_angle) / (optimal_slip * 3))
        else:
            self.state.is_drifting = False
            self.state.drift_intensity *= 0.95  # Fade out
        
        self._last_slip_angle = self.state.slip_angle
        
        return self.state
    
    def _calculate_forces(self, dt: float) -> None:
        """Calculate and apply all forces to the physics body."""
        forward = vec2(np.cos(self.body.angle), np.sin(self.body.angle))
        right = vec2(-forward[1], forward[0])
        
        # Current speed
        speed = self.state.speed
        forward_speed = self.state.forward_speed
        
        # Get effective grip
        effective_grip = self.config.grip * self.state.surface_grip
        
        # --- LONGITUDINAL FORCES ---
        
        # Engine force
        engine_force = 0.0
        if self.state.throttle > 0.01:
            # Torque curve: peak at mid RPM, drops at high speed
            speed_ratio = speed / self.config.top_speed
            torque_curve = 1.0 - speed_ratio * 0.5  # Linear drop
            torque_curve = max(0.3, torque_curve)  # Minimum torque
            
            boost_mult = self.config.boost_power if self.state.boost_active else 1.0
            engine_force = self.config.acceleration * self.config.mass * self.state.throttle * torque_curve * boost_mult
        
        # Brake force
        brake_force = 0.0
        if self.state.brake > 0.01:
            brake_force = self.config.brake_force * self.config.mass * self.state.brake
            # Brake force opposes velocity
            if forward_speed > 0:
                brake_force = -brake_force
            elif forward_speed < 0:
                brake_force = brake_force
        
        # Reverse
        reverse_force = 0.0
        if self.state.throttle > 0.01 and forward_speed < 1.0:
            # Allow reverse when moving slowly forward or stopped
            reverse_force = -self.config.reverse_acceleration * self.config.mass * self.state.throttle
        
        # Drag force (quadratic)
        drag_force = -0.5 * 1.225 * self.config.drag_coefficient * (self.config.width * self.config.length * 0.8) * speed * speed
        if speed > 0:
            drag_force = -abs(drag_force)
        
        # Rolling resistance
        rolling_resistance = -self.config.mass * 9.81 * 0.015 * (1.0 if forward_speed >= 0 else -1.0)
        if abs(forward_speed) < 0.5:
            rolling_resistance = 0.0
        
        # Total longitudinal force
        total_longitudinal = engine_force + brake_force + reverse_force + drag_force + rolling_resistance
        
        # Apply longitudinal force
        long_force_vec = vec2_mul(forward, total_longitudinal)
        self.body.apply_force(long_force_vec)
        
        # --- LATERAL FORCES (Cornering) ---
        
        if abs(forward_speed) > 0.1:
            # Slip angle
            slip = self.state.slip_angle
            
            # Pacejka-like lateral force (simplified)
            # Fy = D * sin(C * arctan(B * slip - E * (B * slip - arctan(B * slip))))
            # Simplified: linear up to optimal, then drop
            optimal_slip = self.config.optimal_slip_angle * effective_grip
            peak_force = self.config.peak_lateral_force * effective_grip * self.config.mass * 9.81
            
            if abs(slip) < optimal_slip:
                # Linear region
                lateral_force_mag = peak_force * (abs(slip) / optimal_slip)
            else:
                # Past peak - drop off
                excess = abs(slip) - optimal_slip
                drop = min(1.0, excess / (optimal_slip * 2))
                lateral_force_mag = peak_force * (1.0 - drop * 0.5)
            
            # Direction opposes slip
            lateral_force_mag *= -1.0 if slip > 0 else 1.0
            
            # Weight transfer effect (simplified)
            # More load on outside tires during cornering
            lat_accel = lateral_force_mag / self.config.mass
            weight_transfer = lat_accel * self.config.mass * 0.3 * self._track_width / 2.0
            
            # Apply lateral force at center of mass
            lat_force_vec = vec2_mul(right, lateral_force_mag)
            self.body.apply_force(lat_force_vec)
            
            # Yaw torque from lateral force (pneumatic trail)
            # Simplified: force applied slightly behind center
            pneumatic_trail = self.config.length * 0.1
            yaw_torque = lateral_force_mag * pneumatic_trail * (1.0 if slip > 0 else -1.0)
            self.body.torque += yaw_torque
        
        # --- STEERING TORQUE ---
        
        # Driver steering input creates yaw torque
        target_steer = self.state.steering * self._max_steer_angle
        
        # Speed-sensitive steering
        speed_factor = clamp(1.0 - speed / self.config.top_speed, 0.3, 1.0)
        target_steer *= speed_factor
        
        # Steering torque (proportional to lateral force potential)
        max_steer_torque = self.config.mass * self.config.steering_sensitivity * effective_grip
        steer_torque = target_steer * max_steer_torque
        
        # Damping
        steer_damping = -self.body.angular_velocity * self.config.mass * 0.5
        
        self.body.torque += steer_torque + steer_damping
        
        # --- DRIFT RECOVERY ---
        
        if self.state.is_drifting:
            # Counter-steer assistance
            counter_steer = -np.sign(self.state.slip_angle) * self.state.drift_intensity * 0.5
            drift_recovery_torque = counter_steer * max_steer_torque * self.config.drift_recovery
            self.body.torque += drift_recovery_torque
        
        # --- DOWNFORCE ---
        
        if self.config.downforce > 0 and speed > 0:
            downforce = self.config.downforce * (speed / self.config.top_speed) ** 2
            # Downforce increases grip but doesn't affect lateral force directly in this model
            # It's factored into effective_grip via surface_grip
            pass
    
    def get_observation(self) -> Dict[str, Any]:
        """Get observation dictionary for AI."""
        forward = vec2(np.cos(self.body.angle), np.sin(self.body.angle))
        right = vec2(-forward[1], forward[0])
        
        return {
            'position': self.state.position.copy(),
            'angle': self.body.angle,
            'velocity': self.body.velocity.copy(),
            'angular_velocity': self.body.angular_velocity,
            'speed': self.state.speed,
            'forward_speed': self.state.forward_speed,
            'lateral_speed': self.state.lateral_speed,
            'slip_angle': self.state.slip_angle,
            'is_drifting': self.state.is_drifting,
            'drift_intensity': self.state.drift_intensity,
            'on_road': self.state.on_road,
            'surface_type': self.state.surface_type,
            'surface_grip': self.state.surface_grip,
            'throttle': self.state.throttle,
            'brake': self.state.brake,
            'steering': self.state.steering,
            'boost_active': self.state.boost_active,
            'boost_available': self.state.boost_cooldown_timer <= 0,
            'forward_vector': forward.copy(),
            'right_vector': right.copy()
        }
    
    def apply_collision_impulse(self, impulse: Vec2, point: Vec2) -> None:
        """Apply collision impulse from physics engine."""
        self.body.apply_impulse(impulse, point)
        self.state.collision_count += 1


import math