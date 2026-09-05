"""
Scripted NPC Opponent AI

Baseline AI for racing opponents with multiple difficulty profiles.
Deterministic, no machine learning required.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

from core.random import DeterministicRNG
from core.math import (
    Vec2, vec2, vec2_add, vec2_sub, vec2_mul, vec2_dot, vec2_length,
    vec2_normalize, vec2_angle, vec2_perp, angle_difference, clamp,
    vec2_distance
)
from track.track import Track, Checkpoint
from vehicles.vehicle import Vehicle, VehicleState
from core.physics import PhysicsWorld


class AIProfile(Enum):
    BEGINNER = "beginner"
    NORMAL = "normal"
    EXPERT = "expert"
    AGGRESSIVE = "aggressive"
    DEFENSIVE = "defensive"
    RANDOMIZED = "randomized"


@dataclass
class AIProfileConfig:
    """Configuration for an AI driving profile."""
    target_speed_factor: float = 0.75      # Fraction of max speed to target
    braking_distance_factor: float = 1.3   # How early to brake (multiplier)
    reaction_time: float = 0.15            # Seconds of input delay
    overtake_threshold: float = 0.7        # Required advantage to attempt overtake
    aggression: float = 0.4                # General aggression level
    error_rate: float = 0.05               # Random mistake probability
    corner_speed_factor: float = 0.9       # Speed in corners relative to optimal
    follow_distance: float = 15.0          # Preferred distance to car ahead
    defense_tendency: float = 0.5          # How much to defend position
    recovery_speed: float = 1.0            # How fast to recover from mistakes


# Default profiles
AI_PROFILES = {
    AIProfile.BEGINNER: AIProfileConfig(
        target_speed_factor=0.5,
        braking_distance_factor=1.8,
        reaction_time=0.3,
        overtake_threshold=0.9,
        aggression=0.1,
        error_rate=0.15,
        corner_speed_factor=0.7,
        follow_distance=25.0,
        defense_tendency=0.2,
        recovery_speed=0.7
    ),
    AIProfile.NORMAL: AIProfileConfig(
        target_speed_factor=0.75,
        braking_distance_factor=1.3,
        reaction_time=0.15,
        overtake_threshold=0.7,
        aggression=0.4,
        error_rate=0.05,
        corner_speed_factor=0.85,
        follow_distance=15.0,
        defense_tendency=0.5,
        recovery_speed=1.0
    ),
    AIProfile.EXPERT: AIProfileConfig(
        target_speed_factor=0.95,
        braking_distance_factor=1.0,
        reaction_time=0.05,
        overtake_threshold=0.5,
        aggression=0.7,
        error_rate=0.01,
        corner_speed_factor=1.0,
        follow_distance=10.0,
        defense_tendency=0.6,
        recovery_speed=1.3
    ),
    AIProfile.AGGRESSIVE: AIProfileConfig(
        target_speed_factor=1.0,
        braking_distance_factor=0.8,
        reaction_time=0.02,
        overtake_threshold=0.3,
        aggression=0.9,
        error_rate=0.02,
        corner_speed_factor=0.95,
        follow_distance=8.0,
        defense_tendency=0.8,
        recovery_speed=1.2
    ),
    AIProfile.DEFENSIVE: AIProfileConfig(
        target_speed_factor=0.8,
        braking_distance_factor=1.5,
        reaction_time=0.1,
        overtake_threshold=0.8,
        aggression=0.2,
        error_rate=0.03,
        corner_speed_factor=0.85,
        follow_distance=20.0,
        defense_tendency=0.9,
        recovery_speed=1.1
    )
}


@dataclass
class AIDecision:
    """AI driving decision for one frame."""
    steering: float = 0.0      # -1 to 1
    throttle: float = 0.0      # 0 to 1
    brake: float = 0.0         # 0 to 1
    boost: bool = False
    target_waypoint: Optional[Vec2] = None
    target_speed: float = 0.0


class NPCAI:
    """
    Scripted NPC AI controller.
    
    Uses racing line following with:
    - Lookahead steering (pure pursuit)
    - Speed planning based on curvature
    - Obstacle/car avoidance
    - Overtaking logic
    - Defensive driving
    - Mistake simulation
    - Deterministic behavior from seed
    """
    
    def __init__(self, vehicle: Vehicle, track: Track, profile: AIProfile,
                 rng: DeterministicRNG, physics_world: PhysicsWorld):
        self.vehicle = vehicle
        self.track = track
        self.profile_config = AI_PROFILES[profile]
        self.rng = rng
        self.physics_world = physics_world
        self.profile = profile
        
        # For randomized profile, generate personality
        if profile == AIProfile.RANDOMIZED:
            self._randomize_personality()
        
        # State
        self.current_target_idx = 0
        self.reaction_timer = 0.0
        self.last_decision = AIDecision()
        self.stuck_timer = 0.0
        self.recovery_mode = False
        self.recovery_timer = 0.0
        self.overtake_target: Optional[Vehicle] = None
        self.overtake_side: int = 0  # -1 left, 1 right
        self.overtake_timer = 0.0
        self.defending = False
        self.defense_timer = 0.0
        
        # Sensor cache
        self._raycast_cache: List[Tuple[bool, float]] = []
    
    def _randomize_personality(self) -> None:
        """Generate random personality for randomized profile."""
        # Base on normal, then vary
        base = AI_PROFILES[AIProfile.NORMAL]
        self.profile_config = AIProfileConfig(
            target_speed_factor=clamp(base.target_speed_factor + self.rng.normal(0, 0.15), 0.3, 1.0),
            braking_distance_factor=clamp(base.braking_distance_factor + self.rng.normal(0, 0.2), 0.5, 2.5),
            reaction_time=clamp(base.reaction_time + abs(self.rng.normal(0, 0.05)), 0.0, 0.5),
            overtake_threshold=clamp(base.overtake_threshold + self.rng.normal(0, 0.15), 0.2, 1.0),
            aggression=clamp(base.aggression + self.rng.normal(0, 0.2), 0.0, 1.0),
            error_rate=clamp(base.error_rate + abs(self.rng.normal(0, 0.03)), 0.0, 0.3),
            corner_speed_factor=clamp(base.corner_speed_factor + self.rng.normal(0, 0.1), 0.5, 1.1),
            follow_distance=clamp(base.follow_distance + self.rng.normal(0, 5.0), 5.0, 40.0),
            defense_tendency=clamp(base.defense_tendency + self.rng.normal(0, 0.15), 0.0, 1.0),
            recovery_speed=clamp(base.recovery_speed + self.rng.normal(0, 0.15), 0.5, 1.5)
        )
    
    def update(self, dt: float, opponents: List[Vehicle]) -> AIDecision:
        """Update AI and return driving decision."""
        state = self.vehicle.state
        
        # Update timers
        self.reaction_timer += dt
        
        # Check if stuck
        if state.speed < 2.0:
            self.stuck_timer += dt
        else:
            self.stuck_timer = 0.0
        
        # Recovery mode
        if self.stuck_timer > 3.0 or self.recovery_mode:
            return self._recovery_behavior(dt)
        
        # Scan environment
        self._scan_environment(opponents)
        
        # Make decision (with reaction delay)
        if self.reaction_timer >= self.profile_config.reaction_time:
            self.reaction_timer = 0.0
            self.last_decision = self._make_decision(opponents)
        
        # Apply errors
        decision = self._apply_errors(self.last_decision)
        
        return decision
    
    def _scan_environment(self, opponents: List[Vehicle]) -> None:
        """Scan track ahead and detect nearby cars."""
        state = self.vehicle.state
        forward = vec2(np.cos(state.angle), np.sin(state.angle))
        right = vec2(-forward[1], forward[0])
        
        # Raycast sensors
        sensor_angles = [-1.2, -0.6, -0.3, -0.1, 0.0, 0.1, 0.3, 0.6, 1.2]
        self._raycast_cache = []
        
        for angle_offset in sensor_angles:
            ray_angle = state.angle + angle_offset
            direction = vec2(np.cos(ray_angle), np.sin(ray_angle))
            hit, dist, _, hit_id = self.physics_world.raycast(
                state.position, direction, 60.0, ignore_body=self.vehicle.body.id
            )
            self._raycast_cache.append((hit, dist, hit_id))
        
        # Find closest opponent ahead
        self.closest_ahead = None
        self.closest_ahead_dist = float('inf')
        self.closest_behind = None
        self.closest_behind_dist = float('inf')
        
        for opp in opponents:
            if opp is self.vehicle:
                continue
            to_opp = vec2_sub(opp.state.position, state.position)
            dist = vec2_length(to_opp)
            forward_dot = vec2_dot(vec2_normalize(to_opp) if dist > 0 else vec2(0, 0), forward)
            
            if forward_dot > 0.3 and dist < self.closest_ahead_dist:
                self.closest_ahead = opp
                self.closest_ahead_dist = dist
            elif forward_dot < -0.3 and dist < self.closest_behind_dist:
                self.closest_behind = opp
                self.closest_behind_dist = dist
    
    def _make_decision(self, opponents: List[Vehicle]) -> AIDecision:
        """Core decision making."""
        state = self.vehicle.state
        forward = vec2(np.cos(state.angle), np.sin(state.angle))
        right = vec2(-forward[1], forward[0])
        
        # Find target waypoint on racing line
        target_idx = self._find_target_waypoint()
        target_point = self.track.racing_line[target_idx].position
        target_speed = self.track.racing_line[target_idx].target_speed * self.profile_config.target_speed_factor
        target_speed *= self.profile_config.corner_speed_factor
        
        # Steering: pure pursuit
        to_target = vec2_sub(target_point, state.position)
        target_dist = vec2_length(to_target)
        
        if target_dist > 1.0:
            # Calculate steering to point
            target_dir = vec2_normalize(to_target)
            cross = forward[0] * target_dir[1] - forward[1] * target_dir[0]
            steering = clamp(cross * 2.0, -1.0, 1.0)
        else:
            steering = 0.0
        
        # Speed control
        speed_error = target_speed - state.forward_speed
        
        if speed_error > 1.0:
            throttle = 1.0
            brake = 0.0
        elif speed_error > -2.0:
            throttle = 0.3
            brake = 0.0
        else:
            throttle = 0.0
            brake = clamp(-speed_error / 10.0, 0.0, 1.0)
        
        # Obstacle avoidance from raycasts
        front_hit, front_dist, _ = self._raycast_cache[4]  # Center ray
        left_hit, left_dist, _ = self._raycast_cache[2]
        right_hit, right_dist, _ = self._raycast_cache[6]
        
        if front_hit and front_dist < 20.0:
            # Obstacle ahead - brake and steer around
            brake = max(brake, 0.7)
            if left_dist > right_dist:
                steering -= 0.5
            else:
                steering += 0.5
        
        # Car following / overtaking
        if self.closest_ahead and self.closest_ahead_dist < self.profile_config.follow_distance:
            # Check if we can overtake
            if self._should_overtake():
                return self._execute_overtake(state, forward, right, target_speed)
            else:
                # Follow at safe distance
                throttle = 0.0
                brake = max(brake, 0.4)
        
        # Defensive driving
        if self.closest_behind and self.closest_behind_dist < 15.0 and self.profile_config.defense_tendency > 0.5:
            self.defending = True
            self.defense_timer = 2.0
            # Move to inside line
            steering = self._defensive_line(steering, state, forward, right)
        
        if self.defense_timer > 0:
            self.defense_timer -= self.vehicle.state.dt if hasattr(self.vehicle.state, 'dt') else 1/60
            if self.defense_timer <= 0:
                self.defending = False
        
        # Corner anticipation - brake early for upcoming corners
        lookahead_idx = min(target_idx + 10, len(self.track.racing_line) - 1)
        future_curvature = self.track.racing_line[lookahead_idx].curvature
        if future_curvature > 0.02:
            braking_dist = future_curvature * 500 * self.profile_config.braking_distance_factor
            if target_dist < braking_dist:
                brake = max(brake, 0.5)
                throttle = 0.0
        
        return AIDecision(
            steering=clamp(steering, -1.0, 1.0),
            throttle=clamp(throttle, 0.0, 1.0),
            brake=clamp(brake, 0.0, 1.0),
            target_waypoint=target_point,
            target_speed=target_speed
        )
    
    def _find_target_waypoint(self) -> int:
        """Find the racing line waypoint to target (lookahead)."""
        state = self.vehicle.state
        
        # Find closest point on racing line
        best_idx = 0
        best_dist = float('inf')
        
        search_start = max(0, self.current_target_idx - 20)
        search_end = min(len(self.track.racing_line), self.current_target_idx + 100)
        
        for i in range(search_start, search_end):
            dist = vec2_distance(state.position, self.track.racing_line[i].position)
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        
        # Lookahead based on speed
        lookahead = max(5, int(state.speed * 0.5))
        target_idx = (best_idx + lookahead) % len(self.track.racing_line)
        self.current_target_idx = target_idx
        
        return target_idx
    
    def _should_overtake(self) -> bool:
        """Decide whether to attempt overtake."""
        if self.closest_ahead is None:
            return False
        
        if self.overtake_timer > 0:
            self.overtake_timer -= 1/60
            return True
        
        # Speed advantage
        speed_diff = self.vehicle.state.speed - self.closest_ahead.state.speed
        speed_advantage = speed_diff / max(self.vehicle.state.speed, 1.0)
        
        # Space to maneuver
        _, left_dist, _ = self._raycast_cache[2]
        _, right_dist, _ = self._raycast_cache[6]
        has_space = max(left_dist, right_dist) > 8.0
        
        # Aggression check
        if (speed_advantage > self.profile_config.overtake_threshold and 
            has_space and 
            self.rng.random() < self.profile_config.aggression):
            
            self.overtake_target = self.closest_ahead
            self.overtake_side = -1 if left_dist > right_dist else 1
            self.overtake_timer = 3.0
            return True
        
        return False
    
    def _execute_overtake(self, state: VehicleState, forward: Vec2, right: Vec2, target_speed: float) -> AIDecision:
        """Execute overtaking maneuver."""
        # Steer to overtake side
        steering = self.overtake_side * 0.8
        
        # Full throttle
        throttle = 1.0
        brake = 0.0
        
        # Boost if available
        boost = self.vehicle.config.has_boost and state.boost_available
        
        return AIDecision(
            steering=clamp(steering, -1.0, 1.0),
            throttle=throttle,
            brake=brake,
            boost=boost,
            target_speed=target_speed * 1.1
        )
    
    def _defensive_line(self, base_steering: float, state: VehicleState, 
                        forward: Vec2, right: Vec2) -> float:
        """Adjust line to defend position."""
        if self.closest_behind is None:
            return base_steering
        
        to_behind = vec2_sub(self.closest_behind.state.position, state.position)
        behind_side = vec2_dot(vec2_normalize(to_behind), right)
        
        # Move to block
        if behind_side > 0:  # Behind on right
            return base_steering + 0.3
        else:  # Behind on left
            return base_steering - 0.3
    
    def _recovery_behavior(self, dt: float) -> AIDecision:
        """Recovery when stuck or off track."""
        state = self.vehicle.state
        
        if not self.recovery_mode:
            self.recovery_mode = True
            self.recovery_timer = 0.0
        
        self.recovery_timer += dt
        
        # Reverse to get unstuck
        if self.recovery_timer < 1.5:
            return AIDecision(steering=0.0, throttle=0.0, brake=1.0)
        elif self.recovery_timer < 3.0:
            # Turn wheels and reverse
            steering = 1.0 if self.rng.random() > 0.5 else -1.0
            return AIDecision(steering=steering, throttle=0.5, brake=0.0)
        else:
            # Try to go forward
            self.recovery_mode = False
            self.stuck_timer = 0.0
            target_idx = self._find_target_waypoint()
            target_point = self.track.racing_line[target_idx].position
            to_target = vec2_sub(target_point, state.position)
            target_dir = vec2_normalize(to_target) if vec2_length(to_target) > 0 else vec2(1, 0)
            forward = vec2(np.cos(state.angle), np.sin(state.angle))
            cross = forward[0] * target_dir[1] - forward[1] * target_dir[0]
            steering = clamp(cross * 2.0, -1.0, 1.0)
            return AIDecision(steering=steering, throttle=0.8, brake=0.0)
    
    def _apply_errors(self, decision: AIDecision) -> AIDecision:
        """Apply random errors based on error rate."""
        if self.rng.random() < self.profile_config.error_rate:
            # Random mistake
            error_type = self.rng.integers(0, 4)
            if error_type == 0:
                decision.steering += self.rng.normal(0, 0.3)
            elif error_type == 1:
                decision.throttle *= 0.5
            elif error_type == 2:
                decision.brake = max(decision.brake, 0.3)
            elif error_type == 3:
                decision.steering *= -1  # Wrong way momentarily
        
        decision.steering = clamp(decision.steering, -1.0, 1.0)
        decision.throttle = clamp(decision.throttle, 0.0, 1.0)
        decision.brake = clamp(decision.brake, 0.0, 1.0)
        
        return decision


def create_npc_ai(vehicle: Vehicle, track: Track, profile: AIProfile,
                  rng: DeterministicRNG, physics_world: PhysicsWorld) -> NPCAI:
    """Factory function to create NPC AI."""
    return NPCAI(vehicle, track, profile, rng, physics_world)


def get_ai_profile(profile: AIProfile) -> AIProfileConfig:
    """Get AI profile configuration."""
    return AI_PROFILES[profile]


def list_ai_profiles() -> List[AIProfile]:
    """List all available AI profiles."""
    return list(AIProfile)


import numpy as np
from core.math import clamp