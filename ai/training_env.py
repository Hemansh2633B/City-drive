"""
AI Training Environment

Gymnasium-compatible environment for reinforcement learning.
Supports vector observations, sensor observations, and image observations.
"""

import numpy as np
import hashlib
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import gymnasium as gym
from gymnasium import spaces

from core.random import DeterministicRNG
from core.config import SimulationContext, Config
from core.math import Vec2, vec2, vec2_add, vec2_sub, vec2_mul, vec2_dot, vec2_length, vec2_normalize, vec2_angle, vec2_perp
from track.track import Track, Checkpoint
from track.generator import TrackGenerator, TrackGeneratorConfig
from vehicles.vehicle import Vehicle, VehicleConfig, VehicleState
from vehicles.presets import get_vehicle_preset
from core.physics import PhysicsWorld, RigidBody, PhysicsMaterial


class ObservationMode(Enum):
    VECTOR = "vector"
    SENSOR = "sensor"
    IMAGE = "image"
    COMBINED = "combined"


class ActionMode(Enum):
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"


@dataclass
class TrainingConfig:
    """Configuration for training environment."""
    # Observation
    observation_mode: ObservationMode = ObservationMode.VECTOR
    sensor_config: Dict[str, Any] = field(default_factory=lambda: {
        'num_rays': 16,
        'max_distance': 50.0,
        'angles': [-3.14159, -2.35619, -1.5708, -0.7854, 0.0, 0.7854, 1.5708, 2.35619, 3.14159]
    })
    image_resolution: Tuple[int, int] = (84, 84)
    image_grayscale: bool = True
    
    # Action
    action_mode: ActionMode = ActionMode.CONTINUOUS
    
    # Rewards
    rewards: Dict[str, float] = field(default_factory=lambda: {
        'progress_per_meter': 0.1,
        'checkpoint_bonus': 10.0,
        'lap_bonus': 100.0,
        'race_win_bonus': 500.0,
        'speed_bonus_factor': 0.01,
        'collision_penalty': -5.0,
        'offroad_penalty_per_second': -2.0,
        'backward_penalty': -1.0,
        'spin_penalty': -0.5,
        'idle_penalty': -0.1,
        'wrong_way_penalty': -3.0,
        'overtake_bonus': 20.0,
        'overtaken_penalty': -10.0
    })
    
    # Termination
    termination: Dict[str, Any] = field(default_factory=lambda: {
        'max_offroad_time': 5.0,
        'max_collisions': 10,
        'max_idle_time': 10.0,
        'max_wrong_way_time': 3.0,
        'min_speed_threshold': 1.0
    })
    
    # Episode
    max_episode_steps: int = 10000
    max_race_time: float = 300.0
    
    # Curriculum
    curriculum: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': True,
        'stages': [
            {'name': 'easy', 'complexity': 0.2, 'road_width_multiplier': 1.5, 'speed_multiplier': 0.6, 'max_opponents': 0},
            {'name': 'medium', 'complexity': 0.4, 'road_width_multiplier': 1.2, 'speed_multiplier': 0.8, 'max_opponents': 1},
            {'name': 'hard', 'complexity': 0.6, 'road_width_multiplier': 1.0, 'speed_multiplier': 1.0, 'max_opponents': 3},
            {'name': 'expert', 'complexity': 0.8, 'road_width_multiplier': 0.8, 'speed_multiplier': 1.2, 'max_opponents': 4}
        ]
    })


class RacingEnv(gym.Env):
    """
    Gymnasium-compatible racing environment.
    
    Observation spaces:
    - Vector: Vehicle state + track info + opponent info
    - Sensor: Raycast distances around vehicle
    - Image: Top-down camera view (optional)
    - Combined: Vector + Sensor
    
    Action spaces:
    - Continuous: [steering, throttle, brake] in [-1, 1] x [0, 1] x [0, 1]
    - Discrete: 5 actions (left, right, accelerate, brake, none)
    """
    
    metadata = {
        'render_modes': ['human', 'rgb_array', 'none'],
        'render_fps': 60
    }
    
    def __init__(self, config: Optional[TrainingConfig] = None, 
                 game_config: Optional[Config] = None,
                 render_mode: Optional[str] = None):
        super().__init__()
        
        self.train_config = config or TrainingConfig()
        self.game_config = game_config
        self.render_mode = render_mode
        
        # Spaces will be defined after track generation
        self.observation_space = None
        self.action_space = None
        
        # Simulation state
        self.sim_context: Optional[SimulationContext] = None
        self.physics_world: Optional[PhysicsWorld] = None
        self.track: Optional[Track] = None
        self.track_generator: Optional[TrackGenerator] = None
        self.vehicle: Optional[Vehicle] = None
        self.vehicle_body: Optional[RigidBody] = None
        self.opponents: List[Vehicle] = []
        self.opponent_bodies: List[RigidBody] = []
        
        # Episode state
        self.current_step = 0
        self.episode_reward = 0.0
        self.episode_info: Dict[str, Any] = {}
        self.last_checkpoint = 0
        self.checkpoints_crossed = 0
        self.laps_completed = 0
        self.wrong_way_time = 0.0
        self.idle_time = 0.0
        self.offroad_time = 0.0
        self.collision_count = 0
        self.last_position = vec2(0, 0)
        self.best_lap_time = float('inf')
        self.current_lap_time = 0.0
        self.lap_times: List[float] = []
        
        # Curriculum
        self.curriculum_stage = 0
        
        # Telemetry
        self.telemetry_buffer: List[Dict] = []
        self.telemetry_enabled = True
        self._render_surface = None
        self._render_font = None
        self._render_small_font = None

        # Initialize spaces (will be finalized in reset)
        self._define_spaces()
    
    def _define_spaces(self) -> None:
        """Define observation and action spaces."""
        # Action space
        if self.train_config.action_mode == ActionMode.CONTINUOUS:
            # steering [-1, 1], throttle [0, 1], brake [0, 1]
            self.action_space = spaces.Box(
                low=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
                high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
                dtype=np.float32
            )
        else:
            # Discrete: 0=left, 1=right, 2=accelerate, 3=brake, 4=none
            self.action_space = spaces.Discrete(5)
        
        # Observation space - will be sized properly in reset
        # For now, create a placeholder
        if self.train_config.observation_mode == ObservationMode.VECTOR:
            # Approximate size: vehicle state (~30) + track info (~10) + opponents (~20 each)
            obs_size = 80
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
            )
        elif self.train_config.observation_mode == ObservationMode.SENSOR:
            num_rays = self.train_config.sensor_config.get('num_rays', 16)
            obs_size = num_rays + 6  # ray distances + six vehicle-state values
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
            )
        elif self.train_config.observation_mode == ObservationMode.IMAGE:
            h, w = self.train_config.image_resolution
            c = 1 if self.train_config.image_grayscale else 3
            self.observation_space = spaces.Box(
                low=0, high=255, shape=(h, w, c), dtype=np.uint8
            )
        else:  # COMBINED
            num_rays = self.train_config.sensor_config.get('num_rays', 16)
            obs_size = 80 + num_rays + 6
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
            )
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment for new episode."""
        # Gymnasium requires an integer seed. Convert readable string seeds
        # deterministically for its internal RNG while preserving the original
        # seed for the track generator and episode metadata.
        gym_seed = seed
        if isinstance(gym_seed, str):
            gym_seed = int.from_bytes(
                hashlib.sha256(gym_seed.encode("utf-8")).digest()[:4], "big"
            )
        elif gym_seed is not None:
            gym_seed = int(gym_seed)
        super().reset(seed=gym_seed)
        
        # Use provided seed or generate one
        if seed is not None:
            episode_seed = seed
        elif options and 'seed' in options:
            episode_seed = options['seed']
        else:
            episode_seed = self.np_random.integers(0, 2**31 - 1)
        
        # Apply curriculum if enabled
        if self.train_config.curriculum.get('enabled', False):
            stages = self.train_config.curriculum.get('stages', [])
            if not stages:
                stages = [{'name': 'standard', 'complexity': 0.35,
                           'road_width_multiplier': 1.0, 'speed_multiplier': 1.0}]
            stage_index = min(max(int(self.curriculum_stage), 0), len(stages) - 1)
            stage = stages[stage_index]
            complexity = stage['complexity']
            road_width_mult = stage['road_width_multiplier']
            speed_mult = stage['speed_multiplier']
        else:
            complexity = 0.35
            road_width_mult = 1.0
            speed_mult = 1.0

        # A single-agent RL environment: no opponent cars or race-mode logic.
        max_opponents = 0
        
        # Create simulation context
        if self.game_config is None:
            from core.config import load_config
            self.game_config = load_config()
        
        self.sim_context = SimulationContext(self.game_config, episode_seed)
        
        # Create physics world
        self.physics_world = PhysicsWorld(self.game_config, self.sim_context.get_rng('physics'))
        
        # Generate track
        track_rng = self.sim_context.get_rng('track')
        track_gen_config = TrackGeneratorConfig()
        self.track_generator = TrackGenerator(track_gen_config, track_rng)
        
        reset_options = options or {}
        track_theme = reset_options.get("theme", "asphalt")
        track_complexity = reset_options.get("complexity", complexity)
        track_width = reset_options.get(
            "road_width", self.game_config.track.min_road_width * road_width_mult
        )
        self.track = self.track_generator.generate(
            seed=episode_seed,
            theme=track_theme,
            complexity=track_complexity,
            road_width=track_width
        )
        if not self.track.is_valid:
            raise RuntimeError(
                f"Seed {episode_seed!r} generated an invalid track: "
                + "; ".join(self.track.validation_errors)
            )
        
        # Create player vehicle
        vehicle_rng = self.sim_context.get_rng('vehicle')
        vehicle_config = get_vehicle_preset("balanced")
        vehicle_config.top_speed *= speed_mult
        vehicle_config.acceleration *= speed_mult
        
        self.vehicle_body = RigidBody(
            mass=vehicle_config.mass,
            width=vehicle_config.width,
            height=vehicle_config.length,
            material=PhysicsMaterial(friction=1.0, restitution=0.1)
        )
        self.physics_world.add_body(self.vehicle_body)
        
        self.vehicle = Vehicle(vehicle_config, self.vehicle_body, vehicle_rng)
        
        # Spawn at start position
        spawn_pos, spawn_angle = self.track.get_spawn_position(0)
        self.vehicle.reset(spawn_pos, spawn_angle)
        
        # Create opponents if needed
        self.opponents = []
        self.opponent_bodies = []
        for i in range(min(max_opponents, 4)):
            opp_rng = self.sim_context.get_rng(f'opponent_{i}')
            opp_config = get_vehicle_preset("balanced")
            opp_config.top_speed *= speed_mult * 0.9  # Slightly slower
            opp_config.acceleration *= speed_mult * 0.9
            
            opp_body = RigidBody(
                mass=opp_config.mass,
                width=opp_config.width,
                height=opp_config.length,
                material=PhysicsMaterial(friction=1.0, restitution=0.1)
            )
            self.physics_world.add_body(opp_body)
            
            opp_vehicle = Vehicle(opp_config, opp_body, opp_rng)
            
            spawn_pos, spawn_angle = self.track.get_spawn_position(i + 1)
            opp_vehicle.reset(spawn_pos, spawn_angle)
            
            self.opponents.append(opp_vehicle)
            self.opponent_bodies.append(opp_body)
        
        # Reset episode state
        self.current_step = 0
        self.episode_reward = 0.0
        self.last_checkpoint = 0
        self.checkpoints_crossed = 0
        self.laps_completed = 0
        self.wrong_way_time = 0.0
        self.idle_time = 0.0
        self.offroad_time = 0.0
        self.collision_count = 0
        self.last_position = self.vehicle.state.position.copy()
        self.current_lap_time = 0.0
        self.lap_times = []
        self.telemetry_buffer = []
        
        # Get initial observation
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one environment step."""
        # Parse action
        if self.train_config.action_mode == ActionMode.CONTINUOUS:
            steering = float(np.clip(action[0], -1.0, 1.0))
            throttle = float(np.clip(action[1], 0.0, 1.0))
            brake = float(np.clip(action[2], 0.0, 1.0))
            boost = False
        else:
            # Discrete actions
            steering = 0.0
            throttle = 0.0
            brake = 0.0
            if action == 0:  # Left
                steering = -1.0
            elif action == 1:  # Right
                steering = 1.0
            elif action == 2:  # Accelerate
                throttle = 1.0
            elif action == 3:  # Brake
                brake = 1.0
            # action == 4: None (coast)
        
        # Apply input to vehicle
        self.vehicle.set_input(throttle, brake, steering, boost)
        
        # Step physics
        dt = self.sim_context.dt
        collisions = self.physics_world.step(dt)
        
        # Update vehicle state
        surface_type, on_road = self._get_surface_at_position(self.vehicle.state.position)
        self.vehicle.step(dt, surface_type, on_road)
        
        # Update opponents (simple AI for now)
        for opp in self.opponents:
            opp_surface, opp_on_road = self._get_surface_at_position(opp.state.position)
            opp.step(dt, opp_surface, opp_on_road)
        
        # Process collisions
        for col in collisions:
            if col.body_a == self.vehicle_body.id or col.body_b == self.vehicle_body.id:
                self.collision_count += 1
        
        # Calculate reward
        reward = self._calculate_reward()
        self.episode_reward += reward
        
        # Check termination
        terminated, truncated = self._check_termination()
        
        # Update episode state
        self.current_step += 1
        self.sim_context.advance_step()
        self.current_lap_time += dt
        
        # Log telemetry
        if self.telemetry_enabled and self.current_step % self.game_config.telemetry.log_frequency == 0:
            self._log_telemetry(action, reward)
        
        # Get observation and info
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _get_surface_at_position(self, position: Vec2) -> Tuple[str, bool]:
        """Determine surface type and whether on road at position."""
        if self.track is None:
            return "asphalt", True
        
        # Simple check: distance to centerline
        min_dist = float('inf')
        for i in range(len(self.track.centerline) - 1):
            # Distance to segment
            from core.math import point_line_distance
            dist = point_line_distance(position, self.track.centerline[i], self.track.centerline[i + 1])
            min_dist = min(min_dist, dist)
        
        half_width = self.track.road_width * 0.5
        on_road = min_dist <= half_width * 1.1  # Small margin
        
        return self.track.theme, on_road
    
    def _calculate_reward(self) -> float:
        """Calculate step reward."""
        r = self.train_config.rewards
        reward = 0.0
        state = self.vehicle.state
        
        # Progress reward
        progress = vec2_length(vec2_sub(state.position, self.last_position))
        if state.forward_speed > 0:
            reward += r['progress_per_meter'] * progress
        else:
            reward += r['backward_penalty'] * progress
        
        self.last_position = state.position.copy()
        
        # Speed bonus
        if state.forward_speed > 5.0:
            reward += r['speed_bonus_factor'] * state.forward_speed
        
        # Checkpoint crossing
        next_cp_idx = (self.last_checkpoint + 1) % len(self.track.checkpoints)
        next_cp = self.track.checkpoints[next_cp_idx]
        
        if next_cp.contains_point(state.position):
            if next_cp_idx == 0 and self.checkpoints_crossed > 0:
                # Completed a lap
                self.laps_completed += 1
                self.lap_times.append(self.current_lap_time)
                if self.current_lap_time < self.best_lap_time:
                    self.best_lap_time = self.current_lap_time
                reward += r['lap_bonus']
                self.current_lap_time = 0.0
            else:
                reward += r['checkpoint_bonus']
            
            self.last_checkpoint = next_cp_idx
            self.checkpoints_crossed += 1
        
        # Penalties
        if self.collision_count > 0:
            reward += r['collision_penalty'] * self.collision_count
            self.collision_count = 0  # Only penalize once per collision
        
        if not state.on_road:
            reward += r['offroad_penalty_per_second'] * self.sim_context.dt
            self.offroad_time += self.sim_context.dt
        else:
            self.offroad_time = 0.0
        
        if state.forward_speed < -1.0:
            reward += r['backward_penalty'] * self.sim_context.dt
        
        if state.is_drifting:
            reward += r['spin_penalty'] * self.sim_context.dt
        
        if state.speed < self.train_config.termination['min_speed_threshold']:
            self.idle_time += self.sim_context.dt
            if self.idle_time > 1.0:
                reward += r['idle_penalty'] * self.sim_context.dt
        else:
            self.idle_time = 0.0
        
        # Wrong way detection
        if self.last_checkpoint < len(self.track.checkpoints):
            cp = self.track.checkpoints[self.last_checkpoint]
            to_next = vec2_sub(self.track.checkpoints[(self.last_checkpoint + 1) % len(self.track.checkpoints)].position, state.position)
            forward = vec2(np.cos(state.angle), np.sin(state.angle))
            if vec2_dot(to_next, forward) < -0.5:
                self.wrong_way_time += self.sim_context.dt
                if self.wrong_way_time > 1.0:
                    reward += r['wrong_way_penalty'] * self.sim_context.dt
            else:
                self.wrong_way_time = 0.0
        
        # Overtake detection (simplified)
        for opp in self.opponents:
            # Check if we overtook
            pass  # Implement if needed
        
        return reward
    
    def _check_termination(self) -> Tuple[bool, bool]:
        """Check if episode should terminate."""
        t = self.train_config.termination
        state = self.vehicle.state
        
        # Race completed
        if self.laps_completed >= self.track.num_laps:
            return True, False
        
        # Max steps
        if self.current_step >= self.train_config.max_episode_steps:
            return False, True
        
        # Max time
        if self.sim_context.time >= self.train_config.max_race_time:
            return False, True
        
        # Offroad too long
        if self.offroad_time > t['max_offroad_time']:
            return True, False
        
        # Too many collisions
        if state.collision_count >= t['max_collisions']:
            return True, False
        
        # Idle too long
        if self.idle_time > t['max_idle_time']:
            return True, False
        
        # Wrong way too long
        if self.wrong_way_time > t['max_wrong_way_time']:
            return True, False
        
        return False, False
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation based on mode."""
        mode = self.train_config.observation_mode
        
        if mode == ObservationMode.VECTOR:
            return self._get_vector_observation()
        elif mode == ObservationMode.SENSOR:
            return self._get_sensor_observation()
        elif mode == ObservationMode.IMAGE:
            return self._get_image_observation()
        else:  # COMBINED
            vec_obs = self._get_vector_observation()
            sensor_obs = self._get_sensor_observation()
            return np.concatenate([vec_obs, sensor_obs]).astype(np.float32)
    
    def _get_vector_observation(self) -> np.ndarray:
        """Get vector observation."""
        state = self.vehicle.state
        obs = []
        
        # Vehicle state
        obs.extend([
            state.speed / 100.0,  # Normalized
            state.forward_speed / 100.0,
            state.lateral_speed / 50.0,
            state.angular_velocity / 10.0,
            state.slip_angle / 1.0,
            float(state.is_drifting),
            state.drift_intensity,
            float(state.on_road),
            state.surface_grip,
            state.throttle,
            state.brake,
            state.steering,
            float(state.boost_active),
            float(state.boost_cooldown_timer <= 0.0),
        ])
        
        # Position relative to next checkpoint
        next_cp = self.track.checkpoints[(self.last_checkpoint + 1) % len(self.track.checkpoints)]
        to_cp = vec2_sub(next_cp.position, state.position)
        dist_to_cp = vec2_length(to_cp)
        dir_to_cp = vec2_normalize(to_cp) if dist_to_cp > 0 else vec2(0, 0)
        forward = vec2(np.cos(state.angle), np.sin(state.angle))
        
        obs.extend([
            dist_to_cp / 200.0,  # Normalized
            vec2_dot(dir_to_cp, forward),  # Alignment with checkpoint
            vec2_dot(dir_to_cp, vec2(-forward[1], forward[0])),  # Lateral offset
        ])
        
        # Racing line info
        rl_point = self.track.get_racing_line_point(self.checkpoints_crossed / max(len(self.track.checkpoints), 1))
        to_rl = vec2_sub(rl_point.position, state.position)
        obs.extend([
            vec2_length(to_rl) / 50.0,
            vec2_dot(vec2_normalize(to_rl) if vec2_length(to_rl) > 0 else vec2(0, 0), forward),
            rl_point.target_speed / 100.0,
            rl_point.curvature * 50.0,
        ])
        
        # Race progress
        obs.extend([
            self.laps_completed / max(self.track.num_laps, 1),
            self.checkpoints_crossed / max(len(self.track.checkpoints), 1),
            self.current_step / self.train_config.max_episode_steps,
        ])
        
        # Opponent info (closest few)
        for opp in self.opponents[:3]:
            to_opp = vec2_sub(opp.state.position, state.position)
            dist = vec2_length(to_opp)
            rel_vel = vec2_sub(opp.state.velocity, state.velocity)
            obs.extend([
                min(dist / 100.0, 1.0),
                vec2_dot(vec2_normalize(to_opp) if dist > 0 else vec2(0, 0), forward),
                opp.state.speed / 100.0,
            ])
        
        # Pad to fixed size if needed
        obs = np.array(obs, dtype=np.float32)
        expected_size = self.observation_space.shape[0]
        if len(obs) < expected_size:
            obs = np.pad(obs, (0, expected_size - len(obs)))
        elif len(obs) > expected_size:
            obs = obs[:expected_size]
        
        return obs
    
    def _get_sensor_observation(self) -> np.ndarray:
        """Get raycast sensor observation."""
        sensor_cfg = self.train_config.sensor_config
        num_rays = sensor_cfg.get('num_rays', 16)
        max_dist = sensor_cfg.get('max_distance', 50.0)
        angles = sensor_cfg.get('angles', [])
        
        if len(angles) != num_rays:
            # Generate evenly spaced angles
            angles = np.linspace(-np.pi, np.pi, num_rays, endpoint=False).tolist()
        
        state = self.vehicle.state
        forward_angle = state.angle
        
        readings = []
        for angle_offset in angles:
            ray_angle = forward_angle + angle_offset
            direction = vec2(np.cos(ray_angle), np.sin(ray_angle))
            
            hit, dist, _, _ = self.physics_world.raycast(
                state.position, direction, max_dist, ignore_body=self.vehicle_body.id
            )
            
            # Normalize distance
            readings.append(dist / max_dist if hit else 1.0)
        
        # Add minimal vehicle state
        vehicle_state = np.array([
            state.speed / 100.0,
            state.forward_speed / 100.0,
            state.lateral_speed / 50.0,
            state.angular_velocity / 10.0,
            state.slip_angle / 1.0,
            float(state.on_road),
        ], dtype=np.float32)
        
        return np.concatenate([np.array(readings, dtype=np.float32), vehicle_state])
    
    def _get_image_observation(self) -> np.ndarray:
        """Get camera image observation (placeholder for headless)."""
        # In headless mode, return zeros
        h, w = self.train_config.image_resolution
        c = 1 if self.train_config.image_grayscale else 3
        return np.zeros((h, w, c), dtype=np.uint8)
    
    def _get_info(self) -> Dict[str, Any]:
        """Get info dictionary."""
        state = self.vehicle.state
        return {
            'step': self.current_step,
            'time': self.sim_context.time,
            'position': state.position.copy(),
            'angle': state.angle,
            'speed': state.speed,
            'lap': self.laps_completed,
            'checkpoint': self.last_checkpoint,
            'checkpoints_crossed': self.checkpoints_crossed,
            'laps_completed': self.laps_completed,
            'best_lap_time': self.best_lap_time if self.best_lap_time != float('inf') else 0.0,
            'current_lap_time': self.current_lap_time,
            'lap_times': self.lap_times.copy(),
            'collision_count': state.collision_count,
            'offroad_time': self.offroad_time,
            'reward': self.episode_reward,
            'seed': self.sim_context.seed,
            'track_length': self.track.total_length,
            'on_road': state.on_road,
            'is_drifting': state.is_drifting,
            'track_valid': bool(self.track.is_valid),
            'track_errors': list(self.track.validation_errors),
            'total_checkpoints': len(self.track.checkpoints),
            'next_checkpoint': (self.last_checkpoint + 1) % max(1, len(self.track.checkpoints))
        }
    
    def _log_telemetry(self, action: np.ndarray, reward: float) -> None:
        """Log telemetry data."""
        self.telemetry_buffer.append({
            'step': self.current_step,
            'time': self.sim_context.time,
            'position': self.vehicle.state.position.tolist(),
            'angle': self.vehicle.state.angle,
            'speed': self.vehicle.state.speed,
            'action': action.tolist(),
            'reward': reward,
            'total_reward': self.episode_reward,
            'lap': self.laps_completed,
            'checkpoint': self.last_checkpoint
        })
    
    def render(self) -> Optional[np.ndarray]:
        """
        Draw a live top-down training viewer.

        Human mode opens a pygame window. rgb_array returns an RGB image for
        recording/evaluation. Rendering is separate from step(), so headless
        training pays no rendering cost.
        """
        if self.render_mode not in ("human", "rgb_array") or self.track is None:
            return None

        try:
            import pygame
        except ImportError:
            if self.render_mode == "human":
                raise RuntimeError("Install pygame-ce to use --render")
            return None

        width, height = 1000, 720
        if not pygame.get_init():
            pygame.init()

        if self.render_mode == "human":
            if not pygame.display.get_init():
                pygame.display.init()
            if self._render_surface is None:
                self._render_surface = pygame.display.set_mode((width, height))
                pygame.display.set_caption("City Drive | RL Training Viewer")
            surface = self._render_surface
        else:
            surface = pygame.Surface((width, height))

        surface.fill((23, 31, 40))
        points = self.track.centerline
        xs = [float(p[0]) for p in points]
        ys = [float(p[1]) for p in points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        span_x = max(max_x - min_x, 1.0)
        span_y = max(max_y - min_y, 1.0)
        padding = 70
        scale = min((width - 2 * padding) / span_x, (height - 2 * padding) / span_y)
        x_offset = (width - span_x * scale) * 0.5
        y_offset = (height - span_y * scale) * 0.5

        def screen_point(point):
            return (
                int(x_offset + (float(point[0]) - min_x) * scale),
                int(height - (y_offset + (float(point[1]) - min_y) * scale))
            )

        path_points = [screen_point(p) for p in points]
        road_px = max(8, min(48, int(self.track.road_width * scale)))
        if len(path_points) >= 2:
            pygame.draw.lines(surface, (9, 13, 18), False, path_points, road_px + 8)
            pygame.draw.lines(surface, (65, 76, 87), False, path_points, road_px)
            pygame.draw.lines(surface, (120, 132, 142), False, path_points, max(1, road_px // 24))

        for checkpoint in self.track.checkpoints:
            left, right = checkpoint.get_bounds()
            color = (245, 194, 66) if checkpoint.is_start_finish else (68, 191, 175)
            pygame.draw.line(surface, color, screen_point(left), screen_point(right), 2)

        state = self.vehicle.state if self.vehicle is not None else None
        if state is not None:
            position = np.asarray(state.position, dtype=float)
            forward = np.array([np.cos(state.angle), np.sin(state.angle)])
            side = np.array([-forward[1], forward[0]])
            nose = position + forward * 3.0
            rear = position - forward * 2.0
            car_points = [
                screen_point(nose),
                screen_point(rear + side * 1.15),
                screen_point(rear - side * 1.15)
            ]
            pygame.draw.polygon(surface, (77, 170, 255), car_points)
            pygame.draw.polygon(surface, (225, 242, 255), car_points, 2)

        if self._render_font is None:
            self._render_font = pygame.font.Font(None, 25)
            self._render_small_font = pygame.font.Font(None, 20)

        speed = state.speed * 3.6 if state is not None else 0.0
        hud_lines = [
            ("CITY DRIVE  /  REINFORCEMENT LEARNING", (235, 243, 250)),
            (f"Seed: {self.track.seed}    Map: {'VALID' if self.track.is_valid else 'INVALID'}",
             (113, 230, 185) if self.track.is_valid else (255, 110, 110)),
            (f"Episode step: {self.current_step}    Speed: {speed:.1f} km/h",
             (215, 225, 235)),
            (f"Reward: {self.episode_reward:.2f}    Checkpoints: {self.checkpoints_crossed}",
             (215, 225, 235)),
            (f"Laps completed: {self.laps_completed}/{self.track.num_laps}",
             (215, 225, 235)),
            ("Blue arrow = learning agent     Gold = start/finish     Teal = checkpoint",
             (156, 174, 190))
        ]
        pygame.draw.rect(surface, (14, 21, 29), (0, 0, width, 142))
        for i, (label, color) in enumerate(hud_lines):
            font = self._render_font if i == 0 else self._render_small_font
            surface.blit(font.render(label, True, color), (20, 14 + i * 21))

        if self.render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.display.quit()
                    self._render_surface = None
                    return None
            pygame.display.flip()
            return None

        pixels = pygame.surfarray.array3d(surface)
        return np.transpose(pixels, (1, 0, 2)).copy()

    def close(self) -> None:
        """Close viewer resources, if they were opened."""
        if self._render_surface is not None:
            try:
                import pygame
                if pygame.display.get_init():
                    pygame.display.quit()
            finally:
                self._render_surface = None

    def get_telemetry(self) -> List[Dict]:
        """Get collected telemetry."""
        return self.telemetry_buffer.copy()
    
    def set_curriculum_stage(self, stage: int) -> None:
        """Set curriculum learning stage."""
        max_stage = len(self.train_config.curriculum.get('stages', [])) - 1
        self.curriculum_stage = max(0, min(stage, max_stage))
    
    def advance_curriculum(self) -> bool:
        """Advance to next curriculum stage. Returns True if advanced."""
        max_stage = len(self.train_config.curriculum.get('stages', [])) - 1
        if self.curriculum_stage < max_stage:
            self.curriculum_stage += 1
            return True
        return False


# Convenience function for creating env
def make_racing_env(config: Optional[TrainingConfig] = None, 
                    game_config: Optional[Config] = None,
                    render_mode: Optional[str] = None) -> RacingEnv:
    """Factory function to create racing environment."""
    return RacingEnv(config, game_config, render_mode)