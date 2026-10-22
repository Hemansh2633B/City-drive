"""
1v1 and 1v4 Race Modes

Multi-car race modes with human/AI participants.
"""

from __future__ import annotations

import numpy as np
from typing import List, Dict, Optional, Any, Tuple, Union
from dataclasses import dataclass
from enum import Enum

from core.config import SimulationContext, Config
from core.random import DeterministicRNG
from core.math import Vec2, vec2
from track.track import Track
from track.generator import TrackGenerator, TrackGeneratorConfig
from vehicles.vehicle import Vehicle, VehicleConfig
from vehicles.presets import get_vehicle_preset
from ai.npc_ai import NPCAI, AIProfile, create_npc_ai
from modes.race_system import RaceManager, RaceConfig, RaceState, CarState, RaceResult
from core.physics import PhysicsWorld, RigidBody, PhysicsMaterial


class RaceMode(Enum):
    TRAINING = "training"
    ONE_V_ONE = "1v1"
    ONE_V_FOUR = "1v4"


@dataclass
class RaceModeConfig:
    """Base configuration for race modes."""
    mode: RaceMode
    seed: Optional[Union[int, str]] = None
    theme: str = "asphalt"
    complexity: Optional[float] = None
    road_width: Optional[float] = None
    num_laps: int = 3
    vehicle_preset: str = "balanced"
    ai_profile: str = "normal"
    catchup_enabled: bool = False
    catchup_strength: float = 0.1
    ghost_car: bool = False
    
    # Player settings
    player_vehicle_preset: str = "balanced"
    player_is_human: bool = True


class BaseRaceMode:
    """Base class for race modes."""
    
    def __init__(self, config: RaceModeConfig, game_config: Config):
        self.config = config
        self.game_config = game_config
        
        # Simulation
        self.sim_context: Optional[SimulationContext] = None
        self.physics_world: Optional[PhysicsWorld] = None
        self.track: Optional[Track] = None
        self.track_generator: Optional[TrackGenerator] = None
        
        # Vehicles
        self.vehicles: List[Vehicle] = []
        self.vehicle_bodies: List[RigidBody] = []
        self.npc_ais: List[NPCAI] = []
        
        # Race management
        self.race_manager: Optional[RaceManager] = None
        
        # State
        self.running = False
        self.paused = False
        self.finished = False
        
        # Rendering
        self.render_callback = None
    
    def initialize(self, seed: Optional[Union[int, str]] = None) -> None:
        """Initialize the race with a seed."""
        race_seed = seed if seed is not None else self.config.seed
        if race_seed is None:
            race_seed = np.random.randint(0, 2**31 - 1)
        
        # Create simulation context
        self.sim_context = SimulationContext(self.game_config, race_seed)
        
        # Create physics world
        self.physics_world = PhysicsWorld(self.game_config, self.sim_context.get_rng('physics'))
        
        # Generate track
        track_rng = self.sim_context.get_rng('track')
        track_gen_config = TrackGeneratorConfig()
        self.track_generator = TrackGenerator(track_gen_config, track_rng)
        
        complexity = self.config.complexity or 0.5
        road_width = self.config.road_width or self.game_config.track.min_road_width
        
        self.track = self.track_generator.generate(
            seed=race_seed,
            theme=self.config.theme,
            complexity=complexity,
            road_width=road_width
        )
        
        # Create vehicles
        self._create_vehicles()
        
        # Create race manager
        race_config = RaceConfig(
            num_laps=self.config.num_laps,
            num_opponents=len(self.vehicles) - 1,
            seed=race_seed,
            theme=self.config.theme,
            vehicle_preset=self.config.vehicle_preset,
            ai_profile=self.config.ai_profile,
            catchup_enabled=self.config.catchup_enabled,
            catchup_strength=self.config.catchup_strength,
            ghost_car=self.config.ghost_car
        )
        
        participant_infos = []
        for i, vehicle in enumerate(self.vehicles):
            participant_infos.append({
                'name': vehicle.state.name if hasattr(vehicle.state, 'name') else f'Car {i+1}',
                'is_player': (i == 0 and self.config.player_is_human),
                'vehicle_id': vehicle.body.id
            })
        
        self.race_manager = RaceManager(
            race_config, self.track, self.physics_world, self.vehicles, participant_infos
        )
        
        # Start countdown
        self.race_manager.race_state = RaceState.COUNTDOWN
        self.running = True
    
    def _create_vehicles(self) -> None:
        """Create vehicles for this race mode. Override in subclasses."""
        raise NotImplementedError
    
    def step(self, dt: float, player_input: Optional[Dict] = None) -> Dict[str, Any]:
        """Step the race simulation."""
        if not self.running or self.paused:
            return self._get_state()
        
        sim_time = self.sim_context.time
        
        # Handle countdown
        if self.race_manager.race_state == RaceState.COUNTDOWN:
            started, countdown = self.race_manager.update_countdown(dt)
            if started:
                self.race_manager.start_race(sim_time)
        
        # Apply player input
        if player_input and self.config.player_is_human:
            player_vehicle = self.vehicles[0]
            player_vehicle.set_input(
                player_input.get('throttle', 0.0),
                player_input.get('brake', 0.0),
                player_input.get('steering', 0.0),
                player_input.get('boost', False)
            )
        
        # Update NPC AIs
        for npc_ai, vehicle in zip(self.npc_ais, self.vehicles[1:]):
            decision = npc_ai.update(dt, self.vehicles)
            vehicle.set_input(decision.throttle, decision.brake, decision.steering, decision.boost)
        
        # Step physics
        collisions = self.physics_world.step(dt)
        
        # Update vehicle states
        for vehicle in self.vehicles:
            surface_type, on_road = self._get_surface_at_position(vehicle.state.position)
            vehicle.step(dt, surface_type, on_road)
        
        # Update race manager
        self.race_manager.update(dt, sim_time)
        
        # Advance simulation
        self.sim_context.advance_step()
        
        # Check finish
        if self.race_manager.race_state == RaceState.FINISHED:
            self.finished = True
            self.running = False
        
        # Render callback
        if self.render_callback:
            self.render_callback(self._get_render_data())
        
        return self._get_state()
    
    def _get_surface_at_position(self, position: Vec2) -> Tuple[str, bool]:
        """Get surface type and on-road status at position."""
        if self.track is None:
            return "asphalt", True
        
        min_dist = float('inf')
        for i in range(len(self.track.centerline) - 1):
            from core.math import point_line_distance
            dist = point_line_distance(position, self.track.centerline[i], self.track.centerline[i + 1])
            min_dist = min(min_dist, dist)
        
        half_width = self.track.road_width * 0.5
        on_road = min_dist <= half_width * 1.1
        
        return self.track.theme, on_road
    
    def _get_state(self) -> Dict[str, Any]:
        """Get current race state for UI/telemetry."""
        state = {
            'running': self.running,
            'paused': self.paused,
            'finished': self.finished,
            'race_state': self.race_manager.race_state.value if self.race_manager else 'uninitialized',
            'countdown': self.race_manager.current_countdown if self.race_manager else 0,
            'sim_time': self.sim_context.time if self.sim_context else 0,
            'seed': self.sim_context.seed if self.sim_context else None,
            'track': self.track.to_dict() if self.track else None,
            'participants': []
        }
        
        if self.race_manager:
            for p in self.race_manager.participants:
                vehicle = next((v for v in self.vehicles if v.body.id == p.vehicle_id), None)
                v_state = vehicle.state if vehicle else None
                
                state['participants'].append({
                    'id': p.id,
                    'name': p.name,
                    'position': p.position,
                    'state': p.state.value,
                    'lap': p.current_lap,
                    'total_laps': p.total_laps,
                    'checkpoint': p.last_checkpoint,
                    'race_time': p.race_time,
                    'current_lap_time': p.current_lap_time,
                    'best_lap_time': p.best_lap_time if p.best_lap_time != float('inf') else 0,
                    'lap_times': [lt.time for lt in p.lap_times],
                    'speed': v_state.speed if v_state else 0,
                    'max_speed': p.max_speed,
                    'collisions': p.collisions,
                    'offroad_time': p.offroad_time,
                    'is_player': p.is_player
                })
        
        return state
    
    def _get_render_data(self) -> Dict[str, Any]:
        """Get data for rendering."""
        return {
            'track': self.track,
            'vehicles': self.vehicles,
            'race_state': self._get_state(),
            'camera_target': self.vehicles[0].state.position if self.vehicles else vec2(0, 0)
        }
    
    def set_render_callback(self, callback) -> None:
        """Set render callback function."""
        self.render_callback = callback
    
    def pause(self) -> None:
        """Pause the race."""
        self.paused = True
        if self.race_manager:
            self.race_manager.race_state = RaceState.PAUSED
    
    def resume(self) -> None:
        """Resume the race."""
        self.paused = False
        if self.race_manager and self.race_manager.race_state == RaceState.PAUSED:
            self.race_manager.race_state = RaceState.RACING
    
    def get_result(self) -> Optional[RaceResult]:
        """Get final race result."""
        if self.race_manager and self.finished:
            return self.race_manager.get_race_result()
        return None
    
    def set_player_input(self, throttle: float, brake: float, steering: float, boost: bool = False) -> None:
        """Set player input directly."""
        if self.vehicles and self.config.player_is_human:
            self.vehicles[0].set_input(throttle, brake, steering, boost)


class OneVOneMode(BaseRaceMode):
    """1v1 race mode: Player vs 1 opponent."""
    
    def __init__(self, config: RaceModeConfig, game_config: Config):
        super().__init__(config, game_config)
        self.config.mode = RaceMode.ONE_V_ONE
    
    def _create_vehicles(self) -> None:
        """Create player and one opponent vehicle."""
        # Player vehicle
        player_rng = self.sim_context.get_rng('player')
        player_config = get_vehicle_preset(self.config.player_vehicle_preset)
        
        player_body = RigidBody(
            mass=player_config.mass,
            width=player_config.width,
            height=player_config.length,
            material=PhysicsMaterial(friction=1.0, restitution=0.1)
        )
        self.physics_world.add_body(player_body)
        
        player_vehicle = Vehicle(player_config, player_body, player_rng)
        player_vehicle.state.name = "Player"
        
        spawn_pos, spawn_angle = self.track.get_spawn_position(0)
        player_vehicle.reset(spawn_pos, spawn_angle)
        
        self.vehicles.append(player_vehicle)
        self.vehicle_bodies.append(player_body)
        
        # Opponent vehicle
        opp_rng = self.sim_context.get_rng('opponent_0')
        opp_config = get_vehicle_preset(self.config.vehicle_preset)
        
        opp_body = RigidBody(
            mass=opp_config.mass,
            width=opp_config.width,
            height=opp_config.length,
            material=PhysicsMaterial(friction=1.0, restitution=0.1)
        )
        self.physics_world.add_body(opp_body)
        
        opp_vehicle = Vehicle(opp_config, opp_body, opp_rng)
        opp_vehicle.state.name = "Opponent"
        
        spawn_pos, spawn_angle = self.track.get_spawn_position(1)
        opp_vehicle.reset(spawn_pos, spawn_angle)
        
        self.vehicles.append(opp_vehicle)
        self.vehicle_bodies.append(opp_body)
        
        # Create NPC AI for opponent
        try:
            ai_profile = AIProfile(self.config.ai_profile)
        except ValueError:
            ai_profile = AIProfile.NORMAL
        
        npc_ai = create_npc_ai(opp_vehicle, self.track, ai_profile, opp_rng, self.physics_world)
        self.npc_ais.append(npc_ai)


class OneVFourMode(BaseRaceMode):
    """1v4 race mode: Player vs 4 opponents."""
    
    def __init__(self, config: RaceModeConfig, game_config: Config):
        super().__init__(config, game_config)
        self.config.mode = RaceMode.ONE_V_FOUR
    
    def _create_vehicles(self) -> None:
        """Create player and four opponent vehicles."""
        # Player vehicle
        player_rng = self.sim_context.get_rng('player')
        player_config = get_vehicle_preset(self.config.player_vehicle_preset)
        
        player_body = RigidBody(
            mass=player_config.mass,
            width=player_config.width,
            height=player_config.length,
            material=PhysicsMaterial(friction=1.0, restitution=0.1)
        )
        self.physics_world.add_body(player_body)
        
        player_vehicle = Vehicle(player_config, player_body, player_rng)
        player_vehicle.state.name = "Player"
        
        spawn_pos, spawn_angle = self.track.get_spawn_position(0)
        player_vehicle.reset(spawn_pos, spawn_angle)
        
        self.vehicles.append(player_vehicle)
        self.vehicle_bodies.append(player_body)
        
        # Four opponent vehicles with varied profiles
        profiles = [AIProfile.NORMAL, AIProfile.EXPERT, AIProfile.AGGRESSIVE, AIProfile.DEFENSIVE]
        
        for i in range(4):
            opp_rng = self.sim_context.get_rng(f'opponent_{i}')
            opp_config = get_vehicle_preset(self.config.vehicle_preset)
            
            # Slight variation in opponent performance
            variation = 1.0 + (i - 1.5) * 0.05  # -7.5% to +7.5%
            opp_config.top_speed *= variation
            opp_config.acceleration *= variation
            
            opp_body = RigidBody(
                mass=opp_config.mass,
                width=opp_config.width,
                height=opp_config.length,
                material=PhysicsMaterial(friction=1.0, restitution=0.1)
            )
            self.physics_world.add_body(opp_body)
            
            opp_vehicle = Vehicle(opp_config, opp_body, opp_rng)
            opp_vehicle.state.name = f"Opponent {i+1}"
            
            spawn_pos, spawn_angle = self.track.get_spawn_position(i + 1)
            opp_vehicle.reset(spawn_pos, spawn_angle)
            
            self.vehicles.append(opp_vehicle)
            self.vehicle_bodies.append(opp_body)
            
            # Assign AI profile (cycle through)
            ai_profile = profiles[i % len(profiles)]
            npc_ai = create_npc_ai(opp_vehicle, self.track, ai_profile, opp_rng, self.physics_world)
            self.npc_ais.append(npc_ai)


class AIVsAIMode(BaseRaceMode):
    """AI vs AI race mode for benchmarking."""
    
    def __init__(self, config: RaceModeConfig, game_config: Config, num_ai: int = 5):
        super().__init__(config, game_config)
        self.num_ai = num_ai
        self.config.player_is_human = False
    
    def _create_vehicles(self) -> None:
        """Create multiple AI vehicles."""
        profiles = [AIProfile.BEGINNER, AIProfile.NORMAL, AIProfile.EXPERT, 
                   AIProfile.AGGRESSIVE, AIProfile.DEFENSIVE]
        
        for i in range(self.num_ai):
            veh_rng = self.sim_context.get_rng(f'ai_{i}')
            veh_config = get_vehicle_preset(self.config.vehicle_preset)
            
            # Variation
            variation = 1.0 + (i - (self.num_ai - 1) / 2) * 0.03
            veh_config.top_speed *= variation
            veh_config.acceleration *= variation
            
            veh_body = RigidBody(
                mass=veh_config.mass,
                width=veh_config.width,
                height=veh_config.length,
                material=PhysicsMaterial(friction=1.0, restitution=0.1)
            )
            self.physics_world.add_body(veh_body)
            
            vehicle = Vehicle(veh_config, veh_body, veh_rng)
            vehicle.state.name = f"AI {i+1}"
            
            spawn_pos, spawn_angle = self.track.get_spawn_position(i)
            vehicle.reset(spawn_pos, spawn_angle)
            
            self.vehicles.append(vehicle)
            self.vehicle_bodies.append(veh_body)
            
            # AI profile
            ai_profile = profiles[i % len(profiles)]
            npc_ai = create_npc_ai(vehicle, self.track, ai_profile, veh_rng, self.physics_world)
            self.npc_ais.append(npc_ai)


def create_race_mode(mode: RaceMode, config: RaceModeConfig, game_config: Config) -> BaseRaceMode:
    """Factory function to create race mode."""
    if mode == RaceMode.ONE_V_ONE:
        return OneVOneMode(config, game_config)
    elif mode == RaceMode.ONE_V_FOUR:
        return OneVFourMode(config, game_config)
    elif mode == RaceMode.TRAINING:
        # Training mode uses different class
        from modes.training_mode import TrainingMode, TrainingModeConfig
        train_config = TrainingModeConfig(
            seed=config.seed,
            theme=config.theme,
            complexity=config.complexity,
            road_width=config.road_width,
            vehicle_preset=config.vehicle_preset
        )
        return TrainingMode(train_config, game_config)
    else:
        raise ValueError(f"Unknown race mode: {mode}")


from typing import Union