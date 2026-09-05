"""
Main Game Entry Point

Handles game initialization, main loop, and mode management.
"""

import pygame
import numpy as np
from typing import Optional, Dict, Any, List
import sys
import os

from core.config import load_config, Config, init_config, SimulationContext
from core.random import DeterministicRNG
from track.track import Track
from track.generator import TrackGenerator, TrackGeneratorConfig
from vehicles.vehicle import Vehicle, VehicleConfig
from vehicles.presets import get_vehicle_preset
from ai.npc_ai import NPCAI, AIProfile, create_npc_ai
from modes.race_modes import (
    RaceMode, RaceModeConfig, OneVOneMode, OneVFourMode, AIVsAIMode, create_race_mode
)
from modes.training_mode import TrainingMode, TrainingModeConfig
from modes.race_system import RaceState
from ui.renderer import Renderer, create_renderer
from ui.menus import MenuManager
from telemetry.telemetry import TelemetryRecorder, ReplayPlayer
from core.physics import PhysicsWorld, RigidBody, PhysicsMaterial


class Game:
    """
    Main game class.
    
    Manages:
    - Configuration
    - Renderer and menus
    - Race modes
    - Input handling
    - Main loop
    """
    
    def __init__(self, config: Config, headless: bool = False):
        self.config = config
        self.headless = headless
        self.running = True
        self.clock = None
        
        # Renderer
        self.renderer: Optional[Renderer] = None
        self.menu_manager: Optional[MenuManager] = None
        
        # Current mode
        self.current_mode: Optional[Any] = None
        self.mode_type: Optional[RaceMode] = None
        
        # Telemetry
        self.telemetry_recorder = TelemetryRecorder(
            config.telemetry.to_dict() if hasattr(config.telemetry, 'to_dict') else config.telemetry
        )
        
        # State
        self.selected_vehicle = "balanced"
        self.selected_seed: Optional[Union[int, str]] = None
        self.selected_ai_profile = "normal"
        self.selected_mode = RaceMode.ONE_V_ONE
        
        # Callbacks
        self._setup_callbacks()
    
    def _setup_callbacks(self) -> None:
        """Set up menu callbacks."""
        # These will be set after menu_manager is created
        pass
    
    def initialize(self) -> bool:
        """Initialize game systems."""
        try:
            if not self.headless:
                # Initialize pygame
                pygame.init()
                pygame.display.set_caption("AI Racing Simulator")
                
                # Create renderer
                self.renderer = create_renderer(self.config.to_dict())
                self.renderer.initialize()
                
                # Create menu manager
                self.menu_manager = MenuManager(self.renderer, self.config.to_dict())
                self.menu_manager.on_mode_selected = self._on_mode_selected
                self.menu_manager.on_seed_entered_callback = self._on_seed_entered
                self.menu_manager.on_vehicle_selected_callback = self._on_vehicle_selected
                self.menu_manager.on_quit = self._on_quit
                self.menu_manager.on_restart = self._on_restart
                self.menu_manager.on_resume = self._on_resume
            
            self.clock = pygame.time.Clock() if not self.headless else None
            
            return True
        except Exception as e:
            print(f"Failed to initialize game: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _on_mode_selected(self, mode: RaceMode) -> None:
        """Handle mode selection from menu."""
        self.selected_mode = mode
        
        if mode == RaceMode.ONE_V_ONE:
            self.menu_manager.seed_menu.input_active = True
        elif mode == RaceMode.ONE_V_FOUR:
            self.menu_manager.seed_menu.input_active = True
        elif mode == RaceMode.TRAINING:
            self.menu_manager.seed_menu.input_active = True
    
    def _on_seed_entered(self, seed: str) -> None:
        """Handle seed entry."""
        self.selected_seed = seed
        self._start_race()
    
    def _on_vehicle_selected(self, vehicle: str) -> None:
        """Handle vehicle selection."""
        self.selected_vehicle = vehicle
        self._start_race()
    
    def _on_quit(self) -> None:
        """Handle quit."""
        self.running = False
    
    def _on_restart(self) -> None:
        """Handle race restart."""
        self._start_race()
    
    def _on_resume(self) -> None:
        """Handle resume from pause."""
        if self.current_mode:
            self.current_mode.resume()
            self.menu_manager.hide_pause()
    
    def _start_race(self) -> None:
        """Start the selected race mode."""
        # Clean up previous mode
        if self.current_mode:
            # Save telemetry
            if hasattr(self.current_mode, 'get_result'):
                result = self.current_mode.get_result()
                if result:
                    self.telemetry_recorder.finish_recording({
                        'winner': result.winner_id,
                        'standings': [
                            {
                                'name': p.name,
                                'position': p.position,
                                'state': p.state.value,
                                'race_time': p.race_time,
                                'best_lap_time': p.best_lap_time if p.best_lap_time != float('inf') else 0,
                                'lap_times': [lt.time for lt in p.lap_times]
                            }
                            for p in result.participants
                        ]
                    })
        
        # Create mode config
        mode_config = RaceModeConfig(
            mode=self.selected_mode,
            seed=self.selected_seed,
            theme=self.config.track.get('theme', 'asphalt') if hasattr(self.config.track, 'get') else 'asphalt',
            complexity=0.5,
            road_width=self.config.track.get('min_road_width', 12.0) if hasattr(self.config.track, 'get') else 12.0,
            num_laps=3,
            vehicle_preset=self.selected_vehicle,
            ai_profile=self.selected_ai_profile,
            catchup_enabled=self.config.get('race', {}).get('catchup_enabled', False),
            catchup_strength=0.1,
            ghost_car=True,
            player_vehicle_preset=self.selected_vehicle,
            player_is_human=True
        )
        
        # Create race mode
        try:
            self.current_mode = create_race_mode(self.selected_mode, mode_config, self.config)
            self.current_mode.initialize(self.selected_seed)
            
            # Set up renderer
            if self.renderer and self.current_mode.track:
                self.renderer.set_track(self.current_mode.track)
                self.renderer.set_vehicles(self.current_mode.vehicles)
            
            # Set up telemetry
            if self.current_mode.vehicles:
                participant_infos = []
                for i, v in enumerate(self.current_mode.vehicles):
                    participant_infos.append({
                        'name': v.state.name if hasattr(v.state, 'name') else f'Car {i+1}',
                        'is_player': (i == 0),
                        'vehicle_id': v.body.id
                    })
                self.telemetry_recorder.start_recording(
                    seed=self.current_mode.sim_context.seed,
                    mode=self.selected_mode.value,
                    track_config=self.current_mode.track.to_dict() if self.current_mode.track else {},
                    vehicle_config=self.current_mode.vehicles[0].config.__dict__ if self.current_mode.vehicles else {},
                    physics_config={},
                    ai_config={},
                    training_config={},
                    participants=participant_infos
                )
            
            # Set render callback
            if self.current_mode:
                self.current_mode.set_render_callback(self._on_render_data)
            
            print(f"Started {self.selected_mode.value} race with seed {self.selected_seed}")
            
        except Exception as e:
            print(f"Failed to start race: {e}")
            import traceback
            traceback.print_exc()
    
    def _on_render_data(self, data: Dict[str, Any]) -> None:
        """Callback for race mode render data."""
        if self.renderer:
            self.renderer.set_race_state(data.get('race_state', {}))
            
            # Record telemetry
            if self.current_mode and self.current_mode.vehicles:
                player_vehicle = self.current_mode.vehicles[0]
                state = player_vehicle.state
                
                frame_data = {
                    'time': self.current_mode.sim_context.time,
                    'position': state.position.tolist(),
                    'angle': state.angle,
                    'velocity': state.velocity.tolist(),
                    'angular_velocity': state.angular_velocity,
                    'speed': state.speed,
                    'forward_speed': state.forward_speed,
                    'lateral_speed': state.lateral_speed,
                    'slip_angle': state.slip_angle,
                    'is_drifting': state.is_drifting,
                    'drift_intensity': state.drift_intensity,
                    'on_road': state.on_road,
                    'surface_type': state.surface_type,
                    'surface_grip': state.surface_grip,
                    'throttle': state.throttle,
                    'brake': state.brake,
                    'steering': state.steering,
                    'boost_active': state.boost_active,
                    'lap': 0,
                    'checkpoint': 0,
                    'checkpoints_crossed': 0,
                    'race_position': 1,
                    'race_time': 0,
                    'current_lap_time': 0,
                    'best_lap_time': 0,
                    'reward': 0,
                    'total_reward': 0,
                    'action': [state.steering, state.throttle, state.brake]
                }
                
                # Add race info if available
                if self.current_mode.race_manager:
                    player_participant = self.current_mode.race_manager.get_participant_by_vehicle(player_vehicle.body.id)
                    if player_participant:
                        frame_data['lap'] = player_participant.current_lap
                        frame_data['checkpoint'] = player_participant.last_checkpoint
                        frame_data['checkpoints_crossed'] = player_participant.checkpoints_crossed
                        frame_data['race_position'] = player_participant.position
                        frame_data['race_time'] = player_participant.race_time
                        frame_data['current_lap_time'] = player_participant.current_lap_time
                        frame_data['best_lap_time'] = player_participant.best_lap_time if player_participant.best_lap_time != float('inf') else 0
                
                self.telemetry_recorder.record_frame(frame_data)
    
    def handle_events(self) -> None:
        """Handle pygame events."""
        if self.headless:
            return
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return
            
            # Let renderer handle some events
            renderer_action = self.renderer.handle_event(event)
            if renderer_action == "pause" and self.current_mode and self.current_mode.running:
                self.current_mode.pause()
                self.menu_manager.show_pause()
            
            # Let menu handle events
            if self.menu_manager:
                menu_action = self.menu_manager.handle_event(event)
                # Menu actions are handled via callbacks
            
            # Player input during race
            if self.current_mode and self.current_mode.running and not self.current_mode.paused:
                self._handle_player_input(event)
    
    def _handle_player_input(self, event: pygame.event.Event) -> None:
        """Handle player keyboard input."""
        keys = pygame.key.get_pressed()
        
        throttle = 0.0
        brake = 0.0
        steering = 0.0
        boost = False
        
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            throttle = 1.0
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            brake = 1.0
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            steering = -1.0
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            steering = 1.0
        if keys[pygame.K_LSHIFT] or keys[pygame.K_SPACE]:
            boost = True
        
        if self.current_mode:
            self.current_mode.set_player_input(throttle, brake, steering, boost)
    
    def update(self, dt: float) -> None:
        """Update game state."""
        if self.current_mode and self.current_mode.running:
            # Get player input for headless mode
            player_input = None
            if self.headless:
                player_input = {'throttle': 0, 'brake': 0, 'steering': 0, 'boost': False}
            
            state = self.current_mode.step(dt, player_input)
            
            # Update renderer
            if self.renderer and self.current_mode.vehicles:
                self.renderer.set_vehicles(self.current_mode.vehicles)
                self.renderer.update_camera(dt, self.current_mode.vehicles[0])
            
            # Check if finished
            if self.current_mode.finished:
                result = self.current_mode.get_result()
                if result and self.menu_manager:
                    # Format results for menu
                    standings = []
                    for p in result.participants:
                        standings.append({
                            'name': p.name,
                            'position': p.position,
                            'state': p.state.value,
                            'race_time': p.race_time,
                            'best_lap_time': p.best_lap_time if p.best_lap_time != float('inf') else 0,
                            'lap_times': [lt.time for lt in p.lap_times]
                        })
                    
                    winner = next((p for p in result.participants if p.position == 1), None)
                    results = {
                        'winner': {'name': winner.name, 'total_time': winner.race_time} if winner else None,
                        'standings': standings
                    }
                    self.menu_manager.show_results(results)
    
    def render(self, dt: float) -> None:
        """Render frame."""
        if self.headless or not self.renderer:
            return
        
        self.renderer.render(dt)
        self.menu_manager.render()
        pygame.display.flip()
    
    def run(self) -> int:
        """Main game loop."""
        if not self.initialize():
            return 1
        
        print("Game initialized. Starting main loop...")
        
        dt = 1/60  # Fixed timestep for rendering
        accumulator = 0.0
        
        while self.running:
            # Handle events
            self.handle_events()
            
            if not self.running:
                break
            
            # Fixed timestep simulation
            if self.current_mode and self.current_mode.running:
                # Use physics timestep
                sim_dt = self.config.game.fixed_dt
                accumulator += dt
                
                while accumulator >= sim_dt:
                    self.update(sim_dt)
                    accumulator -= sim_dt
            
            # Render
            self.render(dt)
            
            # Cap frame rate
            if self.clock:
                dt = self.clock.tick(60) / 1000.0
        
        self.cleanup()
        return 0
    
    def cleanup(self) -> None:
        """Clean up resources."""
        if self.current_mode and hasattr(self.current_mode, 'get_result'):
            result = self.current_mode.get_result()
            if result:
                self.telemetry_recorder.finish_recording({})
        
        if self.renderer:
            self.renderer.cleanup()
        
        pygame.quit()


def create_game(config: Config, headless: bool = False) -> Game:
    """Factory function to create game."""
    return Game(config, headless)


if __name__ == "__main__":
    # Load config
    config = load_config()
    
    # Create and run game
    game = create_game(config)
    sys.exit(game.run())