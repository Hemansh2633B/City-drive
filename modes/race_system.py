"""
Race System

Core race logic: lap tracking, checkpoint validation, position ranking, race states.
"""

import numpy as np
from typing import Union, List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from core.math import Vec2, vec2, vec2_sub, vec2_length, vec2_normalize, vec2_dot


class RaceState(Enum):
    WAITING = "waiting"
    COUNTDOWN = "countdown"
    RACING = "racing"
    FINISHED = "finished"
    PAUSED = "paused"


class CarState(Enum):
    RACING = "racing"
    DNF = "dnf"  # Did not finish
    FINISHED = "finished"


@dataclass
class LapTime:
    """Single lap time record."""
    lap_number: int
    time: float
    sector_times: List[float] = field(default_factory=list)
    is_valid: bool = True  # False if cut corners, etc.


@dataclass
class RaceParticipant:
    """A participant in the race."""
    # Identity
    id: int
    name: str
    is_player: bool = False
    vehicle_id: int = -1
    
    # Race state
    state: CarState = CarState.RACING
    current_lap: int = 0
    total_laps: int = 0
    last_checkpoint: int = 0
    checkpoints_crossed: int = 0
    wrong_way: bool = False
    wrong_way_timer: float = 0.0
    
    # Timing
    race_time: float = 0.0
    lap_times: List[LapTime] = field(default_factory=list)
    current_lap_time: float = 0.0
    best_lap_time: float = float('inf')
    sector_times: List[float] = field(default_factory=list)
    last_sector_time: float = 0.0
    current_sector: int = 0
    
    # Position
    position: int = 1
    sort_key: Tuple[int, int, float] = field(default_factory=lambda: (0, 0, 0.0))
    
    # Stats
    distance_traveled: float = 0.0
    max_speed: float = 0.0
    collisions: int = 0
    offroad_time: float = 0.0
    
    def get_total_time(self) -> float:
        """Get total race time."""
        return self.race_time
    
    def get_sort_key(self) -> Tuple[int, int, float]:
        """
        Get sorting key for race ranking.
        Returns: (-laps_completed, -checkpoints_crossed, progress_to_next_checkpoint)
        Higher is better for first two, lower is better for third.
        """
        # For sorting: we want descending laps, descending checkpoints, ascending progress
        # So we negate laps and checkpoints for ascending sort
        return (-self.current_lap, -self.checkpoints_crossed, self.sort_key[2])


@dataclass
class RaceResult:
    """Final race results."""
    participants: List[RaceParticipant]
    winner_id: int
    race_duration: float
    seed: Union[int, str]
    track_name: str
    num_laps: int
    
    def get_standings(self) -> List[RaceParticipant]:
        """Get participants sorted by final position."""
        return sorted(self.participants, key=lambda p: p.position)


@dataclass
class RaceConfig:
    """Race configuration."""
    num_laps: int = 3
    num_opponents: int = 0
    seed: Optional[Union[int, str]] = None
    theme: str = "asphalt"
    complexity: Optional[float] = None
    road_width: Optional[float] = None
    vehicle_preset: str = "balanced"
    ai_profile: str = "normal"
    catchup_enabled: bool = False
    catchup_strength: float = 0.1
    ghost_car: bool = False


class RaceManager:
    """
    Manages race state, lap counting, checkpoint validation, and ranking.
    
    Features:
    - Deterministic lap/checkpoint tracking
    - Accurate position ranking
    - DNF detection
    - Sector timing
    - Catch-up/rubber-banding (optional)
    - Ghost car recording
    """
    
    def __init__(self, config: RaceConfig, track: 'Track', physics_world: 'PhysicsWorld',
                 vehicles: List['Vehicle'], participant_infos: List[Dict]):
        self.config = config
        self.track = track
        self.physics_world = physics_world
        self.vehicles = vehicles
        
        # Participants
        self.participants: List[RaceParticipant] = []
        for i, info in enumerate(participant_infos):
            p = RaceParticipant(
                id=i,
                name=info.get('name', f'Car {i+1}'),
                is_player=info.get('is_player', False),
                vehicle_id=info.get('vehicle_id', -1),
                total_laps=config.num_laps
            )
            self.participants.append(p)
        
        # Race state
        self.race_state = RaceState.WAITING
        self.countdown_timer = 3.0
        self.race_start_time = 0.0
        self.race_end_time = 0.0
        
        # Countdown
        self.countdown_numbers = [3, 2, 1, 0]  # 3, 2, 1, GO!
        self.current_countdown = 0
        
        # Ghost recording
        self.ghost_data: List[Dict] = []
        self.record_ghost = config.ghost_car
        
        # Catch-up
        self.catchup_enabled = config.catchup_enabled
        self.catchup_strength = config.catchup_strength
        
        # Initialize sort keys
        self._update_sort_keys()
    
    def start_race(self, sim_time: float) -> None:
        """Start the race (after countdown)."""
        self.race_state = RaceState.RACING
        self.race_start_time = sim_time
        for p in self.participants:
            p.race_time = 0.0
    
    def update_countdown(self, dt: float) -> Tuple[bool, Optional[int]]:
        """Update countdown timer. Returns (race_started, countdown_number)."""
        if self.race_state != RaceState.COUNTDOWN:
            return False, None
        
        self.countdown_timer -= dt
        if self.countdown_timer <= 0:
            self.current_countdown += 1
            if self.current_countdown >= len(self.countdown_numbers):
                return True, None  # Race starts
            self.countdown_timer = 1.0
            return False, self.countdown_numbers[self.current_countdown - 1]
        
        return False, self.countdown_numbers[self.current_countdown - 1] if self.current_countdown > 0 else 3
    
    def update(self, dt: float, sim_time: float) -> None:
        """Update race state for all participants."""
        if self.race_state != RaceState.RACING:
            return
        
        # Update each participant
        for i, (participant, vehicle) in enumerate(zip(self.participants, self.vehicles)):
            if participant.state != CarState.RACING:
                continue
            
            state = vehicle.state
            
            # Update timing
            participant.race_time += dt
            participant.current_lap_time += dt
            participant.distance_traveled += state.speed * dt
            participant.max_speed = max(participant.max_speed, state.speed)
            
            # Update offroad time
            if not state.on_road:
                participant.offroad_time += dt
            
            # Update collisions
            participant.collisions = state.collision_count
            
            # Check checkpoints
            self._check_checkpoints(participant, vehicle)
            
            # Check wrong way
            self._check_wrong_way(participant, vehicle, dt)
            
            # Check if finished
            if participant.current_lap >= participant.total_laps:
                participant.state = CarState.FINISHED
                if self.race_end_time == 0:
                    self.race_end_time = sim_time
        
        # Update rankings
        self._update_rankings()
        
        # Apply catch-up if enabled
        if self.catchup_enabled:
            self._apply_catchup()
        
        # Record ghost data
        if self.record_ghost and self.participants[0].is_player:
            self._record_ghost()
        
        # Check if race finished (all finished or DNF)
        if all(p.state != CarState.RACING for p in self.participants):
            self.race_state = RaceState.FINISHED
            if self.race_end_time == 0:
                self.race_end_time = sim_time
    
    def _check_checkpoints(self, participant: RaceParticipant, vehicle: 'Vehicle') -> None:
        """Check if participant crossed checkpoints."""
        state = vehicle.state
        next_cp_idx = (participant.last_checkpoint + 1) % len(self.track.checkpoints)
        next_cp = self.track.checkpoints[next_cp_idx]
        
        if next_cp.contains_point(state.position):
            # Crossed checkpoint in correct direction
            participant.last_checkpoint = next_cp_idx
            participant.checkpoints_crossed += 1
            
            # Sector timing
            sector_time = participant.current_lap_time - participant.last_sector_time
            participant.sector_times.append(sector_time)
            participant.last_sector_time = participant.current_lap_time
            participant.current_sector = (participant.current_sector + 1) % len(self.track.checkpoints)
            
            # Lap completion
            if next_cp_idx == 0 and participant.checkpoints_crossed > 0:
                # Completed a lap
                participant.current_lap += 1
                lap_time = LapTime(
                    lap_number=participant.current_lap,
                    time=participant.current_lap_time,
                    sector_times=participant.sector_times.copy(),
                    is_valid=participant.wrong_way_timer == 0
                )
                participant.lap_times.append(lap_time)
                
                if participant.current_lap_time < participant.best_lap_time:
                    participant.best_lap_time = participant.current_lap_time
                
                # Reset for next lap
                participant.current_lap_time = 0.0
                participant.sector_times = []
                participant.last_sector_time = 0.0
                participant.current_sector = 0
    
    def _check_wrong_way(self, participant: RaceParticipant, vehicle: 'Vehicle', dt: float) -> None:
        """Detect wrong-way driving."""
        state = vehicle.state
        if participant.last_checkpoint < len(self.track.checkpoints):
            cp = self.track.checkpoints[participant.last_checkpoint]
            next_cp = self.track.checkpoints[(participant.last_checkpoint + 1) % len(self.track.checkpoints)]
            to_next = vec2_sub(next_cp.position, state.position)
            forward = vec2(np.cos(state.angle), np.sin(state.angle))
            
            if vec2_dot(to_next, forward) < -0.3:
                participant.wrong_way_timer += dt
                if participant.wrong_way_timer > 2.0:
                    participant.wrong_way = True
            else:
                participant.wrong_way_timer = max(0, participant.wrong_way_timer - dt)
                if participant.wrong_way_timer == 0:
                    participant.wrong_way = False
    
    def _update_rankings(self) -> None:
        """Update position rankings based on progress."""
        # Calculate sort keys for all racing participants
        for p in self.participants:
            if p.state == CarState.RACING:
                # Progress to next checkpoint
                next_cp_idx = (p.last_checkpoint + 1) % len(self.track.checkpoints)
                next_cp = self.track.checkpoints[next_cp_idx]
                
                # Find vehicle
                vehicle = None
                for v in self.vehicles:
                    if v.body.id == p.vehicle_id:
                        vehicle = v
                        break
                
                if vehicle:
                    to_cp = vec2_sub(next_cp.position, vehicle.state.position)
                    progress = 1.0 - min(vec2_length(to_cp) / 200.0, 1.0)  # Normalized 0-1
                    p.sort_key = (-p.current_lap, -p.checkpoints_crossed, -progress)
                else:
                    p.sort_key = (-p.current_lap, -p.checkpoints_crossed, 0)
            elif p.state == CarState.FINISHED:
                p.sort_key = (-p.total_laps - 1, 0, 0)  # Finished first
            else:  # DNF
                p.sort_key = (100, 0, 0)  # Last
        
        # Sort and assign positions
        sorted_participants = sorted(self.participants, key=lambda p: p.sort_key)
        for pos, p in enumerate(sorted_participants):
            p.position = pos + 1
    
    def _apply_catchup(self) -> None:
        """Apply catch-up/rubber-banding to trailing cars."""
        if len(self.participants) < 2:
            return
        
        # Find leader (among racing cars)
        racing = [p for p in self.participants if p.state == CarState.RACING]
        if not racing:
            return
        
        leader = min(racing, key=lambda p: p.sort_key)
        
        for p in racing:
            if p is leader:
                continue
            
            # Calculate gap
            leader_vehicle = next((v for v in self.vehicles if v.body.id == leader.vehicle_id), None)
            p_vehicle = next((v for v in self.vehicles if v.body.id == p.vehicle_id), None)
            
            if leader_vehicle and p_vehicle:
                gap = vec2_length(vec2_sub(leader_vehicle.state.position, p_vehicle.state.position))
                
                if gap > 50.0:  # Only if far behind
                    boost = min(self.catchup_strength * (gap / 100.0), 0.3)
                    # Apply as temporary speed boost
                    p_vehicle.config.top_speed *= (1.0 + boost)
                    p_vehicle.config.acceleration *= (1.0 + boost)
    
    def _record_ghost(self) -> None:
        """Record ghost car data for replay."""
        player = self.participants[0]
        vehicle = next((v for v in self.vehicles if v.body.id == player.vehicle_id), None)
        if vehicle:
            self.ghost_data.append({
                'time': player.race_time,
                'position': vehicle.state.position.tolist(),
                'angle': vehicle.state.angle,
                'speed': vehicle.state.speed,
                'throttle': vehicle.state.throttle,
                'brake': vehicle.state.brake,
                'steering': vehicle.state.steering
            })
    
    def _update_sort_keys(self) -> None:
        """Initialize sort keys."""
        for p in self.participants:
            p.sort_key = (0, 0, 0)
    
    def get_race_result(self) -> RaceResult:
        """Get final race results."""
        # Final sort
        self._update_rankings()
        
        # Assign final positions
        finished = [p for p in self.participants if p.state == CarState.FINISHED]
        racing = [p for p in self.participants if p.state == CarState.RACING]
        dnf = [p for p in self.participants if p.state == CarState.DNF]
        
        # Sort finished by total time
        finished.sort(key=lambda p: p.race_time)
        # Sort racing by progress
        racing.sort(key=lambda p: p.sort_key)
        
        # Combine
        all_sorted = finished + racing + dnf
        for i, p in enumerate(all_sorted):
            p.position = i + 1
        
        winner = all_sorted[0] if all_sorted else None
        
        return RaceResult(
            participants=self.participants,
            winner_id=winner.id if winner else -1,
            race_duration=self.race_end_time - self.race_start_time,
            seed=self.config.seed,
            track_name=self.track.theme,
            num_laps=self.config.num_laps
        )
    
    def set_dnf(self, participant_id: int) -> None:
        """Mark participant as DNF."""
        if 0 <= participant_id < len(self.participants):
            self.participants[participant_id].state = CarState.DNF
    
    def get_participant(self, participant_id: int) -> Optional[RaceParticipant]:
        """Get participant by ID."""
        if 0 <= participant_id < len(self.participants):
            return self.participants[participant_id]
        return None
    
    def get_participant_by_vehicle(self, vehicle_id: int) -> Optional[RaceParticipant]:
        """Get participant by vehicle ID."""
        for p in self.participants:
            if p.vehicle_id == vehicle_id:
                return p
        return None


from core.math import vec2
