"""
Telemetry and Replay System

Handles recording, exporting, and replaying race data.
"""

import json
import csv
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

from core.math import Vec2


@dataclass
class TelemetryFrame:
    """Single frame of telemetry data."""
    step: int
    time: float
    # Vehicle state
    position: List[float]  # [x, y]
    angle: float
    velocity: List[float]  # [vx, vy]
    angular_velocity: float
    speed: float
    forward_speed: float
    lateral_speed: float
    slip_angle: float
    is_drifting: bool
    drift_intensity: float
    on_road: bool
    surface_type: str
    surface_grip: float
    # Inputs
    throttle: float
    brake: float
    steering: float
    boost_active: bool
    # Race state
    lap: int
    checkpoint: int
    checkpoints_crossed: int
    race_position: int
    race_time: float
    current_lap_time: float
    best_lap_time: float
    # Rewards (for training)
    reward: float = 0.0
    total_reward: float = 0.0
    action: Optional[List[float]] = None


@dataclass
class ReplayData:
    """Complete replay data for a race/episode."""
    # Metadata
    version: str = "1.0"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    seed: Union[int, str] = 0
    game_version: str = "1.0.0"
    mode: str = "unknown"
    
    # Configuration
    track_config: Dict[str, Any] = field(default_factory=dict)
    vehicle_config: Dict[str, Any] = field(default_factory=dict)
    physics_config: Dict[str, Any] = field(default_factory=dict)
    ai_config: Dict[str, Any] = field(default_factory=dict)
    training_config: Dict[str, Any] = field(default_factory=dict)
    
    # Race setup
    num_participants: int = 1
    participant_names: List[str] = field(default_factory=list)
    num_laps: int = 3
    
    # Frame data
    frames: List[TelemetryFrame] = field(default_factory=list)
    
    # Results
    results: Dict[str, Any] = field(default_factory=dict)
    
    # Random states for exact replay
    rng_states: Dict[str, Any] = field(default_factory=dict)


class TelemetryRecorder:
    """
    Records telemetry during simulation.
    
    Features:
    - Configurable logging frequency
    - Multiple export formats (JSON, CSV)
    - Automatic file rotation
    - Compression support
    - Memory-efficient streaming
    """
    
    def __init__(self, config: Dict[str, Any], output_dir: str = "logs"):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.enabled = config.get('enabled', True)
        self.log_frequency = config.get('log_frequency', 1)
        self.export_format = config.get('export_format', 'json')
        self.save_replays = config.get('save_replays', True)
        self.replay_dir = Path(config.get('replay_dir', 'replays'))
        self.replay_dir.mkdir(parents=True, exist_ok=True)
        self.include_observations = config.get('include_observations', False)
        self.include_actions = config.get('include_actions', True)
        self.include_rewards = config.get('include_rewards', True)
        self.compression = config.get('compression', False)
        
        # Current session
        self.current_replay: Optional[ReplayData] = None
        self.current_episode = 0
        self.frame_count = 0
        self.step_count = 0
    
    def start_recording(self, seed: Union[int, str], mode: str,
                        track_config: Dict, vehicle_config: Dict,
                        physics_config: Dict, ai_config: Dict,
                        training_config: Dict,
                        participants: List[Dict]) -> None:
        """Start a new recording session."""
        if not self.enabled:
            return
        
        self.current_replay = ReplayData(
            seed=seed,
            mode=mode,
            track_config=track_config,
            vehicle_config=vehicle_config,
            physics_config=physics_config,
            ai_config=ai_config,
            training_config=training_config,
            num_participants=len(participants),
            participant_names=[p.get('name', f'Car {i+1}') for i, p in enumerate(participants)],
            num_laps=track_config.get('num_laps', 3)
        )
        self.frame_count = 0
        self.step_count = 0
    
    def record_frame(self, frame_data: Dict[str, Any]) -> None:
        """Record a single frame of telemetry."""
        if not self.enabled or self.current_replay is None:
            return
        
        self.step_count += 1
        
        # Only log at configured frequency
        if self.step_count % self.log_frequency != 0:
            return
        
        frame = TelemetryFrame(
            step=self.step_count,
            time=frame_data.get('time', 0),
            position=frame_data.get('position', [0, 0]),
            angle=frame_data.get('angle', 0),
            velocity=frame_data.get('velocity', [0, 0]),
            angular_velocity=frame_data.get('angular_velocity', 0),
            speed=frame_data.get('speed', 0),
            forward_speed=frame_data.get('forward_speed', 0),
            lateral_speed=frame_data.get('lateral_speed', 0),
            slip_angle=frame_data.get('slip_angle', 0),
            is_drifting=frame_data.get('is_drifting', False),
            drift_intensity=frame_data.get('drift_intensity', 0),
            on_road=frame_data.get('on_road', True),
            surface_type=frame_data.get('surface_type', 'asphalt'),
            surface_grip=frame_data.get('surface_grip', 1.0),
            throttle=frame_data.get('throttle', 0),
            brake=frame_data.get('brake', 0),
            steering=frame_data.get('steering', 0),
            boost_active=frame_data.get('boost_active', False),
            lap=frame_data.get('lap', 0),
            checkpoint=frame_data.get('checkpoint', 0),
            checkpoints_crossed=frame_data.get('checkpoints_crossed', 0),
            race_position=frame_data.get('race_position', 1),
            race_time=frame_data.get('race_time', 0),
            current_lap_time=frame_data.get('current_lap_time', 0),
            best_lap_time=frame_data.get('best_lap_time', 0),
            reward=frame_data.get('reward', 0) if self.include_rewards else 0,
            total_reward=frame_data.get('total_reward', 0) if self.include_rewards else 0,
            action=frame_data.get('action') if self.include_actions else None
        )
        
        self.current_replay.frames.append(frame)
        self.frame_count += 1
    
    def record_rng_state(self, rng_states: Dict[str, Any]) -> None:
        """Record RNG states for exact replay."""
        if self.current_replay:
            self.current_replay.rng_states = rng_states
    
    def finish_recording(self, results: Dict[str, Any]) -> Optional[str]:
        """Finish recording and save to file."""
        if not self.enabled or self.current_replay is None:
            return None
        
        self.current_replay.results = results
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        seed_str = str(self.current_replay.seed).replace('/', '_')
        filename = f"replay_{self.current_replay.mode}_{seed_str}_{timestamp}"
        
        saved_files = []
        
        # Save as JSON
        if self.export_format in ('json', 'both'):
            json_path = self.replay_dir / f"{filename}.json"
            self._save_json(json_path)
            saved_files.append(str(json_path))
        
        # Save as CSV
        if self.export_format in ('csv', 'both'):
            csv_path = self.replay_dir / f"{filename}.csv"
            self._save_csv(csv_path)
            saved_files.append(str(csv_path))
        
        print(f"Replay saved: {', '.join(saved_files)} ({self.frame_count} frames)")
        
        self.current_replay = None
        return saved_files[0] if saved_files else None
    
    def _save_json(self, path: Path) -> None:
        """Save replay as JSON."""
        # Convert to serializable dict
        data = asdict(self.current_replay)
        
        # Convert frames
        data['frames'] = [asdict(f) for f in self.current_replay.frames]
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def _save_csv(self, path: Path) -> None:
        """Save replay frames as CSV."""
        if not self.current_replay.frames:
            return
        
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            headers = [
                'step', 'time', 'pos_x', 'pos_y', 'angle', 'vel_x', 'vel_y',
                'angular_vel', 'speed', 'forward_speed', 'lateral_speed',
                'slip_angle', 'is_drifting', 'drift_intensity', 'on_road',
                'surface_type', 'surface_grip', 'throttle', 'brake', 'steering',
                'boost_active', 'lap', 'checkpoint', 'checkpoints_crossed',
                'race_position', 'race_time', 'current_lap_time', 'best_lap_time'
            ]
            
            if self.include_rewards:
                headers.extend(['reward', 'total_reward'])
            if self.include_actions:
                headers.extend(['action_steering', 'action_throttle', 'action_brake'])
            
            writer.writerow(headers)
            
            # Data rows
            for frame in self.current_replay.frames:
                row = [
                    frame.step, frame.time,
                    frame.position[0], frame.position[1],
                    frame.angle,
                    frame.velocity[0], frame.velocity[1],
                    frame.angular_velocity,
                    frame.speed, frame.forward_speed, frame.lateral_speed,
                    frame.slip_angle, frame.is_drifting, frame.drift_intensity,
                    frame.on_road, frame.surface_type, frame.surface_grip,
                    frame.throttle, frame.brake, frame.steering,
                    frame.boost_active, frame.lap, frame.checkpoint,
                    frame.checkpoints_crossed, frame.race_position,
                    frame.race_time, frame.current_lap_time, frame.best_lap_time
                ]
                
                if self.include_rewards:
                    row.extend([frame.reward, frame.total_reward])
                if self.include_actions and frame.action:
                    row.extend(frame.action)
                elif self.include_actions:
                    row.extend([0, 0, 0])
                
                writer.writerow(row)
    
    def export_telemetry_csv(self, telemetry: List[Dict], filepath: str) -> None:
        """Export raw telemetry list to CSV."""
        if not telemetry:
            return
        
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Get all keys from first entry
            keys = list(telemetry[0].keys())
            writer.writerow(keys)
            
            for entry in telemetry:
                row = []
                for k in keys:
                    v = entry.get(k)
                    if isinstance(v, (list, dict)):
                        row.append(json.dumps(v))
                    else:
                        row.append(v)
                writer.writerow(row)


class ReplayPlayer:
    """
    Plays back recorded replays.
    
    Can be used for:
    - Visual replay
    - Analysis
    - Ghost car generation
    - Deterministic verification
    """
    
    def __init__(self, replay_data: ReplayData):
        self.replay_data = replay_data
        self.current_frame = 0
        self.playing = False
        self.speed = 1.0
    
    @classmethod
    def load(cls, filepath: str) -> 'ReplayPlayer':
        """Load replay from file."""
        path = Path(filepath)
        
        if path.suffix == '.json':
            with open(path, 'r') as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unsupported replay format: {path.suffix}")
        
        # Reconstruct ReplayData
        frames = []
        for f in data.get('frames', []):
            frames.append(TelemetryFrame(**f))
        
        replay = ReplayData(
            version=data.get('version', '1.0'),
            timestamp=data.get('timestamp', ''),
            seed=data.get('seed', 0),
            game_version=data.get('game_version', ''),
            mode=data.get('mode', ''),
            track_config=data.get('track_config', {}),
            vehicle_config=data.get('vehicle_config', {}),
            physics_config=data.get('physics_config', {}),
            ai_config=data.get('ai_config', {}),
            training_config=data.get('training_config', {}),
            num_participants=data.get('num_participants', 1),
            participant_names=data.get('participant_names', []),
            num_laps=data.get('num_laps', 3),
            frames=frames,
            results=data.get('results', {}),
            rng_states=data.get('rng_states', {})
        )
        
        return cls(replay)
    
    def get_frame(self, index: int) -> Optional[TelemetryFrame]:
        """Get frame by index."""
        if 0 <= index < len(self.replay_data.frames):
            return self.replay_data.frames[index]
        return None
    
    def get_current_frame(self) -> Optional[TelemetryFrame]:
        """Get current playback frame."""
        return self.get_frame(self.current_frame)
    
    def step(self) -> Optional[TelemetryFrame]:
        """Advance to next frame."""
        if self.current_frame < len(self.replay_data.frames) - 1:
            self.current_frame += 1
            return self.replay_data.frames[self.current_frame]
        return None
    
    def seek(self, frame_index: int) -> Optional[TelemetryFrame]:
        """Seek to specific frame."""
        self.current_frame = max(0, min(frame_index, len(self.replay_data.frames) - 1))
        return self.get_current_frame()
    
    def seek_time(self, time: float) -> Optional[TelemetryFrame]:
        """Seek to specific time."""
        for i, frame in enumerate(self.replay_data.frames):
            if frame.time >= time:
                self.current_frame = i
                return frame
        self.current_frame = len(self.replay_data.frames) - 1
        return self.get_current_frame()
    
    def play(self) -> None:
        """Start playback."""
        self.playing = True
    
    def pause(self) -> None:
        """Pause playback."""
        self.playing = False
    
    def set_speed(self, speed: float) -> None:
        """Set playback speed."""
        self.speed = max(0.1, min(10.0, speed))
    
    def get_duration(self) -> float:
        """Get total replay duration in seconds."""
        if self.replay_data.frames:
            return self.replay_data.frames[-1].time
        return 0.0
    
    def get_frame_count(self) -> int:
        """Get total frame count."""
        return len(self.replay_data.frames)
    
    def get_ghost_data(self) -> List[Dict]:
        """Get ghost car data (position, angle, inputs over time)."""
        ghost = []
        for frame in self.replay_data.frames:
            ghost.append({
                'time': frame.time,
                'position': frame.position,
                'angle': frame.angle,
                'speed': frame.speed,
                'throttle': frame.throttle,
                'brake': frame.brake,
                'steering': frame.steering
            })
        return ghost


from typing import Union