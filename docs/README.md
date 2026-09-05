# AI Racing Simulator - Documentation

## Table of Contents
1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Architecture Overview](#architecture-overview)
4. [Game Modes](#game-modes)
5. [Track Generation](#track-generation)
6. [Vehicle Physics](#vehicle-physics)
7. [AI Training Environment](#ai-training-environment)
8. [NPC Opponent AI](#npc-opponent-ai)
9. [Race System](#race-system)
10. [CLI Reference](#cli-reference)
11. [Configuration](#configuration)
12. [Telemetry & Replays](#telemetry--replays)
13. [Extending the Simulator](#extending-the-simulator)
14. [Testing](#testing)
15. [Troubleshooting](#troubleshooting)

---

## Installation

### Requirements
- Python 3.10+
- numpy >= 1.24.0
- gymnasium >= 0.29.0
- pygame >= 2.5.0
- pyyaml >= 6.0

### Setup
```bash
# Clone or navigate to project
cd /path/to/race

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import race; print('Installation successful')"
```

### Optional Dependencies
For faster math operations (automatically used if available):
```bash
pip install numba
```

For RL training with Stable Baselines3:
```bash
pip install stable-baselines3
```

---

## Quick Start

### Run a 1v1 Race (with graphics)
```bash
python scripts/game.py --mode 1v1 --seed 12345
```

### Run a 1v4 Race
```bash
python scripts/game.py --mode 1v4 --seed "daily-race-001"
```

### Run Training (headless)
```bash
python scripts/train.py --mode training --seed 12345 --episodes 100
```

### Evaluate an Agent
```bash
python scripts/evaluate.py --agent random --seeds 1 2 3 4 5
```

### Run Tests
```bash
python -m pytest tests/ -v
```

---

## Architecture Overview

### Project Structure
```
race/
├── core/           # Core systems (random, config, math, physics)
├── track/          # Procedural track generation & representation
├── vehicles/       # Vehicle physics, configs, presets
├── ai/             # Training env (Gymnasium) & NPC AI
├── modes/          # Game modes (training, 1v1, 1v4)
├── ui/             # Renderer, menus, HUD
├── telemetry/      # Logging, replays, CSV/JSON export
├── scripts/        # CLI entry points
├── configs/        # YAML configuration files
├── tests/          # Automated tests
└── assets/         # Sprites, fonts (future)
```

### Key Design Principles

1. **Determinism First**: Every system uses fixed timesteps and seeded RNGs. Same seed = same result.

2. **Modular Architecture**: Clean separation between physics, track, vehicles, AI, and rendering.

3. **Gymnasium Compatible**: Training environment follows standard RL interface.

4. **Headless Capable**: All simulation runs without rendering for fast training.

5. **Extensible**: Easy to add new tracks, vehicles, AI profiles, and observation types.

### Core Systems

#### Deterministic Random (`core/random.py`)
- `DeterministicRNG`: PCG64-based RNG with serializable state
- `SeededRandomStreams`: Multiple independent streams from master seed
- Supports save/restore for exact replays

#### Configuration (`core/config.py`)
- YAML-based configuration with nested attribute access
- `SimulationContext`: Holds sim state, RNG streams, timestep info

#### Math (`core/math.py`)
- JIT-compiled 2D vector operations (numba)
- Geometry: intersection, distance, SAT collision
- Angle handling, interpolation utilities

#### Physics (`core/physics.py`)
- Fixed timestep (60 Hz default, 4 substeps)
- SAT-based collision detection
- Impulse-based resolution with friction
- Spatial hashing broad phase
- Raycasting support

---

## Game Modes

### 1. Training Mode (`modes/training_mode.py`)
Single-car AI training environment.

**Features:**
- Gymnasium-compatible `RacingEnv`
- Curriculum learning (4 stages: easy → expert)
- Headless operation
- Configurable rewards, observations, actions
- Replay recording
- Telemetry export (JSON/CSV)

**Usage:**
```python
from modes.training_mode import TrainingMode, TrainingModeConfig
from core.config import load_config

config = load_config()
train_config = TrainingModeConfig(
    seed=12345,
    max_episodes=1000,
    curriculum_enabled=True
)
training_mode = TrainingMode(train_config, config)

# Run with your agent
result = training_mode.run_episode(your_agent)
```

### 2. 1v1 Mode (`modes/race_modes.py`)
Player vs one AI opponent.

**Features:**
- Human vs AI or AI vs AI
- Multiple AI difficulty profiles
- Lap-based racing with position tracking
- Optional catch-up/rubber-banding
- Ghost car recording

### 3. 1v4 Mode (`modes/race_modes.py`)
Player vs four AI opponents.

**Features:**
- 5 total racers with varied AI profiles
- Independent driving behaviors
- Overtaking, defensive driving, recovery
- Accurate race ranking

### 4. AI vs AI Benchmark
```bash
python scripts/game.py --mode ai_vs_ai --seed 12345
```

---

## Track Generation

### Seed-Based Generation
Every track is generated from a seed (integer or string):
```python
from track.generator import TrackGenerator, TrackGeneratorConfig
from core.random import DeterministicRNG

config = TrackGeneratorConfig()
rng = DeterministicRNG(12345)
generator = TrackGenerator(config, rng)

track = generator.generate(seed=12345, theme="asphalt", complexity=0.5)
```

### Track Properties
- **Centerline**: Ordered points forming closed loop
- **Boundaries**: Left/right edge polylines
- **Checkpoints**: Ordered validation gates
- **Racing Line**: Optimal path with target speeds
- **Spawn Positions**: Grid positions for up to 5 cars

### Validation
Tracks are automatically validated:
- Closed loop (start ≈ end)
- Minimum length
- Checkpoint reachability
- No self-intersections
- Valid spawn positions

### Themes
Visual themes (affect colors & surface friction):
- `asphalt` - Standard circuit
- `desert` - Sandy, lower grip
- `snow` - Icy, very low grip
- `grassland` - Green, moderate grip
- `neon` - Cyberpunk, high grip
- `industrial` - Concrete
- `dirt` - Rally style, variable grip

### Complexity Control
```python
track = generator.generate(
    seed=12345,
    complexity=0.8,        # 0.2 (simple) to 0.8 (complex)
    road_width=12.0,       # meters
    theme="neon"
)
```

---

## Vehicle Physics

### Vehicle Configuration (`vehicles/vehicle.py`)
```python
from vehicles.presets import get_vehicle_preset, VEHICLE_PRESETS

# Built-in presets
config = get_vehicle_preset("balanced")   # All-rounder
config = get_vehicle_preset("speedster")  # High speed, low grip
config = get_vehicle_preset("gripper")    # High grip, lower speed
config = get_vehicle_preset("drift_king") # Drift-oriented
config = get_vehicle_preset("rookie")     # Forgiving

# Custom config
from vehicles.vehicle import VehicleConfig
custom = VehicleConfig(
    top_speed=50.0,
    acceleration=15.0,
    grip=1.2,
    drift_factor=0.4
)
```

### Physics Model
- **Pacejka-like tire model** (simplified)
- Weight transfer during acceleration/braking/cornering
- Drift/slide dynamics with recovery
- Surface-dependent grip modifiers
- Aerodynamic drag
- Optional boost/nitro

### Surface Grip Modifiers
| Surface | Grip Multiplier |
|---------|-----------------|
| asphalt | 1.0 |
| desert | 0.85 |
| snow | 0.4 |
| grassland | 0.9 |
| neon | 1.1 |
| dirt | 0.7 |
| grass | 0.3 |
| sand | 0.25 |
| ice | 0.15 |

---

## AI Training Environment

### Gymnasium Interface (`ai/training_env.py`)
```python
from ai.training_env import make_racing_env, TrainingConfig, ObservationMode, ActionMode

config = TrainingConfig(
    observation_mode=ObservationMode.VECTOR,
    action_mode=ActionMode.CONTINUOUS,
    rewards={...},  # Customize rewards
    termination={...}  # Customize termination
)

env = make_racing_env(config)

# Standard RL loop
obs, info = env.reset(seed=12345)
done = False
while not done:
    action = agent.act(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
```

### Observation Modes

#### Vector (Default)
```python
# ~80-dim vector containing:
# - Vehicle state (speed, slip, position, inputs, etc.)
# - Next checkpoint (distance, direction)
# - Racing line reference (offset, curvature, target speed)
# - Race progress (lap, checkpoints, time)
# - Nearby opponents (relative pos, speed)
```

#### Sensor (Raycast)
```python
# Configurable raycasts around vehicle
config.sensor_config = {
    'num_rays': 16,
    'max_distance': 50.0,
    'angles': [-pi, -3pi/4, -pi/2, -pi/4, 0, pi/4, pi/2, 3pi/4, pi]
}
```

#### Image
```python
config.image_resolution = (84, 84)
config.image_grayscale = True
# Returns top-down camera view (requires rendering)
```

#### Combined
Vector + Sensor concatenated.

### Action Spaces

#### Continuous (Default)
```python
# Box([-1, 0, 0], [1, 1, 1])
# [steering, throttle, brake]
# steering: -1.0 (left) to 1.0 (right)
# throttle: 0.0 to 1.0
# brake: 0.0 to 1.0
```

#### Discrete
```python
# 5 actions: LEFT, RIGHT, ACCELERATE, BRAKE, NONE
```

### Reward Configuration
```python
config.rewards = {
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
}
```

### Termination Conditions
```python
config.termination = {
    'max_offroad_time': 5.0,      # seconds
    'max_collisions': 10,
    'max_idle_time': 10.0,        # seconds
    'max_wrong_way_time': 3.0,    # seconds
    'min_speed_threshold': 1.0    # m/s
}
```

### Curriculum Learning
```python
config.curriculum = {
    'enabled': True,
    'stages': [
        {'name': 'easy', 'complexity': 0.2, 'road_width_mult': 1.5, 'speed_mult': 0.6, 'max_opponents': 0},
        {'name': 'medium', 'complexity': 0.4, 'road_width_mult': 1.2, 'speed_mult': 0.8, 'max_opponents': 1},
        {'name': 'hard', 'complexity': 0.6, 'road_width_mult': 1.0, 'speed_mult': 1.0, 'max_opponents': 3},
        {'name': 'expert', 'complexity': 0.8, 'road_width_mult': 0.8, 'speed_mult': 1.2, 'max_opponents': 4}
    ]
}
```

---

## NPC Opponent AI

### AI Profiles (`ai/npc_ai.py`)
```python
from ai.npc_ai import AIProfile, create_npc_ai

profiles = [
    AIProfile.BEGINNER,    # Slow, safe, frequent braking
    AIProfile.NORMAL,      # Balanced
    AIProfile.EXPERT,      # Efficient racing line
    AIProfile.AGGRESSIVE,  # Late braking, defensive
    AIProfile.DEFENSIVE,   # Protects position
    AIProfile.RANDOMIZED   # Procedural personality
]

npc_ai = create_npc_ai(vehicle, track, AIProfile.EXPERT, rng, physics_world)

# In simulation loop:
decision = npc_ai.update(dt, all_vehicles)
vehicle.set_input(decision.throttle, decision.brake, decision.steering, decision.boost)
```

### AI Behaviors
- **Pure Pursuit** path following with lookahead
- **Speed planning** based on curvature
- **Obstacle avoidance** via raycasts
- **Overtaking logic** with space checking
- **Defensive driving** when challenged
- **Recovery** from spins/off-track
- **Configurable mistakes** (error_rate)

---

## Race System

### Race Manager (`modes/race_system.py`)
Handles:
- Lap counting & checkpoint validation
- Position ranking (laps → checkpoints → progress)
- DNF detection
- Sector timing
- Catch-up/rubber-banding
- Ghost car recording

### Race States
- `WAITING` - Pre-race
- `COUNTDOWN` - 3, 2, 1, GO!
- `RACING` - Active race
- `FINISHED` - Complete
- `PAUSED` - Paused

### Ranking Algorithm
Cars ranked by:
1. Completed laps (descending)
2. Checkpoints crossed (descending)
3. Progress to next checkpoint (ascending distance)

---

## CLI Reference

### `scripts/game.py` - Main Game
```bash
python scripts/game.py [OPTIONS]

Options:
  --mode MODE           Game mode: training, 1v1, 1v4, ai_vs_ai (default: 1v1)
  --seed SEED           Track seed (int or string)
  --random-seed         Use random seed
  --daily-seed          Use today's daily seed
  --vehicle PRESET      Vehicle: balanced, speedster, gripper, drift_king, rookie
  --theme THEME         Track theme: asphalt, desert, snow, grassland, neon, industrial, dirt
  --complexity FLOAT    Track complexity 0.0-1.0 (default: 0.5)
  --road-width FLOAT    Road width in meters
  --laps INT            Number of laps (default: 3)
  --ai-profile PROFILE  AI: beginner, normal, expert, aggressive, defensive
  --headless            Run without rendering
  --catchup             Enable catch-up/rubber-banding
  --config FILE         Config file path
```

### `scripts/train.py` - Training
```bash
python scripts/train.py [OPTIONS]

Options:
  --mode MODE           training, 1v1, 1v4 (default: training)
  --seed SEED           Track seed
  --random-seed         Random seed
  --episodes INT        Training episodes (default: 100)
  --max-steps INT       Max steps per episode (default: 10000)
  --agent TYPE          Agent: random, heuristic, or model path
  --obs-mode MODE       vector, sensor, image, combined
  --action-mode MODE    continuous, discrete
  --curriculum/--no-curriculum  Enable curriculum (default: on)
  --curriculum-stage INT  Starting stage (default: 0)
  --vehicle PRESET      Vehicle preset
  --theme THEME         Track theme
  --complexity FLOAT    Track complexity
  --road-width FLOAT    Road width
  --output DIR          Output directory (default: training_results)
  --save-replays        Save replays (default: on)
  --replay-interval N   Save every N episodes (default: 100)
  --eval-interval N     Evaluate every N episodes (default: 100)
  --eval-episodes N     Episodes per eval (default: 10)
  --eval-seeds SEEDS    Evaluation seeds
  --config FILE         Config file
```

### `scripts/evaluate.py` - Evaluation
```bash
python scripts/evaluate.py [OPTIONS]

Options:
  --agent TYPE          Agent: random, heuristic, or model path
  --seeds SEEDS         Seeds to test (default: benchmark suite)
  --num-seeds N         Number of random seeds (default: 10)
  --episodes-per-seed N Episodes per seed (default: 1)
  --max-steps INT       Max steps (default: 10000)
  --render              Render evaluation
  --obs-mode MODE       vector, sensor, image, combined
  --action-mode MODE    continuous, discrete
  --config FILE         Config file
  --output DIR          Output directory (default: evaluation_results)
  --vehicle PRESET      Vehicle preset
  --theme THEME         Track theme
```

---

## Configuration

### Config File (`configs/default.yaml`)
All settings in YAML format:

```yaml
game:
  fixed_dt: 0.016666666666666666  # 60 Hz
  max_episode_steps: 10000
  max_race_time: 300.0
  physics_substeps: 4

track:
  min_complexity: 0.2
  max_complexity: 0.8
  min_road_width: 8.0
  max_road_width: 16.0
  themes:
    asphalt:
      surface_friction: 1.0
      offroad_friction: 0.3
      color_road: [50, 50, 50]
      ...

vehicle:
  presets:
    balanced:
      top_speed: 45.0
      acceleration: 12.0
      ...

training:
  observation_mode: "vector"
  sensor_config:
    num_rays: 16
    max_distance: 50.0
  rewards:
    progress_per_meter: 0.1
    checkpoint_bonus: 10.0
    ...
  termination:
    max_offroad_time: 5.0
    ...

npc:
  profiles:
    beginner:
      target_speed_factor: 0.5
      ...

render:
  width: 1280
  height: 720
  show_minimap: true
  show_debug_overlay: false

telemetry:
  enabled: true
  export_format: "json"
  save_replays: true
```

### Loading Custom Config
```python
from core.config import load_config, load_config_with_overrides

# Single config
config = load_config("configs/my_config.yaml")

# Base + overrides
config = load_config_with_overrides("configs/default.yaml", ["configs/override.yaml"])
```

---

## Telemetry & Replays

### Recording
Automatic during races:
```python
from telemetry.telemetry import TelemetryRecorder

recorder = TelemetryRecorder(config.telemetry)
recorder.start_recording(seed, mode, track_config, vehicle_config, ...)

# During simulation:
recorder.record_frame({
    'time': sim_time,
    'position': pos.tolist(),
    'speed': speed,
    'throttle': throttle,
    # ...
})

# After race:
recorder.finish_recording(results)
```

### Replay Files
Saved to `replays/` directory:
- `replay_{mode}_{seed}_{timestamp}.json` - Full data
- `replay_{mode}_{seed}_{timestamp}.csv` - Frame data only

### Playback
```python
from telemetry.telemetry import ReplayPlayer

player = ReplayPlayer.load("replays/replay_1v1_12345_20240101_120000.json")

# Get ghost data for rendering
ghost = player.get_ghost_data()

# Seek to time
frame = player.seek_time(30.0)  # 30 seconds in
```

### Export Formats

#### JSON
Complete replay with all metadata, RNG states, and frames.

#### CSV
Frame-by-frame telemetry for analysis:
```csv
step,time,pos_x,pos_y,angle,vel_x,vel_y,angular_vel,speed,forward_speed,lateral_speed,slip_angle,is_drifting,drift_intensity,on_road,surface_type,surface_grip,throttle,brake,steering,boost_active,lap,checkpoint,checkpoints_crossed,race_position,race_time,current_lap_time,best_lap_time,reward,total_reward,action_steering,action_throttle,action_brake
```

---

## Extending the Simulator

### Adding a New Vehicle Preset
```python
# In vehicles/presets.py
VEHICLE_PRESETS["my_car"] = VehicleConfig(
    top_speed=55.0,
    acceleration=13.0,
    brake_force=20.0,
    steering_sensitivity=3.8,
    grip=1.1,
    mass=1150.0,
    width=2.0,
    length=4.4,
    drift_factor=0.25,
    drag_coefficient=0.30,
    downforce=200.0
)
```

### Adding a New Track Theme
```python
# In configs/default.yaml, under track.themes:
my_theme:
  surface_friction: 0.95
  offroad_friction: 0.28
  color_road: [60, 60, 60]
  color_grass: [40, 100, 40]
  color_line: [255, 255, 200]
```

### Adding a New AI Profile
```python
# In ai/npc_ai.py
AI_PROFILES[AIProfile.MY_PROFILE] = AIProfileConfig(
    target_speed_factor=0.85,
    braking_distance_factor=1.2,
    reaction_time=0.1,
    overtake_threshold=0.6,
    aggression=0.6,
    error_rate=0.03,
    corner_speed_factor=0.9,
    follow_distance=12.0,
    defense_tendency=0.5,
    recovery_speed=1.1
)
```

### Custom Observation Space
```python
# Subclass RacingEnv and override _get_observation()
class MyRacingEnv(RacingEnv):
    def _get_observation(self):
        # Return custom observation
        return my_custom_observation()
```

### Custom Reward Function
```python
# In TrainingConfig.rewards dict or subclass RacingEnv:
def _calculate_reward(self):
    # Custom reward logic
    return my_reward
```

---

## Testing

### Run All Tests
```bash
python -m pytest tests/ -v
```

### Run Specific Test
```bash
python -m pytest tests/test_determinism.py::TestTrackGeneration::test_same_seed_same_track -v
```

### Test Categories
- `TestDeterministicRNG` - RNG determinism
- `TestMath` - Math utilities
- `TestTrackGeneration` - Track generation & validation
- `TestVehiclePhysics` - Vehicle physics determinism
- `TestPhysicsWorld` - Collision, raycast
- `TestSimulationContext` - Context state management
- `TestFullRaceDeterminism` - End-to-end race determinism

### Writing New Tests
```python
def test_my_feature(self):
    # Arrange
    rng = DeterministicRNG(12345)
    
    # Act
    result = my_function(rng)
    
    # Assert
    assert result == expected_value
```

---

## Troubleshooting

### Common Issues

#### "Module not found" errors
```bash
# Ensure you're in project root and venv is activated
cd /path/to/race
source venv/bin/activate
pip install -r requirements.txt
```

#### Pygame not working (headless servers)
```bash
# Install dummy video driver
export SDL_VIDEODRIVER=dummy
# Or run with --headless flag
python scripts/train.py --headless
```

#### Slow training
- Use `--headless` flag
- Reduce `physics_substeps` in config (but may affect stability)
- Use `ObservationMode.VECTOR` (fastest)
- Disable telemetry logging: `telemetry.enabled: false`

#### Determinism failures
- Ensure all RNGs use `DeterministicRNG` with proper seeds
- Check that no system uses `np.random` or `random` directly
- Verify `SimulationContext` is used for all random needs

#### Import errors in scripts
```bash
# Run from project root
cd /path/to/race
python scripts/train.py  # Not python scripts/train.py from elsewhere
```

### Debug Tools
- Press `F1` in-game to toggle debug overlay
- `F2` - Racing line
- `F3` - Sensors
- `F4` - Hitboxes
- `F5` - Minimap
- `+/-` - Zoom camera

### Logs & Replays
- Logs: `logs/` directory
- Replays: `replays/` directory
- Enable debug overlay to see real-time physics info

---

## Performance Tips

1. **Headless Training**: Always use `--headless` for training
2. **Vector Observations**: Fastest observation mode
3. **Batch Evaluation**: Run multiple seeds in sequence
4. **Fixed Timestep**: Don't change `fixed_dt` unless necessary
5. **Numba**: Install for faster math operations

---

## License

MIT License - See LICENSE file for details.

---

## Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit pull request

---

## Support

For issues, questions, or contributions:
- Check existing issues
- Review troubleshooting section
- Run tests to verify installation