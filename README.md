# AI Racing Simulator

A 2D top-down racing simulator designed for AI agent training, reinforcement learning research, and autonomous driving benchmarking.

## Features

- **Deterministic Simulation** - Same seed always produces identical results for reproducible experiments
- **AI Training Ready** - Gymnasium-compatible environment with vector/sensor/image observations
- **Procedural Tracks** - Seed-based generation with validation (straights, hairpins, chicanes, S-curves)
- **Vehicle Physics** - Pacejka-like tire model, drift dynamics, surface-dependent grip
- **Multiple Game Modes** - Training (single car), 1v1, 1v4 races
- **Scripted NPC AI** - 6 difficulty profiles (Beginner to Expert, Aggressive, Defensive)
- **Curriculum Learning** - 4 difficulty stages that progressively increase complexity
- **Headless & GUI** - Fast headless training or visual pygame rendering
- **Telemetry & Replays** - JSON/CSV export, exact replay capability

## Quick Start

### Installation

```bash
git clone https://github.com/yourusername/ai-racing-simulator.git
cd ai-racing-simulator

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run a Race (with GUI)

```bash
# 1v1 race against AI
python scripts/game.py --mode 1v1 --seed 12345

# 1v4 race
python scripts/game.py --mode 1v4 --seed "daily-race-001"
```

### Train an AI Agent (Headless)

```bash
# Random agent baseline
python scripts/train.py --mode training --seed 12345 --episodes 100

# With custom agent
python scripts/train.py --agent path/to/model.zip --episodes 1000
```

### Evaluate Agents

```bash
# Benchmark on standard seeds
python scripts/evaluate.py --agent random --seeds 1 2 3 4 5

# Evaluate trained model
python scripts/evaluate.py --agent path/to/model.zip --num-seeds 20
```

## Controls (GUI Mode)

| Key | Action |
|-----|--------|
| ↑ / W | Throttle |
| ↓ / S | Brake / Reverse |
| ← / A | Steer Left |
| → / D | Steer Right |
| Shift / Space | Boost (if available) |
| ESC | Pause Menu |
| F1 | Toggle Debug Overlay |
| F2 | Toggle Racing Line |
| F3 | Toggle Sensors |
| F4 | Toggle Hitboxes |
| F5 | Toggle Minimap |
| + / - | Zoom Camera |

## Game Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| **Training** | Single car, no opponents, configurable rewards | RL training, curriculum learning |
| **1v1** | Player vs 1 AI opponent | Human vs AI, AI vs AI testing |
| **1v4** | Player vs 4 AI opponents | Competitive racing, overtaking logic |
| **AI vs AI** | 5 AI cars racing | Benchmarking, tournaments |

## AI Training

### Observation Modes

```python
from ai.training_env import RacingEnv, TrainingConfig, ObservationMode

config = TrainingConfig(
    observation_mode=ObservationMode.VECTOR,    # 80-dim state vector
    # observation_mode=ObservationMode.SENSOR,   # 16 raycasts + state
    # observation_mode=ObservationMode.IMAGE,    # 84x84 top-down camera
    # observation_mode=ObservationMode.COMBINED, # Vector + Sensor
)
```

### Action Spaces

```python
# Continuous: [steering, throttle, brake] in [-1,1] x [0,1] x [0,1]
# Discrete: 5 actions (LEFT, RIGHT, ACCELERATE, BRAKE, NONE)
```

### Reward Configuration

```python
config = TrainingConfig(
    rewards={
        'progress_per_meter': 0.1,
        'checkpoint_bonus': 10.0,
        'lap_bonus': 100.0,
        'collision_penalty': -5.0,
        'offroad_penalty_per_second': -2.0,
        # ... more rewards
    }
)
```

### Curriculum Learning

4 built-in stages that auto-advance:

| Stage | Complexity | Road Width | Speed | Opponents |
|-------|------------|------------|-------|-----------|
| Easy | 0.2 | 1.5x | 0.6x | 0 |
| Medium | 0.4 | 1.2x | 0.8x | 1 |
| Hard | 0.6 | 1.0x | 1.0x | 3 |
| Expert | 0.8 | 0.8x | 1.2x | 4 |

## Track Generation

Tracks are generated from seeds (integer or string):

```python
from track.generator import TrackGenerator, TrackGeneratorConfig
from core.random import DeterministicRNG

config = TrackGeneratorConfig()
rng = DeterministicRNG(12345)
generator = TrackGenerator(config, rng)

track = generator.generate(
    seed=12345,           # or "my-custom-seed"
    theme="asphalt",      # asphalt, desert, snow, grassland, neon, industrial, dirt
    complexity=0.5,       # 0.0 (simple) to 1.0 (complex)
    road_width=12.0       # meters
)
```

**Themes:** asphalt, desert, snow, grassland, neon, industrial, dirt  
**Benchmark Seeds:** 1, 42, 123, 456, 789, 1000-1004, "daily-race-001", etc.

## Project Structure

```
race/
├── core/           # Deterministic RNG, config, math, physics
├── track/          # Procedural track generation & validation
├── vehicles/       # Vehicle physics, presets
├── ai/             # Training env (Gymnasium), NPC AI
├── modes/          # Training, 1v1, 1v4 race modes
├── ui/             # Pygame renderer, menus, HUD
├── telemetry/      # JSON/CSV replay system
├── scripts/        # CLI entry points
├── configs/        # YAML configuration
├── tests/          # 25 determinism tests
└── docs/           # Documentation
```

## Configuration

Edit `configs/default.yaml` or create custom configs:

```yaml
game:
  fixed_dt: 0.016666666666666666  # 60 Hz physics
  max_episode_steps: 10000

vehicle:
  presets:
    balanced:
      top_speed: 45.0
      acceleration: 12.0
      grip: 1.0
    # ... speedster, gripper, drift_king, rookie

training:
  curriculum:
    enabled: true
    # ... stages

render:
  width: 1280
  height: 720
  show_minimap: true
  show_debug_overlay: false
```

## Reproducibility

Every race is fully deterministic. Same seed = identical results:

```bash
# These produce EXACTLY the same race
python scripts/game.py --seed 12345
python scripts/game.py --seed 12345
```

Replays include complete state for exact reproduction.

## Running Tests

```bash
# All tests
python -m pytest tests/ -v

# Specific test class
python -m pytest tests/test_determinism.py::TestTrackGeneration -v
```

## Extending

### Add Vehicle Preset
```python
# vehicles/presets.py
VEHICLE_PRESETS["my_car"] = VehicleConfig(
    top_speed=55.0,
    acceleration=13.0,
    grip=1.1,
    # ...
)
```

### Add AI Profile
```python
# ai/npc_ai.py
AI_PROFILES[AIProfile.MY_PROFILE] = AIProfileConfig(
    target_speed_factor=0.85,
    aggression=0.6,
    # ...
)
```

### Custom Track Theme
```yaml
# configs/default.yaml
track:
  themes:
    my_theme:
      surface_friction: 0.95
      offroad_friction: 0.28
      color_road: [60, 60, 60]
      # ...
```

## Requirements

- Python 3.10+
- numpy >= 1.24
- gymnasium >= 0.29
- pygame-ce >= 2.5 (for GUI)
- pyyaml >= 6.0

Optional:
- numba >= 0.58 (faster math)
- stable-baselines3 >= 2.0 (RL algorithms)

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

If you use this simulator in research, please cite:

```bibtex
@software{ai_racing_simulator,
  title = {AI Racing Simulator},
  author = {AI Racing Simulator Contributors},
  year = {2025},
  url = {https://github.com/yourusername/ai-racing-simulator}
}
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## Support

- Check [documentation](docs/README.md) for detailed guides
- Review [troubleshooting](docs/README.md#troubleshooting) for common issues
- Open an issue for bugs or feature requests