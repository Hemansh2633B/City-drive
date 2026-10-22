# City Drive RL

City Drive is a 2D top-down driving environment built for **single-agent reinforcement learning**. Its primary workflow is training and evaluating driving policies—not player-versus-player racing.

## Focused feature set

- **Gymnasium environment** with vector, ray-sensor, and combined observations.
- **PPO training** through Stable-Baselines3, with resume-from-checkpoint support.
- **Seeded procedural circuits**: repeat a seed to reproduce the same course.
- **Course validation**: generated centerlines form closed, non-self-intersecting radial loops; checkpoint spacing and spawn positions are validated.
- **Live training viewer** showing the map, learning agent, speed, accumulated reward, checkpoint count, lap completion, and seed.
- **Training records**: episode reward, rolling reward, episode length, laps/checkpoints, and map validity are written to CSV.
- **Checkpoints and experiment config** are saved to disk so runs can be reproduced or resumed.
- **Headless mode** is available by default for faster training. The viewer is optional.

## Install

Python 3.10 or newer is recommended.

```bash
git clone https://github.com/Hemansh2633B/City-drive.git
cd City-drive

python -m venv .venv
source .venv/bin/activate       # Linux/macOS
# .venv\Scripts\activate      # Windows

python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Train an agent

Start a reproducible PPO run with a live viewer:

```bash
python main.py train --seed "research-run-01" --timesteps 250000 --render
```

Run headless for speed:

```bash
python main.py train --seed 12345 --timesteps 100000
```

The trainer uses PPO by default. It writes:

```text
training_runs/city_drive/
├── checkpoints/        # periodic PPO checkpoints
├── metrics.csv         # one row per completed episode
├── run_config.json     # settings and normalized seed
└── final_model.zip     # final trained model
```

To continue from a saved checkpoint:

```bash
python main.py train \
  --resume training_runs/city_drive/checkpoints/city_drive_ppo_25000_steps.zip \
  --seed 12345 --timesteps 100000
```

To smoke-test the environment without training a neural policy:

```bash
python main.py train --agent random --seed "smoke-test" --timesteps 20000 --render
```

The random baseline is only for validating the environment and viewer; it does not learn.

## Track and environment settings

```bash
python main.py train \
  --seed "curvy-map-07" \
  --theme asphalt \
  --complexity 0.55 \
  --road-width 12 \
  --obs-mode vector \
  --action-mode continuous \
  --timesteps 300000
```

- **Seed** accepts an integer or text. Reusing it with the same configuration reproduces the initial track and random stream.
- **Complexity** is clamped to `0.0–1.0`. Seeded shape harmonics and phases vary each closed course while keeping the radius positive, avoiding self-intersections from track closure.
- **Road width** is set in simulation metres; the supported range is 8–16 m.
- **Observations**: `vector` (default), `sensor` (ray distances plus vehicle state), or `combined`.
- **Actions**: `continuous` (steering, throttle, brake) or `discrete`.
- **Curriculum** changes difficulty over time without adding NPCs or opponent cars.

For faster iteration, lower `--timesteps` and use `--no-curriculum`. To slow down or speed up PPO, change `--learning-rate`, `--n-steps`, and `--batch-size`.

## View training metrics

Every completed episode appends a row to `metrics.csv`. Columns include total reward, rolling mean reward, step count, checkpoints crossed, laps completed, track validity, collision count, and off-road time.

Optional TensorBoard logging:

```bash
pip install tensorboard
python main.py train --tensorboard --timesteps 250000
tensorboard --logdir training_runs/city_drive/tensorboard
```

## Test

```bash
python main.py test
# or
pytest tests/ -q
```

## Project layout

```text
ai/             Gymnasium training environment
core/           Physics, math, deterministic random streams, config
track/          Seeded course generation, validation, checkpoints
vehicles/       Vehicle dynamics and presets
scripts/train.py PPO trainer, checkpoints, metrics and live rendering
scripts/evaluate.py
ui/             Interactive viewer/menu components
telemetry/      Telemetry and replay support
tests/          Determinism and RL environment regression tests
```

The main launcher exposes `train`, `eval`, and `test`. Competitive race modes are not part of the primary workflow.
