# City Drive RL

**A small, seeded driving environment for Gymnasium and reinforcement-learning experiments.**

The goal is a Gym-style workflow: install the environment, create it with `gym.make()`, call `reset()` and `step()`, and plug in your preferred agent. Competitive race modes are not part of the main workflow.

## Install

### Install the environment only

This is the lightest option. The simulation and environment run on the CPU; it does not require PyTorch or pygame.

```bash
python -m pip install "city-drive-rl @ git+https://github.com/Hemansh2633B/City-drive.git@ai-training-focus"
```

### Install PPO training and the live viewer

```bash
python -m pip install "city-drive-rl[full] @ git+https://github.com/Hemansh2633B/City-drive.git@ai-training-focus"
```

The ref above installs the current refactor branch. The project is **not yet published on PyPI**, so `pip install city-drive-rl` by itself will not work until a release is published.

### Install from a local clone (development)

```bash
git clone https://github.com/Hemansh2633B/City-drive.git
cd City-drive
git switch ai-training-focus
python -m pip install -e ".[full,dev]"
```

Use a virtual environment if other Python projects on your machine need different dependencies.

## 1. Minimal Gymnasium example

```python
import gymnasium as gym
import city_drive_rl  # registers CityDrive-v0

env = gym.make("CityDrive-v0")
observation, info = env.reset(seed=42)

for _ in range(1_000):
    action = env.action_space.sample()  # replace this with your agent
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()  # next episode, new deterministic RNG draw

env.close()
```

This is the basic Gymnasium API. The action is either a 3-value continuous vector (steering, throttle, brake) or a discrete action depending on configuration. The default environment uses continuous actions and a vector observation.

## 2. Configure a course

```python
import gymnasium as gym
import city_drive_rl
from ai.training_env import TrainingConfig, ObservationMode, ActionMode

config = TrainingConfig(
    observation_mode=ObservationMode.VECTOR,
    action_mode=ActionMode.CONTINUOUS,
    max_episode_steps=5_000,
)
config.track_theme = "asphalt"
config.track_complexity = 0.45
config.road_width = 12.0
config.curriculum["enabled"] = False

env = gym.make("CityDrive-v0", config=config)
observation, info = env.reset(seed="practice-map-01")
print(info["track_valid"], info["seed"])
env.close()
```

Supported observations are `vector`, `sensor`, and `combined`. Image observations are rejected explicitly for now rather than returning fake all-zero images. Supported actions are `continuous` and `discrete`.

## 3. Train PPO

After installing the `full` extra:

```bash
city-drive-train --seed "research-run-01" --timesteps 250000 --render
```

Or run headless for less overhead:

```bash
city-drive-train --seed 12345 --timesteps 100000 --device cpu
```

The trainer uses Stable-Baselines3 PPO. It supports model resume, periodic checkpoints, progress output, episode metrics, optional TensorBoard logs, and the live viewer.

Example custom training with Stable-Baselines3:

```python
import gymnasium as gym
import city_drive_rl
from stable_baselines3 import PPO

env = gym.make("CityDrive-v0")
model = PPO("MlpPolicy", env, device="auto", verbose=1)
model.learn(total_timesteps=250_000)
model.save("city_drive_ppo")
env.close()
```

Outputs from the CLI go to `training_runs/city_drive/` by default:

```text
checkpoints/       # periodic model checkpoints
metrics.csv        # completed episode metrics and rolling reward
run_config.json    # run settings and seed
final_model.zip    # final PPO model
```

Use `--resume path/to/checkpoint.zip` to resume a saved PPO run. `--agent random` is only an environment/viewer smoke test; it does not learn.

## CPU and GPU

- **The simulator is CPU-based.** Track generation, vehicle physics and Gymnasium stepping use NumPy/Python on the CPU.
- **The policy can train on CPU or a supported GPU.** Use `--device cpu`, `--device cuda`, or `--device auto` for PPO. `auto` selects CUDA only when the installed PyTorch build and driver support it.
- **CUDA is optional.** Install the correct PyTorch build for your operating system, GPU and driver using the official [PyTorch install selector](https://pytorch.org/get-started/locally/), then install this project’s `rl` or `full` extra. A GPU flag cannot add GPU support to a CPU-only PyTorch build.
- CPU-only users can run and train small policies without CUDA. Rendering is optional and requires the `render` extra.

The physics engine itself does not become GPU-accelerated just because the PPO policy uses a GPU; vectorized environments and faster physics would be separate performance improvements.

## Reproducibility and course validation

Pass an integer or text seed to `reset(seed=...)`. The same seed and same environment settings reproduce the same track and spawn. A new seed selects a new procedural course.

The course generator builds closed radial-loop circuits with bounded variation, then validates closure, checkpoint spacing and the starting position. Checkpoints are counted only when the car crosses the gate in the forward direction. Course generation is designed to avoid self-intersecting centerlines by construction; that is not a proof that every driving policy can finish every course.

Curriculum learning is optional. It changes course difficulty gradually and **does not add NPC opponents** to the training environment.

## See training live

```bash
city-drive-train --seed 42 --timesteps 100000 --render
```

The pygame viewer shows the course, agent, seed, speed, reward, checkpoint progress, lap count and map validity. Close the viewer to stop displaying frames; use headless mode for faster runs. Per-episode metrics are written to CSV and can also be opened in a spreadsheet.

Optional TensorBoard:

```bash
python -m pip install "city-drive-rl[full,tensorboard] @ git+https://github.com/Hemansh2633B/City-drive.git@ai-training-focus"
city-drive-train --tensorboard --timesteps 250000
tensorboard --logdir training_runs/city_drive/tensorboard
```

## Test the environment

From the repository root:

```bash
python -m pip install -e ".[full,dev]"
python -m pytest tests/ -q
```

GitHub Actions runs the environment tests on supported Python versions. Test the exact environment and hardware combination you plan to use before starting long training runs.

## Project layout

- `city_drive_rl/`: Gymnasium environment registration
- `ai/training_env.py`: observations, actions, rewards, reset/step and viewer
- `track/`: seeded track generation, validation and checkpoints
- `core/`: physics, geometry, deterministic RNG and configuration
- `scripts/train.py`: PPO CLI, checkpoint saves and metrics
- `tests/`: determinism and RL API regression tests

## License

MIT. See [LICENSE](LICENSE).


## GPU-native batched simulation and evolution

The default Gymnasium `RacingEnv` remains the CPU reference environment. An optional
PyTorch tensor backend is available for high-throughput evolutionary policy training:
it stores independent vehicle state for thousands of agents on one explicit device
and does not call `RacingEnv.step()` once per candidate.

Install PyTorch with a build appropriate for your GPU and driver, then install the
optional training dependencies:

```bash
python -m pip install -e ".[gpu,dev,render]"
```

Run the 10,000-policy example (batching controls concurrent simulation memory):

```bash
python scripts/train_gpu_evolution.py --config configs/gpu_evolution.yaml
```

Run a measured throughput / memory benchmark and save machine-readable JSON:

```bash
python scripts/benchmark_gpu_env.py --device cuda \\
  --batches 1 64 256 1024 4096 10000 \\
  --steps 250 --output training_runs/gpu_benchmark.json
```

For a small end-to-end demo that trains a population and renders the selected
champion, use a desktop/display environment with the render extra installed:

```bash
python scripts/demo_gpu_training.py --device cuda \\
  --population-size 512 --simulation-batch-size 256 \\
  --generations 3 --rollout-steps 1000
```

The environment backend defaults to CUDA and fails loudly if CUDA is not available;
`--device cpu` is an explicit debugging mode, not an automatic fallback. Vector,
sensor and combined observations are supported; image observations are not. The
existing focused single-agent environment has no static track-boundary or opponent
collision bodies, so its collision counter does not currently distinguish policies.
The GPU design, source-derived behavior, known limits, commands and benchmark schema
are documented in [docs/GPU_TRAINING.md](docs/GPU_TRAINING.md). CUDA throughput and
speedup should only be claimed from an executed benchmark on the target hardware.
