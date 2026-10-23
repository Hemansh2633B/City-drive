# GPU-native batched simulation and evolutionary training

## Environment audit and source semantics

The reference workflow is `RacingEnv.reset()` and `RacingEnv.step()` in
`ai/training_env.py`. Reset creates the deterministic simulation context,
generates/validates a track on the CPU, creates one balanced vehicle and its
rigid body, and initializes checkpoint/reward counters. A step clamps the
action, calls `PhysicsWorld.step(dt)`, updates vehicle state and computes next
step forces in `Vehicle.step()`, checks the nearest centerline segment for
road membership, applies reward/checkpoint/lap logic, checks termination, then
constructs observations and info.

Important source-derived implementation details:

- Vehicle defaults use a 2 m width and 4.5 m length; default track theme is
  asphalt and simulation time-step is 1/60 s with four physics substeps.
- Rigid-body integration uses semi-implicit Euler. `RigidBody.integrate()`
  clears force/torque after each substep, so the accumulated vehicle force is
  applied at the first substep only. The tensor backend preserves this order.
- `Vehicle._calculate_forces()` includes throttle/torque curve, braking,
  reverse force, quadratic drag, rolling resistance, lateral tire force,
  steering torque, angular damping and drift-recovery torque. It computes a
  weight-transfer value but does not apply it.
- Progress reward is Euclidean movement distance with forward/backward weighting.
  Additional terms are speed, forward checkpoint crossing, lap completion,
  collision penalty, off-road duration, reverse driving, drifting, idling and
  wrong-way driving. Termination covers lap completion, max steps/time,
  excessive off-road time, collision count, idle time and wrong-way duration.
- Vector observations have 80 values; current populated features occupy the
  initial 24 entries and remaining entries are zero-padded. Sensor mode concatenates
  normalized ray distances and six vehicle state values.
- In the focused RL environment, reset explicitly sets `max_opponents = 0`.
  The physics world receives the player body only; track boundaries are stored
  for rendering/geometry, not inserted into the collision world as static bodies.
  Thus there are no player-vs-track-boundary or agent-vs-agent rigid-body collision
  tests in the existing training path. Its road-departure condition is nearest
  centerline distance <= `road_width * 0.5 * 1.1`. The GPU port preserves this
  rule and does not invent a collision penalty or approximate it as a collision.
- The current physics world has no static track raycast surfaces; sensor rays in
  this single-body configuration report maximum range. The GPU observation path
  reproduces that actual behavior rather than pretending it has a road sensor
  that the reference implementation does not supply.

## Backend architecture

`ai/gpu_training_env.py` maintains one persistent set of tensors per simulation
slot. State, actions, observation vectors, reward values, checkpoint/lap indices
and episode flags are stored on the explicit device. The default is CUDA and
fails loudly if CUDA is unavailable; `device="cpu"` is an explicit debugging
backend, not an automatic fallback.

- `ai/gpu_physics.py`: batched rigid-body integration and vectorized vehicle-force
  formulas, using float32 by default (float64 supported for parity debugging).
- `ai/gpu_collision.py`: chunked batched distance-to-centerline and forward gate
  crossing geometry. Immutable geometry is uploaded once when the track changes.
- `ai/gpu_observations.py`: vector/sensor/combined observations constructed from
  tensor state without a CPU policy loop.
- `ai/gpu_rewards.py`: reward, checkpoint/lap updates and termination masks.
- `ai/gpu_evolution.py`: stacked per-candidate MLP weights, batched policy inference,
  chunked candidate evaluation, top-five selection, cloning, Gaussian mutation,
  checkpoints and candidate metrics.
- `scripts/train_gpu_evolution.py`: configurable CLI.
- `scripts/benchmark_gpu_env.py`: CPU reference and tensor pipeline benchmark.
- `scripts/demo_gpu_training.py`: training plus selected-champion viewer.

Population size is independent of concurrent simulation batch size. A 10,000-policy
population can be evaluated in chunks of 1,024 slots while candidate weights stay
on device. The last partial chunk is padded by repeating one candidate; padded
results are discarded. A candidate's policy weight tensors and its environment
slot are paired by index.

## Installation and usage

Install the correct PyTorch build for the host's GPU/driver first using the
official PyTorch install selector. The optional `gpu` extra installs PyTorch
from the standard Python package index, but environment-specific CUDA wheel
selection may still be needed on some hosts.

```bash
python -m pip install -e ".[gpu,dev,render]"
python -m pytest tests/test_gpu_parity.py tests/test_gpu_evolution.py -q
```

Run the 10k-policy example:

```bash
python scripts/train_gpu_evolution.py --config configs/gpu_evolution.yaml
```

For video output, install the optional imageio/FFmpeg extra and pass `--video-output`
to the demo. This records only the selected champion, not the full population:

```bash
python -m pip install -e ".[gpu,render,video]"
python scripts/demo_gpu_training.py --device cuda --video-output champion.mp4
```

Start with a smaller smoke test on CUDA:

```bash
python scripts/train_gpu_evolution.py --device cuda \\
  --population-size 128 --simulation-batch-size 64 \\
  --generations 2 --rollout-steps 200 --output-dir training_runs/gpu_smoke
```

Benchmark end-to-end policy inference + environment stepping:

```bash
python scripts/benchmark_gpu_env.py --device cuda --steps 250 \\
  --batches 1 64 256 1024 4096 10000 \\
  --output training_runs/gpu_benchmark.json
```

CPU tensor-backend debugging (does not claim CUDA performance):

```bash
python scripts/train_gpu_evolution.py --device cpu \\
  --population-size 32 --simulation-batch-size 16 \\
  --generations 1 --rollout-steps 50 --output-dir /tmp/city_drive_gpu
python scripts/benchmark_gpu_env.py --device cpu --batches 1 64 --steps 25
```

Train a small run and visualize the final selected champion (requires the render
extra and a display):

```bash
python scripts/demo_gpu_training.py --device cuda \\
  --population-size 512 --simulation-batch-size 256 \\
  --generations 3 --rollout-steps 1000 --render-every 1
```

The viewer syncs only the champion state for visualization; it does not step the
CPU environment while the GPU rollout is running.

## Fitness and collision eligibility

Fitness is configurable through `fitness_weights` in
`configs/gpu_evolution.yaml`. It is the weighted sum of per-seed averages for
`reward`, `distance_traveled`, `laps_completed`, `offroad_time`, and
`collision_count`. The default preserves cumulative reward ranking
(`reward: 1.0`; all other weights `0.0`). Cost-like metrics such as off-road
time and collisions should normally receive negative weights. Every weight must
be finite; unknown component names are rejected.

All fitness components are averaged across the configured track seeds. All candidates see the same tracks within a generation;
track seeds change across generations to reduce overfitting. With `elitism: true` (the default), the chosen top-five candidates are copied
unchanged into the next generation before mutated offspring refill the rest of the
population. With `elitism: false`, all slots become mutated clones of selected
parents; `keep_previous_elites` is then rejected because no previous elites are
preserved.

The elite selector first restricts ranking to candidates with
`collision_count == 0` (when at least five qualify), then selects five by the
configured fitness. If fewer
than five are eligible, the generation is marked `insufficient_collision_free_elites`
and the explicit `collision_fallback` applies:

- `allow_collisions`: fill remaining elite slots by fitness rank (and keep the
  flag in history/checkpoints).
- `keep_previous_elites`: preserve last generation's elite slots; this errors
  on the first generation because no prior elite set exists.
- `raise`: fail immediately rather than weakening eligibility.

Because the audited single-agent reference environment currently has no collision
bodies, `collision_count` remains zero for all candidates under the existing
physics setup. Zero-collision eligibility therefore does not distinguish policies
until the base environment adds real collision-body semantics. Off-road time and
lap/progress metrics are logged separately and are not silently re-labeled as
collisions.

## Validation and performance claims

CPU-versus-tensor tests compare the reference reset, one-step rewards and
observations, centerline distance, forward checkpoint crossings, observation
shapes, seeded behavior, partial resets, and inactive-slot independence. The
tests use numerical tolerances because PyTorch and NumPy can differ slightly in
floating point calculations.

This implementation does not claim measured CUDA parity, throughput, or speedup
from code inspection alone. The benchmark outputs per-batch wall time,
simulation-steps/s, agent-steps/s, completed episodes and episodes/s (when an
episode finishes in the timed horizon), isolated policy inference, per-step physics,
road-boundary, reward, termination and observation timings, initialization/track
generation/geometry-transfer time, peak allocated and reserved CUDA memory,
device/version metadata, best-effort sampled GPU utilization, and process CPU
utilization normalized to host logical-core capacity. It uses CUDA synchronization
around timed regions. CPU reference comparison is capped at 64
agents by default because it instantiates a separate Python/Gym environment per
agent; use `--cpu-max-batch` to expand it deliberately. Sizes skipped or failing allocation are marked as such rather than filled with fabricated numbers.
GPU utilization is reported as a best-effort sample when the driver exposes it, not
as a time-averaged utilization trace.

Remaining constraints: track generation is CPU-side and transfers immutable
geometry once per track; image observations are unsupported; the current reference
environment does not model static boundary or inter-agent collision bodies;
rendering is CPU-side and synchronizes only the chosen champion; CUDA-specific
tests and performance must be run on the target GPU. torch.compile, CUDA graphs
and custom kernels remain follow-up optimizations only if profiling justifies them.
