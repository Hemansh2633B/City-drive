"""Run a small GPU evolution experiment and render the selected champion."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from ai.gpu_evolution import EvolutionConfig, GPUEvolutionTrainer, load_elite_policy
from ai.gpu_observations import build_observations
from ai.gpu_training_env import GPUTrainingEnv
from ai.training_env import RacingEnv, TrainingConfig
from core.config import load_config


def copy_champion_to_reference(gpu_env: GPUTrainingEnv, reference: RacingEnv) -> None:
    """Synchronize only slot 0 for visualization, outside the simulation hot path."""
    state = reference.vehicle.state
    body = reference.vehicle_body
    body.position = gpu_env.position[0].detach().cpu().numpy().astype(np.float64)
    body.angle = float(gpu_env.angle[0].detach().cpu())
    body.velocity = gpu_env.velocity[0].detach().cpu().numpy().astype(np.float64)
    body.angular_velocity = float(gpu_env.angular_velocity[0].detach().cpu())
    state.position = body.position.copy()
    state.angle = body.angle
    state.velocity = body.velocity.copy()
    state.angular_velocity = body.angular_velocity
    state.speed = float(gpu_env.speed[0].detach().cpu())
    state.forward_speed = float(gpu_env.forward_speed[0].detach().cpu())
    state.lateral_speed = float(gpu_env.lateral_speed[0].detach().cpu())
    state.local_velocity = np.array([state.forward_speed, state.lateral_speed])
    state.slip_angle = float(gpu_env.slip_angle[0].detach().cpu())
    state.is_drifting = bool(gpu_env.is_drifting[0].detach().cpu())
    state.drift_intensity = float(gpu_env.drift_intensity[0].detach().cpu())
    state.on_road = bool(gpu_env.on_road[0].detach().cpu())
    state.surface_type = reference.track.theme
    state.surface_grip = float(gpu_env.surface_grip[0].detach().cpu())
    state.throttle = float(gpu_env.throttle[0].detach().cpu())
    state.brake = float(gpu_env.brake[0].detach().cpu())
    state.steering = float(gpu_env.steering[0].detach().cpu())
    state.boost_active = bool(gpu_env.boost_active[0].detach().cpu())
    body._corners_dirty = True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--population-size", type=int, default=512)
    parser.add_argument("--simulation-batch-size", type=int, default=256)
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--rollout-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="training_runs/gpu_demo")
    parser.add_argument("--render-every", type=int, default=1)
    parser.add_argument(
        "--video-output",
        default=None,
        help="Optional MP4/GIF output path; requires the optional video extra.",
    )
    args = parser.parse_args()

    if args.population_size < 5 or args.simulation_batch_size < 1 or args.generations < 1 or args.rollout_steps < 1:
        parser.error("population size must be >= 5; batch size, generations and rollout steps must be positive")
    evo_config = EvolutionConfig(
        population_size=args.population_size,
        simulation_batch_size=args.simulation_batch_size,
        generations=args.generations,
        rollout_steps=args.rollout_steps,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    training_config = TrainingConfig()
    training_config.curriculum["enabled"] = False

    trainer = GPUEvolutionTrainer(evo_config, training_config, device=args.device)
    try:
        result = trainer.train()
    finally:
        trainer.close()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Cannot render champion: CUDA is not available.")
    reference = RacingEnv(
        training_config,
        load_config(),
        render_mode="rgb_array" if args.video_output else "human",
    )
    sim = None
    video_writer = None
    try:
        reference.reset(seed=args.seed + 9001)
        if args.video_output:
            try:
                import imageio.v2 as imageio
            except ImportError as exc:
                raise RuntimeError(
                    "Video capture requires the optional dependency: "
                    "python -m pip install -e '.[video]'"
                ) from exc
            video_writer = imageio.get_writer(args.video_output, fps=60)
        policy = load_elite_policy(
            Path(result["final_elite_checkpoint"]),
            device=args.device,
            seed=args.seed,
        )
        sim = GPUTrainingEnv.from_env(
            reference, batch_size=1, device=args.device
        )
        observations = sim.reset()
        candidate = torch.zeros(1, dtype=torch.long, device=sim.device)

        for step in range(args.rollout_steps):
            action = policy.actions(observations, candidate)
            observations, _, terminated, truncated, _ = sim.step(action)
            if step % max(args.render_every, 1) == 0:
                copy_champion_to_reference(sim, reference)
                frame = reference.render()
                if video_writer is not None:
                    if frame is None:
                        raise RuntimeError(
                            "Renderer returned no frame; install pygame-ce with "
                            "the render extra and ensure the selected track is initialized."
                        )
                    video_writer.append_data(frame)
            if bool(terminated[0].detach().cpu()) or bool(truncated[0].detach().cpu()):
                break
        print(
            f"Champion demo finished after {step + 1} steps; "
            f"reward={float(sim.episode_reward[0].detach().cpu()):.3f}, "
            f"laps={int(sim.laps_completed[0].detach().cpu())}."
        )
    finally:
        if video_writer is not None:
            video_writer.close()
            print(f"Champion video saved to {args.video_output}")
        if sim is not None:
            sim.close()
        reference.close()


if __name__ == "__main__":
    main()
