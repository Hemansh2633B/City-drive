"""CLI entry point for the GPU-native evolutionary training backend."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml
import torch

from ai.gpu_evolution import EvolutionConfig, GPUEvolutionTrainer
from ai.training_env import ActionMode, ObservationMode, TrainingConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/gpu_evolution.yaml")
    parser.add_argument("--device", choices=("cuda", "cpu"), default=None)
    parser.add_argument("--population-size", type=int, default=None)
    parser.add_argument("--simulation-batch-size", type=int, default=None)
    parser.add_argument("--generations", type=int, default=None)
    parser.add_argument("--rollout-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config)
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}

    configured_device = data.pop("device", "cuda")
    device = args.device or configured_device
    obs_name = data.pop("observation_mode", "vector")
    max_episode_steps = int(data.pop("max_episode_steps", 5000))
    track_theme = data.pop("track_theme", "asphalt")
    track_complexity = data.pop("track_complexity", 0.45)
    road_width = data.pop("road_width", 12.0)
    dtype_name = data.pop("dtype", "float32")
    dtype = torch.float64 if str(dtype_name).lower() == "float64" else torch.float32

    overrides = {
        "population_size": args.population_size,
        "simulation_batch_size": args.simulation_batch_size,
        "generations": args.generations,
        "rollout_steps": args.rollout_steps,
        "seed": args.seed,
        "output_dir": args.output_dir,
    }
    for key, value in overrides.items():
        if value is not None:
            data[key] = value
    if "track_seeds" in data:
        data["track_seeds"] = tuple(int(value) for value in data["track_seeds"])
    data["dtype"] = dtype
    evolution_config = EvolutionConfig(**data)

    training_config = TrainingConfig(
        observation_mode=ObservationMode(str(obs_name).lower()),
        action_mode=ActionMode.CONTINUOUS,
        max_episode_steps=max_episode_steps,
        track_theme=str(track_theme),
        track_complexity=float(track_complexity),
        road_width=float(road_width),
    )
    training_config.curriculum["enabled"] = False

    trainer = GPUEvolutionTrainer(
        evolution_config,
        training_config,
        device=device,
    )
    try:
        result = trainer.train()
        print(f"Final elite checkpoint: {result['final_elite_checkpoint']}")
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
