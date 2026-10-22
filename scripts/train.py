#!/usr/bin/env python3
"""
Focused reinforcement-learning trainer for City Drive.

The default path trains a PPO policy in the single-agent Gymnasium environment,
saves periodic checkpoints, records episode metrics, and can show the live car.
"""

import argparse
import csv
import hashlib
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.config import load_config
from ai.training_env import RacingEnv, TrainingConfig, ObservationMode, ActionMode


def parse_seed(value: str) -> int:
    """Convert integer or readable seed text into a stable 32-bit integer."""
    try:
        return int(value) % (2 ** 32)
    except (TypeError, ValueError):
        return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:4], "big")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a single-agent driving policy with reproducible seeded tracks."
    )
    parser.add_argument("--seed", default="12345",
                        help="Integer or text seed (same seed reproduces the same run)")
    parser.add_argument("--timesteps", type=int, default=250_000,
                        help="Total environment steps")
    parser.add_argument("--max-steps", type=int, default=5000,
                        help="Maximum steps per episode")
    parser.add_argument("--agent", choices=["ppo", "random"], default="ppo",
                        help="PPO trains a policy; random is only a smoke-test baseline")
    parser.add_argument("--resume", default=None,
                        help="Resume training from a Stable-Baselines3 PPO .zip checkpoint")
    parser.add_argument("--obs-mode", choices=["vector", "sensor", "combined"],
                        default="vector", help="Observation representation")
    parser.add_argument("--action-mode", choices=["continuous", "discrete"],
                        default="continuous", help="Vehicle control action space")
    parser.add_argument("--theme",
                        choices=["asphalt", "desert", "snow", "grassland", "neon", "industrial", "dirt"],
                        default="asphalt", help="Visual/surface theme for generated tracks")
    parser.add_argument("--complexity", type=float, default=None,
                        help="Pin track shape complexity to 0-1 (otherwise curriculum controls it)")
    parser.add_argument("--road-width", type=float, default=None,
                        help="Pin road width to 8-16 metres (otherwise curriculum controls it)")
    parser.add_argument("--no-curriculum", action="store_true",
                        help="Keep track difficulty constant")
    parser.add_argument("--render", action="store_true",
                        help="Show the live top-down training window")
    parser.add_argument("--render-every", type=int, default=4,
                        help="Render every N environment steps when --render is enabled")
    parser.add_argument("--checkpoint-interval", type=int, default=25_000,
                        help="Save a PPO checkpoint every N steps")
    parser.add_argument("--output", default="training_runs/city_drive",
                        help="Directory for metrics, logs, and saved models")
    parser.add_argument("--tensorboard", action="store_true",
                        help="Write TensorBoard logs (requires tensorboard)")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=2048,
                        help="PPO rollout length")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto",
                        help="Stable-Baselines3 device: auto, cpu, or cuda")
    return parser.parse_args()


METRIC_FIELDS = [
    "episode", "timesteps", "reward", "rolling_reward", "length",
    "laps_completed", "checkpoints_crossed", "seed", "track_valid",
    "collisions", "offroad_time"
]


def create_training_config(args) -> TrainingConfig:
    config = TrainingConfig()
    config.observation_mode = ObservationMode(args.obs_mode)
    config.action_mode = ActionMode(args.action_mode)
    config.max_episode_steps = max(1, int(args.max_steps))
    config.track_theme = args.theme
    config.track_complexity = (
        None if args.complexity is None
        else min(1.0, max(0.0, float(args.complexity)))
    )
    config.road_width = (
        None if args.road_width is None
        else max(8.0, min(16.0, float(args.road_width)))
    )
    config.curriculum["enabled"] = not args.no_curriculum
    # Never inject NPCs/opponents into a single-agent learning task.
    for stage in config.curriculum.get("stages", []):
        stage["max_opponents"] = 0
    return config


def write_csv_row(path: Path, record: dict) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=METRIC_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({key: record.get(key, "") for key in METRIC_FIELDS})


def run_random_baseline(args, training_config, game_config, output_dir, seed):
    """Run a random-policy smoke test so the environment can be checked without SB3."""
    env = RacingEnv(
        config=training_config,
        game_config=game_config,
        render_mode="human" if args.render else None
    )
    obs, info = env.reset(seed=seed)
    if args.render:
        env.render()

    rewards = []
    episode_reward = 0.0
    episode_steps = 0
    episode_number = 0
    for step in range(max(1, args.timesteps)):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += float(reward)
        episode_steps += 1

        if args.render and step % max(1, args.render_every) == 0:
            env.render()

        if terminated or truncated:
            episode_number += 1
            rewards.append(episode_reward)
            rolling = float(np.mean(rewards[-20:]))
            record = {
                "episode": episode_number,
                "timesteps": step + 1,
                "reward": round(episode_reward, 4),
                "rolling_reward": round(rolling, 4),
                "length": episode_steps,
                "laps_completed": info.get("laps_completed", 0),
                "checkpoints_crossed": info.get("checkpoints_crossed", 0),
                "seed": info.get("seed", seed),
                "track_valid": info.get("track_valid", False),
                "collisions": info.get("collision_count", 0),
                "offroad_time": info.get("offroad_time", 0.0)
            }
            write_csv_row(output_dir / "metrics.csv", record)
            print(
                f"Episode {episode_number:5d} | steps={step + 1:8d} | "
                f"reward={episode_reward:9.2f} | rolling20={rolling:9.2f} | "
                f"laps={record['laps_completed']} | seed={record['seed']}"
            )
            episode_reward = 0.0
            episode_steps = 0
            obs, info = env.reset()

    env.close()
    print("Random baseline finished. This mode validates the environment; it does not learn.")
    print(f"Metrics: {output_dir / 'metrics.csv'}")


def run_ppo(args, training_config, game_config, output_dir, seed):
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv
    except ImportError as exc:
        raise SystemExit(
            "PPO training requires Stable-Baselines3. Install with: "
            "pip install 'stable-baselines3>=2.3' tensorboard"
        ) from exc

    render_mode = "human" if args.render else None
    env = DummyVecEnv([
        lambda: Monitor(RacingEnv(
            config=training_config,
            game_config=game_config,
            render_mode=render_mode
        ))
    ])

    class TrainingDashboardCallback(BaseCallback):
        def __init__(self):
            super().__init__(verbose=0)
            self.episode_rewards = deque(maxlen=20)
            self.episode_number = 0
            self.last_render_step = 0
            self.metrics_path = output_dir / "metrics.csv"

        def _on_step(self) -> bool:
            if args.render and (
                self.num_timesteps - self.last_render_step >= max(1, args.render_every)
            ):
                self.training_env.env_method("render")
                self.last_render_step = self.num_timesteps

            for info in self.locals.get("infos", []):
                episode = info.get("episode")
                if not episode:
                    continue

                self.episode_number += 1
                reward = float(episode.get("r", 0.0))
                self.episode_rewards.append(reward)
                rolling = float(np.mean(self.episode_rewards))
                record = {
                    "episode": self.episode_number,
                    "timesteps": self.num_timesteps,
                    "reward": round(reward, 4),
                    "rolling_reward": round(rolling, 4),
                    "length": int(episode.get("l", 0)),
                    "laps_completed": info.get("laps_completed", 0),
                    "checkpoints_crossed": info.get("checkpoints_crossed", 0),
                    "seed": info.get("seed", seed),
                    "track_valid": info.get("track_valid", False),
                    "collisions": info.get("collision_count", 0),
                    "offroad_time": info.get("offroad_time", 0.0)
                }
                write_csv_row(self.metrics_path, record)
                print(
                    f"Episode {self.episode_number:5d} | steps={self.num_timesteps:8d} | "
                    f"reward={reward:9.2f} | rolling20={rolling:9.2f} | "
                    f"laps={record['laps_completed']} | valid_map={record['track_valid']}"
                )
            return True

    callback_list = [TrainingDashboardCallback()]
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if args.checkpoint_interval > 0:
        callback_list.append(CheckpointCallback(
            save_freq=max(1, int(args.checkpoint_interval)),
            save_path=str(checkpoint_dir),
            name_prefix="city_drive_ppo"
        ))

    tensorboard_path = str(output_dir / "tensorboard") if args.tensorboard else None
    if args.tensorboard:
        try:
            import tensorboard  # noqa: F401
        except ImportError as exc:
            raise SystemExit("Install TensorBoard with: pip install tensorboard") from exc

    if args.resume:
        model = PPO.load(args.resume, env=env, device=args.device)
        model.set_random_seed(seed)
        model.tensorboard_log = tensorboard_path
        print(f"Resuming PPO from {args.resume}")
    else:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=args.learning_rate,
            n_steps=max(2, int(args.n_steps)),
            batch_size=max(2, int(args.batch_size)),
            seed=seed,
            verbose=0,
            device=args.device,
            tensorboard_log=tensorboard_path
        )

    (output_dir / "run_config.json").write_text(json.dumps({
        "seed": args.seed,
        "seed_integer": seed,
        "timesteps": args.timesteps,
        "agent": "PPO",
        "observation_mode": args.obs_mode,
        "action_mode": args.action_mode,
        "theme": args.theme,
        "complexity": training_config.track_complexity,
        "road_width": training_config.road_width,
        "render": args.render,
        "learning_rate": args.learning_rate,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size
    }, indent=2), encoding="utf-8")

    print("City Drive | AI training")
    print(f"Seed: {args.seed} (normalized: {seed})")
    print(f"Agent: PPO | Observation: {args.obs_mode} | Action: {args.action_mode}")
    complexity_label = (
        f"{training_config.track_complexity:.2f}"
        if training_config.track_complexity is not None else "curriculum/default"
    )
    width_label = (
        f"{training_config.road_width:.1f} m"
        if training_config.road_width is not None else "curriculum/default"
    )
    print(f"Track: {args.theme} | Complexity: {complexity_label} | Road width: {width_label}")
    print(f"Timesteps: {args.timesteps:,} | Live viewer: {'on' if args.render else 'off'}")
    try:
        model.learn(
            total_timesteps=max(1, int(args.timesteps)),
            callback=CallbackList(callback_list),
            reset_num_timesteps=args.resume is None,
            tb_log_name="city_drive_ppo",
            progress_bar=False
        )
        final_path = output_dir / "final_model"
        model.save(str(final_path))
        print(f"Training complete. Model: {final_path}.zip")
        print(f"Metrics: {output_dir / 'metrics.csv'}")
        if args.tensorboard:
            print(f"TensorBoard: tensorboard --logdir {output_dir / 'tensorboard'}")
    finally:
        env.close()


def main():
    args = parse_args()
    if args.timesteps < 1:
        raise SystemExit("--timesteps must be greater than zero")
    if args.complexity is not None and not 0.0 <= args.complexity <= 1.0:
        raise SystemExit("--complexity must be between 0 and 1")
    if args.road_width is not None and not 8 <= args.road_width <= 16:
        raise SystemExit("--road-width must be between 8 and 16 metres")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    game_config = load_config()
    training_config = create_training_config(args)
    seed = parse_seed(args.seed)

    if args.agent == "random" and args.resume:
        raise SystemExit("--resume only works with --agent ppo")
    if args.agent == "random":
        run_random_baseline(args, training_config, game_config, output_dir, seed)
    else:
        run_ppo(args, training_config, game_config, output_dir, seed)


if __name__ == "__main__":
    main()
