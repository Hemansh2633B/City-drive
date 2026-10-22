#!/usr/bin/env python3
"""Evaluate a driving policy on reproducible track seeds.

This is a single-agent environment, so reports track completion and driving
quality. It deliberately does not report race position or win rate.
"""

import sys
import csv
import json
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import load_config
from ai.training_env import (
    RacingEnv, TrainingConfig, ObservationMode, ActionMode, make_racing_env,
)


BENCHMARK_SEEDS = [
    1, 42, 123, 456, 789, 1000, 1001, 1002, 1003, 1004,
    "benchmark-easy-1", "benchmark-easy-2", "benchmark-easy-3",
    "benchmark-medium-1", "benchmark-medium-2", "benchmark-medium-3",
    "benchmark-hard-1", "benchmark-hard-2", "benchmark-hard-3",
    "benchmark-expert-1", "benchmark-expert-2", "benchmark-expert-3",
    "daily-race-001", "daily-race-002", "daily-race-003",
    "championship-1", "championship-2", "championship-3",
    "time-trial-1", "time-trial-2", "time-trial-3",
]


class RandomAgent:
    """Random-policy baseline. Useful for smoke tests, not learning."""

    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, obs):
        return self.action_space.sample()


class HeuristicAgent:
    """Simple no-input baseline; not a track-following controller."""

    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, obs):
        if hasattr(self.action_space, "shape"):
            return np.zeros(self.action_space.shape, dtype=np.float32)
        return 4  # discrete "coast"


class SB3Agent:
    """Wrapper for Stable-Baselines3 policies."""

    def __init__(self, model):
        self.model = model

    def act(self, obs):
        action, _ = self.model.predict(obs, deterministic=True)
        return action


def _space_signature(space):
    """Return a small stable signature for checking saved policy compatibility."""
    return (
        type(space).__name__,
        tuple(getattr(space, "shape", ()) or ()),
        getattr(space, "n", None),
    )


def create_agent(agent_type: str, env: RacingEnv):
    """Create a baseline or load a PPO checkpoint.

    A requested trained model must load successfully. Falling back to a random
    policy would make evaluation results misleading, so errors are propagated.
    """
    if agent_type == "random":
        return RandomAgent(env.action_space)
    if agent_type == "heuristic":
        return HeuristicAgent(env.action_space)

    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise RuntimeError(
            "Loading a trained PPO model requires Stable-Baselines3. "
            "Install it with: pip install 'city-drive-rl[rl]'"
        ) from exc

    try:
        model = PPO.load(agent_type)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load trained model '{agent_type}'. Check the path, "
            "checkpoint file, and installed Stable-Baselines3 version."
        ) from exc

    if _space_signature(model.observation_space) != _space_signature(env.observation_space):
        raise ValueError(
            "Model observation space does not match this environment: "
            f"model={_space_signature(model.observation_space)}, "
            f"environment={_space_signature(env.observation_space)}. "
            "Use the same --obs-mode as training."
        )
    if _space_signature(model.action_space) != _space_signature(env.action_space):
        raise ValueError(
            "Model action space does not match this environment: "
            f"model={_space_signature(model.action_space)}, "
            f"environment={_space_signature(env.action_space)}. "
            "Use the same --action-mode as training."
        )
    return SB3Agent(model)


def run_episode(
    env: RacingEnv,
    agent,
    seed: Optional[Union[int, str]] = None,
    render: bool = False,
    max_steps: int = 10000,
) -> Dict[str, Any]:
    """Run one episode and return driving metrics (not race-place metrics)."""
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")

    obs, info = env.reset(seed=seed)
    total_reward = 0.0
    steps = 0
    terminated = False
    truncated = False

    while not terminated and not truncated and steps < max_steps:
        action = agent.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        steps += 1
        if render:
            env.render()

    laps_completed = int(info.get("laps_completed", 0))
    target_laps = int(getattr(env.track, "num_laps", 1))
    lap_times = [float(t) for t in info.get("lap_times", []) if float(t) > 0]
    raw_best_lap = float(info.get("best_lap_time", 0.0) or 0.0)
    step_limit_reached = bool(
        not terminated and not truncated and steps >= max_steps
    )

    return {
        "seed": seed,
        "reward": float(total_reward),
        "steps": int(steps),
        "laps_completed": laps_completed,
        "target_laps": target_laps,
        "best_lap_time": raw_best_lap if raw_best_lap > 0 else None,
        "lap_times": lap_times,
        "completed": laps_completed >= target_laps,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "step_limit_reached": step_limit_reached,
        "collision_count": int(info.get("collision_count", 0) or 0),
        "offroad_time": float(
            info.get("total_offroad_time", info.get("offroad_time", 0.0)) or 0.0
        ),
        "track_valid": bool(info.get("track_valid", True)),
    }


def evaluate_agent(
    agent,
    seeds: List[Union[int, str]],
    config: TrainingConfig,
    game_config,
    render: bool = False,
    max_steps: int = 10000,
    episodes_per_seed: int = 1,
) -> Dict[str, Any]:
    """Evaluate one policy across seeds and optional repeat episodes per seed."""
    if not seeds:
        raise ValueError("At least one evaluation seed is required")
    if episodes_per_seed < 1:
        raise ValueError("episodes_per_seed must be at least 1")
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")

    results = []
    total_episodes = len(seeds) * episodes_per_seed
    episode_number = 0

    for base_seed in seeds:
        for repeat in range(episodes_per_seed):
            episode_number += 1
            # Keep the requested seed for the first run; use a deterministic,
            # distinct text seed for repetitions of that track family.
            episode_seed = (
                base_seed if repeat == 0 else f"{base_seed}-repeat-{repeat + 1}"
            )
            print(f"Evaluating episode {episode_number}/{total_episodes} on seed: {episode_seed}")
            env = make_racing_env(
                config, game_config, render_mode="human" if render else None
            )
            try:
                result = run_episode(
                    env, agent, seed=episode_seed, render=render, max_steps=max_steps
                )
                results.append(result)
                best = (
                    f"{result['best_lap_time']:.2f}s"
                    if result["best_lap_time"] is not None else "n/a"
                )
                print(
                    f"  Reward: {result['reward']:.2f}, "
                    f"Laps: {result['laps_completed']}/{result['target_laps']}, "
                    f"Best lap: {best}, Track complete: {result['completed']}"
                )
            finally:
                env.close()

    rewards = [r["reward"] for r in results]
    laps = [r["laps_completed"] for r in results]
    lap_times = [time for r in results for time in r["lap_times"]]
    completed = [r["completed"] for r in results]
    collisions = [r["collision_count"] for r in results]
    offroad = [r["offroad_time"] for r in results]

    return {
        "num_seeds": len(seeds),
        "episodes_per_seed": episodes_per_seed,
        "num_episodes": len(results),
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
        "mean_laps": float(np.mean(laps)),
        "max_laps": int(np.max(laps)),
        "completion_rate": float(np.mean(completed)),
        "termination_rate": float(np.mean([r["terminated"] for r in results])),
        "truncation_rate": float(np.mean([r["truncated"] for r in results])),
        "step_limit_rate": float(np.mean([r["step_limit_reached"] for r in results])),
        "mean_collisions": float(np.mean(collisions)),
        "mean_offroad_time": float(np.mean(offroad)),
        "mean_lap_time": float(np.mean(lap_times)) if lap_times else None,
        "best_lap_time": float(np.min(lap_times)) if lap_times else None,
        "results": results,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a City Drive AI policy")
    parser.add_argument(
        "--agent", type=str, default="random",
        help="Agent type: random, heuristic, or path to a trained PPO .zip model",
    )
    parser.add_argument(
        "--seeds", type=str, nargs="+", default=None,
        help="Seeds to evaluate on (default: benchmark seeds)",
    )
    parser.add_argument(
        "--num-seeds", type=int, default=10,
        help="Number of benchmark seeds if --seeds is not provided",
    )
    parser.add_argument("--episodes-per-seed", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--obs-mode", choices=["vector", "sensor", "image", "combined"],
        default="vector",
    )
    parser.add_argument(
        "--action-mode", choices=["continuous", "discrete"], default="continuous"
    )
    parser.add_argument("--config", type=str, default=None, help="Config file path")
    parser.add_argument("--output", type=str, default="evaluation_results")
    parser.add_argument(
        "--vehicle", choices=["balanced", "speedster", "gripper", "drift_king", "rookie"],
        default="balanced",
    )
    parser.add_argument(
        "--theme",
        choices=["asphalt", "desert", "snow", "grassland", "neon", "industrial", "dirt"],
        default="asphalt",
    )
    return parser.parse_args()


def _format_seconds(value):
    return f"{value:.2f}s" if value is not None else "n/a"


def main():
    args = parse_args()
    if args.max_steps < 1:
        raise SystemExit("--max-steps must be greater than zero")
    if args.episodes_per_seed < 1:
        raise SystemExit("--episodes-per-seed must be greater than zero")
    if not args.seeds and args.num_seeds < 1:
        raise SystemExit("--num-seeds must be greater than zero")

    config = load_config(args.config)
    if args.seeds:
        seeds = []
        for value in args.seeds:
            try:
                seeds.append(int(value))
            except ValueError:
                seeds.append(value)
    else:
        seeds = BENCHMARK_SEEDS[:args.num_seeds]
        if len(seeds) < args.num_seeds:
            seeds.extend(
                f"extra-benchmark-{i}"
                for i in range(len(seeds) + 1, args.num_seeds + 1)
            )

    train_config = TrainingConfig()
    train_config.observation_mode = ObservationMode(args.obs_mode)
    train_config.action_mode = ActionMode(args.action_mode)
    train_config.max_episode_steps = args.max_steps
    train_config.track_theme = args.theme

    game_config = config
    env = make_racing_env(train_config, game_config)
    try:
        agent = create_agent(args.agent, env)
    finally:
        env.close()

    print(f"Evaluating {args.agent} on {len(seeds)} seed(s)...")
    print(f"Seeds: {seeds}")
    summary = evaluate_agent(
        agent, seeds, train_config, game_config, render=args.render,
        max_steps=args.max_steps, episodes_per_seed=args.episodes_per_seed,
    )

    print("\n=== EVALUATION SUMMARY ===")
    print(f"Episodes tested: {summary['num_episodes']}")
    print(f"Mean reward: {summary['mean_reward']:.2f} ± {summary['std_reward']:.2f}")
    print(f"Reward range: [{summary['min_reward']:.2f}, {summary['max_reward']:.2f}]")
    print(f"Mean laps: {summary['mean_laps']:.2f} (max: {summary['max_laps']})")
    print(f"Track completion rate: {summary['completion_rate'] * 100:.1f}%")
    print(f"Termination rate: {summary['termination_rate'] * 100:.1f}%")
    print(f"Truncation rate: {summary['truncation_rate'] * 100:.1f}%")
    print(f"Evaluation step-limit rate: {summary['step_limit_rate'] * 100:.1f}%")
    print(f"Mean collisions: {summary['mean_collisions']:.2f}")
    print(f"Mean off-road time: {summary['mean_offroad_time']:.2f}s")
    print(f"Mean lap time: {_format_seconds(summary['mean_lap_time'])}")
    print(f"Best lap time: {_format_seconds(summary['best_lap_time'])}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "evaluation.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, default=str)

    with (output_dir / "evaluation.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = [
            "seed", "reward", "steps", "laps_completed", "target_laps",
            "best_lap_time", "completed", "terminated", "truncated",
            "step_limit_reached", "collision_count", "offroad_time", "track_valid",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for result in summary["results"]:
            writer.writerow({field: result.get(field) for field in fields})

    print(f"\nResults saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
