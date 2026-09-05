#!/usr/bin/env python3
"""
Evaluation Script

Evaluates trained agents on benchmark seeds.
"""

import sys
import os
import argparse
import numpy as np
import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import load_config
from ai.training_env import RacingEnv, TrainingConfig, ObservationMode, ActionMode, make_racing_env


# Benchmark seeds for consistent evaluation
BENCHMARK_SEEDS = [
    1, 42, 123, 456, 789,
    1000, 1001, 1002, 1003, 1004,
    "benchmark-easy-1", "benchmark-easy-2", "benchmark-easy-3",
    "benchmark-medium-1", "benchmark-medium-2", "benchmark-medium-3",
    "benchmark-hard-1", "benchmark-hard-2", "benchmark-hard-3",
    "benchmark-expert-1", "benchmark-expert-2", "benchmark-expert-3",
    "daily-race-001", "daily-race-002", "daily-race-003",
    "championship-1", "championship-2", "championship-3",
    "time-trial-1", "time-trial-2", "time-trial-3"
]


class RandomAgent:
    """Random agent for baseline."""
    
    def __init__(self, action_space):
        self.action_space = action_space
    
    def act(self, obs):
        return self.action_space.sample()


class HeuristicAgent:
    """Simple heuristic agent."""
    
    def __init__(self, action_space):
        self.action_space = action_space
    
    def act(self, obs):
        # Very simple heuristic
        if hasattr(self.action_space, 'shape'):
            return np.zeros(self.action_space.shape, dtype=np.float32)
        return 0


def create_agent(agent_type: str, env: RacingEnv):
    """Create agent by type."""
    if agent_type == "random":
        return RandomAgent(env.action_space)
    elif agent_type == "heuristic":
        return HeuristicAgent(env.action_space)
    else:
        # Try to load a trained model
        try:
            from stable_baselines3 import PPO
            model = PPO.load(agent_type)
            return SB3Agent(model, env.action_space)
        except ImportError:
            print("stable_baselines3 not installed, using random agent")
            return RandomAgent(env.action_space)
        except Exception as e:
            print(f"Failed to load model {agent_type}: {e}")
            return RandomAgent(env.action_space)


class SB3Agent:
    """Wrapper for Stable Baselines3 agents."""
    
    def __init__(self, model, action_space):
        self.model = model
        self.action_space = action_space
    
    def act(self, obs):
        action, _ = self.model.predict(obs, deterministic=True)
        return action


def run_episode(env: RacingEnv, agent, seed: Optional[Union[int, str]] = None, 
                render: bool = False, max_steps: int = 10000) -> Dict[str, Any]:
    """Run a single evaluation episode."""
    obs, info = env.reset(seed=seed)
    
    total_reward = 0.0
    steps = 0
    done = False
    truncated = False
    
    lap_times = []
    positions = []
    
    while not done and not truncated and steps < max_steps:
        action = agent.act(obs)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        
        if render:
            env.render()
        
        # Track lap times
        if 'lap_times' in info and info['lap_times']:
            lap_times = info['lap_times']
        
        positions.append(info.get('position', 1))
    
    return {
        'seed': seed,
        'reward': total_reward,
        'steps': steps,
        'laps_completed': info.get('laps_completed', 0),
        'best_lap_time': info.get('best_lap_time', 0),
        'lap_times': lap_times,
        'final_position': info.get('position', 1),
        'mean_position': np.mean(positions) if positions else 1,
        'collision_count': info.get('collision_count', 0),
        'offroad_time': info.get('offroad_time', 0),
        'completed': done and not truncated
    }


def evaluate_agent(agent, seeds: List[Union[int, str]], config: TrainingConfig,
                   game_config, render: bool = False, max_steps: int = 10000) -> Dict[str, Any]:
    """Evaluate agent on multiple seeds."""
    
    results = []
    
    for seed in seeds:
        print(f"Evaluating on seed: {seed}")
        
        # Create env for this seed
        env = make_racing_env(config, game_config, render_mode='human' if render else None)
        
        try:
            result = run_episode(env, agent, seed=seed, render=render, max_steps=max_steps)
            results.append(result)
            
            print(f"  Reward: {result['reward']:.2f}, Laps: {result['laps_completed']}, "
                  f"Best Lap: {result['best_lap_time']:.2f}s, Position: {result['final_position']}")
        finally:
            env.close()
    
    # Aggregate statistics
    rewards = [r['reward'] for r in results]
    laps = [r['laps_completed'] for r in results]
    lap_times = []
    for r in results:
        lap_times.extend(r['lap_times'])
    positions = [r['final_position'] for r in results]
    completed = [r['completed'] for r in results]
    collisions = [r['collision_count'] for r in results]
    offroad = [r['offroad_time'] for r in results]
    
    summary = {
        'num_seeds': len(seeds),
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'min_reward': np.min(rewards),
        'max_reward': np.max(rewards),
        'mean_laps': np.mean(laps),
        'max_laps': np.max(laps),
        'completion_rate': np.mean(completed),
        'mean_final_position': np.mean(positions),
        'win_rate': np.mean([p == 1 for p in positions]),
        'mean_collisions': np.mean(collisions),
        'mean_offroad_time': np.mean(offroad),
        'mean_lap_time': np.mean(lap_times) if lap_times else 0,
        'best_lap_time': np.min(lap_times) if lap_times else float('inf'),
        'results': results
    }
    
    return summary


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate AI Racing Agent")
    
    parser.add_argument("--agent", type=str, default="random",
                       help="Agent type: random, heuristic, or path to model")
    parser.add_argument("--seeds", type=str, nargs="+", default=None,
                       help="Seeds to evaluate on (default: benchmark seeds)")
    parser.add_argument("--num-seeds", type=int, default=10,
                       help="Number of random seeds if --seeds not provided")
    parser.add_argument("--episodes-per-seed", type=int, default=1,
                       help="Episodes per seed")
    parser.add_argument("--max-steps", type=int, default=10000,
                       help="Max steps per episode")
    parser.add_argument("--render", action="store_true",
                       help="Render evaluation")
    parser.add_argument("--obs-mode", type=str, default="vector",
                       choices=["vector", "sensor", "image", "combined"])
    parser.add_argument("--action-mode", type=str, default="continuous",
                       choices=["continuous", "discrete"])
    parser.add_argument("--config", type=str, default=None,
                       help="Config file path")
    parser.add_argument("--output", type=str, default="evaluation_results",
                       help="Output directory")
    parser.add_argument("--vehicle", type=str, default="balanced",
                       choices=["balanced", "speedster", "gripper", "drift_king", "rookie"])
    parser.add_argument("--theme", type=str, default="asphalt",
                       choices=["asphalt", "desert", "snow", "grassland", "neon", "industrial", "dirt"])
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Determine seeds
    if args.seeds:
        seeds = []
        for s in args.seeds:
            try:
                seeds.append(int(s))
            except ValueError:
                seeds.append(s)
    else:
        seeds = BENCHMARK_SEEDS[:args.num_seeds]
    
    # Create training config
    train_config = TrainingConfig()
    train_config.observation_mode = ObservationMode(args.obs_mode)
    train_config.action_mode = ActionMode(args.action_mode)
    train_config.max_episode_steps = args.max_steps
    
    # Create dummy env to get agent
    game_config = config
    env = make_racing_env(train_config, game_config)
    agent = create_agent(args.agent, env)
    env.close()
    
    # Run evaluation
    print(f"Evaluating {args.agent} on {len(seeds)} seeds...")
    print(f"Seeds: {seeds}")
    
    summary = evaluate_agent(agent, seeds, train_config, game_config, 
                           render=args.render, max_steps=args.max_steps)
    
    # Print summary
    print("\n=== EVALUATION SUMMARY ===")
    print(f"Seeds tested: {summary['num_seeds']}")
    print(f"Mean reward: {summary['mean_reward']:.2f} ± {summary['std_reward']:.2f}")
    print(f"Reward range: [{summary['min_reward']:.2f}, {summary['max_reward']:.2f}]")
    print(f"Mean laps: {summary['mean_laps']:.2f} (max: {summary['max_laps']})")
    print(f"Completion rate: {summary['completion_rate']*100:.1f}%")
    print(f"Win rate: {summary['win_rate']*100:.1f}%")
    print(f"Mean position: {summary['mean_final_position']:.2f}")
    print(f"Mean collisions: {summary['mean_collisions']:.2f}")
    print(f"Mean offroad time: {summary['mean_offroad_time']:.2f}s")
    print(f"Mean lap time: {summary['mean_lap_time']:.2f}s")
    print(f"Best lap time: {summary['best_lap_time']:.2f}s")
    
    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # JSON
    with open(output_dir / "evaluation.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    # CSV
    with open(output_dir / "evaluation.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['seed', 'reward', 'steps', 'laps', 'best_lap', 'position', 'completed', 'collisions', 'offroad'])
        for r in summary['results']:
            writer.writerow([
                r['seed'], r['reward'], r['steps'], r['laps_completed'],
                r['best_lap_time'], r['final_position'], r['completed'],
                r['collision_count'], r['offroad_time']
            ])
    
    print(f"\nResults saved to: {output_dir}")
    
    return 0


if __name__ == "__main__":
    from typing import Union
    sys.exit(main())