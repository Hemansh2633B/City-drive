#!/usr/bin/env python3
"""
Training Script

Runs AI training in headless mode with configurable parameters.
"""

import sys
import os
import argparse
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import load_config
from modes.training_mode import TrainingMode, TrainingModeConfig
from ai.training_env import RacingEnv, TrainingConfig, ObservationMode, ActionMode


class RandomAgent:
    """Random agent for testing."""
    
    def __init__(self, action_space):
        self.action_space = action_space
    
    def act(self, obs):
        return self.action_space.sample()
    
    def update(self, transitions):
        pass


class HeuristicAgent:
    """Simple heuristic agent that follows racing line."""
    
    def __init__(self, action_space):
        self.action_space = action_space
    
    def act(self, obs):
        # Simple heuristic: steer toward center, moderate throttle
        # This is a placeholder - real implementation would use obs data
        if hasattr(self.action_space, 'sample'):
            return self.action_space.sample()
        return np.array([0.0, 0.5, 0.0], dtype=np.float32)
    
    def update(self, transitions):
        pass


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
    
    def update(self, transitions):
        pass


def parse_args():
    parser = argparse.ArgumentParser(description="AI Racing Training Script")
    
    # Mode
    parser.add_argument("--mode", type=str, default="training",
                       choices=["training", "1v1", "1v4"],
                       help="Game mode")
    
    # Seed
    parser.add_argument("--seed", type=str, default=None,
                       help="Track seed (string or integer)")
    parser.add_argument("--random-seed", action="store_true",
                       help="Use random seed")
    
    # Episodes
    parser.add_argument("--episodes", type=int, default=100,
                       help="Number of training episodes")
    parser.add_argument("--max-steps", type=int, default=10000,
                       help="Max steps per episode")
    
    # Agent
    parser.add_argument("--agent", type=str, default="random",
                       help="Agent type: random, heuristic, or path to model")
    
    # Observation/Action
    parser.add_argument("--obs-mode", type=str, default="vector",
                       choices=["vector", "sensor", "image", "combined"],
                       help="Observation mode")
    parser.add_argument("--action-mode", type=str, default="continuous",
                       choices=["continuous", "discrete"],
                       help="Action mode")
    
    # Curriculum
    parser.add_argument("--curriculum", action="store_true", default=True,
                       help="Enable curriculum learning")
    parser.add_argument("--no-curriculum", dest="curriculum", action="store_false",
                       help="Disable curriculum learning")
    parser.add_argument("--curriculum-stage", type=int, default=0,
                       help="Starting curriculum stage")
    
    # Vehicle/Track
    parser.add_argument("--vehicle", type=str, default="balanced",
                       choices=["balanced", "speedster", "gripper", "drift_king", "rookie"],
                       help="Vehicle preset")
    parser.add_argument("--theme", type=str, default="asphalt",
                       choices=["asphalt", "desert", "snow", "grassland", "neon", "industrial", "dirt"],
                       help="Track theme")
    parser.add_argument("--complexity", type=float, default=None,
                       help="Track complexity (0.0-1.0)")
    parser.add_argument("--road-width", type=float, default=None,
                       help="Road width in meters")
    
    # Output
    parser.add_argument("--output", type=str, default="training_results",
                       help="Output directory for results")
    parser.add_argument("--save-replays", action="store_true", default=True,
                       help="Save replay files")
    parser.add_argument("--replay-interval", type=int, default=100,
                       help="Save replay every N episodes")
    
    # Evaluation
    parser.add_argument("--eval-interval", type=int, default=100,
                       help="Evaluate every N episodes")
    parser.add_argument("--eval-episodes", type=int, default=10,
                       help="Episodes per evaluation")
    parser.add_argument("--eval-seeds", type=str, nargs="+", default=None,
                       help="Seeds for evaluation (space separated)")
    
    # Config
    parser.add_argument("--config", type=str, default=None,
                       help="Path to config file")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Parse seed
    seed = args.seed
    if args.random_seed or seed is None:
        seed = np.random.randint(0, 2**31 - 1)
    else:
        try:
            seed = int(seed)
        except ValueError:
            pass  # Keep as string
    
    # Create training config
    train_config = TrainingModeConfig(
        seed=seed,
        theme=args.theme,
        complexity=args.complexity,
        road_width=args.road_width,
        vehicle_preset=args.vehicle,
        max_episodes=args.episodes,
        max_steps_per_episode=args.max_steps,
        curriculum_enabled=args.curriculum,
        curriculum_stage=args.curriculum_stage,
        save_replays=args.save_replays,
        replay_interval=args.replay_interval
    )
    
    # Create training mode
    training_mode = TrainingMode(train_config, config)
    
    # Create environment to get spaces
    env = training_mode.create_env(seed)
    
    # Create agent
    agent = create_agent(args.agent, env)
    env.close()
    
    # Parse eval seeds
    eval_seeds = None
    if args.eval_seeds:
        eval_seeds = []
        for s in args.eval_seeds:
            try:
                eval_seeds.append(int(s))
            except ValueError:
                eval_seeds.append(s)
    
    # Run training
    print(f"Starting training: {args.episodes} episodes, seed={seed}")
    print(f"Agent: {args.agent}, Obs: {args.obs_mode}, Action: {args.action_mode}")
    
    try:
        stats = training_mode.run_training(
            agent,
            num_episodes=args.episodes,
            eval_interval=args.eval_interval,
            eval_episodes=args.eval_episodes,
            eval_seeds=eval_seeds
        )
        
        # Export results
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        training_mode.export_telemetry(str(output_dir / "telemetry"))
        
        # Print final stats
        final_stats = training_mode.get_statistics()
        print("\n=== Training Complete ===")
        print(f"Episodes: {final_stats['episodes_completed']}")
        print(f"Total Steps: {final_stats['total_steps']}")
        print(f"Best Reward: {final_stats['best_reward']:.2f}")
        print(f"Best Lap Time: {final_stats['best_lap_time']:.2f}s")
        print(f"Mean Reward: {final_stats['mean_reward']:.2f}")
        print(f"Results saved to: {output_dir}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted")
        training_mode.export_telemetry(str(output_dir / "telemetry_interrupted"))
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())