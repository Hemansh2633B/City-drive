"""
Training Mode

Single-car AI training environment with curriculum learning support.
"""

import numpy as np
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from core.config import SimulationContext, Config
from core.random import DeterministicRNG
from core.math import Vec2, vec2
from track.track import Track
from track.generator import TrackGenerator, TrackGeneratorConfig
from vehicles.vehicle import Vehicle, VehicleConfig
from vehicles.presets import get_vehicle_preset
from ai.training_env import RacingEnv, TrainingConfig, make_racing_env
from core.physics import PhysicsWorld, RigidBody, PhysicsMaterial


@dataclass
class TrainingModeConfig:
    """Configuration for training mode."""
    # Environment
    seed: Optional[Union[int, str]] = None
    theme: str = "asphalt"
    complexity: Optional[float] = None
    road_width: Optional[float] = None
    vehicle_preset: str = "balanced"
    
    # Training
    max_episodes: int = 1000
    max_steps_per_episode: int = 10000
    
    # Curriculum
    curriculum_enabled: bool = True
    curriculum_stage: int = 0
    
    # Replay
    save_replays: bool = True
    replay_interval: int = 100  # Save replay every N episodes
    
    # Telemetry
    telemetry_interval: int = 1
    export_format: str = "json"  # "json", "csv", "both"


class TrainingMode:
    """
    Training mode for single-car AI learning.
    
    Features:
    - Gymnasium-compatible environment
    - Curriculum learning
    - Episode management
    - Replay recording
    - Telemetry export
    - Headless operation
    - Deterministic seeds
    """
    
    def __init__(self, config: TrainingModeConfig, game_config: Config):
        self.config = config
        self.game_config = game_config
        
        # Training environment
        self.train_config = TrainingConfig()
        self.train_config.max_episode_steps = config.max_steps_per_episode
        
        if config.curriculum_enabled:
            self.train_config.curriculum['enabled'] = True
            self.train_config.curriculum['stages'] = self._get_curriculum_stages()
        else:
            self.train_config.curriculum['enabled'] = False
        
        self.env: Optional[RacingEnv] = None
        self.current_episode = 0
        self.total_steps = 0
        self.best_reward = -float('inf')
        self.best_lap_time = float('inf')
        
        # Episode stats
        self.episode_rewards: List[float] = []
        self.episode_lap_times: List[List[float]] = []
        self.episode_lengths: List[int] = []
        
        # Replays
        self.replays: List[Dict] = []
        
        # Random seed
        self.master_seed = config.seed if config.seed is not None else np.random.randint(0, 2**31 - 1)
    
    def _get_curriculum_stages(self) -> List[Dict]:
        """Get curriculum stages from game config."""
        return self.game_config.training.curriculum.stages
    
    def create_env(self, seed: Optional[Union[int, str]] = None) -> RacingEnv:
        """Create training environment with specific seed."""
        episode_seed = seed if seed is not None else self.master_seed
        
        # Add episode offset for variety if using same seed
        if seed is None:
            episode_seed = int(DeterministicRNG(self.master_seed, sequence=self.current_episode).random_raw())
        else:
            episode_seed = int(seed) if isinstance(seed, (int, np.integer)) else seed
        
        env = make_racing_env(self.train_config, self.game_config, render_mode=None)
        
        # Manually reset with our seed
        obs, info = env.reset(seed=episode_seed)
        return env
    
    def run_episode(self, agent, seed: Optional[Union[int, str]] = None, 
                    render: bool = False) -> Dict[str, Any]:
        """
        Run a single training episode.
        
        Args:
            agent: Agent with act(obs) -> action method
            seed: Episode seed (None for auto)
            render: Whether to render (requires render_mode)
            
        Returns:
            Episode statistics dictionary
        """
        env = self.create_env(seed)
        
        obs, info = env.reset(seed=seed)
        done = False
        truncated = False
        episode_reward = 0.0
        episode_steps = 0
        
        while not done and not truncated:
            action = agent.act(obs)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            episode_steps += 1
            
            if render:
                env.render()
        
        # Collect results
        result = {
            'episode': self.current_episode,
            'seed': env.sim_context.seed,
            'reward': episode_reward,
            'steps': episode_steps,
            'laps_completed': info.get('laps_completed', 0),
            'best_lap_time': info.get('best_lap_time', 0),
            'lap_times': info.get('lap_times', []),
            'collision_count': info.get('collision_count', 0),
            'offroad_time': info.get('offroad_time', 0),
            'telemetry': env.get_telemetry()
        }
        
        # Update stats
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_steps)
        self.episode_lap_times.append(info.get('lap_times', []))
        
        if episode_reward > self.best_reward:
            self.best_reward = episode_reward
        
        if info.get('best_lap_time', float('inf')) < self.best_lap_time:
            self.best_lap_time = info.get('best_lap_time', float('inf'))
        
        # Save replay if interval
        if self.config.save_replays and self.current_episode % self.config.replay_interval == 0:
            self.replays.append(result)
        
        self.current_episode += 1
        self.total_steps += episode_steps
        
        env.close()
        return result
    
    def run_training(self, agent, num_episodes: Optional[int] = None,
                     eval_interval: int = 100, eval_episodes: int = 10,
                     eval_seeds: Optional[List[Union[int, str]]] = None) -> Dict[str, Any]:
        """
        Run full training loop with evaluation.
        
        Args:
            agent: Agent with act(obs) -> action and update(transitions) methods
            num_episodes: Number of episodes (default from config)
            eval_interval: Evaluate every N episodes
            eval_episodes: Episodes per evaluation
            eval_seeds: Fixed seeds for evaluation
            
        Returns:
            Training statistics
        """
        num_episodes = num_episodes or self.config.max_episodes
        eval_seeds = eval_seeds or list(range(1000, 1000 + eval_episodes))
        
        training_stats = {
            'episodes': [],
            'evaluations': [],
            'curriculum_progress': []
        }
        
        for episode in range(num_episodes):
            # Run training episode
            result = self.run_episode(agent)
            training_stats['episodes'].append(result)
            
            # Curriculum advancement check
            if (self.config.curriculum_enabled and 
                self.current_episode > 0 and 
                self.current_episode % 200 == 0):
                if self.env and self.env.advance_curriculum():
                    training_stats['curriculum_progress'].append({
                        'episode': self.current_episode,
                        'stage': self.env.curriculum_stage
                    })
            
            # Evaluation
            if episode % eval_interval == 0 and episode > 0:
                eval_results = self.evaluate(agent, eval_seeds)
                training_stats['evaluations'].append({
                    'episode': episode,
                    'results': eval_results
                })
                
                print(f"Episode {episode}: eval_reward={eval_results['mean_reward']:.2f}, "
                      f"mean_laps={eval_results['mean_laps']:.2f}")
            
            # Agent update (if it supports batch updates)
            if hasattr(agent, 'update') and len(training_stats['episodes']) >= 10:
                recent = training_stats['episodes'][-10:]
                # Convert to transitions format for agent
                agent.update(recent)
        
        # Final evaluation
        final_eval = self.evaluate(agent, eval_seeds)
        training_stats['evaluations'].append({
            'episode': num_episodes,
            'results': final_eval,
            'final': True
        })
        
        return training_stats
    
    def evaluate(self, agent, seeds: List[Union[int, str]]) -> Dict[str, Any]:
        """Evaluate agent on fixed seeds."""
        rewards = []
        laps = []
        lap_times = []
        
        for seed in seeds:
            result = self.run_episode(agent, seed=seed)
            rewards.append(result['reward'])
            laps.append(result['laps_completed'])
            if result['lap_times']:
                lap_times.extend(result['lap_times'])
        
        return {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'mean_laps': np.mean(laps),
            'max_laps': max(laps),
            'mean_lap_time': np.mean(lap_times) if lap_times else 0,
            'best_lap_time': min(lap_times) if lap_times else float('inf'),
            'seeds_tested': len(seeds)
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get training statistics."""
        return {
            'episodes_completed': self.current_episode,
            'total_steps': self.total_steps,
            'best_reward': self.best_reward,
            'best_lap_time': self.best_lap_time,
            'mean_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0,
            'mean_episode_length': np.mean(self.episode_lengths) if self.episode_lengths else 0,
            'recent_rewards': self.episode_rewards[-100:] if self.episode_rewards else [],
            'replay_count': len(self.replays)
        }
    
    def export_telemetry(self, filepath: str) -> None:
        """Export all telemetry to file."""
        import json
        import csv
        
        data = {
            'config': self.config.__dict__,
            'statistics': self.get_statistics(),
            'episodes': self.episode_rewards,
            'replays': self.replays
        }
        
        if self.config.export_format in ('json', 'both'):
            with open(filepath + '.json', 'w') as f:
                json.dump(data, f, indent=2, default=str)
        
        if self.config.export_format in ('csv', 'both'):
            with open(filepath + '.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['episode', 'reward', 'steps', 'laps', 'best_lap_time'])
                for i, (reward, length, laps) in enumerate(zip(
                    self.episode_rewards, self.episode_lengths, self.episode_lap_times
                )):
                    best_lap = min(laps) if laps else 0
                    writer.writerow([i, reward, length, len(laps), best_lap])


from typing import Union