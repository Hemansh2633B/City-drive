"""
Tests for AI Training Environment & Fallbacks

Verifies Gymnasium environment compliance, observation/action spaces,
curriculum progression, track generator fallbacks, and agent load fallbacks.
"""

import pytest
import numpy as np
import gymnasium as gym

from ai.training_env import RacingEnv, TrainingConfig, ObservationMode, ActionMode, make_racing_env
from track.generator import TrackGenerator, TrackGeneratorConfig
from core.random import DeterministicRNG
from core.config import load_config
from scripts.train import create_agent
from ui.renderer import Renderer


class TestRacingEnv:
    """Tests for Gymnasium RacingEnv."""

    def test_env_creation_and_reset(self):
        config = TrainingConfig(observation_mode=ObservationMode.VECTOR, max_episode_steps=100)
        env = RacingEnv(config=config)
        obs, info = env.reset(seed=12345)

        assert isinstance(obs, np.ndarray)
        assert obs.shape == env.observation_space.shape
        assert info['seed'] == 12345
        assert 'speed' in info
        env.close()

    def test_env_stepping_continuous(self):
        config = TrainingConfig(
            observation_mode=ObservationMode.VECTOR,
            action_mode=ActionMode.CONTINUOUS,
            max_episode_steps=10
        )
        env = RacingEnv(config=config)
        obs, info = env.reset(seed=42)

        action = np.array([0.0, 1.0, 0.0], dtype=np.float32)  # Full throttle
        next_obs, reward, terminated, truncated, info = env.step(action)

        assert next_obs.shape == env.observation_space.shape
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert info['step'] == 1
        env.close()

    def test_env_stepping_discrete(self):
        config = TrainingConfig(
            observation_mode=ObservationMode.SENSOR,
            action_mode=ActionMode.DISCRETE,
            max_episode_steps=10
        )
        env = RacingEnv(config=config)
        obs, info = env.reset(seed=42)

        action = 2  # Accelerate
        next_obs, reward, terminated, truncated, info = env.step(action)

        assert next_obs.shape == env.observation_space.shape
        assert isinstance(reward, float)
        env.close()

    def test_combined_observation_mode(self):
        config = TrainingConfig(
            observation_mode=ObservationMode.COMBINED,
            max_episode_steps=10
        )
        env = RacingEnv(config=config)
        obs, info = env.reset(seed=999)

        assert obs.shape[0] == env.observation_space.shape[0]
        env.close()

    def test_curriculum_advancement(self):
        config = TrainingConfig()
        env = RacingEnv(config=config)
        env.reset(seed=1)

        assert env.curriculum_stage == 0
        advanced = env.advance_curriculum()
        assert advanced is True
        assert env.curriculum_stage == 1
        env.close()


class TestFallbacks:
    """Tests for system fallbacks."""

    def test_track_generator_fallback_oval(self):
        """Track generator falls back to seed-based oval track when invalid."""
        config = TrackGeneratorConfig(max_validation_attempts=1)
        rng = DeterministicRNG(12345)
        generator = TrackGenerator(config, rng)

        # Force _validate_track to return False to trigger fallback
        generator._validate_track = lambda track: False

        track = generator.generate(seed=12345, theme="asphalt")

        assert track.is_valid is True
        assert len(track.centerline) > 0
        assert len(track.checkpoints) >= 4

    def test_agent_loading_fallback(self):
        """Agent loader falls back to RandomAgent if model file does not exist."""
        config = TrainingConfig()
        env = RacingEnv(config=config)
        env.reset(seed=123)

        agent = create_agent("non_existent_model_file.zip", env)
        action = agent.act(env.observation_space.sample())

        assert action is not None
        env.close()

    def test_renderer_headless_fallback(self):
        """Renderer handles missing/headless display gracefully."""
        config = load_config()
        renderer = Renderer(config)
        renderer.initialize()  # Should initialize in dummy mode if display fails

        # Call render safely
        renderer.render(0.016)
        renderer.cleanup()
