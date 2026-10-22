"""Regression tests for the focused single-agent training workflow."""

import numpy as np
import pytest
import sys
from types import SimpleNamespace

from ai.training_env import ActionMode, ObservationMode, RacingEnv, TrainingConfig
from core.config import load_config
from core.math import vec2
from core.random import DeterministicRNG
from track.generator import TrackGenerator, TrackGeneratorConfig
from track.track import Checkpoint


def test_checkpoint_uses_forward_and_lateral_distances():
    checkpoint = Checkpoint(
        position=vec2(0.0, 0.0),
        direction=vec2(1.0, 0.0),
        width=10.0,
        index=0,
    )

    assert checkpoint.contains_point(vec2(1.0, 0.0))
    assert not checkpoint.contains_point(vec2(-1.0, 0.0))
    assert not checkpoint.contains_point(vec2(1.0, 6.0))
    assert checkpoint.crossed_by(vec2(-1.0, 0.0), vec2(1.0, 0.0))
    assert not checkpoint.crossed_by(vec2(1.0, 0.0), vec2(-1.0, 0.0))
    assert not checkpoint.crossed_by(vec2(-1.0, 6.0), vec2(1.0, 6.0))
    # Being well in front of the gate is not a new crossing.
    assert not checkpoint.crossed_by(vec2(2.0, 0.0), vec2(3.0, 0.0))


def test_seeded_tracks_are_valid_closed_loops():
    generator = TrackGenerator(TrackGeneratorConfig(), DeterministicRNG(7))

    tracks = [
        generator.generate(seed, complexity=0.55, road_width=12.0, length=600.0)
        for seed in (7, 42, "map-alpha", "map-beta")
    ]

    for track in tracks:
        assert track.is_valid, track.validation_errors
        assert len(track.centerline) >= 4
        assert len(track.checkpoints) >= generator.config.min_checkpoints
        assert np.allclose(track.centerline[0], track.centerline[-1])
        assert len(track.start_positions) >= 1
        assert len(track.start_positions) == len(track.start_angles)

    assert not np.allclose(tracks[2].centerline[20], tracks[3].centerline[20])


def test_text_seed_reproduces_training_map_and_spawn():
    game_config = load_config()
    training_config = TrainingConfig()
    training_config.curriculum["enabled"] = False

    env = RacingEnv(training_config, game_config)
    try:
        obs1, info1 = env.reset(seed="reproducible-map-alpha")
        points1 = [point.copy() for point in env.track.centerline]
        spawn1 = env.vehicle.state.position.copy()

        assert env.track.is_valid
        assert info1["track_valid"]
        assert env.opponents == []
        assert env.observation_space.contains(obs1)

        obs2, info2 = env.reset(seed="reproducible-map-alpha")
        assert info2["track_valid"]
        assert len(points1) == len(env.track.centerline)
        assert all(np.allclose(a, b) for a, b in zip(points1, env.track.centerline))
        assert np.allclose(spawn1, env.vehicle.state.position)
        assert env.observation_space.contains(obs2)
    finally:
        env.close()


def test_sensor_and_combined_observation_spaces_match_outputs():
    game_config = load_config()
    for mode in (ObservationMode.SENSOR, ObservationMode.COMBINED):
        config = TrainingConfig(observation_mode=mode, action_mode=ActionMode.CONTINUOUS)
        config.curriculum["enabled"] = False
        env = RacingEnv(config, game_config)
        try:
            observation, info = env.reset(seed=1234)
            assert info["track_valid"]
            assert env.observation_space.contains(observation)
            assert env.action_space.shape == (3,)
        finally:
            env.close()


def test_registered_gymnasium_environment_and_checker():
    import gymnasium as gym
    import city_drive_rl
    from gymnasium.utils.env_checker import check_env

    assert city_drive_rl.ENV_ID == "CityDrive-v0"
    env = gym.make(city_drive_rl.ENV_ID)
    try:
        observation, info = env.reset(seed=2026)
        assert env.observation_space.contains(observation)
        assert info["track_valid"]

        observation, reward, terminated, truncated, info = env.step(
            env.action_space.sample()
        )
        assert env.observation_space.contains(observation)
        assert np.isfinite(reward)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert info["track_valid"]
    finally:
        env.close()

    raw_env = city_drive_rl.RacingEnv()
    try:
        check_env(raw_env, skip_render_check=True)
    finally:
        raw_env.close()


def test_image_observations_fail_loudly_instead_of_returning_fake_data():
    config = TrainingConfig(observation_mode=ObservationMode.IMAGE)
    try:
        RacingEnv(config=config, game_config=load_config())
    except ValueError as exc:
        assert "Image observations are not implemented" in str(exc)
    else:
        raise AssertionError("Unsupported image observations must not silently return zeros")


def test_vehicle_presets_do_not_mutate_across_resets():
    game_config = load_config()
    config = TrainingConfig()
    config.curriculum["enabled"] = True
    env = RacingEnv(config, game_config)
    try:
        env.reset(seed=321)
        first_speed = env.vehicle.config.top_speed
        env.reset(seed=321)
        second_speed = env.vehicle.config.top_speed
        assert np.isclose(first_speed, second_speed)
        assert np.isclose(first_speed, 45.0 * config.curriculum["stages"][0]["speed_multiplier"])
    finally:
        env.close()



def test_evaluation_module_imports_and_exposes_cli():
    # Importing used to fail at function definition because Union was missing.
    from scripts import evaluate

    assert callable(evaluate.main)
    assert callable(evaluate.run_episode)


def test_failed_trained_model_load_does_not_fall_back_to_random(monkeypatch):
    from scripts import evaluate

    def fail_load(_path):
        raise FileNotFoundError("checkpoint missing")

    fake_sb3 = SimpleNamespace(PPO=SimpleNamespace(load=fail_load))
    monkeypatch.setitem(sys.modules, "stable_baselines3", fake_sb3)

    env = RacingEnv(TrainingConfig(), load_config())
    try:
        with pytest.raises(RuntimeError, match="Failed to load trained model"):
            evaluate.create_agent("missing-model.zip", env)
    finally:
        env.close()


def test_evaluation_reports_track_completion_not_race_position():
    from scripts import evaluate

    config = TrainingConfig()
    config.curriculum["enabled"] = False
    config.max_episode_steps = 10
    game_config = load_config()
    env = RacingEnv(config, game_config)
    try:
        agent = evaluate.RandomAgent(env.action_space)
        result = evaluate.run_episode(env, agent, seed=42, max_steps=1)
        assert result["steps"] == 1
        assert result["step_limit_reached"]
        assert result["completed"] is False
        assert result["target_laps"] >= 1
        assert result["offroad_time"] >= 0.0
        assert "final_position" not in result
        assert "mean_position" not in result
    finally:
        env.close()


def test_evaluation_summary_has_no_misleading_win_or_position_rates():
    from scripts import evaluate

    config = TrainingConfig()
    config.curriculum["enabled"] = False
    config.max_episode_steps = 10
    summary = evaluate.evaluate_agent(
        evaluate.RandomAgent(RacingEnv(config, load_config()).action_space),
        [42],
        config,
        load_config(),
        max_steps=1,
        episodes_per_seed=1,
    )
    assert summary["num_episodes"] == 1
    assert 0.0 <= summary["completion_rate"] <= 1.0
    assert "win_rate" not in summary
    assert "mean_final_position" not in summary
    assert summary["best_lap_time"] is None
