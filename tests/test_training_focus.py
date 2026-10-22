"""Regression tests for the focused single-agent training workflow."""

import numpy as np

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
