"""Tests for batched MLP inference and reproducible evolutionary operations."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from ai.gpu_evolution import BatchedPolicyPopulation, EvolutionConfig, GPUEvolutionTrainer
from ai.training_env import TrainingConfig


def test_batched_policy_initialization_is_seeded_and_outputs_valid_actions():
    first = BatchedPolicyPopulation(8, 80, 16, device="cpu", seed=17)
    second = BatchedPolicyPopulation(8, 80, 16, device="cpu", seed=17)
    torch.testing.assert_close(first.w1, second.w1)
    torch.testing.assert_close(first.w2, second.w2)
    observations = torch.randn(8, 80)
    actions = first.actions(observations)
    assert actions.shape == (8, 3)
    assert torch.all(actions[:, 0] >= -1.0)
    assert torch.all(actions[:, 0] <= 1.0)
    assert torch.all(actions[:, 1:] >= 0.0)
    assert torch.all(actions[:, 1:] <= 1.0)


def test_mutation_only_changes_selected_candidates():
    policies = BatchedPolicyPopulation(8, 20, 8, device="cpu", seed=19)
    before = policies.w1.clone()
    policies.mutate(torch.tensor([5, 6]), sigma=0.2)
    torch.testing.assert_close(policies.w1[:5], before[:5])
    assert not torch.equal(policies.w1[5:], before[5:])


def test_evolution_config_requires_explicit_valid_fallback():
    with pytest.raises(ValueError, match="collision_fallback"):
        EvolutionConfig(collision_fallback="silent_relaxation")
    with pytest.raises(ValueError, match="population_size"):
        EvolutionConfig(population_size=4)
    configurable = EvolutionConfig(
        population_size=8,
        fitness_weights={"reward": 2.0, "offroad_time": -3.0},
    )
    assert configurable.fitness_weights["reward"] == 2.0
    assert configurable.fitness_weights["distance_traveled"] == 0.0
    assert configurable.fitness_weights["offroad_time"] == -3.0
    with pytest.raises(ValueError, match="unsupported fitness weight keys"):
        EvolutionConfig(population_size=8, fitness_weights={"unknown_component": 1.0})


def test_selection_prefers_collision_free_and_flags_insufficient_set():
    config = EvolutionConfig(
        population_size=8,
        simulation_batch_size=4,
        generations=1,
        rollout_steps=1,
        collision_fallback="allow_collisions",
        output_dir="/tmp/city-drive-gpu-test",
    )
    trainer = GPUEvolutionTrainer(config, TrainingConfig(), device="cpu")
    try:
        metrics = {
            "fitness": torch.tensor([100.0, 90.0, 80.0, 70.0, 60.0, 50.0, 40.0, 30.0]),
            "zero_collision": torch.tensor([False, False, True, True, True, False, False, False]),
        }
        selected, insufficient = trainer.select_elites(metrics)
        assert insufficient is True
        assert selected.tolist()[:3] == [2, 3, 4]
        assert len(set(selected.tolist())) == 5
    finally:
        trainer.close()


def test_elitism_preserves_selected_policies_and_can_be_disabled():
    config = EvolutionConfig(
        population_size=8,
        simulation_batch_size=4,
        generations=1,
        rollout_steps=1,
        elitism=True,
        mutation_sigma=0.25,
        mutation_probability=1.0,
        output_dir="/tmp/city-drive-gpu-elitism-test",
    )
    trainer = GPUEvolutionTrainer(config, TrainingConfig(), device="cpu")
    try:
        before = trainer.population.w1.clone()
        selected = torch.tensor([7, 6, 5, 4, 3])
        trainer.next_generation(selected)
        torch.testing.assert_close(trainer.population.w1[:5], before[selected])
        assert trainer.previous_elite_indices.tolist() == [0, 1, 2, 3, 4]
    finally:
        trainer.close()

    no_elitism = EvolutionConfig(
        population_size=8,
        simulation_batch_size=4,
        generations=1,
        rollout_steps=1,
        elitism=False,
        mutation_sigma=0.25,
        mutation_probability=1.0,
        output_dir="/tmp/city-drive-gpu-no-elitism-test",
    )
    trainer = GPUEvolutionTrainer(no_elitism, TrainingConfig(), device="cpu")
    try:
        before = trainer.population.w1.clone()
        trainer.next_generation(torch.tensor([7, 6, 5, 4, 3]))
        assert trainer.previous_elite_indices is None
        assert not torch.equal(trainer.population.w1, before)
    finally:
        trainer.close()

    with pytest.raises(ValueError, match="requires elitism=True"):
        EvolutionConfig(
            population_size=8,
            elitism=False,
            collision_fallback="keep_previous_elites",
        )
