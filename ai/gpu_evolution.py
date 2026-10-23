"""Batched evolutionary policy training for GPUTrainingEnv.

The policy population is a stack of tensors (one small MLP per candidate), so
inference uses batched matrix multiplication rather than per-policy Python
modules. Track generation, metrics serialization and checkpointing are boundary
operations; candidate evaluation and policy inference stay on the selected device.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch

from ai.gpu_training_env import GPUTrainingEnv
from ai.training_env import ActionMode, ObservationMode, RacingEnv, TrainingConfig
from core.config import load_config


class BatchedPolicyPopulation:
    """Parameter tensors and inference for population members sharing an MLP shape."""

    def __init__(
        self,
        population_size: int,
        observation_size: int,
        hidden_size: int = 64,
        *,
        device: torch.device | str = "cuda",
        seed: int = 0,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if population_size < 1:
            raise ValueError("population_size must be >= 1")
        if observation_size < 1 or hidden_size < 1:
            raise ValueError("observation_size and hidden_size must be positive")
        self.population_size = int(population_size)
        self.observation_size = int(observation_size)
        self.hidden_size = int(hidden_size)
        self.device = torch.device(device)
        self.dtype = dtype
        generator = torch.Generator(device=self.device)
        generator.manual_seed(int(seed))

        def normal(shape: tuple[int, ...], fan_in: int) -> torch.Tensor:
            return torch.randn(shape, generator=generator, device=self.device, dtype=self.dtype) / math.sqrt(fan_in)

        p, i, h = self.population_size, self.observation_size, self.hidden_size
        self.w1 = normal((p, i, h), i)
        self.b1 = torch.zeros((p, h), device=self.device, dtype=self.dtype)
        self.w2 = normal((p, h, 3), h)
        self.b2 = torch.zeros((p, 3), device=self.device, dtype=self.dtype)
        self.generator = generator

    @torch.no_grad()
    def actions(
        self,
        observations: torch.Tensor,
        candidate_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute continuous [steering, throttle, brake] outputs in one batched op."""
        if observations.ndim != 2 or observations.shape[1] != self.observation_size:
            raise ValueError(
                f"observations must have shape [batch, {self.observation_size}]"
            )
        count = observations.shape[0]
        if candidate_indices is None:
            if count != self.population_size:
                raise ValueError("candidate_indices is required for a policy subset")
            candidate_indices = torch.arange(self.population_size, device=self.device)
        candidate_indices = candidate_indices.to(device=self.device, dtype=torch.long)
        first = torch.bmm(
            observations.unsqueeze(1),
            self.w1.index_select(0, candidate_indices),
        ).squeeze(1) + self.b1.index_select(0, candidate_indices)
        hidden = torch.tanh(first)
        logits = torch.bmm(
            hidden.unsqueeze(1),
            self.w2.index_select(0, candidate_indices),
        ).squeeze(1) + self.b2.index_select(0, candidate_indices)
        steering = torch.tanh(logits[:, 0])
        throttle_brake = torch.sigmoid(logits[:, 1:])
        return torch.cat((steering[:, None], throttle_brake), dim=1)

    @torch.no_grad()
    def copy_candidates(self, indices: torch.Tensor, target_indices: torch.Tensor) -> None:
        indices = indices.to(device=self.device, dtype=torch.long)
        target_indices = target_indices.to(device=self.device, dtype=torch.long)
        self.w1[target_indices] = self.w1[indices]
        self.b1[target_indices] = self.b1[indices]
        self.w2[target_indices] = self.w2[indices]
        self.b2[target_indices] = self.b2[indices]

    @torch.no_grad()
    def mutate(
        self,
        indices: torch.Tensor,
        *,
        sigma: float,
        mutation_probability: float = 1.0,
    ) -> None:
        """Mutate selected policy parameters with element-wise Gaussian noise."""
        indices = indices.to(device=self.device, dtype=torch.long)
        if sigma < 0.0:
            raise ValueError("mutation sigma cannot be negative")
        for parameter in (self.w1, self.b1, self.w2, self.b2):
            target = parameter.index_select(0, indices)
            noise = torch.randn(
                target.shape,
                generator=self.generator,
                device=self.device,
                dtype=target.dtype,
            )
            if mutation_probability < 1.0:
                mask = torch.rand(
                    target.shape,
                    generator=self.generator,
                    device=self.device,
                    dtype=target.dtype,
                ) < mutation_probability
                noise = noise * mask
            parameter[indices] = target + sigma * noise

    def state_dict(self) -> dict[str, Any]:
        """Return device-neutral checkpoint tensors (sync occurs at checkpoint boundary)."""
        return {
            "w1": self.w1.detach().cpu(),
            "b1": self.b1.detach().cpu(),
            "w2": self.w2.detach().cpu(),
            "b2": self.b2.detach().cpu(),
            "population_size": self.population_size,
            "observation_size": self.observation_size,
            "hidden_size": self.hidden_size,
        }


class EvolutionConfig:
    """Simple validated configuration kept dependency-free for CLI and notebooks."""

    def __init__(
        self,
        population_size: int = 10_000,
        simulation_batch_size: int = 1_024,
        generations: int = 50,
        rollout_steps: int = 2_000,
        hidden_size: int = 64,
        elites: int = 5,
        elitism: bool = True,
        mutation_sigma: float = 0.05,
        mutation_probability: float = 0.1,
        collision_fallback: str = "allow_collisions",
        fitness_weights: Optional[dict[str, float]] = None,
        seed: int = 42,
        track_seeds: Sequence[int] = (1001, 2002, 3003),
        output_dir: str | Path = "training_runs/gpu_evolution",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if population_size < 5:
            raise ValueError("population_size must be >= 5")
        if simulation_batch_size < 1:
            raise ValueError("simulation_batch_size must be >= 1")
        if generations < 1 or rollout_steps < 1:
            raise ValueError("generations and rollout_steps must be >= 1")
        if elites != 5:
            raise ValueError("this workflow currently selects exactly five elites")
        if collision_fallback not in {"allow_collisions", "keep_previous_elites", "raise"}:
            raise ValueError(
                "collision_fallback must be allow_collisions, keep_previous_elites, or raise"
            )
        if not elitism and collision_fallback == "keep_previous_elites":
            raise ValueError("collision_fallback='keep_previous_elites' requires elitism=True")
        if mutation_sigma < 0 or not 0 <= mutation_probability <= 1:
            raise ValueError("invalid mutation configuration")
        self.population_size = int(population_size)
        self.simulation_batch_size = int(simulation_batch_size)
        self.generations = int(generations)
        self.rollout_steps = int(rollout_steps)
        self.hidden_size = int(hidden_size)
        self.elites = int(elites)
        self.elitism = bool(elitism)
        self.mutation_sigma = float(mutation_sigma)
        self.mutation_probability = float(mutation_probability)
        self.collision_fallback = collision_fallback
        default_fitness_weights = {
            "reward": 1.0,
            "distance_traveled": 0.0,
            "laps_completed": 0.0,
            "offroad_time": 0.0,
            "collision_count": 0.0,
        }
        unknown_weights = set(fitness_weights or {}) - set(default_fitness_weights)
        if unknown_weights:
            raise ValueError(f"unsupported fitness weight keys: {sorted(unknown_weights)}")
        selected_weights = {**default_fitness_weights, **(fitness_weights or {})}
        if not all(math.isfinite(float(value)) for value in selected_weights.values()):
            raise ValueError("fitness weights must be finite numbers")
        self.fitness_weights = {key: float(value) for key, value in selected_weights.items()}
        self.seed = int(seed)
        self.track_seeds = tuple(int(s) for s in track_seeds)
        if not self.track_seeds:
            raise ValueError("track_seeds must include at least one seed")
        self.output_dir = Path(output_dir)
        self.dtype = dtype


class GPUEvolutionTrainer:
    """Evaluate 10k+ batched policies against independent GPU-resident simulations."""

    def __init__(
        self,
        config: Optional[EvolutionConfig] = None,
        training_config: Optional[TrainingConfig] = None,
        *,
        device: str | torch.device = "cuda",
        reference_env: Optional[RacingEnv] = None,
    ) -> None:
        self.config = config or EvolutionConfig()
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "GPUEvolutionTrainer requested CUDA, but CUDA is unavailable. "
                "Use a CUDA-enabled PyTorch install or explicitly pass device='cpu'."
            )
        self.train_config = training_config or TrainingConfig()
        self.train_config.curriculum["enabled"] = False
        if self.train_config.action_mode != ActionMode.CONTINUOUS:
            raise ValueError("GPU evolutionary MLP currently requires continuous actions")
        if self.train_config.observation_mode == ObservationMode.IMAGE:
            raise ValueError("GPU evolution does not support image observations")

        self.reference_env = reference_env or RacingEnv(
            self.train_config, load_config()
        )
        if self.reference_env.track is None:
            self.reference_env.reset(seed=self.config.track_seeds[0])
        self.env = GPUTrainingEnv.from_env(
            self.reference_env,
            batch_size=self.config.simulation_batch_size,
            device=self.device,
            dtype=self.config.dtype,
        )
        obs_size = int(self.env.reset().shape[1])
        self.population = BatchedPolicyPopulation(
            self.config.population_size,
            obs_size,
            self.config.hidden_size,
            device=self.device,
            seed=self.config.seed,
            dtype=self.config.dtype,
        )
        self.history: list[dict[str, Any]] = []
        self.previous_elite_indices: Optional[torch.Tensor] = None
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def evaluate_generation(self, generation: int) -> dict[str, torch.Tensor]:
        """Evaluate all candidates across the same set of fresh tracks for fairness."""
        cfg = self.config
        n = cfg.population_size
        batch_size = cfg.simulation_batch_size
        fitness_sum = torch.zeros(n, dtype=self.config.dtype, device=self.device)
        offroad_sum = torch.zeros_like(fitness_sum)
        collision_sum = torch.zeros(n, dtype=self.config.dtype, device=self.device)
        laps_sum = torch.zeros_like(fitness_sum)
        distance_sum = torch.zeros_like(fitness_sum)
        completed_sum = torch.zeros_like(fitness_sum)
        total_samples = 0

        for seed_offset, base_seed in enumerate(cfg.track_seeds):
            # Vary maps by generation while ensuring every candidate in this
            # generation sees the exact same seeded track set.
            track_seed = int(base_seed + generation * 100_003)
            self.reference_env.reset(seed=track_seed)
            self.env.set_track_from_env(self.reference_env)

            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                indices = torch.arange(start, end, device=self.device, dtype=torch.long)
                valid_count = end - start
                if valid_count < batch_size:
                    padding = indices[0].expand(batch_size - valid_count)
                    policy_indices = torch.cat((indices, padding))
                else:
                    policy_indices = indices
                self.env.reset()

                for _ in range(cfg.rollout_steps):
                    observations = self._get_observations()
                    actions = self.population.actions(observations, policy_indices)
                    self.env.step(actions)

                # Metrics synchronize only at the evaluation boundary, not per step.
                fitness_sum[start:end] += self.env.episode_reward[:valid_count]
                offroad_sum[start:end] += self.env.total_offroad_time[:valid_count]
                collision_sum[start:end] += self.env.collision_count[:valid_count].to(self.config.dtype)
                laps_sum[start:end] += self.env.laps_completed[:valid_count].to(self.config.dtype)
                distance_sum[start:end] += self.env.distance_traveled[:valid_count]
                completed_sum[start:end] += (
                    self.env.terminated[:valid_count] & (self.env.laps_completed[:valid_count] >= self.env.num_laps)
                ).to(self.config.dtype)
                total_samples += 1

        tracks = float(len(cfg.track_seeds))
        reward_mean = fitness_sum / tracks
        offroad_mean = offroad_sum / tracks
        collision_mean = collision_sum / tracks
        laps_mean = laps_sum / tracks
        distance_mean = distance_sum / tracks
        weights = cfg.fitness_weights
        fitness = (
            weights["reward"] * reward_mean
            + weights["distance_traveled"] * distance_mean
            + weights["laps_completed"] * laps_mean
            + weights["offroad_time"] * offroad_mean
            + weights["collision_count"] * collision_mean
        )
        return {
            "fitness": fitness,
            "reward": reward_mean,
            "offroad_time": offroad_mean,
            "collision_count": collision_mean,
            "laps_completed": laps_mean,
            "distance_traveled": distance_mean,
            "completed_episodes": completed_sum / tracks,
            "zero_collision": collision_sum == 0,
        }

    def _get_observations(self) -> torch.Tensor:
        # Public observation construction remains on device; keeping a method
        # wrapper makes alternate observation modes easy to instrument.
        from ai.gpu_observations import build_observations
        return build_observations(self.env)

    @torch.no_grad()
    def select_elites(
        self, metrics: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, bool]:
        fitness = metrics["fitness"]
        eligible = metrics["zero_collision"]
        eligible_indices = torch.nonzero(eligible, as_tuple=False).flatten()
        insufficient = eligible_indices.numel() < self.config.elites

        if not insufficient:
            eligible_scores = fitness.index_select(0, eligible_indices)
            order = torch.topk(
                eligible_scores, k=self.config.elites, largest=True, sorted=True
            ).indices
            selected = eligible_indices.index_select(0, order)
            return selected, False

        strategy = self.config.collision_fallback
        if strategy == "raise":
            raise RuntimeError(
                f"generation has only {eligible_indices.numel()} collision-free candidates; "
                f"{self.config.elites} are required"
            )
        if strategy == "keep_previous_elites":
            if self.previous_elite_indices is None:
                raise RuntimeError(
                    "insufficient collision-free candidates in generation 0 and "
                    "collision_fallback='keep_previous_elites' has no previous elites"
                )
            return self.previous_elite_indices.clone(), True

        # Explicit fallback: keep zero-collision candidates first, then fill
        # remaining slots using highest reward among the remaining candidates.
        order = torch.argsort(fitness, descending=True)
        selected_list = eligible_indices[
            torch.argsort(fitness.index_select(0, eligible_indices), descending=True)
        ].tolist()
        # This selection happens at a generation boundary; avoid duplicate indices.
        chosen = selected_list[: self.config.elites]
        chosen_set = set(chosen)
        for idx in order.tolist():
            if idx not in chosen_set:
                chosen.append(idx)
                chosen_set.add(idx)
            if len(chosen) == self.config.elites:
                break
        return torch.as_tensor(chosen, dtype=torch.long, device=self.device), True

    @torch.no_grad()
    def next_generation(self, elite_indices: torch.Tensor) -> None:
        """Clone the selected top five and mutate offspring; optionally retain elites."""
        n = self.config.population_size
        elites = elite_indices.to(device=self.device, dtype=torch.long)
        if elites.numel() != self.config.elites:
            raise ValueError("exactly five elite indices are required")

        if self.config.elitism:
            # Preserve selected top-five parameter tensors unchanged in slots [0:5].
            self.population.copy_candidates(
                elites,
                torch.arange(self.config.elites, device=self.device),
            )
            self.previous_elite_indices = torch.arange(self.config.elites, device=self.device)
            children = torch.arange(self.config.elites, n, device=self.device)
            parent_positions = torch.randint(
                0,
                self.config.elites,
                (children.numel(),),
                generator=self.population.generator,
                device=self.device,
            )
            parent_indices = parent_positions
        else:
            # Without elitism, every destination is replaced by a mutated clone
            # of one of the selected elites; no original slot survives unchanged.
            self.previous_elite_indices = None
            children = torch.arange(n, device=self.device)
            parent_positions = torch.randint(
                0,
                self.config.elites,
                (children.numel(),),
                generator=self.population.generator,
                device=self.device,
            )
            parent_indices = elites.index_select(0, parent_positions)

        self.population.copy_candidates(parent_indices, children)
        self.population.mutate(
            children,
            sigma=self.config.mutation_sigma,
            mutation_probability=self.config.mutation_probability,
        )

    def _write_metrics(self, generation: int, metrics: dict[str, torch.Tensor]) -> None:
        csv_path = self.config.output_dir / "candidate_metrics.csv"
        exists = csv_path.exists()
        n = self.config.population_size
        host = {
            key: value.detach().cpu().numpy()
            for key, value in metrics.items()
            if key != "zero_collision"
        }
        with csv_path.open("a", newline="", encoding="utf-8") as f:
            fieldnames = [
                "generation", "candidate", "fitness", "reward", "collision_count",
                "offroad_time", "laps_completed", "distance_traveled",
                "completed_episodes", "zero_collision",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not exists:
                writer.writeheader()
            zero_collision = metrics["zero_collision"].detach().cpu().numpy()
            for index in range(n):
                writer.writerow({
                    "generation": generation,
                    "candidate": index,
                    "fitness": float(host["fitness"][index]),
                    "reward": float(host["reward"][index]),
                    "collision_count": float(host["collision_count"][index]),
                    "offroad_time": float(host["offroad_time"][index]),
                    "laps_completed": float(host["laps_completed"][index]),
                    "distance_traveled": float(host["distance_traveled"][index]),
                    "completed_episodes": float(host["completed_episodes"][index]),
                    "zero_collision": bool(zero_collision[index]),
                })

    @torch.no_grad()
    def _save_checkpoint(
        self, generation: int, metrics: dict[str, torch.Tensor], elite_indices: torch.Tensor
    ) -> None:
        path = self.config.output_dir / f"generation_{generation:04d}.pt"
        torch.save({
            "generation": generation,
            "policy_population": self.population.state_dict(),
            "elite_indices": elite_indices.detach().cpu(),
            "metrics": {
                key: value.detach().cpu()
                for key, value in metrics.items()
            },
            "config": {
                "population_size": self.config.population_size,
                "simulation_batch_size": self.config.simulation_batch_size,
                "generations": self.config.generations,
                "rollout_steps": self.config.rollout_steps,
                "hidden_size": self.config.hidden_size,
                "elitism": self.config.elitism,
                "mutation_sigma": self.config.mutation_sigma,
                "mutation_probability": self.config.mutation_probability,
                "collision_fallback": self.config.collision_fallback,
                "fitness_weights": self.config.fitness_weights,
                "seed": self.config.seed,
                "track_seeds": list(self.config.track_seeds),
            },
        }, path)

    def train(self) -> dict[str, Any]:
        """Run configured generations and persist per-generation artifacts."""
        for generation in range(self.config.generations):
            metrics = self.evaluate_generation(generation)
            elite_indices, insufficient = self.select_elites(metrics)
            best_idx = int(torch.argmax(metrics["fitness"]).detach().cpu())
            row = {
                "generation": generation,
                "best_fitness": float(metrics["fitness"].max().detach().cpu()),
                "mean_fitness": float(metrics["fitness"].mean().detach().cpu()),
                "best_candidate": best_idx,
                "collision_free_candidates": int(metrics["zero_collision"].sum().detach().cpu()),
                "insufficient_collision_free_elites": bool(insufficient),
                "elite_indices": elite_indices.detach().cpu().tolist(),
                "best_laps": float(metrics["laps_completed"][best_idx].detach().cpu()),
                "best_distance": float(metrics["distance_traveled"][best_idx].detach().cpu()),
            }
            self.history.append(row)
            self._write_metrics(generation, metrics)
            self._save_checkpoint(generation, metrics, elite_indices)
            (self.config.output_dir / "fitness_history.json").write_text(
                json.dumps(self.history, indent=2) + "\n", encoding="utf-8"
            )
            print(
                f"generation={generation} best={row['best_fitness']:.3f} "
                f"mean={row['mean_fitness']:.3f} zero_collision={row['collision_free_candidates']} "
                f"insufficient_elites={row['insufficient_collision_free_elites']}"
            )
            if generation + 1 < self.config.generations:
                self.next_generation(elite_indices)

        final_elites = self.history[-1]["elite_indices"] if self.history else []
        final_path = self.config.output_dir / "elite_policies.pt"
        selected = torch.as_tensor(final_elites, dtype=torch.long, device=self.device)
        torch.save({
            "elite_policy_population": {
                key: tensor.index_select(0, selected).detach().cpu()
                for key, tensor in (
                    ("w1", self.population.w1),
                    ("b1", self.population.b1),
                    ("w2", self.population.w2),
                    ("b2", self.population.b2),
                )
            },
            "history": self.history,
            "observation_size": self.population.observation_size,
            "hidden_size": self.population.hidden_size,
        }, final_path)
        return {
            "history": self.history,
            "final_elite_checkpoint": str(final_path),
            "output_dir": str(self.config.output_dir),
        }

    def close(self) -> None:
        self.env.close()
        self.reference_env.close()


def load_elite_policy(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cuda",
    seed: int = 0,
) -> BatchedPolicyPopulation:
    """Load the saved top-five set into a five-candidate batched policy stack."""
    data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    elite = data["elite_policy_population"]
    w1 = elite["w1"]
    population = BatchedPolicyPopulation(
        population_size=w1.shape[0],
        observation_size=int(data["observation_size"]),
        hidden_size=int(data["hidden_size"]),
        device=device,
        seed=seed,
        dtype=w1.dtype,
    )
    for key in ("w1", "b1", "w2", "b2"):
        getattr(population, key).copy_(elite[key].to(device=device))
    return population
