"""Reproducible CPU-reference versus batched tensor-backend benchmark.

GPU benchmark sizes default to 1, 64, 256, 1024, 4096 and 10000. For safety,
CPU reference comparison is capped at 64 independent RacingEnv instances by
default; increase --cpu-max-batch for a more expensive measurement. No speedup
or performance numbers are hard-coded; this script writes measurements from the
machine it runs on to JSON.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from ai.gpu_evolution import BatchedPolicyPopulation
from ai.gpu_training_env import GPUTrainingEnv
from ai.training_env import ActionMode, TrainingConfig, RacingEnv
from core.config import load_config


DEFAULT_BATCHES = [1, 64, 256, 1024, 4096, 10000]


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def cpu_benchmark(batch_size: int, steps: int, seed: int, training_config: TrainingConfig) -> dict:
    envs = []
    try:
        for index in range(batch_size):
            env = RacingEnv(training_config, load_config())
            env.reset(seed=seed + index)
            envs.append(env)
        action = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        for env in envs:
            env.step(action)  # warm-up
        start = time.perf_counter()
        episode_ends = 0
        for step in range(steps):
            for env in envs:
                _, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    episode_ends += 1
                    env.reset(seed=seed + step + 1)
        elapsed = time.perf_counter() - start
        return {
            "batch_size": batch_size,
            "steps": steps,
            "wall_seconds": elapsed,
            "simulation_steps_per_second": steps / elapsed,
            "agent_steps_per_second": batch_size * steps / elapsed,
            "completed_or_reset_episodes": episode_ends,
            "status": "measured",
        }
    finally:
        for env in envs:
            env.close()


def tensor_benchmark(
    batch_size: int,
    steps: int,
    seed: int,
    device: torch.device,
    training_config: TrainingConfig,
) -> dict:
    reference = RacingEnv(training_config, load_config())
    try:
        reference.reset(seed=seed)
        env = GPUTrainingEnv.from_env(reference, batch_size=batch_size, device=device)
        policy = BatchedPolicyPopulation(
            batch_size,
            env.reset().shape[1],
            hidden_size=32,
            device=device,
            seed=seed,
        )
        obs = env.reset()
        actions = torch.zeros((batch_size, 3), dtype=env.dtype, device=device)
        actions[:, 1] = 1.0
        # Warm-up both environment and batched policy.
        for _ in range(min(20, steps)):
            actions = policy.actions(obs)
            obs, *_ = env.step(actions)
        synchronize(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        # Isolate policy inference from environment stepping, with proper CUDA sync.
        synchronize(device)
        policy_start = time.perf_counter()
        for _ in range(steps):
            policy.actions(obs)
        synchronize(device)
        policy_elapsed = time.perf_counter() - policy_start

        # End-to-end steady-state throughput (policy + physics/collision/reward/obs).
        synchronize(device)
        start = time.perf_counter()
        reward_sum = torch.zeros(batch_size, dtype=env.dtype, device=device)
        for _ in range(steps):
            actions = policy.actions(obs)
            obs, rewards, _, _, _ = env.step(actions)
            reward_sum.add_(rewards)
        synchronize(device)
        elapsed = time.perf_counter() - start
        allocated = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        reserved = torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0
        total_memory = torch.cuda.get_device_properties(device).total_memory if device.type == "cuda" else None
        env.close()

        # Opt-in synchronization inside step partitions physics integration,
        # boundary-distance work, vehicle forces, rewards/termination and obs.
        profile_env = GPUTrainingEnv.from_env(
            reference, batch_size=batch_size, device=device, profile_timing=True
        )
        profile_env.reset()
        profile_actions = torch.zeros((batch_size, 3), dtype=profile_env.dtype, device=device)
        profile_actions[:, 1] = 1.0
        profile_samples = []
        profile_steps = min(max(steps, 5), 25)
        for _ in range(3):
            profile_env.step(profile_actions)
        for _ in range(profile_steps):
            profile_env.step(profile_actions)
            profile_samples.append(profile_env.last_step_timings.copy())
        synchronize(device)
        phase_keys = sorted({key for sample in profile_samples for key in sample})
        component_timings = {
            key: float(np.mean([sample.get(key, 0.0) for sample in profile_samples]))
            for key in phase_keys
        }
        profile_env.close()
        return {
            "batch_size": batch_size,
            "steps": steps,
            "wall_seconds": elapsed,
            "simulation_steps_per_second": steps / elapsed,
            "agent_steps_per_second": batch_size * steps / elapsed,
            "policy_inference_seconds": policy_elapsed,
            "policy_inference_seconds_per_batch_step": policy_elapsed / steps,
            "environment_components_seconds_per_step": component_timings,
            "mean_reward_per_agent": float(reward_sum.mean().detach().cpu()),
            "peak_gpu_allocated_bytes": int(allocated),
            "peak_gpu_reserved_bytes": int(reserved),
            "gpu_total_memory_bytes": int(total_memory) if total_memory is not None else None,
            "status": "measured",
        }
    finally:
        reference.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batches", nargs="+", type=int, default=DEFAULT_BATCHES)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu-max-batch", type=int, default=64)
    parser.add_argument("--output", default="training_runs/gpu_benchmark.json")
    args = parser.parse_args()

    if args.steps < 1 or any(value < 1 for value in args.batches):
        parser.error("batch sizes and steps must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA benchmark requested but CUDA is unavailable. Install a CUDA-enabled "
            "PyTorch build or explicitly run --device cpu to benchmark the tensor backend."
        )

    training_config = TrainingConfig(action_mode=ActionMode.CONTINUOUS)
    training_config.curriculum["enabled"] = False
    results = {
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
            "dtype": "float32",
            "seed": args.seed,
            "steps_per_batch": args.steps,
        },
        "results": [],
        "notes": [
            "GPU measurements include policy inference and environment observation/reward work.",
            "Track generation and initial tensor upload are excluded from steady-state timing.",
            "CPU reference benchmark is limited by --cpu-max-batch; skipped entries are not measurements.",
            "The focused reference environment has no static track-boundary or opponent collision bodies.",
        ],
    }

    for batch in args.batches:
        print(f"Benchmarking tensor backend batch={batch} device={device}...")
        try:
            tensor_result = tensor_benchmark(
                batch, args.steps, args.seed, device, training_config
            )
            results["results"].append({"backend": "batched_tensor", **tensor_result})
        except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            results["results"].append({
                "backend": "batched_tensor",
                "batch_size": batch,
                "status": "failed",
                "error": str(exc),
            })
            if batch == args.batches[0]:
                raise

        if batch <= args.cpu_max_batch:
            print(f"Benchmarking CPU reference batch={batch}...")
            try:
                results["results"].append({
                    "backend": "cpu_reference",
                    **cpu_benchmark(batch, args.steps, args.seed, training_config),
                })
            except Exception as exc:
                results["results"].append({
                    "backend": "cpu_reference",
                    "batch_size": batch,
                    "status": "failed",
                    "error": str(exc),
                })
        else:
            results["results"].append({
                "backend": "cpu_reference",
                "batch_size": batch,
                "status": "skipped",
                "reason": f"batch exceeds cpu-max-batch={args.cpu_max_batch}",
            })

    # Compute only comparable same-batch speedups from measured timings.
    cpu_times = {
        row["batch_size"]: row["wall_seconds"]
        for row in results["results"]
        if row["backend"] == "cpu_reference" and row["status"] == "measured"
    }
    for row in results["results"]:
        if row["backend"] == "batched_tensor" and row["status"] == "measured":
            cpu_time = cpu_times.get(row["batch_size"])
            row["cpu_over_tensor_speedup"] = (
                cpu_time / row["wall_seconds"]
                if cpu_time is not None and device.type == "cuda"
                else None
            )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Benchmark results written to {output}")


if __name__ == "__main__":
    main()
