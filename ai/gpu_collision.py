"""CUDA-vectorized geometry helpers for the single-car training environment.

The reference RacingEnv does not create static road-boundary bodies. Its road
membership rule is distance to the nearest centerline segment <= 1.1 * half
road width. The physics world also contains no opponent bodies in the focused
single-agent training workflow, so separate population slots must not collide
with one another.
"""
from __future__ import annotations

import torch


def nearest_centerline_distance(
    positions: torch.Tensor,
    starts: torch.Tensor,
    segments: torch.Tensor,
    segment_length_sq: torch.Tensor,
    *,
    chunk_size: int = 64,
) -> torch.Tensor:
    """Return nearest polyline distance for each position without CPU copies.

    positions: [batch, 2]; starts/segments: [segments, 2]. Segment chunking
    bounds temporary memory for large agent populations and detailed tracks.
    """
    if positions.ndim != 2 or positions.shape[-1] != 2:
        raise ValueError("positions must have shape [batch, 2]")
    if starts.shape != segments.shape or starts.ndim != 2 or starts.shape[-1] != 2:
        raise ValueError("starts and segments must both have shape [segments, 2]")
    if starts.shape[0] == 0:
        return torch.full(
            (positions.shape[0],), float("inf"), dtype=positions.dtype,
            device=positions.device,
        )
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")

    best_sq = torch.full(
        (positions.shape[0],), float("inf"), dtype=positions.dtype,
        device=positions.device,
    )
    for begin in range(0, starts.shape[0], chunk_size):
        end = min(begin + chunk_size, starts.shape[0])
        seg = segments[begin:end]
        rel = positions[:, None, :] - starts[None, begin:end, :]
        denom = segment_length_sq[None, begin:end].clamp_min(1e-12)
        fraction = ((rel * seg[None, :, :]).sum(dim=-1) / denom).clamp(0.0, 1.0)
        closest = starts[None, begin:end, :] + fraction[..., None] * seg[None, :, :]
        delta = positions[:, None, :] - closest
        distance_sq = (delta * delta).sum(dim=-1).amin(dim=1)
        best_sq = torch.minimum(best_sq, distance_sq)
    return best_sq.clamp_min(0.0).sqrt()


def forward_checkpoint_crossing(
    previous_position: torch.Tensor,
    position: torch.Tensor,
    checkpoint_positions: torch.Tensor,
    checkpoint_directions: torch.Tensor,
    checkpoint_widths: torch.Tensor,
    target_checkpoint: torch.Tensor,
) -> torch.Tensor:
    """Vectorized equivalent of Checkpoint.crossed_by for one target per slot."""
    if checkpoint_positions.shape[0] == 0:
        return torch.zeros(
            position.shape[0], dtype=torch.bool, device=position.device
        )
    cp_pos = checkpoint_positions[target_checkpoint]
    cp_dir = checkpoint_directions[target_checkpoint]
    width = checkpoint_widths[target_checkpoint]

    previous_offset = previous_position - cp_pos
    current_offset = position - cp_pos
    previous_side = (previous_offset * cp_dir).sum(dim=1)
    current_side = (current_offset * cp_dir).sum(dim=1)
    travel = current_side - previous_side
    positive_travel = travel > 1e-9
    safe_travel = torch.where(positive_travel, travel, torch.ones_like(travel))
    fraction = (-previous_side / safe_travel).clamp(0.0, 1.0)
    crossing = previous_position + (position - previous_position) * fraction[:, None]
    lateral = crossing - cp_pos
    perpendicular = torch.stack((-cp_dir[:, 1], cp_dir[:, 0]), dim=1)
    lateral_distance = (lateral * perpendicular).sum(dim=1).abs()

    return (
        (previous_side <= 0.0)
        & (current_side >= 0.0)
        & positive_travel
        & (lateral_distance <= width * 0.5)
    )
