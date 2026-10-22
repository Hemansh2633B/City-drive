"""
Procedural Track Generator

Generates deterministic, validated racing tracks from seeds.
Supports multiple track types, complexity levels, and themes.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass
import math

from core.random import DeterministicRNG
from core.math import (
    Vec2, vec2, vec2_add, vec2_sub, vec2_mul, vec2_normalize, vec2_perp,
    vec2_length, vec2_angle, vec2_rotate, vec2_lerp, angle_difference, clamp,
    vec2_distance, point_line_distance
)
from track.track import Track, TrackSegment, Checkpoint, RacingLinePoint


@dataclass
class TrackGeneratorConfig:
    """Configuration for track generation."""
    min_complexity: float = 0.2
    max_complexity: float = 0.8
    min_road_width: float = 8.0
    max_road_width: float = 16.0
    min_track_length: float = 400.0
    max_track_length: float = 2000.0
    min_checkpoints: int = 8
    max_checkpoints: int = 24
    min_corner_radius: float = 5.0
    max_corner_radius: float = 50.0
    enable_hairpins: bool = True
    enable_chicanes: bool = True
    enable_s_curves: bool = True
    max_validation_attempts: int = 10


class TrackGenerator:
    """
    Deterministic procedural track generator.
    
    Uses a seed to generate consistent tracks with:
    - Closed loop centerlines
    - Varied corner types
    - Proper checkpoints
    - Racing line computation
    - Validation and regeneration
    """
    
    def __init__(self, config: TrackGeneratorConfig, rng: DeterministicRNG):
        self.config = config
        self.rng = rng
    
    def generate(self, seed: Union[int, str], theme: str = "asphalt",
                 complexity: Optional[float] = None,
                 road_width: Optional[float] = None,
                 length: Optional[float] = None) -> Track:
        """
        Generate a track from seed.
        
        Args:
            seed: Seed for deterministic generation
            theme: Visual theme name
            complexity: Override complexity (0.0 to 1.0)
            road_width: Override road width
            length: Override track length
            
        Returns:
            Generated and validated Track
        """
        # Create track-specific RNG from seed
        track_rng = DeterministicRNG(seed)
        
        # Determine parameters
        if complexity is None:
            complexity = track_rng.uniform(self.config.min_complexity, self.config.max_complexity)
        else:
            complexity = clamp(complexity, 0.0, 1.0)
        
        if road_width is None:
            road_width = track_rng.uniform(self.config.min_road_width, self.config.max_road_width)
        else:
            road_width = clamp(road_width, self.config.min_road_width, self.config.max_road_width)
        
        if length is None:
            length = track_rng.uniform(self.config.min_track_length, self.config.max_track_length)
        else:
            length = clamp(length, self.config.min_track_length, self.config.max_track_length)
        
        # Generate track with validation attempts
        for attempt in range(self.config.max_validation_attempts):
            attempt_rng = DeterministicRNG(seed, sequence=attempt)
            track = self._generate_track(attempt_rng, theme, complexity, road_width, length)
            
            if self._validate_track(track):
                track.is_valid = True
                return track
            
            # Track invalid, try next attempt
            track.validation_errors.append(f"Attempt {attempt + 1} failed validation")
        
        # All attempts failed, return best attempt
        track.is_valid = False
        return track
    
    def _generate_track(self, rng: DeterministicRNG, theme: str,
                        complexity: float, road_width: float, target_length: float) -> Track:
        """Generate a single track attempt."""
        track = Track(
            seed=rng.seed,
            theme=theme,
            complexity=complexity,
            road_width=road_width,
            length=target_length
        )
        
        # Determine number of major corners based on complexity
        num_corners = int(self.config.min_checkpoints + complexity * 
                         (self.config.max_checkpoints - self.config.min_checkpoints))
        num_corners = max(4, num_corners)  # At least 4 corners for a loop
        
        # Generate centerline using spline-based approach
        centerline = self._generate_centerline(rng, num_corners, target_length, complexity)
        
        # Ensure it's a closed loop
        centerline = self._close_loop(centerline)
        
        # Build track from centerline
        track.centerline = centerline
        track.total_length = self._calculate_length(centerline)
        
        # Generate segments with boundaries
        track.segments = self._build_segments(centerline, road_width, rng)
        
        # Build left/right boundaries
        track.left_boundary, track.right_boundary = self._build_boundaries(centerline, road_width)
        
        # Generate checkpoints
        track.checkpoints = self._generate_checkpoints(centerline, road_width)
        
        # Generate racing line
        track.racing_line = self._generate_racing_line(centerline, road_width, rng)
        
        # Generate spawn positions
        track.start_positions, track.start_angles = self._generate_spawn_positions(
            centerline, road_width, track.checkpoints[0] if track.checkpoints else None
        )
        
        # Theme data
        track.theme_data = self._get_theme_data(theme)
        
        return track
    
    def _generate_centerline(self, rng: DeterministicRNG, num_corners: int,
                             target_length: float, complexity: float) -> List[Vec2]:
        """
        Generate a simple, smooth, closed circuit from a deterministic seed.

        The old generator stitched independent turns together and then bridged the
        remaining gap with a spline. That could create abrupt joins or ambiguous
        crossings. This polar-loop construction keeps the radius positive at every
        angle, so the centerline is closed and non-self-intersecting by construction.
        Complexity changes the shape gently rather than making courses impossible.
        """
        samples = 160
        complexity = clamp(complexity, 0.0, 1.0)

        # An ellipse gives a more natural driving line than a perfect circle.
        # Ramanujan's perimeter approximation makes generated lengths close to the
        # requested target before the small, bounded shape variations are applied.
        aspect_x, aspect_y = 1.18, 0.86
        perimeter_factor = math.pi * (
            3.0 * (aspect_x + aspect_y)
            - math.sqrt((3.0 * aspect_x + aspect_y) * (aspect_x + 3.0 * aspect_y))
        )
        radius = max(1.0, target_length / perimeter_factor)

        # Seed selects both shape frequency and phase, so seeds produce genuinely
        # different circuits. The summed amplitude remains below 0.12; radius stays
        # positive, which prevents the radial loop from crossing itself.
        phases = [rng.uniform(0.0, 2.0 * math.pi) for _ in range(3)]
        max_lobes = 3 + int(2.0 * complexity)
        lobes = int(rng.integers(2, max_lobes + 1))
        primary_amp = 0.035 + 0.035 * complexity
        secondary_amp = 0.015 + 0.020 * complexity
        tertiary_amp = 0.005 + 0.005 * complexity

        points: List[Vec2] = []
        for i in range(samples + 1):
            theta = 2.0 * math.pi * i / samples
            radial = (
                1.0
                + primary_amp * math.sin(lobes * theta + phases[0])
                + secondary_amp * math.sin((lobes + 2) * theta + phases[1])
                + tertiary_amp * math.sin((lobes + 3) * theta + phases[2])
            )
            x = radius * aspect_x * radial * math.cos(theta)
            y = radius * aspect_y * radial * math.sin(theta)
            points.append(vec2(x, y))

        # Exact closure avoids microscopic gaps from floating-point trig values.
        points[-1] = points[0].copy()
        return points

    def _choose_corner_type(self, rng: DeterministicRNG, complexity: float) -> str:
        """Choose corner type based on complexity and randomness."""
        r = rng.random()
        
        # Base probabilities
        probs = {
            "straight": 0.3,
            "corner": 0.4,
            "hairpin": 0.1,
            "chicane": 0.1,
            "s_curve": 0.1
        }
        
        # Adjust based on complexity
        probs["straight"] *= (1.0 - complexity * 0.5)
        probs["corner"] *= (1.0 + complexity * 0.2)
        probs["hairpin"] *= (1.0 + complexity)
        probs["chicane"] *= (1.0 + complexity * 1.5)
        probs["s_curve"] *= (1.0 + complexity)
        
        # Disable if not enabled
        if not self.config.enable_hairpins:
            probs["hairpin"] = 0
        if not self.config.enable_chicanes:
            probs["chicane"] = 0
        if not self.config.enable_s_curves:
            probs["s_curve"] = 0
        
        # Normalize
        total = sum(probs.values())
        probs = {k: v/total for k, v in probs.items()}
        
        # Select
        cumsum = 0
        for corner_type, prob in probs.items():
            cumsum += prob
            if r <= cumsum:
                return corner_type
        
        return "corner"
    
    def _close_loop(self, points: List[Vec2]) -> List[Vec2]:
        """Ensure the track forms a closed loop."""
        if len(points) < 4:
            return points
        
        start = points[0]
        end = points[-1]
        gap = vec2_distance(start, end)
        
        if gap < 5.0:
            # Already close enough, just ensure closure
            points[-1] = start
            return points
        
        # Add connecting segment
        # Create a smooth connection using a curve
        num_segments = max(3, int(gap / 20.0))
        start_dir = vec2_normalize(vec2_sub(points[1], points[0]))
        end_dir = vec2_normalize(vec2_sub(points[-1], points[-2]))
        
        # Use cubic bezier for smooth connection
        p0 = end
        p3 = start
        # Control points extending from endpoints
        handle_len = gap * 0.4
        p1 = vec2_add(end, vec2_mul(end_dir, handle_len))
        p2 = vec2_add(start, vec2_mul(vec2_mul(start_dir, -1), handle_len))
        
        for i in range(1, num_segments):
            t = i / num_segments
            # Cubic bezier
            mt = 1 - t
            point = vec2_add(
                vec2_add(
                    vec2_add(
                        vec2_mul(p0, mt**3),
                        vec2_mul(p1, 3 * mt**2 * t)
                    ),
                    vec2_mul(p2, 3 * mt * t**2)
                ),
                vec2_mul(p3, t**3)
            )
            points.append(point)
        
        points[-1] = start  # Ensure exact closure
        return points
    
    def _calculate_length(self, points: List[Vec2]) -> float:
        """Calculate total track length."""
        length = 0.0
        for i in range(len(points) - 1):
            length += vec2_distance(points[i], points[i + 1])
        return length
    
    def _build_segments(self, centerline: List[Vec2], road_width: float, 
                        rng: DeterministicRNG) -> List[TrackSegment]:
        """Build track segments from centerline."""
        segments = []
        
        for i in range(len(centerline) - 1):
            start = centerline[i]
            end = centerline[i + 1]
            
            direction = vec2_normalize(vec2_sub(end, start))
            perp = vec2_perp(direction)
            
            left_start = vec2_add(start, vec2_mul(perp, road_width * 0.5))
            left_end = vec2_add(end, vec2_mul(perp, road_width * 0.5))
            right_start = vec2_sub(start, vec2_mul(perp, road_width * 0.5))
            right_end = vec2_sub(end, vec2_mul(perp, road_width * 0.5))
            
            # Determine segment type (simplified)
            seg_type = "straight"
            corner_radius = 0.0
            corner_direction = 0
            
            if i > 0 and i < len(centerline) - 2:
                prev_dir = vec2_normalize(vec2_sub(centerline[i], centerline[i - 1]))
                next_dir = vec2_normalize(vec2_sub(centerline[i + 2], centerline[i + 1]))
                angle_change = angle_difference(vec2_angle(prev_dir), vec2_angle(next_dir))
                
                if abs(angle_change) > 0.2:
                    seg_type = "corner"
                    corner_radius = road_width / (2 * abs(math.sin(angle_change / 2))) if abs(math.sin(angle_change / 2)) > 0.01 else 50.0
                    corner_direction = 1 if angle_change > 0 else -1
            
            segments.append(TrackSegment(
                start=start,
                end=end,
                width=road_width,
                left_boundary=(left_start, left_end),
                right_boundary=(right_start, right_end),
                segment_type=seg_type,
                corner_radius=corner_radius,
                corner_direction=corner_direction
            ))
        
        return segments
    
    def _build_boundaries(self, centerline: List[Vec2], road_width: float) -> Tuple[List[Vec2], List[Vec2]]:
        """Build left and right boundary polylines."""
        left = []
        right = []
        half_width = road_width * 0.5
        
        for i in range(len(centerline) - 1):
            start = centerline[i]
            end = centerline[i + 1]
            direction = vec2_normalize(vec2_sub(end, start))
            perp = vec2_perp(direction)
            
            left.append(vec2_add(start, vec2_mul(perp, half_width)))
            right.append(vec2_sub(start, vec2_mul(perp, half_width)))
        
        # Add final point
        if len(centerline) > 1:
            last = centerline[-1]
            prev = centerline[-2]
            direction = vec2_normalize(vec2_sub(last, prev))
            perp = vec2_perp(direction)
            left.append(vec2_add(last, vec2_mul(perp, half_width)))
            right.append(vec2_sub(last, vec2_mul(perp, half_width)))
        
        return left, right
    
    def _generate_checkpoints(self, centerline: List[Vec2], road_width: float) -> List[Checkpoint]:
        """Generate evenly spaced checkpoints along centerline."""
        checkpoints = []
        
        # Choose spacing from actual loop length so short seeds still have
        # enough checkpoints and long seeds do not create an excessive number.
        total_length = self._calculate_length(centerline)
        desired_count = int(round(total_length / 60.0))
        desired_count = max(
            self.config.min_checkpoints,
            min(self.config.max_checkpoints, desired_count)
        )
        target_spacing = total_length / (desired_count + 0.5)
        accumulated = 0.0
        last_checkpoint_idx = 0
        
        for i in range(len(centerline) - 1):
            seg_len = vec2_distance(centerline[i], centerline[i + 1])
            accumulated += seg_len
            
            if accumulated >= target_spacing or i == len(centerline) - 2:
                # Place checkpoint at this position
                pos = centerline[i]
                if i < len(centerline) - 1:
                    direction = vec2_normalize(vec2_sub(centerline[i + 1], centerline[i]))
                else:
                    direction = vec2_normalize(vec2_sub(centerline[i], centerline[i - 1]))
                
                cp = Checkpoint(
                    position=pos,
                    direction=direction,
                    width=road_width * 1.2,
                    index=len(checkpoints),
                    is_start_finish=(len(checkpoints) == 0)
                )
                checkpoints.append(cp)
                accumulated -= target_spacing
                last_checkpoint_idx = i
        
        # Ensure at least minimum checkpoints
        while len(checkpoints) < 4:
            # Add more checkpoints by subdividing
            new_checkpoints = []
            for i in range(len(checkpoints)):
                new_checkpoints.append(checkpoints[i])
                if i < len(checkpoints) - 1:
                    mid_pos = vec2_lerp(checkpoints[i].position, checkpoints[i + 1].position, 0.5)
                    mid_dir = vec2_normalize(vec2_add(checkpoints[i].direction, checkpoints[i + 1].direction))
                    new_checkpoints.append(Checkpoint(
                        position=mid_pos,
                        direction=mid_dir,
                        width=road_width * 1.2,
                        index=len(new_checkpoints)
                    ))
            if len(checkpoints) == len(new_checkpoints):
                break
            checkpoints = new_checkpoints
        
        # Re-index
        for i, cp in enumerate(checkpoints):
            cp.index = i
            cp.is_start_finish = (i == 0)
        
        return checkpoints
    
    def _generate_racing_line(self, centerline: List[Vec2], road_width: float,
                              rng: DeterministicRNG) -> List[RacingLinePoint]:
        """Generate optimal racing line (slightly inside corners)."""
        racing_line = []
        half_width = road_width * 0.5
        
        for i in range(len(centerline)):
            pos = centerline[i]
            
            # Get direction
            if i < len(centerline) - 1:
                direction = vec2_normalize(vec2_sub(centerline[i + 1], centerline[i]))
            else:
                direction = vec2_normalize(vec2_sub(centerline[i], centerline[i - 1]))
            
            # Calculate curvature
            curvature = 0.0
            if i > 0 and i < len(centerline) - 1:
                prev_dir = vec2_normalize(vec2_sub(centerline[i], centerline[i - 1]))
                next_dir = vec2_normalize(vec2_sub(centerline[i + 1], centerline[i]))
                angle_change = angle_difference(vec2_angle(prev_dir), vec2_angle(next_dir))
                seg_len = (vec2_distance(centerline[i - 1], centerline[i]) + 
                          vec2_distance(centerline[i], centerline[i + 1])) * 0.5
                if seg_len > 0:
                    curvature = abs(angle_change) / seg_len
            
            # Racing line hugs inside of corners
            offset = 0.0
            if curvature > 0.01:
                # Move toward inside of corner
                perp = vec2_perp(direction)
                # Determine which side is inside
                if i > 0:
                    prev_pos = centerline[i - 1]
                    to_prev = vec2_normalize(vec2_sub(prev_pos, pos))
                    cross = to_prev[0] * direction[1] - to_prev[1] * direction[0]
                    offset = -cross * half_width * 0.6 * min(1.0, curvature * 20.0)
            
            line_pos = vec2_add(pos, vec2_mul(vec2_perp(direction), offset))
            
            # Target speed based on curvature
            if curvature > 0:
                target_speed = min(50.0, math.sqrt(9.81 * road_width * 0.5 / max(curvature, 0.001)))
            else:
                target_speed = 60.0
            
            racing_line.append(RacingLinePoint(
                position=line_pos,
                direction=direction,
                target_speed=target_speed,
                curvature=curvature
            ))
        
        return racing_line
    
    def _generate_spawn_positions(self, centerline: List[Vec2], road_width: float,
                                  start_checkpoint: Optional[Checkpoint]) -> Tuple[List[Vec2], List[float]]:
        """Generate a compact starting grid that fits inside the road."""
        positions: List[Vec2] = []
        angles: List[float] = []

        if start_checkpoint:
            perp = vec2_perp(start_checkpoint.direction)
            # Keep every grid slot comfortably inside the road. The previous fixed
            # 3.5 m spacing placed the first car outside narrow tracks at reset.
            lateral_spacing = min(1.5, max(0.5, road_width * 0.18))
            offsets = [0.0, -lateral_spacing, lateral_spacing,
                       -2.0 * lateral_spacing, 2.0 * lateral_spacing]
            for i in range(5):
                offset = offsets[i]
                pos = vec2_add(start_checkpoint.position, vec2_mul(perp, offset))
                pos = vec2_sub(pos, vec2_mul(start_checkpoint.direction, 5.0 + i * 2.5))
                positions.append(pos)
                angles.append(vec2_angle(start_checkpoint.direction))
        else:
            for i in range(5):
                positions.append(vec2(float(i * 2), 0.0))
                angles.append(0.0)

        return positions, angles

    def _get_theme_data(self, theme: str) -> Dict[str, Any]:
        """Get theme-specific data."""
        themes = {
            "asphalt": {"surface_friction": 1.0, "offroad_friction": 0.3},
            "desert": {"surface_friction": 0.85, "offroad_friction": 0.25},
            "snow": {"surface_friction": 0.4, "offroad_friction": 0.15},
            "grassland": {"surface_friction": 0.9, "offroad_friction": 0.35},
            "neon": {"surface_friction": 1.1, "offroad_friction": 0.2},
            "industrial": {"surface_friction": 1.0, "offroad_friction": 0.3},
            "dirt": {"surface_friction": 0.7, "offroad_friction": 0.5}
        }
        return themes.get(theme, themes["asphalt"])
    
    def _validate_track(self, track: Track) -> bool:
        """Validate track meets all requirements."""
        errors = []
        
        # Check centerline forms loop
        if len(track.centerline) < 4:
            errors.append("Centerline too short")
        
        start = track.centerline[0]
        end = track.centerline[-1]
        if vec2_distance(start, end) > 1.0:
            errors.append("Track does not form closed loop")
        
        # Check track length
        if track.total_length < self.config.min_track_length * 0.5:
            errors.append(f"Track too short: {track.total_length:.1f}m")
        if track.total_length > self.config.max_track_length * 1.5:
            errors.append(f"Track too long: {track.total_length:.1f}m")
        
        # Check checkpoints
        if len(track.checkpoints) < self.config.min_checkpoints:
            errors.append(f"Too few checkpoints: {len(track.checkpoints)}")
        
        # Checkpoints should follow the route and remain reasonably spaced.
        nominal_checkpoint_gap = track.total_length / max(1, len(track.checkpoints))
        max_checkpoint_gap = max(track.road_width * 10, nominal_checkpoint_gap * 1.5)
        for i, cp in enumerate(track.checkpoints):
            next_cp = track.checkpoints[(i + 1) % len(track.checkpoints)]
            dist = vec2_distance(cp.position, next_cp.position)
            if dist > max_checkpoint_gap:
                errors.append(f"Checkpoint {i} to {next_cp.index} too far: {dist:.1f}m")
        
        # Check boundaries don't self-intersect (simplified)
        # Check spawn positions are valid
        if len(track.start_positions) < 1:
            errors.append("No valid spawn positions")
        
        track.validation_errors = errors
        return len(errors) == 0


from typing import Union
from core.math import clamp
import math