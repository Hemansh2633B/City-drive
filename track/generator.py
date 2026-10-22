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
    - Validation and regeneration with fallback
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
            Generated and validated Track (or fallback track if validation fails)
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
            try:
                attempt_rng = DeterministicRNG(seed, sequence=attempt)
                track = self._generate_track(attempt_rng, theme, complexity, road_width, length)

                if self._validate_track(track):
                    track.is_valid = True
                    return track
            except Exception as e:
                pass

        # Fallback to guaranteed valid seed-based oval track
        return self._generate_fallback_oval_track(seed, theme, road_width)

    def _generate_fallback_oval_track(self, seed: Union[int, str], theme: str, road_width: float) -> Track:
        """Generate a simple guaranteed-valid oval track as fallback."""
        track = Track(
            seed=seed,
            theme=theme,
            complexity=0.1,
            road_width=road_width,
            length=600.0,
            is_valid=True
        )
        
        # Generate smooth oval centerline
        num_points = 32
        radius_x = 100.0
        radius_y = 50.0
        centerline = []
        for i in range(num_points):
            angle = 2.0 * math.pi * i / num_points
            x = radius_x * math.cos(angle)
            y = radius_y * math.sin(angle)
            centerline.append(vec2(x, y))
        centerline.append(centerline[0]) # Closed loop

        track.centerline = centerline
        track.total_length = self._calculate_length(centerline)
        track.segments = self._build_segments(centerline, road_width, DeterministicRNG(seed))
        track.left_boundary, track.right_boundary = self._build_boundaries(centerline, road_width)
        track.checkpoints = self._generate_checkpoints(centerline, road_width)
        track.racing_line = self._generate_racing_line(centerline, road_width, DeterministicRNG(seed))
        track.start_positions, track.start_angles = self._generate_spawn_positions(
            centerline, road_width, track.checkpoints[0] if track.checkpoints else None
        )
        track.theme_data = self._get_theme_data(theme)
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
        """Generate track centerline using parametric curve."""
        points = []
        
        current_pos = vec2(0.0, 0.0)
        current_angle = 0.0
        
        corner_params = []
        for i in range(num_corners):
            corner_type = self._choose_corner_type(rng, complexity)
            
            if corner_type == "straight":
                length = rng.uniform(30.0, 100.0)
                corner_params.append(("straight", length, 0.0, 0))
            elif corner_type == "corner":
                radius = rng.uniform(self.config.min_corner_radius, self.config.max_corner_radius)
                angle = rng.uniform(0.5, 1.5) * (1 if rng.random() > 0.5 else -1)
                length = abs(angle) * radius
                corner_params.append(("corner", length, radius, 1 if angle > 0 else -1))
            elif corner_type == "hairpin":
                radius = rng.uniform(self.config.min_corner_radius, 15.0)
                angle = math.pi * (0.9 + rng.random() * 0.2) * (1 if rng.random() > 0.5 else -1)
                length = abs(angle) * radius
                corner_params.append(("hairpin", length, radius, 1 if angle > 0 else -1))
            elif corner_type == "chicane":
                radius = rng.uniform(8.0, 20.0)
                angle1 = rng.uniform(0.6, 1.0) * (1 if rng.random() > 0.5 else -1)
                corner_params.append(("chicane", 0, radius, 1 if angle1 > 0 else -1))
            elif corner_type == "s_curve":
                radius = rng.uniform(15.0, 35.0)
                angle1 = rng.uniform(0.5, 1.0) * (1 if rng.random() > 0.5 else -1)
                corner_params.append(("s_curve", 0, radius, 1 if angle1 > 0 else -1))
        
        total_param_length = sum(p[1] for p in corner_params if p[0] not in ("chicane", "s_curve"))
        for p in corner_params:
            if p[0] in ("chicane", "s_curve"):
                total_param_length += p[2] * 2.5
        
        scale = target_length / max(total_param_length, 1.0)
        
        points.append(current_pos)
        
        for i, (ctype, length, radius, direction) in enumerate(corner_params):
            if ctype == "straight":
                length *= scale
                for _ in range(max(2, int(length / 10.0))):
                    current_pos = vec2_add(current_pos, vec2_mul(vec2_normalize(vec2(math.cos(current_angle), math.sin(current_angle))), length / max(2, int(length / 10.0))))
                    points.append(current_pos)
            
            elif ctype == "corner":
                radius *= scale
                length = abs(length * scale)
                angle = length / radius * direction
                steps = max(3, int(abs(angle) * 5))
                for step in range(1, steps + 1):
                    a = angle * step / steps
                    offset = vec2_rotate(vec2(radius * direction, 0), current_angle + a - direction * math.pi/2)
                    center = vec2_add(current_pos, vec2_sub(offset, vec2_rotate(vec2(radius * direction, 0), current_angle - direction * math.pi/2)))
                    points.append(center)
                current_angle += angle
                current_pos = points[-1]
            
            elif ctype == "hairpin":
                radius *= scale
                angle = math.pi * direction * 0.95
                steps = max(5, int(abs(angle) * 8))
                for step in range(1, steps + 1):
                    a = angle * step / steps
                    offset = vec2_rotate(vec2(radius * direction, 0), current_angle + a - direction * math.pi/2)
                    center = vec2_add(current_pos, vec2_sub(offset, vec2_rotate(vec2(radius * direction, 0), current_angle - direction * math.pi/2)))
                    points.append(center)
                current_angle += angle
                current_pos = points[-1]
            
            elif ctype == "chicane":
                radius *= scale
                angle1 = rng.uniform(0.6, 1.0) * direction
                angle2 = -angle1 * rng.uniform(0.8, 1.2)
                
                steps1 = max(3, int(abs(angle1) * 5))
                for step in range(1, steps1 + 1):
                    a = angle1 * step / steps1
                    offset = vec2_rotate(vec2(radius * direction, 0), current_angle + a - direction * math.pi/2)
                    center = vec2_add(current_pos, vec2_sub(offset, vec2_rotate(vec2(radius * direction, 0), current_angle - direction * math.pi/2)))
                    points.append(center)
                current_angle += angle1
                current_pos = points[-1]
                
                mid_length = radius * 1.5
                for _ in range(2):
                    current_pos = vec2_add(current_pos, vec2_mul(vec2_normalize(vec2(math.cos(current_angle), math.sin(current_angle))), mid_length / 2))
                    points.append(current_pos)
                
                steps2 = max(3, int(abs(angle2) * 5))
                for step in range(1, steps2 + 1):
                    a = angle2 * step / steps2
                    offset = vec2_rotate(vec2(radius * -direction, 0), current_angle + a + direction * math.pi/2)
                    center = vec2_add(current_pos, vec2_sub(offset, vec2_rotate(vec2(radius * -direction, 0), current_angle + direction * math.pi/2)))
                    points.append(center)
                current_angle += angle2
                current_pos = points[-1]
            
            elif ctype == "s_curve":
                radius *= scale
                angle1 = rng.uniform(0.5, 1.0) * direction
                angle2 = -angle1 * rng.uniform(0.7, 1.0)
                
                steps1 = max(4, int(abs(angle1) * 6))
                for step in range(1, steps1 + 1):
                    a = angle1 * step / steps1
                    offset = vec2_rotate(vec2(radius * direction, 0), current_angle + a - direction * math.pi/2)
                    center = vec2_add(current_pos, vec2_sub(offset, vec2_rotate(vec2(radius * direction, 0), current_angle - direction * math.pi/2)))
                    points.append(center)
                current_angle += angle1
                current_pos = points[-1]
                
                steps2 = max(4, int(abs(angle2) * 6))
                for step in range(1, steps2 + 1):
                    a = angle2 * step / steps2
                    offset = vec2_rotate(vec2(radius * -direction, 0), current_angle + a + direction * math.pi/2)
                    center = vec2_add(current_pos, vec2_sub(offset, vec2_rotate(vec2(radius * -direction, 0), current_angle + direction * math.pi/2)))
                    points.append(center)
                current_angle += angle2
                current_pos = points[-1]
        
        return points
    
    def _choose_corner_type(self, rng: DeterministicRNG, complexity: float) -> str:
        """Choose corner type based on complexity and randomness."""
        r = rng.random()
        
        probs = {
            "straight": 0.3,
            "corner": 0.4,
            "hairpin": 0.1,
            "chicane": 0.1,
            "s_curve": 0.1
        }
        
        probs["straight"] *= (1.0 - complexity * 0.5)
        probs["corner"] *= (1.0 + complexity * 0.2)
        probs["hairpin"] *= (1.0 + complexity)
        probs["chicane"] *= (1.0 + complexity * 1.5)
        probs["s_curve"] *= (1.0 + complexity)
        
        if not self.config.enable_hairpins:
            probs["hairpin"] = 0
        if not self.config.enable_chicanes:
            probs["chicane"] = 0
        if not self.config.enable_s_curves:
            probs["s_curve"] = 0
        
        total = sum(probs.values())
        probs = {k: v/total for k, v in probs.items()}
        
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
            points[-1] = start
            return points
        
        num_segments = max(3, int(gap / 20.0))
        start_dir = vec2_normalize(vec2_sub(points[1], points[0]))
        end_dir = vec2_normalize(vec2_sub(points[-1], points[-2]))
        
        p0 = end
        p3 = start
        handle_len = gap * 0.4
        p1 = vec2_add(end, vec2_mul(end_dir, handle_len))
        p2 = vec2_add(start, vec2_mul(vec2_mul(start_dir, -1), handle_len))
        
        for i in range(1, num_segments):
            t = i / num_segments
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
        
        points[-1] = start
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
        
        target_spacing = 60.0
        accumulated = 0.0
        last_checkpoint_idx = 0
        
        for i in range(len(centerline) - 1):
            seg_len = vec2_distance(centerline[i], centerline[i + 1])
            accumulated += seg_len
            
            if accumulated >= target_spacing or i == len(centerline) - 2:
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
                accumulated = 0.0
                last_checkpoint_idx = i
        
        while len(checkpoints) < 4:
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
            
            if i < len(centerline) - 1:
                direction = vec2_normalize(vec2_sub(centerline[i + 1], centerline[i]))
            else:
                direction = vec2_normalize(vec2_sub(centerline[i], centerline[i - 1]))
            
            curvature = 0.0
            if i > 0 and i < len(centerline) - 1:
                prev_dir = vec2_normalize(vec2_sub(centerline[i], centerline[i - 1]))
                next_dir = vec2_normalize(vec2_sub(centerline[i + 1], centerline[i]))
                angle_change = angle_difference(vec2_angle(prev_dir), vec2_angle(next_dir))
                seg_len = (vec2_distance(centerline[i - 1], centerline[i]) + 
                          vec2_distance(centerline[i], centerline[i + 1])) * 0.5
                if seg_len > 0:
                    curvature = abs(angle_change) / seg_len
            
            offset = 0.0
            if curvature > 0.01:
                if i > 0:
                    prev_pos = centerline[i - 1]
                    to_prev = vec2_normalize(vec2_sub(prev_pos, pos))
                    cross = to_prev[0] * direction[1] - to_prev[1] * direction[0]
                    offset = -cross * half_width * 0.6 * min(1.0, curvature * 20.0)
            
            line_pos = vec2_add(pos, vec2_mul(vec2_perp(direction), offset))
            
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
        """Generate starting grid positions."""
        positions = []
        angles = []
        
        if start_checkpoint:
            perp = vec2_perp(start_checkpoint.direction)
            for i in range(5):
                offset = (i - 2) * 3.5
                pos = vec2_add(start_checkpoint.position, vec2_mul(perp, offset))
                pos = vec2_sub(pos, vec2_mul(start_checkpoint.direction, 5.0 + i * 3.0))
                positions.append(pos)
                angles.append(vec2_angle(start_checkpoint.direction))
        else:
            for i in range(5):
                positions.append(vec2(float(i * 3), 0.0))
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
        
        if len(track.centerline) < 4:
            errors.append("Centerline too short")
        
        start = track.centerline[0]
        end = track.centerline[-1]
        if vec2_distance(start, end) > 1.0:
            errors.append("Track does not form closed loop")
        
        if track.total_length < self.config.min_track_length * 0.5:
            errors.append(f"Track too short: {track.total_length:.1f}m")
        if track.total_length > self.config.max_track_length * 1.5:
            errors.append(f"Track too long: {track.total_length:.1f}m")
        
        if len(track.checkpoints) < self.config.min_checkpoints:
            errors.append(f"Too few checkpoints: {len(track.checkpoints)}")
        
        for i, cp in enumerate(track.checkpoints):
            next_cp = track.checkpoints[(i + 1) % len(track.checkpoints)]
            dist = vec2_distance(cp.position, next_cp.position)
            if dist > track.road_width * 10:
                errors.append(f"Checkpoint {i} to {next_cp.index} too far: {dist:.1f}m")
        
        if len(track.start_positions) < 1:
            errors.append("No valid spawn positions")
        
        track.validation_errors = errors
        return len(errors) == 0
