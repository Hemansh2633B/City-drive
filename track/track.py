"""
Track Representation

Data structures for tracks, checkpoints, racing lines, and track boundaries.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
from core.math import Vec2, vec2, vec2_add, vec2_sub, vec2_mul, vec2_normalize, vec2_perp, vec2_length, vec2_angle, vec2_rotate


@dataclass
class TrackSegment:
    """A segment of the track centerline."""
    start: Vec2
    end: Vec2
    width: float
    left_boundary: Tuple[Vec2, Vec2]  # (start, end)
    right_boundary: Tuple[Vec2, Vec2]  # (start, end)
    segment_type: str = "straight"  # "straight", "corner", "hairpin", "chicane", "s_curve"
    corner_radius: float = 0.0
    corner_direction: int = 0  # -1 for left, 1 for right


@dataclass
class Checkpoint:
    """Race checkpoint with position and direction."""
    position: Vec2
    direction: Vec2  # Normalized direction vector
    width: float
    index: int
    is_start_finish: bool = False
    
    def get_bounds(self) -> Tuple[Vec2, Vec2]:
        """Get the two endpoints of the checkpoint line."""
        half_width = self.width * 0.5
        perp = vec2_perp(self.direction)
        left = vec2_add(self.position, vec2_mul(perp, half_width))
        right = vec2_sub(self.position, vec2_mul(perp, half_width))
        return left, right
    
    def contains_point(self, point: Vec2) -> bool:
        """Check if point has crossed this checkpoint (in correct direction)."""
        left, right = self.get_bounds()
        # Check if point is on the correct side of the line
        to_point = vec2_sub(point, self.position)
        cross = to_point[0] * self.direction[1] - to_point[1] * self.direction[0]
        return cross > 0  # Crossed in forward direction


@dataclass
class RacingLinePoint:
    """A point on the optimal racing line."""
    position: Vec2
    direction: Vec2
    target_speed: float
    curvature: float  # 1/radius, 0 for straight


@dataclass
class Track:
    """
    Complete track representation.
    
    Contains centerline, boundaries, checkpoints, racing line, and metadata.
    """
    # Generation parameters
    seed: Union[int, str]
    theme: str
    complexity: float
    road_width: float
    length: float
    
    # Geometry
    centerline: List[Vec2] = field(default_factory=list)  # Ordered points forming loop
    segments: List[TrackSegment] = field(default_factory=list)
    checkpoints: List[Checkpoint] = field(default_factory=list)
    racing_line: List[RacingLinePoint] = field(default_factory=list)
    
    # Boundaries (for collision/rendering)
    left_boundary: List[Vec2] = field(default_factory=list)
    right_boundary: List[Vec2] = field(default_factory=list)
    
    # Spawn positions
    start_positions: List[Vec2] = field(default_factory=list)
    start_angles: List[float] = field(default_factory=list)
    
    # Properties
    num_laps: int = 3
    total_length: float = 0.0
    
    # Validation
    is_valid: bool = False
    validation_errors: List[str] = field(default_factory=list)
    
    # Theme data
    theme_data: Dict[str, Any] = field(default_factory=dict)
    
    def get_checkpoint(self, index: int) -> Checkpoint:
        """Get checkpoint by index (wraps around)."""
        return self.checkpoints[index % len(self.checkpoints)]
    
    def get_next_checkpoint(self, current_index: int) -> Checkpoint:
        """Get next checkpoint in sequence."""
        return self.checkpoints[(current_index + 1) % len(self.checkpoints)]
    
    def get_racing_line_point(self, progress: float) -> RacingLinePoint:
        """Get racing line point at progress (0 to 1)."""
        if not self.racing_line:
            return RacingLinePoint(vec2(0, 0), vec2(1, 0), 0.0, 0.0)
        
        idx = progress * (len(self.racing_line) - 1)
        i = int(idx)
        t = idx - i
        
        if i >= len(self.racing_line) - 1:
            return self.racing_line[-1]
        
        a = self.racing_line[i]
        b = self.racing_line[i + 1]
        
        return RacingLinePoint(
            position=vec2_add(a.position, vec2_mul(vec2_sub(b.position, a.position), t)),
            direction=vec2_normalize(vec2_add(a.direction, vec2_mul(vec2_sub(b.direction, a.direction), t))),
            target_speed=a.target_speed + (b.target_speed - a.target_speed) * t,
            curvature=a.curvature + (b.curvature - a.curvature) * t
        )
    
    def get_spawn_position(self, index: int) -> Tuple[Vec2, float]:
        """Get spawn position and angle for grid position."""
        if index < len(self.start_positions):
            return self.start_positions[index], self.start_angles[index]
        # Fallback to start/finish line
        if self.checkpoints:
            cp = self.checkpoints[0]
            offset = (index - len(self.start_positions)) * 3.0
            perp = vec2_perp(cp.direction)
            pos = vec2_add(cp.position, vec2_mul(perp, offset))
            return pos, vec2_angle(cp.direction)
        return vec2(0, 0), 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize track to dictionary."""
        return {
            'seed': self.seed,
            'theme': self.theme,
            'complexity': self.complexity,
            'road_width': self.road_width,
            'length': self.length,
            'centerline': [(float(p[0]), float(p[1])) for p in self.centerline],
            'segments': [
                {
                    'start': (float(s.start[0]), float(s.start[1])),
                    'end': (float(s.end[0]), float(s.end[1])),
                    'width': s.width,
                    'type': s.segment_type,
                    'corner_radius': s.corner_radius,
                    'corner_direction': s.corner_direction
                }
                for s in self.segments
            ],
            'checkpoints': [
                {
                    'position': (float(c.position[0]), float(c.position[1])),
                    'direction': (float(c.direction[0]), float(c.direction[1])),
                    'width': c.width,
                    'index': c.index,
                    'is_start_finish': c.is_start_finish
                }
                for c in self.checkpoints
            ],
            'racing_line': [
                {
                    'position': (float(r.position[0]), float(r.position[1])),
                    'direction': (float(r.direction[0]), float(r.direction[1])),
                    'target_speed': r.target_speed,
                    'curvature': r.curvature
                }
                for r in self.racing_line
            ],
            'left_boundary': [(float(p[0]), float(p[1])) for p in self.left_boundary],
            'right_boundary': [(float(p[0]), float(p[1])) for p in self.right_boundary],
            'start_positions': [(float(p[0]), float(p[1])) for p in self.start_positions],
            'start_angles': [float(a) for a in self.start_angles],
            'num_laps': self.num_laps,
            'total_length': self.total_length,
            'is_valid': self.is_valid,
            'validation_errors': self.validation_errors,
            'theme_data': self.theme_data
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Track':
        """Deserialize track from dictionary."""
        track = cls(
            seed=data['seed'],
            theme=data['theme'],
            complexity=data['complexity'],
            road_width=data['road_width'],
            length=data['length']
        )
        
        track.centerline = [vec2(p[0], p[1]) for p in data['centerline']]
        track.segments = [
            TrackSegment(
                start=vec2(s['start'][0], s['start'][1]),
                end=vec2(s['end'][0], s['end'][1]),
                width=s['width'],
                left_boundary=(vec2(0, 0), vec2(0, 0)),  # Will be rebuilt
                right_boundary=(vec2(0, 0), vec2(0, 0)),
                segment_type=s['type'],
                corner_radius=s['corner_radius'],
                corner_direction=s['corner_direction']
            )
            for s in data['segments']
        ]
        track.checkpoints = [
            Checkpoint(
                position=vec2(c['position'][0], c['position'][1]),
                direction=vec2(c['direction'][0], c['direction'][1]),
                width=c['width'],
                index=c['index'],
                is_start_finish=c['is_start_finish']
            )
            for c in data['checkpoints']
        ]
        track.racing_line = [
            RacingLinePoint(
                position=vec2(r['position'][0], r['position'][1]),
                direction=vec2(r['direction'][0], r['direction'][1]),
                target_speed=r['target_speed'],
                curvature=r['curvature']
            )
            for r in data['racing_line']
        ]
        track.left_boundary = [vec2(p[0], p[1]) for p in data['left_boundary']]
        track.right_boundary = [vec2(p[0], p[1]) for p in data['right_boundary']]
        track.start_positions = [vec2(p[0], p[1]) for p in data['start_positions']]
        track.start_angles = data['start_angles']
        track.num_laps = data['num_laps']
        track.total_length = data['total_length']
        track.is_valid = data['is_valid']
        track.validation_errors = data['validation_errors']
        track.theme_data = data.get('theme_data', {})
        
        return track


from typing import Union