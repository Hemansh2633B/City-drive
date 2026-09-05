"""
Core Package

Core systems for the racing simulator:
- random: Deterministic random number generation
- config: Configuration management
- math: 2D vector math and geometry
- physics: Deterministic physics engine
"""

from core.random import DeterministicRNG, SeededRandomStreams, create_rng, seed_from_string
from core.config import Config, load_config, init_config, get_config, SimulationContext, create_simulation_context
from core.math import (
    Vec2, vec2, vec2_zero, vec2_add, vec2_sub, vec2_mul, vec2_div,
    vec2_dot, vec2_cross, vec2_length, vec2_length_sq, vec2_normalize,
    vec2_lerp, vec2_rotate, vec2_angle, vec2_distance, vec2_distance_sq,
    vec2_perp, vec2_clamp_length, line_intersection, point_line_distance,
    closest_point_on_segment, circle_rect_intersection, rect_rect_intersection,
    rotate_point, get_rect_corners, sat_collision, angle_difference,
    angle_lerp, clamp, lerp, smoothstep, smootherstep
)
from core.physics import PhysicsMaterial, RigidBody, CollisionInfo, PhysicsWorld

__all__ = [
    # Random
    'DeterministicRNG', 'SeededRandomStreams', 'create_rng', 'seed_from_string',
    # Config
    'Config', 'load_config', 'init_config', 'get_config', 'SimulationContext', 'create_simulation_context',
    # Math
    'Vec2', 'vec2', 'vec2_zero', 'vec2_add', 'vec2_sub', 'vec2_mul', 'vec2_div',
    'vec2_dot', 'vec2_cross', 'vec2_length', 'vec2_length_sq', 'vec2_normalize',
    'vec2_lerp', 'vec2_rotate', 'vec2_angle', 'vec2_distance', 'vec2_distance_sq',
    'vec2_perp', 'vec2_clamp_length', 'line_intersection', 'point_line_distance',
    'closest_point_on_segment', 'circle_rect_intersection', 'rect_rect_intersection',
    'rotate_point', 'get_rect_corners', 'sat_collision', 'angle_difference',
    'angle_lerp', 'clamp', 'lerp', 'smoothstep', 'smootherstep',
    # Physics
    'PhysicsMaterial', 'RigidBody', 'CollisionInfo', 'PhysicsWorld'
]