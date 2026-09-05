"""
Core Math Utilities

2D vector math, geometry, and physics helpers for deterministic simulation.
All operations use float64 for precision.
"""

import numpy as np
from typing import Tuple, Optional, Union

# Try to import numba for JIT compilation
try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def jit(*args, **kwargs):
        """No-op decorator when numba not available."""
        def decorator(func):
            return func
        if args and callable(args[0]):
            return args[0]
        return decorator


# Type aliases
Vec2 = np.ndarray  # shape (2,), dtype float64
Vec2Array = np.ndarray  # shape (N, 2), dtype float64


@jit(nopython=True, cache=True)
def vec2(x: float, y: float) -> Vec2:
    """Create a 2D vector."""
    return np.array([x, y], dtype=np.float64)


@jit(nopython=True, cache=True)
def vec2_zero() -> Vec2:
    """Create zero vector."""
    return np.array([0.0, 0.0], dtype=np.float64)


@jit(nopython=True, cache=True)
def vec2_add(a: Vec2, b: Vec2) -> Vec2:
    """Add two vectors."""
    return np.array([a[0] + b[0], a[1] + b[1]], dtype=np.float64)


@jit(nopython=True, cache=True)
def vec2_sub(a: Vec2, b: Vec2) -> Vec2:
    """Subtract b from a."""
    return np.array([a[0] - b[0], a[1] - b[1]], dtype=np.float64)


@jit(nopython=True, cache=True)
def vec2_mul(a: Vec2, s: float) -> Vec2:
    """Multiply vector by scalar."""
    return np.array([a[0] * s, a[1] * s], dtype=np.float64)


@jit(nopython=True, cache=True)
def vec2_div(a: Vec2, s: float) -> Vec2:
    """Divide vector by scalar."""
    return np.array([a[0] / s, a[1] / s], dtype=np.float64)


@jit(nopython=True, cache=True)
def vec2_dot(a: Vec2, b: Vec2) -> float:
    """Dot product."""
    return a[0] * b[0] + a[1] * b[1]


@jit(nopython=True, cache=True)
def vec2_cross(a: Vec2, b: Vec2) -> float:
    """2D cross product (returns scalar)."""
    return a[0] * b[1] - a[1] * b[0]


@jit(nopython=True, cache=True)
def vec2_length_sq(v: Vec2) -> float:
    """Squared length."""
    return v[0] * v[0] + v[1] * v[1]


@jit(nopython=True, cache=True)
def vec2_length(v: Vec2) -> float:
    """Length."""
    return np.sqrt(v[0] * v[0] + v[1] * v[1])


@jit(nopython=True, cache=True)
def vec2_normalize(v: Vec2) -> Vec2:
    """Normalize vector (returns zero if length is 0)."""
    l = vec2_length(v)
    if l > 0:
        return np.array([v[0] / l, v[1] / l], dtype=np.float64)
    return vec2_zero()


@jit(nopython=True, cache=True)
def vec2_lerp(a: Vec2, b: Vec2, t: float) -> Vec2:
    """Linear interpolation."""
    return np.array([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t], dtype=np.float64)


@jit(nopython=True, cache=True)
def vec2_rotate(v: Vec2, angle: float) -> Vec2:
    """Rotate vector by angle (radians)."""
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([v[0] * c - v[1] * s, v[0] * s + v[1] * c], dtype=np.float64)


@jit(nopython=True, cache=True)
def vec2_angle(v: Vec2) -> float:
    """Angle of vector in radians."""
    return np.arctan2(v[1], v[0])


@jit(nopython=True, cache=True)
def vec2_distance(a: Vec2, b: Vec2) -> float:
    """Distance between two points."""
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return np.sqrt(dx * dx + dy * dy)


@jit(nopython=True, cache=True)
def vec2_distance_sq(a: Vec2, b: Vec2) -> float:
    """Squared distance between two points."""
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy


@jit(nopython=True, cache=True)
def vec2_perp(v: Vec2) -> Vec2:
    """Perpendicular vector (rotated 90 degrees CCW)."""
    return np.array([-v[1], v[0]], dtype=np.float64)


@jit(nopython=True, cache=True)
def vec2_clamp_length(v: Vec2, max_length: float) -> Vec2:
    """Clamp vector length to maximum."""
    l = vec2_length(v)
    if l > max_length and l > 0:
        return np.array([v[0] * max_length / l, v[1] * max_length / l], dtype=np.float64)
    return v


@jit(nopython=True, cache=True)
def line_intersection(p1: Vec2, p2: Vec2, p3: Vec2, p4: Vec2) -> Tuple[bool, Vec2]:
    """
    Check if two line segments intersect.
    Returns (intersects, intersection_point).
    """
    x1, y1 = p1[0], p1[1]
    x2, y2 = p2[0], p2[1]
    x3, y3 = p3[0], p3[1]
    x4, y4 = p4[0], p4[1]
    
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    
    if abs(denom) < 1e-10:
        return False, vec2_zero()
    
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
    
    if 0 <= t <= 1 and 0 <= u <= 1:
        return True, np.array([x1 + t * (x2 - x1), y1 + t * (y2 - y1)], dtype=np.float64)
    
    return False, vec2_zero()


@jit(nopython=True, cache=True)
def point_line_distance(point: Vec2, line_start: Vec2, line_end: Vec2) -> float:
    """Distance from point to line segment."""
    l2 = vec2_distance_sq(line_start, line_end)
    if l2 == 0:
        return vec2_distance(point, line_start)
    
    t = max(0, min(1, vec2_dot(vec2_sub(point, line_start), vec2_sub(line_end, line_start)) / l2))
    projection = vec2_add(line_start, vec2_mul(vec2_sub(line_end, line_start), t))
    return vec2_distance(point, projection)


@jit(nopython=True, cache=True)
def closest_point_on_segment(point: Vec2, line_start: Vec2, line_end: Vec2) -> Vec2:
    """Closest point on line segment to point."""
    l2 = vec2_distance_sq(line_start, line_end)
    if l2 == 0:
        return line_start
    
    t = max(0, min(1, vec2_dot(vec2_sub(point, line_start), vec2_sub(line_end, line_start)) / l2))
    return vec2_add(line_start, vec2_mul(vec2_sub(line_end, line_start), t))


@jit(nopython=True, cache=True)
def circle_rect_intersection(cx: float, cy: float, r: float,
                              rx: float, ry: float, rw: float, rh: float) -> bool:
    """Check if circle intersects rectangle."""
    # Find closest point on rectangle to circle center
    closest_x = max(rx, min(cx, rx + rw))
    closest_y = max(ry, min(cy, ry + rh))
    
    dx = cx - closest_x
    dy = cy - closest_y
    return dx * dx + dy * dy <= r * r


@jit(nopython=True, cache=True)
def rect_rect_intersection(ax: float, ay: float, aw: float, ah: float,
                            bx: float, by: float, bw: float, bh: float) -> bool:
    """Check if two rectangles intersect."""
    return not (ax + aw < bx or bx + bw < ax or ay + ah < by or by + bh < ay)


@jit(nopython=True, cache=True)
def rotate_point(point: Vec2, center: Vec2, angle: float) -> Vec2:
    """Rotate point around center by angle."""
    s = np.sin(angle)
    c = np.cos(angle)
    px = point[0] - center[0]
    py = point[1] - center[1]
    return np.array([px * c - py * s + center[0], px * s + py * c + center[1]], dtype=np.float64)


@jit(nopython=True, cache=True)
def get_rect_corners(x: float, y: float, w: float, h: float, angle: float) -> Vec2Array:
    """Get corners of rotated rectangle."""
    hw = w * 0.5
    hh = h * 0.5
    corners = np.array([
        [-hw, -hh],
        [hw, -hh],
        [hw, hh],
        [-hw, hh]
    ], dtype=np.float64)
    
    c = np.cos(angle)
    s = np.sin(angle)
    rot = np.array([[c, -s], [s, c]], dtype=np.float64)
    rotated = corners @ rot.T
    rotated[:, 0] += x
    rotated[:, 1] += y
    return rotated


@jit(nopython=True, cache=True)
def sat_collision(poly1: Vec2Array, poly2: Vec2Array) -> Tuple[bool, Vec2, float]:
    """
    Separating Axis Theorem collision detection for convex polygons.
    Returns (colliding, normal, penetration_depth).
    Normal points from poly1 to poly2.
    """
    min_penetration = float('inf')
    collision_normal = vec2_zero()
    
    def project(poly: Vec2Array, axis: Vec2) -> Tuple[float, float]:
        dots = poly[:, 0] * axis[0] + poly[:, 1] * axis[1]
        return np.min(dots), np.max(dots)
    
    def get_axes(poly: Vec2Array):
        axes = []
        n = len(poly)
        for i in range(n):
            p1 = poly[i]
            p2 = poly[(i + 1) % n]
            edge = vec2_sub(p2, p1)
            axis = vec2_normalize(vec2_perp(edge))
            axes.append(axis)
        return axes
    
    # Check all axes from both polygons
    for axis in get_axes(poly1):
        min1, max1 = project(poly1, axis)
        min2, max2 = project(poly2, axis)
        
        if max1 < min2 or max2 < min1:
            return False, vec2_zero(), 0.0
        
        penetration = min(max1, max2) - max(min1, min2)
        if penetration < min_penetration:
            min_penetration = penetration
            # Determine direction
            center1 = np.mean(poly1, axis=0)
            center2 = np.mean(poly2, axis=0)
            dir_vec = vec2_sub(center2, center1)
            if vec2_dot(dir_vec, axis) < 0:
                collision_normal = np.array([-axis[0], -axis[1]], dtype=np.float64)
            else:
                collision_normal = axis
    
    for axis in get_axes(poly2):
        min1, max1 = project(poly1, axis)
        min2, max2 = project(poly2, axis)
        
        if max1 < min2 or max2 < min1:
            return False, vec2_zero(), 0.0
        
        penetration = min(max1, max2) - max(min1, min2)
        if penetration < min_penetration:
            min_penetration = penetration
            center1 = np.mean(poly1, axis=0)
            center2 = np.mean(poly2, axis=0)
            dir_vec = vec2_sub(center2, center1)
            if vec2_dot(dir_vec, axis) < 0:
                collision_normal = np.array([-axis[0], -axis[1]], dtype=np.float64)
            else:
                collision_normal = axis
    
    return True, collision_normal, min_penetration


# Non-jitted helpers for convenience
def vec2_from_tuple(t: Tuple[float, float]) -> Vec2:
    return np.array([float(t[0]), float(t[1])], dtype=np.float64)


def vec2_to_tuple(v: Vec2) -> Tuple[float, float]:
    return (float(v[0]), float(v[1]))


def angle_difference(a1: float, a2: float) -> float:
    """Signed difference between two angles (-pi to pi)."""
    diff = a2 - a1
    while diff > np.pi:
        diff -= 2 * np.pi
    while diff < -np.pi:
        diff += 2 * np.pi
    return diff


def angle_lerp(a1: float, a2: float, t: float) -> float:
    """Linear interpolation of angles (shortest path)."""
    diff = angle_difference(a1, a2)
    return a1 + diff * t


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value to range."""
    return max(min_val, min(max_val, value))


def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation."""
    return a + (b - a) * t


def smoothstep(edge0: float, edge1: float, x: float) -> float:
    """Smooth Hermite interpolation."""
    t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def smootherstep(edge0: float, edge1: float, x: float) -> float:
    """Smoother interpolation."""
    t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * t * (t * (t * 6 - 15) + 10)