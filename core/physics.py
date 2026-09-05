"""
Core Physics Engine

Deterministic 2D rigid body physics for vehicles.
Fixed timestep, deterministic collision resolution.
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass, field
from core.math import (
    Vec2, vec2, vec2_zero, vec2_add, vec2_sub, vec2_mul, vec2_dot,
    vec2_length, vec2_normalize, vec2_rotate, vec2_angle, vec2_perp,
    get_rect_corners, sat_collision, angle_difference, clamp,
    vec2_distance
)
from core.random import DeterministicRNG


@dataclass
class PhysicsMaterial:
    """Physical material properties."""
    friction: float = 1.0
    restitution: float = 0.1  # Bounciness
    density: float = 1.0


@dataclass
class CollisionInfo:
    """Information about a collision."""
    body_a: int
    body_b: int
    normal: Vec2
    penetration: float
    point: Vec2
    relative_velocity: Vec2


@dataclass
class RigidBody:
    """2D rigid body for physics simulation."""
    # Identity
    id: int = 0
    
    # Transform
    position: Vec2 = field(default_factory=vec2_zero)
    angle: float = 0.0
    
    # Velocity
    velocity: Vec2 = field(default_factory=vec2_zero)
    angular_velocity: float = 0.0
    
    # Forces
    force: Vec2 = field(default_factory=vec2_zero)
    torque: float = 0.0
    
    # Physical properties
    mass: float = 1.0
    inv_mass: float = 1.0
    inertia: float = 1.0
    inv_inertia: float = 1.0
    
    # Shape
    width: float = 1.0
    height: float = 1.0
    
    # Material
    material: PhysicsMaterial = field(default_factory=PhysicsMaterial)
    
    # State
    is_static: bool = False
    is_sensor: bool = False
    enabled: bool = True
    
    # Collision callbacks
    on_collision: Optional[callable] = None
    
    # Derived/cached
    _corners: Optional[np.ndarray] = None
    _corners_dirty: bool = True
    
    def __post_init__(self):
        if self.mass > 0:
            self.inv_mass = 1.0 / self.mass
            # Moment of inertia for rectangle
            self.inertia = self.mass * (self.width**2 + self.height**2) / 12.0
            self.inv_inertia = 1.0 / self.inertia
        else:
            self.inv_mass = 0.0
            self.inv_inertia = 0.0
            self.is_static = True
    
    def get_corners(self) -> np.ndarray:
        """Get world-space corners of the body."""
        if self._corners_dirty or self._corners is None:
            self._corners = get_rect_corners(
                self.position[0], self.position[1],
                self.width, self.height, self.angle
            )
            self._corners_dirty = False
        return self._corners
    
    def apply_force(self, force: Vec2, point: Optional[Vec2] = None) -> None:
        """Apply force at world point (or center of mass if None)."""
        self.force = vec2_add(self.force, force)
        if point is not None:
            r = vec2_sub(point, self.position)
            self.torque += r[0] * force[1] - r[1] * force[0]
    
    def apply_impulse(self, impulse: Vec2, point: Optional[Vec2] = None) -> None:
        """Apply instantaneous impulse."""
        if self.is_static:
            return
        self.velocity = vec2_add(self.velocity, vec2_mul(impulse, self.inv_mass))
        if point is not None:
            r = vec2_sub(point, self.position)
            self.angular_velocity += (r[0] * impulse[1] - r[1] * impulse[0]) * self.inv_inertia
    
    def clear_forces(self) -> None:
        """Clear accumulated forces."""
        self.force = vec2_zero()
        self.torque = 0.0
    
    def integrate(self, dt: float) -> None:
        """Integrate physics (semi-implicit Euler)."""
        if self.is_static:
            return
        
        # Linear
        acceleration = vec2_mul(self.force, self.inv_mass)
        self.velocity = vec2_add(self.velocity, vec2_mul(acceleration, dt))
        self.position = vec2_add(self.position, vec2_mul(self.velocity, dt))
        
        # Angular
        angular_accel = self.torque * self.inv_inertia
        self.angular_velocity += angular_accel * dt
        self.angle += self.angular_velocity * dt
        
        # Normalize angle
        while self.angle > np.pi:
            self.angle -= 2 * np.pi
        while self.angle < -np.pi:
            self.angle += 2 * np.pi
        
        self._corners_dirty = True
        self.clear_forces()


class PhysicsWorld:
    """
    Deterministic 2D physics world.
    
    Features:
    - Fixed timestep simulation
    - SAT-based collision detection
    - Impulse-based collision resolution
    - Broad phase with spatial hashing
    - Deterministic collision ordering
    """
    
    def __init__(self, config, rng: DeterministicRNG):
        self.config = config
        self.rng = rng
        self.bodies: List[RigidBody] = []
        self.static_bodies: List[RigidBody] = []
        self._next_body_id = 1
        self._spatial_hash: dict = {}
        self._cell_size = 50.0  # meters
        
        # Collision pairs for deterministic ordering
        self._collision_pairs: List[Tuple[int, int]] = []
    
    def add_body(self, body: RigidBody) -> int:
        """Add a body to the world."""
        body.id = self._next_body_id
        self._next_body_id += 1
        
        if body.is_static:
            self.static_bodies.append(body)
        else:
            self.bodies.append(body)
        
        return body.id
    
    def remove_body(self, body_id: int) -> bool:
        """Remove a body by ID."""
        for i, body in enumerate(self.bodies):
            if body.id == body_id:
                self.bodies.pop(i)
                return True
        for i, body in enumerate(self.static_bodies):
            if body.id == body_id:
                self.static_bodies.pop(i)
                return True
        return False
    
    def get_body(self, body_id: int) -> Optional[RigidBody]:
        """Get body by ID."""
        for body in self.bodies:
            if body.id == body_id:
                return body
        for body in self.static_bodies:
            if body.id == body_id:
                return body
        return None
    
    def _build_spatial_hash(self) -> None:
        """Build spatial hash for broad phase."""
        self._spatial_hash.clear()
        all_bodies = self.bodies + self.static_bodies
        
        for body in all_bodies:
            if not body.enabled:
                continue
            corners = body.get_corners()
            min_x = np.min(corners[:, 0])
            max_x = np.max(corners[:, 0])
            min_y = np.min(corners[:, 1])
            max_y = np.max(corners[:, 1])
            
            min_cell_x = int(min_x / self._cell_size)
            max_cell_x = int(max_x / self._cell_size)
            min_cell_y = int(min_y / self._cell_size)
            max_cell_y = int(max_y / self._cell_size)
            
            for cx in range(min_cell_x, max_cell_x + 1):
                for cy in range(min_cell_y, max_cell_y + 1):
                    key = (cx, cy)
                    if key not in self._spatial_hash:
                        self._spatial_hash[key] = []
                    self._spatial_hash[key].append(body.id)
    
    def _get_potential_pairs(self) -> List[Tuple[int, int]]:
        """Get potential collision pairs from spatial hash."""
        pairs = set()
        
        for cell_bodies in self._spatial_hash.values():
            for i in range(len(cell_bodies)):
                for j in range(i + 1, len(cell_bodies)):
                    a_id, b_id = cell_bodies[i], cell_bodies[j]
                    if a_id > b_id:
                        a_id, b_id = b_id, a_id
                    pairs.add((a_id, b_id))
        
        # Sort for deterministic ordering
        return sorted(list(pairs))
    
    def _check_collision(self, body_a: RigidBody, body_b: RigidBody) -> Optional[CollisionInfo]:
        """Check collision between two bodies using SAT."""
        if not body_a.enabled or not body_b.enabled:
            return None
        if body_a.is_static and body_b.is_static:
            return None
        if body_a.is_sensor or body_b.is_sensor:
            # Sensor collision - just detect overlap
            pass
        
        poly_a = body_a.get_corners()
        poly_b = body_b.get_corners()
        
        colliding, normal, penetration = sat_collision(poly_a, poly_b)
        
        if colliding:
            # Calculate collision point (average of overlapping vertices)
            # For simplicity, use center of penetration
            center_a = np.mean(poly_a, axis=0)
            center_b = np.mean(poly_b, axis=0)
            point = (center_a + center_b) * 0.5
            
            # Relative velocity
            rel_vel = vec2_sub(body_b.velocity, body_a.velocity)
            
            return CollisionInfo(
                body_a=body_a.id,
                body_b=body_b.id,
                normal=normal,
                penetration=penetration,
                point=point,
                relative_velocity=rel_vel
            )
        
        return None
    
    def _resolve_collision(self, info: CollisionInfo) -> None:
        """Resolve collision using impulse-based method."""
        body_a = self.get_body(info.body_a)
        body_b = self.get_body(info.body_b)
        
        if body_a is None or body_b is None:
            return
        
        # Combined material properties
        friction = min(body_a.material.friction, body_b.material.friction)
        restitution = max(body_a.material.restitution, body_b.material.restitution)
        
        # Relative velocity
        rv = info.relative_velocity
        vel_along_normal = vec2_dot(rv, info.normal)
        
        # Don't resolve if separating
        if vel_along_normal > 0:
            return
        
        # Calculate impulse scalar
        inv_mass_sum = body_a.inv_mass + body_b.inv_mass
        
        # Contact point relative to centers
        ra = vec2_sub(info.point, body_a.position)
        rb = vec2_sub(info.point, body_b.position)
        
        # Angular contributions
        ra_cross_n = ra[0] * info.normal[1] - ra[1] * info.normal[0]
        rb_cross_n = rb[0] * info.normal[1] - rb[1] * info.normal[0]
        
        inv_inertia_sum = (body_a.inv_inertia * ra_cross_n * ra_cross_n + 
                          body_b.inv_inertia * rb_cross_n * rb_cross_n)
        
        j = -(1 + restitution) * vel_along_normal
        j /= inv_mass_sum + inv_inertia_sum
        
        # Apply impulse
        impulse = vec2_mul(info.normal, j)
        
        if not body_a.is_static:
            body_a.velocity = vec2_sub(body_a.velocity, vec2_mul(impulse, body_a.inv_mass))
            body_a.angular_velocity -= ra_cross_n * j * body_a.inv_inertia
        
        if not body_b.is_static:
            body_b.velocity = vec2_add(body_b.velocity, vec2_mul(impulse, body_b.inv_mass))
            body_b.angular_velocity += rb_cross_n * j * body_b.inv_inertia
        
        # Friction impulse
        rv = vec2_sub(body_b.velocity, body_a.velocity)
        tangent = vec2_sub(rv, vec2_mul(info.normal, vec2_dot(rv, info.normal)))
        tangent_len = vec2_length(tangent)
        
        if tangent_len > 1e-6:
            tangent = vec2_mul(tangent, 1.0 / tangent_len)
            
            jt = -vec2_dot(rv, tangent)
            jt /= inv_mass_sum + inv_inertia_sum
            
            # Clamp friction
            if jt > j * friction:
                jt = j * friction
            elif jt < -j * friction:
                jt = -j * friction
            
            friction_impulse = vec2_mul(tangent, jt)
            
            if not body_a.is_static:
                body_a.velocity = vec2_sub(body_a.velocity, vec2_mul(friction_impulse, body_a.inv_mass))
                ra_cross_t = ra[0] * tangent[1] - ra[1] * tangent[0]
                body_a.angular_velocity -= ra_cross_t * jt * body_a.inv_inertia
            
            if not body_b.is_static:
                body_b.velocity = vec2_add(body_b.velocity, vec2_mul(friction_impulse, body_b.inv_mass))
                rb_cross_t = rb[0] * tangent[1] - rb[1] * tangent[0]
                body_b.angular_velocity += rb_cross_t * jt * body_b.inv_inertia
        
        # Position correction (penetration resolution)
        if info.penetration > 0.001:
            correction = vec2_mul(info.normal, info.penetration * 0.5)
            if not body_a.is_static:
                body_a.position = vec2_sub(body_a.position, vec2_mul(correction, body_a.inv_mass / inv_mass_sum))
            if not body_b.is_static:
                body_b.position = vec2_add(body_b.position, vec2_mul(correction, body_b.inv_mass / inv_mass_sum))
    
    def step(self, dt: float) -> List[CollisionInfo]:
        """Step the physics world by dt seconds."""
        sub_dt = dt / self.config.game.physics_substeps
        all_collisions = []
        
        for _ in range(self.config.game.physics_substeps):
            # Integrate
            for body in self.bodies:
                body.integrate(sub_dt)
            
            # Broad phase
            self._build_spatial_hash()
            pairs = self._get_potential_pairs()
            
            # Narrow phase
            collisions = []
            for a_id, b_id in pairs:
                body_a = self.get_body(a_id)
                body_b = self.get_body(b_id)
                
                if body_a is None or body_b is None:
                    continue
                
                info = self._check_collision(body_a, body_b)
                if info:
                    collisions.append(info)
            
            # Sort collisions for deterministic resolution
            collisions.sort(key=lambda c: (c.body_a, c.body_b))
            
            # Resolve collisions
            for info in collisions:
                self._resolve_collision(info)
            
            all_collisions.extend(collisions)
        
        return all_collisions
    
    def raycast(self, origin: Vec2, direction: Vec2, max_distance: float,
                ignore_body: Optional[int] = None) -> Tuple[bool, float, Vec2, int]:
        """
        Cast a ray and return first hit.
        Returns (hit, distance, hit_point, body_id).
        """
        # Normalize direction
        dir_len = vec2_length(direction)
        if dir_len == 0:
            return False, max_distance, vec2_zero(), -1
        direction = vec2_mul(direction, 1.0 / dir_len)
        
        end_point = vec2_add(origin, vec2_mul(direction, max_distance))
        closest_dist = max_distance
        hit_point = vec2_zero()
        hit_body_id = -1
        hit = False
        
        all_bodies = self.bodies + self.static_bodies
        
        for body in all_bodies:
            if not body.enabled or body.id == ignore_body:
                continue
            
            corners = body.get_corners()
            n = len(corners)
            
            for i in range(n):
                p1 = corners[i]
                p2 = corners[(i + 1) % n]
                
                # Line segment intersection
                intersects, point = line_intersection(origin, end_point, p1, p2)
                if intersects:
                    dist = vec2_distance(origin, point)
                    if dist < closest_dist:
                        closest_dist = dist
                        hit_point = point
                        hit_body_id = body.id
                        hit = True
        
        return hit, closest_dist, hit_point, hit_body_id


# Import needed for collision detection
from core.math import line_intersection