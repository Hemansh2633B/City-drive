"""
Pygame Renderer

Handles all rendering: track, vehicles, HUD, debug overlay, menus.
"""

import pygame
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from core.math import Vec2, vec2, vec2_add, vec2_sub, vec2_mul, vec2_normalize, vec2_rotate, vec2_angle
from track.track import Track, TrackSegment, Checkpoint
from vehicles.vehicle import Vehicle, VehicleState
from telemetry.telemetry import ReplayPlayer


class RenderLayer(Enum):
    BACKGROUND = 0
    TRACK = 1
    TRACK_DETAILS = 2
    VEHICLES = 3
    EFFECTS = 4
    HUD = 5
    DEBUG = 6
    MENU = 7


@dataclass
class Camera:
    """Camera for following vehicles."""
    position: Vec2 = field(default_factory=lambda: vec2(0, 0))
    target_position: Vec2 = field(default_factory=lambda: vec2(0, 0))
    zoom: float = 1.0
    target_zoom: float = 1.0
    smoothing: float = 0.1
    min_zoom: float = 0.3
    max_zoom: float = 3.0
    screen_width: int = 1280
    screen_height: int = 720
    
    def update(self, dt: float) -> None:
        """Update camera position and zoom."""
        # Smooth follow
        self.position = vec2_add(self.position, 
            vec2_mul(vec2_sub(self.target_position, self.position), self.smoothing))
        
        # Smooth zoom
        self.zoom += (self.target_zoom - self.zoom) * self.smoothing
        self.zoom = max(self.min_zoom, min(self.max_zoom, self.zoom))
    
    def world_to_screen(self, world_pos: Vec2) -> Tuple[int, int]:
        """Convert world coordinates to screen coordinates."""
        rel_x = (world_pos[0] - self.position[0]) * self.zoom
        rel_y = (world_pos[1] - self.position[1]) * self.zoom
        
        screen_x = int(self.screen_width / 2 + rel_x)
        screen_y = int(self.screen_height / 2 + rel_y)
        
        return screen_x, screen_y
    
    def screen_to_world(self, screen_pos: Tuple[int, int]) -> Vec2:
        """Convert screen coordinates to world coordinates."""
        rel_x = (screen_pos[0] - self.screen_width / 2) / self.zoom
        rel_y = (screen_pos[1] - self.screen_height / 2) / self.zoom
        
        return vec2(self.position[0] + rel_x, self.position[1] + rel_y)
    
    def set_target(self, position: Vec2, zoom: Optional[float] = None) -> None:
        """Set camera target."""
        self.target_position = position
        if zoom is not None:
            self.target_zoom = zoom


class Renderer:
    """
    Main renderer for the racing game.
    
    Features:
    - Track rendering with boundaries, checkpoints, racing line
    - Vehicle rendering with rotation, effects
    - HUD with position, lap times, speed, minimap
    - Debug overlay with sensors, physics info
    - Menu system
    - Replay playback
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.render_config = config.get('render', {})
        
        # Screen
        self.width = self.render_config.get('width', 1280)
        self.height = self.render_config.get('height', 720)
        self.screen = None
        self.clock = None
        
        # Camera
        self.camera = Camera(
            screen_width=self.width,
            screen_height=self.height,
            zoom=self.render_config.get('camera_zoom', 1.0),
            smoothing=self.render_config.get('camera_smoothing', 0.1),
            min_zoom=self.render_config.get('min_zoom', 0.3),
            max_zoom=self.render_config.get('max_zoom', 3.0)
        )
        
        # Colors
        colors = self.render_config.get('colors', {})
        self.bg_color = colors.get('background', [30, 30, 40])
        self.road_color = colors.get('road', [50, 50, 50])
        self.grass_color = colors.get('grass', [34, 139, 34])
        self.line_color = colors.get('line', [255, 255, 255])
        self.checkpoint_color = colors.get('checkpoint', [0, 255, 0])
        self.start_finish_color = colors.get('start_finish', [255, 255, 0])
        self.car_player_color = colors.get('car_player', [0, 100, 255])
        self.car_ai_color = colors.get('car_ai', [255, 100, 0])
        self.car_npc_color = colors.get('car_npc', [200, 50, 50])
        self.sensor_color = colors.get('sensor_ray', [0, 255, 255, 128])
        self.racing_line_color = colors.get('racing_line', [255, 255, 0, 128])
        
        # Visual options
        self.show_minimap = self.render_config.get('show_minimap', True)
        self.minimap_size = self.render_config.get('minimap_size', 200)
        self.minimap_position = self.render_config.get('minimap_position', 'top-right')
        self.show_racing_line = self.render_config.get('show_racing_line', False)
        self.show_checkpoints = self.render_config.get('show_checkpoints', True)
        self.show_sensors = self.render_config.get('show_sensors', False)
        self.show_hitboxes = self.render_config.get('show_hitboxes', False)
        self.show_debug_overlay = self.render_config.get('show_debug_overlay', False)
        
        # Fonts
        self.font_large = None
        self.font_medium = None
        self.font_small = None
        self.font_mono = None
        
        # Assets
        self.car_sprite = None
        self._create_car_sprite()
        
        # State
        self.track: Optional[Track] = None
        self.vehicles: List[Vehicle] = []
        self.race_state: Dict[str, Any] = {}
        self.replay_player: Optional[ReplayPlayer] = None
        
        # Debug
        self.debug_info: Dict[str, Any] = {}
    
    def initialize(self) -> None:
        """Initialize pygame and create window."""
        pygame.init()
        pygame.display.set_caption("AI Racing Simulator")
        
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()
        
        # Fonts
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 20)
        self.font_mono = pygame.font.SysFont('monospace', 14)
    
    def _create_car_sprite(self) -> None:
        """Create a simple car sprite surface."""
        # Create a simple car shape
        w, h = 40, 20
        self.car_sprite = pygame.Surface((w, h), pygame.SRCALPHA)
        
        # Car body
        pygame.draw.rect(self.car_sprite, (100, 100, 100), (2, 2, w-4, h-4), border_radius=3)
        # Front indicator
        pygame.draw.polygon(self.car_sprite, (255, 255, 255), [
            (w-2, h//2 - 4), (w-2, h//2 + 4), (w+2, h//2)
        ])
        # Windows
        pygame.draw.rect(self.car_sprite, (50, 150, 255), (6, 5, w-14, h-10), border_radius=2)
    
    def set_track(self, track: Track) -> None:
        """Set current track."""
        self.track = track
    
    def set_vehicles(self, vehicles: List[Vehicle]) -> None:
        """Set current vehicles."""
        self.vehicles = vehicles
    
    def set_race_state(self, state: Dict[str, Any]) -> None:
        """Set race state for HUD."""
        self.race_state = state
    
    def set_replay_player(self, player: ReplayPlayer) -> None:
        """Set replay player for playback."""
        self.replay_player = player
    
    def update_camera(self, dt: float, target_vehicle: Optional[Vehicle] = None) -> None:
        """Update camera to follow target vehicle."""
        if target_vehicle:
            self.camera.set_target(target_vehicle.state.position)
        
        self.camera.update(dt)
    
    def render(self, dt: float) -> None:
        """Main render function."""
        if not self.screen:
            return
        
        # Clear screen
        self.screen.fill(self.bg_color)
        
        if self.track:
            # Render track
            self._render_track()
            
            # Render racing line
            if self.show_racing_line:
                self._render_racing_line()
            
            # Render checkpoints
            if self.show_checkpoints:
                self._render_checkpoints()
        
        # Render vehicles
        for vehicle in self.vehicles:
            self._render_vehicle(vehicle)
        
        # Render sensors
        if self.show_sensors and self.vehicles:
            self._render_sensors(self.vehicles[0])
        
        # Render hitboxes
        if self.show_hitboxes:
            self._render_hitboxes()
        
        # Render HUD
        self._render_hud()
        
        # Render minimap
        if self.show_minimap and self.track:
            self._render_minimap()
        
        # Render debug overlay
        if self.show_debug_overlay:
            self._render_debug_overlay()
        
        # Render countdown
        self._render_countdown()
        
        pygame.display.flip()
    
    def _render_track(self) -> None:
        """Render track boundaries and surface."""
        if not self.track:
            return
        
        # Render grass background (larger area)
        # Draw a large rect covering the track bounds
        all_points = self.track.left_boundary + self.track.right_boundary[::-1]
        if len(all_points) > 2:
            screen_points = [self.camera.world_to_screen(p) for p in all_points]
            
            # Expand for grass
            grass_points = []
            for i, p in enumerate(all_points):
                # Calculate outward normal
                prev_p = all_points[i - 1]
                next_p = all_points[(i + 1) % len(all_points)]
                edge = vec2_sub(next_p, prev_p)
                length = np.sqrt(edge[0]**2 + edge[1]**2)
                if length > 0:
                    normal = vec2(-edge[1]/length, edge[0]/length)
                    # Determine if this is left or right boundary
                    if i < len(self.track.left_boundary):
                        grass_p = vec2_add(p, vec2_mul(normal, 100))
                    else:
                        grass_p = vec2_sub(p, vec2_mul(normal, 100))
                    grass_points.append(self.camera.world_to_screen(grass_p))
            
            # Draw grass (simplified - just draw a large rect)
            bounds = self._get_track_bounds()
            if bounds:
                min_x, min_y, max_x, max_y = bounds
                grass_rect = pygame.Rect(
                    self.camera.world_to_screen(vec2(min_x - 50, min_y - 50))[0],
                    self.camera.world_to_screen(vec2(min_x - 50, min_y - 50))[1],
                    (max_x - min_x + 100) * self.camera.zoom,
                    (max_y - min_y + 100) * self.camera.zoom
                )
                pygame.draw.rect(self.screen, self.grass_color, grass_rect)
        
        # Render road surface (polygon between boundaries)
        if len(self.track.left_boundary) > 1 and len(self.track.right_boundary) > 1:
            road_points = []
            for p in self.track.left_boundary:
                road_points.append(self.camera.world_to_screen(p))
            for p in reversed(self.track.right_boundary):
                road_points.append(self.camera.world_to_screen(p))
            
            if len(road_points) > 2:
                pygame.draw.polygon(self.screen, self.road_color, road_points)
        
        # Render track edges (lines)
        self._render_polyline(self.track.left_boundary, self.line_color, 2)
        self._render_polyline(self.track.right_boundary, self.line_color, 2)
        
        # Render centerline (dashed)
        self._render_dashed_polyline(self.track.centerline, (150, 150, 150), 1, 10, 10)
    
    def _render_polyline(self, points: List[Vec2], color: Tuple[int, int, int], width: int) -> None:
        """Render a polyline."""
        if len(points) < 2:
            return
        
        screen_points = [self.camera.world_to_screen(p) for p in points]
        pygame.draw.lines(self.screen, color, False, screen_points, width)
    
    def _render_dashed_polyline(self, points: List[Vec2], color: Tuple[int, int, int], 
                                 width: int, dash_len: int, gap_len: int) -> None:
        """Render a dashed polyline."""
        if len(points) < 2:
            return
        
        for i in range(len(points) - 1):
            p1 = self.camera.world_to_screen(points[i])
            p2 = self.camera.world_to_screen(points[i + 1])
            
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            dist = np.sqrt(dx*dx + dy*dy)
            
            if dist == 0:
                continue
            
            dash_count = int(dist / (dash_len + gap_len))
            ux, uy = dx / dist, dy / dist
            
            for d in range(dash_count):
                start_x = p1[0] + ux * d * (dash_len + gap_len)
                start_y = p1[1] + uy * d * (dash_len + gap_len)
                end_x = start_x + ux * dash_len
                end_y = start_y + uy * dash_len
                
                pygame.draw.line(self.screen, color, 
                               (int(start_x), int(start_y)), 
                               (int(end_x), int(end_y)), width)
    
    def _render_racing_line(self) -> None:
        """Render optimal racing line."""
        if not self.track or not self.track.racing_line:
            return
        
        points = [rl.position for rl in self.track.racing_line]
        screen_points = [self.camera.world_to_screen(p) for p in points]
        
        if len(screen_points) > 1:
            # Draw as semi-transparent yellow line
            color = (*self.racing_line_color[:3], 128) if len(self.racing_line_color) == 4 else self.racing_line_color
            # Create surface for alpha
            line_surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            pygame.draw.lines(line_surf, color, False, screen_points, 2)
            self.screen.blit(line_surf, (0, 0))
    
    def _render_checkpoints(self) -> None:
        """Render checkpoints."""
        if not self.track or not self.track.checkpoints:
            return
        
        for i, cp in enumerate(self.track.checkpoints):
            left, right = cp.get_bounds()
            left_screen = self.camera.world_to_screen(left)
            right_screen = self.camera.world_to_screen(right)
            
            # Start/finish line is special
            if cp.is_start_finish:
                color = self.start_finish_color
                width = 4
            else:
                color = self.checkpoint_color
                width = 2
            
            pygame.draw.line(self.screen, color, left_screen, right_screen, width)
            
            # Checkpoint number
            mid = vec2((left[0] + right[0]) / 2, (left[1] + right[1]) / 2)
            mid_screen = self.camera.world_to_screen(mid)
            text = self.font_small.render(str(i), True, (255, 255, 255))
            self.screen.blit(text, (mid_screen[0] - 5, mid_screen[1] - 10))
    
    def _render_vehicle(self, vehicle: Vehicle) -> None:
        """Render a vehicle."""
        state = vehicle.state
        pos = self.camera.world_to_screen(state.position)
        angle_deg = -np.degrees(state.angle)  # Pygame Y is down
        
        # Determine color
        if hasattr(state, 'name') and state.name == "Player":
            color = self.car_player_color
        elif hasattr(state, 'name') and "Opponent" in state.name:
            color = self.car_npc_color
        else:
            color = self.car_ai_color
        
        # Rotate and draw car sprite
        if self.car_sprite:
            # Tint the sprite
            tinted = self.car_sprite.copy()
            tinted.fill(color[:3] + (255,), special_flags=pygame.BLEND_RGBA_MULT)
            
            rotated = pygame.transform.rotate(tinted, angle_deg)
            rect = rotated.get_rect(center=pos)
            self.screen.blit(rotated, rect)
        else:
            # Fallback: draw rectangle
            w, h = int(vehicle.config.width * self.camera.zoom), int(vehicle.config.length * self.camera.zoom)
            surf = pygame.Surface((w, h), pygame.SRCALPHA)
            pygame.draw.rect(surf, color, (0, 0, w, h), border_radius=3)
            # Front indicator
            pygame.draw.polygon(surf, (255, 255, 255), [
                (w-2, h//2 - 4), (w-2, h//2 + 4), (w+2, h//2)
            ])
            rotated = pygame.transform.rotate(surf, angle_deg)
            rect = rotated.get_rect(center=pos)
            self.screen.blit(rotated, rect)
        
        # Drift effect
        if state.is_drifting:
            # Draw tire smoke particles
            rear_offset = vec2_rotate(vec2(-vehicle.config.length * 0.4, 0), state.angle)
            rear_pos = vec2_add(state.position, rear_offset)
            rear_screen = self.camera.world_to_screen(rear_pos)
            
            alpha = int(255 * state.drift_intensity)
            for _ in range(3):
                offset_x = np.random.uniform(-10, 10)
                offset_y = np.random.uniform(-5, 5)
                pygame.draw.circle(self.screen, (200, 200, 200, alpha),
                                 (int(rear_screen[0] + offset_x), int(rear_screen[1] + offset_y)),
                                 int(3 * self.camera.zoom))
        
        # Boost effect
        if state.boost_active:
            rear_offset = vec2_rotate(vec2(-vehicle.config.length * 0.5, 0), state.angle)
            rear_pos = vec2_add(state.position, rear_offset)
            rear_screen = self.camera.world_to_screen(rear_pos)
            
            for i in range(5):
                offset = np.random.uniform(-5, 5)
                pygame.draw.line(self.screen, (255, 100, 0),
                               rear_screen,
                               (rear_screen[0] - int(15 * self.camera.zoom), 
                                rear_screen[1] + int(offset * self.camera.zoom)), 2)
    
    def _render_sensors(self, vehicle: Vehicle) -> None:
        """Render raycast sensors."""
        state = vehicle.state
        sensor_cfg = self.config.get('training', {}).get('sensor_config', {})
        num_rays = sensor_cfg.get('num_rays', 16)
        max_dist = sensor_cfg.get('max_distance', 50.0)
        angles = sensor_cfg.get('angles', [])
        
        if len(angles) != num_rays:
            angles = np.linspace(-np.pi, np.pi, num_rays, endpoint=False).tolist()
        
        # We need physics world for raycast - simplified here
        # In practice, this would use the physics world
        pass
    
    def _render_hitboxes(self) -> None:
        """Render vehicle hitboxes."""
        for vehicle in self.vehicles:
            from core.math import get_rect_corners
            corners = get_rect_corners(
                vehicle.state.position[0], vehicle.state.position[1],
                vehicle.config.width, vehicle.config.length, vehicle.state.angle
            )
            screen_corners = [self.camera.world_to_screen(c) for c in corners]
            pygame.draw.lines(self.screen, (255, 0, 255), True, screen_corners, 1)
    
    def _render_hud(self) -> None:
        """Render heads-up display."""
        if not self.race_state or 'participants' not in self.race_state:
            return
        
        # Find player
        player = None
        for p in self.race_state['participants']:
            if p.get('is_player', False):
                player = p
                break
        
        if not player and self.race_state['participants']:
            player = self.race_state['participants'][0]
        
        if not player:
            return
        
        # Position
        pos_text = f"POS: {player['position']}/{len(self.race_state['participants'])}"
        pos_surf = self.font_large.render(pos_text, True, (255, 255, 255))
        self.screen.blit(pos_surf, (20, 20))
        
        # Lap
        lap_text = f"LAP: {player['lap']}/{player['total_laps']}"
        lap_surf = self.font_medium.render(lap_text, True, (255, 255, 255))
        self.screen.blit(lap_surf, (20, 70))
        
        # Speed
        speed_kmh = player['speed'] * 3.6
        speed_text = f"{speed_kmh:.0f} km/h"
        speed_surf = self.font_large.render(speed_text, True, (255, 255, 100))
        self.screen.blit(speed_surf, (20, 110))
        
        # Lap times
        if player['lap_times']:
            best = min(player['lap_times'])
            current = player['current_lap_time']
            best_text = f"BEST: {best:.2f}s"
            curr_text = f"CURR: {current:.2f}s"
            best_surf = self.font_small.render(best_text, True, (100, 255, 100))
            curr_surf = self.font_small.render(curr_text, True, (255, 255, 255))
            self.screen.blit(best_surf, (20, 160))
            self.screen.blit(curr_surf, (20, 180))
        
        # Race time
        time_text = f"TIME: {player['race_time']:.1f}s"
        time_surf = self.font_medium.render(time_text, True, (255, 255, 255))
        self.screen.blit(time_surf, (20, 210))
        
        # Seed
        if 'seed' in self.race_state:
            seed_text = f"SEED: {self.race_state['seed']}"
            seed_surf = self.font_small.render(seed_text, True, (150, 150, 150))
            self.screen.blit(seed_surf, (20, self.height - 30))
        
        # Checkpoint indicator
        if self.track and self.track.checkpoints:
            next_cp_idx = player['checkpoint'] % len(self.track.checkpoints)
            next_cp = self.track.checkpoints[next_cp_idx]
            
            # Direction to next checkpoint
            if self.vehicles:
                vehicle = self.vehicles[0]
                to_cp = vec2_sub(next_cp.position, vehicle.state.position)
                dist = np.sqrt(to_cp[0]**2 + to_cp[1]**2)
                
                if dist > 0:
                    cp_dir = vec2_normalize(to_cp)
                    forward = vec2(np.cos(vehicle.state.angle), np.sin(vehicle.state.angle))
                    right = vec2(-forward[1], forward[0])
                    
                    # Angle to checkpoint
                    dot = vec2_dot(cp_dir, forward)
                    cross = vec2_dot(cp_dir, right)
                    angle = np.arctan2(cross, dot)
                    
                    # Draw arrow indicator
                    indicator_x = self.width - 80
                    indicator_y = self.height // 2
                    arrow_len = 40
                    
                    # Rotate arrow
                    end_x = indicator_x + int(np.sin(angle) * arrow_len)
                    end_y = indicator_y - int(np.cos(angle) * arrow_len)
                    
                    pygame.draw.line(self.screen, (0, 255, 0), 
                                   (indicator_x, indicator_y), (end_x, end_y), 3)
                    pygame.draw.circle(self.screen, (0, 255, 0), (indicator_x, indicator_y), 10, 2)
                    
                    # Distance
                    dist_text = f"{dist:.0f}m"
                    dist_surf = self.font_small.render(dist_text, True, (0, 255, 0))
                    self.screen.blit(dist_surf, (indicator_x - 20, indicator_y + 25))
    
    def _render_minimap(self) -> None:
        """Render minimap in corner."""
        if not self.track:
            return
        
        size = self.minimap_size
        
        # Position
        if self.minimap_position == 'top-right':
            map_x = self.width - size - 10
            map_y = 10
        elif self.minimap_position == 'top-left':
            map_x = 10
            map_y = 10
        elif self.minimap_position == 'bottom-right':
            map_x = self.width - size - 10
            map_y = self.height - size - 10
        else:  # bottom-left
            map_x = 10
            map_y = self.height - size - 10
        
        # Background
        pygame.draw.rect(self.screen, (20, 20, 30, 200), (map_x, map_y, size, size))
        pygame.draw.rect(self.screen, (100, 100, 100), (map_x, map_y, size, size), 2)
        
        # Calculate track bounds
        bounds = self._get_track_bounds()
        if not bounds:
            return
        
        min_x, min_y, max_x, max_y = bounds
        track_w = max_x - min_x
        track_h = max_y - min_y
        scale = min(size / track_w, size / track_h) * 0.9
        offset_x = map_x + size / 2 - (min_x + max_x) / 2 * scale
        offset_y = map_y + size / 2 - (min_y + max_y) / 2 * scale
        
        # Draw track boundaries
        for boundary in [self.track.left_boundary, self.track.right_boundary]:
            if len(boundary) > 1:
                points = []
                for p in boundary:
                    px = int(offset_x + p[0] * scale)
                    py = int(offset_y + p[1] * scale)
                    points.append((px, py))
                pygame.draw.lines(self.screen, (100, 100, 100), False, points, 1)
        
        # Draw vehicles
        for vehicle in self.vehicles:
            pos = vehicle.state.position
            px = int(offset_x + pos[0] * scale)
            py = int(offset_y + pos[1] * scale)
            
            color = self.car_player_color if vehicle.state.name == "Player" else self.car_npc_color
            pygame.draw.circle(self.screen, color, (px, py), max(3, int(2 * scale)))
        
        # Draw checkpoints
        for i, cp in enumerate(self.track.checkpoints):
            px = int(offset_x + cp.position[0] * scale)
            py = int(offset_y + cp.position[1] * scale)
            color = self.start_finish_color if cp.is_start_finish else self.checkpoint_color
            pygame.draw.circle(self.screen, color, (px, py), 3)
    
    def _render_countdown(self) -> None:
        """Render race countdown."""
        if not self.race_state:
            return
        
        race_state = self.race_state.get('race_state', '')
        countdown = self.race_state.get('countdown', 0)
        
        if race_state == 'countdown' and countdown > 0:
            # Large countdown number
            if countdown == 0:
                text = "GO!"
                color = (0, 255, 0)
            else:
                text = str(countdown)
                color = (255, 255, 0)
            
            # Pulsing effect
            scale = 1.0 + 0.2 * np.sin(pygame.time.get_ticks() * 0.01)
            font = pygame.font.Font(None, int(120 * scale))
            text_surf = font.render(text, True, color)
            rect = text_surf.get_rect(center=(self.width // 2, self.height // 2))
            self.screen.blit(text_surf, rect)
    
    def _render_debug_overlay(self) -> None:
        """Render debug information overlay."""
        y = 10
        x = self.width - 350
        
        # Background
        overlay_surf = pygame.Surface((340, self.height - 20), pygame.SRCALPHA)
        overlay_surf.fill((0, 0, 0, 180))
        self.screen.blit(overlay_surf, (x, y))
        
        lines = [
            "=== DEBUG OVERLAY ===",
            f"FPS: {self.clock.get_fps():.1f}" if self.clock else "FPS: N/A",
            f"Sim Time: {self.race_state.get('sim_time', 0):.2f}s",
            f"Seed: {self.race_state.get('seed', 'N/A')}",
            "",
            "--- VEHICLE ---",
        ]
        
        if self.vehicles:
            v = self.vehicles[0]
            s = v.state
            lines.extend([
                f"Pos: ({s.position[0]:.1f}, {s.position[1]:.1f})",
                f"Angle: {np.degrees(s.angle):.1f}°",
                f"Speed: {s.speed:.2f} m/s ({s.speed*3.6:.1f} km/h)",
                f"Fwd: {s.forward_speed:.2f}, Lat: {s.lateral_speed:.2f}",
                f"Ang Vel: {s.angular_velocity:.3f}",
                f"Slip: {np.degrees(s.slip_angle):.1f}°",
                f"Drifting: {s.is_drifting} ({s.drift_intensity:.2f})",
                f"On Road: {s.on_road}",
                f"Surface: {s.surface_type} ({s.surface_grip:.2f})",
                f"Input: T={s.throttle:.2f} B={s.brake:.2f} S={s.steering:.2f}",
                f"Boost: {s.boost_active}",
                f"Collisions: {s.collision_count}",
                f"Distance: {s.distance_traveled:.1f}m",
            ])
        
        lines.append("")
        lines.append("--- RACE ---")
        if self.race_state.get('participants'):
            for p in self.race_state['participants'][:5]:
                lines.append(f"P{p['position']}: {p['name']} L{p['lap']}/{p['total_laps']} CP{p['checkpoint']} T={p['race_time']:.1f}")
        
        # Render lines
        for i, line in enumerate(lines):
            color = (255, 255, 100) if line.startswith("===") else (200, 200, 200)
            if line.startswith("---"):
                color = (100, 255, 100)
            text = self.font_mono.render(line, True, color)
            self.screen.blit(text, (x + 10, y + 10 + i * 16))
    
    def _get_track_bounds(self) -> Optional[Tuple[float, float, float, float]]:
        """Get track bounding box."""
        if not self.track or not self.track.centerline:
            return None
        
        points = np.array(self.track.centerline)
        min_x, min_y = np.min(points, axis=0)
        max_x, max_y = np.max(points, axis=0)
        
        return (min_x, min_y, max_x, max_y)
    
    def handle_event(self, event: pygame.event.Event) -> Optional[str]:
        """Handle pygame events. Returns action string if any."""
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                return "pause"
            elif event.key == pygame.K_F1:
                self.show_debug_overlay = not self.show_debug_overlay
            elif event.key == pygame.K_F2:
                self.show_racing_line = not self.show_racing_line
            elif event.key == pygame.K_F3:
                self.show_sensors = not self.show_sensors
            elif event.key == pygame.K_F4:
                self.show_hitboxes = not self.show_hitboxes
            elif event.key == pygame.K_F5:
                self.show_minimap = not self.show_minimap
            elif event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
                self.camera.target_zoom = min(self.camera.max_zoom, self.camera.target_zoom * 1.2)
            elif event.key == pygame.K_MINUS:
                self.camera.target_zoom = max(self.camera.min_zoom, self.camera.target_zoom / 1.2)
        
        return None
    
    def cleanup(self) -> None:
        """Clean up pygame."""
        pygame.quit()


def create_renderer(config: Dict[str, Any]) -> Renderer:
    """Factory function to create renderer."""
    return Renderer(config)