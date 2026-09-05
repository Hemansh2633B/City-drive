"""
Menu System

Main menu, mode selection, settings, and other UI screens.
"""

import pygame
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum

from ui.renderer import Renderer


class MenuState(Enum):
    MAIN = "main"
    MODE_SELECT = "mode_select"
    TRAINING_CONFIG = "training_config"
    RACE_SETUP = "race_setup"
    SEED_ENTRY = "seed_entry"
    VEHICLE_SELECT = "vehicle_select"
    SETTINGS = "settings"
    PAUSE = "pause"
    RESULTS = "results"
    REPLAY_BROWSER = "replay_browser"


@dataclass
class MenuItem:
    """A single menu item."""
    text: str
    action: Optional[str] = None
    callback: Optional[Callable] = None
    enabled: bool = True
    value: Any = None


class Menu:
    """Base menu class."""
    
    def __init__(self, renderer: Renderer, title: str):
        self.renderer = renderer
        self.title = title
        self.items: List[MenuItem] = []
        self.selected_index = 0
        self.visible = True
        self.parent: Optional['Menu'] = None
    
    def add_item(self, text: str, action: str = None, callback: Callable = None, 
                 enabled: bool = True, value: Any = None) -> None:
        """Add a menu item."""
        self.items.append(MenuItem(text, action, callback, enabled, value))
    
    def handle_event(self, event: pygame.event.Event) -> Optional[str]:
        """Handle input events. Returns action string."""
        if not self.visible:
            return None
        
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                self._move_selection(-1)
            elif event.key == pygame.K_DOWN:
                self._move_selection(1)
            elif event.key == pygame.K_RETURN or event.key == pygame.K_SPACE:
                return self._activate_selected()
            elif event.key == pygame.K_ESCAPE:
                return "back"
        
        return None
    
    def _move_selection(self, direction: int) -> None:
        """Move selection up/down, skipping disabled items."""
        if not self.items:
            return
        
        original = self.selected_index
        while True:
            self.selected_index = (self.selected_index + direction) % len(self.items)
            if self.items[self.selected_index].enabled:
                break
            if self.selected_index == original:
                break  # All items disabled
    
    def _activate_selected(self) -> Optional[str]:
        """Activate the currently selected item."""
        if not self.items or self.selected_index >= len(self.items):
            return None
        
        item = self.items[self.selected_index]
        if not item.enabled:
            return None
        
        if item.callback:
            return item.callback()
        return item.action
    
    def render(self) -> None:
        """Render the menu."""
        if not self.visible:
            return
        
        screen = self.renderer.screen
        width = self.renderer.width
        height = self.renderer.height
        
        # Semi-transparent background
        overlay = pygame.Surface((width, height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 200))
        screen.blit(overlay, (0, 0))
        
        # Title
        title_surf = self.renderer.font_large.render(self.title, True, (255, 255, 255))
        title_rect = title_surf.get_rect(center=(width // 2, 80))
        screen.blit(title_surf, title_rect)
        
        # Menu items
        start_y = 180
        spacing = 50
        
        for i, item in enumerate(self.items):
            color = (255, 255, 255) if item.enabled else (100, 100, 100)
            
            if i == self.selected_index:
                # Highlighted
                text = f"> {item.text} <"
                color = (255, 255, 0)
            else:
                text = f"  {item.text}  "
            
            text_surf = self.renderer.font_medium.render(text, True, color)
            text_rect = text_surf.get_rect(center=(width // 2, start_y + i * spacing))
            screen.blit(text_surf, text_rect)
    
    def set_visible(self, visible: bool) -> None:
        """Show/hide menu."""
        self.visible = visible
        if visible:
            self.selected_index = 0


class MainMenu(Menu):
    """Main menu."""
    
    def __init__(self, renderer: Renderer):
        super().__init__(renderer, "AI RACING SIMULATOR")
        self.add_item("Training Mode", "mode_training")
        self.add_item("1v1 Race", "mode_1v1")
        self.add_item("1v4 Race", "mode_1v4")
        self.add_item("AI vs AI Benchmark", "mode_ai_vs_ai")
        self.add_item("Replay Browser", "replay_browser")
        self.add_item("Settings", "settings")
        self.add_item("Quit", "quit")


class ModeSelectMenu(Menu):
    """Mode selection menu (legacy - now in main)."""
    
    def __init__(self, renderer: Renderer):
        super().__init__(renderer, "SELECT MODE")
        self.add_item("Training Mode", "mode_training")
        self.add_item("1v1 Race", "mode_1v1")
        self.add_item("1v4 Race", "mode_1v4")
        self.add_item("AI vs AI (5 cars)", "mode_ai_vs_ai")
        self.add_item("Back", "back")


class SeedEntryMenu(Menu):
    """Seed entry menu."""
    
    def __init__(self, renderer: Renderer, on_seed_entered: Callable[[str], None]):
        super().__init__(renderer, "ENTER SEED")
        self.on_seed_entered = on_seed_entered
        self.input_text = ""
        self.input_active = True
        
        self.add_item("Random Seed", "random_seed")
        self.add_item("Daily Seed", "daily_seed")
        self.add_item("Recent Seeds", "recent_seeds")
        self.add_item("Back", "back")
    
    def handle_event(self, event: pygame.event.Event) -> Optional[str]:
        if not self.visible:
            return None
        
        if self.input_active:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    if self.input_text:
                        self.on_seed_entered(self.input_text)
                        self.input_text = ""
                    return "seed_entered"
                elif event.key == pygame.K_BACKSPACE:
                    self.input_text = self.input_text[:-1]
                elif event.key == pygame.K_ESCAPE:
                    self.input_active = False
                else:
                    # Only allow alphanumeric and some special chars
                    if event.unicode.isprintable() and len(self.input_text) < 50:
                        self.input_text += event.unicode
            return None
        
        # Normal menu navigation
        result = super().handle_event(event)
        
        if result == "random_seed":
            import numpy as np
            seed = str(np.random.randint(0, 2**31 - 1))
            self.on_seed_entered(seed)
            return "seed_entered"
        elif result == "daily_seed":
            from datetime import date
            today = date.today()
            seed = f"daily-{today.year}{today.month:02d}{today.day:02d}"
            self.on_seed_entered(seed)
            return "seed_entered"
        elif result == "recent_seeds":
            return "show_recent"
        
        return result
    
    def render(self) -> None:
        super().render()
        
        if self.input_active:
            screen = self.renderer.screen
            width = self.renderer.width
            
            # Input box
            box_rect = pygame.Rect(width // 2 - 200, 120, 400, 40)
            pygame.draw.rect(screen, (50, 50, 60), box_rect)
            pygame.draw.rect(screen, (255, 255, 0), box_rect, 2)
            
            # Input text with cursor
            cursor = "|" if (pygame.time.get_ticks() // 500) % 2 == 0 else ""
            display_text = self.input_text + cursor
            text_surf = self.renderer.font_medium.render(display_text, True, (255, 255, 255))
            text_rect = text_surf.get_rect(midleft=(box_rect.left + 10, box_rect.centery))
            screen.blit(text_surf, text_rect)
            
            # Hint
            hint = self.renderer.font_small.render("Type seed and press ENTER, or select option below", True, (150, 150, 150))
            hint_rect = hint.get_rect(center=(width // 2, 175))
            screen.blit(hint, hint_rect)


class VehicleSelectMenu(Menu):
    """Vehicle selection menu."""
    
    def __init__(self, renderer: Renderer, vehicles: List[str], on_select: Callable[[str], None]):
        super().__init__(renderer, "SELECT VEHICLE")
        self.on_select = on_select
        
        for v in vehicles:
            self.add_item(v.capitalize(), callback=lambda v=v: self._select_vehicle(v))
        self.add_item("Back", "back")
    
    def _select_vehicle(self, vehicle: str) -> str:
        self.on_select(vehicle)
        return "vehicle_selected"


class SettingsMenu(Menu):
    """Settings menu."""
    
    def __init__(self, renderer: Renderer, config: Dict[str, Any]):
        super().__init__(renderer, "SETTINGS")
        self.config = config
        self._build_items()
    
    def _build_items(self) -> None:
        self.items.clear()
        
        # Graphics
        self.add_item(f"Resolution: {self.config['render']['width']}x{self.config['render']['height']}", 
                      callback=self._cycle_resolution)
        self.add_item(f"VSync: {'On' if self.config['render'].get('vsync', True) else 'Off'}", 
                      callback=self._toggle_vsync)
        self.add_item(f"Show Minimap: {'On' if self.config['render'].get('show_minimap', True) else 'Off'}", 
                      callback=self._toggle_minimap)
        self.add_item(f"Show Racing Line: {'On' if self.config['render'].get('show_racing_line', False) else 'Off'}", 
                      callback=self._toggle_racing_line)
        self.add_item(f"Show Sensors: {'On' if self.config['render'].get('show_sensors', False) else 'Off'}", 
                      callback=self._toggle_sensors)
        
        # Audio
        self.add_item(f"Master Volume: {int(self.config.get('audio', {}).get('master_volume', 1.0) * 100)}%", 
                      callback=self._cycle_volume)
        
        # Gameplay
        self.add_item(f"Catch-up: {'On' if self.config.get('race', {}).get('catchup_enabled', False) else 'Off'}", 
                      callback=self._toggle_catchup)
        
        self.add_item("Back", "back")
    
    def _cycle_resolution(self) -> str:
        resolutions = [(1280, 720), (1920, 1080), (2560, 1440), (3840, 2160)]
        current = (self.config['render']['width'], self.config['render']['height'])
        idx = resolutions.index(current) if current in resolutions else 0
        idx = (idx + 1) % len(resolutions)
        self.config['render']['width'], self.config['render']['height'] = resolutions[idx]
        self._build_items()
        return "resolution_changed"
    
    def _toggle_vsync(self) -> str:
        self.config['render']['vsync'] = not self.config['render'].get('vsync', True)
        self._build_items()
        return "settings_changed"
    
    def _toggle_minimap(self) -> str:
        self.config['render']['show_minimap'] = not self.config['render'].get('show_minimap', True)
        self._build_items()
        return "settings_changed"
    
    def _toggle_racing_line(self) -> str:
        self.config['render']['show_racing_line'] = not self.config['render'].get('show_racing_line', False)
        self._build_items()
        return "settings_changed"
    
    def _toggle_sensors(self) -> str:
        self.config['render']['show_sensors'] = not self.config['render'].get('show_sensors', False)
        self._build_items()
        return "settings_changed"
    
    def _cycle_volume(self) -> str:
        vols = [0.0, 0.25, 0.5, 0.75, 1.0]
        current = self.config.get('audio', {}).get('master_volume', 1.0)
        idx = vols.index(current) if current in vols else 0
        idx = (idx + 1) % len(vols)
        if 'audio' not in self.config:
            self.config['audio'] = {}
        self.config['audio']['master_volume'] = vols[idx]
        self._build_items()
        return "settings_changed"
    
    def _toggle_catchup(self) -> str:
        if 'race' not in self.config:
            self.config['race'] = {}
        self.config['race']['catchup_enabled'] = not self.config['race'].get('catchup_enabled', False)
        self._build_items()
        return "settings_changed"


class PauseMenu(Menu):
    """Pause menu during race."""
    
    def __init__(self, renderer: Renderer):
        super().__init__(renderer, "PAUSED")
        self.add_item("Resume", "resume")
        self.add_item("Restart Race", "restart")
        self.add_item("Settings", "settings")
        self.add_item("Quit to Menu", "quit")


class ResultsMenu(Menu):
    """Race results screen."""
    
    def __init__(self, renderer: Renderer, results: Dict[str, Any]):
        super().__init__(renderer, "RACE RESULTS")
        self.results = results
        self._build_items()
    
    def _build_items(self) -> None:
        self.items.clear()
        
        # Winner
        if self.results.get('winner'):
            winner = self.results['winner']
            self.add_item(f"WINNER: {winner['name']} - {winner['total_time']:.2f}s", enabled=False)
        
        self.add_item("", enabled=False)  # Spacer
        
        # Standings
        for i, p in enumerate(self.results.get('standings', [])):
            status = "FINISHED" if p['state'] == 'finished' else ("DNF" if p['state'] == 'dnf' else "RACING")
            time_str = f"{p['race_time']:.2f}s" if p['state'] == 'finished' else ""
            best_lap = f"Best: {p['best_lap_time']:.2f}s" if p['best_lap_time'] > 0 else ""
            self.add_item(f"{i+1}. {p['name']} - {status} {time_str} {best_lap}", enabled=False)
        
        self.add_item("", enabled=False)
        self.add_item("Race Again", "restart")
        self.add_item("Main Menu", "main_menu")


class ReplayBrowserMenu(Menu):
    """Replay browser menu."""
    
    def __init__(self, renderer: Renderer, replay_files: List[str], on_load: Callable[[str], None]):
        super().__init__(renderer, "REPLAY BROWSER")
        self.on_load = on_load
        self.replay_files = replay_files
        self._build_items()
    
    def _build_items(self) -> None:
        self.items.clear()
        
        for f in self.replay_files[:20]:  # Limit display
            name = f.split('/')[-1].replace('.json', '').replace('.csv', '')
            self.add_item(name, callback=lambda f=f: self._load_replay(f))
        
        if not self.replay_files:
            self.add_item("No replays found", enabled=False)
        
        self.add_item("Back", "back")
    
    def _load_replay(self, filepath: str) -> str:
        self.on_load(filepath)
        return "replay_loaded"


class MenuManager:
    """Manages menu stack and transitions."""
    
    def __init__(self, renderer: Renderer, config: Dict[str, Any]):
        self.renderer = renderer
        self.config = config
        self.menu_stack: List[Menu] = []
        self.current_menu: Optional[Menu] = None
        
        # Create menus
        self.main_menu = MainMenu(renderer)
        self.seed_menu = SeedEntryMenu(renderer, self._on_seed_entered)
        self.vehicle_menu = VehicleSelectMenu(renderer, 
            ["balanced", "speedster", "gripper", "drift_king", "rookie"],
            self._on_vehicle_selected)
        self.settings_menu = SettingsMenu(renderer, config)
        self.pause_menu = PauseMenu(renderer)
        
        self._push_menu(self.main_menu)
        
        # Callbacks
        self.on_mode_selected: Optional[Callable] = None
        self.on_seed_entered_callback: Optional[Callable] = None
        self.on_vehicle_selected_callback: Optional[Callable] = None
        self.on_quit: Optional[Callable] = None
        self.on_restart: Optional[Callable] = None
        self.on_resume: Optional[Callable] = None
    
    def _push_menu(self, menu: Menu) -> None:
        """Push menu onto stack."""
        if self.current_menu:
            self.current_menu.set_visible(False)
        self.menu_stack.append(menu)
        self.current_menu = menu
        self.current_menu.set_visible(True)
    
    def _pop_menu(self) -> None:
        """Pop current menu."""
        if self.menu_stack:
            self.menu_stack.pop().set_visible(False)
        if self.menu_stack:
            self.current_menu = self.menu_stack[-1]
            self.current_menu.set_visible(True)
        else:
            self.current_menu = None
    
    def handle_event(self, event: pygame.event.Event) -> Optional[str]:
        """Handle events for current menu."""
        if not self.current_menu:
            return None
        
        action = self.current_menu.handle_event(event)
        
        if action:
            return self._handle_action(action)
        
        return None
    
    def _handle_action(self, action: str) -> Optional[str]:
        """Handle menu actions."""
        if action == "back":
            self._pop_menu()
            return None
        
        elif action == "mode_training":
            self._push_menu(self.seed_menu)
            self.seed_menu.input_active = True
            return None
        
        elif action == "mode_1v1" or action == "mode_1v4" or action == "mode_ai_vs_ai":
            self._push_menu(self.vehicle_menu)
            return None
        
        elif action == "replay_browser":
            self._show_replay_browser()
            return None
        
        elif action == "settings":
            self._push_menu(self.settings_menu)
            return None
        
        elif action == "quit":
            if self.on_quit:
                self.on_quit()
            return "quit"
        
        elif action == "resume":
            if self.on_resume:
                self.on_resume()
            return "resume"
        
        elif action == "restart":
            if self.on_restart:
                self.on_restart()
            return "restart"
        
        elif action == "main_menu":
            self._pop_to_main()
            return None
        
        elif action == "seed_entered":
            self._pop_menu()
            return None
        
        elif action == "show_recent":
            self._show_recent_seeds()
            return None
        
        elif action == "vehicle_selected":
            self._pop_menu()
            return None
        
        elif action == "replay_loaded":
            self._pop_to_main()
            return None
        
        return action
    
    def _on_seed_entered(self, seed: str) -> None:
        if self.on_seed_entered_callback:
            self.on_seed_entered_callback(seed)
    
    def _on_vehicle_selected(self, vehicle: str) -> None:
        if self.on_vehicle_selected_callback:
            self.on_vehicle_selected_callback(vehicle)
    
    def _pop_to_main(self) -> None:
        while len(self.menu_stack) > 1:
            self._pop_menu()
    
    def _show_replay_browser(self) -> None:
        import glob
        files = glob.glob("replays/*.json") + glob.glob("replays/*.csv")
        files.sort(key=lambda x: -os.path.getmtime(x))
        
        browser = ReplayBrowserMenu(self.renderer, files, self._on_replay_load)
        self._push_menu(browser)
    
    def _on_replay_load(self, filepath: str) -> None:
        # Would load replay
        pass
    
    def _show_recent_seeds(self) -> None:
        # Show recent seeds menu
        pass
    
    def show_pause(self) -> None:
        """Show pause menu."""
        self._push_menu(self.pause_menu)
    
    def hide_pause(self) -> None:
        """Hide pause menu."""
        if self.current_menu == self.pause_menu:
            self._pop_menu()
    
    def show_results(self, results: Dict[str, Any]) -> None:
        """Show results menu."""
        results_menu = ResultsMenu(self.renderer, results)
        self._push_menu(results_menu)
    
    def render(self) -> None:
        """Render current menu."""
        if self.current_menu:
            self.current_menu.render()
    
    def is_menu_open(self) -> bool:
        """Check if any menu is open."""
        return self.current_menu is not None and self.current_menu.visible


import os