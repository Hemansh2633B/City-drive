"""
Configuration Management

Loads and manages game configuration from YAML files with support for
environment-specific overrides and validation.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union
import yaml

from core.random import DeterministicRNG, SeededRandomStreams


class Config:
    """Configuration manager with nested attribute access."""
    
    def __init__(self, data: Dict[str, Any]):
        self._data = data
        self._wrap_dicts()
    
    def _wrap_dicts(self) -> None:
        """Recursively wrap nested dicts as Config objects."""
        for key, value in self._data.items():
            if isinstance(value, dict):
                self._data[key] = Config(value)
    
    def __getattr__(self, name: str) -> Any:
        if name in self._data:
            return self._data[name]
        raise AttributeError(f"Config has no attribute '{name}'")
    
    def __getitem__(self, key: str) -> Any:
        return self._data[key]
    
    def __setitem__(self, key: str, value: Any) -> None:
        self._data[key] = value
        if isinstance(value, dict):
            self._data[key] = Config(value)
    
    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert back to plain dict."""
        result = {}
        for key, value in self._data.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result
    
    def update(self, other: Dict[str, Any]) -> None:
        """Update with another dict, recursively merging."""
        for key, value in other.items():
            if key in self._data and isinstance(self._data[key], Config) and isinstance(value, dict):
                self._data[key].update(value)
            else:
                self._data[key] = value
                if isinstance(value, dict):
                    self._data[key] = Config(value)


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file. If None, uses default.
    
    Returns:
        Config object with nested attribute access.
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent / "configs" / "default.yaml"
    
    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)
    
    return Config(data)


def load_config_with_overrides(base_path: str, override_paths: list[str]) -> Config:
    """Load base config and apply overrides in order."""
    config = load_config(base_path)
    for override_path in override_paths:
        with open(override_path, 'r') as f:
            override_data = yaml.safe_load(f)
        config.update(override_data)
    return config


# Global config instance (set by init_config)
_global_config: Optional[Config] = None


def init_config(config_path: Optional[str] = None) -> Config:
    """Initialize global configuration."""
    global _global_config
    _global_config = load_config(config_path)
    return _global_config


def get_config() -> Config:
    """Get global configuration instance."""
    if _global_config is None:
        return init_config()
    return _global_config


class SimulationContext:
    """
    Context object that holds all simulation state for deterministic execution.
    
    This includes:
    - Configuration
    - Random number generators
    - Current step/frame count
    - Physics parameters
    """
    
    def __init__(self, config: Config, seed: Optional[Union[int, str, bytes]] = None):
        self.config = config
        self.seed = seed
        self.rng_streams = SeededRandomStreams(seed if seed is not None else 0)
        self.step_count = 0
        self.frame_count = 0
        self.time = 0.0
        self.dt = config.game.fixed_dt
        self.substeps = config.game.physics_substeps
        self.sub_dt = self.dt / self.substeps
    
    def get_rng(self, name: str) -> DeterministicRNG:
        """Get a named random stream."""
        return self.rng_streams.get_stream(name)
    
    def advance_step(self) -> None:
        """Advance simulation by one step."""
        self.step_count += 1
        self.time = self.step_count * self.dt
    
    def advance_frame(self) -> None:
        """Advance by one render frame."""
        self.frame_count += 1
    
    def get_state(self) -> dict:
        """Get serializable state for replay."""
        return {
            'seed': self.seed,
            'step_count': self.step_count,
            'frame_count': self.frame_count,
            'time': self.time,
            'rng_state': self.rng_streams.get_state()
        }
    
    def set_state(self, state: dict) -> None:
        """Restore state from serialized dict."""
        self.step_count = state['step_count']
        self.frame_count = state['frame_count']
        self.time = state['time']
        self.rng_streams.set_state(state['rng_state'])


def create_simulation_context(config: Optional[Config] = None, 
                              seed: Optional[Union[int, str, bytes]] = None) -> SimulationContext:
    """Factory function to create a simulation context."""
    if config is None:
        config = get_config()
    return SimulationContext(config, seed)
