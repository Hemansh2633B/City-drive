"""
Vehicle Presets and Factory

Predefined vehicle configurations and factory functions.
"""

from dataclasses import dataclass
from typing import Dict
from vehicles.vehicle import VehicleConfig


# Vehicle presets matching the config
VEHICLE_PRESETS: Dict[str, VehicleConfig] = {
    "balanced": VehicleConfig(
        top_speed=45.0,
        acceleration=12.0,
        brake_force=18.0,
        steering_sensitivity=3.5,
        grip=1.0,
        mass=1200.0,
        width=2.0,
        length=4.5,
        drift_factor=0.3,
        drag_coefficient=0.32,
        downforce=0.0
    ),
    "speedster": VehicleConfig(
        top_speed=65.0,
        acceleration=15.0,
        brake_force=15.0,
        steering_sensitivity=2.8,
        grip=0.7,
        mass=1000.0,
        width=1.9,
        length=4.3,
        drift_factor=0.5,
        drag_coefficient=0.28,
        downforce=0.0
    ),
    "gripper": VehicleConfig(
        top_speed=35.0,
        acceleration=10.0,
        brake_force=22.0,
        steering_sensitivity=4.2,
        grip=1.4,
        mass=1400.0,
        width=2.1,
        length=4.7,
        drift_factor=0.15,
        drag_coefficient=0.38,
        downforce=500.0
    ),
    # Additional presets for variety
    "drift_king": VehicleConfig(
        top_speed=50.0,
        acceleration=14.0,
        brake_force=16.0,
        steering_sensitivity=4.0,
        grip=0.85,
        mass=1100.0,
        width=1.95,
        length=4.4,
        drift_factor=0.7,
        drag_coefficient=0.30,
        downforce=0.0
    ),
    "rookie": VehicleConfig(
        top_speed=30.0,
        acceleration=8.0,
        brake_force=20.0,
        steering_sensitivity=3.0,
        grip=1.2,
        mass=1300.0,
        width=2.1,
        length=4.6,
        drift_factor=0.1,
        drag_coefficient=0.35,
        downforce=200.0
    )
}


def get_vehicle_preset(name: str) -> VehicleConfig:
    """Get a vehicle preset by name."""
    if name not in VEHICLE_PRESETS:
        raise ValueError(f"Unknown vehicle preset: {name}. Available: {list(VEHICLE_PRESETS.keys())}")
    return VEHICLE_PRESETS[name]


def list_vehicle_presets() -> list:
    """List all available vehicle preset names."""
    return list(VEHICLE_PRESETS.keys())


def create_custom_vehicle(**kwargs) -> VehicleConfig:
    """Create a custom vehicle configuration."""
    # Start with balanced as base
    config = VehicleConfig()
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Unknown vehicle config parameter: {key}")
    return config