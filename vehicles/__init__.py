"""
Vehicles Package

Vehicle physics, configurations, and presets.
"""

from vehicles.vehicle import Vehicle, VehicleConfig, VehicleState
from vehicles.presets import VEHICLE_PRESETS, get_vehicle_preset, list_vehicle_presets, create_custom_vehicle

__all__ = [
    'Vehicle', 'VehicleConfig', 'VehicleState',
    'VEHICLE_PRESETS', 'get_vehicle_preset', 'list_vehicle_presets', 'create_custom_vehicle'
]