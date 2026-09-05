"""
Modes Package

Race modes: Training, 1v1, 1v4, and race system.
"""

from modes.race_system import RaceManager, RaceConfig, RaceState, CarState, RaceResult, LapTime, RaceParticipant
from modes.training_mode import TrainingMode, TrainingModeConfig
from modes.race_modes import (
    RaceMode, RaceModeConfig, BaseRaceMode, OneVOneMode, OneVFourMode, AIVsAIMode, create_race_mode
)

__all__ = [
    # Race system
    'RaceManager', 'RaceConfig', 'RaceState', 'CarState', 'RaceResult', 'LapTime', 'RaceParticipant',
    # Training mode
    'TrainingMode', 'TrainingModeConfig',
    # Race modes
    'RaceMode', 'RaceModeConfig', 'BaseRaceMode', 'OneVOneMode', 'OneVFourMode', 'AIVsAIMode', 'create_race_mode'
]