"""
AI Package

Training environment and NPC opponent AI.
"""

from ai.training_env import RacingEnv, TrainingConfig, ObservationMode, ActionMode, make_racing_env
from ai.npc_ai import NPCAI, AIProfile, AIProfileConfig, create_npc_ai, get_ai_profile, list_ai_profiles, AI_PROFILES

__all__ = [
    # Training env
    'RacingEnv', 'TrainingConfig', 'ObservationMode', 'ActionMode', 'make_racing_env',
    # NPC AI
    'NPCAI', 'AIProfile', 'AIProfileConfig', 'create_npc_ai', 'get_ai_profile', 'list_ai_profiles', 'AI_PROFILES'
]