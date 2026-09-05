#!/usr/bin/env python3
"""
Game Script

Runs the racing game with rendering for human play or AI vs AI.
"""

import sys
import os
import argparse
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import load_config
from game import create_game
from modes.race_modes import RaceMode


def parse_args():
    parser = argparse.ArgumentParser(description="AI Racing Simulator")
    
    # Mode
    parser.add_argument("--mode", type=str, default="1v1",
                       choices=["training", "1v1", "1v4", "ai_vs_ai"],
                       help="Game mode")
    
    # Seed
    parser.add_argument("--seed", type=str, default=None,
                       help="Track seed (string or integer)")
    parser.add_argument("--random-seed", action="store_true",
                       help="Use random seed")
    parser.add_argument("--daily-seed", action="store_true",
                       help="Use daily seed")
    
    # Vehicle/Track
    parser.add_argument("--vehicle", type=str, default="balanced",
                       choices=["balanced", "speedster", "gripper", "drift_king", "rookie"],
                       help="Vehicle preset")
    parser.add_argument("--theme", type=str, default="asphalt",
                       choices=["asphalt", "desert", "snow", "grassland", "neon", "industrial", "dirt"],
                       help="Track theme")
    parser.add_argument("--complexity", type=float, default=0.5,
                       help="Track complexity (0.0-1.0)")
    parser.add_argument("--road-width", type=float, default=None,
                       help="Road width in meters")
    parser.add_argument("--laps", type=int, default=3,
                       help="Number of laps")
    parser.add_argument("--ai-profile", type=str, default="normal",
                       choices=["beginner", "normal", "expert", "aggressive", "defensive"],
                       help="AI opponent profile")
    
    # Options
    parser.add_argument("--headless", action="store_true",
                       help="Run in headless mode (no rendering)")
    parser.add_argument("--catchup", action="store_true",
                       help="Enable catch-up/rubber-banding")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to config file")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Parse seed
    seed = args.seed
    if args.daily_seed:
        from datetime import date
        today = date.today()
        seed = f"daily-{today.year}{today.month:02d}{today.day:02d}"
    elif args.random_seed or seed is None:
        seed = np.random.randint(0, 2**31 - 1)
    else:
        try:
            seed = int(seed)
        except ValueError:
            pass  # Keep as string
    
    # Parse mode
    mode_map = {
        "training": RaceMode.TRAINING,
        "1v1": RaceMode.ONE_V_ONE,
        "1v4": RaceMode.ONE_V_FOUR,
        "ai_vs_ai": RaceMode.ONE_V_FOUR  # Use 1v4 with 5 AI
    }
    mode = mode_map.get(args.mode, RaceMode.ONE_V_ONE)
    
    # Override config with command line args
    if args.catchup:
        if 'race' not in config.to_dict():
            config.race = {}
        config.race.catchup_enabled = True
    
    # Create game
    game = create_game(config, headless=args.headless)
    
    # Set game parameters before running
    game.selected_mode = mode
    game.selected_seed = seed
    game.selected_vehicle = args.vehicle
    game.selected_ai_profile = args.ai_profile
    
    # For AI vs AI, disable human player
    if args.mode == "ai_vs_ai":
        game.config.player_is_human = False
    
    print(f"Starting {mode.value} race")
    print(f"Seed: {seed}")
    print(f"Vehicle: {args.vehicle}")
    print(f"Theme: {args.theme}")
    print(f"Laps: {args.laps}")
    print(f"Headless: {args.headless}")
    
    # Run game
    return game.run()


if __name__ == "__main__":
    sys.exit(main())