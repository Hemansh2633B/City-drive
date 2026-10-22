#!/usr/bin/env python3
"""Optional interactive single-agent training viewer."""

import sys
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import load_config
from game import create_game
from modes.race_modes import RaceMode


def parse_args():
    parser = argparse.ArgumentParser(description="City Drive interactive training viewer")
    parser.add_argument("--seed", type=str, default="12345",
                        help="Reproducible integer or text seed")
    parser.add_argument("--random-seed", action="store_true",
                        help="Choose a random seed")
    parser.add_argument("--daily-seed", action="store_true",
                        help="Use a date-based seed")
    parser.add_argument("--vehicle", choices=["balanced", "speedster", "gripper", "drift_king", "rookie"],
                        default="balanced", help="Vehicle preset for the viewer")
    parser.add_argument("--headless", action="store_true", help="Disable pygame display")
    parser.add_argument("--config", default=None, help="Optional YAML config file")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    seed = args.seed
    if args.daily_seed:
        from datetime import date
        today = date.today()
        seed = f"daily-{today.year}{today.month:02d}{today.day:02d}"
    elif args.random_seed:
        seed = int(np.random.randint(0, 2**31 - 1))
    else:
        try:
            seed = int(seed)
        except ValueError:
            pass

    game = create_game(config, headless=args.headless)
    game.selected_mode = RaceMode.TRAINING
    game.selected_seed = seed
    game.selected_vehicle = args.vehicle
    print("City Drive | single-agent training viewer")
    print(f"Seed: {seed}")
    print(f"Vehicle: {args.vehicle}")
    return game.run()


if __name__ == "__main__":
    sys.exit(main())
