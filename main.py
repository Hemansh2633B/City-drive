#!/usr/bin/env python3
"""
Main Entry Point

Simple launcher for the AI Racing Simulator.
"""

import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="AI Racing Simulator")
    parser.add_argument("command", nargs="?", default="game",
                       choices=["game", "train", "eval", "test"],
                       help="Command to run")
    parser.add_argument("--headless", action="store_true", help="Run headless")
    parser.add_argument("--mode", default="1v1", help="Game mode")
    parser.add_argument("--seed", default=None, help="Track seed")
    parser.add_argument("--config", default=None, help="Config file")
    
    # Parse known args to determine command
    args, remaining = parser.parse_known_args()
    
    if args.command == "game":
        from scripts.game import main as game_main
        sys.argv = [sys.argv[0]] + remaining
        return game_main()
    elif args.command == "train":
        from scripts.train import main as train_main
        sys.argv = [sys.argv[0]] + remaining
        return train_main()
    elif args.command == "eval":
        from scripts.evaluate import main as eval_main
        sys.argv = [sys.argv[0]] + remaining
        return eval_main()
    elif args.command == "test":
        import pytest
        sys.argv = ["pytest", "tests/"] + remaining
        return pytest.main()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())