#!/usr/bin/env python3
"""
AI Training Launcher & Main Entry Point

Focused lightweight launcher for reinforcement learning training, evaluation, and testing.
"""

import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="AI Racing Simulator Training Launcher")
    parser.add_argument("command", nargs="?", default="train",
                       choices=["train", "eval", "game", "test"],
                       help="Command to run: 'train' for AI training, 'eval' for evaluation, 'test' for tests")
    parser.add_argument("--mode", default="training", help="Training mode")
    parser.add_argument("--seed", default=None, help="Track seed")
    parser.add_argument("--config", default=None, help="Config file path")
    
    # Parse known args
    args, remaining = parser.parse_known_args()
    
    if args.command in ("train", "game"):
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
