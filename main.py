#!/usr/bin/env python3
"""Command-line entry point for the City Drive RL training environment."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="City Drive RL: train, evaluate, or test a single-agent driving environment."
    )
    parser.add_argument(
        "command", nargs="?", default="train", choices=["train", "eval", "test"],
        help="Command to run (defaults to train)"
    )
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining

    if args.command == "train":
        from scripts.train import main as train_main
        return train_main()
    if args.command == "eval":
        from scripts.evaluate import main as eval_main
        return eval_main()
    if args.command == "test":
        import pytest
        return pytest.main(["tests/", *remaining])
    return 0


if __name__ == "__main__":
    sys.exit(main())
