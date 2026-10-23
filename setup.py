#!/usr/bin/env python3
"""Package metadata for City Drive RL."""

from pathlib import Path
from setuptools import setup, find_packages

ROOT = Path(__file__).parent
readme_path = ROOT / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

setup(
    name="city-drive-rl",
    version="2.1.0",
    author="City Drive Contributors",
    description="Seeded 2D driving environment for Gymnasium and reinforcement learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Hemansh2633B/City-drive",
    packages=find_packages(exclude=("tests", "tests.*")),
    py_modules=["main"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "gymnasium>=0.29.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "rl": ["stable-baselines3>=2.3.0"],
        "render": ["pygame-ce>=2.5.0"],
        "gpu": ["torch>=2.2.0"],
        "dev": ["pytest>=7.4.0", "pytest-cov>=4.1.0", "ruff", "build"],
        "tensorboard": ["tensorboard>=2.14.0"],
        "full": [
            "stable-baselines3>=2.3.0",
            "pygame-ce>=2.5.0",
            "tensorboard>=2.14.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "city-drive=main:main",
            "city-drive-train=scripts.train:main",
            "city-drive-view=scripts.game:main",
            "city-drive-eval=scripts.evaluate:main",
            "city-drive-train-gpu=scripts.train_gpu_evolution:main",
        ],
    },
    include_package_data=True,
    package_data={"configs": ["default.yaml", "gpu_evolution.yaml"]},
)
