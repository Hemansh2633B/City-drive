#!/usr/bin/env python3
"""
Setup script for AI Racing Simulator.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read requirements
requirements = []
with open("requirements.txt") as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#"):
            requirements.append(line)

# Read long description
long_description = ""
readme_path = Path("README.md")
if readme_path.exists():
    long_description = readme_path.read_text()

setup(
    name="ai-racing-simulator",
    version="1.0.0",
    author="AI Racing Team",
    description="2D top-down racing simulator for AI training and research",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ai-racing-simulator",
    packages=find_packages(include=["race", "race.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Games/Entertainment :: Simulation",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "rl": ["stable-baselines3>=2.0.0"],
        "dev": ["pytest>=7.4.0", "pytest-cov>=4.1.0", "black", "ruff"],
        "numba": ["numba>=0.58.0"],
    },
    entry_points={
        "console_scripts": [
            "ai-racing=main:main",
            "ai-racing-game=scripts.game:main",
            "ai-racing-train=scripts.train:main",
            "ai-racing-eval=scripts.evaluate:main",
        ],
    },
    include_package_data=True,
    package_data={
        "race": ["configs/*.yaml"],
    },
)