#!/usr/bin/env python3
"""Package metadata for City Drive RL."""

from pathlib import Path
from setuptools import setup, find_packages

requirements = []
with open(Path(__file__).with_name("requirements.txt"), encoding="utf-8") as stream:
    for line in stream:
        line = line.strip()
        if line and not line.startswith("#"):
            requirements.append(line)

readme_path = Path(__file__).with_name("README.md")
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

setup(
    name="city-drive-rl",
    version="2.0.0",
    author="City Drive Contributors",
    description="Seeded 2D driving environment for single-agent reinforcement learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Hemansh2633B/City-drive",
    packages=find_packages(),
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
    install_requires=requirements,
    extras_require={
        "dev": ["pytest>=7.4.0", "pytest-cov>=4.1.0", "ruff"],
        "numba": ["numba>=0.58.0"],
        "tensorboard": ["tensorboard>=2.14.0"],
    },
    entry_points={
        "console_scripts": [
            "city-drive=main:main",
            "city-drive-train=scripts.train:main",
            "city-drive-view=scripts.game:main",
            "city-drive-eval=scripts.evaluate:main",
        ],
    },
    include_package_data=True,
)
