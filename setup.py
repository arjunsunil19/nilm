#!/usr/bin/env python
"""
NILM Package Setup

Installation script for the NILM (Non-Intrusive Load Monitoring) package.
"""

from setuptools import setup, find_packages
import os

# Read the README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
def read_requirements(filename):
    """Read requirements from file."""
    requirements = []
    if os.path.exists(filename):
        with open(filename, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    requirements.append(line)
    return requirements

setup(
    name="nilm",
    version="0.1.0",
    author="NILM Team",
    author_email="",
    description="Non-Intrusive Load Monitoring using Deep Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/arjunsunil19/nilm",
    packages=find_packages(exclude=["tests", "tests.*", "examples"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "torch>=1.9.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.0.0",
            "pandas>=1.3.0",
        ],
        "full": [
            "pandas>=1.3.0",
            "matplotlib>=3.4.0",
            "tensorboard>=2.6.0",
            "jupyter>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "nilm-train=examples.train_model:main",
        ],
    },
)
