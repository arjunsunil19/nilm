"""
Data Module for NILM

Contains utilities for loading and preprocessing energy consumption data.
"""

from .data_loader import DataLoader
from .preprocessor import DataPreprocessor

__all__ = [
    "DataLoader",
    "DataPreprocessor",
]
