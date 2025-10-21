"""
Space Decay Prediction Package

A machine learning package for classifying space objects and predicting orbital decay.
"""

__version__ = "1.0.0"
__author__ = "thenithin342"

from .data_preprocessing import DataPreprocessor
from .feature_engineering import FeatureEngineer
from .model import SpaceObjectClassifier
from .utils import load_model, save_model

__all__ = [
    "DataPreprocessor",
    "FeatureEngineer",
    "SpaceObjectClassifier",
    "load_model",
    "save_model",
]

