"""
Space Decay Prediction Package

A machine learning package for classifying space objects and predicting orbital decay.
"""

__version__ = "1.0.0"
__author__ = "thenithin342"

# Legacy modules
from .data_preprocessing import DataPreprocessor
from .feature_engineering import FeatureEngineer
from .model import SpaceObjectClassifier

# Utility modules
from .utils import load_model, save_model
from .logger import logger, get_logger, LoggerConfig
from .exception import (
    CustomException,
    DataValidationError,
    ModelTrainingError,
    DataPreprocessingError,
    FeatureEngineeringError,
    ModelLoadingError,
    PredictionError,
)

# Components
from .components import (
    DataIngestion,
    DataTransformation,
    ModelTrainer,
)

# Pipelines
from .pipeline import (
    TrainPipeline,
    PredictPipeline,
    CustomData,
)

__all__ = [
    # Legacy
    "DataPreprocessor",
    "FeatureEngineer",
    "SpaceObjectClassifier",
    # Utils
    "load_model",
    "save_model",
    # Logger
    "logger",
    "get_logger",
    "LoggerConfig",
    # Exceptions
    "CustomException",
    "DataValidationError",
    "ModelTrainingError",
    "DataPreprocessingError",
    "FeatureEngineeringError",
    "ModelLoadingError",
    "PredictionError",
    # Components
    "DataIngestion",
    "DataTransformation",
    "ModelTrainer",
    # Pipelines
    "TrainPipeline",
    "PredictPipeline",
    "CustomData",
]

