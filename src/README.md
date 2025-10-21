# Source Code Structure

This directory contains the complete ML project source code organized into modular components.

## 📁 Directory Structure

```
src/
├── __init__.py                 # Main package initialization
├── logger.py                   # Logging configuration
├── exception.py                # Custom exception classes
├── utils.py                    # Utility functions
│
├── components/                 # ML Pipeline Components
│   ├── __init__.py
│   ├── data_ingestion.py      # Data loading and splitting
│   ├── data_transformation.py # Data preprocessing and feature engineering
│   └── model_trainer.py       # Model training and evaluation
│
├── pipeline/                   # ML Pipelines
│   ├── __init__.py
│   ├── train_pipeline.py      # End-to-end training pipeline
│   └── predict_pipeline.py    # Prediction pipeline
│
└── [Legacy modules]           # Original modules (for backward compatibility)
    ├── data_preprocessing.py
    ├── feature_engineering.py
    └── model.py
```

## 🔧 Components

### Data Ingestion (`components/data_ingestion.py`)
- Loads raw data from CSV files
- Performs train-test split
- Saves data to artifacts directory
- Validates data quality

### Data Transformation (`components/data_transformation.py`)
- Handles missing values
- Scales numerical features
- Preprocesses categorical features
- Creates preprocessing pipelines
- Saves preprocessor for reuse

### Model Trainer (`components/model_trainer.py`)
- Trains multiple ML models
- Performs hyperparameter tuning using GridSearchCV
- Evaluates model performance
- Selects best model based on accuracy
- Saves trained model

## 🚀 Pipelines

### Training Pipeline (`pipeline/train_pipeline.py`)
Orchestrates the complete training workflow:
1. Data Ingestion
2. Data Transformation
3. Model Training

**Usage:**
```python
from src.pipeline import TrainPipeline

pipeline = TrainPipeline()
accuracy = pipeline.run_pipeline(
    data_path="data.csv",
    target_column="target",
    test_size=0.2
)
```

### Prediction Pipeline (`pipeline/predict_pipeline.py`)
Handles predictions on new data:
- Loads trained model and preprocessor
- Transforms input data
- Makes predictions

**Usage:**
```python
from src.pipeline import PredictPipeline, CustomData
import pandas as pd

# Method 1: Using DataFrame
pipeline = PredictPipeline()
predictions = pipeline.predict(df)

# Method 2: Using CustomData
custom_data = CustomData(feature1=1.0, feature2=2.0)
df = custom_data.get_data_as_dataframe()
predictions = pipeline.predict(df)
```

## 📝 Logger

Centralized logging system that:
- Creates timestamped log files in `logs/` directory
- Logs to both file and console
- Captures detailed information (timestamp, line number, level, message)

**Usage:**
```python
from src.logger import logger

logger.info("Information message")
logger.warning("Warning message")
logger.error("Error message")
```

## ⚠️ Exception Handling

Custom exception classes for better error tracking:
- `CustomException` - Base exception with detailed error info
- `DataValidationError` - Data validation errors
- `ModelTrainingError` - Model training errors
- `DataPreprocessingError` - Preprocessing errors
- `FeatureEngineeringError` - Feature engineering errors
- `ModelLoadingError` - Model loading errors
- `PredictionError` - Prediction errors

**Usage:**
```python
from src.exception import CustomException, DataValidationError
import sys

try:
    # Your code
    pass
except Exception as e:
    raise CustomException(str(e), sys)
```

## 🛠️ Utilities

Helper functions in `utils.py`:
- `save_model()` - Save trained models
- `load_model()` - Load trained models
- `load_dataset()` - Load datasets from CSV
- `split_features_target()` - Split features and target
- `get_class_distribution()` - Get class distribution stats
- `check_class_imbalance()` - Check for class imbalance

## 📦 Artifacts

All trained models, preprocessors, and data splits are saved to the `artifacts/` directory:
```
artifacts/
├── raw.csv              # Raw data
├── train.csv           # Training data
├── test.csv            # Test data
├── preprocessor.pkl    # Fitted preprocessor
└── model.pkl           # Trained model
```

## 🚫 Git Ignore

The following are automatically ignored by git:
- `logs/` directory (created by logger)
- `artifacts/` directory (created during training)
- Python cache files
- Virtual environments

## 🔄 Migration from Legacy Code

The legacy modules (`data_preprocessing.py`, `feature_engineering.py`, `model.py`) are still available for backward compatibility. New development should use the component-based architecture.

