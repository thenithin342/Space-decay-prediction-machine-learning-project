"""
Prediction Pipeline

Handles prediction on new data using trained models.
"""

import sys
import os
import shutil
import pandas as pd
from src.exception import CustomException, PredictionError
from src.logger import logger
from src.utils import load_model

# Determine project root directory (parent of src directory)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")
DEFAULT_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model.pkl")
DEFAULT_PREPROCESSOR_PATH = os.path.join(ARTIFACTS_DIR, "preprocessor.pkl")
FALLBACK_MODEL_PATH = os.path.join(PROJECT_ROOT, "notebook", "data", "best_model.pkl")
FALLBACK_PREPROCESSOR_PATH = os.path.join(PROJECT_ROOT, "notebook", "data", "scaler.pkl")


def ensure_model_artifact():
    """Ensure model and preprocessor artifacts exist, copying from fallback if needed."""
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    # If the primary model is missing but a fallback exists, copy it
    if not os.path.exists(DEFAULT_MODEL_PATH) and os.path.exists(FALLBACK_MODEL_PATH):
        shutil.copyfile(FALLBACK_MODEL_PATH, DEFAULT_MODEL_PATH)
        logger.info(f"Copied fallback model from {FALLBACK_MODEL_PATH} to {DEFAULT_MODEL_PATH}")
    # Ensure the matching preprocessor exists as well
    if not os.path.exists(DEFAULT_PREPROCESSOR_PATH) and os.path.exists(FALLBACK_PREPROCESSOR_PATH):
        shutil.copyfile(FALLBACK_PREPROCESSOR_PATH, DEFAULT_PREPROCESSOR_PATH)
        logger.info(f"Copied fallback preprocessor from {FALLBACK_PREPROCESSOR_PATH} to {DEFAULT_PREPROCESSOR_PATH}")


class PredictPipeline:
    """
    Prediction pipeline for making predictions on new data.
    """
    
    def __init__(self):
        """
        Initialize PredictPipeline.
        """
        logger.info("PredictPipeline initialized")
        # Ensure artifacts exist before trying to load them
        ensure_model_artifact()
    
    def predict(self, features):
        """
        Make predictions on input features.
        
        Parameters:
        -----------
        features : pd.DataFrame
            Input features for prediction
            
        Returns:
        --------
        np.array
            Predictions
        """
        try:
            logger.info("Starting prediction process")
            
            # Use absolute paths based on project root
            model_path = DEFAULT_MODEL_PATH
            preprocessor_path = DEFAULT_PREPROCESSOR_PATH
            
            logger.info(f"Loading model from: {model_path}")
            logger.info(f"Loading preprocessor from: {preprocessor_path}")
            model = load_model(filepath=model_path)
            preprocessor = load_model(filepath=preprocessor_path)
            
            # Transform features
            logger.info("Transforming input features")
            if not isinstance(features, pd.DataFrame):
                raise PredictionError("Features must be a pandas DataFrame", sys)

            # Coerce all columns to numeric where possible to avoid string-to-float errors; non-convertible
            # values become NaN and are handled by imputers in the preprocessor.
            try:
                features_numeric = features.apply(pd.to_numeric, errors="coerce")
            except Exception:
                features_numeric = features

            data_scaled = preprocessor.transform(features_numeric)
            
            # Make predictions
            logger.info("Making predictions")
            predictions = model.predict(data_scaled)
            
            logger.info("Prediction completed successfully")
            return predictions
            
        except FileNotFoundError as e:
            logger.error(f"Model or preprocessor file not found: {str(e)}")
            raise PredictionError(f"Required files not found: {str(e)}", sys)
            
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            raise PredictionError(str(e), sys)


class CustomData:
    """
    Custom data class for creating input data from individual features.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize CustomData with feature values.
        
        Parameters:
        -----------
        **kwargs : dict
            Feature names and their values
        """
        self.data = kwargs
        logger.info(f"CustomData initialized with {len(kwargs)} features")
    
    def get_data_as_dataframe(self):
        """
        Convert input data to DataFrame.
        
        Returns:
        --------
        pd.DataFrame
            Input data as DataFrame
        """
        try:
            logger.info("Converting custom data to DataFrame")
            df = pd.DataFrame([self.data])
            logger.info(f"DataFrame created with shape: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Error creating DataFrame: {str(e)}")
            raise CustomException(str(e), sys)


if __name__ == "__main__":
    # Example usage
    try:
        # Example 1: Direct prediction with DataFrame
        print("Example 1: Prediction with DataFrame")
        
        # Create sample data (adjust column names and values based on your features)
        sample_data = pd.DataFrame({
            'feature1': [1.0],
            'feature2': [2.0],
            'feature3': [3.0],
            # Add more features as needed
        })
        
        # Make prediction
        pipeline = PredictPipeline()
        predictions = pipeline.predict(sample_data)
        print(f"Predictions: {predictions}")
        
        # Example 2: Using CustomData class
        print("\nExample 2: Using CustomData class")
        
        custom_data = CustomData(
            feature1=1.0,
            feature2=2.0,
            feature3=3.0,
            # Add more features as needed
        )
        
        # Convert to DataFrame and predict
        input_df = custom_data.get_data_as_dataframe()
        predictions = pipeline.predict(input_df)
        print(f"Predictions: {predictions}")
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        print(f"Error: {str(e)}")

