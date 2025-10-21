"""
Prediction Pipeline

Handles prediction on new data using trained models.
"""

import sys
import os
import pandas as pd
from src.exception import CustomException, PredictionError
from src.logger import logger
from src.utils import load_model


class PredictPipeline:
    """
    Prediction pipeline for making predictions on new data.
    """
    
    def __init__(self):
        """
        Initialize PredictPipeline.
        """
        logger.info("PredictPipeline initialized")
    
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
            
            # Load preprocessor and model
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            
            logger.info("Loading model and preprocessor")
            model = load_model(filepath=model_path)
            preprocessor = load_model(filepath=preprocessor_path)
            
            # Transform features
            logger.info("Transforming input features")
            data_scaled = preprocessor.transform(features)
            
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

