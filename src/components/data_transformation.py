"""
Data Transformation Component

Handles data preprocessing, feature engineering, and transformations.
"""

import sys
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from src.exception import CustomException, DataPreprocessingError
from src.logger import logger
from src.utils import save_model


@dataclass
class DataTransformationConfig:
    """
    Configuration for data transformation.
    """
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')
    scaler_obj_file_path: str = os.path.join('artifacts', 'scaler.pkl')


class DataTransformation:
    """
    Data Transformation component for preprocessing and feature engineering.
    """
    
    def __init__(self):
        """
        Initialize DataTransformation with configuration.
        """
        self.data_transformation_config = DataTransformationConfig()
        logger.info("DataTransformation component initialized")
    
    def get_data_transformer_object(self, numerical_columns, categorical_columns=None):
        """
        Create preprocessing pipeline for numerical and categorical features.
        
        Parameters:
        -----------
        numerical_columns : list
            List of numerical column names
        categorical_columns : list, optional
            List of categorical column names
            
        Returns:
        --------
        ColumnTransformer
            Preprocessing pipeline object
        """
        try:
            logger.info("Creating data transformation pipeline")
            
            # Numerical pipeline
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )
            
            transformers = [
                ("num_pipeline", num_pipeline, numerical_columns)
            ]
            
            # Add categorical pipeline if categorical columns exist
            if categorical_columns:
                cat_pipeline = Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                    ]
                )
                transformers.append(("cat_pipeline", cat_pipeline, categorical_columns))
            
            preprocessor = ColumnTransformer(transformers)
            
            logger.info("Data transformation pipeline created successfully")
            return preprocessor
            
        except Exception as e:
            logger.error(f"Error creating data transformer: {str(e)}")
            raise CustomException(str(e), sys)
    
    def initiate_data_transformation(self, train_path, test_path, target_column):
        """
        Apply data transformation to train and test data.
        
        Parameters:
        -----------
        train_path : str
            Path to training data
        test_path : str
            Path to test data
        target_column : str
            Name of the target column
            
        Returns:
        --------
        tuple
            (train_arr, test_arr, preprocessor_path)
        """
        try:
            logger.info("Starting data transformation")
            
            # Read train and test data
            logger.info(f"Reading train data from {train_path}")
            train_df = pd.read_csv(train_path)
            
            logger.info(f"Reading test data from {test_path}")
            test_df = pd.read_csv(test_path)
            
            logger.info(f"Train data shape: {train_df.shape}")
            logger.info(f"Test data shape: {test_df.shape}")
            
            # Separate features and target
            logger.info("Separating features and target variable")
            
            input_feature_train_df = train_df.drop(columns=[target_column], axis=1)
            target_feature_train_df = train_df[target_column]
            
            input_feature_test_df = test_df.drop(columns=[target_column], axis=1)
            target_feature_test_df = test_df[target_column]
            
            # Get numerical columns
            numerical_columns = input_feature_train_df.select_dtypes(
                include=['int64', 'float64']
            ).columns.tolist()
            
            # Get categorical columns
            categorical_columns = input_feature_train_df.select_dtypes(
                include=['object']
            ).columns.tolist()
            
            logger.info(f"Numerical columns: {numerical_columns}")
            logger.info(f"Categorical columns: {categorical_columns}")
            
            # Create and apply preprocessing object
            logger.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object(
                numerical_columns=numerical_columns,
                categorical_columns=categorical_columns if categorical_columns else None
            )
            
            logger.info("Applying preprocessing on training dataframe")
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            
            logger.info("Applying preprocessing on test dataframe")
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            
            # Combine features and target
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]
            
            logger.info(f"Transformed train array shape: {train_arr.shape}")
            logger.info(f"Transformed test array shape: {test_arr.shape}")
            
            # Save preprocessing object
            logger.info("Saving preprocessing object")
            save_model(
                model=preprocessing_obj,
                filepath=self.data_transformation_config.preprocessor_obj_file_path
            )
            
            logger.info("Data transformation completed successfully")
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
            
        except Exception as e:
            logger.error(f"Error in data transformation: {str(e)}")
            raise DataPreprocessingError(str(e), sys)

