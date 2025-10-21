"""
Data Ingestion Component

Handles data loading, validation, and initial splitting.
"""

import os
import sys
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.exception import CustomException, DataValidationError
from src.logger import logger


@dataclass
class DataIngestionConfig:
    """
    Configuration for data ingestion.
    """
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'raw.csv')


class DataIngestion:
    """
    Data Ingestion component for loading and splitting data.
    """
    
    def __init__(self):
        """
        Initialize DataIngestion with configuration.
        """
        self.ingestion_config = DataIngestionConfig()
        logger.info("DataIngestion component initialized")
    
    def initiate_data_ingestion(self, data_path: str, test_size: float = 0.2):
        """
        Read data from source, split into train and test, and save to artifacts.
        
        Parameters:
        -----------
        data_path : str
            Path to the source data file
        test_size : float
            Proportion of data to use for testing (default: 0.2)
            
        Returns:
        --------
        tuple
            (train_data_path, test_data_path)
        """
        logger.info("Entered the data ingestion method or component")
        
        try:
            # Read the dataset
            logger.info(f"Reading dataset from {data_path}")
            df = pd.read_csv(data_path)
            logger.info(f"Dataset loaded successfully. Shape: {df.shape}")
            
            # Create artifacts directory
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            logger.info("Artifacts directory created")
            
            # Save raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logger.info(f"Raw data saved to {self.ingestion_config.raw_data_path}")
            
            # Validate data
            if df.empty:
                raise DataValidationError("Dataset is empty", sys)
            
            logger.info("Initiating train-test split")
            train_set, test_set = train_test_split(df, test_size=test_size, random_state=42)
            
            # Save train and test sets
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            
            logger.info(f"Train set saved to {self.ingestion_config.train_data_path}")
            logger.info(f"Test set saved to {self.ingestion_config.test_data_path}")
            logger.info(f"Train set shape: {train_set.shape}, Test set shape: {test_set.shape}")
            
            logger.info("Data ingestion completed successfully")
            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
            
        except FileNotFoundError as e:
            logger.error(f"Data file not found: {str(e)}")
            raise CustomException(f"Data file not found: {str(e)}", sys)
            
        except Exception as e:
            logger.error(f"Error in data ingestion: {str(e)}")
            raise CustomException(str(e), sys)

