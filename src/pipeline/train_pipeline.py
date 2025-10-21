"""
Training Pipeline

End-to-end training pipeline orchestrating all components.
"""

import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logger


class TrainPipeline:
    """
    Training pipeline that orchestrates data ingestion, transformation, and model training.
    """
    
    def __init__(self):
        """
        Initialize TrainPipeline.
        """
        logger.info("TrainPipeline initialized")
    
    def run_pipeline(self, data_path: str, target_column: str, test_size: float = 0.2):
        """
        Execute the complete training pipeline.
        
        Parameters:
        -----------
        data_path : str
            Path to the source data file
        target_column : str
            Name of the target column
        test_size : float
            Proportion of data to use for testing (default: 0.2)
            
        Returns:
        --------
        float
            Best model accuracy score
        """
        try:
            logger.info("=" * 70)
            logger.info("Training Pipeline Started")
            logger.info("=" * 70)
            
            # Step 1: Data Ingestion
            logger.info("Step 1: Data Ingestion")
            data_ingestion = DataIngestion()
            train_data_path, test_data_path = data_ingestion.initiate_data_ingestion(
                data_path=data_path,
                test_size=test_size
            )
            
            # Step 2: Data Transformation
            logger.info("Step 2: Data Transformation")
            data_transformation = DataTransformation()
            train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(
                train_path=train_data_path,
                test_path=test_data_path,
                target_column=target_column
            )
            
            # Step 3: Model Training
            logger.info("Step 3: Model Training")
            model_trainer = ModelTrainer()
            model_score = model_trainer.initiate_model_trainer(
                train_array=train_arr,
                test_array=test_arr
            )
            
            logger.info("=" * 70)
            logger.info(f"Training Pipeline Completed Successfully!")
            logger.info(f"Best Model Accuracy: {model_score:.4f}")
            logger.info("=" * 70)
            
            return model_score
            
        except Exception as e:
            logger.error("Error in training pipeline")
            raise CustomException(str(e), sys)


if __name__ == "__main__":
    # Example usage
    try:
        # Initialize pipeline
        pipeline = TrainPipeline()
        
        # Run training pipeline
        # Adjust these parameters according to your data
        accuracy = pipeline.run_pipeline(
            data_path="space_decay.csv",  # Update with your data path
            target_column="Decay",  # Update with your target column name
            test_size=0.2
        )
        
        print(f"\nTraining completed! Best model accuracy: {accuracy:.4f}")
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}")
        print(f"Error: {str(e)}")

