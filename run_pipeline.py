"""
Complete Pipeline Runner
Executes data ingestion and notebook analyses
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.components.data_ingestion import DataIngestion
from src.logger import logger

def main():
    """
    Run the complete data science pipeline
    """
    try:
        # Step 1: Data Ingestion
        logger.info("=" * 80)
        logger.info("STEP 1: DATA INGESTION")
        logger.info("=" * 80)
        
        data_ingestion = DataIngestion()
        data_path = "space_decay.csv"  # Update this path if needed
        
        if not os.path.exists(data_path):
            # Try alternative path in notebook/data
            data_path = "notebook/data/space_decay.csv"
        
        train_path, test_path = data_ingestion.initiate_data_ingestion(data_path)
        
        logger.info(f"✓ Data ingestion completed successfully!")
        logger.info(f"  Train data: {train_path}")
        logger.info(f"  Test data: {test_path}")
        
        # Step 2: Run EDA Notebook (instructions)
        logger.info("\n" + "=" * 80)
        logger.info("STEP 2: EXPLORATORY DATA ANALYSIS")
        logger.info("=" * 80)
        logger.info("Please run the following notebook:")
        logger.info("  notebook/1_EDA_Analysis.ipynb")
        logger.info("\nTo run in Jupyter:")
        logger.info("  cd notebook")
        logger.info("  jupyter notebook 1_EDA_Analysis.ipynb")
        
        # Step 3: Run Model Training Notebook (instructions)
        logger.info("\n" + "=" * 80)
        logger.info("STEP 3: MODEL TRAINING")
        logger.info("=" * 80)
        logger.info("After completing EDA, run:")
        logger.info("  notebook/2_Model_Training.ipynb")
        
        logger.info("\n" + "=" * 80)
        logger.info("PIPELINE SETUP COMPLETED!")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()

