"""
Custom Exception Handling Module

Provides custom exception classes and error handling utilities.
"""

import sys
from typing import Optional


def error_message_detail(error: Exception, error_detail: sys) -> str:
    """
    Generate detailed error message with file name and line number.
    
    Parameters:
    -----------
    error : Exception
        The exception that was raised
    error_detail : sys
        System module to extract exception info
        
    Returns:
    --------
    str
        Formatted error message with details
    """
    _, _, exc_tb = error_detail.exc_info()
    
    if exc_tb is not None:
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
        error_message = (
            f"Error occurred in python script name [{file_name}] "
            f"line number [{line_number}] error message [{str(error)}]"
        )
    else:
        error_message = f"Error occurred: {str(error)}"
    
    return error_message


class CustomException(Exception):
    """
    Custom exception class for the project.
    
    This exception captures detailed information about where the error occurred,
    including the file name, line number, and error message.
    """
    
    def __init__(self, error_message: str, error_detail: Optional[sys] = None):
        """
        Initialize custom exception.
        
        Parameters:
        -----------
        error_message : str
            The error message
        error_detail : sys, optional
            System module to extract exception info
        """
        super().__init__(error_message)
        
        if error_detail is None:
            error_detail = sys
        
        self.error_message = error_message_detail(
            error=Exception(error_message),
            error_detail=error_detail
        )
    
    def __str__(self) -> str:
        """
        String representation of the exception.
        
        Returns:
        --------
        str
            Detailed error message
        """
        return self.error_message


class DataValidationError(CustomException):
    """
    Exception raised for data validation errors.
    """
    pass


class ModelTrainingError(CustomException):
    """
    Exception raised for model training errors.
    """
    pass


class DataPreprocessingError(CustomException):
    """
    Exception raised for data preprocessing errors.
    """
    pass


class FeatureEngineeringError(CustomException):
    """
    Exception raised for feature engineering errors.
    """
    pass


class ModelLoadingError(CustomException):
    """
    Exception raised for model loading errors.
    """
    pass


class PredictionError(CustomException):
    """
    Exception raised for prediction errors.
    """
    pass

