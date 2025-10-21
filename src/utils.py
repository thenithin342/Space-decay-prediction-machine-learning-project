"""
Utility Functions Module

Helper functions for model persistence and data handling.
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path


def save_model(model, filepath):
    """
    Save a trained model to disk.
    
    Parameters:
    -----------
    model : object
        Trained model object
    filepath : str or Path
        Path to save the model
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model saved to: {filepath}")


def load_model(filepath):
    """
    Load a trained model from disk.
    
    Parameters:
    -----------
    filepath : str or Path
        Path to the saved model
        
    Returns:
    --------
    object
        Loaded model
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")
    
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    
    print(f"Model loaded from: {filepath}")
    return model


def save_scaler(scaler, filepath):
    """
    Save a fitted scaler to disk.
    
    Parameters:
    -----------
    scaler : object
        Fitted scaler object
    filepath : str or Path
        Path to save the scaler
    """
    save_model(scaler, filepath)


def load_scaler(filepath):
    """
    Load a fitted scaler from disk.
    
    Parameters:
    -----------
    filepath : str or Path
        Path to the saved scaler
        
    Returns:
    --------
    object
        Loaded scaler
    """
    return load_model(filepath)


def load_dataset(filepath, **kwargs):
    """
    Load dataset from CSV file.
    
    Parameters:
    -----------
    filepath : str or Path
        Path to the dataset
    **kwargs : dict
        Additional arguments for pd.read_csv
        
    Returns:
    --------
    pd.DataFrame
        Loaded dataset
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset not found: {filepath}")
    
    df = pd.read_csv(filepath, **kwargs)
    print(f"Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    
    return df


def split_features_target(df, target_column):
    """
    Split dataframe into features and target.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    target_column : str
        Name of target column
        
    Returns:
    --------
    tuple
        (X, y) features and target
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataframe")
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    return X, y


def get_class_distribution(y):
    """
    Get class distribution statistics.
    
    Parameters:
    -----------
    y : pd.Series or np.array
        Target labels
        
    Returns:
    --------
    pd.DataFrame
        Class distribution statistics
    """
    if isinstance(y, np.ndarray):
        y = pd.Series(y)
    
    counts = y.value_counts().sort_index()
    percentages = (counts / len(y) * 100).round(2)
    
    dist_df = pd.DataFrame({
        'Class': counts.index,
        'Count': counts.values,
        'Percentage': percentages.values
    })
    
    return dist_df


def check_class_imbalance(y, threshold=3.0):
    """
    Check for class imbalance in target variable.
    
    Parameters:
    -----------
    y : pd.Series or np.array
        Target labels
    threshold : float
        Imbalance ratio threshold
        
    Returns:
    --------
    dict
        Imbalance statistics and recommendation
    """
    if isinstance(y, np.ndarray):
        y = pd.Series(y)
    
    counts = y.value_counts()
    max_count = counts.max()
    min_count = counts.min()
    imbalance_ratio = max_count / min_count
    
    is_imbalanced = imbalance_ratio > threshold
    
    result = {
        'imbalance_ratio': imbalance_ratio,
        'is_imbalanced': is_imbalanced,
        'max_class': counts.idxmax(),
        'min_class': counts.idxmin(),
        'recommendation': 'Use SMOTE or class weights' if is_imbalanced else 'Classes are balanced'
    }
    
    return result


def print_dataset_info(df):
    """
    Print comprehensive dataset information.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    """
    print("=" * 60)
    print("DATASET INFORMATION")
    print("=" * 60)
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"\nData Types:")
    print(df.dtypes.value_counts())
    print(f"\nMemory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"\nMissing Values:")
        print(missing[missing > 0].sort_values(ascending=False))
    else:
        print(f"\nMissing Values: None")
    
    duplicates = df.duplicated().sum()
    print(f"Duplicate Rows: {duplicates}")
    print("=" * 60)

