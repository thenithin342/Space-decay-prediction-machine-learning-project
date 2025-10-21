"""
Data Preprocessing Module

Handles data cleaning, missing value imputation, and outlier detection.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


class DataPreprocessor:
    """
    Class for preprocessing space debris dataset.
    
    Handles:
    - Missing value imputation
    - Duplicate removal
    - Outlier detection
    - Feature encoding
    - Feature scaling
    """
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = None
        
    def handle_missing_values(self, df, strategy='auto'):
        """
        Handle missing values in the dataset.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        strategy : str
            'auto', 'drop', or 'fill'
            
        Returns:
        --------
        pd.DataFrame
            Cleaned dataframe
        """
        df_clean = df.copy()
        
        # Drop columns with >70% missing values
        if strategy in ['auto', 'drop']:
            threshold = 0.7
            missing_percent = df_clean.isnull().sum() / len(df_clean)
            cols_to_drop = missing_percent[missing_percent > threshold].index.tolist()
            if cols_to_drop:
                print(f"Dropping columns with >{threshold*100}% missing: {cols_to_drop}")
                df_clean = df_clean.drop(columns=cols_to_drop)
        
        # Fill missing values
        if strategy in ['auto', 'fill']:
            # Numerical columns: fill with median
            numerical_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns
            for col in numerical_cols:
                if df_clean[col].isnull().sum() > 0:
                    df_clean[col].fillna(df_clean[col].median(), inplace=True)
            
            # Categorical columns: fill with mode
            categorical_cols = df_clean.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if df_clean[col].isnull().sum() > 0:
                    mode_value = df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown'
                    df_clean[col].fillna(mode_value, inplace=True)
        
        return df_clean
    
    def remove_duplicates(self, df):
        """Remove duplicate rows from dataframe."""
        initial_shape = df.shape
        df_clean = df.drop_duplicates()
        removed = initial_shape[0] - df_clean.shape[0]
        if removed > 0:
            print(f"Removed {removed} duplicate rows")
        return df_clean
    
    def detect_outliers(self, df, columns=None, method='iqr'):
        """
        Detect outliers using IQR method.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        columns : list
            Columns to check for outliers
        method : str
            'iqr' or 'zscore'
            
        Returns:
        --------
        dict
            Dictionary with outlier counts per column
        """
        if columns is None:
            columns = df.select_dtypes(include=['int64', 'float64']).columns[:10]
        
        outlier_summary = {}
        
        for col in columns:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
                outlier_summary[col] = len(outliers)
        
        return outlier_summary
    
    def encode_categorical_features(self, df, columns=None, target_column=None):
        """
        Encode categorical features using LabelEncoder.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        columns : list
            Columns to encode (if None, encodes all object columns)
        target_column : str
            Target column to exclude from encoding
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with encoded features
        """
        df_encoded = df.copy()
        
        if columns is None:
            columns = df_encoded.select_dtypes(include=['object']).columns
        
        for col in columns:
            if col != target_column:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                self.label_encoders[col] = le
        
        return df_encoded
    
    def scale_features(self, X_train, X_val=None, X_test=None):
        """
        Scale features using StandardScaler.
        
        Parameters:
        -----------
        X_train : pd.DataFrame or np.array
            Training features
        X_val : pd.DataFrame or np.array
            Validation features
        X_test : pd.DataFrame or np.array
            Test features
            
        Returns:
        --------
        tuple
            Scaled train, validation, and test sets
        """
        # Fit scaler on training data only
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        results = [X_train_scaled]
        
        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            results.append(X_val_scaled)
        
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            results.append(X_test_scaled)
        
        return tuple(results) if len(results) > 1 else results[0]
    
    def preprocess_pipeline(self, df, target_column=None):
        """
        Complete preprocessing pipeline.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        target_column : str
            Name of target column
            
        Returns:
        --------
        pd.DataFrame
            Preprocessed dataframe
        """
        print("Starting preprocessing pipeline...")
        
        # Handle missing values
        print("Step 1: Handling missing values...")
        df = self.handle_missing_values(df)
        
        # Remove duplicates
        print("Step 2: Removing duplicates...")
        df = self.remove_duplicates(df)
        
        # Encode categorical features
        print("Step 3: Encoding categorical features...")
        df = self.encode_categorical_features(df, target_column=target_column)
        
        print("Preprocessing complete!")
        return df

