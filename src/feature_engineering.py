"""
Feature Engineering Module

Creates new features and performs feature selection.
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA


class FeatureEngineer:
    """
    Class for feature engineering and selection.
    
    Handles:
    - Feature creation
    - Feature selection
    - PCA transformation
    - Domain-specific transformations
    """
    
    def __init__(self):
        self.pca = None
        self.selected_features = None
        
    def create_orbit_categories(self, df):
        """
        Create categorical features from orbital parameters.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with new categorical features
        """
        df_eng = df.copy()
        
        # Eccentricity category
        if 'ECCENTRICITY' in df_eng.columns:
            df_eng['orbit_eccentricity_category'] = pd.cut(
                df_eng['ECCENTRICITY'],
                bins=[0, 0.1, 0.4, 1.0],
                labels=['Circular', 'Elliptical', 'Highly_Elliptical']
            )
            df_eng['orbit_eccentricity_category'] = df_eng['orbit_eccentricity_category'].cat.codes
        
        # Altitude category
        if 'SEMIMAJOR_AXIS' in df_eng.columns:
            df_eng['altitude_category'] = pd.cut(
                df_eng['SEMIMAJOR_AXIS'],
                bins=[0, 8000, 20000, 42000, np.inf],
                labels=['LEO', 'MEO', 'GEO', 'Beyond_GEO']
            )
            df_eng['altitude_category'] = df_eng['altitude_category'].cat.codes
        
        return df_eng
    
    def select_features(self, df, feature_list=None):
        """
        Select relevant features for modeling.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        feature_list : list
            List of features to select (if None, uses default)
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with selected features
        """
        if feature_list is None:
            # Default feature list
            feature_list = [
                'MEAN_MOTION', 'ECCENTRICITY', 'INCLINATION', 
                'RA_OF_ASC_NODE', 'ARG_OF_PERICENTER', 'MEAN_ANOMALY',
                'BSTAR', 'MEAN_MOTION_DOT', 'SEMIMAJOR_AXIS', 
                'PERIOD', 'APOAPSIS', 'PERIAPSIS', 'REV_AT_EPOCH',
                'COUNTRY_CODE', 'CLASSIFICATION_TYPE'
            ]
        
        # Filter to existing columns
        available_features = [f for f in feature_list if f in df.columns]
        self.selected_features = available_features
        
        return df[available_features]
    
    def apply_pca(self, X, n_components=0.95, fit=True):
        """
        Apply Principal Component Analysis.
        
        Parameters:
        -----------
        X : np.array or pd.DataFrame
            Input features
        n_components : int or float
            Number of components or variance ratio
        fit : bool
            Whether to fit PCA (True) or transform only (False)
            
        Returns:
        --------
        np.array
            Transformed features
        """
        if fit:
            self.pca = PCA(n_components=n_components)
            X_transformed = self.pca.fit_transform(X)
            print(f"PCA: Reduced to {self.pca.n_components_} components")
            print(f"Explained variance: {self.pca.explained_variance_ratio_.sum():.2%}")
        else:
            if self.pca is None:
                raise ValueError("PCA not fitted. Set fit=True first.")
            X_transformed = self.pca.transform(X)
        
        return X_transformed
    
    def get_feature_importance_from_pca(self, feature_names):
        """
        Get feature contributions to principal components.
        
        Parameters:
        -----------
        feature_names : list
            List of original feature names
            
        Returns:
        --------
        pd.DataFrame
            Feature loadings for each PC
        """
        if self.pca is None:
            raise ValueError("PCA not fitted yet.")
        
        loadings = self.pca.components_.T * np.sqrt(self.pca.explained_variance_)
        loadings_df = pd.DataFrame(
            loadings,
            columns=[f'PC{i+1}' for i in range(self.pca.n_components_)],
            index=feature_names
        )
        
        return loadings_df
    
    def create_interaction_features(self, df, feature_pairs=None):
        """
        Create interaction features between pairs of features.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        feature_pairs : list of tuples
            Pairs of features to interact
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with interaction features
        """
        df_int = df.copy()
        
        if feature_pairs is None:
            # Default interactions for orbital mechanics
            feature_pairs = [
                ('MEAN_MOTION', 'SEMIMAJOR_AXIS'),
                ('ECCENTRICITY', 'PERIOD'),
            ]
        
        for feat1, feat2 in feature_pairs:
            if feat1 in df_int.columns and feat2 in df_int.columns:
                df_int[f'{feat1}_x_{feat2}'] = df_int[feat1] * df_int[feat2]
        
        return df_int

