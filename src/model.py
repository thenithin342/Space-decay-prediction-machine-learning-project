"""
Model Training and Prediction Module

Handles model training, evaluation, and prediction.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score
)

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False


class SpaceObjectClassifier:
    """
    Classifier for space objects (DEBRIS, PAYLOAD, ROCKET BODY, TBA).
    
    Supports:
    - Multiple algorithms (Random Forest, Logistic Regression)
    - Class imbalance handling (SMOTE)
    - Cross-validation
    - Comprehensive evaluation
    """
    
    def __init__(self, model_type='random_forest', random_state=42):
        """
        Initialize classifier.
        
        Parameters:
        -----------
        model_type : str
            'random_forest' or 'logistic_regression'
        random_state : int
            Random seed for reproducibility
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.classes_ = None
        
        self._init_model()
    
    def _init_model(self):
        """Initialize the model based on model_type."""
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=self.random_state,
                n_jobs=-1
            )
        elif self.model_type == 'logistic_regression':
            self.model = LogisticRegression(
                max_iter=1000,
                random_state=self.random_state
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, X_train, y_train, use_smote=False):
        """
        Train the model.
        
        Parameters:
        -----------
        X_train : np.array or pd.DataFrame
            Training features
        y_train : np.array or pd.Series
            Training labels
        use_smote : bool
            Whether to apply SMOTE for class imbalance
            
        Returns:
        --------
        self
        """
        # Apply SMOTE if requested and available
        if use_smote and SMOTE_AVAILABLE:
            print("Applying SMOTE for class balancing...")
            smote = SMOTE(random_state=self.random_state)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            print(f"Training set after SMOTE: {X_train.shape[0]} samples")
        elif use_smote and not SMOTE_AVAILABLE:
            print("Warning: SMOTE not available. Install with: pip install imbalanced-learn")
        
        # Train model
        print(f"Training {self.model_type}...")
        self.model.fit(X_train, y_train)
        self.classes_ = self.model.classes_
        print("Training complete!")
        
        return self
    
    def predict(self, X):
        """
        Make predictions.
        
        Parameters:
        -----------
        X : np.array or pd.DataFrame
            Features to predict
            
        Returns:
        --------
        np.array
            Predictions
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Parameters:
        -----------
        X : np.array or pd.DataFrame
            Features to predict
            
        Returns:
        --------
        np.array
            Class probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance.
        
        Parameters:
        -----------
        X_test : np.array or pd.DataFrame
            Test features
        y_test : np.array or pd.Series
            Test labels
            
        Returns:
        --------
        dict
            Dictionary with evaluation metrics
        """
        y_pred = self.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted')
        }
        
        print("=" * 50)
        print(f"MODEL EVALUATION - {self.model_type.upper()}")
        print("=" * 50)
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score:  {metrics['f1_score']:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return metrics
    
    def cross_validate(self, X, y, cv=5):
        """
        Perform cross-validation.
        
        Parameters:
        -----------
        X : np.array or pd.DataFrame
            Features
        y : np.array or pd.Series
            Labels
        cv : int
            Number of folds
            
        Returns:
        --------
        dict
            Cross-validation results
        """
        print(f"Performing {cv}-fold cross-validation...")
        
        cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        scores = cross_val_score(self.model, X, y, cv=cv_strategy, scoring='accuracy', n_jobs=-1)
        
        results = {
            'mean_score': scores.mean(),
            'std_score': scores.std(),
            'scores': scores
        }
        
        print(f"CV Accuracy: {results['mean_score']:.4f} (+/- {results['std_score']:.4f})")
        
        return results
    
    def get_feature_importance(self, feature_names=None):
        """
        Get feature importance (for Random Forest).
        
        Parameters:
        -----------
        feature_names : list
            Names of features
            
        Returns:
        --------
        pd.DataFrame
            Feature importance scores
        """
        if self.model_type != 'random_forest':
            print("Feature importance only available for Random Forest")
            return None
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(self.model.feature_importances_))]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df

