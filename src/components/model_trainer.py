"""
Model Trainer Component

Handles model training, evaluation, and hyperparameter tuning.
"""

import os
import sys
from dataclasses import dataclass
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException, ModelTrainingError
from src.logger import logger
from src.utils import save_model


@dataclass
class ModelTrainerConfig:
    """
    Configuration for model training.
    """
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    """
    Model Trainer component for training and evaluating ML models.
    """
    
    def __init__(self):
        """
        Initialize ModelTrainer with configuration.
        """
        self.model_trainer_config = ModelTrainerConfig()
        logger.info("ModelTrainer component initialized")
    
    def evaluate_models(self, X_train, y_train, X_test, y_test, models, params=None):
        """
        Evaluate multiple models and return their scores.
        
        Parameters:
        -----------
        X_train : array-like
            Training features
        y_train : array-like
            Training target
        X_test : array-like
            Test features
        y_test : array-like
            Test target
        models : dict
            Dictionary of model name and model object
        params : dict, optional
            Dictionary of model name and hyperparameters for grid search
            
        Returns:
        --------
        dict
            Dictionary of model name and test accuracy
        """
        try:
            logger.info("Starting model evaluation")
            report = {}
            
            for model_name, model in models.items():
                logger.info(f"Training {model_name}")
                
                # Hyperparameter tuning if params provided
                if params and model_name in params:
                    logger.info(f"Performing GridSearchCV for {model_name}")
                    gs = GridSearchCV(model, params[model_name], cv=3, n_jobs=-1, verbose=1)
                    gs.fit(X_train, y_train)
                    model = gs.best_estimator_
                    logger.info(f"Best parameters for {model_name}: {gs.best_params_}")
                else:
                    model.fit(X_train, y_train)
                
                # Predictions
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)
                
                # Evaluate
                train_accuracy = accuracy_score(y_train, y_train_pred)
                test_accuracy = accuracy_score(y_test, y_test_pred)
                
                report[model_name] = {
                    'model': model,
                    'train_accuracy': train_accuracy,
                    'test_accuracy': test_accuracy
                }
                
                logger.info(f"{model_name} - Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")
            
            logger.info("Model evaluation completed")
            return report
            
        except Exception as e:
            logger.error(f"Error in model evaluation: {str(e)}")
            raise ModelTrainingError(str(e), sys)
    
    def initiate_model_trainer(self, train_array, test_array):
        """
        Train models and select the best one.
        
        Parameters:
        -----------
        train_array : np.array
            Training data with features and target
        test_array : np.array
            Test data with features and target
            
        Returns:
        --------
        float
            Best model test accuracy
        """
        try:
            logger.info("Starting model training process")
            logger.info("Splitting training and test input data")
            
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]
            
            logger.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
            logger.info(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
            
            # Define models
            models = {
                "Random Forest": RandomForestClassifier(random_state=42),
                "Gradient Boosting": GradientBoostingClassifier(random_state=42),
                "Decision Tree": DecisionTreeClassifier(random_state=42),
                "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
                "SVM": SVC(random_state=42)
            }
            
            # Define hyperparameters for tuning
            params = {
                "Random Forest": {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20],
                },
                "Gradient Boosting": {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                },
                "Decision Tree": {
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10]
                },
                "Logistic Regression": {
                    'C': [0.1, 1.0, 10.0],
                    'penalty': ['l2']
                },
                "SVM": {
                    'C': [0.1, 1.0, 10.0],
                    'kernel': ['linear', 'rbf']
                }
            }
            
            # Evaluate models
            logger.info("Evaluating multiple models")
            model_report = self.evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                params=params
            )
            
            # Get best model
            best_model_name = max(
                model_report,
                key=lambda x: model_report[x]['test_accuracy']
            )
            best_model = model_report[best_model_name]['model']
            best_model_score = model_report[best_model_name]['test_accuracy']
            
            logger.info(f"Best model: {best_model_name}")
            logger.info(f"Best model test accuracy: {best_model_score:.4f}")
            
            # Check if best model meets minimum threshold
            if best_model_score < 0.6:
                logger.warning(f"Best model accuracy {best_model_score:.4f} is below threshold")
                raise ModelTrainingError(
                    "No best model found with accuracy above 0.6",
                    sys
                )
            
            # Save best model
            logger.info("Saving best model")
            save_model(
                model=best_model,
                filepath=self.model_trainer_config.trained_model_file_path
            )
            
            # Generate classification report
            y_pred = best_model.predict(X_test)
            logger.info(f"Classification Report for {best_model_name}:")
            logger.info(f"\n{classification_report(y_test, y_pred)}")
            
            logger.info("Model training completed successfully")
            
            return best_model_score
            
        except Exception as e:
            logger.error(f"Error in model training: {str(e)}")
            raise ModelTrainingError(str(e), sys)

