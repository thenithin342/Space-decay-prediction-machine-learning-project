"""
Example Usage of Space Decay Prediction Package

This script demonstrates how to use the package for:
1. Loading and preprocessing data
2. Feature engineering
3. Model training and evaluation
4. Making predictions
"""

import pandas as pd
from sklearn.model_selection import train_test_split

# Import from our package
from src.data_preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.model import SpaceObjectClassifier
from src.utils import (
    load_dataset, save_model, load_model,
    check_class_imbalance, print_dataset_info
)


def main():
    """Main execution function."""
    
    print("=" * 70)
    print("SPACE DECAY PREDICTION - EXAMPLE USAGE")
    print("=" * 70)
    
    # ==================== STEP 1: Load Data ====================
    print("\n[STEP 1] Loading dataset...")
    df = load_dataset('space_decay.csv')
    print_dataset_info(df)
    
    # ==================== STEP 2: Preprocess Data ====================
    print("\n[STEP 2] Preprocessing data...")
    preprocessor = DataPreprocessor()
    df_clean = preprocessor.preprocess_pipeline(df, target_column='OBJECT_TYPE')
    
    # ==================== STEP 3: Feature Engineering ====================
    print("\n[STEP 3] Feature engineering...")
    engineer = FeatureEngineer()
    
    # Create orbital categories
    df_engineered = engineer.create_orbit_categories(df_clean)
    
    # Select features
    feature_columns = engineer.select_features(df_engineered)
    
    # ==================== STEP 4: Prepare Data for Modeling ====================
    print("\n[STEP 4] Preparing data for modeling...")
    
    # Define features and target
    if 'OBJECT_TYPE_encoded' in df_engineered.columns:
        target_col = 'OBJECT_TYPE_encoded'
    elif 'OBJECT_TYPE' in df_engineered.columns:
        target_col = 'OBJECT_TYPE'
    else:
        raise ValueError("Target column not found")
    
    # Get feature columns (exclude target)
    X = df_engineered.drop(columns=[col for col in df_engineered.columns if 'OBJECT_TYPE' in col])
    X = X.select_dtypes(include=['int64', 'float64'])
    y = df_engineered[target_col]
    
    # Check for class imbalance
    imbalance_info = check_class_imbalance(y)
    print(f"\nClass Imbalance Check:")
    print(f"  Imbalance Ratio: {imbalance_info['imbalance_ratio']:.2f}")
    print(f"  Recommendation: {imbalance_info['recommendation']}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nData Split:")
    print(f"  Training: {X_train.shape[0]} samples")
    print(f"  Test: {X_test.shape[0]} samples")
    
    # Scale features
    X_train_scaled, X_test_scaled = preprocessor.scale_features(X_train, X_test=X_test)
    print(f"  Features scaled: {X_train.shape[1]} features")
    
    # ==================== STEP 5: Train Model ====================
    print("\n[STEP 5] Training model...")
    
    # Initialize classifier
    classifier = SpaceObjectClassifier(model_type='random_forest', random_state=42)
    
    # Train with SMOTE if imbalanced
    use_smote = imbalance_info['is_imbalanced']
    classifier.train(X_train_scaled, y_train, use_smote=use_smote)
    
    # ==================== STEP 6: Evaluate Model ====================
    print("\n[STEP 6] Evaluating model...")
    metrics = classifier.evaluate(X_test_scaled, y_test)
    
    # Cross-validation
    print("\n[STEP 6.1] Cross-validation...")
    cv_results = classifier.cross_validate(X_train_scaled, y_train, cv=5)
    
    # Feature importance
    print("\n[STEP 6.2] Feature importance...")
    importance = classifier.get_feature_importance(feature_names=X_train.columns.tolist())
    if importance is not None:
        print("\nTop 10 Important Features:")
        print(importance.head(10).to_string(index=False))
    
    # ==================== STEP 7: Save Model ====================
    print("\n[STEP 7] Saving model and scaler...")
    save_model(classifier, 'best_model.pkl')
    save_model(preprocessor.scaler, 'scaler.pkl')
    
    # ==================== STEP 8: Make Predictions ====================
    print("\n[STEP 8] Making predictions on new data...")
    
    # Load saved model
    loaded_classifier = load_model('best_model.pkl')
    loaded_scaler = load_model('scaler.pkl')
    
    # Example prediction
    sample = X_test.iloc[0:1]
    sample_scaled = loaded_scaler.transform(sample)
    prediction = loaded_classifier.predict(sample_scaled)
    prediction_proba = loaded_classifier.predict_proba(sample_scaled)
    
    print(f"\nExample Prediction:")
    print(f"  Predicted Class: {prediction[0]}")
    print(f"  Actual Class: {y_test.iloc[0]}")
    print(f"  Prediction Probabilities: {prediction_proba[0]}")
    
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()

