"""
Prediction Script

Simple script to make predictions using trained model.
"""

import pandas as pd
from src.utils import load_model


def predict_space_object(features_dict):
    """
    Predict space object type from orbital parameters.
    
    Parameters:
    -----------
    features_dict : dict
        Dictionary with feature values
        
    Returns:
    --------
    dict
        Prediction result with class and probabilities
    """
    # Load trained model and scaler
    model = load_model('best_model.pkl')
    scaler = load_model('scaler.pkl')
    
    # Convert to DataFrame
    df = pd.DataFrame([features_dict])
    
    # Scale features
    df_scaled = scaler.transform(df)
    
    # Predict
    prediction = model.predict(df_scaled)[0]
    probabilities = model.predict_proba(df_scaled)[0]
    
    # Class mapping
    class_names = {0: 'DEBRIS', 1: 'PAYLOAD', 2: 'ROCKET BODY', 3: 'TBA'}
    
    result = {
        'predicted_class': class_names.get(prediction, prediction),
        'class_code': prediction,
        'probabilities': {
            class_names.get(i, i): prob 
            for i, prob in enumerate(probabilities)
        }
    }
    
    return result


def main():
    """Example usage."""
    
    # Example orbital parameters
    sample_data = {
        'MEAN_MOTION': 12.96964393,
        'ECCENTRICITY': 0.0026771,
        'INCLINATION': 90.273,
        'RA_OF_ASC_NODE': 325.7206,
        'ARG_OF_PERICENTER': 182.0403,
        'MEAN_ANOMALY': 178.0613,
        'BSTAR': 0.0017732,
        'MEAN_MOTION_DOT': 5.19e-06,
        'SEMIMAJOR_AXIS': 7652.144,
        'PERIOD': 111.028,
        'APOAPSIS': 1294.495,
        'PERIAPSIS': 1253.524,
        'REV_AT_EPOCH': 36475,
        'COUNTRY_CODE': 50,  # Encoded value
        'CLASSIFICATION_TYPE': 0,  # Encoded value
        'orbit_eccentricity_category': 0,
        'altitude_category': 0
    }
    
    print("Making prediction for space object...")
    print(f"Orbital Parameters: {sample_data}")
    
    result = predict_space_object(sample_data)
    
    print("\n" + "=" * 50)
    print("PREDICTION RESULT")
    print("=" * 50)
    print(f"Predicted Object Type: {result['predicted_class']}")
    print(f"\nClass Probabilities:")
    for class_name, prob in result['probabilities'].items():
        print(f"  {class_name}: {prob:.4f} ({prob*100:.2f}%)")
    print("=" * 50)


if __name__ == "__main__":
    main()

