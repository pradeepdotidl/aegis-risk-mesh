import os
import joblib
import numpy as np
from typing import Tuple, List
from core.models import RiskFeature
from ml_engine.preprocessor import vectorize_features, EXPECTED_FEATURES

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "rf_risk_v1.joblib")

# Keep the model loaded in memory for fast inference
_model_instance = None

def _get_model():
    global _model_instance
    if _model_instance is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"ML Model not found at {MODEL_PATH}. Run train_baseline.py first.")
        _model_instance = joblib.load(MODEL_PATH)
    return _model_instance

def predict_risk_probability(features: List[RiskFeature]) -> Tuple[float, float, List[str]]:
    """
    Called by the Optimizer Agent.
    Takes agent features, vectorizes them, and returns:
    (Probability of Risk Event, Model Confidence, Key Drivers)
    """
    model = _get_model()
    
    # 1. Preprocess
    X_input = vectorize_features(features)
    
    # 2. Predict Probability (returns array like [[prob_safe, prob_risk]])
    probabilities = model.predict_proba(X_input)[0]
    risk_probability = float(probabilities[1])
    
    # 3. Calculate Confidence based on feature reliability from the Researcher
    # If the agent found highly reliable data, our ML confidence goes up.
    if features:
        avg_data_reliability = sum(f.reliability_score for f in features) / len(features)
    else:
        avg_data_reliability = 0.1
        
    # Model confidence is a blend of the ML certainty (distance from 0.5) and data reliability
    ml_certainty = abs(risk_probability - 0.5) * 2 
    overall_confidence = (ml_certainty * 0.4) + (avg_data_reliability * 0.6)
    
    # 4. Extract Key Drivers (Which feature contributed most?)
    # Using feature importances from the Random Forest
    importances = model.feature_importances_
    driver_indices = np.argsort(importances)[::-1]
    key_drivers = [EXPECTED_FEATURES[i] for i in driver_indices[:2]] # Top 2 drivers
    
    return risk_probability, overall_confidence, key_drivers