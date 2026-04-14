import numpy as np
from typing import List
from core.models import RiskFeature

# We expect the agent to find these specific features for our V1 model
EXPECTED_FEATURES = [
    "Market Volatility Index",
    "Internal Liquidity Ratio",
    "Sentiment Negative Keyword Frequency"
]

def vectorize_features(features: List[RiskFeature]) -> np.ndarray:
    """
    Maps dynamic agent findings into a strict 1D numpy array for the ML model.
    Applies default values if the agent failed to find a specific feature.
    """
    feature_dict = {f.feature_name: f.value for f in features}
    
    vector = []
    for expected in EXPECTED_FEATURES:
        # If the agent found it, use it. Otherwise, assume a baseline/safe mean.
        val = feature_dict.get(expected, get_baseline_mean(expected))
        vector.append(val)
        
    return np.array(vector).reshape(1, -1)

def get_baseline_mean(feature_name: str) -> float:
    """Provides a neutral fallback value if the Researcher agent missed a data point."""
    baselines = {
        "Market Volatility Index": 0.3,
        "Internal Liquidity Ratio": 1.2,
        "Sentiment Negative Keyword Frequency": 10.0
    }
    return baselines.get(feature_name, 0.0)