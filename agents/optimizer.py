from typing import Dict, Any
from core.state import AgenticMeshState
from core.models import OptimizerPrediction
from ml_engine.predictor import predict_risk_probability # Import our actual ML engine

async def optimizer_node(state: AgenticMeshState) -> Dict[str, Any]:
    features = state.get('gathered_features', [])
    if not features:
        return {"errors": ["No features provided to Optimizer."], "current_agent": "supervisor"}

    print(f"🧮 [Optimizer] Pushing {len(features)} features into local ML engine...")
    
    # Execute actual ML inference on the M4
    risk_prob, confidence, key_drivers = predict_risk_probability(features)
    
    prediction = OptimizerPrediction(
        probability_of_event=risk_prob,
        confidence_interval=0.05,
        key_drivers=key_drivers
    )
    
    print(f"📊 [Optimizer] Actual ML Risk: {risk_prob:.2f} | Confidence: {confidence:.2f}")

    return {
        "ml_prediction": prediction,
        "prediction_confidence": confidence,
        "current_agent": "supervisor"
    }