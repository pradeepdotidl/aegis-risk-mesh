from pydantic import BaseModel, Field
from typing import List, Optional

class RiskRequest(BaseModel):
    """Payload received from the frontend/user to initiate analysis."""
    entity_name: str = Field(..., description="The company, asset, or system to analyze.")
    analysis_timeframe: str = Field("30_days", description="Time horizon for the risk prediction.")
    specific_concerns: Optional[List[str]] = Field(default=[], description="User-defined focus areas (e.g., 'liquidity', 'supply chain').")

class RiskFeature(BaseModel):
    """A specific data point found by the Researcher agent."""
    feature_name: str
    value: float
    source: str
    reliability_score: float = Field(ge=0.0, le=1.0)

class OptimizerPrediction(BaseModel):
    """The raw output from the local ML model."""
    probability_of_event: float = Field(ge=0.0, le=1.0)
    confidence_interval: float
    key_drivers: List[str]

class FinalRiskAssessment(BaseModel):
    """The final payload sent via GraphQL/WebSockets to the client."""
    entity_name: str
    overall_risk_score: float = Field(ge=0.0, le=100.0)
    risk_category: str # e.g., "Critical", "Moderate", "Safe"
    optimizer_prediction: OptimizerPrediction
    researcher_summary: str
    refinement_cycles_used: int