from typing import TypedDict, Annotated, List, Optional
import operator
from core.models import RiskFeature, OptimizerPrediction

class AgenticMeshState(TypedDict):
    """
    The LangGraph State Dictionary. 
    This is the 'shared memory' passed between agents during execution.
    """
    # Original Request Data
    entity_name: str
    analysis_timeframe: str
    
    # Researcher State
    gathered_features: Annotated[List[RiskFeature], operator.add]
    research_complete: bool
    context_documents: List[str] # Raw text chunks from MCP/RAG
    
    # Optimizer State
    ml_prediction: Optional[OptimizerPrediction]
    prediction_confidence: float
    
    # Supervisor State
    current_agent: str # Tracks who currently has control ('researcher', 'optimizer', 'supervisor')
    refinement_cycles: int
    errors: Annotated[List[str], operator.add]
    mesh_status: str # 'initializing', 'researching', 'computing', 'negotiating', 'complete'