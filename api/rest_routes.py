from fastapi import APIRouter, HTTPException
from core.models import RiskRequest, FinalRiskAssessment
from agents.supervisor import build_aegis_graph

print("🚀 DEBUG: rest_routes.py is being imported!")

router = APIRouter()
mesh = build_aegis_graph()

@router.post("/analyze", response_model=FinalRiskAssessment)
async def trigger_risk_analysis(request: RiskRequest):
    """
    Standard REST endpoint. Triggers the Agentic Mesh and waits for completion.
    """
    initial_state = {
        "entity_name": request.entity_name,
        "analysis_timeframe": request.analysis_timeframe,
        "gathered_features": [],
        "refinement_cycles": 0
    }
    
    try:
        # Execute the LangGraph workflow
        final_state = await mesh.ainvoke(initial_state)
        # Map the raw state back to our strict FinalRiskAssessment model
        assessment = FinalRiskAssessment(
            entity_name=final_state["entity_name"],
            overall_risk_score=final_state["ml_prediction"].probability_of_event * 100,
            risk_category="Critical" if final_state["ml_prediction"].probability_of_event > 0.7 else "Safe",
            optimizer_prediction=final_state["ml_prediction"],
            researcher_summary=f"Gathered {len(final_state['gathered_features'])} reliable features.",
            refinement_cycles_used=final_state["refinement_cycles"]
        )
        return assessment
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))