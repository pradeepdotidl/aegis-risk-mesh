from fastapi import APIRouter, HTTPException, Request
from core.models import RiskRequest, FinalRiskAssessment

print("🚀 DEBUG: rest_routes.py is being imported!")

router = APIRouter()

@router.post("/analyze", response_model=FinalRiskAssessment)
async def trigger_risk_analysis(request: Request, body: RiskRequest):
    """
    Standard REST endpoint. Triggers the Agentic Mesh and waits for completion.
    """
    # 1. Pull the mesh from the app state
    mesh = request.app.state.mesh
    
    initial_state = {
        "entity_name": body.entity_name, 
        "analysis_timeframe": body.analysis_timeframe,
        "gathered_features": [],
        "refinement_cycles": 0,
        "errors": []
    }
    
    try:
        # 2. Execute the LangGraph workflow
        final_state = await mesh.ainvoke(initial_state)
        
        # 3. Defensive Check
        prediction = final_state.get("ml_prediction")
        
        if prediction is None:
            raise HTTPException(
                status_code=404, 
                detail={
                    "status": "Incomplete Analysis",
                    "reason": f"No local financial data found for '{body.entity_name}' after 3 attempts.",
                    "cycles": final_state.get("refinement_cycles")
                }
            )

        # 4. Success Path
        return FinalRiskAssessment(
            entity_name=final_state["entity_name"],
            overall_risk_score=prediction.probability_of_event * 100,
            risk_category="Critical" if prediction.probability_of_event > 0.7 else "Safe",
            optimizer_prediction=prediction,
            researcher_summary=f"Analysis verified after {final_state.get('refinement_cycles')} cycles.",
            refinement_cycles_used=final_state.get("refinement_cycles")
        )
        
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"❌ Mesh Execution Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))