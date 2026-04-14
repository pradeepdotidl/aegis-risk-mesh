from langgraph.graph import StateGraph, END
from core.state import AgenticMeshState
from core.config import settings
from agents.researcher import researcher_node
from agents.optimizer import optimizer_node

def supervisor_router(state: AgenticMeshState) -> str:
    """
    Decides if the graph should loop back for more research or finish.
    """
    confidence = state.get("prediction_confidence", 0.0)
    cycles = state.get("refinement_cycles", 0)
    
    print(f"🧠 [Supervisor] Evaluating state. Current Confidence: {confidence:.2f}, Cycles: {cycles}")
    
    # Check if we hit our ML confidence threshold OR if we've looped too many times
    if confidence >= settings.RISK_CONFIDENCE_THRESHOLD or cycles >= 3:
        print("✅ [Supervisor] Confidence threshold met or max cycles reached. Ending mesh execution.")
        return END
    
    print("⚠️ [Supervisor] Confidence too low. Initiating A2A refinement loop back to Researcher.")
    return "researcher"

def increment_cycle(state: AgenticMeshState):
    """Helper node to track A2A negotiation cycles."""
    return {"refinement_cycles": state.get("refinement_cycles", 0) + 1}

def build_aegis_graph():
    """Compiles the LangGraph State Machine."""
    workflow = StateGraph(AgenticMeshState)
    
    # Add Nodes (The Agents)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("optimizer", optimizer_node)
    workflow.add_node("cycle_tracker", increment_cycle)
    
    # Define Edges (The Flow)
    workflow.set_entry_point("researcher")
    workflow.add_edge("researcher", "optimizer")
    workflow.add_edge("optimizer", "cycle_tracker")
    
    # Conditional Edge (The Logic)
    workflow.add_conditional_edges(
        "cycle_tracker",
        supervisor_router,
        {
            "researcher": "researcher", # Loop back
            END: END                    # Finish
        }
    )
    return workflow.compile()

# This is the executable mesh that the FastAPI endpoint will call
aegis_mesh = build_aegis_graph()