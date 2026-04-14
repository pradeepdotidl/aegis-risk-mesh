import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from agents.supervisor import build_aegis_graph

router = APIRouter()
mesh = build_aegis_graph()

@router.websocket("/ws/mesh-telemetry")
async def mesh_telemetry_endpoint(websocket: WebSocket):
    """
    Connect to this WS to watch the agents 'think' in real-time.
    """
    await websocket.accept()
    
    try:
        # Wait for the client to send the initial analysis target
        data = await websocket.receive_text()
        request_data = json.loads(data)
        initial_state = {
            "entity_name": request_data.get("entity_name", "Unknown"),
            "analysis_timeframe": request_data.get("analysis_timeframe", "30_days"),
            "gathered_features": [],
            "refinement_cycles": 0
        }
        
        # Stream the graph execution step-by-step
        async for event in mesh.astream(initial_state):
            for agent_node, state_update in event.items():
                # Broadcast which agent just finished and what they did
                message = {
                    "agent": agent_node,
                    "status": "completed_task",
                    "cycles": state_update.get("refinement_cycles", 0),
                    "current_confidence": state_update.get("prediction_confidence", 0.0)
                }
                await websocket.send_json(message)
                
        await websocket.send_json({"status": "MESH_COMPLETE"})
        
    except WebSocketDisconnect:
        print("Client disconnected from telemetry stream.")
    except Exception as e:
        await websocket.send_json({"error": str(e)})