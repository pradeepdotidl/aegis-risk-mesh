import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from strawberry.fastapi import GraphQLRouter

# Import our API modules
from api.rest_routes import router as rest_router
from api.websocket_routes import router as ws_router
from api.graphql_schema import schema
from core.config import settings
from agents.supervisor import build_aegis_graph

# --- LIFESPAN MANAGER ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs ONCE when the server starts.
    """
    print("🚀 [1/3] Initializing LangGraph Mesh...")
    try:
        app.state.mesh = build_aegis_graph()
        print("✅ [2/3] Agentic Mesh Compiled Successfully.")
    except Exception as e:
        print(f"❌ ERROR: Failed to build Agentic Mesh: {e}")
        raise e
    
    yield
    print("🛑 [3/3] Shutting down Aegis Mesh.")

# --- APP SETUP ---
app = FastAPI(
    title=settings.PROJECT_NAME,
    description="A multi-agent risk intelligence mesh powered by Google ADK, LangGraph, and local ML.",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Routers
app.include_router(rest_router, prefix=settings.API_V1_STR)
app.include_router(ws_router)

# Mount GraphQL
graphql_app = GraphQLRouter(schema)
app.include_router(graphql_app, prefix="/graphql")

@app.get("/health")
async def health_check():
    return {
        "status": "Aegis Mesh is Online", 
        "mesh_initialized": hasattr(app.state, 'mesh')
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=False)