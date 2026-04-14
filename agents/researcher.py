import json
from typing import Dict, Any, List
from langchain_community.llms import Ollama
from core.state import AgenticMeshState
from core.models import RiskFeature
from core.config import settings

# Initialize our local LLM (Ollama)
llm = Ollama(base_url=settings.OLLAMA_BASE_URL, model=settings.PRIMARY_LLM_MODEL)

async def call_mcp_tool(entity_name: str) -> str:
    """
    Simulates calling the MCP server tool 'read_local_financial_report'.
    In a full production ADK setup, you'd use the MCP Client session here.
    """
    # For now, we simulate the MCP response. 
    # In your GitHub, you can point to your mcp_server/server.py as the source.
    from mcp_server.server import read_local_financial_report
    return read_local_financial_report(entity_name)

async def researcher_node(state: AgenticMeshState) -> Dict[str, Any]:
    print(f"🕵️  [Researcher] Contextualizing risk for: {state['entity_name']}...")

    # 1. Fetch raw text via MCP
    raw_context = await call_mcp_tool(state['entity_name'])
    
    # 2. Use LLM to extract structured features from raw text
    # This is the "Agentic" part that proves expertise
    extraction_prompt = f"""
    You are a Risk Analyst Agent. Extract exactly 3 metrics from the text below.
    Text: {raw_context}
    
    Respond ONLY with a valid JSON list of objects. 
    Each object MUST have: "feature_name", "value" (float), and "reliability_score" (float 0-1).
    Example: [[{{"feature_name": "Metric Name", "value": 0.5, "reliability_score": 0.9}}]]
    """
    
    print(f"🧠 [Researcher] LLM is analyzing raw MCP data...")
    response = llm.invoke(extraction_prompt)
    
    try:
        # Parse the LLM's JSON response
        clean_json = response.strip().replace("```json", "").replace("```", "").strip()
        extracted_data = json.loads(clean_json)
        
        features = [RiskFeature(source="MCP_Local_Docs", **d) for d in extracted_data]
    except Exception as e:
        print(f"⚠️ [Researcher] LLM response parsing failed. Falling back to baseline. Error: {e}")
        # Fallback if the LLM hallucinates bad JSON
        features = [RiskFeature(feature_name="Extraction Failure", value=0.5, source="System_Fallback", reliability_score=0.1)]

    return {
        "gathered_features": features,
        "context_documents": [raw_context],
        "research_complete": True,
        "current_agent": "optimizer"
    }