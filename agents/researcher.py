import json
from typing import Dict, Any, List
from langchain_ollama import OllamaLLM
from core.state import AgenticMeshState
from core.models import RiskFeature
from core.config import settings

# Initialize our local LLM with the updated package
llm = OllamaLLM(base_url=settings.OLLAMA_BASE_URL, model=settings.PRIMARY_LLM_MODEL)

async def call_mcp_tool(entity_name: str) -> str:
    """
    Calls the MCP server tool. 
    """
    try:
        from mcp_server.server import read_local_financial_report
        return read_local_financial_report(entity_name)
    except Exception as e:
        return f"ERROR_SYSTEM_EXCEPTION: {str(e)}"

async def researcher_node(state: AgenticMeshState) -> Dict[str, Any]:
    print(f"🕵️  [Researcher] Contextualizing risk for: {state['entity_name']}...")

    # 1. Fetch raw text via MCP
    raw_context = await call_mcp_tool(state['entity_name'])
    
    # 2. THE GATEKEEPER: Check for MCP Error Strings
    if "ERROR_" in raw_context:
        print(f"🛑 [Researcher] Data Ingestion Failed: {raw_context}")
        return {
            "gathered_features": [],
            "context_documents": [raw_context],
            "research_complete": False,
            "errors": [raw_context],
            "current_agent": "supervisor"
        }

    # 3. Use LLM to extract structured features ONLY if data was found
    extraction_prompt = f"""
    You are a Risk Analyst Agent. You must extract exactly 3 metrics from the text below.
    If the text does not contain enough data, do not invent metrics.
    
    Text: {raw_context}
    
    Respond ONLY with a valid JSON list of objects. 
    Each object MUST have: "feature_name", "value" (float), and "reliability_score" (float 0-1).
    Example: [{{"feature_name": "Metric Name", "value": 0.5, "reliability_score": 0.9}}]
    """
    
    print(f"🧠 [Researcher] Valid data found. LLM is analyzing raw MCP data...")
    response = llm.invoke(extraction_prompt)
    
    try:
        clean_json = response.strip().replace("```json", "").replace("```", "").strip()
        extracted_data = json.loads(clean_json)
        
        if not isinstance(extracted_data, list):
            extracted_data = [extracted_data]

        features = [RiskFeature(source="MCP_Local_Docs", **d) for d in extracted_data]
    except Exception as e:
        print(f"⚠️ [Researcher] LLM response parsing failed. Error: {e}")
        features = [
            RiskFeature(
                feature_name="Extraction Integrity Check", 
                value=0.0, 
                source="System_Error", 
                reliability_score=0.0
            )
        ]

    return {
        "gathered_features": features,
        "context_documents": [raw_context],
        "research_complete": True,
        "current_agent": "optimizer"
    }