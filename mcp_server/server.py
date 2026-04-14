# mcp_server/server.py
from mcp.server.fastmcp import FastMCP
import os

# Initialize the MCP Server
mcp = FastMCP("Aegis Local Finance DB")

# Mock directory where your sensitive M4 files live
LOCAL_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/secure_docs"))

@mcp.tool()
def read_local_financial_report(entity_name: str) -> str:
    """
    Exposed via MCP to the Researcher Agent.
    Searches local M4 storage for financial data on a given entity.
    """
    print(f"🔒 [MCP Server] Request received to access local files for: {entity_name}")
    
    # In a real app, you would use PyPDF2 or standard file I/O here
    file_path = os.path.join(LOCAL_DATA_DIR, f"{entity_name.lower()}_report.txt")
    
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return f.read()
            
    return f"No local financial data found for {entity_name}."

if __name__ == "__main__":
    # Runs the server locally, usually on stdio or SSE
    print("🚀 Starting Aegis MCP Server on Mac M4...")
    mcp.run()