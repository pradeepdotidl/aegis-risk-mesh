from mcp.server.fastmcp import FastMCP
import os

mcp = FastMCP("Aegis Local Finance DB")

# Ensure the path is handled strictly across different environments
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL_DATA_DIR = os.path.join(BASE_DIR, "data", "secure_docs")

# Create the directory if it's missing during dev
if not os.path.exists(LOCAL_DATA_DIR):
    os.makedirs(LOCAL_DATA_DIR, exist_ok=True)
    print(f"📁 [System] Created missing directory at: {LOCAL_DATA_DIR}")

@mcp.tool()
def read_local_financial_report(entity_name: str) -> str:
    """
    Searches local storage. Returns file content or an explicit ERROR_NOT_FOUND string.
    """
    print(f"🔒 [MCP Server] Searching for: {entity_name}")

    if not os.path.exists(LOCAL_DATA_DIR):
        return f"ERROR_DIRECTORY_MISSING: {LOCAL_DATA_DIR}"

    file_path = os.path.join(LOCAL_DATA_DIR, f"{entity_name.lower()}_report.txt")
    
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
            
    return f"ERROR_FILE_NOT_FOUND: {entity_name}"

if __name__ == "__main__":
    print("🚀 Starting Aegis MCP Server...")
    mcp.run()