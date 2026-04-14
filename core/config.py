import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # API & App Settings
    PROJECT_NAME: str = "Project Aegis: Risk Intelligence Mesh"
    API_V1_STR: str = "/api/v1"
    
    # Local Inference (Mac M4 + Ollama)
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    PRIMARY_LLM_MODEL: str = os.getenv("PRIMARY_LLM_MODEL", "llama3.2:3b")
    
    # ML & RAG Thresholds
    RISK_CONFIDENCE_THRESHOLD: float = 0.80 # Agents loop if confidence is below 80%
    VECTOR_DB_PATH: str = "./data/chroma_db"

    class Config:
        env_file = ".env"

settings = Settings()