from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import Literal

class Settings(BaseSettings):
    """
    Main configuration class for King AI v2.
    Loads values from environment variables or a .env file.
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    # --- Database Settings ---
    # The primary PostgreSQL connection string (asyncpg)
    database_url: str
    # Redis connection string for caching and task queueing
    redis_url: str = "redis://localhost:6379"
    
    # --- Ollama / LLM Settings ---
    # The base URL for the Ollama server (e.g., http://localhost:11434)
    ollama_url: str
    # The specific model name to use (e.g., llama3.1:70b or 8b)
    ollama_model: str = "llama3.1:8b"
    
    # --- Pinecone Settings (Vector Store) ---
    # Optional API key for Pinecone vector database integration
    pinecone_api_key: str | None = None
    # The index name to use in Pinecone
    pinecone_index: str = "king-ai"
    # The Pinecone environment/region (e.g., us-east-1)
    pinecone_environment: str = "us-east-1"
    
    # --- Risk & Evolution Controls ---
    # Controls how much autonomy the AI has: conservative, moderate, or aggressive
    risk_profile: Literal["conservative", "moderate", "aggressive"] = "moderate"
    # Max number of self-modification proposals allowed per hour
    max_evolutions_per_hour: int = 5
    # Whether autonomous mode is enabled
    enable_autonomous_mode: bool = False
    
    # --- API Server Settings ---
    # Host for the FastAPI server
    api_host: str = "0.0.0.0"
    # Port for the FastAPI server
    api_port: int = 8000

# Create a singleton instance of settings to be used across the application
settings = Settings()
