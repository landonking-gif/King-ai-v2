from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import Literal, Optional

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
    database_url: str = Field(..., env="DATABASE_URL")
    # Redis connection string for caching and task queueing
    redis_url: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    
    # --- Ollama / LLM Settings ---
    # The base URL for the Ollama server (e.g., http://localhost:11434)
    ollama_url: str = Field(..., env="OLLAMA_URL")
    # The specific model name to use (e.g., llama3.1:70b or 8b)
    ollama_model: str = Field(default="llama3.1:8b", env="OLLAMA_MODEL")
    
    # --- Pinecone Settings (Vector Store) ---
    # Optional API key for Pinecone vector database integration
    pinecone_api_key: str | None = Field(default=None, env="PINECONE_API_KEY")
    # The index name to use in Pinecone
    pinecone_index: str = Field(default="king-ai", env="PINECONE_INDEX")
    
    # --- Risk & Evolution Controls ---
    # Controls how much autonomy the AI has: conservative, moderate, or aggressive
    risk_profile: Literal["conservative", "moderate", "aggressive"] = "moderate"
    # Max number of self-modification proposals allowed per hour
    max_evolutions_per_hour: int = Field(default=5)
    
    # --- API Server Settings ---
    # Host for the FastAPI server
    api_host: str = Field(default="0.0.0.0")
    # Port for the FastAPI server
    api_port: int = Field(default=8000)
    
    class Config:
        env_file = ".env" # Path to the environment file
        extra = "ignore" # Ignore extra environment variables not defined here

# Singleton instance
settings = Settings()
