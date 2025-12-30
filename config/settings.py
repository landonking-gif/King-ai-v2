from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Literal

class Settings(BaseSettings):
    """
    Main configuration class for King AI v2.
    Loads values from environment variables or a .env file.
    """
    
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
    
    # --- Shopify Configuration ---
    # Shopify store name (e.g., "my-store" from my-store.myshopify.com)
    SHOPIFY_SHOP_NAME: str | None = Field(default=None, env="SHOPIFY_SHOP_NAME")
    # Shopify Admin API access token
    SHOPIFY_ACCESS_TOKEN: str | None = Field(default=None, env="SHOPIFY_ACCESS_TOKEN")
    # Secret for verifying Shopify webhooks
    SHOPIFY_WEBHOOK_SECRET: str | None = Field(default=None, env="SHOPIFY_WEBHOOK_SECRET")
    # Shopify API version to use
    SHOPIFY_API_VERSION: str = Field(default="2024-10", env="SHOPIFY_API_VERSION")
    
    class Config:
        env_file = ".env" # Path to the environment file
        extra = "ignore" # Ignore extra environment variables not defined here

# Create a singleton instance of settings to be used across the application
settings = Settings()
