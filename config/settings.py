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
    database_url: str = Field(..., validation_alias="DATABASE_URL")
    # Redis connection string for caching and task queueing
    redis_url: str = Field(default="redis://localhost:6379", validation_alias="REDIS_URL")
    
    # --- Ollama / LLM Settings ---
    # The base URL for the Ollama server (e.g., http://localhost:11434)
    ollama_url: str = Field(..., validation_alias="OLLAMA_URL")
    # The specific model name to use (e.g., llama3.1:70b or 8b)
    ollama_model: str = Field(default="llama3.1:8b", validation_alias="OLLAMA_MODEL")

    # --- vLLM Settings (Optional) ---
    # Base URL for a vLLM OpenAI-compatible server
    vllm_url: Optional[str] = Field(default=None, validation_alias="VLLM_URL")
    # Model identifier to request from vLLM
    vllm_model: str = Field(
        default="meta-llama/Llama-3.1-70B-Instruct",
        validation_alias="VLLM_MODEL",
    )

    # --- Claude Settings (Optional) ---
    # Claude model name (API key is read from ANTHROPIC_API_KEY)
    claude_model: str = Field(
        default="claude-3-5-sonnet-20241022",
        validation_alias="CLAUDE_MODEL",
    )
    
    # --- Pinecone Settings (Vector Store) ---
    # Optional API key for Pinecone vector database integration
    pinecone_api_key: str | None = Field(default=None, validation_alias="PINECONE_API_KEY")
    # The index name to use in Pinecone
    pinecone_index: str = Field(default="king-ai", validation_alias="PINECONE_INDEX")
    
    # --- Risk & Evolution Controls ---
    # Controls how much autonomy the AI has: conservative, moderate, or aggressive
    risk_profile: Literal["conservative", "moderate", "aggressive"] = "moderate"
    # Max number of self-modification proposals allowed per hour
    max_evolutions_per_hour: int = Field(default=5)
    # Enable autonomous operation mode (self-driven without user prompts)
    enable_autonomous_mode: bool = Field(default=False)
    # Enable self-modification capabilities
    enable_self_modification: bool = Field(default=True)
    # Confidence threshold for evolution proposals (0.0 - 1.0)
    evolution_confidence_threshold: float = Field(default=0.8)
    # Daily limit for evolution proposals
    evolution_daily_limit: int = Field(default=1)
    # Sandbox timeout for testing evolution proposals (seconds)
    evolution_sandbox_timeout: int = Field(default=300)
    # Require tests to pass before applying evolution
    evolution_require_tests: bool = Field(default=True)
    
    # --- Approval Settings ---
    # Maximum amount (in dollars) that can be auto-approved
    max_auto_approve_amount: float = Field(default=100.0)
    # Hours before an approval request expires
    approval_expiry_hours: int = Field(default=24)
    
    # --- Scheduler Settings ---
    # Enable background task scheduler
    enable_scheduler: bool = Field(default=True)
    # Default interval for KPI review (hours)
    kpi_review_interval_hours: int = Field(default=6)
    # Default interval for health checks (hours)
    health_check_interval_hours: int = Field(default=1)
    
    # --- API Server Settings ---
    # Host for the FastAPI server
    api_host: str = Field(default="0.0.0.0")
    # Port for the FastAPI server
    api_port: int = Field(default=8000)

# Singleton instance
settings = Settings()
