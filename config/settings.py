from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Literal, Optional

class Settings(BaseSettings):
    """
    Main configuration class for King AI v2.
    Loads values from environment variables or a .env file.
    """
    
    # --- Database Settings ---
    database_url: str = Field(..., env="DATABASE_URL")
    redis_url: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    
    # --- LLM Provider Settings ---
    # Primary: vLLM for production
    vllm_url: Optional[str] = Field(default=None, env="VLLM_URL")
    vllm_model: str = Field(
        default="meta-llama/Llama-3.1-70B-Instruct",
        env="VLLM_MODEL"
    )
    
    # Secondary: Ollama for development/fallback
    ollama_url: str = Field(default="http://localhost:11434", env="OLLAMA_URL")
    ollama_model: str = Field(default="llama3.1:8b", env="OLLAMA_MODEL")
    
    # Tertiary: Cloud fallbacks (API keys via env)
    # GEMINI_API_KEY loaded directly from env
    # ANTHROPIC_API_KEY for future Claude integration
    
    # --- Vector Store Settings ---
    pinecone_api_key: Optional[str] = Field(default=None, env="PINECONE_API_KEY")
    pinecone_index: str = Field(default="king-ai", env="PINECONE_INDEX")
    pinecone_environment: str = Field(default="us-east-1", env="PINECONE_ENV")
    
    # --- Risk & Evolution Controls ---
    risk_profile: Literal["conservative", "moderate", "aggressive"] = Field(
        default="moderate",
        env="RISK_PROFILE"
    )
    max_evolutions_per_hour: int = Field(default=5, env="MAX_EVOLUTIONS_PER_HOUR")
    evolution_confidence_threshold: float = Field(
        default=0.8,
        env="EVOLUTION_CONFIDENCE_THRESHOLD"
    )
    
    # --- API Server Settings ---
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_workers: int = Field(default=4, env="API_WORKERS")
    
    # --- AWS Settings ---
    aws_region: str = Field(default="us-east-1", env="AWS_REGION")
    sqs_inference_queue: Optional[str] = Field(default=None, env="SQS_INFERENCE_QUEUE")
    s3_artifacts_bucket: Optional[str] = Field(default=None, env="S3_ARTIFACTS_BUCKET")
    
    # --- Monitoring Settings ---
    datadog_enabled: bool = Field(default=False, env="DD_ENABLED")
    datadog_api_key: Optional[str] = Field(default=None, env="DD_API_KEY")
    datadog_app_key: Optional[str] = Field(default=None, env="DD_APP_KEY")
    
    # --- Security Settings ---
    jwt_secret: str = Field(default="change-me-in-production", env="JWT_SECRET")
    jwt_algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    api_rate_limit: int = Field(default=100, env="API_RATE_LIMIT")  # requests per minute
    
    # --- Feature Flags ---
    enable_autonomous_mode: bool = Field(default=False, env="ENABLE_AUTONOMOUS_MODE")
    enable_self_modification: bool = Field(default=True, env="ENABLE_SELF_MODIFICATION")
    enable_vllm: bool = Field(default=False, env="ENABLE_VLLM")
    
    class Config:
        env_file = ".env"
        extra = "ignore"


# Singleton instance
settings = Settings()
