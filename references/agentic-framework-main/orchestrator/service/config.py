"""
Configuration module for the Orchestrator service.

Uses pydantic-settings for environment variable management with type validation.
"""

from typing import Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class OrchestratorConfig(BaseSettings):
    """Configuration for the Lead Agent/Orchestrator service."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Service Configuration
    host: str = Field(default="0.0.0.0", description="Host to bind the service")
    port: int = Field(default=8000, description="Port to bind the service")
    reload: bool = Field(default=False, description="Enable auto-reload for development")

    # LLM Provider Configuration
    default_llm_provider: str = Field(
        default="anthropic", description="Default LLM provider (anthropic, openai, azure)"
    )
    anthropic_api_key: Optional[str] = Field(
        default=None, description="Anthropic API key"
    )
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    azure_openai_key: Optional[str] = Field(
        default=None, description="Azure OpenAI API key"
    )
    azure_openai_endpoint: Optional[str] = Field(
        default=None, description="Azure OpenAI endpoint"
    )

    # Database Configuration
    postgres_url: str = Field(
        default="postgresql://user:password@localhost:5432/agentic_framework",
        description="PostgreSQL connection URL",
    )
    redis_url: str = Field(
        default="redis://localhost:6379/0", description="Redis connection URL"
    )

    # Vector Database Configuration
    milvus_url: str = Field(
        default="http://localhost:19530", description="Milvus server URL"
    )
    chroma_path: str = Field(
        default="./data/chroma", description="ChromaDB persistence path"
    )
    use_chroma_dev: bool = Field(
        default=True, description="Use ChromaDB for development (Milvus for production)"
    )

    # Object Storage Configuration
    minio_endpoint: str = Field(
        default="localhost:9000", description="MinIO endpoint"
    )
    minio_access_key: str = Field(default="minioadmin", description="MinIO access key")
    minio_secret_key: str = Field(default="minioadmin", description="MinIO secret key")
    minio_bucket: str = Field(
        default="agent-artifacts", description="MinIO bucket for artifacts"
    )
    minio_secure: bool = Field(
        default=False, description="Use secure connection for MinIO"
    )

    # Dependent Services
    mcp_gateway_url: str = Field(
        default="http://localhost:8080", description="MCP Gateway service URL"
    )
    subagent_manager_url: str = Field(
        default="http://localhost:8001", description="Subagent Manager service URL"
    )
    memory_service_url: str = Field(
        default="http://localhost:8002", description="Memory Service URL"
    )
    code_executor_url: str = Field(
        default="http://localhost:8003", description="Code Executor service URL"
    )

    # Security Configuration
    jwt_secret_key: str = Field(
        default="your-secret-key-change-in-production",
        description="JWT secret key for token signing",
    )
    jwt_algorithm: str = Field(default="HS256", description="JWT algorithm")
    access_token_expire_minutes: int = Field(
        default=60, description="Access token expiration time in minutes"
    )

    # Memory Configuration
    memory_session_ttl_hours: int = Field(
        default=72, description="Session memory TTL in hours"
    )
    memory_retention_years: int = Field(
        default=2, description="Long-term memory retention in years"
    )
    memory_compaction_threshold_tokens: int = Field(
        default=8000, description="Token threshold for memory compaction"
    )

    # Workflow Configuration
    manifest_schema_path: str = Field(
        default="./docs/schemas/manifest_schema.json",
        description="Path to manifest JSON schema",
    )
    max_workflow_timeout_seconds: int = Field(
        default=3600, description="Maximum workflow execution timeout"
    )
    max_subagent_retries: int = Field(
        default=3, description="Maximum retries for subagent calls"
    )

    # Observability Configuration
    otel_exporter_otlp_endpoint: str = Field(
        default="http://localhost:4317", description="OpenTelemetry OTLP endpoint"
    )
    prometheus_port: int = Field(default=9090, description="Prometheus metrics port")
    enable_tracing: bool = Field(
        default=True, description="Enable OpenTelemetry tracing"
    )
    enable_metrics: bool = Field(
        default=True, description="Enable Prometheus metrics"
    )
    log_level: str = Field(
        default="INFO", description="Logging level (DEBUG, INFO, WARNING, ERROR)"
    )

    # Policy Configuration
    require_human_approval_default: bool = Field(
        default=False, description="Default value for requiring human approval"
    )
    enable_pii_check: bool = Field(
        default=True, description="Enable PII checking in artifacts"
    )

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level is one of the allowed values."""
        allowed_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v_upper = v.upper()
        if v_upper not in allowed_levels:
            raise ValueError(f"log_level must be one of {allowed_levels}")
        return v_upper

    @field_validator("default_llm_provider")
    @classmethod
    def validate_llm_provider(cls, v: str) -> str:
        """Validate LLM provider is supported."""
        allowed_providers = {"anthropic", "openai", "azure"}
        v_lower = v.lower()
        if v_lower not in allowed_providers:
            raise ValueError(f"default_llm_provider must be one of {allowed_providers}")
        return v_lower


# Global config instance
config = OrchestratorConfig()
