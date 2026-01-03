"""
Configuration management for Subagent Manager service.
"""

from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class SubagentManagerConfig(BaseSettings):
    """Configuration for Subagent Manager service."""

    model_config = SettingsConfigDict(env_prefix="SUBAGENT_", case_sensitive=False)

    # Service configuration
    host: str = Field(default="0.0.0.0", description="Service host")
    port: int = Field(default=8001, description="Service port")
    log_level: str = Field(default="INFO", description="Logging level")

    # Subagent lifecycle
    default_timeout: int = Field(default=300, description="Default subagent timeout (seconds)")
    max_lifetime: int = Field(
        default=3600, description="Maximum subagent lifetime (seconds)"
    )
    cleanup_interval: int = Field(
        default=60, description="Cleanup interval for expired subagents (seconds)"
    )

    # LLM configuration
    llm_provider: str = Field(default="mock", description="Default LLM provider")
    llm_model: str = Field(default="mock-model", description="Default LLM model")
    llm_temperature: float = Field(default=0.7, description="Default temperature")
    llm_max_tokens: int = Field(default=4096, description="Default max tokens")

    # API keys (optional, can use env vars directly)
    anthropic_api_key: Optional[str] = Field(default=None, description="Anthropic API key")
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")

    # MCP Gateway configuration
    mcp_gateway_url: str = Field(
        default="http://localhost:8080", description="MCP Gateway URL for tool invocation"
    )

    # Schema validation
    schema_registry_path: str = Field(
        default="docs/schema_registry", description="Path to JSON schema registry"
    )

    # Resource limits
    max_concurrent_subagents: int = Field(
        default=100, description="Maximum concurrent subagents"
    )
    max_context_tokens: int = Field(
        default=100000, description="Maximum tokens in subagent context"
    )

    # Observability
    enable_metrics: bool = Field(default=True, description="Enable Prometheus metrics")
    enable_tracing: bool = Field(default=True, description="Enable OpenTelemetry tracing")


# Global config instance
config = SubagentManagerConfig()
