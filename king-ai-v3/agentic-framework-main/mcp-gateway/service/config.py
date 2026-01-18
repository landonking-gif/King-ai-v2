"""
MCP Gateway configuration module.

Loads configuration from environment variables and provides settings for the gateway service.
"""

from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """MCP Gateway service settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Service configuration
    mcp_gateway_host: str = "0.0.0.0"
    mcp_gateway_port: int = 8080
    service_name: str = "mcp-gateway"
    api_version: str = "1.0.0"

    # Redis configuration for rate limiting
    redis_url: str = "redis://localhost:6379/0"

    # Security
    jwt_secret_key: str
    jwt_algorithm: str = "HS256"
    ephemeral_token_ttl_minutes: int = 15
    access_token_expire_minutes: int = 60

    # Rate limiting defaults
    default_rate_limit_calls: int = 100
    default_rate_limit_window_seconds: int = 60

    # PII detection
    enable_pii_detection: bool = True
    pii_keywords: list[str] = [
        "ssn",
        "social_security",
        "credit_card",
        "password",
        "api_key",
        "secret",
        "token",
        "bearer",
        "authorization",
    ]

    # Logging
    log_level: str = "INFO"
    enable_provenance_logging: bool = True

    # Database (for future migration from in-memory)
    postgres_url: Optional[str] = None

    # CORS
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:8000"]
    cors_allow_credentials: bool = True
    cors_allow_methods: list[str] = ["*"]
    cors_allow_headers: list[str] = ["*"]


# Global settings instance
settings = Settings()
