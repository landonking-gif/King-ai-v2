"""
Configuration for Code Executor Service.
Module: code-exec/service/config.py
"""

from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class CodeExecSettings(BaseSettings):
    """Configuration settings for Code Executor service."""

    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="CODE_EXEC_", case_sensitive=False
    )

    # Service configuration
    service_name: str = Field(default="code-executor", description="Service name")
    host: str = Field(default="0.0.0.0", description="Service host")
    port: int = Field(default=8002, description="Service port")
    debug: bool = Field(default=False, description="Debug mode")

    # Skills configuration
    skills_directory: str = Field(
        default="/Users/paragpradhan/Projects/Agent framework/agent-framework/code-exec/skills",
        description="Directory containing skill definitions",
    )
    max_execution_time: int = Field(
        default=30, description="Maximum execution time in seconds"
    )

    # Hashing configuration
    hash_algorithm: str = Field(default="sha256", description="Hash algorithm for provenance")

    # Logging configuration
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format",
    )

    # Redis configuration (for provenance logging)
    redis_url: Optional[str] = Field(
        default="redis://localhost:6379/2", description="Redis URL for logs"
    )

    # Embedding model configuration
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Default embedding model",
    )

    # Security
    require_policy_approval_for_side_effects: bool = Field(
        default=True, description="Require policy approval for side-effect skills"
    )


settings = CodeExecSettings()
