"""
Configuration Management for Kautilya.

Module: kautilya/config.py
"""

from typing import Optional, Dict, Any
from pathlib import Path
import yaml
from pydantic import BaseModel, Field


class ProjectConfig(BaseModel):
    """Project configuration."""

    name: str
    version: str = "1.0.0"


class LLMHyperparameters(BaseModel):
    """LLM hyperparameters configuration."""

    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, ge=1)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(default=None, ge=1)
    frequency_penalty: Optional[float] = Field(default=None, ge=-2.0, le=2.0)
    presence_penalty: Optional[float] = Field(default=None, ge=-2.0, le=2.0)
    stop_sequences: Optional[list] = None
    max_retries: int = Field(default=3, ge=0, le=10)


class LLMProviderConfig(BaseModel):
    """LLM provider configuration."""

    default_model: str
    api_key_env: str
    fallback_model: Optional[str] = None
    endpoint: Optional[str] = None  # For local models
    hyperparameters: LLMHyperparameters = Field(default_factory=LLMHyperparameters)


class MCPGatewayConfig(BaseModel):
    """MCP Gateway configuration."""

    url: str = "http://localhost:8080"
    auth: str = "bearer_token"


class OrchestratorConfig(BaseModel):
    """Orchestrator configuration."""

    url: str = "http://localhost:8000"


class DisplayConfig(BaseModel):
    """Display configuration for iteration feedback."""

    mode: str = Field(default="detailed", pattern="^(minimal|detailed)$")
    show_tokens: bool = True
    show_thinking: bool = True
    show_timer: bool = True


class Config(BaseModel):
    """Kautilya configuration."""

    project: Optional[ProjectConfig] = None
    llm_providers: Dict[str, LLMProviderConfig] = Field(default_factory=dict)
    default_provider: str = "anthropic"
    memory_backend: str = "redis"
    vector_db: str = "chroma"
    mcp_gateway: MCPGatewayConfig = Field(default_factory=MCPGatewayConfig)
    orchestrator: OrchestratorConfig = Field(default_factory=OrchestratorConfig)
    display: DisplayConfig = Field(default_factory=DisplayConfig)


def load_config(config_dir: str = ".kautilya") -> Config:
    """
    Load configuration from directory.

    Args:
        config_dir: Configuration directory path

    Returns:
        Loaded configuration
    """
    config_path = Path(config_dir) / "config.yaml"

    if not config_path.exists():
        # Return default configuration
        return Config()

    try:
        with open(config_path, "r") as f:
            data = yaml.safe_load(f) or {}
        return Config(**data)
    except Exception as e:
        print(f"Warning: Failed to load config from {config_path}: {e}")
        return Config()


def save_config(config: Config, config_dir: str = ".kautilya") -> None:
    """
    Save configuration to directory.

    Args:
        config: Configuration to save
        config_dir: Configuration directory path
    """
    config_path = Path(config_dir)
    config_path.mkdir(parents=True, exist_ok=True)

    config_file = config_path / "config.yaml"

    with open(config_file, "w") as f:
        yaml.dump(config.model_dump(exclude_none=True), f, default_flow_style=False)


def get_llm_config_path(config_dir: str = ".kautilya") -> Path:
    """Get path to LLM configuration file."""
    return Path(config_dir) / "llm.yaml"


def load_llm_config(config_dir: str = ".kautilya") -> Dict[str, Any]:
    """Load LLM configuration."""
    llm_config_path = get_llm_config_path(config_dir)

    if not llm_config_path.exists():
        return {}

    with open(llm_config_path, "r") as f:
        return yaml.safe_load(f) or {}


def save_llm_config(llm_config: Dict[str, Any], config_dir: str = ".kautilya") -> None:
    """Save LLM configuration."""
    llm_config_path = get_llm_config_path(config_dir)
    llm_config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(llm_config_path, "w") as f:
        yaml.dump(llm_config, f, default_flow_style=False)
