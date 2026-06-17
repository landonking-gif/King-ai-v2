"""
Application settings loaded from environment variables.

All configuration is driven by .env — no secrets are hardcoded.
"""

import os
from typing import List, Optional
from dataclasses import dataclass, field


def _env_list(key: str, default: str = "") -> List[str]:
    """Parse a comma-separated env var into a list."""
    val = os.getenv(key, default)
    return [s.strip() for s in val.split(",") if s.strip()]


@dataclass
class Settings:
    # Database
    database_url: str = os.getenv("DATABASE_URL", "postgresql+asyncpg://king:password@localhost:5432/kingai")
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379")

    # LLM Providers
    ollama_url: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    anthropic_api_key: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    gemini_api_key: Optional[str] = os.getenv("GEMINI_API_KEY")
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")

    # CORS — explicit origins only (never wildcard + credentials)
    allowed_origins: List[str] = _env_list("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8000")

    # API Server
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))

    # Rate limiting
    enable_rate_limiting: bool = os.getenv("ENABLE_RATE_LIMITING", "true").lower() == "true"
    rate_limit_requests: int = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
    rate_limit_window: int = int(os.getenv("RATE_LIMIT_WINDOW", "60"))

    # Risk & Approvals
    risk_profile: str = os.getenv("RISK_PROFILE", "moderate")
    max_auto_approve_amount: float = float(os.getenv("MAX_AUTO_APPROVE_AMOUNT", "100.0"))
    approval_expiry_hours: int = int(os.getenv("APPROVAL_EXPIRY_HOURS", "24"))

    # Autonomous mode
    enable_autonomous_mode: bool = os.getenv("ENABLE_AUTONOMOUS_MODE", "false").lower() == "true"
    enable_self_modification: bool = os.getenv("ENABLE_SELF_MODIFICATION", "true").lower() == "true"
    enable_scheduler: bool = os.getenv("ENABLE_SCHEDULER", "true").lower() == "true"

    # Evolution
    max_evolutions_per_hour: int = int(os.getenv("MAX_EVOLUTIONS_PER_HOUR", "5"))
    evolution_daily_limit: int = int(os.getenv("EVOLUTION_DAILY_LIMIT", "1"))
    evolution_confidence_threshold: float = float(os.getenv("EVOLUTION_CONFIDENCE_THRESHOLD", "0.8"))
    evolution_require_tests: bool = os.getenv("EVOLUTION_REQUIRE_TESTS", "true").lower() == "true"
    evolution_sandbox_timeout: int = int(os.getenv("EVOLUTION_SANDBOX_TIMEOUT", "300"))

    # Scheduler
    kpi_review_interval_hours: int = int(os.getenv("KPI_REVIEW_INTERVAL_HOURS", "6"))
    health_check_interval_hours: int = int(os.getenv("HEALTH_CHECK_INTERVAL_HOURS", "1"))

    # Docker sandbox
    docker_sandbox_enabled: bool = os.getenv("DOCKER_SANDBOX_ENABLED", "true").lower() == "true"
    docker_sandbox_strict: bool = os.getenv("DOCKER_SANDBOX_STRICT", "true").lower() == "true"

    # Webhook secrets
    stripe_webhook_secret: Optional[str] = os.getenv("STRIPE_WEBHOOK_SECRET")
    shopify_webhook_secret: Optional[str] = os.getenv("SHOPIFY_WEBHOOK_SECRET")
    plaid_webhook_secret: Optional[str] = os.getenv("PLAID_WEBHOOK_SECRET")

    # JWT Authentication
    jwt_secret: str = os.getenv("JWT_SECRET", "change-me-in-production")
    jwt_algorithm: str = os.getenv("JWT_ALGORITHM", "HS256")
    jwt_expiry_minutes: int = int(os.getenv("JWT_EXPIRY_MINUTES", "60"))

    # Integrations
    stripe_api_key: Optional[str] = os.getenv("STRIPE_API_KEY")
    shopify_shop_url: Optional[str] = os.getenv("SHOPIFY_SHOP_URL")
    shopify_access_token: Optional[str] = os.getenv("SHOPIFY_ACCESS_TOKEN")
    plaid_client_id: Optional[str] = os.getenv("PLAID_CLIENT_ID")
    plaid_secret: Optional[str] = os.getenv("PLAID_SECRET")
    plaid_env: str = os.getenv("PLAID_ENV", "sandbox")
    paypal_client_id: Optional[str] = os.getenv("PAYPAL_CLIENT_ID")
    paypal_client_secret: Optional[str] = os.getenv("PAYPAL_CLIENT_SECRET")

    # Monitoring
    dd_api_key: Optional[str] = os.getenv("DD_API_KEY")
    arize_api_key: Optional[str] = os.getenv("ARIZE_API_KEY")
    langchain_api_key: Optional[str] = os.getenv("LANGCHAIN_API_KEY")


settings = Settings()
