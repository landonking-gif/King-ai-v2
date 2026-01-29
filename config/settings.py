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

    # --- Claude/Anthropic Settings (Optional) ---
    # Anthropic API key for Claude
    anthropic_api_key: str | None = Field(default=None, validation_alias="ANTHROPIC_API_KEY")
    # Claude model name
    claude_model: str = Field(
        default="claude-3-5-sonnet-20241022",
        validation_alias="CLAUDE_MODEL",
    )
    
    # --- Gemini Settings ---
    # Single Gemini API key
    gemini_api_key: str | None = Field(default=None, validation_alias="GEMINI_API_KEY")
    # Multiple Gemini API keys (comma-separated)
    gemini_api_keys: str = Field(default="", validation_alias="GEMINI_API_KEYS")

    # --- OpenAI Settings (for DALL-E, Whisper, CLIP) ---
    openai_api_key: str | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    
    # --- Pinecone Settings (Vector Store) ---
    # Optional API key for Pinecone vector database integration
    pinecone_api_key: str | None = Field(default=None, validation_alias="PINECONE_API_KEY")
    # The index name to use in Pinecone
    pinecone_index: str = Field(default="king-ai", validation_alias="PINECONE_INDEX")
    # Pinecone environment
    pinecone_environment: str | None = Field(default=None, validation_alias="PINECONE_ENVIRONMENT")
    
    # --- Stripe Settings (Payments) ---
    stripe_api_key: str | None = Field(default=None, validation_alias="STRIPE_API_KEY")
    stripe_publishable_key: str | None = Field(default=None, validation_alias="STRIPE_PUBLISHABLE_KEY")
    stripe_webhook_secret: str | None = Field(default=None, validation_alias="STRIPE_WEBHOOK_SECRET")
    
    # --- PayPal Settings (Fallback Payments) ---
    paypal_client_id: str | None = Field(default=None, validation_alias="PAYPAL_CLIENT_ID")
    paypal_client_secret: str | None = Field(default=None, validation_alias="PAYPAL_CLIENT_SECRET")
    paypal_sandbox: bool = Field(default=True, validation_alias="PAYPAL_SANDBOX")
    paypal_webhook_id: str | None = Field(default=None, validation_alias="PAYPAL_WEBHOOK_ID")
    
    # --- Plaid Settings (Banking) ---
    plaid_client_id: str | None = Field(default=None, validation_alias="PLAID_CLIENT_ID")
    plaid_secret: str | None = Field(default=None, validation_alias="PLAID_SECRET")
    plaid_env: str = Field(default="sandbox", validation_alias="PLAID_ENV")
    
    # --- Shopify Settings (E-Commerce) ---
    shopify_shop_url: str | None = Field(default=None, validation_alias="SHOPIFY_SHOP_URL")
    shopify_access_token: str | None = Field(default=None, validation_alias="SHOPIFY_ACCESS_TOKEN")
    shopify_api_version: str = Field(default="2024-10", validation_alias="SHOPIFY_API_VERSION")
    
    # --- Google Analytics 4 Settings ---
    ga4_property_id: str | None = Field(default=None, validation_alias="GA4_PROPERTY_ID")
    google_application_credentials: str | None = Field(default=None, validation_alias="GOOGLE_APPLICATION_CREDENTIALS")
    ga4_credentials_json: str | None = Field(default=None, validation_alias="GA4_CREDENTIALS_JSON")
    
    # --- SerpAPI Settings (Web Search) ---
    serpapi_key: str | None = Field(default=None, validation_alias="SERPAPI_KEY")
    
    # --- Twilio Settings (SMS/Voice) ---
    twilio_account_sid: str | None = Field(default=None, validation_alias="TWILIO_ACCOUNT_SID")
    twilio_auth_token: str | None = Field(default=None, validation_alias="TWILIO_AUTH_TOKEN")
    twilio_from_number: str | None = Field(default=None, validation_alias="TWILIO_FROM_NUMBER")
    admin_phone_number: str | None = Field(default=None, validation_alias="ADMIN_PHONE_NUMBER")
    
    # --- Datadog Monitoring ---
    dd_api_key: str | None = Field(default=None, validation_alias="DD_API_KEY")
    dd_app_key: str | None = Field(default=None, validation_alias="DD_APP_KEY")
    datadog_api_key: str | None = Field(default=None, validation_alias="DATADOG_API_KEY")
    datadog_app_key: str | None = Field(default=None, validation_alias="DATADOG_APP_KEY")
    environment: str = Field(default="development", validation_alias="ENVIRONMENT")
    
    # --- Arize ML Observability ---
    arize_api_key: str | None = Field(default=None, validation_alias="ARIZE_API_KEY")
    arize_space_key: str | None = Field(default=None, validation_alias="ARIZE_SPACE_KEY")
    
    # --- LangSmith/LangChain Tracing ---
    langchain_api_key: str | None = Field(default=None, validation_alias="LANGCHAIN_API_KEY")
    langchain_tracing_v2: bool = Field(default=False, validation_alias="LANGCHAIN_TRACING_V2")
    langchain_project: str = Field(default="king-ai-v2", validation_alias="LANGCHAIN_PROJECT")
    
    # --- Email Settings (Gmail) ---
    gmail_user: str | None = Field(default=None, validation_alias="GMAIL_USER")
    gmail_app_password: str | None = Field(default=None, validation_alias="GMAIL_APP_PASSWORD")
    notification_email: str | None = Field(default=None, validation_alias="NOTIFICATION_EMAIL")
    
    # --- Hugging Face Settings ---
    hugging_face_api_keys: str = Field(default="", validation_alias="HUGGING_FACE_API_KEYS")
    
    # --- Risk & Evolution Controls ---
    # Controls how much autonomy the AI has: conservative, moderate, or aggressive
    risk_profile: Literal["conservative", "moderate", "aggressive"] = Field(default="moderate", validation_alias="RISK_PROFILE")
    # Max number of self-modification proposals allowed per hour
    max_evolutions_per_hour: int = Field(default=5, validation_alias="MAX_EVOLUTIONS_PER_HOUR")
    # Enable autonomous operation mode (self-driven without user prompts)
    enable_autonomous_mode: bool = Field(default=False, validation_alias="ENABLE_AUTONOMOUS_MODE")
    # Enable self-modification capabilities
    enable_self_modification: bool = Field(default=True, validation_alias="ENABLE_SELF_MODIFICATION")
    # Confidence threshold for evolution proposals (0.0 - 1.0)
    evolution_confidence_threshold: float = Field(default=0.8, validation_alias="EVOLUTION_CONFIDENCE_THRESHOLD")
    # Daily limit for evolution proposals
    evolution_daily_limit: int = Field(default=120, validation_alias="EVOLUTION_DAILY_LIMIT")
    # Sandbox timeout for testing evolution proposals (seconds)
    evolution_sandbox_timeout: int = Field(default=300, validation_alias="EVOLUTION_SANDBOX_TIMEOUT")
    # Require tests to pass before applying evolution
    evolution_require_tests: bool = Field(default=True, validation_alias="EVOLUTION_REQUIRE_TESTS")
    
    # --- Approval Settings ---
    # Maximum amount (in dollars) that can be auto-approved
    max_auto_approve_amount: float = Field(default=100.0, validation_alias="MAX_AUTO_APPROVE_AMOUNT")
    # Hours before an approval request expires
    approval_expiry_hours: int = Field(default=24, validation_alias="APPROVAL_EXPIRY_HOURS")
    # Default auto-approve setting for new approvals
    auto_approve_default: bool = Field(default=True, validation_alias="AUTO_APPROVE_DEFAULT")
    # Require approval for legal decisions
    require_approval_legal: bool = Field(default=True, validation_alias="REQUIRE_APPROVAL_LEGAL")
    # Require approval for financial decisions
    require_approval_financial: bool = Field(default=True, validation_alias="REQUIRE_APPROVAL_FINANCIAL")

    # --- Scheduler Settings ---
    # Enable background task scheduler
    enable_scheduler: bool = Field(default=True, validation_alias="ENABLE_SCHEDULER")
    # Default interval for KPI review (hours)
    kpi_review_interval_hours: int = Field(default=6, validation_alias="KPI_REVIEW_INTERVAL_HOURS")
    # Default interval for health checks (hours)
    health_check_interval_hours: int = Field(default=1, validation_alias="HEALTH_CHECK_INTERVAL_HOURS")
    # Daily summary email hour (24-hour format)
    daily_summary_hour: int = Field(default=18, validation_alias="DAILY_SUMMARY_HOUR")
    # Daily summary email minute
    daily_summary_minute: int = Field(default=0, validation_alias="DAILY_SUMMARY_MINUTE")
    # Timezone for scheduled tasks
    timezone: str = Field(default="America/Chicago", validation_alias="TIMEZONE")
    
    # --- API Server Settings ---
    # Host for the FastAPI server
    api_host: str = Field(default="0.0.0.0", validation_alias="API_HOST")
    # Port for the FastAPI server
    api_port: int = Field(default=8000, validation_alias="API_PORT")
    # Enable rate limiting on API endpoints
    enable_rate_limiting: bool = Field(default=True, validation_alias="ENABLE_RATE_LIMITING")
    # Number of requests allowed per rate limit window
    rate_limit_requests: int = Field(default=100, validation_alias="RATE_LIMIT_REQUESTS")
    # Rate limit window in seconds
    rate_limit_window: int = Field(default=60, validation_alias="RATE_LIMIT_WINDOW")

    # --- File Paths ---
    # Path for audit logs
    audit_log_path: str = Field(default="./data/audit-logs", validation_alias="AUDIT_LOG_PATH")
    # Path for document storage
    documents_path: str = Field(default="./data/documents", validation_alias="DOCUMENTS_PATH")

    # --- Docker/Sandbox Settings ---
    # Enable Docker sandbox for code execution
    docker_sandbox_enabled: bool = Field(default=True, validation_alias="DOCKER_SANDBOX_ENABLED")
    # Docker image to use for sandbox
    docker_image: str = Field(default="node:20-alpine", validation_alias="DOCKER_IMAGE")

    # --- Business Limits ---
    # Maximum number of concurrent businesses
    max_concurrent_businesses: int = Field(default=3, validation_alias="MAX_CONCURRENT_BUSINESSES")
    # Primary model to use for decisions
    primary_model: str = Field(default="ollama", validation_alias="PRIMARY_MODEL")

    # --- Email Notification Settings (SMTP) ---
    # SMTP host for email notifications
    smtp_host: str = Field(default="smtp.gmail.com", validation_alias="SMTP_HOST")
    # SMTP port
    smtp_port: int = Field(default=587, validation_alias="SMTP_PORT")
    # SMTP username
    smtp_user: str | None = Field(default=None, validation_alias="SMTP_USER")
    # SMTP password/app password
    smtp_password: str | None = Field(default=None, validation_alias="SMTP_PASSWORD")
    # From email address for notifications
    smtp_from_email: str = Field(default="noreply@king-ai.com", validation_alias="SMTP_FROM_EMAIL")
    # Use AWS SES instead of SMTP
    use_ses: bool = Field(default=False, validation_alias="USE_SES")
    # AWS region for SES
    aws_region: str = Field(default="us-east-1", validation_alias="AWS_REGION")

    # --- Supabase Settings (Dashboard) ---
    # Supabase URL for dashboard
    supabase_url: str | None = Field(default=None, validation_alias="NEXT_PUBLIC_SUPABASE_URL")
    # Supabase public key for dashboard
    supabase_publishable_key: str | None = Field(default=None, validation_alias="NEXT_PUBLIC_SUPABASE_PUBLISHABLE_DEFAULT_KEY")

    # --- Agentic Framework Settings (King AI v3) ---
    # URL for the agentic framework orchestrator
    agentic_orchestrator_url: str = Field(default="http://localhost:8001", validation_alias="AGENTIC_ORCHESTRATOR_URL")
    # URL for the MCP gateway
    mcp_gateway_url: str = Field(default="http://localhost:3000", validation_alias="MCP_GATEWAY_URL")
    # URL for the memory service
    memory_service_url: str = Field(default="http://localhost:8002", validation_alias="MEMORY_SERVICE_URL")
    # Target server for Ralph code agent
    ralph_target_server: str = Field(default="100.24.50.240", validation_alias="RALPH_TARGET_SERVER")

# Singleton instance
settings = Settings()
