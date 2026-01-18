"""
Memory Service Configuration.

Module: memory-service/service/config.py
"""

from typing import Literal, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Memory service configuration settings."""

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra="ignore")

    # Service settings
    host: str = "0.0.0.0"
    port: int = 8001
    service_name: str = "memory-service"

    # Redis configuration
    redis_url: str = "redis://localhost:6379/0"
    session_ttl_hours: int = 72

    # PostgreSQL configuration
    postgres_url: str = "postgresql://user:password@localhost:5432/agentic_framework"

    # Vector database configuration
    vector_db_type: Literal["milvus", "chroma"] = "chroma"
    milvus_url: str = "http://localhost:19530"
    chroma_path: str = "./data/chroma"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    # MinIO/S3 configuration
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket: str = "agent-artifacts"
    minio_secure: bool = False

    # Memory configuration
    memory_retention_years: int = 2
    memory_compaction_threshold_tokens: int = 8000
    max_provenance_depth: int = 100

    # Code Executor configuration (for summarization skill)
    code_exec_url: str = "http://localhost:8004"

    # Observability
    otel_exporter_otlp_endpoint: Optional[str] = None
    prometheus_port: int = 9091


settings = Settings()
