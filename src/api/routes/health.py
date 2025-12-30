"""
Health Check Routes - Comprehensive system health monitoring.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
from datetime import datetime
import asyncio

from config.settings import settings
from src.utils.llm_router import LLMRouter, ProviderType
from src.database.connection import get_db
from sqlalchemy import text

router = APIRouter()


class ProviderHealth(BaseModel):
    """Health status for a single provider."""
    name: str
    healthy: bool
    latency_ms: Optional[float] = None
    error: Optional[str] = None


class ComponentHealth(BaseModel):
    """Health status for a system component."""
    name: str
    healthy: bool
    latency_ms: Optional[float] = None
    details: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class SystemHealth(BaseModel):
    """Complete system health report."""
    status: str  # "healthy", "degraded", "unhealthy"
    timestamp: str
    version: str
    environment: str
    components: Dict[str, ComponentHealth]
    llm_providers: Dict[str, ProviderHealth]


@router.get("/", response_model=SystemHealth)
async def full_health_check():
    """
    Comprehensive health check for all system components.
    Returns detailed status for monitoring and alerting.
    """
    import time
    
    components = {}
    llm_providers = {}
    
    # Check Database
    try:
        start = time.time()
        async with get_db() as db:
            await db.execute(text("SELECT 1"))
        latency = (time.time() - start) * 1000
        components["database"] = ComponentHealth(
            name="PostgreSQL",
            healthy=True,
            latency_ms=latency
        )
    except Exception as e:
        components["database"] = ComponentHealth(
            name="PostgreSQL",
            healthy=False,
            error=str(e)
        )
    
    # Check Redis
    try:
        import redis.asyncio as redis
        start = time.time()
        r = redis.from_url(settings.redis_url)
        await r.ping()
        await r.close()
        latency = (time.time() - start) * 1000
        components["redis"] = ComponentHealth(
            name="Redis",
            healthy=True,
            latency_ms=latency
        )
    except Exception as e:
        components["redis"] = ComponentHealth(
            name="Redis",
            healthy=False,
            error=str(e)
        )
    
    # Check LLM Providers
    llm_router = LLMRouter()
    provider_health = await llm_router.health_check_all()
    
    for provider, healthy in provider_health.items():
        llm_providers[provider.value] = ProviderHealth(
            name=provider.value,
            healthy=healthy
        )
    
    # Check Vector Store
    try:
        if settings.pinecone_api_key:
            from src.database.vector_store import VectorStore
            vs = VectorStore()
            if vs.index:
                components["vector_store"] = ComponentHealth(
                    name="Pinecone",
                    healthy=True,
                    details={"index": settings.pinecone_index}
                )
            else:
                components["vector_store"] = ComponentHealth(
                    name="Pinecone",
                    healthy=False,
                    error="Index not available"
                )
        else:
            components["vector_store"] = ComponentHealth(
                name="Pinecone",
                healthy=False,
                error="Not configured"
            )
    except Exception as e:
        components["vector_store"] = ComponentHealth(
            name="Pinecone",
            healthy=False,
            error=str(e)
        )
    
    # Determine overall status
    all_healthy = all(c.healthy for c in components.values())
    any_llm_healthy = any(p.healthy for p in llm_providers.values())
    
    if all_healthy and any_llm_healthy:
        status = "healthy"
    elif components.get("database", ComponentHealth(name="db", healthy=False)).healthy and any_llm_healthy:
        status = "degraded"
    else:
        status = "unhealthy"
    
    return SystemHealth(
        status=status,
        timestamp=datetime.now().isoformat(),
        version="2.0.0",
        environment=settings.risk_profile,
        components=components,
        llm_providers=llm_providers
    )


@router.get("/ready")
async def readiness_check():
    """
    Kubernetes readiness probe.
    Returns 200 if the service can accept traffic.
    """
    try:
        async with get_db() as db:
            await db.execute(text("SELECT 1"))
        return {"ready": True}
    except:
        raise HTTPException(status_code=503, detail="Service not ready")


@router.get("/live")
async def liveness_check():
    """
    Kubernetes liveness probe.
    Returns 200 if the service is alive.
    """
    return {"alive": True}
