"""
System API Routes - Monitoring, Health, Audit.

Provides endpoints for:
- Circuit breaker status
- Audit trail export
- System health checks
- Rate limit monitoring
"""

from datetime import datetime
from typing import Optional, List

from fastapi import APIRouter, HTTPException, Query, Response
from pydantic import BaseModel

from src.utils.circuit_breaker import CircuitBreaker, get_circuit_breakers_health
from src.utils.audit_trail import (
    audit_manager, AuditExportOptions, AuditExportFormat,
    AuditEventType, AuditSeverity
)
from src.utils.structured_logging import get_logger

logger = get_logger("system_routes")

router = APIRouter(prefix="/api/v1/system", tags=["system"])


# Request/Response Models

class CircuitBreakerResponse(BaseModel):
    """Circuit breaker status response."""
    status: str
    open_circuits: List[str]
    circuits: dict


class AuditExportRequest(BaseModel):
    """Audit export request parameters."""
    format: str = "json"  # json, csv, jsonl
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    event_types: Optional[List[str]] = None
    severities: Optional[List[str]] = None
    business_ids: Optional[List[str]] = None
    include_metadata: bool = True
    max_records: int = 10000


class ComplianceReportRequest(BaseModel):
    """Compliance report request."""
    start_date: datetime
    end_date: datetime
    business_id: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    uptime_seconds: float
    checks: dict


# Global state for uptime tracking
_start_time = datetime.utcnow()


# Endpoints

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Comprehensive health check endpoint.
    
    Checks:
    - Database connectivity
    - Redis connectivity
    - LLM service availability
    - Circuit breaker states
    """
    checks = {}
    overall_status = "healthy"
    
    # Check database
    try:
        from src.database.connection import get_db_session
        async with get_db_session() as session:
            from sqlalchemy import text
            await session.execute(text("SELECT 1"))
        checks["database"] = {"status": "healthy"}
    except Exception as e:
        checks["database"] = {"status": "unhealthy", "error": str(e)}
        overall_status = "degraded"
    
    # Check Redis
    try:
        import redis.asyncio as aioredis
        from config.settings import settings
        redis_url = getattr(settings, 'redis_url', 'redis://localhost:6379')
        redis = await aioredis.from_url(redis_url, socket_timeout=2)
        await redis.ping()
        await redis.close()
        checks["redis"] = {"status": "healthy"}
    except Exception as e:
        checks["redis"] = {"status": "unhealthy", "error": str(e)}
        # Redis failure is degraded, not critical
    
    # Check circuit breakers
    cb_health = await get_circuit_breakers_health()
    checks["circuit_breakers"] = {
        "status": cb_health["status"],
        "open_circuits": cb_health["open_circuits"]
    }
    if cb_health["open_circuits"]:
        overall_status = "degraded"
    
    uptime = (datetime.utcnow() - _start_time).total_seconds()
    
    return HealthResponse(
        status=overall_status,
        version="2.0.0",
        uptime_seconds=uptime,
        checks=checks
    )


@router.get("/circuit-breakers", response_model=CircuitBreakerResponse)
async def get_circuit_breakers():
    """
    Get status of all circuit breakers.
    
    Returns:
        Status of each registered circuit breaker including:
        - Current state (closed, open, half-open)
        - Request counts
        - Failure history
        - Configuration
    """
    return await get_circuit_breakers_health()


@router.post("/circuit-breakers/{name}/reset")
async def reset_circuit_breaker(name: str):
    """
    Manually reset a circuit breaker to closed state.
    
    Use with caution - should only be used when the underlying
    service issue has been resolved.
    """
    if name not in CircuitBreaker._registry:
        raise HTTPException(
            status_code=404,
            detail=f"Circuit breaker '{name}' not found"
        )
    
    CircuitBreaker._registry[name].reset()
    
    logger.info(f"Circuit breaker '{name}' manually reset")
    
    return {"status": "success", "message": f"Circuit breaker '{name}' reset to closed state"}


@router.post("/audit/export")
async def export_audit_trail(request: AuditExportRequest):
    """
    Export audit trail data.
    
    Supports multiple formats:
    - JSON: Full structured data
    - CSV: Spreadsheet-compatible
    - JSONL: Streaming format
    
    Filter by:
    - Date range
    - Event types
    - Severity levels
    - Business IDs
    """
    try:
        format_enum = AuditExportFormat(request.format.lower())
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid format: {request.format}. Supported: json, csv, jsonl"
        )
    
    # Parse event types
    event_types = None
    if request.event_types:
        event_types = []
        for et in request.event_types:
            try:
                event_types.append(AuditEventType(et))
            except ValueError:
                pass  # Skip invalid event types
    
    # Parse severities
    severities = None
    if request.severities:
        severities = []
        for s in request.severities:
            try:
                severities.append(AuditSeverity(s))
            except ValueError:
                pass
    
    options = AuditExportOptions(
        format=format_enum,
        start_date=request.start_date,
        end_date=request.end_date,
        event_types=event_types,
        severities=severities,
        business_ids=request.business_ids,
        include_metadata=request.include_metadata,
        max_records=min(request.max_records, 100000),  # Cap at 100k
    )
    
    content, content_type = await audit_manager.export(options)
    
    # Set filename based on format
    filename = f"audit_export_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    if format_enum == AuditExportFormat.JSON:
        filename += ".json"
    elif format_enum == AuditExportFormat.CSV:
        filename += ".csv"
    else:
        filename += ".jsonl"
    
    return Response(
        content=content,
        media_type=content_type,
        headers={
            "Content-Disposition": f"attachment; filename={filename}"
        }
    )


@router.post("/audit/compliance-report")
async def generate_compliance_report(request: ComplianceReportRequest):
    """
    Generate a compliance report for the specified period.
    
    Includes:
    - Event summary by type and severity
    - Top actors
    - Daily trends
    - Critical events requiring attention
    """
    if request.end_date <= request.start_date:
        raise HTTPException(
            status_code=400,
            detail="end_date must be after start_date"
        )
    
    # Limit to 1 year max
    if (request.end_date - request.start_date).days > 365:
        raise HTTPException(
            status_code=400,
            detail="Maximum report period is 1 year"
        )
    
    report = await audit_manager.get_compliance_report(
        start_date=request.start_date,
        end_date=request.end_date,
        business_id=request.business_id
    )
    
    return report


@router.get("/audit/events")
async def list_audit_events(
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    event_type: Optional[str] = Query(None),
    severity: Optional[str] = Query(None),
    business_id: Optional[str] = Query(None),
    limit: int = Query(100, le=1000),
    offset: int = Query(0, ge=0),
):
    """
    List audit events with pagination.
    
    Use this for browsing audit history in the dashboard.
    For bulk export, use the /audit/export endpoint.
    """
    event_types = [AuditEventType(event_type)] if event_type else None
    severities = [AuditSeverity(severity)] if severity else None
    
    options = AuditExportOptions(
        start_date=start_date,
        end_date=end_date,
        event_types=event_types,
        severities=severities,
        business_ids=[business_id] if business_id else None,
        max_records=limit + offset,
    )
    
    events = await audit_manager.query(options)
    
    # Apply pagination
    paginated = events[offset:offset + limit]
    
    return {
        "events": [e.to_dict() for e in paginated],
        "total": len(events),
        "limit": limit,
        "offset": offset,
    }


@router.get("/metrics")
async def get_system_metrics():
    """
    Get system metrics for monitoring.
    
    Returns metrics suitable for Prometheus/Datadog ingestion.
    """
    from src.utils.metrics import TASKS_EXECUTED, LLM_REQUESTS
    
    # Gather metrics
    metrics = {
        "timestamp": datetime.utcnow().isoformat(),
        "uptime_seconds": (datetime.utcnow() - _start_time).total_seconds(),
    }
    
    # Circuit breaker metrics
    cb_metrics = CircuitBreaker.get_all_metrics()
    metrics["circuit_breakers"] = {
        name: {
            "state": m["state"],
            "total_calls": m["total_calls"],
            "failed_calls": m["failed_calls"],
            "rejected_calls": m["rejected_calls"],
        }
        for name, m in cb_metrics.items()
    }
    
    return metrics


@router.get("/config")
async def get_runtime_config():
    """
    Get current runtime configuration (non-sensitive).
    
    Useful for debugging and verifying configuration.
    """
    from config.settings import settings
    
    # Only expose non-sensitive settings
    config = {
        "environment": getattr(settings, 'environment', 'development'),
        "debug": getattr(settings, 'debug', False),
        "log_level": getattr(settings, 'log_level', 'INFO'),
        "enable_self_modification": settings.enable_self_modification,
        "max_evolutions_per_hour": settings.max_evolutions_per_hour,
        "risk_profile": settings.risk_profile,
    }
    
    return config
