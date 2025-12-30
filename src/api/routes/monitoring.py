"""
Monitoring API Routes.
"""
from fastapi import APIRouter, WebSocket
from src.api.websocket import manager, websocket_endpoint
from src.monitoring.monitor import monitor
from src.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/monitoring", tags=["monitoring"])


@router.websocket("/ws")
async def websocket_route(
    websocket: WebSocket,
    user_id: str = None,
    business_id: str = None,
):
    """WebSocket endpoint for real-time updates."""
    await websocket_endpoint(websocket, user_id, business_id)


@router.get("/health")
async def health_check():
    """Get system health status."""
    checks = await monitor.check_health()
    
    overall = "healthy"
    for check in checks.values():
        if check.status == "unhealthy":
            overall = "unhealthy"
            break
        elif check.status == "degraded" and overall == "healthy":
            overall = "degraded"

    return {
        "status": overall,
        "checks": {
            name: {
                "status": c.status,
                "latency_ms": round(c.latency_ms, 2),
                "message": c.message,
            }
            for name, c in checks.items()
        },
    }


@router.get("/metrics")
async def get_metrics():
    """Get system metrics."""
    metrics = await monitor.collect_metrics()
    
    return {
        "cpu_percent": metrics.cpu_percent,
        "memory_percent": metrics.memory_percent,
        "disk_percent": metrics.disk_percent,
        "active_connections": metrics.active_connections,
        "pending_tasks": metrics.pending_tasks,
        "timestamp": metrics.timestamp.isoformat(),
    }


@router.get("/alerts")
async def get_alerts(limit: int = 20):
    """Get recent system alerts."""
    return {
        "alerts": monitor.get_recent_alerts(limit),
    }


@router.get("/connections")
async def get_connections():
    """Get WebSocket connection stats."""
    return manager.get_stats()
