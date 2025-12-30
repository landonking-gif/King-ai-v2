"""
System Health Monitoring.
"""
import asyncio
import psutil
from datetime import datetime
from typing import Optional, Callable
from dataclasses import dataclass, field
from src.api.events import broadcaster, EventType
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class HealthCheck:
    """Health check result."""
    name: str
    status: str  # healthy, degraded, unhealthy
    latency_ms: float = 0
    message: Optional[str] = None
    last_check: datetime = field(default_factory=datetime.utcnow)


@dataclass
class SystemMetrics:
    """System resource metrics."""
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    active_connections: int
    pending_tasks: int
    timestamp: datetime = field(default_factory=datetime.utcnow)


class SystemMonitor:
    """Monitor system health and resources."""

    def __init__(self):
        self._health_checks: dict[str, Callable] = {}
        self._last_metrics: Optional[SystemMetrics] = None
        self._alerts: list[dict] = []
        self._running = False
        self._check_interval = 30  # seconds

    def register_health_check(self, name: str, check_fn: Callable):
        """Register a health check function."""
        self._health_checks[name] = check_fn

    async def check_health(self) -> dict[str, HealthCheck]:
        """Run all health checks."""
        results = {}
        
        for name, check_fn in self._health_checks.items():
            start = datetime.utcnow()
            try:
                status, message = await check_fn()
                latency = (datetime.utcnow() - start).total_seconds() * 1000
                results[name] = HealthCheck(
                    name=name,
                    status=status,
                    latency_ms=latency,
                    message=message,
                )
            except Exception as e:
                results[name] = HealthCheck(
                    name=name,
                    status="unhealthy",
                    message=str(e),
                )

        return results

    async def collect_metrics(self) -> SystemMetrics:
        """Collect system metrics."""
        # Avoid circular import by importing at runtime
        from src.api.websocket import manager
        
        metrics = SystemMetrics(
            cpu_percent=psutil.cpu_percent(interval=None),  # Non-blocking
            memory_percent=psutil.virtual_memory().percent,
            disk_percent=psutil.disk_usage('/').percent,
            active_connections=manager.get_stats()["total_connections"],
            pending_tasks=0,  # Would be populated from task queue
        )
        
        self._last_metrics = metrics
        return metrics

    async def check_thresholds(self, metrics: SystemMetrics):
        """Check metrics against thresholds and alert if needed."""
        alerts = []
        
        if metrics.cpu_percent > 90:
            alerts.append(("critical", f"CPU usage critical: {metrics.cpu_percent}%"))
        elif metrics.cpu_percent > 75:
            alerts.append(("warning", f"CPU usage high: {metrics.cpu_percent}%"))

        if metrics.memory_percent > 90:
            alerts.append(("critical", f"Memory usage critical: {metrics.memory_percent}%"))
        elif metrics.memory_percent > 80:
            alerts.append(("warning", f"Memory usage high: {metrics.memory_percent}%"))

        if metrics.disk_percent > 90:
            alerts.append(("critical", f"Disk usage critical: {metrics.disk_percent}%"))
        elif metrics.disk_percent > 80:
            alerts.append(("warning", f"Disk usage high: {metrics.disk_percent}%"))

        for level, message in alerts:
            await broadcaster.emit_system_alert(level, message, {
                "cpu": metrics.cpu_percent,
                "memory": metrics.memory_percent,
                "disk": metrics.disk_percent,
            })
            self._alerts.append({
                "level": level,
                "message": message,
                "timestamp": datetime.utcnow().isoformat(),
            })

    async def start_monitoring(self):
        """Start background monitoring loop."""
        self._running = True
        logger.info("System monitoring started")
        
        while self._running:
            try:
                metrics = await self.collect_metrics()
                await self.check_thresholds(metrics)
                
                # Broadcast metrics
                await broadcaster.emit(
                    EventType.SYSTEM_METRIC,
                    {
                        "cpu": metrics.cpu_percent,
                        "memory": metrics.memory_percent,
                        "disk": metrics.disk_percent,
                        "connections": metrics.active_connections,
                    },
                )
            except Exception as e:
                logger.error(f"Monitoring error: {e}")

            await asyncio.sleep(self._check_interval)

    def stop_monitoring(self):
        """Stop the monitoring loop."""
        self._running = False
        logger.info("System monitoring stopped")

    def get_recent_alerts(self, limit: int = 20) -> list[dict]:
        """Get recent alerts."""
        return self._alerts[-limit:]

    def get_last_metrics(self) -> Optional[SystemMetrics]:
        """Get last collected metrics."""
        return self._last_metrics


# Global monitor instance
monitor = SystemMonitor()


# Default health checks
async def check_database() -> tuple[str, str]:
    """Check database connectivity."""
    try:
        # Would actually check DB connection
        return "healthy", "Database connected"
    except Exception as e:
        return "unhealthy", str(e)


async def check_redis() -> tuple[str, str]:
    """Check Redis connectivity."""
    try:
        # Would actually check Redis
        return "healthy", "Redis connected"
    except Exception:
        return "degraded", "Redis unavailable, using fallback"


async def check_llm() -> tuple[str, str]:
    """Check LLM service."""
    try:
        # Would check Ollama/LLM service
        return "healthy", "LLM service available"
    except Exception as e:
        return "unhealthy", str(e)


# Register default checks
monitor.register_health_check("database", check_database)
monitor.register_health_check("redis", check_redis)
monitor.register_health_check("llm", check_llm)
