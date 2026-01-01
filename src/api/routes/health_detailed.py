"""
Health Check Aggregator.
Comprehensive health checks for all services with Kubernetes probe support.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Coroutine, Dict, List, Optional
from enum import Enum

from fastapi import APIRouter, Response
from pydantic import BaseModel

from src.utils.structured_logging import get_logger

logger = get_logger("health")


class HealthStatus(str, Enum):
    """Service health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class CheckType(str, Enum):
    """Type of health check."""
    READINESS = "readiness"  # Can accept traffic
    LIVENESS = "liveness"   # Is alive
    STARTUP = "startup"     # Has started successfully


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    name: str
    status: HealthStatus
    message: Optional[str] = None
    latency_ms: float = 0.0
    checked_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "latency_ms": round(self.latency_ms, 2),
            "checked_at": self.checked_at.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class ServiceHealth:
    """Overall service health status."""
    status: HealthStatus
    version: str
    uptime_seconds: float
    checks: List[HealthCheckResult]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "version": self.version,
            "uptime_seconds": round(self.uptime_seconds, 2),
            "timestamp": self.timestamp.isoformat(),
            "checks": [c.to_dict() for c in self.checks],
        }


HealthChecker = Callable[[], Coroutine[Any, Any, HealthCheckResult]]


class HealthAggregator:
    """
    Aggregates health checks from all services.
    
    Features:
    - Multiple check types (readiness, liveness, startup)
    - Parallel check execution
    - Caching to prevent check flooding
    - Kubernetes probe compatible
    """
    
    def __init__(
        self,
        version: str = "1.0.0",
        cache_ttl: float = 5.0,
    ):
        self.version = version
        self.cache_ttl = cache_ttl
        self._start_time = datetime.utcnow()
        self._checks: Dict[str, Dict[str, HealthChecker]] = {
            CheckType.READINESS.value: {},
            CheckType.LIVENESS.value: {},
            CheckType.STARTUP.value: {},
        }
        self._cache: Dict[str, tuple[datetime, ServiceHealth]] = {}
        self._started = False
    
    def register_check(
        self,
        name: str,
        checker: HealthChecker,
        check_type: CheckType = CheckType.READINESS,
    ) -> None:
        """Register a health check."""
        self._checks[check_type.value][name] = checker
        logger.debug(f"Registered health check: {name} ({check_type.value})")
    
    def mark_started(self) -> None:
        """Mark the service as started."""
        self._started = True
    
    async def check(
        self,
        check_type: CheckType = CheckType.READINESS,
        use_cache: bool = True,
    ) -> ServiceHealth:
        """
        Run all health checks of the specified type.
        
        Args:
            check_type: Type of checks to run
            use_cache: Whether to use cached results
            
        Returns:
            Aggregated health status
        """
        cache_key = check_type.value
        
        # Check cache
        if use_cache and cache_key in self._cache:
            cached_time, cached_result = self._cache[cache_key]
            if (datetime.utcnow() - cached_time).total_seconds() < self.cache_ttl:
                return cached_result
        
        # Run checks in parallel
        checkers = self._checks[check_type.value]
        
        if not checkers:
            # No checks registered, assume healthy
            result = ServiceHealth(
                status=HealthStatus.HEALTHY,
                version=self.version,
                uptime_seconds=self._uptime_seconds(),
                checks=[],
            )
            self._cache[cache_key] = (datetime.utcnow(), result)
            return result
        
        # Execute all checks
        check_results = await asyncio.gather(
            *[self._run_check(name, checker) for name, checker in checkers.items()],
            return_exceptions=True,
        )
        
        # Process results
        results: List[HealthCheckResult] = []
        for r in check_results:
            if isinstance(r, Exception):
                results.append(HealthCheckResult(
                    name="unknown",
                    status=HealthStatus.UNHEALTHY,
                    message=str(r),
                ))
            else:
                results.append(r)
        
        # Determine overall status
        overall_status = self._aggregate_status(results)
        
        result = ServiceHealth(
            status=overall_status,
            version=self.version,
            uptime_seconds=self._uptime_seconds(),
            checks=results,
        )
        
        # Cache result
        self._cache[cache_key] = (datetime.utcnow(), result)
        
        return result
    
    async def _run_check(
        self,
        name: str,
        checker: HealthChecker,
    ) -> HealthCheckResult:
        """Run a single health check with timing."""
        start = datetime.utcnow()
        
        try:
            result = await asyncio.wait_for(checker(), timeout=10.0)
            result.latency_ms = (datetime.utcnow() - start).total_seconds() * 1000
            return result
        
        except asyncio.TimeoutError:
            return HealthCheckResult(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message="Health check timed out",
                latency_ms=(datetime.utcnow() - start).total_seconds() * 1000,
            )
        
        except Exception as e:
            return HealthCheckResult(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Check failed: {str(e)}",
                latency_ms=(datetime.utcnow() - start).total_seconds() * 1000,
            )
    
    def _aggregate_status(self, results: List[HealthCheckResult]) -> HealthStatus:
        """Aggregate individual check statuses."""
        if not results:
            return HealthStatus.HEALTHY
        
        statuses = [r.status for r in results]
        
        if all(s == HealthStatus.HEALTHY for s in statuses):
            return HealthStatus.HEALTHY
        elif any(s == HealthStatus.UNHEALTHY for s in statuses):
            return HealthStatus.UNHEALTHY
        elif any(s == HealthStatus.DEGRADED for s in statuses):
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.UNKNOWN
    
    def _uptime_seconds(self) -> float:
        """Get service uptime in seconds."""
        return (datetime.utcnow() - self._start_time).total_seconds()
    
    def is_healthy(self, check_type: CheckType = CheckType.READINESS) -> bool:
        """Quick check if service is healthy."""
        cache_key = check_type.value
        if cache_key in self._cache:
            _, result = self._cache[cache_key]
            return result.status == HealthStatus.HEALTHY
        return True  # Assume healthy if no checks run yet


# Global instance
health_aggregator = HealthAggregator()


def get_health_aggregator() -> HealthAggregator:
    """Get the global health aggregator instance."""
    return health_aggregator


# Pre-built health checkers
async def check_database() -> HealthCheckResult:
    """Check database connectivity."""
    try:
        from src.database.connection_pool import get_pool
        pool = await get_pool()
        
        if pool.is_healthy:
            metrics = pool.get_metrics()
            return HealthCheckResult(
                name="database",
                status=HealthStatus.HEALTHY,
                message="Database connection pool healthy",
                metadata=metrics,
            )
        else:
            return HealthCheckResult(
                name="database",
                status=HealthStatus.UNHEALTHY,
                message="Database connection pool unhealthy",
            )
    except Exception as e:
        return HealthCheckResult(
            name="database",
            status=HealthStatus.UNHEALTHY,
            message=f"Database check failed: {str(e)}",
        )


async def check_redis() -> HealthCheckResult:
    """Check Redis connectivity."""
    try:
        from config.settings import settings
        import redis.asyncio as redis
        
        client = redis.from_url(settings.redis_url)
        await client.ping()
        await client.close()
        
        return HealthCheckResult(
            name="redis",
            status=HealthStatus.HEALTHY,
            message="Redis connection successful",
        )
    except Exception as e:
        return HealthCheckResult(
            name="redis",
            status=HealthStatus.UNHEALTHY,
            message=f"Redis check failed: {str(e)}",
        )


async def check_llm() -> HealthCheckResult:
    """Check LLM provider connectivity."""
    try:
        from src.utils.llm_router import get_router
        router = get_router()
        
        # Try to get provider health
        if hasattr(router, 'get_health'):
            health = await router.get_health()
            status = HealthStatus.HEALTHY if health.get("available") else HealthStatus.DEGRADED
            return HealthCheckResult(
                name="llm",
                status=status,
                message="LLM provider available",
                metadata=health,
            )
        
        return HealthCheckResult(
            name="llm",
            status=HealthStatus.HEALTHY,
            message="LLM router initialized",
        )
    except Exception as e:
        return HealthCheckResult(
            name="llm",
            status=HealthStatus.DEGRADED,
            message=f"LLM check: {str(e)}",
        )


async def check_task_queue() -> HealthCheckResult:
    """Check task queue status."""
    try:
        from src.services.task_queue import get_task_queue
        queue = get_task_queue()
        
        stats = await queue.get_stats()
        
        status = HealthStatus.HEALTHY
        if stats.get("dead_letter_queue", 0) > 100:
            status = HealthStatus.DEGRADED
        
        return HealthCheckResult(
            name="task_queue",
            status=status,
            message="Task queue operational",
            metadata=stats,
        )
    except Exception as e:
        return HealthCheckResult(
            name="task_queue",
            status=HealthStatus.DEGRADED,
            message=f"Task queue check: {str(e)}",
        )


# FastAPI router
def create_health_router(aggregator: Optional[HealthAggregator] = None) -> APIRouter:
    """
    Create FastAPI router for health endpoints.
    
    Usage:
        from src.api.routes.health_detailed import create_health_router
        
        app.include_router(create_health_router())
    """
    router = APIRouter(tags=["Health"])
    agg = aggregator or health_aggregator
    
    @router.get("/health")
    async def health():
        """Basic health check."""
        return {"status": "ok"}
    
    @router.get("/health/ready")
    async def readiness(response: Response):
        """Kubernetes readiness probe."""
        result = await agg.check(CheckType.READINESS)
        
        if result.status != HealthStatus.HEALTHY:
            response.status_code = 503
        
        return result.to_dict()
    
    @router.get("/health/live")
    async def liveness(response: Response):
        """Kubernetes liveness probe."""
        result = await agg.check(CheckType.LIVENESS)
        
        if result.status == HealthStatus.UNHEALTHY:
            response.status_code = 503
        
        return result.to_dict()
    
    @router.get("/health/startup")
    async def startup(response: Response):
        """Kubernetes startup probe."""
        if not agg._started:
            response.status_code = 503
            return {"status": "starting", "ready": False}
        
        result = await agg.check(CheckType.STARTUP)
        
        if result.status != HealthStatus.HEALTHY:
            response.status_code = 503
        
        return result.to_dict()
    
    @router.get("/health/detailed")
    async def detailed():
        """Detailed health information."""
        readiness = await agg.check(CheckType.READINESS, use_cache=False)
        
        return {
            "service": "king-ai",
            "version": agg.version,
            "status": readiness.status.value,
            "uptime_seconds": readiness.uptime_seconds,
            "checks": [c.to_dict() for c in readiness.checks],
            "timestamp": datetime.utcnow().isoformat(),
        }
    
    return router


def setup_default_checks(aggregator: Optional[HealthAggregator] = None) -> None:
    """Setup default health checks."""
    agg = aggregator or health_aggregator
    
    # Readiness checks
    agg.register_check("database", check_database, CheckType.READINESS)
    agg.register_check("redis", check_redis, CheckType.READINESS)
    agg.register_check("llm", check_llm, CheckType.READINESS)
    agg.register_check("task_queue", check_task_queue, CheckType.READINESS)
    
    # Liveness checks (simpler, just verify process is responsive)
    async def simple_liveness() -> HealthCheckResult:
        return HealthCheckResult(
            name="process",
            status=HealthStatus.HEALTHY,
            message="Process responsive",
        )
    
    agg.register_check("process", simple_liveness, CheckType.LIVENESS)
    
    # Startup checks
    agg.register_check("database", check_database, CheckType.STARTUP)
    
    logger.info("Default health checks configured")
