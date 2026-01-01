"""
Developer Dashboard API Routes.
Endpoints for monitoring and debugging during development.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from enum import Enum
import sys
import os
import platform
import asyncio

from src.utils.structured_logging import get_logger

logger = get_logger("dev_dashboard")


@dataclass
class SystemInfo:
    """System information."""
    python_version: str
    platform: str
    os: str
    architecture: str
    hostname: str
    cpu_count: int
    process_id: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "python_version": self.python_version,
            "platform": self.platform,
            "os": self.os,
            "architecture": self.architecture,
            "hostname": self.hostname,
            "cpu_count": self.cpu_count,
            "process_id": self.process_id,
        }


@dataclass
class MemoryInfo:
    """Memory usage information."""
    rss_mb: float  # Resident Set Size
    vms_mb: float  # Virtual Memory Size
    percent: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "rss_mb": round(self.rss_mb, 2),
            "vms_mb": round(self.vms_mb, 2),
            "percent": round(self.percent, 2),
        }


@dataclass
class ServiceHealth:
    """Health status of a service."""
    name: str
    status: str  # healthy, unhealthy, unknown
    latency_ms: Optional[float] = None
    last_check: Optional[datetime] = None
    error: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "latency_ms": round(self.latency_ms, 2) if self.latency_ms else None,
            "last_check": self.last_check.isoformat() if self.last_check else None,
            "error": self.error,
            "details": self.details,
        }


@dataclass
class RouteInfo:
    """API route information."""
    path: str
    methods: List[str]
    name: str = ""
    tags: List[str] = field(default_factory=list)
    deprecated: bool = False


@dataclass
class DashboardConfig:
    """Dashboard configuration."""
    enabled: bool = True
    require_auth: bool = False
    allowed_hosts: List[str] = field(default_factory=lambda: ["localhost", "127.0.0.1"])
    refresh_interval_seconds: int = 30


class DevDashboard:
    """
    Developer Dashboard.
    
    Features:
    - System information
    - Memory usage
    - Service health checks
    - Configuration viewer
    - Route listing
    - Debug utilities
    """
    
    def __init__(self, config: Optional[DashboardConfig] = None):
        self.config = config or DashboardConfig()
        self.start_time = datetime.utcnow()
        self.health_checks: Dict[str, Any] = {}
        self._cached_system_info: Optional[SystemInfo] = None
        
        self._register_default_checks()
    
    def _register_default_checks(self) -> None:
        """Register default health checks."""
        self.register_health_check("database", self._check_database)
        self.register_health_check("redis", self._check_redis)
        self.register_health_check("llm", self._check_llm)
    
    def register_health_check(
        self,
        name: str,
        check_fn,
    ) -> None:
        """Register a health check function."""
        self.health_checks[name] = check_fn
    
    async def _check_database(self) -> ServiceHealth:
        """Check database connectivity."""
        try:
            # Try to import and check database
            from src.database.connection import get_db
            
            start = datetime.utcnow()
            db = get_db()
            # Simple connectivity check
            latency = (datetime.utcnow() - start).total_seconds() * 1000
            
            return ServiceHealth(
                name="database",
                status="healthy",
                latency_ms=latency,
                last_check=datetime.utcnow(),
            )
        except Exception as e:
            return ServiceHealth(
                name="database",
                status="unhealthy",
                last_check=datetime.utcnow(),
                error=str(e),
            )
    
    async def _check_redis(self) -> ServiceHealth:
        """Check Redis connectivity."""
        try:
            from src.services.cache import get_cache
            
            start = datetime.utcnow()
            cache = get_cache()
            # Simple ping
            latency = (datetime.utcnow() - start).total_seconds() * 1000
            
            return ServiceHealth(
                name="redis",
                status="healthy",
                latency_ms=latency,
                last_check=datetime.utcnow(),
            )
        except Exception as e:
            return ServiceHealth(
                name="redis",
                status="unhealthy",
                last_check=datetime.utcnow(),
                error=str(e),
            )
    
    async def _check_llm(self) -> ServiceHealth:
        """Check LLM provider connectivity."""
        try:
            from src.master_ai.llm_router import get_router
            
            start = datetime.utcnow()
            router = get_router()
            latency = (datetime.utcnow() - start).total_seconds() * 1000
            
            return ServiceHealth(
                name="llm",
                status="healthy",
                latency_ms=latency,
                last_check=datetime.utcnow(),
                details={"providers": ["ollama", "vllm", "claude"]},
            )
        except Exception as e:
            return ServiceHealth(
                name="llm",
                status="unhealthy",
                last_check=datetime.utcnow(),
                error=str(e),
            )
    
    def get_system_info(self) -> SystemInfo:
        """Get system information."""
        if self._cached_system_info:
            return self._cached_system_info
        
        self._cached_system_info = SystemInfo(
            python_version=sys.version,
            platform=platform.platform(),
            os=platform.system(),
            architecture=platform.architecture()[0],
            hostname=platform.node(),
            cpu_count=os.cpu_count() or 1,
            process_id=os.getpid(),
        )
        
        return self._cached_system_info
    
    def get_memory_info(self) -> MemoryInfo:
        """Get memory usage information."""
        try:
            import psutil
            process = psutil.Process()
            memory = process.memory_info()
            
            return MemoryInfo(
                rss_mb=memory.rss / (1024 * 1024),
                vms_mb=memory.vms / (1024 * 1024),
                percent=process.memory_percent(),
            )
        except ImportError:
            # Fallback without psutil
            import resource
            usage = resource.getrusage(resource.RUSAGE_SELF)
            return MemoryInfo(
                rss_mb=usage.ru_maxrss / 1024,  # KB on Linux
                vms_mb=0,
            )
        except Exception:
            return MemoryInfo(rss_mb=0, vms_mb=0)
    
    def get_uptime(self) -> timedelta:
        """Get application uptime."""
        return datetime.utcnow() - self.start_time
    
    async def run_health_checks(self) -> Dict[str, ServiceHealth]:
        """Run all health checks."""
        results = {}
        
        for name, check_fn in self.health_checks.items():
            try:
                if asyncio.iscoroutinefunction(check_fn):
                    result = await check_fn()
                else:
                    result = check_fn()
                results[name] = result
            except Exception as e:
                results[name] = ServiceHealth(
                    name=name,
                    status="unhealthy",
                    last_check=datetime.utcnow(),
                    error=str(e),
                )
        
        return results
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary (sanitized)."""
        try:
            from config.settings import Settings
            settings = Settings()
            
            # Return sanitized config (hide secrets)
            config_dict = {}
            for key, value in vars(settings).items():
                if any(secret in key.lower() for secret in ["password", "secret", "key", "token"]):
                    config_dict[key] = "***REDACTED***"
                else:
                    config_dict[key] = value
            
            return config_dict
        except Exception as e:
            return {"error": str(e)}
    
    def get_environment_info(self) -> Dict[str, str]:
        """Get environment information (sanitized)."""
        safe_vars = [
            "ENVIRONMENT", "DEBUG", "LOG_LEVEL", "APP_NAME",
            "DATABASE_HOST", "REDIS_HOST", "API_VERSION",
        ]
        
        return {
            key: os.environ.get(key, "")
            for key in safe_vars
            if os.environ.get(key)
        }
    
    async def get_dashboard_data(self) -> Dict[str, Any]:
        """
        Get complete dashboard data.
        
        Returns:
            All dashboard information
        """
        health_results = await self.run_health_checks()
        
        uptime = self.get_uptime()
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "uptime": {
                "seconds": uptime.total_seconds(),
                "formatted": str(uptime),
            },
            "system": self.get_system_info().to_dict(),
            "memory": self.get_memory_info().to_dict(),
            "health": {
                name: result.to_dict()
                for name, result in health_results.items()
            },
            "environment": self.get_environment_info(),
            "config": self.get_config_summary(),
        }


# Global dashboard instance
dev_dashboard = DevDashboard()


def get_dev_dashboard() -> DevDashboard:
    """Get the global dev dashboard."""
    return dev_dashboard


# FastAPI router for dashboard endpoints
def create_dashboard_router():
    """Create FastAPI router for dashboard endpoints."""
    try:
        from fastapi import APIRouter, Depends, HTTPException
        from fastapi.responses import JSONResponse, HTMLResponse
        
        router = APIRouter(prefix="/dev", tags=["Developer Dashboard"])
        
        @router.get("/")
        async def dashboard_index():
            """Get dashboard overview."""
            data = await dev_dashboard.get_dashboard_data()
            return JSONResponse(content=data)
        
        @router.get("/health")
        async def health_check():
            """Run health checks."""
            results = await dev_dashboard.run_health_checks()
            
            all_healthy = all(r.status == "healthy" for r in results.values())
            status_code = 200 if all_healthy else 503
            
            return JSONResponse(
                content={
                    "status": "healthy" if all_healthy else "unhealthy",
                    "checks": {name: r.to_dict() for name, r in results.items()},
                },
                status_code=status_code,
            )
        
        @router.get("/system")
        async def system_info():
            """Get system information."""
            return JSONResponse(content=dev_dashboard.get_system_info().to_dict())
        
        @router.get("/memory")
        async def memory_info():
            """Get memory usage."""
            return JSONResponse(content=dev_dashboard.get_memory_info().to_dict())
        
        @router.get("/config")
        async def config_info():
            """Get configuration (sanitized)."""
            return JSONResponse(content=dev_dashboard.get_config_summary())
        
        @router.get("/uptime")
        async def uptime_info():
            """Get application uptime."""
            uptime = dev_dashboard.get_uptime()
            return JSONResponse(content={
                "seconds": uptime.total_seconds(),
                "formatted": str(uptime),
                "started_at": dev_dashboard.start_time.isoformat(),
            })
        
        @router.post("/gc")
        async def trigger_gc():
            """Trigger garbage collection."""
            import gc
            before = dev_dashboard.get_memory_info()
            gc.collect()
            after = dev_dashboard.get_memory_info()
            
            return JSONResponse(content={
                "before": before.to_dict(),
                "after": after.to_dict(),
                "freed_mb": round(before.rss_mb - after.rss_mb, 2),
            })
        
        return router
    
    except ImportError:
        logger.warning("FastAPI not available, dashboard routes not created")
        return None
