"""
Graceful Shutdown Handler.
Manages clean application shutdown with request draining and state persistence.
"""

import asyncio
import signal
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Coroutine, Any, Optional, List, Dict
from enum import Enum

from src.utils.structured_logging import get_logger

logger = get_logger("shutdown")


class ShutdownState(str, Enum):
    """Application shutdown state."""
    RUNNING = "running"
    DRAINING = "draining"
    SHUTTING_DOWN = "shutting_down"
    STOPPED = "stopped"


@dataclass
class ShutdownConfig:
    """Shutdown configuration."""
    drain_timeout: float = 30.0  # Max time to wait for requests to drain
    shutdown_timeout: float = 60.0  # Max total shutdown time
    force_after: float = 90.0  # Force exit after this time
    grace_period: float = 5.0  # Grace period before starting drain


@dataclass
class ActiveRequest:
    """Tracks an active request."""
    id: str
    path: str
    method: str
    started_at: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def duration_seconds(self) -> float:
        return (datetime.utcnow() - self.started_at).total_seconds()


ShutdownCallback = Callable[[], Coroutine[Any, Any, None]]


class GracefulShutdownManager:
    """
    Manages graceful application shutdown.
    
    Features:
    - Request draining before shutdown
    - Callback registration for cleanup tasks
    - State persistence for pending approvals
    - Configurable timeouts
    - Signal handling (SIGTERM, SIGINT)
    """
    
    def __init__(self, config: Optional[ShutdownConfig] = None):
        self.config = config or ShutdownConfig()
        self._state = ShutdownState.RUNNING
        self._active_requests: Dict[str, ActiveRequest] = {}
        self._cleanup_callbacks: List[tuple[str, ShutdownCallback, int]] = []  # (name, callback, priority)
        self._shutdown_event = asyncio.Event()
        self._drain_complete = asyncio.Event()
        self._shutdown_started_at: Optional[datetime] = None
        self._lock = asyncio.Lock()
    
    def register_cleanup(
        self,
        name: str,
        callback: ShutdownCallback,
        priority: int = 50,
    ) -> None:
        """
        Register a cleanup callback for shutdown.
        
        Args:
            name: Descriptive name for logging
            callback: Async function to call during shutdown
            priority: Lower numbers run first (0-100)
        """
        self._cleanup_callbacks.append((name, callback, priority))
        # Sort by priority
        self._cleanup_callbacks.sort(key=lambda x: x[2])
        logger.debug(f"Registered cleanup callback: {name} (priority={priority})")
    
    def track_request(self, request_id: str, path: str, method: str) -> None:
        """Track an active request."""
        if self._state != ShutdownState.RUNNING:
            raise ShutdownInProgressError("Server is shutting down")
        
        self._active_requests[request_id] = ActiveRequest(
            id=request_id,
            path=path,
            method=method,
        )
    
    def complete_request(self, request_id: str) -> None:
        """Mark a request as complete."""
        self._active_requests.pop(request_id, None)
        
        # Check if drain is complete
        if self._state == ShutdownState.DRAINING and not self._active_requests:
            self._drain_complete.set()
    
    @property
    def is_shutting_down(self) -> bool:
        """Check if shutdown is in progress."""
        return self._state != ShutdownState.RUNNING
    
    @property
    def active_request_count(self) -> int:
        """Get count of active requests."""
        return len(self._active_requests)
    
    def get_active_requests(self) -> List[Dict[str, Any]]:
        """Get list of active requests."""
        return [
            {
                "id": req.id,
                "path": req.path,
                "method": req.method,
                "duration_seconds": req.duration_seconds,
            }
            for req in self._active_requests.values()
        ]
    
    async def initiate_shutdown(self, reason: str = "requested") -> None:
        """
        Initiate graceful shutdown.
        
        Args:
            reason: Why shutdown was initiated
        """
        async with self._lock:
            if self._state != ShutdownState.RUNNING:
                logger.warning("Shutdown already in progress")
                return
            
            self._shutdown_started_at = datetime.utcnow()
            logger.info(f"Initiating graceful shutdown: {reason}")
            
            # Grace period
            logger.info(f"Grace period: {self.config.grace_period}s")
            await asyncio.sleep(self.config.grace_period)
            
            # Start draining
            await self._drain_requests()
            
            # Run cleanup callbacks
            await self._run_cleanup()
            
            self._state = ShutdownState.STOPPED
            self._shutdown_event.set()
            logger.info("Shutdown complete")
    
    async def _drain_requests(self) -> None:
        """Wait for active requests to complete."""
        self._state = ShutdownState.DRAINING
        
        if not self._active_requests:
            logger.info("No active requests to drain")
            self._drain_complete.set()
            return
        
        logger.info(
            f"Draining {len(self._active_requests)} active requests",
            timeout=self.config.drain_timeout,
        )
        
        try:
            await asyncio.wait_for(
                self._drain_complete.wait(),
                timeout=self.config.drain_timeout,
            )
            logger.info("All requests drained successfully")
        except asyncio.TimeoutError:
            remaining = len(self._active_requests)
            logger.warning(
                f"Drain timeout reached, {remaining} requests still active",
                requests=self.get_active_requests(),
            )
    
    async def _run_cleanup(self) -> None:
        """Run all registered cleanup callbacks."""
        self._state = ShutdownState.SHUTTING_DOWN
        
        for name, callback, priority in self._cleanup_callbacks:
            try:
                logger.info(f"Running cleanup: {name}")
                await asyncio.wait_for(
                    callback(),
                    timeout=10.0,  # Per-callback timeout
                )
                logger.info(f"Cleanup complete: {name}")
            except asyncio.TimeoutError:
                logger.error(f"Cleanup timeout: {name}")
            except Exception as e:
                logger.error(f"Cleanup error in {name}: {e}")
    
    async def wait_for_shutdown(self) -> None:
        """Wait for shutdown to complete."""
        await self._shutdown_event.wait()
    
    def setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        loop = asyncio.get_event_loop()
        
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(
                    sig,
                    lambda s=sig: asyncio.create_task(
                        self._handle_signal(s)
                    ),
                )
                logger.debug(f"Registered signal handler for {sig.name}")
            except NotImplementedError:
                # Windows doesn't support add_signal_handler
                signal.signal(sig, self._sync_signal_handler)
    
    def _sync_signal_handler(self, signum: int, frame) -> None:
        """Synchronous signal handler for Windows."""
        logger.info(f"Received signal {signum}")
        asyncio.create_task(self.initiate_shutdown(f"signal {signum}"))
    
    async def _handle_signal(self, sig: signal.Signals) -> None:
        """Handle shutdown signal."""
        logger.info(f"Received {sig.name}")
        await self.initiate_shutdown(f"signal {sig.name}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current shutdown status."""
        elapsed = None
        if self._shutdown_started_at:
            elapsed = (datetime.utcnow() - self._shutdown_started_at).total_seconds()
        
        return {
            "state": self._state.value,
            "active_requests": self.active_request_count,
            "cleanup_callbacks": len(self._cleanup_callbacks),
            "shutdown_elapsed_seconds": elapsed,
        }


class ShutdownInProgressError(Exception):
    """Raised when trying to accept requests during shutdown."""
    pass


# Global shutdown manager instance
shutdown_manager = GracefulShutdownManager()


def get_shutdown_manager() -> GracefulShutdownManager:
    """Get the global shutdown manager instance."""
    return shutdown_manager


# FastAPI middleware helper
class ShutdownMiddleware:
    """
    FastAPI middleware for graceful shutdown handling.
    
    Usage:
        app = FastAPI()
        app.add_middleware(ShutdownMiddleware)
    """
    
    def __init__(self, app):
        self.app = app
        self.manager = shutdown_manager
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        # Check if shutting down
        if self.manager.is_shutting_down:
            # Return 503 Service Unavailable
            await send({
                "type": "http.response.start",
                "status": 503,
                "headers": [
                    [b"content-type", b"application/json"],
                    [b"retry-after", b"30"],
                ],
            })
            await send({
                "type": "http.response.body",
                "body": b'{"error": "Server is shutting down", "retry_after": 30}',
            })
            return
        
        # Generate request ID
        import uuid
        request_id = str(uuid.uuid4())
        path = scope.get("path", "/")
        method = scope.get("method", "GET")
        
        try:
            self.manager.track_request(request_id, path, method)
            await self.app(scope, receive, send)
        finally:
            self.manager.complete_request(request_id)


# FastAPI lifespan helper
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan_with_shutdown(app):
    """
    FastAPI lifespan context manager with graceful shutdown.
    
    Usage:
        from src.api.shutdown import lifespan_with_shutdown
        
        app = FastAPI(lifespan=lifespan_with_shutdown)
    """
    # Startup
    shutdown_manager.setup_signal_handlers()
    
    # Register cleanup for connection pool
    try:
        from src.database.connection_pool import connection_pool
        shutdown_manager.register_cleanup(
            "database_pool",
            connection_pool.close,
            priority=90,  # Close database last
        )
    except ImportError:
        pass
    
    yield
    
    # Shutdown
    await shutdown_manager.initiate_shutdown("application shutdown")
