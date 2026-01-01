"""
Database Connection Pool Manager.
Provides advanced connection pooling with health checks, auto-reconnect, and metrics.
"""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, AsyncIterator
from enum import Enum

from sqlalchemy.ext.asyncio import (
    create_async_engine,
    AsyncSession,
    async_sessionmaker,
    AsyncEngine,
)
from sqlalchemy import text, event
from sqlalchemy.pool import QueuePool

from src.utils.structured_logging import get_logger
from config.settings import settings

logger = get_logger("connection_pool")


class PoolHealth(str, Enum):
    """Connection pool health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    RECOVERING = "recovering"


@dataclass
class PoolMetrics:
    """Connection pool metrics."""
    total_connections: int = 0
    active_connections: int = 0
    idle_connections: int = 0
    overflow_connections: int = 0
    checked_out: int = 0
    checked_in: int = 0
    connections_created: int = 0
    connections_recycled: int = 0
    failed_connections: int = 0
    avg_acquisition_time_ms: float = 0.0
    last_health_check: Optional[datetime] = None
    health_status: PoolHealth = PoolHealth.HEALTHY


@dataclass
class PoolConfig:
    """Connection pool configuration."""
    min_size: int = 5
    max_size: int = 20
    max_overflow: int = 10
    pool_timeout: float = 30.0
    pool_recycle: int = 3600  # Recycle connections after 1 hour
    pool_pre_ping: bool = True
    echo: bool = False
    health_check_interval: int = 30
    retry_attempts: int = 3
    retry_delay: float = 1.0


class ConnectionPoolManager:
    """
    Advanced database connection pool manager.
    
    Features:
    - Connection pooling with configurable sizes
    - Health checks and auto-reconnect
    - Metrics tracking
    - Graceful degradation
    """
    
    def __init__(self, config: Optional[PoolConfig] = None):
        self.config = config or PoolConfig()
        self._engine: Optional[AsyncEngine] = None
        self._session_factory: Optional[async_sessionmaker] = None
        self._metrics = PoolMetrics()
        self._health_check_task: Optional[asyncio.Task] = None
        self._is_initialized = False
        self._acquisition_times: list[float] = []
        self._lock = asyncio.Lock()
    
    async def initialize(self, database_url: Optional[str] = None) -> None:
        """Initialize the connection pool."""
        if self._is_initialized:
            return
        
        async with self._lock:
            if self._is_initialized:
                return
            
            url = database_url or settings.database_url
            
            self._engine = create_async_engine(
                url,
                poolclass=QueuePool,
                pool_size=self.config.min_size,
                max_overflow=self.config.max_overflow,
                pool_timeout=self.config.pool_timeout,
                pool_recycle=self.config.pool_recycle,
                pool_pre_ping=self.config.pool_pre_ping,
                echo=self.config.echo,
            )
            
            self._session_factory = async_sessionmaker(
                self._engine,
                class_=AsyncSession,
                expire_on_commit=False,
            )
            
            # Register pool event listeners
            self._register_pool_events()
            
            # Start health check task
            self._health_check_task = asyncio.create_task(
                self._health_check_loop()
            )
            
            self._is_initialized = True
            logger.info(
                "Connection pool initialized",
                min_size=self.config.min_size,
                max_size=self.config.max_size + self.config.max_overflow,
            )
    
    def _register_pool_events(self) -> None:
        """Register SQLAlchemy pool event listeners."""
        if not self._engine:
            return
        
        pool = self._engine.pool
        
        @event.listens_for(pool, "checkout")
        def on_checkout(dbapi_conn, connection_record, connection_proxy):
            self._metrics.checked_out += 1
        
        @event.listens_for(pool, "checkin")
        def on_checkin(dbapi_conn, connection_record):
            self._metrics.checked_in += 1
        
        @event.listens_for(pool, "connect")
        def on_connect(dbapi_conn, connection_record):
            self._metrics.connections_created += 1
        
        @event.listens_for(pool, "invalidate")
        def on_invalidate(dbapi_conn, connection_record, exception):
            self._metrics.failed_connections += 1
            logger.warning("Connection invalidated", error=str(exception))
    
    async def _health_check_loop(self) -> None:
        """Periodic health check loop."""
        while True:
            try:
                await asyncio.sleep(self.config.health_check_interval)
                await self._perform_health_check()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Health check error", error=str(e))
    
    async def _perform_health_check(self) -> bool:
        """Perform a database health check."""
        try:
            async with self.acquire() as session:
                result = await session.execute(text("SELECT 1"))
                result.close()
            
            self._metrics.health_status = PoolHealth.HEALTHY
            self._metrics.last_health_check = datetime.utcnow()
            return True
        
        except Exception as e:
            logger.error("Health check failed", error=str(e))
            self._metrics.health_status = PoolHealth.UNHEALTHY
            self._metrics.last_health_check = datetime.utcnow()
            return False
    
    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[AsyncSession]:
        """
        Acquire a database session from the pool.
        
        Usage:
            async with pool.acquire() as session:
                result = await session.execute(query)
        """
        if not self._is_initialized:
            await self.initialize()
        
        start_time = datetime.utcnow()
        session: Optional[AsyncSession] = None
        
        for attempt in range(self.config.retry_attempts):
            try:
                session = self._session_factory()
                
                # Track acquisition time
                elapsed = (datetime.utcnow() - start_time).total_seconds() * 1000
                self._acquisition_times.append(elapsed)
                if len(self._acquisition_times) > 100:
                    self._acquisition_times.pop(0)
                self._metrics.avg_acquisition_time_ms = sum(self._acquisition_times) / len(self._acquisition_times)
                
                yield session
                
                await session.commit()
                break
                
            except Exception as e:
                if session:
                    await session.rollback()
                
                if attempt < self.config.retry_attempts - 1:
                    logger.warning(
                        "Database connection failed, retrying",
                        attempt=attempt + 1,
                        error=str(e),
                    )
                    await asyncio.sleep(self.config.retry_delay * (attempt + 1))
                else:
                    logger.error("Database connection failed after retries", error=str(e))
                    raise
            
            finally:
                if session:
                    await session.close()
    
    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[AsyncSession]:
        """
        Acquire a session with explicit transaction management.
        
        Usage:
            async with pool.transaction() as session:
                # All operations in a single transaction
                await session.execute(query1)
                await session.execute(query2)
        """
        async with self.acquire() as session:
            async with session.begin():
                yield session
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current pool metrics."""
        pool_status = {}
        if self._engine and hasattr(self._engine.pool, "status"):
            pool = self._engine.pool
            pool_status = {
                "size": pool.size(),
                "checked_in": pool.checkedin(),
                "checked_out": pool.checkedout(),
                "overflow": pool.overflow(),
            }
        
        return {
            "health_status": self._metrics.health_status.value,
            "last_health_check": self._metrics.last_health_check.isoformat() if self._metrics.last_health_check else None,
            "connections_created": self._metrics.connections_created,
            "failed_connections": self._metrics.failed_connections,
            "avg_acquisition_time_ms": round(self._metrics.avg_acquisition_time_ms, 2),
            "checked_out_total": self._metrics.checked_out,
            "checked_in_total": self._metrics.checked_in,
            **pool_status,
        }
    
    async def close(self) -> None:
        """Close the connection pool."""
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        if self._engine:
            await self._engine.dispose()
            self._engine = None
            self._session_factory = None
        
        self._is_initialized = False
        logger.info("Connection pool closed")
    
    async def refresh(self) -> None:
        """Refresh all connections in the pool."""
        if self._engine:
            await self._engine.dispose()
            
            # Re-create engine with same config
            await self.initialize()
            logger.info("Connection pool refreshed")
    
    @property
    def is_healthy(self) -> bool:
        """Check if the pool is healthy."""
        return self._metrics.health_status == PoolHealth.HEALTHY
    
    @property
    def engine(self) -> Optional[AsyncEngine]:
        """Get the underlying SQLAlchemy engine."""
        return self._engine


# Global connection pool instance
connection_pool = ConnectionPoolManager()


async def get_pool() -> ConnectionPoolManager:
    """Get the global connection pool instance."""
    if not connection_pool._is_initialized:
        await connection_pool.initialize()
    return connection_pool


@asynccontextmanager
async def get_db() -> AsyncIterator[AsyncSession]:
    """
    Get a database session from the pool.
    
    Convenience function for backward compatibility.
    """
    pool = await get_pool()
    async with pool.acquire() as session:
        yield session


# Alias for backward compatibility
get_db_session = get_db
