"""
Request Correlation ID Middleware.
Provides distributed tracing support with unique request IDs.
"""

import uuid
import time
import contextvars
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from fastapi import FastAPI

from src.utils.structured_logging import get_logger

logger = get_logger("correlation")

# Context variable for request correlation
correlation_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "correlation_id", default=None
)
request_context_var: contextvars.ContextVar[Optional[Dict[str, Any]]] = contextvars.ContextVar(
    "request_context", default=None
)


# Header names
CORRELATION_ID_HEADER = "X-Correlation-ID"
REQUEST_ID_HEADER = "X-Request-ID"
TRACE_ID_HEADER = "X-Trace-ID"
SPAN_ID_HEADER = "X-Span-ID"
PARENT_SPAN_HEADER = "X-Parent-Span-ID"


def get_correlation_id() -> Optional[str]:
    """Get the current correlation ID from context."""
    return correlation_id_var.get()


def set_correlation_id(correlation_id: str) -> None:
    """Set the correlation ID in context."""
    correlation_id_var.set(correlation_id)


def get_request_context() -> Optional[Dict[str, Any]]:
    """Get the current request context."""
    return request_context_var.get()


def generate_id() -> str:
    """Generate a unique ID for tracing."""
    return str(uuid.uuid4())


@dataclass
class TraceContext:
    """
    Distributed tracing context.
    
    Contains all IDs needed for distributed tracing across services.
    """
    correlation_id: str  # Unique ID for the entire request chain
    request_id: str  # Unique ID for this specific request
    trace_id: str  # Trace ID for distributed tracing
    span_id: str  # Span ID for this operation
    parent_span_id: Optional[str] = None  # Parent span for nested operations
    service_name: str = "king-ai"
    started_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_headers(self) -> Dict[str, str]:
        """Convert to HTTP headers for propagation."""
        headers = {
            CORRELATION_ID_HEADER: self.correlation_id,
            REQUEST_ID_HEADER: self.request_id,
            TRACE_ID_HEADER: self.trace_id,
            SPAN_ID_HEADER: self.span_id,
        }
        if self.parent_span_id:
            headers[PARENT_SPAN_HEADER] = self.parent_span_id
        return headers
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "correlation_id": self.correlation_id,
            "request_id": self.request_id,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "service": self.service_name,
        }
    
    def create_child_span(self) -> "TraceContext":
        """Create a child span for nested operations."""
        return TraceContext(
            correlation_id=self.correlation_id,
            request_id=self.request_id,
            trace_id=self.trace_id,
            span_id=generate_id(),
            parent_span_id=self.span_id,
            service_name=self.service_name,
        )
    
    @classmethod
    def from_headers(cls, headers: Dict[str, str]) -> "TraceContext":
        """Create TraceContext from incoming headers."""
        # Use existing IDs or generate new ones
        correlation_id = headers.get(CORRELATION_ID_HEADER) or generate_id()
        trace_id = headers.get(TRACE_ID_HEADER) or correlation_id
        parent_span_id = headers.get(SPAN_ID_HEADER)
        
        return cls(
            correlation_id=correlation_id,
            request_id=generate_id(),
            trace_id=trace_id,
            span_id=generate_id(),
            parent_span_id=parent_span_id,
        )


class CorrelationMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for request correlation and tracing.
    
    Features:
    - Generates or propagates correlation IDs
    - Adds request timing
    - Enriches logs with trace context
    - Propagates headers for distributed tracing
    """
    
    def __init__(
        self,
        app: FastAPI,
        service_name: str = "king-ai",
        log_requests: bool = True,
        log_responses: bool = True,
        exclude_paths: Optional[list[str]] = None,
    ):
        super().__init__(app)
        self.service_name = service_name
        self.log_requests = log_requests
        self.log_responses = log_responses
        self.exclude_paths = exclude_paths or ["/health", "/metrics", "/favicon.ico"]
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip excluded paths
        if request.url.path in self.exclude_paths:
            return await call_next(request)
        
        # Create trace context from headers
        headers = dict(request.headers)
        trace_context = TraceContext.from_headers(headers)
        trace_context.service_name = self.service_name
        
        # Set context variables
        correlation_id_var.set(trace_context.correlation_id)
        request_context_var.set({
            **trace_context.to_dict(),
            "method": request.method,
            "path": request.url.path,
            "client_ip": request.client.host if request.client else None,
        })
        
        # Log request
        if self.log_requests:
            logger.info(
                f"Request started: {request.method} {request.url.path}",
                **trace_context.to_dict(),
                method=request.method,
                path=request.url.path,
                query=str(request.query_params),
                client_ip=request.client.host if request.client else None,
            )
        
        # Process request
        start_time = time.perf_counter()
        
        try:
            response = await call_next(request)
        except Exception as e:
            # Log error
            logger.error(
                f"Request failed: {request.method} {request.url.path}",
                **trace_context.to_dict(),
                error=str(e),
                error_type=type(e).__name__,
            )
            raise
        
        # Calculate duration
        duration_ms = (time.perf_counter() - start_time) * 1000
        
        # Add correlation headers to response
        response.headers[CORRELATION_ID_HEADER] = trace_context.correlation_id
        response.headers[REQUEST_ID_HEADER] = trace_context.request_id
        response.headers[TRACE_ID_HEADER] = trace_context.trace_id
        response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"
        
        # Log response
        if self.log_responses:
            log_level = "info" if response.status_code < 400 else "warning"
            getattr(logger, log_level)(
                f"Request completed: {request.method} {request.url.path}",
                **trace_context.to_dict(),
                status_code=response.status_code,
                duration_ms=round(duration_ms, 2),
            )
        
        return response


def with_correlation(func: Callable) -> Callable:
    """
    Decorator to propagate correlation context to async functions.
    
    Usage:
        @with_correlation
        async def my_function():
            logger.info("This log includes correlation ID")
    """
    import functools
    
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        # Context is already set by middleware
        return await func(*args, **kwargs)
    
    return wrapper


class CorrelatedLogger:
    """
    Logger wrapper that automatically includes correlation context.
    
    Usage:
        logger = CorrelatedLogger("my_module")
        logger.info("This message includes correlation context")
    """
    
    def __init__(self, name: str):
        self._logger = get_logger(name)
    
    def _add_context(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Add correlation context to log kwargs."""
        correlation_id = get_correlation_id()
        if correlation_id:
            kwargs["correlation_id"] = correlation_id
        
        context = get_request_context()
        if context:
            kwargs.setdefault("trace_id", context.get("trace_id"))
            kwargs.setdefault("span_id", context.get("span_id"))
        
        return kwargs
    
    def debug(self, msg: str, **kwargs):
        self._logger.debug(msg, **self._add_context(kwargs))
    
    def info(self, msg: str, **kwargs):
        self._logger.info(msg, **self._add_context(kwargs))
    
    def warning(self, msg: str, **kwargs):
        self._logger.warning(msg, **self._add_context(kwargs))
    
    def error(self, msg: str, **kwargs):
        self._logger.error(msg, **self._add_context(kwargs))
    
    def critical(self, msg: str, **kwargs):
        self._logger.critical(msg, **self._add_context(kwargs))


def setup_correlation_middleware(app: FastAPI, service_name: str = "king-ai") -> None:
    """
    Setup correlation middleware on a FastAPI app.
    
    Usage:
        from src.api.middleware.correlation import setup_correlation_middleware
        
        app = FastAPI()
        setup_correlation_middleware(app)
    """
    app.add_middleware(
        CorrelationMiddleware,
        service_name=service_name,
    )


# For use in async HTTP client calls
def get_correlation_headers() -> Dict[str, str]:
    """
    Get correlation headers for outgoing HTTP requests.
    
    Usage:
        async with aiohttp.ClientSession() as session:
            headers = get_correlation_headers()
            async with session.get(url, headers=headers) as response:
                ...
    """
    correlation_id = get_correlation_id()
    context = get_request_context()
    
    headers = {}
    if correlation_id:
        headers[CORRELATION_ID_HEADER] = correlation_id
    
    if context:
        if context.get("trace_id"):
            headers[TRACE_ID_HEADER] = context["trace_id"]
        if context.get("span_id"):
            headers[SPAN_ID_HEADER] = context["span_id"]
    
    return headers
