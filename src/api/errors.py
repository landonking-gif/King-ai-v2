"""
Structured Error Responses.
Consistent error handling across the API.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Type
from enum import Enum
import traceback
import uuid

from src.utils.structured_logging import get_logger

logger = get_logger("errors")


class ErrorCode(str, Enum):
    """Standard error codes."""
    # General errors (1xxx)
    UNKNOWN = "E1000"
    INTERNAL_ERROR = "E1001"
    NOT_IMPLEMENTED = "E1002"
    SERVICE_UNAVAILABLE = "E1003"
    TIMEOUT = "E1004"
    
    # Validation errors (2xxx)
    VALIDATION_ERROR = "E2000"
    MISSING_FIELD = "E2001"
    INVALID_FORMAT = "E2002"
    VALUE_OUT_OF_RANGE = "E2003"
    TYPE_MISMATCH = "E2004"
    CONSTRAINT_VIOLATION = "E2005"
    
    # Authentication/Authorization (3xxx)
    UNAUTHORIZED = "E3000"
    FORBIDDEN = "E3001"
    TOKEN_EXPIRED = "E3002"
    INVALID_TOKEN = "E3003"
    INSUFFICIENT_PERMISSIONS = "E3004"
    
    # Resource errors (4xxx)
    NOT_FOUND = "E4000"
    ALREADY_EXISTS = "E4001"
    CONFLICT = "E4002"
    GONE = "E4003"
    LOCKED = "E4004"
    
    # Rate limiting (5xxx)
    RATE_LIMITED = "E5000"
    QUOTA_EXCEEDED = "E5001"
    TOO_MANY_REQUESTS = "E5002"
    
    # External service errors (6xxx)
    EXTERNAL_SERVICE_ERROR = "E6000"
    LLM_ERROR = "E6001"
    DATABASE_ERROR = "E6002"
    REDIS_ERROR = "E6003"
    PAYMENT_ERROR = "E6004"
    
    # Business logic errors (7xxx)
    BUSINESS_RULE_VIOLATION = "E7000"
    APPROVAL_REQUIRED = "E7001"
    INSUFFICIENT_FUNDS = "E7002"
    OPERATION_NOT_ALLOWED = "E7003"


# HTTP status code mapping
ERROR_STATUS_CODES: Dict[ErrorCode, int] = {
    ErrorCode.UNKNOWN: 500,
    ErrorCode.INTERNAL_ERROR: 500,
    ErrorCode.NOT_IMPLEMENTED: 501,
    ErrorCode.SERVICE_UNAVAILABLE: 503,
    ErrorCode.TIMEOUT: 504,
    
    ErrorCode.VALIDATION_ERROR: 400,
    ErrorCode.MISSING_FIELD: 400,
    ErrorCode.INVALID_FORMAT: 400,
    ErrorCode.VALUE_OUT_OF_RANGE: 400,
    ErrorCode.TYPE_MISMATCH: 400,
    ErrorCode.CONSTRAINT_VIOLATION: 422,
    
    ErrorCode.UNAUTHORIZED: 401,
    ErrorCode.FORBIDDEN: 403,
    ErrorCode.TOKEN_EXPIRED: 401,
    ErrorCode.INVALID_TOKEN: 401,
    ErrorCode.INSUFFICIENT_PERMISSIONS: 403,
    
    ErrorCode.NOT_FOUND: 404,
    ErrorCode.ALREADY_EXISTS: 409,
    ErrorCode.CONFLICT: 409,
    ErrorCode.GONE: 410,
    ErrorCode.LOCKED: 423,
    
    ErrorCode.RATE_LIMITED: 429,
    ErrorCode.QUOTA_EXCEEDED: 429,
    ErrorCode.TOO_MANY_REQUESTS: 429,
    
    ErrorCode.EXTERNAL_SERVICE_ERROR: 502,
    ErrorCode.LLM_ERROR: 502,
    ErrorCode.DATABASE_ERROR: 503,
    ErrorCode.REDIS_ERROR: 503,
    ErrorCode.PAYMENT_ERROR: 502,
    
    ErrorCode.BUSINESS_RULE_VIOLATION: 422,
    ErrorCode.APPROVAL_REQUIRED: 403,
    ErrorCode.INSUFFICIENT_FUNDS: 402,
    ErrorCode.OPERATION_NOT_ALLOWED: 403,
}


@dataclass
class FieldError:
    """Error for a specific field."""
    field: str
    message: str
    code: str = "invalid"
    value: Any = None


@dataclass
class ErrorDetail:
    """Detailed error information."""
    code: ErrorCode
    message: str
    field_errors: List[FieldError] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)
    help_url: Optional[str] = None
    retry_after: Optional[int] = None  # seconds


@dataclass
class ErrorResponse:
    """Structured error response."""
    error: ErrorDetail
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    path: Optional[str] = None
    method: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON response."""
        response = {
            "error": {
                "code": self.error.code.value,
                "message": self.error.message,
            },
            "request_id": self.request_id,
            "timestamp": self.timestamp.isoformat(),
        }
        
        if self.error.field_errors:
            response["error"]["field_errors"] = [
                {
                    "field": fe.field,
                    "message": fe.message,
                    "code": fe.code,
                }
                for fe in self.error.field_errors
            ]
        
        if self.error.details:
            response["error"]["details"] = self.error.details
        
        if self.error.help_url:
            response["error"]["help_url"] = self.error.help_url
        
        if self.error.retry_after:
            response["error"]["retry_after"] = self.error.retry_after
        
        if self.path:
            response["path"] = self.path
        
        if self.method:
            response["method"] = self.method
        
        return response
    
    @property
    def status_code(self) -> int:
        """Get HTTP status code for this error."""
        return ERROR_STATUS_CODES.get(self.error.code, 500)


class APIException(Exception):
    """Base exception for API errors."""
    
    def __init__(
        self,
        code: ErrorCode,
        message: str,
        field_errors: List[FieldError] = None,
        details: Dict[str, Any] = None,
        help_url: Optional[str] = None,
        retry_after: Optional[int] = None,
    ):
        super().__init__(message)
        self.code = code
        self.message = message
        self.field_errors = field_errors or []
        self.details = details or {}
        self.help_url = help_url
        self.retry_after = retry_after
    
    def to_response(
        self,
        request_id: Optional[str] = None,
        path: Optional[str] = None,
        method: Optional[str] = None,
    ) -> ErrorResponse:
        """Convert to error response."""
        return ErrorResponse(
            error=ErrorDetail(
                code=self.code,
                message=self.message,
                field_errors=self.field_errors,
                details=self.details,
                help_url=self.help_url,
                retry_after=self.retry_after,
            ),
            request_id=request_id or str(uuid.uuid4()),
            path=path,
            method=method,
        )
    
    @property
    def status_code(self) -> int:
        return ERROR_STATUS_CODES.get(self.code, 500)


# Specific exception classes
class ValidationException(APIException):
    """Validation error exception."""
    
    def __init__(
        self,
        message: str = "Validation failed",
        field_errors: List[FieldError] = None,
        details: Dict[str, Any] = None,
    ):
        super().__init__(
            code=ErrorCode.VALIDATION_ERROR,
            message=message,
            field_errors=field_errors,
            details=details,
        )


class NotFoundException(APIException):
    """Resource not found exception."""
    
    def __init__(
        self,
        resource_type: str = "Resource",
        resource_id: Any = None,
    ):
        message = f"{resource_type} not found"
        if resource_id:
            message = f"{resource_type} with id '{resource_id}' not found"
        
        super().__init__(
            code=ErrorCode.NOT_FOUND,
            message=message,
            details={"resource_type": resource_type, "resource_id": resource_id},
        )


class UnauthorizedException(APIException):
    """Unauthorized exception."""
    
    def __init__(
        self,
        message: str = "Authentication required",
    ):
        super().__init__(
            code=ErrorCode.UNAUTHORIZED,
            message=message,
        )


class ForbiddenException(APIException):
    """Forbidden exception."""
    
    def __init__(
        self,
        message: str = "Access denied",
        required_permission: Optional[str] = None,
    ):
        details = {}
        if required_permission:
            details["required_permission"] = required_permission
        
        super().__init__(
            code=ErrorCode.FORBIDDEN,
            message=message,
            details=details,
        )


class RateLimitedException(APIException):
    """Rate limited exception."""
    
    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: int = 60,
        limit: Optional[int] = None,
    ):
        details = {"retry_after_seconds": retry_after}
        if limit:
            details["limit"] = limit
        
        super().__init__(
            code=ErrorCode.RATE_LIMITED,
            message=message,
            details=details,
            retry_after=retry_after,
        )


class ExternalServiceException(APIException):
    """External service error exception."""
    
    def __init__(
        self,
        service: str,
        message: str = None,
        original_error: Optional[str] = None,
    ):
        super().__init__(
            code=ErrorCode.EXTERNAL_SERVICE_ERROR,
            message=message or f"External service error: {service}",
            details={
                "service": service,
                "original_error": original_error,
            },
        )


class BusinessRuleException(APIException):
    """Business rule violation exception."""
    
    def __init__(
        self,
        rule: str,
        message: str,
        details: Dict[str, Any] = None,
    ):
        super().__init__(
            code=ErrorCode.BUSINESS_RULE_VIOLATION,
            message=message,
            details={"rule": rule, **(details or {})},
        )


class ErrorHandler:
    """
    Handles exceptions and converts to structured responses.
    
    Features:
    - Exception to response conversion
    - Logging with context
    - Stack trace handling (dev vs prod)
    - Exception mapping
    """
    
    def __init__(
        self,
        include_stack_trace: bool = False,
        log_errors: bool = True,
    ):
        self.include_stack_trace = include_stack_trace
        self.log_errors = log_errors
        self._exception_handlers: Dict[Type[Exception], ErrorCode] = {}
    
    def register_exception(
        self,
        exception_type: Type[Exception],
        error_code: ErrorCode,
    ) -> None:
        """Register exception type to error code mapping."""
        self._exception_handlers[exception_type] = error_code
    
    def handle(
        self,
        exception: Exception,
        request_id: Optional[str] = None,
        path: Optional[str] = None,
        method: Optional[str] = None,
    ) -> ErrorResponse:
        """Handle an exception and return structured response."""
        request_id = request_id or str(uuid.uuid4())
        
        # Handle APIException
        if isinstance(exception, APIException):
            response = exception.to_response(request_id, path, method)
            
            if self.log_errors:
                logger.warning(
                    f"API Error: {exception.code.value} - {exception.message}",
                    extra={"request_id": request_id, "path": path},
                )
            
            return response
        
        # Check registered handlers
        for exc_type, error_code in self._exception_handlers.items():
            if isinstance(exception, exc_type):
                return self._create_response(
                    error_code,
                    str(exception),
                    exception,
                    request_id,
                    path,
                    method,
                )
        
        # Handle unknown exceptions
        if self.log_errors:
            logger.error(
                f"Unhandled exception: {type(exception).__name__} - {str(exception)}",
                extra={
                    "request_id": request_id,
                    "path": path,
                    "traceback": traceback.format_exc(),
                },
            )
        
        return self._create_response(
            ErrorCode.INTERNAL_ERROR,
            "An internal error occurred",
            exception,
            request_id,
            path,
            method,
        )
    
    def _create_response(
        self,
        code: ErrorCode,
        message: str,
        exception: Exception,
        request_id: str,
        path: Optional[str],
        method: Optional[str],
    ) -> ErrorResponse:
        """Create error response from exception."""
        details = {}
        
        if self.include_stack_trace:
            details["exception_type"] = type(exception).__name__
            details["stack_trace"] = traceback.format_exc()
        
        return ErrorResponse(
            error=ErrorDetail(
                code=code,
                message=message,
                details=details if details else {},
            ),
            request_id=request_id,
            path=path,
            method=method,
        )


# FastAPI integration
def create_exception_handlers(handler: ErrorHandler):
    """Create FastAPI exception handlers."""
    from fastapi import Request
    from fastapi.responses import JSONResponse
    from fastapi.exceptions import RequestValidationError
    from starlette.exceptions import HTTPException as StarletteHTTPException
    
    async def api_exception_handler(request: Request, exc: APIException):
        """Handle APIException."""
        request_id = getattr(request.state, "request_id", None)
        response = exc.to_response(
            request_id=request_id,
            path=str(request.url.path),
            method=request.method,
        )
        return JSONResponse(
            status_code=response.status_code,
            content=response.to_dict(),
        )
    
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ):
        """Handle Pydantic validation errors."""
        field_errors = []
        for error in exc.errors():
            field = ".".join(str(loc) for loc in error["loc"])
            field_errors.append(FieldError(
                field=field,
                message=error["msg"],
                code=error["type"],
            ))
        
        api_exc = ValidationException(
            message="Request validation failed",
            field_errors=field_errors,
        )
        
        return await api_exception_handler(request, api_exc)
    
    async def http_exception_handler(
        request: Request, exc: StarletteHTTPException
    ):
        """Handle Starlette HTTP exceptions."""
        code_map = {
            400: ErrorCode.VALIDATION_ERROR,
            401: ErrorCode.UNAUTHORIZED,
            403: ErrorCode.FORBIDDEN,
            404: ErrorCode.NOT_FOUND,
            405: ErrorCode.OPERATION_NOT_ALLOWED,
            409: ErrorCode.CONFLICT,
            429: ErrorCode.RATE_LIMITED,
            500: ErrorCode.INTERNAL_ERROR,
            502: ErrorCode.EXTERNAL_SERVICE_ERROR,
            503: ErrorCode.SERVICE_UNAVAILABLE,
        }
        
        error_code = code_map.get(exc.status_code, ErrorCode.UNKNOWN)
        api_exc = APIException(
            code=error_code,
            message=str(exc.detail),
        )
        
        return await api_exception_handler(request, api_exc)
    
    async def generic_exception_handler(request: Request, exc: Exception):
        """Handle all other exceptions."""
        request_id = getattr(request.state, "request_id", None)
        response = handler.handle(
            exc,
            request_id=request_id,
            path=str(request.url.path),
            method=request.method,
        )
        return JSONResponse(
            status_code=response.status_code,
            content=response.to_dict(),
        )
    
    return {
        APIException: api_exception_handler,
        RequestValidationError: validation_exception_handler,
        StarletteHTTPException: http_exception_handler,
        Exception: generic_exception_handler,
    }


# Global error handler
error_handler = ErrorHandler(include_stack_trace=False)


def get_error_handler() -> ErrorHandler:
    """Get the global error handler."""
    return error_handler
