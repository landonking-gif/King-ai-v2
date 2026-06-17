"""
Authentication middleware for King AI v2 API.

Provides JWT-based authentication for all API routes except
public endpoints (health checks, webhooks).
"""

import hmac
import time
from typing import Optional, List

import jwt
from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from config.settings import settings
from src.utils.structured_logging import get_logger

logger = get_logger("auth")


# Paths that do NOT require authentication
PUBLIC_PATHS: List[str] = [
    "/api/health",
    "/api/health/",
    "/docs",
    "/openapi.json",
    "/redoc",
]

# Path prefixes that do NOT require authentication
PUBLIC_PREFIXES: List[str] = [
    "/api/webhooks",
]


class AuthMiddleware(BaseHTTPMiddleware):
    """
    JWT authentication middleware.

    Validates Bearer tokens on all routes except public paths.
    Also accepts X-API-Key for service-to-service auth.
    """

    def __init__(self, app, **kwargs):
        super().__init__(app, **kwargs)
        self.jwt_secret = settings.jwt_secret
        self.jwt_algorithm = settings.jwt_algorithm

    def _is_public_path(self, path: str) -> bool:
        """Check if the path is exempt from authentication."""
        if path in PUBLIC_PATHS:
            return True
        for prefix in PUBLIC_PREFIXES:
            if path.startswith(prefix):
                return True
        return False

    def _verify_jwt(self, token: str) -> Optional[dict]:
        """Verify and decode a JWT token."""
        try:
            payload = jwt.decode(
                token,
                self.jwt_secret,
                algorithms=[self.jwt_algorithm],
            )
            if "exp" in payload and payload["exp"] < time.time():
                return None
            return payload
        except jwt.InvalidTokenError:
            return None

    def _verify_api_key(self, api_key: str) -> bool:
        """Verify an API key (constant-time comparison)."""
        valid_keys = []
        if settings.stripe_api_key:
            valid_keys.append(settings.stripe_api_key)
        if settings.openai_api_key:
            valid_keys.append(settings.openai_api_key)
        for valid_key in valid_keys:
            if hmac.compare_digest(api_key, valid_key):
                return True
        return False

    async def dispatch(self, request: Request, call_next):
        """Authenticate request before passing to route handler."""
        path = request.url.path

        if self._is_public_path(path):
            return await call_next(request)

        if request.method == "OPTIONS":
            return await call_next(request)

        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
            payload = self._verify_jwt(token)
            if payload:
                request.state.user_id = payload.get("sub")
                request.state.user_role = payload.get("role", "user")
                request.state.authenticated = True
                return await call_next(request)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token",
                headers={"WWW-Authenticate": "Bearer"},
            )

        api_key = request.headers.get("X-API-Key")
        if api_key:
            if self._verify_api_key(api_key):
                request.state.user_id = "api_service"
                request.state.user_role = "service"
                request.state.authenticated = True
                return await call_next(request)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key",
            )

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
