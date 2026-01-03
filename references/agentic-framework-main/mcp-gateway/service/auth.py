"""
Authentication and token management for MCP Gateway.

Provides JWT-based ephemeral token generation and validation for scoped direct access.
"""

from typing import Optional
from datetime import datetime, timedelta
import secrets
from jose import JWTError, jwt

from .config import settings
from .models import TokenScope, EphemeralTokenRequest, EphemeralTokenResponse


class TokenManager:
    """Manager for ephemeral JWT tokens with scoping and TTL."""

    def __init__(self) -> None:
        """Initialize token manager with configuration."""
        self.secret_key = settings.jwt_secret_key
        self.algorithm = settings.jwt_algorithm
        self.default_ttl_minutes = settings.ephemeral_token_ttl_minutes

    async def mint_token(self, request: EphemeralTokenRequest) -> EphemeralTokenResponse:
        """
        Generate an ephemeral JWT token with specified scope and TTL.

        Args:
            request: Token request with scope and optional TTL

        Returns:
            Token response with JWT and expiration details
        """
        ttl_minutes = request.ttl_minutes or self.default_ttl_minutes
        expires_at = datetime.utcnow() + timedelta(minutes=ttl_minutes)

        # Create JWT payload
        payload = {
            "jti": secrets.token_urlsafe(16),  # JWT ID for tracking
            "iat": datetime.utcnow().timestamp(),  # Issued at
            "exp": expires_at.timestamp(),  # Expiration
            "scope": {
                "tool_ids": request.scope.tool_ids,
                "actor_id": request.scope.actor_id,
                "actor_type": request.scope.actor_type,
                "max_invocations": request.scope.max_invocations,
            },
            "invocation_count": 0,  # Track usage
        }

        # Generate JWT
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

        return EphemeralTokenResponse(
            token=token, expires_at=expires_at, scope=request.scope
        )

    async def validate_token(self, token: str) -> Optional[TokenScope]:
        """
        Validate a JWT token and extract its scope.

        Args:
            token: JWT token to validate

        Returns:
            Token scope if valid, None if invalid or expired
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])

            # Extract scope
            scope_data = payload.get("scope")
            if not scope_data:
                return None

            scope = TokenScope(
                tool_ids=scope_data["tool_ids"],
                actor_id=scope_data["actor_id"],
                actor_type=scope_data["actor_type"],
                max_invocations=scope_data.get("max_invocations"),
            )

            return scope

        except JWTError:
            return None

    async def check_token_permission(
        self, token: str, tool_id: str
    ) -> tuple[bool, Optional[str]]:
        """
        Check if a token has permission to access a specific tool.

        Args:
            token: JWT token
            tool_id: Tool ID to check access for

        Returns:
            Tuple of (has_permission, error_message)
        """
        scope = await self.validate_token(token)

        if scope is None:
            return False, "Invalid or expired token"

        if tool_id not in scope.tool_ids:
            return False, f"Token does not have permission for tool '{tool_id}'"

        # Check invocation count if max_invocations is set
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            invocation_count = payload.get("invocation_count", 0)

            if scope.max_invocations is not None and invocation_count >= scope.max_invocations:
                return False, "Token invocation limit exceeded"

        except JWTError:
            return False, "Invalid token"

        return True, None

    async def increment_invocation_count(self, token: str) -> Optional[str]:
        """
        Increment the invocation count in a token.

        Note: In production, this should be tracked in Redis/DB rather than in the JWT itself.
        For Sprint 0, we're demonstrating the pattern.

        Args:
            token: JWT token

        Returns:
            New token with incremented count, or None if invalid
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            payload["invocation_count"] = payload.get("invocation_count", 0) + 1

            # Re-encode with updated count
            new_token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
            return new_token

        except JWTError:
            return None


# Global token manager instance
_token_manager_instance: Optional[TokenManager] = None


def get_token_manager() -> TokenManager:
    """
    Get the global token manager instance (singleton pattern).

    Returns:
        Global TokenManager instance
    """
    global _token_manager_instance
    if _token_manager_instance is None:
        _token_manager_instance = TokenManager()
    return _token_manager_instance
