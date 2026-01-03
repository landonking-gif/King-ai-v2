"""
MCP Gateway Client for Kautilya.

Module: kautilya/mcp_gateway_client.py

Provides client interface to interact with the MCP Gateway service
for registering and managing external MCP servers.
"""

from typing import Any, Optional, Dict, List
import os
import httpx
from pydantic import BaseModel, Field


class ToolParameter(BaseModel):
    """Schema for a tool parameter."""

    name: str
    type: str
    description: str
    required: bool = False
    default: Optional[Any] = None


class ToolSchema(BaseModel):
    """Schema definition for a single tool."""

    name: str
    description: str
    parameters: List[ToolParameter] = Field(default_factory=list)
    returns: Optional[str] = None


class RateLimitConfig(BaseModel):
    """Rate limiting configuration."""

    max_calls: int
    window_seconds: int


class MCPServerRegistration(BaseModel):
    """Registration schema for an external MCP server."""

    tool_id: str
    name: str
    version: str
    owner: str
    contact: str
    endpoint: Optional[str] = None  # Required for SSE transport
    tools: List[ToolSchema]
    auth_flow: str = "none"  # none, api_key, oauth2, ephemeral_token
    transport: str = "sse"  # sse (remote) or stdio (local subprocess)
    command: Optional[List[str]] = None  # Required for stdio transport
    classification: List[str] = Field(default_factory=list)
    rate_limits: Optional[RateLimitConfig] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MCPGatewayClient:
    """Client for MCP Gateway API."""

    def __init__(self, gateway_url: Optional[str] = None, timeout: float = 30.0):
        """
        Initialize MCP Gateway client.

        Args:
            gateway_url: Gateway base URL (defaults to env var or localhost)
            timeout: Request timeout in seconds
        """
        self.gateway_url = (
            gateway_url
            or os.getenv("MCP_GATEWAY_URL", "http://localhost:8080")
        ).rstrip("/")
        self.client = httpx.AsyncClient(timeout=timeout)
        self.sync_client = httpx.Client(timeout=timeout)

    async def register_server(
        self, registration: MCPServerRegistration
    ) -> Dict[str, Any]:
        """
        Register an external MCP server with the gateway.

        Args:
            registration: Server registration details

        Returns:
            Registration response

        Raises:
            httpx.HTTPStatusError: If registration fails
        """
        response = await self.client.post(
            f"{self.gateway_url}/catalog/register",
            json=registration.model_dump(),
        )
        response.raise_for_status()
        return response.json()

    def register_server_sync(
        self, registration: MCPServerRegistration
    ) -> Dict[str, Any]:
        """Synchronous version of register_server."""
        response = self.sync_client.post(
            f"{self.gateway_url}/catalog/register",
            json=registration.model_dump(),
        )
        response.raise_for_status()
        return response.json()

    async def list_servers(self, enabled_only: bool = True) -> List[Dict[str, Any]]:
        """
        List all registered MCP servers.

        Args:
            enabled_only: If True, return only enabled servers

        Returns:
            List of server catalog entries
        """
        response = await self.client.get(
            f"{self.gateway_url}/catalog/tools", params={"enabled_only": enabled_only}
        )
        response.raise_for_status()
        data = response.json()
        return data.get("servers", [])

    def list_servers_sync(self, enabled_only: bool = True) -> List[Dict[str, Any]]:
        """Synchronous version of list_servers."""
        response = self.sync_client.get(
            f"{self.gateway_url}/catalog/tools", params={"enabled_only": enabled_only}
        )
        response.raise_for_status()
        data = response.json()
        return data.get("servers", [])

    async def test_connection(self) -> bool:
        """
        Test connection to MCP Gateway.

        Returns:
            True if gateway is reachable and healthy
        """
        try:
            response = await self.client.get(f"{self.gateway_url}/health")
            response.raise_for_status()
            return True
        except Exception:
            return False

    def test_connection_sync(self) -> bool:
        """Synchronous version of test_connection."""
        try:
            response = self.sync_client.get(f"{self.gateway_url}/health")
            response.raise_for_status()
            return True
        except Exception:
            return False

    async def unregister_server(self, tool_id: str) -> Dict[str, Any]:
        """
        Unregister a server from the gateway.

        Args:
            tool_id: Server ID to unregister

        Returns:
            Unregistration response
        """
        response = await self.client.delete(
            f"{self.gateway_url}/catalog/tools/{tool_id}"
        )
        response.raise_for_status()
        return response.json()

    def unregister_server_sync(self, tool_id: str) -> Dict[str, Any]:
        """Synchronous version of unregister_server."""
        response = self.sync_client.delete(
            f"{self.gateway_url}/catalog/tools/{tool_id}"
        )
        response.raise_for_status()
        return response.json()

    async def enable_server(self, tool_id: str) -> Dict[str, Any]:
        """Enable a registered server."""
        response = await self.client.patch(
            f"{self.gateway_url}/catalog/tools/{tool_id}/enable"
        )
        response.raise_for_status()
        return response.json()

    def enable_server_sync(self, tool_id: str) -> Dict[str, Any]:
        """Synchronous version of enable_server."""
        response = self.sync_client.patch(
            f"{self.gateway_url}/catalog/tools/{tool_id}/enable"
        )
        response.raise_for_status()
        return response.json()

    async def disable_server(self, tool_id: str) -> Dict[str, Any]:
        """Disable a registered server."""
        response = await self.client.patch(
            f"{self.gateway_url}/catalog/tools/{tool_id}/disable"
        )
        response.raise_for_status()
        return response.json()

    def disable_server_sync(self, tool_id: str) -> Dict[str, Any]:
        """Synchronous version of disable_server."""
        response = self.sync_client.patch(
            f"{self.gateway_url}/catalog/tools/{tool_id}/disable"
        )
        response.raise_for_status()
        return response.json()

    async def close(self) -> None:
        """Close HTTP client."""
        await self.client.aclose()

    def close_sync(self) -> None:
        """Close synchronous HTTP client."""
        self.sync_client.close()

    def __enter__(self):
        """Context manager support."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup."""
        self.close_sync()
