"""
Pydantic models for MCP Gateway.

Defines data models for tool catalog, registration, invocation, authentication, and provenance.
"""

from typing import Any, Optional
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, field_validator


class ToolClassification(str, Enum):
    """Security classification for tools."""

    SAFE = "safe"
    PII_RISK = "pii_risk"
    EXTERNAL_CALL = "external_call"
    SIDE_EFFECT = "side_effect"
    REQUIRES_APPROVAL = "requires_approval"


class AuthFlow(str, Enum):
    """Authentication flow type for tools."""

    NONE = "none"
    API_KEY = "api_key"
    OAUTH2 = "oauth2"
    EPHEMERAL_TOKEN = "ephemeral_token"


class RuntimeMode(str, Enum):
    """MCP Gateway runtime mode."""

    ORCHESTRATED = "orchestrated"  # Lead Agent mediated
    SCOPED_DIRECT = "scoped_direct"  # Ephemeral token-based direct access


class MCPTransport(str, Enum):
    """MCP transport mechanism."""

    SSE = "sse"  # Server-Sent Events / Streamable HTTP (remote servers)
    STDIO = "stdio"  # Stdin/stdout subprocess (local servers)


class ToolParameter(BaseModel):
    """Schema for a tool parameter."""

    name: str = Field(..., description="Parameter name")
    type: str = Field(..., description="Parameter type (string, number, boolean, array, object)")
    description: str = Field(..., description="Parameter description")
    required: bool = Field(default=False, description="Whether parameter is required")
    default: Optional[Any] = Field(default=None, description="Default value if not provided")


class ToolSchema(BaseModel):
    """Schema definition for a single tool."""

    name: str = Field(..., description="Tool name (unique identifier)")
    description: str = Field(..., description="Tool description")
    parameters: list[ToolParameter] = Field(
        default_factory=list, description="Tool input parameters"
    )
    returns: Optional[str] = Field(
        default=None, description="Description of return value/output"
    )


class RateLimitConfig(BaseModel):
    """Rate limiting configuration."""

    max_calls: int = Field(..., ge=1, description="Maximum calls allowed in window")
    window_seconds: int = Field(..., ge=1, description="Time window in seconds")


class MCPServerRegistration(BaseModel):
    """Registration schema for an MCP server with its tools."""

    tool_id: str = Field(..., description="Unique identifier for the MCP server")
    name: str = Field(..., description="Human-readable name")
    version: str = Field(..., description="Server version (semver)")
    owner: str = Field(..., description="Owner/team responsible for this server")
    contact: str = Field(..., description="Contact email for support")
    tools: list[ToolSchema] = Field(..., description="List of tools provided by this server")
    auth_flow: AuthFlow = Field(default=AuthFlow.NONE, description="Authentication flow type")
    classification: list[ToolClassification] = Field(
        default_factory=list, description="Security classifications"
    )
    rate_limits: Optional[RateLimitConfig] = Field(
        default=None, description="Rate limiting configuration"
    )
    endpoint: Optional[str] = Field(
        default=None, description="Server endpoint URL (for SSE transport)"
    )
    transport: MCPTransport = Field(
        default=MCPTransport.SSE, description="Transport mechanism (sse or stdio)"
    )
    command: Optional[list[str]] = Field(
        default=None, description="Command to start local MCP server (for stdio transport)"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata (api_key_env, etc.)"
    )

    @field_validator("version")
    @classmethod
    def validate_version(cls, v: str) -> str:
        """Validate version follows semver pattern."""
        parts = v.split(".")
        if len(parts) != 3:
            raise ValueError("Version must follow semver format (e.g., 1.0.0)")
        for part in parts:
            if not part.isdigit():
                raise ValueError("Version parts must be numeric")
        return v


class CatalogEntry(BaseModel):
    """Catalog entry with metadata and registration timestamp."""

    registration: MCPServerRegistration
    registered_at: datetime = Field(default_factory=datetime.utcnow)
    last_used: Optional[datetime] = None
    call_count: int = Field(default=0, description="Total number of calls to this server")
    enabled: bool = Field(default=True, description="Whether server is enabled")


class ToolInvocationRequest(BaseModel):
    """Request to invoke a tool via the gateway."""

    tool_id: str = Field(..., description="MCP server ID")
    tool_name: str = Field(..., description="Specific tool name within the server")
    arguments: dict[str, Any] = Field(default_factory=dict, description="Tool arguments")
    actor_id: str = Field(..., description="ID of the actor invoking the tool (agent/user)")
    actor_type: str = Field(..., description="Type of actor (lead_agent, subagent, user)")
    runtime_mode: RuntimeMode = Field(
        default=RuntimeMode.ORCHESTRATED, description="Runtime mode for this invocation"
    )
    token: Optional[str] = Field(
        default=None, description="Ephemeral token for scoped direct mode"
    )


class ToolInvocationResponse(BaseModel):
    """Response from a tool invocation."""

    success: bool = Field(..., description="Whether invocation succeeded")
    result: Optional[Any] = Field(default=None, description="Tool execution result")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    invocation_id: str = Field(..., description="Unique invocation ID for provenance")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    execution_time_ms: float = Field(..., description="Execution time in milliseconds")
    pii_detected: bool = Field(default=False, description="Whether PII was detected")


class TokenScope(BaseModel):
    """Scope definition for an ephemeral token."""

    tool_ids: list[str] = Field(..., description="List of allowed tool IDs")
    actor_id: str = Field(..., description="Actor this token is issued to")
    actor_type: str = Field(..., description="Type of actor")
    max_invocations: Optional[int] = Field(
        default=None, description="Maximum number of invocations allowed"
    )


class EphemeralTokenRequest(BaseModel):
    """Request to mint an ephemeral token for scoped direct access."""

    scope: TokenScope = Field(..., description="Token scope and permissions")
    ttl_minutes: Optional[int] = Field(
        default=None, description="Token TTL in minutes (uses default if not specified)"
    )


class EphemeralTokenResponse(BaseModel):
    """Response containing minted ephemeral token."""

    token: str = Field(..., description="JWT token")
    expires_at: datetime = Field(..., description="Token expiration timestamp")
    scope: TokenScope = Field(..., description="Token scope")


class ProvenanceLog(BaseModel):
    """Provenance log entry for tool invocations."""

    invocation_id: str = Field(..., description="Unique invocation ID")
    tool_id: str = Field(..., description="MCP server ID")
    tool_name: str = Field(..., description="Tool name")
    actor_id: str = Field(..., description="Actor ID")
    actor_type: str = Field(..., description="Actor type")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    arguments_hash: str = Field(..., description="Hash of input arguments")
    result_hash: Optional[str] = Field(default=None, description="Hash of result")
    success: bool = Field(..., description="Whether invocation succeeded")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    pii_detected: bool = Field(default=False, description="Whether PII was detected")
    runtime_mode: RuntimeMode = Field(..., description="Runtime mode used")
    execution_time_ms: float = Field(..., description="Execution time in milliseconds")


class CatalogListResponse(BaseModel):
    """Response containing list of available tools in catalog."""

    total: int = Field(..., description="Total number of registered servers")
    servers: list[CatalogEntry] = Field(..., description="List of catalog entries")


class ToolSchemaResponse(BaseModel):
    """Response containing detailed schema for a specific tool."""

    tool_id: str = Field(..., description="MCP server ID")
    tool_name: str = Field(..., description="Tool name")
    schema: ToolSchema = Field(..., description="Tool schema definition")
    classification: list[ToolClassification] = Field(
        ..., description="Security classifications"
    )
    rate_limits: Optional[RateLimitConfig] = Field(
        default=None, description="Rate limiting configuration"
    )


class HealthCheckResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Health status")
    version: str = Field(..., description="API version")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    catalog_size: int = Field(..., description="Number of registered servers")
