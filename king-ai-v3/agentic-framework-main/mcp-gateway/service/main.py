"""
MCP Gateway FastAPI application.

Enterprise-grade MCP tool catalog and runtime proxy with authentication,
rate limiting, PII detection, and provenance logging.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file (project root)
# This ensures API keys like FIRECRAWL_API_KEY are available
_env_paths = [
    Path(__file__).parent.parent.parent / ".env",  # agent-framework/.env
    Path(__file__).parent.parent / ".env",  # mcp-gateway/.env
    Path.cwd() / ".env",  # current directory
]
for _env_path in _env_paths:
    if _env_path.exists():
        load_dotenv(_env_path, override=False)

from typing import Any
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from .config import settings
from .models import (
    MCPServerRegistration,
    CatalogListResponse,
    ToolInvocationRequest,
    ToolInvocationResponse,
    EphemeralTokenRequest,
    EphemeralTokenResponse,
    ToolSchemaResponse,
    HealthCheckResponse,
    ToolClassification,
    ProvenanceLog,
)
from .catalog import get_catalog, initialize_sample_tools
from .auth import get_token_manager
from .rate_limit import get_rate_limiter
from .proxy import get_proxy


@asynccontextmanager
async def lifespan(app: FastAPI) -> Any:
    """
    Application lifespan manager.

    Handles startup and shutdown tasks.
    """
    # Startup
    print(f"Starting {settings.service_name} v{settings.api_version}")

    # Initialize Redis connection for rate limiting
    rate_limiter = get_rate_limiter()
    await rate_limiter.connect()

    # Initialize sample tools for Sprint 0
    await initialize_sample_tools()
    print("Initialized sample tools in catalog")

    yield

    # Shutdown
    print("Shutting down MCP Gateway")
    await rate_limiter.disconnect()


# Create FastAPI application
app = FastAPI(
    title="MCP Gateway",
    description="Enterprise MCP tool catalog and runtime proxy",
    version=settings.api_version,
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=settings.cors_allow_methods,
    allow_headers=settings.cors_allow_headers,
)


# Health check endpoint
@app.get("/health", response_model=HealthCheckResponse, tags=["Health"])
async def health_check() -> HealthCheckResponse:
    """
    Health check endpoint.

    Returns service status and basic metrics.
    """
    catalog = get_catalog()
    servers = await catalog.list_all()

    return HealthCheckResponse(
        status="healthy", version=settings.api_version, catalog_size=len(servers)
    )


# Catalog endpoints
@app.get("/catalog/tools", response_model=CatalogListResponse, tags=["Catalog"])
async def list_tools(
    enabled_only: bool = Query(default=True, description="Return only enabled tools")
) -> CatalogListResponse:
    """
    List all available tools in the catalog.

    Args:
        enabled_only: If True, return only enabled tools

    Returns:
        List of catalog entries
    """
    catalog = get_catalog()

    if enabled_only:
        servers = await catalog.list_enabled()
    else:
        servers = await catalog.list_all()

    return CatalogListResponse(total=len(servers), servers=servers)


@app.post("/catalog/register", status_code=status.HTTP_201_CREATED, tags=["Catalog"])
async def register_tool(registration: MCPServerRegistration) -> dict[str, str]:
    """
    Register a new MCP server in the catalog.

    Args:
        registration: Server registration details

    Returns:
        Success message with tool ID

    Raises:
        HTTPException: If tool_id already exists
    """
    catalog = get_catalog()

    try:
        entry = await catalog.register(registration)
        return {
            "message": f"Successfully registered tool '{registration.tool_id}'",
            "tool_id": registration.tool_id,
            "registered_at": entry.registered_at.isoformat(),
        }
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))


@app.delete("/catalog/tools/{tool_id}", tags=["Catalog"])
async def unregister_tool(tool_id: str) -> dict[str, str]:
    """
    Unregister a tool from the catalog.

    Args:
        tool_id: Tool ID to unregister

    Returns:
        Success message

    Raises:
        HTTPException: If tool_id not found
    """
    catalog = get_catalog()

    try:
        await catalog.unregister(tool_id)
        return {"message": f"Successfully unregistered tool '{tool_id}'"}
    except KeyError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@app.patch("/catalog/tools/{tool_id}/enable", tags=["Catalog"])
async def enable_tool(tool_id: str) -> dict[str, str]:
    """
    Enable a tool in the catalog.

    Args:
        tool_id: Tool ID to enable

    Returns:
        Success message

    Raises:
        HTTPException: If tool_id not found
    """
    catalog = get_catalog()

    try:
        await catalog.enable(tool_id)
        return {"message": f"Tool '{tool_id}' enabled"}
    except KeyError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@app.patch("/catalog/tools/{tool_id}/disable", tags=["Catalog"])
async def disable_tool(tool_id: str) -> dict[str, str]:
    """
    Disable a tool in the catalog.

    Args:
        tool_id: Tool ID to disable

    Returns:
        Success message

    Raises:
        HTTPException: If tool_id not found
    """
    catalog = get_catalog()

    try:
        await catalog.disable(tool_id)
        return {"message": f"Tool '{tool_id}' disabled"}
    except KeyError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@app.get(
    "/catalog/tools/{tool_id}/schema/{tool_name}",
    response_model=ToolSchemaResponse,
    tags=["Catalog"],
)
async def get_tool_schema(tool_id: str, tool_name: str) -> ToolSchemaResponse:
    """
    Get detailed schema for a specific tool.

    Args:
        tool_id: Server ID
        tool_name: Tool name

    Returns:
        Tool schema with classification and rate limits

    Raises:
        HTTPException: If tool not found
    """
    catalog = get_catalog()

    entry = await catalog.get(tool_id)
    if entry is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Tool '{tool_id}' not found"
        )

    schema = await catalog.get_tool_schema(tool_id, tool_name)
    if schema is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tool '{tool_name}' not found in server '{tool_id}'",
        )

    return ToolSchemaResponse(
        tool_id=tool_id,
        tool_name=tool_name,
        schema=schema,
        classification=entry.registration.classification,
        rate_limits=entry.registration.rate_limits,
    )


@app.get("/catalog/search", tags=["Catalog"])
async def search_by_classification(
    classification: ToolClassification = Query(..., description="Classification to search for")
) -> CatalogListResponse:
    """
    Search for tools by classification.

    Args:
        classification: Security classification

    Returns:
        List of matching tools
    """
    catalog = get_catalog()
    servers = await catalog.search_by_classification(classification)

    return CatalogListResponse(total=len(servers), servers=servers)


# Tool invocation endpoint
@app.post("/tools/invoke", response_model=ToolInvocationResponse, tags=["Invocation"])
async def invoke_tool(request: ToolInvocationRequest) -> ToolInvocationResponse:
    """
    Invoke a tool via the gateway proxy.

    Handles authorization, rate limiting, PII detection, and provenance logging.

    Args:
        request: Tool invocation request

    Returns:
        Tool invocation response

    Raises:
        HTTPException: On validation, authorization, or rate limit errors
    """
    proxy = get_proxy()

    try:
        response = await proxy.invoke(request)
        return response
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error: {str(e)}",
        )


# Authentication endpoints
@app.post(
    "/auth/token", response_model=EphemeralTokenResponse, tags=["Authentication"]
)
async def mint_ephemeral_token(
    request: EphemeralTokenRequest,
) -> EphemeralTokenResponse:
    """
    Mint an ephemeral JWT token for scoped direct access.

    Args:
        request: Token request with scope and TTL

    Returns:
        Token response with JWT and expiration
    """
    token_manager = get_token_manager()
    return await token_manager.mint_token(request)


@app.post("/auth/validate", tags=["Authentication"])
async def validate_token(token: str = Query(..., description="Token to validate")) -> dict[str, Any]:
    """
    Validate a token and return its scope.

    Args:
        token: JWT token to validate

    Returns:
        Token scope and validity status

    Raises:
        HTTPException: If token is invalid
    """
    token_manager = get_token_manager()
    scope = await token_manager.validate_token(token)

    if scope is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token"
        )

    return {
        "valid": True,
        "scope": scope.model_dump(),
    }


# Provenance endpoints
@app.get("/provenance/logs", response_model=list[ProvenanceLog], tags=["Provenance"])
async def get_provenance_logs(
    limit: int = Query(default=100, ge=1, le=1000, description="Maximum logs to return"),
    tool_id: str = Query(default=None, description="Filter by tool ID"),
) -> list[ProvenanceLog]:
    """
    Retrieve provenance logs for tool invocations.

    Args:
        limit: Maximum number of logs to return
        tool_id: Optional filter by tool ID

    Returns:
        List of provenance log entries
    """
    proxy = get_proxy()
    return await proxy.get_provenance_logs(limit=limit, tool_id=tool_id)


# Error handlers
@app.exception_handler(ValueError)
async def value_error_handler(request: Any, exc: ValueError) -> JSONResponse:
    """Handle ValueError exceptions."""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST, content={"detail": str(exc)}
    )


@app.exception_handler(PermissionError)
async def permission_error_handler(request: Any, exc: PermissionError) -> JSONResponse:
    """Handle PermissionError exceptions."""
    return JSONResponse(
        status_code=status.HTTP_403_FORBIDDEN, content={"detail": str(exc)}
    )


def main() -> None:
    """Run the MCP Gateway service."""
    uvicorn.run(
        "mcp_gateway.service.main:app",
        host=settings.mcp_gateway_host,
        port=settings.mcp_gateway_port,
        reload=True,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
