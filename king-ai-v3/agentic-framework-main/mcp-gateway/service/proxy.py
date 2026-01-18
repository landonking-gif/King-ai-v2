"""
Tool invocation proxy with PII detection and provenance logging.

Handles proxying tool invocations to MCP servers with security checks and logging.
"""

from typing import Any, Optional
import hashlib
import json
import secrets
import time
from datetime import datetime
import re
import anyio

from .config import settings
from .models import (
    ToolInvocationRequest,
    ToolInvocationResponse,
    ProvenanceLog,
    RuntimeMode,
)
from .catalog import get_catalog
from .auth import get_token_manager
from .rate_limit import get_rate_limiter


class PIIDetector:
    """Simple keyword-based PII detector for Sprint 0."""

    def __init__(self) -> None:
        """Initialize PII detector with keyword patterns."""
        self.keywords = settings.pii_keywords
        # Compile regex patterns for common PII formats
        self.patterns = [
            re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # SSN format
            re.compile(r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b"),  # Credit card
            re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),  # Email
            re.compile(r"\b(?:api[_-]?key|secret|token|password)\s*[:=]\s*\S+", re.IGNORECASE),
        ]

    def detect(self, data: Any) -> bool:
        """
        Detect potential PII in data.

        Args:
            data: Data to check (any JSON-serializable type)

        Returns:
            True if potential PII detected, False otherwise
        """
        if not settings.enable_pii_detection:
            return False

        # Convert data to string for analysis
        data_str = json.dumps(data).lower()

        # Check keywords
        for keyword in self.keywords:
            if keyword.lower() in data_str:
                return True

        # Check patterns
        for pattern in self.patterns:
            if pattern.search(data_str):
                return True

        return False


class ToolProxy:
    """Proxy for tool invocations with security and logging."""

    def __init__(self) -> None:
        """Initialize tool proxy with dependencies."""
        self.catalog = get_catalog()
        self.token_manager = get_token_manager()
        self.rate_limiter = get_rate_limiter()
        self.pii_detector = PIIDetector()
        self._provenance_logs: list[ProvenanceLog] = []

    async def invoke(self, request: ToolInvocationRequest) -> ToolInvocationResponse:
        """
        Invoke a tool via the gateway with all security checks.

        Args:
            request: Tool invocation request

        Returns:
            Tool invocation response

        Raises:
            ValueError: If tool not found or validation fails
            PermissionError: If authorization fails
        """
        start_time = time.time()
        invocation_id = secrets.token_urlsafe(16)

        try:
            # 1. Validate tool exists and is enabled
            entry = await self.catalog.get(request.tool_id)
            if entry is None:
                raise ValueError(f"Tool '{request.tool_id}' not found in catalog")

            if not entry.enabled:
                raise ValueError(f"Tool '{request.tool_id}' is disabled")

            # 2. Check authorization based on runtime mode
            if request.runtime_mode == RuntimeMode.SCOPED_DIRECT:
                if request.token is None:
                    raise PermissionError("Token required for scoped direct mode")

                has_permission, error_msg = await self.token_manager.check_token_permission(
                    request.token, request.tool_id
                )
                if not has_permission:
                    raise PermissionError(error_msg or "Token validation failed")

            # 3. Check rate limits
            rate_limit = entry.registration.rate_limits
            if rate_limit is not None:
                is_allowed, retry_after = await self.rate_limiter.check_and_increment(
                    request.tool_id, request.actor_id, rate_limit
                )
                if not is_allowed:
                    raise PermissionError(
                        f"Rate limit exceeded. Retry after {retry_after} seconds."
                    )

            # 4. PII detection
            pii_detected = self.pii_detector.detect(request.arguments)
            if pii_detected:
                # Log warning but continue (in production, might block or require approval)
                print(f"WARNING: Potential PII detected in request {invocation_id}")

            # 5. Validate tool schema
            tool_schema = await self.catalog.get_tool_schema(
                request.tool_id, request.tool_name
            )
            if tool_schema is None:
                raise ValueError(
                    f"Tool '{request.tool_name}' not found in server '{request.tool_id}'"
                )

            # 6. Execute tool via MCP protocol (remote) or mock (local)
            result = await self._execute_tool(
                request.tool_id, request.tool_name, request.arguments
            )

            # 7. Update usage statistics
            await self.catalog.update_usage(request.tool_id)

            # 8. Create success response
            execution_time = (time.time() - start_time) * 1000
            response = ToolInvocationResponse(
                success=True,
                result=result,
                invocation_id=invocation_id,
                execution_time_ms=execution_time,
                pii_detected=pii_detected,
            )

            # 9. Log provenance
            await self._log_provenance(request, response, invocation_id)

            return response

        except Exception as e:
            # Handle errors and log provenance
            execution_time = (time.time() - start_time) * 1000
            response = ToolInvocationResponse(
                success=False,
                error=str(e),
                invocation_id=invocation_id,
                execution_time_ms=execution_time,
                pii_detected=False,
            )

            await self._log_provenance(request, response, invocation_id)
            raise

    async def _execute_tool(
        self, tool_id: str, tool_name: str, arguments: dict[str, Any]
    ) -> Any:
        """
        Execute tool via MCP protocol.

        Supports:
        - Remote MCP servers via SSE transport
        - Local MCP servers via stdio transport
        - Mock tools for internal/testing

        Args:
            tool_id: Tool server ID
            tool_name: Tool name
            arguments: Tool arguments

        Returns:
            Result from the MCP server
        """
        from .mcp_client import (
            MCPClientConfig,
            MCPTransport as ClientTransport,
            get_mcp_client,
        )
        from .models import MCPTransport

        # Get tool registration
        entry = await self.catalog.get(tool_id)
        if entry is None:
            raise ValueError(f"Tool '{tool_id}' not found")

        registration = entry.registration
        transport = registration.transport
        metadata = registration.metadata

        # Check if this is a real MCP server or mock
        if transport == MCPTransport.SSE and registration.endpoint:
            # Remote MCP server via SSE
            config = MCPClientConfig(
                tool_id=tool_id,
                transport=ClientTransport.SSE,
                endpoint=registration.endpoint,
                api_key_env=metadata.get("api_key_env"),
                timeout=60.0,
            )
            return await self._execute_via_mcp_client(config, tool_name, arguments)

        elif transport == MCPTransport.STDIO and registration.command:
            # Local MCP server via stdio
            config = MCPClientConfig(
                tool_id=tool_id,
                transport=ClientTransport.STDIO,
                command=registration.command,
                api_key_env=metadata.get("api_key_env"),
                timeout=60.0,
                env_vars=metadata.get("env_vars", {}),
            )
            return await self._execute_via_mcp_client(config, tool_name, arguments)

        else:
            # Fall back to mock for internal tools
            return await self._execute_tool_mock(tool_id, tool_name, arguments)

    async def _execute_via_mcp_client(
        self,
        config: Any,  # MCPClientConfig
        tool_name: str,
        arguments: dict[str, Any],
    ) -> Any:
        """
        Execute tool using proper MCP client.

        Handles initialization, tool call, and error handling.

        Args:
            config: MCP client configuration
            tool_name: Tool name to call
            arguments: Tool arguments

        Returns:
            Tool execution result
        """
        from .mcp_client import get_mcp_client

        try:
            client = await get_mcp_client(config)
            result = await client.call_tool(tool_name, arguments)
            return result
        except Exception as e:
            raise ValueError(f"MCP tool execution failed: {str(e)}")

    async def _execute_tool_mock(
        self, tool_id: str, tool_name: str, arguments: dict[str, Any]
    ) -> Any:
        """
        Mock tool execution for local/internal tools.

        Args:
            tool_id: Tool server ID
            tool_name: Tool name
            arguments: Tool arguments

        Returns:
            Mock result based on tool type
        """
        # Simulate network delay
        await anyio.sleep(0.1)

        # Return mock results based on tool
        if tool_id == "web_search" and tool_name == "search":
            return [
                {
                    "title": f"Result for: {arguments.get('query', 'unknown')}",
                    "url": "https://example.com/result1",
                    "snippet": "This is a mock search result snippet",
                }
            ]

        elif tool_id == "file_operations" and tool_name == "read_file":
            return f"Mock contents of file: {arguments.get('path', 'unknown')}"

        elif tool_id == "file_operations" and tool_name == "write_file":
            return {"success": True, "bytes_written": len(arguments.get("content", ""))}

        elif tool_id == "database_query" and tool_name == "execute_query":
            return [
                {"id": 1, "name": "Mock Record 1"},
                {"id": 2, "name": "Mock Record 2"},
            ]

        elif tool_id == "text_processing" and tool_name == "summarize":
            text = arguments.get("text", "")
            max_length = arguments.get("max_length", 200)
            return text[:max_length] + "..." if len(text) > max_length else text

        elif tool_id == "text_processing" and tool_name == "extract_entities":
            return [
                {"name": "Example Entity", "type": "ORGANIZATION", "confidence": 0.95},
                {"name": "Mock Location", "type": "LOCATION", "confidence": 0.87},
            ]

        else:
            return {"status": "success", "message": f"Mock execution of {tool_name}"}

    async def _log_provenance(
        self,
        request: ToolInvocationRequest,
        response: ToolInvocationResponse,
        invocation_id: str,
    ) -> None:
        """
        Log provenance information for the invocation.

        Args:
            request: Original request
            response: Response generated
            invocation_id: Unique invocation ID
        """
        if not settings.enable_provenance_logging:
            return

        # Hash arguments and result for provenance
        arguments_hash = hashlib.sha256(
            json.dumps(request.arguments, sort_keys=True).encode()
        ).hexdigest()

        result_hash = None
        if response.result is not None:
            result_hash = hashlib.sha256(
                json.dumps(response.result, sort_keys=True).encode()
            ).hexdigest()

        log_entry = ProvenanceLog(
            invocation_id=invocation_id,
            tool_id=request.tool_id,
            tool_name=request.tool_name,
            actor_id=request.actor_id,
            actor_type=request.actor_type,
            arguments_hash=arguments_hash,
            result_hash=result_hash,
            success=response.success,
            error=response.error,
            pii_detected=response.pii_detected,
            runtime_mode=request.runtime_mode,
            execution_time_ms=response.execution_time_ms,
        )

        # Store in memory for Sprint 0 (in production, write to database)
        self._provenance_logs.append(log_entry)

        # Print for visibility during development
        print(f"[PROVENANCE] {log_entry.model_dump_json()}")

    async def get_provenance_logs(
        self, limit: int = 100, tool_id: Optional[str] = None
    ) -> list[ProvenanceLog]:
        """
        Retrieve provenance logs.

        Args:
            limit: Maximum number of logs to return
            tool_id: Optional filter by tool ID

        Returns:
            List of provenance log entries
        """
        logs = self._provenance_logs

        if tool_id is not None:
            logs = [log for log in logs if log.tool_id == tool_id]

        # Return most recent logs
        return logs[-limit:]


# Global proxy instance
_proxy_instance: Optional[ToolProxy] = None


def get_proxy() -> ToolProxy:
    """
    Get the global tool proxy instance (singleton pattern).

    Returns:
        Global ToolProxy instance
    """
    global _proxy_instance
    if _proxy_instance is None:
        _proxy_instance = ToolProxy()
    return _proxy_instance
