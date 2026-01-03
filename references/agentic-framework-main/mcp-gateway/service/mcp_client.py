"""
MCP Client implementations for different transport mechanisms.

Module: mcp-gateway/service/mcp_client.py

Supports:
- Remote MCP servers via SSE/Streamable HTTP transport
- Local MCP servers via stdio transport (subprocess)
"""

import os
import json
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class MCPTransport(str, Enum):
    """Supported MCP transport mechanisms."""
    SSE = "sse"  # Server-Sent Events / Streamable HTTP
    STDIO = "stdio"  # Local subprocess via stdin/stdout


@dataclass
class MCPMessage:
    """MCP JSON-RPC message."""
    jsonrpc: str = "2.0"
    id: Optional[int] = None
    method: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        msg = {"jsonrpc": self.jsonrpc}
        if self.id is not None:
            msg["id"] = self.id
        if self.method is not None:
            msg["method"] = self.method
        if self.params is not None:
            msg["params"] = self.params
        if self.result is not None:
            msg["result"] = self.result
        if self.error is not None:
            msg["error"] = self.error
        return msg

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MCPMessage":
        """Create from dictionary."""
        return cls(
            jsonrpc=data.get("jsonrpc", "2.0"),
            id=data.get("id"),
            method=data.get("method"),
            params=data.get("params"),
            result=data.get("result"),
            error=data.get("error"),
        )


@dataclass
class MCPClientConfig:
    """Configuration for MCP client."""
    tool_id: str
    transport: MCPTransport
    endpoint: Optional[str] = None  # For SSE transport
    command: Optional[List[str]] = None  # For stdio transport
    api_key_env: Optional[str] = None
    timeout: float = 60.0
    env_vars: Dict[str, str] = field(default_factory=dict)


class MCPClientBase(ABC):
    """Base class for MCP clients."""

    def __init__(self, config: MCPClientConfig):
        """Initialize MCP client."""
        self.config = config
        self._initialized = False
        self._message_id = 0
        self._server_info: Optional[Dict[str, Any]] = None

    def _next_id(self) -> int:
        """Get next message ID."""
        self._message_id += 1
        return self._message_id

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to MCP server."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to MCP server."""
        pass

    @abstractmethod
    async def send_message(self, message: MCPMessage) -> MCPMessage:
        """Send message and wait for response."""
        pass

    async def initialize(self) -> Dict[str, Any]:
        """
        Initialize MCP session with server.

        Returns:
            Server capabilities and info
        """
        if self._initialized:
            return self._server_info or {}

        # Send initialize request
        init_msg = MCPMessage(
            id=self._next_id(),
            method="initialize",
            params={
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {}
                },
                "clientInfo": {
                    "name": "kautilya-gateway",
                    "version": "1.0.0"
                }
            }
        )

        response = await self.send_message(init_msg)

        if response.error:
            raise ValueError(f"Initialize failed: {response.error}")

        self._server_info = response.result
        self._initialized = True

        # Send initialized notification
        await self.send_notification("notifications/initialized", {})

        logger.info(f"MCP session initialized for {self.config.tool_id}")
        return self._server_info or {}

    async def send_notification(self, method: str, params: Dict[str, Any]) -> None:
        """Send notification (no response expected)."""
        notif = MCPMessage(method=method, params=params)
        await self.send_message(notif)

    async def call_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> Any:
        """
        Call a tool on the MCP server.

        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments

        Returns:
            Tool execution result
        """
        # Ensure initialized
        if not self._initialized:
            await self.initialize()

        # Send tools/call request
        call_msg = MCPMessage(
            id=self._next_id(),
            method="tools/call",
            params={
                "name": tool_name,
                "arguments": arguments
            }
        )

        response = await self.send_message(call_msg)

        if response.error:
            raise ValueError(
                f"Tool call failed: {response.error.get('message', str(response.error))}"
            )

        # Extract content from result
        result = response.result
        if isinstance(result, dict) and "content" in result:
            # MCP tools return content array
            content = result["content"]
            if isinstance(content, list) and len(content) > 0:
                # Return text content if available
                for item in content:
                    if item.get("type") == "text":
                        return item.get("text")
                return content
            return content
        return result

    async def list_tools(self) -> List[Dict[str, Any]]:
        """Get list of available tools from server."""
        if not self._initialized:
            await self.initialize()

        list_msg = MCPMessage(
            id=self._next_id(),
            method="tools/list",
            params={}
        )

        response = await self.send_message(list_msg)

        if response.error:
            raise ValueError(f"List tools failed: {response.error}")

        return response.result.get("tools", []) if response.result else []


class RemoteMCPClient(MCPClientBase):
    """
    MCP client for remote servers using SSE/Streamable HTTP transport.

    Handles the MCP protocol over HTTP with Server-Sent Events responses.
    """

    def __init__(self, config: MCPClientConfig):
        """Initialize remote MCP client."""
        super().__init__(config)
        self._http_client: Optional[Any] = None
        self._session_id: Optional[str] = None

    async def connect(self) -> None:
        """Establish HTTP client connection."""
        import httpx
        self._http_client = httpx.AsyncClient(timeout=self.config.timeout)

    async def disconnect(self) -> None:
        """Close HTTP client."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
        self._initialized = False

    def _get_headers(self) -> Dict[str, str]:
        """Build request headers with authentication."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }

        # Add API key if configured
        if self.config.api_key_env:
            api_key = os.getenv(self.config.api_key_env)
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

        # Add session ID if we have one
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id

        return headers

    async def send_message(self, message: MCPMessage) -> MCPMessage:
        """Send message via HTTP POST and handle SSE response."""
        import httpx

        if not self._http_client:
            await self.connect()

        endpoint = self.config.endpoint
        if not endpoint:
            raise ValueError("Endpoint URL is required for remote MCP client")

        headers = self._get_headers()

        try:
            # For notifications (no id), we don't expect a response
            if message.id is None:
                await self._http_client.post(
                    endpoint,
                    json=message.to_dict(),
                    headers=headers,
                )
                return MCPMessage()

            # For requests, handle streaming response
            async with self._http_client.stream(
                "POST",
                endpoint,
                json=message.to_dict(),
                headers=headers,
            ) as response:
                # Check for session ID in response headers
                if "mcp-session-id" in response.headers:
                    self._session_id = response.headers["mcp-session-id"]

                if response.status_code != 200:
                    error_text = await response.aread()
                    raise ValueError(
                        f"MCP server error ({response.status_code}): {error_text.decode()[:500]}"
                    )

                content_type = response.headers.get("content-type", "")

                # Handle SSE response
                if "text/event-stream" in content_type:
                    return await self._parse_sse_stream(response)

                # Handle JSON response
                body = await response.aread()
                data = json.loads(body)
                return MCPMessage.from_dict(data)

        except httpx.TimeoutException:
            raise ValueError(f"Timeout connecting to MCP server at {endpoint}")
        except httpx.ConnectError as e:
            raise ValueError(f"Cannot connect to MCP server at {endpoint}: {e}")

    async def _parse_sse_stream(self, response: Any) -> MCPMessage:
        """Parse SSE stream and extract the final result."""
        result_message = MCPMessage()

        async for line in response.aiter_lines():
            line = line.strip()
            if not line:
                continue

            if line.startswith("data:"):
                data_str = line[5:].strip()
                if data_str:
                    try:
                        data = json.loads(data_str)
                        # Update result message with latest data
                        if "result" in data:
                            result_message.result = data["result"]
                        if "error" in data:
                            result_message.error = data["error"]
                        if "id" in data:
                            result_message.id = data["id"]
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse SSE data: {data_str[:100]}")

            elif line.startswith("event:"):
                event_type = line[6:].strip()
                logger.debug(f"SSE event: {event_type}")

        return result_message


class StdioMCPClient(MCPClientBase):
    """
    MCP client for local servers using stdio transport.

    Spawns the MCP server as a subprocess and communicates via stdin/stdout.
    """

    def __init__(self, config: MCPClientConfig):
        """Initialize stdio MCP client."""
        super().__init__(config)
        self._process: Optional[asyncio.subprocess.Process] = None
        self._read_lock = asyncio.Lock()
        self._write_lock = asyncio.Lock()

    async def connect(self) -> None:
        """Spawn MCP server subprocess."""
        if not self.config.command:
            raise ValueError("Command is required for stdio MCP client")

        # Build environment with any configured env vars
        env = os.environ.copy()
        env.update(self.config.env_vars)

        # Add API key to environment if configured
        if self.config.api_key_env:
            api_key = os.getenv(self.config.api_key_env)
            if api_key:
                env[self.config.api_key_env] = api_key

        logger.info(f"Starting MCP server: {' '.join(self.config.command)}")

        self._process = await asyncio.create_subprocess_exec(
            *self.config.command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )

        # Give process time to start
        await asyncio.sleep(0.5)

        # Check if process started successfully
        if self._process.returncode is not None:
            stderr = await self._process.stderr.read() if self._process.stderr else b""
            raise ValueError(
                f"MCP server failed to start: {stderr.decode()[:500]}"
            )

        logger.info(f"MCP server started (PID: {self._process.pid})")

    async def disconnect(self) -> None:
        """Stop MCP server subprocess."""
        if self._process:
            try:
                self._process.terminate()
                await asyncio.wait_for(self._process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self._process.kill()
                await self._process.wait()
            finally:
                self._process = None
                self._initialized = False
                logger.info(f"MCP server stopped for {self.config.tool_id}")

    async def send_message(self, message: MCPMessage) -> MCPMessage:
        """Send message via stdin and read response from stdout."""
        if not self._process or not self._process.stdin or not self._process.stdout:
            await self.connect()

        assert self._process and self._process.stdin and self._process.stdout

        # Write message to stdin
        async with self._write_lock:
            msg_bytes = (message.to_json() + "\n").encode()
            self._process.stdin.write(msg_bytes)
            await self._process.stdin.drain()

        # For notifications, don't wait for response
        if message.id is None:
            return MCPMessage()

        # Read response from stdout
        async with self._read_lock:
            try:
                line = await asyncio.wait_for(
                    self._process.stdout.readline(),
                    timeout=self.config.timeout
                )
                if not line:
                    raise ValueError("MCP server closed connection")

                data = json.loads(line.decode())
                return MCPMessage.from_dict(data)

            except asyncio.TimeoutError:
                raise ValueError(f"Timeout waiting for MCP server response")
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON from MCP server: {e}")


# Client cache for reusing connections
_client_cache: Dict[str, MCPClientBase] = {}
_cache_lock = asyncio.Lock()


async def get_mcp_client(config: MCPClientConfig) -> MCPClientBase:
    """
    Get or create an MCP client for the given configuration.

    Clients are cached and reused for efficiency.

    Args:
        config: Client configuration

    Returns:
        Connected MCP client
    """
    async with _cache_lock:
        cache_key = f"{config.tool_id}:{config.transport.value}"

        if cache_key in _client_cache:
            client = _client_cache[cache_key]
            # Check if still connected
            if config.transport == MCPTransport.STDIO:
                stdio_client = client
                if isinstance(stdio_client, StdioMCPClient):
                    if stdio_client._process and stdio_client._process.returncode is None:
                        return client
            else:
                return client

        # Create new client
        if config.transport == MCPTransport.SSE:
            client = RemoteMCPClient(config)
        elif config.transport == MCPTransport.STDIO:
            client = StdioMCPClient(config)
        else:
            raise ValueError(f"Unknown transport: {config.transport}")

        await client.connect()
        _client_cache[cache_key] = client

        return client


async def close_all_clients() -> None:
    """Close all cached MCP clients."""
    async with _cache_lock:
        for client in _client_cache.values():
            try:
                await client.disconnect()
            except Exception as e:
                logger.warning(f"Error closing MCP client: {e}")
        _client_cache.clear()
