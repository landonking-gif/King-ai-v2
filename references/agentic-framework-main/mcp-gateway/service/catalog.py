"""
MCP Tool Catalog management.

Provides in-memory catalog for MCP server registrations with CRUD operations.
Can be migrated to database backend in production.
"""

from typing import Optional
from datetime import datetime
import anyio
from anyio import Lock

from .models import (
    MCPServerRegistration,
    CatalogEntry,
    ToolSchema,
    ToolParameter,
    ToolClassification,
    AuthFlow,
    RateLimitConfig,
)


class ToolCatalog:
    """In-memory catalog for registered MCP servers and tools."""

    def __init__(self) -> None:
        """Initialize empty catalog with lock for thread safety."""
        self._catalog: dict[str, CatalogEntry] = {}
        self._lock = Lock()

    async def register(self, registration: MCPServerRegistration) -> CatalogEntry:
        """
        Register a new MCP server in the catalog.

        Args:
            registration: Server registration details

        Returns:
            Created catalog entry

        Raises:
            ValueError: If tool_id already exists
        """
        async with self._lock:
            if registration.tool_id in self._catalog:
                raise ValueError(f"Tool ID '{registration.tool_id}' is already registered")

            entry = CatalogEntry(registration=registration)
            self._catalog[registration.tool_id] = entry
            return entry

    async def unregister(self, tool_id: str) -> None:
        """
        Remove a server from the catalog.

        Args:
            tool_id: Server ID to remove

        Raises:
            KeyError: If tool_id does not exist
        """
        async with self._lock:
            if tool_id not in self._catalog:
                raise KeyError(f"Tool ID '{tool_id}' not found in catalog")
            del self._catalog[tool_id]

    async def get(self, tool_id: str) -> Optional[CatalogEntry]:
        """
        Retrieve a catalog entry by tool ID.

        Args:
            tool_id: Server ID to retrieve

        Returns:
            Catalog entry if found, None otherwise
        """
        return self._catalog.get(tool_id)

    async def list_all(self) -> list[CatalogEntry]:
        """
        List all registered servers in the catalog.

        Returns:
            List of all catalog entries
        """
        return list(self._catalog.values())

    async def list_enabled(self) -> list[CatalogEntry]:
        """
        List only enabled servers.

        Returns:
            List of enabled catalog entries
        """
        return [entry for entry in self._catalog.values() if entry.enabled]

    async def enable(self, tool_id: str) -> None:
        """
        Enable a server in the catalog.

        Args:
            tool_id: Server ID to enable

        Raises:
            KeyError: If tool_id does not exist
        """
        async with self._lock:
            if tool_id not in self._catalog:
                raise KeyError(f"Tool ID '{tool_id}' not found in catalog")
            self._catalog[tool_id].enabled = True

    async def disable(self, tool_id: str) -> None:
        """
        Disable a server in the catalog.

        Args:
            tool_id: Server ID to disable

        Raises:
            KeyError: If tool_id does not exist
        """
        async with self._lock:
            if tool_id not in self._catalog:
                raise KeyError(f"Tool ID '{tool_id}' not found in catalog")
            self._catalog[tool_id].enabled = False

    async def get_tool_schema(self, tool_id: str, tool_name: str) -> Optional[ToolSchema]:
        """
        Get schema for a specific tool within a server.

        Args:
            tool_id: Server ID
            tool_name: Tool name

        Returns:
            Tool schema if found, None otherwise
        """
        entry = await self.get(tool_id)
        if entry is None:
            return None

        for tool in entry.registration.tools:
            if tool.name == tool_name:
                return tool

        return None

    async def update_usage(self, tool_id: str) -> None:
        """
        Update usage statistics for a server.

        Args:
            tool_id: Server ID

        Raises:
            KeyError: If tool_id does not exist
        """
        async with self._lock:
            if tool_id not in self._catalog:
                raise KeyError(f"Tool ID '{tool_id}' not found in catalog")

            entry = self._catalog[tool_id]
            entry.call_count += 1
            entry.last_used = datetime.utcnow()

    async def search_by_classification(
        self, classification: ToolClassification
    ) -> list[CatalogEntry]:
        """
        Search for servers by classification.

        Args:
            classification: Classification to search for

        Returns:
            List of matching catalog entries
        """
        return [
            entry
            for entry in self._catalog.values()
            if classification in entry.registration.classification
        ]


# Global catalog instance
_catalog_instance: Optional[ToolCatalog] = None


def get_catalog() -> ToolCatalog:
    """
    Get the global catalog instance (singleton pattern).

    Returns:
        Global ToolCatalog instance
    """
    global _catalog_instance
    if _catalog_instance is None:
        _catalog_instance = ToolCatalog()
    return _catalog_instance


async def initialize_sample_tools() -> None:
    """Initialize catalog with sample tool registrations for Sprint 0."""
    catalog = get_catalog()

    # Web search tool
    web_search = MCPServerRegistration(
        tool_id="web_search",
        name="Web Search",
        version="1.0.0",
        owner="platform-team",
        contact="platform@example.com",
        tools=[
            ToolSchema(
                name="search",
                description="Search the web for information",
                parameters=[
                    ToolParameter(
                        name="query",
                        type="string",
                        description="Search query",
                        required=True,
                    ),
                    ToolParameter(
                        name="max_results",
                        type="number",
                        description="Maximum number of results",
                        required=False,
                        default=10,
                    ),
                ],
                returns="List of search results with title, url, and snippet",
            )
        ],
        auth_flow=AuthFlow.API_KEY,
        classification=[ToolClassification.EXTERNAL_CALL],
        rate_limits=RateLimitConfig(max_calls=100, window_seconds=60),
        endpoint="http://localhost:9001/search",
    )

    # File operations tool
    file_ops = MCPServerRegistration(
        tool_id="file_operations",
        name="File Operations",
        version="1.0.0",
        owner="platform-team",
        contact="platform@example.com",
        tools=[
            ToolSchema(
                name="read_file",
                description="Read contents of a file",
                parameters=[
                    ToolParameter(
                        name="path",
                        type="string",
                        description="File path to read",
                        required=True,
                    )
                ],
                returns="File contents as string",
            ),
            ToolSchema(
                name="write_file",
                description="Write contents to a file",
                parameters=[
                    ToolParameter(
                        name="path",
                        type="string",
                        description="File path to write",
                        required=True,
                    ),
                    ToolParameter(
                        name="content",
                        type="string",
                        description="Content to write",
                        required=True,
                    ),
                ],
                returns="Success confirmation",
            ),
        ],
        auth_flow=AuthFlow.NONE,
        classification=[ToolClassification.SIDE_EFFECT],
        rate_limits=RateLimitConfig(max_calls=50, window_seconds=60),
        endpoint="http://localhost:9002/files",
    )

    # Database query tool
    database_query = MCPServerRegistration(
        tool_id="database_query",
        name="Database Query",
        version="1.0.0",
        owner="data-team",
        contact="data@example.com",
        tools=[
            ToolSchema(
                name="execute_query",
                description="Execute a database query",
                parameters=[
                    ToolParameter(
                        name="query",
                        type="string",
                        description="SQL query to execute",
                        required=True,
                    ),
                    ToolParameter(
                        name="params",
                        type="object",
                        description="Query parameters",
                        required=False,
                        default={},
                    ),
                ],
                returns="Query results as array of objects",
            )
        ],
        auth_flow=AuthFlow.EPHEMERAL_TOKEN,
        classification=[ToolClassification.PII_RISK, ToolClassification.REQUIRES_APPROVAL],
        rate_limits=RateLimitConfig(max_calls=30, window_seconds=60),
        endpoint="http://localhost:9003/db",
    )

    # Text processing tool (safe)
    text_processing = MCPServerRegistration(
        tool_id="text_processing",
        name="Text Processing",
        version="1.0.0",
        owner="platform-team",
        contact="platform@example.com",
        tools=[
            ToolSchema(
                name="summarize",
                description="Summarize text content",
                parameters=[
                    ToolParameter(
                        name="text",
                        type="string",
                        description="Text to summarize",
                        required=True,
                    ),
                    ToolParameter(
                        name="max_length",
                        type="number",
                        description="Maximum summary length",
                        required=False,
                        default=200,
                    ),
                ],
                returns="Summarized text",
            ),
            ToolSchema(
                name="extract_entities",
                description="Extract named entities from text",
                parameters=[
                    ToolParameter(
                        name="text",
                        type="string",
                        description="Text to analyze",
                        required=True,
                    )
                ],
                returns="List of extracted entities with type and confidence",
            ),
        ],
        auth_flow=AuthFlow.NONE,
        classification=[ToolClassification.SAFE],
        rate_limits=RateLimitConfig(max_calls=200, window_seconds=60),
        endpoint="http://localhost:9004/text",
    )

    # Register all sample tools
    await catalog.register(web_search)
    await catalog.register(file_ops)
    await catalog.register(database_query)
    await catalog.register(text_processing)
