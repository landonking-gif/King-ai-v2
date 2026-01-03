"""
Example client for MCP Gateway.

Demonstrates how to interact with the gateway programmatically.
"""

import asyncio
import httpx
from typing import Any, Optional


class MCPGatewayClient:
    """Client for interacting with MCP Gateway."""

    def __init__(self, base_url: str = "http://localhost:8080") -> None:
        """
        Initialize client.

        Args:
            base_url: Gateway base URL
        """
        self.base_url = base_url
        self.client = httpx.AsyncClient(base_url=base_url, timeout=30.0)

    async def close(self) -> None:
        """Close HTTP client."""
        await self.client.aclose()

    async def health_check(self) -> dict[str, Any]:
        """
        Check gateway health.

        Returns:
            Health status
        """
        response = await self.client.get("/health")
        response.raise_for_status()
        return response.json()

    async def list_tools(self, enabled_only: bool = True) -> dict[str, Any]:
        """
        List available tools.

        Args:
            enabled_only: Return only enabled tools

        Returns:
            Catalog listing
        """
        response = await self.client.get(
            "/catalog/tools", params={"enabled_only": enabled_only}
        )
        response.raise_for_status()
        return response.json()

    async def get_tool_schema(self, tool_id: str, tool_name: str) -> dict[str, Any]:
        """
        Get tool schema.

        Args:
            tool_id: Server ID
            tool_name: Tool name

        Returns:
            Tool schema
        """
        response = await self.client.get(f"/catalog/tools/{tool_id}/schema/{tool_name}")
        response.raise_for_status()
        return response.json()

    async def mint_token(
        self,
        tool_ids: list[str],
        actor_id: str,
        actor_type: str = "subagent",
        max_invocations: Optional[int] = None,
        ttl_minutes: Optional[int] = None,
    ) -> dict[str, Any]:
        """
        Mint ephemeral token.

        Args:
            tool_ids: Allowed tool IDs
            actor_id: Actor identifier
            actor_type: Actor type
            max_invocations: Maximum invocations allowed
            ttl_minutes: Token TTL in minutes

        Returns:
            Token response
        """
        payload = {
            "scope": {
                "tool_ids": tool_ids,
                "actor_id": actor_id,
                "actor_type": actor_type,
                "max_invocations": max_invocations,
            },
            "ttl_minutes": ttl_minutes,
        }
        response = await self.client.post("/auth/token", json=payload)
        response.raise_for_status()
        return response.json()

    async def invoke_tool(
        self,
        tool_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        actor_id: str,
        actor_type: str = "lead_agent",
        runtime_mode: str = "orchestrated",
        token: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Invoke a tool.

        Args:
            tool_id: Server ID
            tool_name: Tool name
            arguments: Tool arguments
            actor_id: Actor identifier
            actor_type: Actor type
            runtime_mode: Runtime mode (orchestrated or scoped_direct)
            token: Ephemeral token (required for scoped_direct mode)

        Returns:
            Invocation response
        """
        payload = {
            "tool_id": tool_id,
            "tool_name": tool_name,
            "arguments": arguments,
            "actor_id": actor_id,
            "actor_type": actor_type,
            "runtime_mode": runtime_mode,
            "token": token,
        }
        response = await self.client.post("/tools/invoke", json=payload)
        response.raise_for_status()
        return response.json()

    async def get_provenance_logs(
        self, limit: int = 100, tool_id: Optional[str] = None
    ) -> list[dict[str, Any]]:
        """
        Get provenance logs.

        Args:
            limit: Maximum logs to return
            tool_id: Optional filter by tool ID

        Returns:
            List of provenance logs
        """
        params = {"limit": limit}
        if tool_id:
            params["tool_id"] = tool_id

        response = await self.client.get("/provenance/logs", params=params)
        response.raise_for_status()
        return response.json()


async def main() -> None:
    """Example usage of the MCP Gateway client."""
    client = MCPGatewayClient()

    try:
        # 1. Health check
        print("=" * 60)
        print("1. Health Check")
        print("=" * 60)
        health = await client.health_check()
        print(f"Status: {health['status']}")
        print(f"Version: {health['version']}")
        print(f"Catalog size: {health['catalog_size']}")
        print()

        # 2. List tools
        print("=" * 60)
        print("2. List Available Tools")
        print("=" * 60)
        catalog = await client.list_tools()
        print(f"Total tools: {catalog['total']}")
        for server in catalog["servers"]:
            reg = server["registration"]
            print(f"\n  - {reg['tool_id']} (v{reg['version']})")
            print(f"    Name: {reg['name']}")
            print(f"    Tools: {', '.join([t['name'] for t in reg['tools']])}")
            print(f"    Classification: {', '.join(reg['classification'])}")
        print()

        # 3. Get specific tool schema
        print("=" * 60)
        print("3. Get Tool Schema")
        print("=" * 60)
        schema = await client.get_tool_schema("text_processing", "summarize")
        print(f"Tool: {schema['tool_name']}")
        print(f"Description: {schema['schema']['description']}")
        print("Parameters:")
        for param in schema["schema"]["parameters"]:
            print(f"  - {param['name']} ({param['type']}): {param['description']}")
        print()

        # 4. Invoke tool in orchestrated mode
        print("=" * 60)
        print("4. Invoke Tool (Orchestrated Mode)")
        print("=" * 60)
        result = await client.invoke_tool(
            tool_id="text_processing",
            tool_name="summarize",
            arguments={
                "text": "This is a long text that demonstrates the text processing capabilities of the MCP Gateway. "
                "The gateway provides tool discovery, authentication, rate limiting, and provenance logging.",
                "max_length": 50,
            },
            actor_id="example_lead_agent",
            actor_type="lead_agent",
        )
        print(f"Success: {result['success']}")
        print(f"Result: {result['result']}")
        print(f"Execution time: {result['execution_time_ms']:.2f}ms")
        print(f"PII detected: {result['pii_detected']}")
        print()

        # 5. Mint token and use scoped direct mode
        print("=" * 60)
        print("5. Mint Token & Invoke (Scoped Direct Mode)")
        print("=" * 60)
        token_response = await client.mint_token(
            tool_ids=["text_processing", "web_search"],
            actor_id="example_subagent",
            actor_type="subagent",
            max_invocations=10,
            ttl_minutes=15,
        )
        print(f"Token minted: {token_response['token'][:50]}...")
        print(f"Expires at: {token_response['expires_at']}")
        print(f"Allowed tools: {', '.join(token_response['scope']['tool_ids'])}")
        print()

        # Use the token
        result = await client.invoke_tool(
            tool_id="text_processing",
            tool_name="extract_entities",
            arguments={"text": "Microsoft is headquartered in Redmond, Washington"},
            actor_id="example_subagent",
            actor_type="subagent",
            runtime_mode="scoped_direct",
            token=token_response["token"],
        )
        print(f"Invocation success: {result['success']}")
        print(f"Entities extracted: {result['result']}")
        print()

        # 6. View provenance logs
        print("=" * 60)
        print("6. Provenance Logs")
        print("=" * 60)
        logs = await client.get_provenance_logs(limit=5)
        print(f"Recent invocations: {len(logs)}")
        for log in logs[-3:]:  # Show last 3
            print(f"\n  [{log['timestamp']}]")
            print(f"  Tool: {log['tool_id']}/{log['tool_name']}")
            print(f"  Actor: {log['actor_type']} ({log['actor_id']})")
            print(f"  Success: {log['success']}")
            print(f"  Execution: {log['execution_time_ms']:.2f}ms")
            print(f"  Mode: {log['runtime_mode']}")

    except Exception as e:
        print(f"Error: {e}")

    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
