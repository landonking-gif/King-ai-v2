"""
MCP Integration Example

Demonstrates how to integrate external tools via Model Context Protocol (MCP).
This example shows the configuration and usage patterns without requiring
actual MCP servers to be running.
"""

import asyncio
from pathlib import Path
from typing import Dict, Any, List
import yaml


class MCPToolSimulator:
    """Simulates MCP tool invocation for demonstration."""

    def __init__(self, config_path: Path):
        """Load MCP configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.servers = self.config.get('mcp_servers', {})

    def list_available_tools(self) -> List[Dict[str, Any]]:
        """List all available MCP tools."""
        tools = []
        for server_id, server_config in self.servers.items():
            for tool_name in server_config.get('tools', []):
                tools.append({
                    "server": server_id,
                    "tool": tool_name,
                    "scopes": server_config.get('scopes', []),
                    "rate_limit": server_config.get('rate_limit', 0)
                })
        return tools

    async def invoke_tool(
        self,
        server_id: str,
        tool_name: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Simulate tool invocation."""
        print(f"   📡 Calling {server_id}.{tool_name}")
        print(f"      Parameters: {params}")

        # Simulate network delay
        await asyncio.sleep(0.3)

        # Simulate different tool responses
        if server_id == "github" and tool_name == "search_repos":
            return self._simulate_github_search(params)
        elif server_id == "filesystem" and tool_name == "file_read":
            return self._simulate_file_read(params)
        elif server_id == "web_search" and tool_name == "search":
            return self._simulate_web_search(params)
        elif server_id == "firecrawl_mcp" and tool_name == "scrape_url":
            return self._simulate_scrape(params)
        else:
            return {"status": "success", "message": f"Tool {tool_name} executed"}

    def _simulate_github_search(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate GitHub repository search."""
        query = params.get("query", "")
        return {
            "status": "success",
            "results": [
                {
                    "name": "agentic-framework",
                    "full_name": "example/agentic-framework",
                    "description": "Enterprise multi-agent orchestration",
                    "stars": 1250,
                    "url": "https://github.com/example/agentic-framework"
                },
                {
                    "name": "llm-orchestrator",
                    "full_name": "example/llm-orchestrator",
                    "description": "LLM workflow management",
                    "stars": 890,
                    "url": "https://github.com/example/llm-orchestrator"
                }
            ],
            "total_count": 2
        }

    def _simulate_file_read(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate file read operation."""
        filepath = params.get("path", "unknown")
        return {
            "status": "success",
            "path": filepath,
            "content": "Sample file content from MCP filesystem tool...",
            "size_bytes": 156,
            "encoding": "utf-8"
        }

    def _simulate_web_search(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate web search."""
        query = params.get("query", "")
        return {
            "status": "success",
            "query": query,
            "results": [
                {
                    "title": "Multi-Agent Systems Overview",
                    "url": "https://example.com/multi-agent-systems",
                    "snippet": "Comprehensive guide to multi-agent architectures..."
                },
                {
                    "title": "Enterprise AI Adoption",
                    "url": "https://example.com/enterprise-ai",
                    "snippet": "How enterprises are adopting AI agent systems..."
                }
            ],
            "total_results": 2
        }

    def _simulate_scrape(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate web scraping."""
        url = params.get("url", "")
        return {
            "status": "success",
            "url": url,
            "title": "Example Page Title",
            "content": "Extracted content from the webpage...",
            "metadata": {
                "author": "John Doe",
                "published": "2024-01-01",
                "word_count": 1500
            }
        }


async def main():
    """Demonstrate MCP integration patterns."""
    print("=" * 60)
    print("MCP Integration Example")
    print("=" * 60)

    # Load MCP configuration
    config_path = Path(__file__).parent / "mcp_config.yaml"

    if not config_path.exists():
        print(f"\n❌ Error: MCP config not found at {config_path}")
        return

    simulator = MCPToolSimulator(config_path)

    print("\n📋 Available MCP Servers:")
    for server_id, server_config in simulator.servers.items():
        print(f"\n   {server_id}:")
        print(f"      Type: {server_config.get('type', 'unknown')}")
        print(f"      Description: {server_config.get('description', 'N/A')}")
        print(f"      Tools: {', '.join(server_config.get('tools', []))}")
        print(f"      Rate Limit: {server_config.get('rate_limit', 'N/A')}/min")

    # Demonstrate tool usage patterns
    print("\n" + "=" * 60)
    print("MCP Tool Usage Patterns")
    print("=" * 60)

    # Pattern 1: GitHub Repository Search
    print("\n[Pattern 1] GitHub Repository Search")
    result1 = await simulator.invoke_tool(
        "github",
        "search_repos",
        {"query": "agentic frameworks", "limit": 5}
    )
    print(f"   ✓ Found {result1['total_count']} repositories")
    for repo in result1['results']:
        print(f"      - {repo['full_name']} ({repo['stars']}⭐)")

    # Pattern 2: File System Access
    print("\n[Pattern 2] File System Access")
    result2 = await simulator.invoke_tool(
        "filesystem",
        "file_read",
        {"path": "/data/config.json"}
    )
    print(f"   ✓ Read {result2['size_bytes']} bytes from {result2['path']}")

    # Pattern 3: Web Search
    print("\n[Pattern 3] Web Search")
    result3 = await simulator.invoke_tool(
        "web_search",
        "search",
        {"query": "multi-agent systems", "max_results": 10}
    )
    print(f"   ✓ Found {result3['total_results']} results for '{result3['query']}'")

    # Pattern 4: Web Scraping
    print("\n[Pattern 4] Web Scraping")
    result4 = await simulator.invoke_tool(
        "firecrawl_mcp",
        "scrape_url",
        {"url": "https://example.com/article"}
    )
    print(f"   ✓ Scraped '{result4['title']}' ({result4['metadata']['word_count']} words)")

    # Show security features
    print("\n" + "=" * 60)
    print("Security & Governance")
    print("=" * 60)

    policies = simulator.config.get('policies', {})
    print(f"\n✓ PII Filtering: {policies.get('pii_filtering', False)}")
    print(f"✓ Max Retries: {policies.get('max_retries', 0)}")
    print(f"✓ Timeout: {policies.get('timeout_seconds', 0)}s")
    print(f"✓ Approval Required For:")
    for tool in policies.get('require_approval_for', []):
        print(f"   - {tool}")

    print("\n💡 What you learned:")
    print("   ✓ How to configure MCP servers")
    print("   ✓ How to define tool scopes and permissions")
    print("   ✓ How to set rate limits")
    print("   ✓ How to invoke MCP tools from agents")
    print("   ✓ How security policies work")

    print("\n📁 Files:")
    print("   - mcp_config.yaml: MCP server configuration")
    print("   - run.py: MCP integration demonstration")

    print("\nNext steps:")
    print("  - Install real MCP servers (firecrawl, github, etc.)")
    print("  - Configure environment variables for API keys")
    print("  - Deploy MCP Gateway: docker-compose up mcp-gateway")
    print("  - Read docs/mcp.md for advanced configuration")


if __name__ == "__main__":
    asyncio.run(main())
