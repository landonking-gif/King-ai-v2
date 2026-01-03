"""
Test script for external MCP server integration via kautilya.

Run this script to verify that:
1. MCP Gateway is accessible
2. Server registration works
3. Server listing works
4. Server management (enable/disable/unregister) works

Usage:
    python examples/test_external_mcp_integration.py
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from kautilya.mcp_gateway_client import (
    MCPGatewayClient,
    MCPServerRegistration,
    ToolSchema,
    ToolParameter,
    RateLimitConfig,
)
from rich.console import Console
from rich.panel import Panel

console = Console()


def test_gateway_connection():
    """Test 1: Gateway connectivity."""
    console.print("\n[bold cyan]Test 1: Gateway Connection[/bold cyan]")

    try:
        with MCPGatewayClient() as client:
            is_connected = client.test_connection_sync()

            if is_connected:
                console.print("[green]✓[/green] Gateway is reachable")
                return True
            else:
                console.print("[red]✗[/red] Gateway not responding")
                return False

    except Exception as e:
        console.print(f"[red]✗[/red] Connection failed: {str(e)}")
        return False


def test_server_registration():
    """Test 2: Server registration."""
    console.print("\n[bold cyan]Test 2: Server Registration[/bold cyan]")

    try:
        with MCPGatewayClient() as client:
            # Create test server registration
            registration = MCPServerRegistration(
                tool_id="test_weather_api",
                name="Test Weather API",
                version="1.0.0",
                owner="test-team",
                contact="test@example.com",
                endpoint="https://weather-api.example.com/v1",
                tools=[
                    ToolSchema(
                        name="get_weather",
                        description="Get weather for a city",
                        parameters=[
                            ToolParameter(
                                name="city",
                                type="string",
                                description="City name",
                                required=True,
                            ),
                            ToolParameter(
                                name="units",
                                type="string",
                                description="Units (metric/imperial)",
                                required=False,
                                default="metric",
                            ),
                        ],
                        returns="Weather data",
                    )
                ],
                auth_flow="api_key",
                classification=["external_call", "safe"],
                rate_limits=RateLimitConfig(max_calls=60, window_seconds=60),
                metadata={"api_key_env": "WEATHER_API_KEY"},
            )

            # Try to register
            result = client.register_server_sync(registration)

            console.print("[green]✓[/green] Server registered successfully")
            console.print(f"[dim]Tool ID: {result.get('tool_id')}[/dim]")
            return True

    except Exception as e:
        if "already registered" in str(e).lower():
            console.print("[yellow]⚠[/yellow] Server already registered (OK)")
            return True
        else:
            console.print(f"[red]✗[/red] Registration failed: {str(e)}")
            return False


def test_server_listing():
    """Test 3: List servers."""
    console.print("\n[bold cyan]Test 3: List Servers[/bold cyan]")

    try:
        with MCPGatewayClient() as client:
            servers = client.list_servers_sync(enabled_only=False)

            console.print(f"[green]✓[/green] Found {len(servers)} registered servers")

            # Find our test server
            test_server = next(
                (
                    s
                    for s in servers
                    if s.get("registration", {}).get("tool_id") == "test_weather_api"
                ),
                None,
            )

            if test_server:
                console.print(
                    "[green]✓[/green] Test server found in list"
                )
                console.print(f"[dim]Name: {test_server['registration']['name']}[/dim]")
                console.print(f"[dim]Enabled: {test_server['enabled']}[/dim]")
            else:
                console.print("[yellow]⚠[/yellow] Test server not found in list")

            return True

    except Exception as e:
        console.print(f"[red]✗[/red] Listing failed: {str(e)}")
        return False


def test_server_management():
    """Test 4: Enable/disable server."""
    console.print("\n[bold cyan]Test 4: Server Management[/bold cyan]")

    try:
        with MCPGatewayClient() as client:
            # Disable
            result = client.disable_server_sync("test_weather_api")
            console.print("[green]✓[/green] Server disabled")

            # Enable
            result = client.enable_server_sync("test_weather_api")
            console.print("[green]✓[/green] Server enabled")

            return True

    except Exception as e:
        console.print(f"[red]✗[/red] Management failed: {str(e)}")
        return False


def test_cleanup():
    """Test 5: Cleanup - unregister test server."""
    console.print("\n[bold cyan]Test 5: Cleanup[/bold cyan]")

    try:
        with MCPGatewayClient() as client:
            result = client.unregister_server_sync("test_weather_api")
            console.print("[green]✓[/green] Test server unregistered")
            return True

    except Exception as e:
        if "not found" in str(e).lower():
            console.print("[yellow]⚠[/yellow] Test server already removed (OK)")
            return True
        else:
            console.print(f"[red]✗[/red] Cleanup failed: {str(e)}")
            return False


def main():
    """Run all tests."""
    console.print(
        Panel(
            "[bold]External MCP Server Integration Test[/bold]\n\n"
            "This will test the kautilya external MCP server integration.\n"
            "Ensure MCP Gateway is running on http://localhost:8080",
            title="Test Suite",
            border_style="cyan",
        )
    )

    tests = [
        ("Gateway Connection", test_gateway_connection),
        ("Server Registration", test_server_registration),
        ("Server Listing", test_server_listing),
        ("Server Management", test_server_management),
        ("Cleanup", test_cleanup),
    ]

    results = []

    for test_name, test_func in tests:
        result = test_func()
        results.append((test_name, result))

    # Summary
    console.print("\n" + "=" * 60)
    console.print("[bold cyan]Test Summary[/bold cyan]\n")

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "[green]PASS[/green]" if result else "[red]FAIL[/red]"
        console.print(f"  {status}  {test_name}")

    console.print(f"\n[bold]Results: {passed}/{total} tests passed[/bold]")

    if passed == total:
        console.print("\n[green]✓ All tests passed![/green]\n")
        sys.exit(0)
    else:
        console.print("\n[red]✗ Some tests failed[/red]")
        console.print("\n[yellow]Troubleshooting:[/yellow]")
        console.print("  1. Ensure MCP Gateway is running:")
        console.print("     cd mcp-gateway && uvicorn service.main:app --port 8080")
        console.print("  2. Check environment variables:")
        console.print("     export MCP_GATEWAY_URL=http://localhost:8080")
        console.print("     export JWT_SECRET_KEY=your-secret-key\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
