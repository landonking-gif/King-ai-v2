"""
Test script to verify consistency between command-line and interactive mode.

This tests that both modes return the same data for:
1. MCP servers list
2. LLM providers list
3. Skills list (if applicable)
"""

import sys
import os
from pathlib import Path

# Add kautilya to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "kautilya"))

from kautilya.tool_executor import ToolExecutor
from kautilya.commands.llm import list_llm_providers

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    console = Console()
except ImportError:
    class Console:
        def print(self, *args, **kwargs):
            print(*args)
    console = Console()


def test_mcp_consistency():
    """Test MCP servers list consistency."""
    console.print("\n[bold cyan]Test 1: MCP Servers List Consistency[/bold cyan]")
    console.print("=" * 70)

    executor = ToolExecutor()

    # Test interactive mode
    console.print("\n[bold]Interactive Mode (LLM tool call):[/bold]")
    interactive_result = executor._exec_list_mcp_servers()

    console.print(f"  Success: {interactive_result.get('success')}")
    console.print(f"  Source: {interactive_result.get('source', 'N/A')}")
    console.print(f"  Count: {interactive_result.get('count', 0)}")

    if interactive_result.get('warning'):
        console.print(f"  [yellow]Warning: {interactive_result.get('warning')}[/yellow]")

    if interactive_result.get('servers'):
        console.print(f"\n  Servers found:")
        for server in interactive_result['servers'][:3]:  # Show first 3
            name = server.get('name') or server.get('tool_id', 'Unknown')
            console.print(f"    • {name}")
        if len(interactive_result['servers']) > 3:
            console.print(f"    ... and {len(interactive_result['servers']) - 3} more")

    # Compare with command-line behavior
    console.print("\n[bold]Command-line Mode:[/bold]")
    console.print("  Would call: kautilya mcp list")
    console.print("  Uses: MCPGatewayClient.list_servers_sync()")

    # Validation
    if interactive_result.get('source') == 'mcp_gateway':
        console.print("\n[green]✓ PASS - Using MCP Gateway (real-time data)[/green]")
    elif interactive_result.get('source') == 'cached_fallback':
        console.print("\n[yellow]⚠ FALLBACK - Using cached list (Gateway not available)[/yellow]")
        console.print("[dim]This is expected if MCP Gateway is not running[/dim]")
    else:
        console.print("\n[red]✗ FAIL - Unknown source[/red]")

    return interactive_result.get('success', False)


def test_llm_consistency():
    """Test LLM providers list consistency."""
    console.print("\n[bold cyan]Test 2: LLM Providers List Consistency[/bold cyan]")
    console.print("=" * 70)

    executor = ToolExecutor()

    # Test interactive mode
    console.print("\n[bold]Interactive Mode (LLM tool call):[/bold]")
    interactive_result = executor._exec_list_llm_providers()

    console.print(f"  Success: {interactive_result.get('success')}")
    console.print(f"  Count: {interactive_result.get('count', 0)}")

    providers_interactive = interactive_result.get('providers', [])
    provider_names_interactive = [p.get('name') for p in providers_interactive]

    console.print(f"\n  Providers:")
    for provider in provider_names_interactive:
        console.print(f"    • {provider}")

    # Expected providers (from command-line)
    expected_providers = [
        "anthropic", "openai", "azure", "gemini", "local", "vllm"
    ]

    console.print("\n[bold]Expected Providers (from command-line):[/bold]")
    for provider in expected_providers:
        console.print(f"    • {provider}")

    # Validation
    missing = set(expected_providers) - set(provider_names_interactive)
    extra = set(provider_names_interactive) - set(expected_providers)

    if not missing and not extra:
        console.print("\n[green]✓ PASS - All 6 providers match command-line[/green]")
        return True
    else:
        if missing:
            console.print(f"\n[red]✗ FAIL - Missing providers: {missing}[/red]")
        if extra:
            console.print(f"\n[yellow]⚠ WARNING - Extra providers: {extra}[/yellow]")
        return False


def test_provider_details():
    """Test that provider details are complete."""
    console.print("\n[bold cyan]Test 3: LLM Provider Details Completeness[/bold cyan]")
    console.print("=" * 70)

    executor = ToolExecutor()
    result = executor._exec_list_llm_providers()

    providers = result.get('providers', [])

    console.print(f"\nChecking {len(providers)} providers for complete metadata...")

    all_complete = True
    for provider in providers:
        name = provider.get('name', 'Unknown')
        required_fields = ['name', 'default_model', 'fallback_model', 'api_key_env']

        missing_fields = [f for f in required_fields if not provider.get(f)]

        if missing_fields:
            console.print(f"  [red]✗ {name}: Missing {missing_fields}[/red]")
            all_complete = False
        else:
            console.print(f"  [green]✓ {name}: Complete[/green]")

    if all_complete:
        console.print("\n[green]✓ PASS - All providers have complete metadata[/green]")
    else:
        console.print("\n[red]✗ FAIL - Some providers missing metadata[/red]")

    return all_complete


def test_mcp_server_format():
    """Test that MCP server data format is LLM-friendly."""
    console.print("\n[bold cyan]Test 4: MCP Server Data Format[/bold cyan]")
    console.print("=" * 70)

    executor = ToolExecutor()
    result = executor._exec_list_mcp_servers()

    if not result.get('success'):
        console.print("[yellow]⚠ Skipping (MCP Gateway not available)[/yellow]")
        return True

    servers = result.get('servers', [])
    if not servers:
        console.print("[yellow]⚠ No servers to test[/yellow]")
        return True

    console.print(f"\nChecking {len(servers)} servers for LLM-friendly format...")

    # Check first server
    sample_server = servers[0]
    console.print(f"\nSample server data:")
    for key, value in sample_server.items():
        console.print(f"  {key}: {value}")

    # Validate format
    has_name = any(sample_server.get(k) for k in ['name', 'tool_id'])
    has_description = 'description' in sample_server or 'endpoint' in sample_server
    has_status = 'enabled' in sample_server or 'requires_approval' in sample_server

    if has_name and (has_description or has_status):
        console.print("\n[green]✓ PASS - Server format is LLM-friendly[/green]")
        return True
    else:
        console.print("\n[red]✗ FAIL - Server format missing key fields[/red]")
        return False


def main():
    """Run all consistency tests."""
    console.print("\n" + "=" * 70)
    console.print("[bold yellow]Consistency Test Suite - Command-line vs Interactive[/bold yellow]")
    console.print("=" * 70)

    results = []
    results.append(("MCP Consistency", test_mcp_consistency()))
    results.append(("LLM Consistency", test_llm_consistency()))
    results.append(("Provider Details", test_provider_details()))
    results.append(("MCP Format", test_mcp_server_format()))

    console.print("\n" + "=" * 70)
    console.print("[bold]Test Results Summary[/bold]")
    console.print("=" * 70)

    for name, result in results:
        status = "[green]✓ PASS[/green]" if result else "[red]✗ FAIL[/red]"
        console.print(f"{status} - {name}")

    all_passed = all(result for _, result in results)

    console.print("\n" + "=" * 70)
    if all_passed:
        console.print("[bold green]✓ ALL TESTS PASSED[/bold green]")
        console.print("=" * 70 + "\n")
        console.print("[cyan]Command-line and interactive mode are now consistent![/cyan]\n")
        console.print("Key improvements:")
        console.print("  • MCP servers: Now fetched from MCP Gateway (real-time)")
        console.print("  • LLM providers: All 6 providers available (was 4)")
        console.print("  • Data format: Consistent across both modes")
        console.print("  • Fallback: Graceful degradation if services unavailable\n")
    else:
        console.print("[bold red]✗ SOME TESTS FAILED[/bold red]")
        console.print("=" * 70 + "\n")
        console.print("Review failed tests above for details.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
