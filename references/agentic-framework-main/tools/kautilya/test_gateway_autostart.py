"""
Test script to verify automatic MCP Gateway startup.

This script tests that the gateway manager correctly:
1. Detects if gateway is running
2. Starts it automatically if needed
3. Handles errors gracefully

Usage:
    python test_gateway_autostart.py
"""

import sys
import time
import subprocess
from pathlib import Path

# Add kautilya to path
sys.path.insert(0, str(Path(__file__).parent))

from kautilya.gateway_manager import GatewayManager
from rich.console import Console
from rich.panel import Panel

console = Console()


def test_gateway_detection():
    """Test 1: Gateway detection."""
    console.print("\n[bold cyan]Test 1: Gateway Detection[/bold cyan]")

    manager = GatewayManager(verbose=True, auto_start=False)

    if manager.is_running():
        console.print("[green]✓[/green] Gateway is running")
        return True
    else:
        console.print("[yellow]○[/yellow] Gateway is not running (will test auto-start)")
        return False


def test_gateway_autostart():
    """Test 2: Automatic gateway startup."""
    console.print("\n[bold cyan]Test 2: Automatic Gateway Startup[/bold cyan]")

    manager = GatewayManager(verbose=True, auto_start=True)

    # Kill any existing gateway first
    try:
        subprocess.run(["pkill", "-f", "uvicorn service.main:app"], capture_output=True)
        time.sleep(1)
    except Exception:
        pass

    # Try to start
    console.print("[dim]Attempting auto-start...[/dim]")

    success = manager.start()

    if success:
        console.print("[green]✓[/green] Gateway started successfully")

        # Verify it's actually running
        if manager.is_running():
            console.print("[green]✓[/green] Gateway is responding to health checks")
            return True
        else:
            console.print("[red]✗[/red] Gateway started but not responding")
            return False
    else:
        console.print("[red]✗[/red] Gateway failed to start")
        return False


def test_gateway_reuse():
    """Test 3: Reusing existing gateway."""
    console.print("\n[bold cyan]Test 3: Gateway Reuse[/bold cyan]")

    manager = GatewayManager(verbose=True, auto_start=True)

    # Should reuse existing gateway
    is_running = manager.ensure_running()

    if is_running:
        console.print("[green]✓[/green] Gateway is running (reused existing)")
        return True
    else:
        console.print("[red]✗[/red] Failed to ensure gateway is running")
        return False


def test_cleanup():
    """Test 4: Cleanup (optional - leaves gateway running)."""
    console.print("\n[bold cyan]Test 4: Cleanup[/bold cyan]")

    console.print(
        "[dim]Leaving gateway running for kautilya use.[/dim]"
    )
    console.print("[dim]To stop manually: pkill -f 'uvicorn service.main:app'[/dim]")
    return True


def main():
    """Run all tests."""
    console.print(
        Panel(
            "[bold]MCP Gateway Auto-Start Test Suite[/bold]\n\n"
            "This tests automatic gateway lifecycle management.",
            title="Gateway Manager Tests",
            border_style="cyan",
        )
    )

    tests = [
        ("Gateway Detection", test_gateway_detection),
        ("Auto-Start", test_gateway_autostart),
        ("Gateway Reuse", test_gateway_reuse),
        ("Cleanup", test_cleanup),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result, None))
        except Exception as e:
            console.print(f"[red]✗[/red] Exception: {str(e)}")
            results.append((test_name, False, str(e)))

    # Summary
    console.print("\n" + "=" * 60)
    console.print("[bold cyan]Test Summary[/bold cyan]\n")

    passed = sum(1 for _, result, _ in results if result)
    total = len(results)

    for test_name, result, error in results:
        if result:
            status = "[green]PASS[/green]"
        else:
            status = "[red]FAIL[/red]"
            if error:
                status += f" ({error[:50]}...)"
        console.print(f"  {status}  {test_name}")

    console.print(f"\n[bold]Results: {passed}/{total} tests passed[/bold]")

    if passed == total:
        console.print("\n[green]✓ All tests passed![/green]")
        console.print("\n[bold cyan]Next step:[/bold cyan]")
        console.print("  Run: kautilya")
        console.print("  Gateway should already be running!\n")
        sys.exit(0)
    else:
        console.print("\n[red]✗ Some tests failed[/red]")
        console.print("\n[yellow]Troubleshooting:[/yellow]")
        console.print("  1. Check if port 8080 is available:")
        console.print("     lsof -i :8080")
        console.print("  2. Verify virtual environment:")
        console.print("     which python")
        console.print("  3. Check dependencies:")
        console.print("     uv pip install -r requirements.txt\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
