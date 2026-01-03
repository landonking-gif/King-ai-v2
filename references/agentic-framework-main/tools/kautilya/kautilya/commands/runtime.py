"""
Runtime Management Commands.

Module: kautilya/commands/runtime.py
"""

from typing import Optional
import subprocess
import httpx
import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich.layout import Layout
from pathlib import Path
import time

console = Console()


@click.command(name="run")
@click.pass_context
def run_cmd(ctx: click.Context) -> None:
    """Run current project in dev mode."""
    config_dir = ctx.obj.get("config_dir", ".kautilya")
    run_project(config_dir)


@click.command(name="status")
@click.pass_context
def status_cmd(ctx: click.Context) -> None:
    """Show running agents, memory usage, connections."""
    config_dir = ctx.obj.get("config_dir", ".kautilya")
    show_status(config_dir)


@click.command(name="logs")
@click.argument("agent", required=False)
@click.pass_context
def logs_cmd(ctx: click.Context, agent: Optional[str]) -> None:
    """Tail logs for agent or all."""
    config_dir = ctx.obj.get("config_dir", ".kautilya")
    tail_logs(agent, config_dir)


def run_project(config_dir: str) -> None:
    """Run the current project in development mode."""
    console.print("\n[bold cyan]Starting Agent Framework Services[/bold cyan]\n")

    # Find docker-compose.yml
    compose_file = Path.cwd().parent / "docker-compose.yml"

    if not compose_file.exists():
        # Try current directory
        compose_file = Path.cwd() / "docker-compose.yml"

    if not compose_file.exists():
        console.print("[yellow]No docker-compose.yml found.[/yellow]")
        console.print("[dim]Make sure you're in an agent-framework project directory.[/dim]")
        return

    console.print(f"[dim]Using compose file: {compose_file}[/dim]\n")

    # Start services
    try:
        console.print("[cyan]Starting services with docker-compose...[/cyan]")

        result = subprocess.run(
            ["docker-compose", "-f", str(compose_file), "up", "-d"],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            console.print("[green]✓[/green] Services started successfully\n")
            console.print(result.stdout)

            # Show service URLs
            success_message = """
[bold]Service URLs:[/bold]
  - Orchestrator:     http://localhost:8000
  - Subagent Manager: http://localhost:8001
  - Memory Service:   http://localhost:8002
  - MCP Gateway:      http://localhost:8080
  - Code Executor:    http://localhost:8004

[bold]Commands:[/bold]
  - View status: kautilya /status
  - View logs:   kautilya /logs
  - Stop:        docker-compose -f {compose_file} down
            """

            console.print(Panel(success_message.strip(), title="[bold green]Services Running[/bold green]"))
        else:
            console.print("[red]✗[/red] Failed to start services\n")
            console.print(result.stderr)

    except FileNotFoundError:
        console.print("[red]docker-compose not found.[/red]")
        console.print("[dim]Install docker-compose to run services.[/dim]")
    except Exception as e:
        console.print(f"[red]Error:[/red] {str(e)}")


def show_status(config_dir: str) -> None:
    """Show status of running services."""
    console.print("\n[bold cyan]Agent Framework Status[/bold cyan]\n")

    services = [
        ("Orchestrator", "http://localhost:8000/health"),
        ("Subagent Manager", "http://localhost:8001/health"),
        ("Memory Service", "http://localhost:8002/health"),
        ("MCP Gateway", "http://localhost:8080/health"),
        ("Code Executor", "http://localhost:8004/health"),
    ]

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Service", style="cyan")
    table.add_column("Status")
    table.add_column("URL")

    async def check_services() -> None:
        async with httpx.AsyncClient(timeout=5.0) as client:
            for name, url in services:
                try:
                    response = await client.get(url)
                    if response.status_code == 200:
                        status = "[green]✓ Running[/green]"
                    else:
                        status = f"[yellow]⚠ Status {response.status_code}[/yellow]"
                except httpx.ConnectError:
                    status = "[red]✗ Not running[/red]"
                except Exception as e:
                    status = f"[red]✗ Error: {str(e)}[/red]"

                table.add_row(name, status, url)

    import anyio
    anyio.run(check_services)

    console.print(table)
    console.print()


def tail_logs(agent_name: Optional[str], config_dir: str) -> None:
    """Tail logs for specified agent or all services."""
    console.print(f"\n[bold cyan]Tailing Logs{f': {agent_name}' if agent_name else ''}[/bold cyan]\n")

    # Find docker-compose.yml
    compose_file = Path.cwd().parent / "docker-compose.yml"

    if not compose_file.exists():
        compose_file = Path.cwd() / "docker-compose.yml"

    if not compose_file.exists():
        console.print("[yellow]No docker-compose.yml found.[/yellow]")
        return

    try:
        # Map agent names to service names
        service_map = {
            "orchestrator": "orchestrator",
            "subagent": "subagent-manager",
            "memory": "memory-service",
            "mcp": "mcp-gateway",
            "executor": "code-exec",
        }

        service_name = service_map.get(agent_name.lower()) if agent_name else None

        cmd = ["docker-compose", "-f", str(compose_file), "logs", "-f"]

        if service_name:
            cmd.append(service_name)

        console.print("[dim]Press Ctrl+C to stop tailing logs[/dim]\n")

        subprocess.run(cmd)

    except KeyboardInterrupt:
        console.print("\n[yellow]Stopped tailing logs[/yellow]")
    except FileNotFoundError:
        console.print("[red]docker-compose not found.[/red]")
    except Exception as e:
        console.print(f"[red]Error:[/red] {str(e)}")


def run_project_programmatic(config_dir: str = ".kautilya") -> dict:
    """
    Programmatic interface for running projects.

    Args:
        config_dir: Configuration directory

    Returns:
        Run status dictionary
    """
    services = []

    # Check for docker-compose
    compose_file = Path.cwd().parent / "docker-compose.yml"
    if not compose_file.exists():
        compose_file = Path.cwd() / "docker-compose.yml"

    if compose_file.exists():
        services = ["orchestrator", "memory-service", "mcp-gateway", "subagent-manager", "code-exec"]

    return {
        "status": "starting",
        "services": services,
        "message": "Use 'docker-compose up -d' to start services",
    }


def show_status_programmatic(config_dir: str = ".kautilya") -> dict:
    """
    Programmatic interface for showing status.

    Args:
        config_dir: Configuration directory

    Returns:
        Status dictionary
    """
    services = []

    # Check service health
    service_urls = {
        "orchestrator": "http://localhost:8000/health",
        "memory-service": "http://localhost:8002/health",
        "mcp-gateway": "http://localhost:8080/health",
        "subagent-manager": "http://localhost:8001/health",
        "code-executor": "http://localhost:8004/health",
    }

    for service, url in service_urls.items():
        try:
            response = httpx.get(url, timeout=2.0)
            services.append({
                "name": service,
                "status": "running" if response.status_code == 200 else "unhealthy",
                "url": url,
            })
        except Exception:
            services.append({
                "name": service,
                "status": "stopped",
                "url": url,
            })

    return {
        "services": services,
        "agents_running": sum(1 for s in services if s["status"] == "running"),
        "memory_usage": "N/A",
    }


def show_logs_programmatic(
    agent_name: Optional[str] = None,
    config_dir: str = ".kautilya",
) -> dict:
    """
    Programmatic interface for showing logs.

    Args:
        agent_name: Agent name to show logs for
        config_dir: Configuration directory

    Returns:
        Logs dictionary
    """
    return {
        "agent": agent_name or "all",
        "logs": [],
        "message": "Use 'docker-compose logs' for detailed logs",
    }
