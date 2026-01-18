"""
MCP Server Management Commands.

Module: kautilya/commands/mcp.py
"""

from typing import Optional, Dict, Any, List
from pathlib import Path
import click
import questionary
import yaml
import os
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


@click.group(name="mcp")
def mcp_cmd() -> None:
    """Manage MCP servers."""
    pass


@mcp_cmd.command(name="add")
@click.argument("server")
@click.pass_context
def mcp_add_cmd(ctx: click.Context, server: str) -> None:
    """Add MCP server to manifest."""
    config_dir = ctx.obj.get("config_dir", ".kautilya")
    add_mcp_server(server, config_dir, interactive=False)


@mcp_cmd.command(name="list")
def mcp_list_cmd() -> None:
    """List registered MCP servers from catalog."""
    list_mcp_servers()


@mcp_cmd.command(name="test")
@click.argument("server")
@click.pass_context
def mcp_test_cmd(ctx: click.Context, server: str) -> None:
    """Test MCP server connection."""
    config_dir = ctx.obj.get("config_dir", ".kautilya")
    test_mcp_server(server, config_dir)


@mcp_cmd.command(name="register-external")
@click.pass_context
def mcp_register_external_cmd(ctx: click.Context) -> None:
    """Register an external MCP server interactively."""
    config_dir = ctx.obj.get("config_dir", ".kautilya")
    register_external_server_interactive(config_dir)


@mcp_cmd.command(name="import")
@click.argument("source")
@click.option("--format", type=click.Choice(["json", "yaml"]), default="yaml")
@click.pass_context
def mcp_import_cmd(ctx: click.Context, source: str, format: str) -> None:
    """Import external MCP server from URL or file."""
    config_dir = ctx.obj.get("config_dir", ".kautilya")
    import_external_server(source, format, config_dir)


@mcp_cmd.command(name="unregister")
@click.argument("tool_id")
@click.pass_context
def mcp_unregister_cmd(ctx: click.Context, tool_id: str) -> None:
    """Unregister an MCP server from the gateway."""
    unregister_server(tool_id)


@mcp_cmd.command(name="enable")
@click.argument("tool_id")
@click.pass_context
def mcp_enable_cmd(ctx: click.Context, tool_id: str) -> None:
    """Enable a registered MCP server."""
    enable_disable_server(tool_id, enable=True)


@mcp_cmd.command(name="disable")
@click.argument("tool_id")
@click.pass_context
def mcp_disable_cmd(ctx: click.Context, tool_id: str) -> None:
    """Disable a registered MCP server."""
    enable_disable_server(tool_id, enable=False)


def add_mcp_server(
    server_name: str,
    config_dir: str,
    interactive: bool = True,
) -> None:
    """Add MCP server to project."""
    console.print(f"\n[bold cyan]Adding MCP Server: {server_name}[/bold cyan]\n")

    # Get server details
    scopes = []
    rate_limit = 60

    if interactive:
        scopes_input = questionary.text(
            "Scopes needed (comma-separated):",
            default="read,write",
        ).ask()
        scopes = [s.strip() for s in scopes_input.split(",") if s.strip()]

        rate_limit = int(
            questionary.text("Rate limit (calls/min):", default="60").ask()
        )

    # Find manifest to add to
    manifests_dir = Path.cwd() / "manifests"
    if not manifests_dir.exists():
        console.print("[yellow]No manifests directory found. Create a manifest first.[/yellow]")
        return

    manifest_files = list(manifests_dir.glob("*.yaml"))
    if not manifest_files:
        console.print("[yellow]No manifests found. Create a manifest first.[/yellow]")
        return

    if interactive and len(manifest_files) > 1:
        manifest_choices = [f.name for f in manifest_files]
        manifest_name = questionary.select(
            "Add to manifest:", choices=manifest_choices
        ).ask()
        manifest_file = manifests_dir / manifest_name
    else:
        manifest_file = manifest_files[0]

    # Load manifest
    with open(manifest_file, "r") as f:
        manifest = yaml.safe_load(f) or {}

    # Add MCP server to tools
    if "tools" not in manifest:
        manifest["tools"] = {}

    if "catalog_ids" not in manifest["tools"]:
        manifest["tools"]["catalog_ids"] = []

    # Add server config
    server_config = {server_name: {"scopes": scopes, "rate_limit": rate_limit}}

    if isinstance(manifest["tools"]["catalog_ids"], list):
        # Convert to dict format
        existing = manifest["tools"]["catalog_ids"]
        manifest["tools"]["catalog_ids"] = [
            s if isinstance(s, dict) else s for s in existing
        ]

    manifest["tools"]["catalog_ids"].append(server_config)

    # Save manifest
    with open(manifest_file, "w") as f:
        yaml.dump(manifest, f, default_flow_style=False)

    success_message = f"""
[green]✓[/green] Added {server_name} MCP to {manifest_file.name}

[bold]Configuration:[/bold]
  Scopes: {', '.join(scopes)}
  Rate limit: {rate_limit} calls/min
    """

    console.print(Panel(success_message.strip(), title="[bold green]MCP Added[/bold green]"))


def list_mcp_servers() -> None:
    """List available MCP servers from MCP Gateway."""
    from ..mcp_gateway_client import MCPGatewayClient

    console.print("\n[bold cyan]Fetching MCP servers from Gateway...[/bold cyan]\n")

    try:
        with MCPGatewayClient() as client:
            # Test connection first
            if not client.test_connection_sync():
                console.print(
                    "[yellow]⚠ MCP Gateway not reachable. Showing cached list.[/yellow]\n"
                )
                _list_cached_servers()
                return

            servers = client.list_servers_sync(enabled_only=False)

            if not servers:
                console.print("[yellow]No MCP servers registered in gateway.[/yellow]")
                console.print(
                    "[dim]Use 'kautilya mcp register-external' to add servers.[/dim]\n"
                )
                return

            table = Table(
                title="Registered MCP Servers",
                show_header=True,
                header_style="bold magenta",
            )
            table.add_column("Tool ID", style="cyan", no_wrap=True)
            table.add_column("Name", style="green")
            table.add_column("Version", style="dim")
            table.add_column("Endpoint", style="blue")
            table.add_column("Status", style="yellow")
            table.add_column("Calls", style="dim")

            for server in servers:
                reg = server.get("registration", {})
                status = "✓ Enabled" if server.get("enabled") else "○ Disabled"
                calls = server.get("call_count", 0)

                # Truncate endpoint for display
                endpoint = reg.get("endpoint") or "N/A"
                if len(endpoint) > 40:
                    endpoint = endpoint[:37] + "..."

                table.add_row(
                    reg.get("tool_id", ""),
                    reg.get("name", ""),
                    reg.get("version", ""),
                    endpoint,
                    status,
                    str(calls),
                )

            console.print(table)
            console.print(
                f"\n[dim]Total: {len(servers)} servers "
                f"({sum(1 for s in servers if s.get('enabled'))} enabled)[/dim]\n"
            )

    except Exception as e:
        console.print(f"[red]Error connecting to MCP Gateway:[/red] {str(e)}")
        console.print("[yellow]Showing cached server list instead:[/yellow]\n")
        _list_cached_servers()


def _list_cached_servers() -> None:
    """List hardcoded MCP servers (fallback when gateway unavailable)."""
    table = Table(
        title="Available MCP Servers (Cached)",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Server", style="cyan")
    table.add_column("Description")
    table.add_column("Status")

    servers = [
        ("filesystem", "Local file operations", "✓ Available"),
        ("github", "GitHub API integration", "✓ Available"),
        ("postgres", "Database queries", "✓ Available"),
        ("slack", "Slack messaging", "○ Requires approval"),
        ("jira", "Jira ticket management", "○ Requires approval"),
        ("web_search", "Web search capability", "✓ Available"),
    ]

    for server, desc, status in servers:
        table.add_row(server, desc, status)

    console.print(table)
    console.print()


def test_mcp_server(server_name: str, config_dir: str) -> None:
    """Test MCP server connection."""
    console.print(f"\n[bold cyan]Testing MCP Server: {server_name}[/bold cyan]\n")

    # TODO: Actually test the connection
    console.print(f"[green]✓[/green] Server: {server_name}")
    console.print("[dim]Note: Full connection test not yet implemented[/dim]")


def add_mcp_server_programmatic(
    server_name: str,
    scopes: Optional[list] = None,
    config_dir: str = ".kautilya",
) -> dict:
    """
    Programmatic interface for adding MCP servers.

    Args:
        server_name: Server name
        scopes: Permission scopes
        config_dir: Configuration directory

    Returns:
        Server configuration dictionary
    """
    scopes = scopes or []

    # Load existing config
    config_path = Path.cwd() / config_dir / "config.yaml"
    config = {}
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}

    if "mcp_servers" not in config:
        config["mcp_servers"] = {}

    config["mcp_servers"][server_name] = {
        "enabled": True,
        "scopes": scopes,
    }

    # Save config
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    return {
        "server": server_name,
        "scopes": scopes,
        "enabled": True,
    }


def register_external_server_interactive(config_dir: str) -> None:
    """Interactively register an external MCP server."""
    from ..mcp_gateway_client import (
        MCPGatewayClient,
        MCPServerRegistration,
        ToolSchema,
        ToolParameter,
        RateLimitConfig,
    )

    console.print("\n[bold cyan]Register External MCP Server[/bold cyan]\n")

    # Collect basic info
    tool_id = questionary.text("Server ID (unique identifier):").ask()
    if not tool_id:
        console.print("[yellow]Registration cancelled.[/yellow]")
        return

    name = questionary.text("Server name:", default=tool_id.replace("_", " ").title()).ask()
    version = questionary.text("Version (semver):", default="1.0.0").ask()
    endpoint = questionary.text("Endpoint URL (e.g., https://api.example.com/v1):").ask()

    if not endpoint:
        console.print("[yellow]Endpoint required. Registration cancelled.[/yellow]")
        return

    # Contact info
    owner = questionary.text("Owner/Team:", default="platform-team").ask()
    contact = questionary.text("Contact email:", default="platform@example.com").ask()

    # Authentication
    auth_flow = questionary.select(
        "Authentication method:",
        choices=["none", "api_key", "oauth2", "ephemeral_token"],
    ).ask()

    metadata = {}
    if auth_flow == "api_key":
        api_key_env = questionary.text(
            "Environment variable for API key:", default=f"{tool_id.upper()}_API_KEY"
        ).ask()
        metadata["api_key_env"] = api_key_env

    # Security classification
    console.print("\n[bold]Security Classification:[/bold]")
    classifications = questionary.checkbox(
        "Select all that apply:",
        choices=[
            "safe",
            "external_call",
            "pii_risk",
            "side_effect",
            "requires_approval",
        ],
    ).ask()

    # Rate limits
    has_rate_limit = questionary.confirm("Configure rate limiting?", default=True).ask()
    rate_limits = None
    if has_rate_limit:
        max_calls = int(
            questionary.text("Max calls per window:", default="100").ask()
        )
        window_seconds = int(
            questionary.text("Window duration (seconds):", default="60").ask()
        )
        rate_limits = RateLimitConfig(max_calls=max_calls, window_seconds=window_seconds)

    # Tools
    console.print("\n[bold]Tool Definitions:[/bold]")
    tools = []
    while True:
        add_tool = questionary.confirm(
            f"Add tool #{len(tools) + 1}?" if tools else "Add first tool?",
            default=True,
        ).ask()

        if not add_tool:
            break

        tool_name = questionary.text("Tool name:").ask()
        tool_desc = questionary.text("Tool description:").ask()

        # Parameters
        parameters = []
        while True:
            add_param = questionary.confirm(
                f"Add parameter #{len(parameters) + 1}?" if parameters else "Add parameter?",
                default=True,
            ).ask()

            if not add_param:
                break

            param_name = questionary.text("Parameter name:").ask()
            param_type = questionary.select(
                "Parameter type:",
                choices=["string", "number", "boolean", "array", "object"],
            ).ask()
            param_desc = questionary.text("Parameter description:").ask()
            param_required = questionary.confirm("Required?", default=False).ask()

            parameters.append(
                ToolParameter(
                    name=param_name,
                    type=param_type,
                    description=param_desc,
                    required=param_required,
                )
            )

        returns_desc = questionary.text("Returns description (optional):").ask()

        tools.append(
            ToolSchema(
                name=tool_name,
                description=tool_desc,
                parameters=parameters,
                returns=returns_desc if returns_desc else None,
            )
        )

    if not tools:
        console.print("[yellow]At least one tool required. Registration cancelled.[/yellow]")
        return

    # Create registration
    registration = MCPServerRegistration(
        tool_id=tool_id,
        name=name,
        version=version,
        owner=owner,
        contact=contact,
        endpoint=endpoint,
        tools=tools,
        auth_flow=auth_flow,
        classification=classifications,
        rate_limits=rate_limits,
        metadata=metadata,
    )

    # Show summary
    console.print("\n[bold cyan]Registration Summary:[/bold cyan]")
    console.print(f"  Tool ID: {tool_id}")
    console.print(f"  Name: {name}")
    console.print(f"  Endpoint: {endpoint}")
    console.print(f"  Auth: {auth_flow}")
    console.print(f"  Tools: {len(tools)}")
    console.print(f"  Classification: {', '.join(classifications)}")

    confirm = questionary.confirm("\nProceed with registration?", default=True).ask()
    if not confirm:
        console.print("[yellow]Registration cancelled.[/yellow]")
        return

    # Register with gateway
    console.print("\n[dim]Registering with MCP Gateway...[/dim]")

    try:
        with MCPGatewayClient() as client:
            result = client.register_server_sync(registration)
            console.print(f"\n[green]✓[/green] Server registered successfully!")
            console.print(f"[dim]Tool ID: {result.get('tool_id')}[/dim]")
            console.print(f"[dim]Registered at: {result.get('registered_at')}[/dim]\n")

            # Save to local config
            _save_external_server_config(tool_id, registration, config_dir)

    except Exception as e:
        console.print(f"\n[red]Registration failed:[/red] {str(e)}\n")


def import_external_server(source: str, format: str, config_dir: str) -> None:
    """Import external MCP server from URL or file."""
    from ..mcp_gateway_client import MCPGatewayClient, MCPServerRegistration
    import json

    console.print(f"\n[bold cyan]Importing MCP server from:[/bold cyan] {source}\n")

    try:
        # Load server definition
        if source.startswith("http://") or source.startswith("https://"):
            import httpx

            response = httpx.get(source)
            response.raise_for_status()
            data = response.json() if format == "json" else yaml.safe_load(response.text)
        else:
            # Local file
            with open(source, "r") as f:
                data = json.load(f) if format == "json" else yaml.safe_load(f)

        # Create registration from data
        registration = MCPServerRegistration(**data)

        console.print(f"[green]✓[/green] Loaded server: {registration.name}")
        console.print(f"[dim]Tool ID: {registration.tool_id}[/dim]")
        console.print(f"[dim]Endpoint: {registration.endpoint}[/dim]")
        console.print(f"[dim]Tools: {len(registration.tools)}[/dim]\n")

        # Safety review
        if "pii_risk" in registration.classification or "side_effect" in registration.classification:
            console.print(
                "[yellow]⚠  Warning: This server has safety flags:[/yellow]"
            )
            for flag in registration.classification:
                console.print(f"  - {flag}")

            confirm = questionary.confirm("\nProceed with registration?", default=False).ask()
            if not confirm:
                console.print("[yellow]Import cancelled.[/yellow]")
                return

        # Register with gateway
        with MCPGatewayClient() as client:
            result = client.register_server_sync(registration)
            console.print(f"[green]✓[/green] Server imported and registered successfully!\n")

            # Save to local config
            _save_external_server_config(registration.tool_id, registration, config_dir)

    except Exception as e:
        console.print(f"[red]Import failed:[/red] {str(e)}\n")


def unregister_server(tool_id: str) -> None:
    """Unregister an MCP server from the gateway."""
    from ..mcp_gateway_client import MCPGatewayClient

    console.print(f"\n[bold yellow]Unregistering server:[/bold yellow] {tool_id}\n")

    confirm = questionary.confirm(
        "Are you sure? This will remove the server from the gateway.",
        default=False,
    ).ask()

    if not confirm:
        console.print("[yellow]Unregister cancelled.[/yellow]")
        return

    try:
        with MCPGatewayClient() as client:
            result = client.unregister_server_sync(tool_id)
            console.print(f"[green]✓[/green] {result.get('message', 'Server unregistered')}\n")

    except Exception as e:
        console.print(f"[red]Unregister failed:[/red] {str(e)}\n")


def enable_disable_server(tool_id: str, enable: bool = True) -> None:
    """Enable or disable an MCP server."""
    from ..mcp_gateway_client import MCPGatewayClient

    action = "Enabling" if enable else "Disabling"
    console.print(f"\n[bold cyan]{action} server:[/bold cyan] {tool_id}\n")

    try:
        with MCPGatewayClient() as client:
            if enable:
                result = client.enable_server_sync(tool_id)
            else:
                result = client.disable_server_sync(tool_id)

            console.print(f"[green]✓[/green] {result.get('message', 'Server updated')}\n")

    except Exception as e:
        console.print(f"[red]Failed:[/red] {str(e)}\n")


def _save_external_server_config(
    tool_id: str, registration: Any, config_dir: str
) -> None:
    """Save external server configuration to local file."""
    config_path = Path.cwd() / config_dir / "external_mcp_servers.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing config
    config = {"servers": []}
    if config_path.exists():
        with open(config_path, "r") as f:
            config = yaml.safe_load(f) or {"servers": []}

    # Remove existing entry for this tool_id
    config["servers"] = [
        s for s in config["servers"] if s.get("tool_id") != tool_id
    ]

    # Add new entry
    config["servers"].append(registration.model_dump())

    # Save
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    console.print(f"[dim]Saved to: {config_path}[/dim]")
