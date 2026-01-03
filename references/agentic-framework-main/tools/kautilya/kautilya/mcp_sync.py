"""
MCP Server Auto-Sync from YAML Files.

Module: kautilya/mcp_sync.py

Automatically syncs MCP server registrations from YAML files to the gateway.
"""

from typing import List, Dict, Any
from pathlib import Path
import yaml
from rich.console import Console

from .mcp_gateway_client import MCPGatewayClient, MCPServerRegistration

console = Console()


def sync_mcp_servers_from_yaml(
    config_dir: str = ".kautilya",
    verbose: bool = False,
) -> None:
    """
    Sync MCP servers from YAML files to the gateway.

    Looks for YAML files in:
    1. examples/ directory
    2. .kautilya/mcp_servers/ directory
    3. mcp_servers/ directory

    Args:
        config_dir: Configuration directory
        verbose: Print detailed sync information
    """
    # Directories to scan for MCP YAML files
    search_paths = [
        Path("examples"),
        Path(config_dir) / "mcp_servers",
        Path("mcp_servers"),
    ]

    yaml_files: List[Path] = []

    # Find all YAML files
    for search_path in search_paths:
        if search_path.exists() and search_path.is_dir():
            yaml_files.extend(search_path.glob("*.yaml"))
            yaml_files.extend(search_path.glob("*.yml"))

    if not yaml_files:
        if verbose:
            console.print("[dim]No MCP YAML files found to sync[/dim]")
        return

    if verbose:
        console.print(f"[dim]Found {len(yaml_files)} MCP YAML file(s) to sync...[/dim]")

    # Try to connect to gateway
    try:
        with MCPGatewayClient() as client:
            # Test connection
            if not client.test_connection_sync():
                if verbose:
                    console.print("[yellow]MCP Gateway not reachable, skipping sync[/yellow]")
                return

            # Get currently registered servers
            registered_servers = client.list_servers_sync(enabled_only=False)
            registered_tool_ids = {
                s.get("registration", {}).get("tool_id"): s
                for s in registered_servers
            }

            synced_count = 0
            updated_count = 0
            skipped_count = 0

            # Process each YAML file
            for yaml_file in yaml_files:
                try:
                    # Load YAML
                    with open(yaml_file, "r") as f:
                        data = yaml.safe_load(f)

                    if not data or "tool_id" not in data:
                        if verbose:
                            console.print(
                                f"[yellow]Skipping {yaml_file.name}: No tool_id found[/yellow]"
                            )
                        skipped_count += 1
                        continue

                    tool_id = data["tool_id"]

                    # Create registration from YAML
                    registration = MCPServerRegistration(**data)

                    # Check if server needs update
                    if tool_id in registered_tool_ids:
                        # Server exists - check if it needs update
                        existing_reg = registered_tool_ids[tool_id]["registration"]

                        # Compare key fields
                        needs_update = (
                            existing_reg.get("name") != registration.name
                            or existing_reg.get("version") != registration.version
                            or existing_reg.get("endpoint") != registration.endpoint
                        )

                        if needs_update:
                            # Update: unregister old, register new
                            if verbose:
                                console.print(
                                    f"[cyan]Updating {tool_id} from {yaml_file.name}[/cyan]"
                                )

                            client.unregister_server_sync(tool_id)
                            client.register_server_sync(registration)
                            updated_count += 1
                        else:
                            if verbose:
                                console.print(
                                    f"[dim]{tool_id} up-to-date (skipping)[/dim]"
                                )
                            skipped_count += 1
                    else:
                        # New server - register
                        if verbose:
                            console.print(
                                f"[green]Registering {tool_id} from {yaml_file.name}[/green]"
                            )

                        client.register_server_sync(registration)
                        synced_count += 1

                except Exception as e:
                    if verbose:
                        console.print(
                            f"[red]Error processing {yaml_file.name}:[/red] {str(e)}"
                        )
                    skipped_count += 1
                    continue

            # Summary
            if verbose and (synced_count > 0 or updated_count > 0):
                console.print(
                    f"[green]✓ MCP Sync: {synced_count} new, {updated_count} updated, {skipped_count} skipped[/green]"
                )

    except Exception as e:
        if verbose:
            console.print(f"[yellow]MCP sync failed:[/yellow] {str(e)}")


def get_mcp_yaml_files(config_dir: str = ".kautilya") -> List[Path]:
    """
    Get list of MCP YAML files from standard locations.

    Args:
        config_dir: Configuration directory

    Returns:
        List of YAML file paths
    """
    search_paths = [
        Path("examples"),
        Path(config_dir) / "mcp_servers",
        Path("mcp_servers"),
    ]

    yaml_files: List[Path] = []

    for search_path in search_paths:
        if search_path.exists() and search_path.is_dir():
            yaml_files.extend(search_path.glob("*.yaml"))
            yaml_files.extend(search_path.glob("*.yml"))

    return yaml_files
