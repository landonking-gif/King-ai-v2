"""
Kautilya Main CLI Application.

Module: kautilya/cli.py
"""

from typing import Optional
import sys
import os
import click
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
# Search upward from this file's location to find project root .env
def _find_project_env() -> Optional[Path]:
    """Find .env file by searching upward from this file's location."""
    current = Path(__file__).resolve().parent

    # Search upward up to 10 levels
    for _ in range(10):
        env_path = current / ".env"
        if env_path.exists():
            return env_path

        parent = current.parent
        if parent == current:
            break  # Reached filesystem root
        current = parent

    return None

_env_path = _find_project_env()
if _env_path:
    load_dotenv(_env_path, override=False)
else:
    # Fallback: try default load_dotenv behavior (CWD)
    load_dotenv(override=False)

from .interactive import InteractiveMode
from .config import Config, load_config
from .gateway_manager import ensure_gateway_running
from .mcp_sync import sync_mcp_servers_from_yaml
from .iteration_display import DisplayMode, OutputMode, set_display_mode, set_output_mode
from .commands import (
    init,
    agent,
    skill,
    llm,
    mcp,
    manifest,
    runtime,
)

console = Console()


@click.group(invoke_without_command=True)
@click.option("--config-dir", default=".kautilya", help="Configuration directory")
@click.option("--version", is_flag=True, help="Show version and exit")
@click.option("--no-gateway", is_flag=True, help="Don't auto-start MCP Gateway")
@click.option(
    "-d", "--display",
    type=click.Choice(["minimal", "detailed"], case_sensitive=False),
    default=None,
    help="Display mode for iteration feedback (minimal=compact, detailed=rich panels)"
)
@click.option(
    "-o", "--output",
    type=click.Choice(["concise", "verbose"], case_sensitive=False),
    default=None,
    help="Output verbosity (concise=direct answers, verbose=include action summaries)"
)
@click.pass_context
def cli(ctx: click.Context, config_dir: str, version: bool, no_gateway: bool, display: Optional[str], output: Optional[str]) -> None:
    """
    Kautilya v1.0 - Enterprise Agentic Framework CLI.

    Interactive CLI for scaffolding agents, configuring LLMs, creating skills,
    and managing manifests.
    """
    if version:
        console.print("[bold green]Kautilya v1.0.0[/bold green]")
        sys.exit(0)

    # Set display mode if specified
    if display:
        set_display_mode(DisplayMode(display.lower()))

    # Set output mode if specified
    if output:
        set_output_mode(OutputMode(output.lower()))

    # Auto-start MCP Gateway (unless disabled)
    if not no_gateway:
        # Check if verbose mode is enabled
        verbose = os.getenv("KAUTILYA_VERBOSE_MODE", "false").lower() == "true"

        # Show a brief startup message
        if verbose:
            console.print("[dim]Checking MCP Gateway...[/dim]")

        # Ensure gateway is running
        gateway_started = ensure_gateway_running(verbose=verbose)

        if not gateway_started and verbose:
            console.print("[yellow]⚠ MCP Gateway could not be started automatically[/yellow]")
            console.print("[dim]MCP commands may not work. Start manually with:[/dim]")
            console.print("[dim]  cd mcp-gateway && ./start.sh[/dim]")
        elif gateway_started:
            # Gateway is running - auto-sync MCP servers from YAML files
            sync_mcp_servers_from_yaml(config_dir=config_dir, verbose=verbose)

    # Load configuration
    ctx.ensure_object(dict)
    ctx.obj["config_dir"] = config_dir
    ctx.obj["config"] = load_config(config_dir)

    # If no command provided, enter interactive mode
    if ctx.invoked_subcommand is None:
        interactive = InteractiveMode(config_dir, ctx.obj["config"])
        interactive.run()


# Register command groups
cli.add_command(init.init_cmd)
cli.add_command(agent.agent_cmd)
cli.add_command(skill.skill_cmd)
cli.add_command(llm.llm_cmd)
cli.add_command(mcp.mcp_cmd)
cli.add_command(manifest.manifest_cmd)
cli.add_command(runtime.run_cmd)
cli.add_command(runtime.status_cmd)
cli.add_command(runtime.logs_cmd)


def main() -> None:
    """Main entry point for kautilya."""
    try:
        cli(obj={})
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
