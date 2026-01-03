"""
Manifest Management Commands.

Module: kautilya/commands/manifest.py
"""

from typing import Optional, List, Dict, Any
from pathlib import Path
import click
import questionary
import yaml
import json
import jsonschema
import httpx
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

console = Console()


@click.group(name="manifest")
def manifest_cmd() -> None:
    """Manage workflow manifests."""
    pass


@manifest_cmd.command(name="new")
@click.pass_context
def manifest_new_cmd(ctx: click.Context) -> None:
    """Create new workflow manifest (guided)."""
    config_dir = ctx.obj.get("config_dir", ".kautilya")
    create_manifest(config_dir, interactive=True)


@manifest_cmd.command(name="validate")
@click.argument("file", required=False)
@click.pass_context
def manifest_validate_cmd(ctx: click.Context, file: Optional[str]) -> None:
    """Validate manifest against schema."""
    config_dir = ctx.obj.get("config_dir", ".kautilya")
    validate_manifest(file, config_dir)


@manifest_cmd.command(name="run")
@click.argument("file", required=False)
@click.pass_context
def manifest_run_cmd(ctx: click.Context, file: Optional[str]) -> None:
    """Execute workflow from manifest."""
    config_dir = ctx.obj.get("config_dir", ".kautilya")
    run_manifest(file, config_dir)


def create_manifest(config_dir: str, interactive: bool = True) -> None:
    """Create a new workflow manifest."""
    console.print("\n[bold cyan]Creating Workflow Manifest[/bold cyan]\n")

    if not interactive:
        console.print("[yellow]Interactive mode required for manifest creation[/yellow]")
        return

    # Get manifest name
    manifest_name = questionary.text(
        "Manifest name:", default="my-workflow"
    ).ask()

    # Get description
    description = questionary.text(
        "Description:", default="Workflow description"
    ).ask()

    # Build steps
    steps: List[Dict[str, Any]] = []
    console.print("[dim]Add workflow steps (press Enter to finish):[/dim]")

    step_id = 1
    while True:
        console.print(f"\n[bold]Step {step_id}:[/bold]")

        add_step = questionary.confirm(
            f"Add step {step_id}?", default=True
        ).ask()

        if not add_step:
            break

        role = questionary.select(
            "Role:",
            choices=["research", "verify", "code", "synthesis", "custom"],
        ).ask()

        # Get available agents
        agents_dir = Path.cwd() / "agents"
        agent_choices = ["<create-new>"]
        if agents_dir.exists():
            agent_choices = [d.name for d in agents_dir.iterdir() if d.is_dir()] + agent_choices

        agent = questionary.select(
            "Agent:", choices=agent_choices
        ).ask()

        if agent == "<create-new>":
            agent = f"{role}-agent"

        capabilities_input = questionary.text(
            "Capabilities (comma-separated):",
            default="web_search,summarize",
        ).ask()
        capabilities = [c.strip() for c in capabilities_input.split(",") if c.strip()]

        timeout = int(questionary.text("Timeout (seconds):", default="30").ask())

        step = {
            "id": f"step-{step_id}",
            "role": role,
            "agent": agent,
            "capabilities": capabilities,
            "inputs": [],
            "outputs": [f"{role}_result"],
            "timeout": timeout,
            "retry_on_failure": True,
        }

        steps.append(step)
        step_id += 1

    if not steps:
        console.print("[yellow]No steps added. Manifest not created.[/yellow]")
        return

    # Memory configuration
    memory_persist_on = questionary.checkbox(
        "Memory persistence:",
        choices=["on_complete", "per_step", "on_error"],
    ).ask() or []

    compaction_strategy = questionary.select(
        "Compaction strategy:",
        choices=["summarize", "truncate", "none"],
    ).ask()

    max_tokens = int(
        questionary.text("Max tokens in context:", default="8000").ask()
    )

    # Policy configuration
    requires_approval = questionary.confirm(
        "Requires human approval?", default=False
    ).ask()

    # Build manifest
    manifest = {
        "manifest_id": manifest_name.lower().replace(" ", "-"),
        "name": manifest_name,
        "version": "1.0.0",
        "description": description,
        "steps": steps,
        "memory": {
            "persist_on": memory_persist_on,
            "compaction": {
                "strategy": compaction_strategy,
                "max_tokens": max_tokens,
            },
        },
        "tools": {
            "catalog_ids": [],
        },
        "policies": {
            "requires_human_approval": requires_approval,
            "allowed_tool_categories": ["safe", "read_only"],
        },
        "metadata": {
            "created_by": "kautilya",
            "tags": [],
            "priority": "normal",
        },
    }

    # Save manifest
    manifests_dir = Path.cwd() / "manifests"
    manifests_dir.mkdir(exist_ok=True)

    manifest_file = manifests_dir / f"{manifest['manifest_id']}.yaml"

    with open(manifest_file, "w") as f:
        yaml.dump(manifest, f, default_flow_style=False, sort_keys=False)

    success_message = f"""
[green]✓[/green] Generated: {manifest_file}

[bold]Configuration:[/bold]
  Steps: {len(steps)}
  Memory: {compaction_strategy} compaction, {max_tokens} tokens
  Approval: {requires_approval}
    """

    console.print(Panel(success_message.strip(), title="[bold green]Manifest Created[/bold green]"))


def validate_manifest(manifest_file: Optional[str], config_dir: str) -> None:
    """Validate manifest against JSON schema."""
    console.print("\n[bold cyan]Validating Manifest[/bold cyan]\n")

    if not manifest_file:
        # Find manifest files
        manifests_dir = Path.cwd() / "manifests"
        if not manifests_dir.exists():
            console.print("[yellow]No manifests directory found.[/yellow]")
            return

        manifest_files = list(manifests_dir.glob("*.yaml"))
        if not manifest_files:
            console.print("[yellow]No manifest files found.[/yellow]")
            return

        manifest_file = str(manifest_files[0])

    manifest_path = Path(manifest_file)
    if not manifest_path.exists():
        console.print(f"[red]Manifest file not found: {manifest_file}[/red]")
        return

    # Load manifest
    with open(manifest_path, "r") as f:
        manifest = yaml.safe_load(f)

    # Load schema (from docs/schema_registry)
    schema_path = Path.cwd().parent / "docs" / "schema_registry" / "manifest.json"

    if not schema_path.exists():
        console.print("[yellow]Manifest schema not found, skipping validation[/yellow]")
        console.print(f"[green]✓[/green] Manifest loaded: {manifest_path}")
        return

    with open(schema_path, "r") as f:
        schema = json.load(f)

    # Validate
    try:
        jsonschema.validate(manifest, schema)
        console.print(f"[green]✓[/green] Manifest is valid: {manifest_path}")
    except jsonschema.ValidationError as e:
        console.print(f"[red]✗[/red] Validation failed: {e.message}")
        console.print(f"[dim]Path: {' -> '.join(str(p) for p in e.path)}[/dim]")


def run_manifest(manifest_file: Optional[str], config_dir: str) -> None:
    """Execute workflow from manifest."""
    console.print("\n[bold cyan]Running Workflow[/bold cyan]\n")

    if not manifest_file:
        # Find manifest files
        manifests_dir = Path.cwd() / "manifests"
        if not manifests_dir.exists():
            console.print("[yellow]No manifests directory found.[/yellow]")
            return

        manifest_files = list(manifests_dir.glob("*.yaml"))
        if not manifest_files:
            console.print("[yellow]No manifest files found.[/yellow]")
            return

        manifest_file = str(manifest_files[0])

    manifest_path = Path(manifest_file)
    if not manifest_path.exists():
        console.print(f"[red]Manifest file not found: {manifest_file}[/red]")
        return

    # Load manifest
    with open(manifest_path, "r") as f:
        manifest = yaml.safe_load(f)

    console.print(f"[dim]Executing manifest: {manifest.get('name', 'Unknown')}[/dim]")

    # TODO: Call orchestrator API to execute workflow
    orchestrator_url = "http://localhost:8000"

    try:
        async def run_workflow() -> None:
            async with httpx.AsyncClient(timeout=300.0) as client:
                response = await client.post(
                    f"{orchestrator_url}/workflows/start",
                    json={
                        "manifest": manifest,
                        "inputs": {},
                    },
                )

                if response.status_code == 200:
                    result = response.json()
                    console.print(f"[green]✓[/green] Workflow started: {result.get('workflow_id')}")
                else:
                    console.print(
                        f"[red]✗[/red] Failed to start workflow: {response.status_code}"
                    )

        import anyio
        anyio.run(run_workflow)

    except httpx.ConnectError:
        console.print(
            f"[yellow]Could not connect to orchestrator at {orchestrator_url}[/yellow]"
        )
        console.print("[dim]Make sure the orchestrator service is running[/dim]")
    except Exception as e:
        console.print(f"[red]Error:[/red] {str(e)}")


def create_manifest_programmatic(
    name: str,
    description: str = "",
    config_dir: str = ".kautilya",
) -> dict:
    """
    Programmatic interface for creating manifests.

    Args:
        name: Manifest name
        description: Manifest description
        config_dir: Configuration directory

    Returns:
        Manifest details dictionary
    """
    manifests_dir = Path.cwd() / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    manifest_file = manifests_dir / f"{name}.yaml"

    manifest = {
        "manifest_id": name,
        "name": name,
        "version": "1.0.0",
        "description": description or f"Workflow: {name}",
        "steps": [
            {
                "id": "step_1",
                "role": "research",
                "capabilities": ["web_search", "summarize"],
                "inputs": ["query"],
                "outputs": ["research_snippet"],
                "timeout": 60,
            },
        ],
        "memory": {
            "persist_on": ["complete"],
            "compaction": {"strategy": "summarize", "max_tokens": 8000},
        },
        "tools": {"catalog_ids": ["filesystem"]},
        "policies": {"requires_human_approval": False},
    }

    with open(manifest_file, "w") as f:
        yaml.dump(manifest, f, default_flow_style=False, sort_keys=False)

    return {
        "manifest_path": str(manifest_file),
        "name": name,
        "description": description,
    }


def validate_manifest_programmatic(
    manifest_file: str,
    config_dir: str = ".kautilya",
) -> dict:
    """
    Programmatic interface for validating manifests.

    Args:
        manifest_file: Path to manifest file
        config_dir: Configuration directory

    Returns:
        Validation result dictionary
    """
    manifest_path = Path(manifest_file)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_file}")

    with open(manifest_path, "r") as f:
        manifest = yaml.safe_load(f)

    warnings = []

    # Basic validation
    if "manifest_id" not in manifest:
        warnings.append("Missing manifest_id")
    if "steps" not in manifest or not manifest["steps"]:
        warnings.append("No steps defined")

    return {
        "valid": len(warnings) == 0,
        "manifest_file": manifest_file,
        "warnings": warnings,
    }
