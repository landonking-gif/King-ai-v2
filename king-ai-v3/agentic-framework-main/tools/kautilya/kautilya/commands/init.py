"""
Project Initialization Command.

Module: kautilya/commands/init.py
"""

from typing import Optional
from pathlib import Path
import click
import questionary
from rich.console import Console
from rich.panel import Panel

from ..config import Config, ProjectConfig, save_config

console = Console()


@click.command(name="init")
@click.option("--name", help="Project name")
@click.option("--provider", help="LLM provider (anthropic/openai/azure/local)")
@click.option("--mcp/--no-mcp", default=True, help="Enable MCP integration")
@click.pass_context
def init_cmd(
    ctx: click.Context, name: Optional[str], provider: Optional[str], mcp: bool
) -> None:
    """Initialize new agent project with guided setup."""
    config_dir = ctx.obj.get("config_dir", ".kautilya")
    initialize_project(config_dir, name, provider, mcp, interactive=False)


def initialize_project(
    config_dir: str,
    project_name: Optional[str] = None,
    provider: Optional[str] = None,
    enable_mcp: bool = True,
    interactive: bool = True,
) -> None:
    """
    Initialize a new agent project.

    Args:
        config_dir: Configuration directory
        project_name: Project name (will prompt if not provided)
        provider: LLM provider (will prompt if not provided)
        enable_mcp: Enable MCP integration
        interactive: Whether to use interactive prompts
    """
    console.print("\n[bold cyan]Initializing new agent project...[/bold cyan]\n")

    # Get project name
    if not project_name and interactive:
        project_name = questionary.text(
            "Project name:", default="my-agent"
        ).ask()
    elif not project_name:
        project_name = "my-agent"

    # Get LLM provider
    if not provider and interactive:
        provider = questionary.select(
            "LLM provider:",
            choices=["anthropic", "openai", "azure", "local"],
        ).ask()
    elif not provider:
        provider = "anthropic"

    # Confirm MCP integration
    if interactive:
        enable_mcp = questionary.confirm(
            "Enable MCP integration?", default=True
        ).ask()

    # Create project structure
    project_path = Path.cwd() / project_name
    project_path.mkdir(exist_ok=True)

    # Create directories
    directories = [
        "agents",
        "skills",
        "manifests",
        "schemas",
        "tests",
        config_dir,
    ]

    for directory in directories:
        (project_path / directory).mkdir(exist_ok=True)

    # Create configuration
    config = Config(
        project=ProjectConfig(name=project_name, version="1.0.0"),
        default_provider=provider,
    )

    # Save configuration
    save_config(config, str(project_path / config_dir))

    # Create default files
    _create_default_files(project_path, project_name, enable_mcp)

    # Show success message
    success_message = f"""
[green]✓[/green] Created agent project structure
[green]✓[/green] Generated default configuration
[green]✓[/green] Configured {provider} as LLM provider

[bold]Next steps:[/bold]
  1. cd {project_name}
  2. Configure API keys: export {_get_api_key_env(provider)}=your-key
  3. Run: kautilya /run
    """

    console.print(Panel(success_message.strip(), title="[bold green]Success[/bold green]"))


def _create_default_files(project_path: Path, project_name: str, enable_mcp: bool) -> None:
    """Create default project files."""
    # Create README
    readme_content = f"""# {project_name}

Enterprise agent project built with Agentic Framework.

## Setup

1. Install dependencies:
   ```bash
   uv venv --python 3.11
   source .venv/bin/activate
   uv pip install -r requirements.txt
   ```

2. Configure environment variables:
   ```bash
   export ANTHROPIC_API_KEY=your-key
   export REDIS_URL=redis://localhost:6379
   export POSTGRES_URL=postgresql://user:password@localhost:5432/agentic_framework
   ```

3. Run the project:
   ```bash
   kautilya /run
   ```

## Project Structure

- `agents/` - Subagent configurations
- `skills/` - Custom skills
- `manifests/` - Workflow manifests
- `schemas/` - Custom artifact schemas
- `tests/` - Test suite

## Documentation

See [Agentic Framework Documentation](https://github.com/yourorg/agent-framework) for details.
"""

    (project_path / "README.md").write_text(readme_content)

    # Create requirements.txt
    requirements = """fastapi>=0.109.0
pydantic>=2.5.0
pydantic-settings>=2.1.0
redis>=5.0.0
psycopg2-binary>=2.9.9
httpx>=0.26.0
anyio>=4.2.0
"""

    (project_path / "requirements.txt").write_text(requirements)

    # Create .gitignore
    gitignore = """.kautilya/
.venv/
__pycache__/
*.pyc
.env
.DS_Store
*.log
"""

    (project_path / ".gitignore").write_text(gitignore)


def _get_api_key_env(provider: str) -> str:
    """Get API key environment variable name for provider."""
    mapping = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "azure": "AZURE_OPENAI_KEY",
        "local": "# No API key needed for local models",
    }
    return mapping.get(provider, "API_KEY")


def initialize_project_programmatic(
    name: str,
    provider: str = "openai",
    enable_mcp: bool = True,
    config_dir: str = ".kautilya",
) -> dict:
    """
    Programmatic interface for project initialization.

    Args:
        name: Project name
        provider: LLM provider
        enable_mcp: Enable MCP integration
        config_dir: Configuration directory

    Returns:
        Dictionary with project details
    """
    from pathlib import Path
    from ..config import Config, ProjectConfig, save_config

    project_path = Path.cwd() / name
    project_path.mkdir(exist_ok=True)

    # Create directories
    directories = ["agents", "skills", "manifests", "schemas", "tests", config_dir]
    files_created = []

    for directory in directories:
        dir_path = project_path / directory
        dir_path.mkdir(exist_ok=True)
        files_created.append(str(dir_path))

    # Create configuration
    config = Config(
        project=ProjectConfig(name=name, version="1.0.0"),
        default_provider=provider,
    )

    # Save configuration
    save_config(config, str(project_path / config_dir))
    files_created.append(str(project_path / config_dir / "config.yaml"))

    # Create default files
    _create_default_files(project_path, name, enable_mcp)
    files_created.extend([
        str(project_path / "README.md"),
        str(project_path / "requirements.txt"),
        str(project_path / ".gitignore"),
    ])

    return {
        "project_path": str(project_path),
        "provider": provider,
        "mcp_enabled": enable_mcp,
        "files_created": files_created,
    }
