"""
Skill Management Commands.

Module: kautilya/commands/skill.py

Supports both native format (skill.yaml + schema.json) and Anthropic format (SKILL.md).
"""

from typing import Optional, List, Dict, Any
from pathlib import Path
import sys
import click
import questionary
import yaml
import json
import urllib.request
import tempfile
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Import skill parser utilities
# Handle importing from code-exec directory (has hyphen, not valid Python package name)
import importlib.util

_project_root = Path(__file__).parents[4]
_skill_parser_path = _project_root / "code-exec" / "service" / "skill_parser.py"
_models_path = _project_root / "code-exec" / "service" / "models.py"

# Load models.py first (no dependencies)
if _models_path.exists():
    spec = importlib.util.spec_from_file_location("models", _models_path)
    models_module = importlib.util.module_from_spec(spec)
    sys.modules["models"] = models_module
    spec.loader.exec_module(models_module)

# Load skill_parser.py (depends on models)
if _skill_parser_path.exists():
    spec = importlib.util.spec_from_file_location("skill_parser", _skill_parser_path)
    skill_parser_module = importlib.util.module_from_spec(spec)
    sys.modules["skill_parser"] = skill_parser_module
    spec.loader.exec_module(skill_parser_module)

    SkillParser = skill_parser_module.SkillParser
    SkillConverter = skill_parser_module.SkillConverter
    FormatDetector = skill_parser_module.FormatDetector
    SkillPackager = skill_parser_module.SkillPackager
    AnthropicSkillMetadata = skill_parser_module.AnthropicSkillMetadata
else:
    raise RuntimeError(f"skill_parser.py not found at {_skill_parser_path}")

# Load skill_registry.py
_skill_registry_path = _project_root / "code-exec" / "service" / "skill_registry.py"
if _skill_registry_path.exists():
    spec = importlib.util.spec_from_file_location("skill_registry", _skill_registry_path)
    skill_registry_module = importlib.util.module_from_spec(spec)
    sys.modules["skill_registry"] = skill_registry_module
    spec.loader.exec_module(skill_registry_module)

    SkillRegistry = skill_registry_module.SkillRegistry
    SkillMetadata = skill_registry_module.SkillMetadata
else:
    SkillRegistry = None
    SkillMetadata = None

console = Console()


@click.group(name="skill")
def skill_cmd() -> None:
    """Manage skills."""
    pass


@skill_cmd.command(name="new")
@click.argument("name")
@click.option("--description", help="Skill description")
@click.option(
    "--format",
    "skill_format",
    type=click.Choice(["anthropic", "native", "hybrid"]),
    default="native",
    help="Skill format (anthropic=SKILL.md, native=skill.yaml+schema.json, hybrid=both)",
)
@click.pass_context
def skill_new_cmd(
    ctx: click.Context,
    name: str,
    description: Optional[str],
    skill_format: str,
) -> None:
    """Scaffold new skill with I/O schema."""
    config_dir = ctx.obj.get("config_dir", ".kautilya")
    create_skill(name, config_dir, description, skill_format, interactive=False)


def create_skill(
    name: str,
    config_dir: str,
    description: Optional[str] = None,
    skill_format: str = "native",
    interactive: bool = True,
) -> None:
    """
    Create a new skill.

    Args:
        name: Skill name
        config_dir: Configuration directory
        description: Skill description
        skill_format: Skill format (anthropic, native, or hybrid)
        interactive: Use interactive prompts
    """
    console.print(f"\n[bold cyan]Creating skill: {name}[/bold cyan]\n")

    # Ask for format if interactive
    if interactive and not skill_format:
        skill_format = questionary.select(
            "Skill format:",
            choices=[
                questionary.Choice("Hybrid (both formats, recommended)", value="hybrid"),
                questionary.Choice("Native (skill.yaml + schema.json)", value="native"),
                questionary.Choice("Anthropic (SKILL.md only)", value="anthropic"),
            ],
        ).ask()

    # Get description
    if not description and interactive:
        description = questionary.text(
            "Skill description:",
            default=f"Execute {name} operation",
        ).ask()
    elif not description:
        description = f"Execute {name} operation"

    # Get input schema fields
    input_fields: List[Dict[str, Any]] = []
    if interactive:
        console.print("[dim]Define input schema (press Enter with empty name to finish):[/dim]")
        while True:
            field_name = questionary.text("Field name (or press Enter to finish):").ask()
            if not field_name:
                break

            field_type = questionary.select(
                f"Type for '{field_name}':",
                choices=["string", "number", "boolean", "array", "object"],
            ).ask()

            required = questionary.confirm(f"Is '{field_name}' required?", default=True).ask()

            input_fields.append({
                "name": field_name,
                "type": field_type,
                "required": required,
            })
    else:
        # Default input field
        input_fields = [{"name": "input", "type": "string", "required": True}]

    # Get output schema fields
    output_fields: List[Dict[str, Any]] = []
    if interactive:
        console.print("[dim]Define output schema (press Enter with empty name to finish):[/dim]")
        while True:
            field_name = questionary.text("Field name (or press Enter to finish):").ask()
            if not field_name:
                break

            field_type = questionary.select(
                f"Type for '{field_name}':",
                choices=["string", "number", "boolean", "array", "object"],
            ).ask()

            output_fields.append({
                "name": field_name,
                "type": field_type,
            })
    else:
        # Default output field
        output_fields = [{"name": "result", "type": "string"}]

    # Get safety flags
    safety_flags: List[str] = []
    if interactive:
        safety_flags = questionary.checkbox(
            "Select safety flags:",
            choices=["pii_risk", "external_call", "side_effect", "file_system", "network_access"],
        ).ask() or []

    # Check if requires approval
    requires_approval = False
    if interactive and safety_flags:
        requires_approval = questionary.confirm(
            "Requires policy approval?", default=False
        ).ask()

    # Create skill directory
    skill_dir = Path.cwd() / "skills" / name.replace("-", "_")
    skill_dir.mkdir(parents=True, exist_ok=True)

    # Create skill.yaml
    skill_config = {
        "name": name,
        "version": "1.0.0",
        "description": description,
        "safety_flags": safety_flags,
        "requires_approval": requires_approval,
        "input_schema": "./schema.json#/input",
        "output_schema": "./schema.json#/output",
        "handler": f"handler.{name.replace('-', '_')}",
    }

    with open(skill_dir / "skill.yaml", "w") as f:
        yaml.dump(skill_config, f, default_flow_style=False)

    # Create schema.json
    schema = _create_json_schema(input_fields, output_fields)
    with open(skill_dir / "schema.json", "w") as f:
        json.dump(schema, f, indent=2)

    # Create handler.py
    handler_code = _generate_handler_code(name, input_fields, output_fields)
    (skill_dir / "handler.py").write_text(handler_code)

    # Create test file
    test_code = _generate_test_code(name, input_fields, output_fields)
    (skill_dir / "test_handler.py").write_text(test_code)

    # Show success
    success_message = f"""
[green]✓[/green] Generated skill: skills/{name.replace("-", "_")}/
  ├── skill.yaml       # Registration metadata
  ├── schema.json      # I/O JSON Schema
  ├── handler.py       # Implementation stub
  └── test_handler.py  # Test template

[bold]Next steps:[/bold]
  1. Implement logic in handler.py
  2. Add tests in test_handler.py
  3. Run tests: pytest skills/{name.replace("-", "_")}/test_handler.py
    """

    console.print(Panel(success_message.strip(), title="[bold green]Skill Created[/bold green]"))


def _create_json_schema(
    input_fields: List[Dict[str, Any]], output_fields: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Create JSON Schema from field definitions."""
    def create_properties(fields: List[Dict[str, Any]]) -> Dict[str, Any]:
        properties = {}
        for field in fields:
            field_schema = {"type": field["type"]}
            if field["type"] == "array":
                field_schema["items"] = {"type": "string"}
            properties[field["name"]] = field_schema
        return properties

    input_required = [f["name"] for f in input_fields if f.get("required", False)]

    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "input": {
            "type": "object",
            "properties": create_properties(input_fields),
            "required": input_required,
        },
        "output": {
            "type": "object",
            "properties": create_properties(output_fields),
            "required": [f["name"] for f in output_fields],
        },
    }


def _generate_handler_code(
    name: str, input_fields: List[Dict[str, Any]], output_fields: List[Dict[str, Any]]
) -> str:
    """Generate handler implementation stub."""
    function_name = name.replace("-", "_")
    input_params = ", ".join(f"{f['name']}: {_python_type(f['type'])}" for f in input_fields)
    output_type = "Dict[str, Any]"

    return f'''"""
{name.replace("-", " ").title()} Skill Handler.

Module: skills/{name.replace("-", "_")}/handler.py
"""

from typing import Dict, Any, List, Optional


def {function_name}({input_params}) -> {output_type}:
    """
    Execute {name} operation.

    Args:
{chr(10).join(f"        {f['name']}: {f.get('description', f['name'])} ({f['type']})" for f in input_fields)}

    Returns:
        Result dictionary with:
{chr(10).join(f"        - {f['name']}: {f.get('description', f['name'])} ({f['type']})" for f in output_fields)}
    """
    # TODO: Implement skill logic here

    # Example implementation (replace with actual logic)
    result = {{
{chr(10).join(f'        "{f["name"]}": None,  # TODO: Compute {f["name"]}' for f in output_fields)}
    }}

    return result
'''


def _generate_test_code(
    name: str, input_fields: List[Dict[str, Any]], output_fields: List[Dict[str, Any]]
) -> str:
    """Generate test template."""
    function_name = name.replace("-", "_")

    return f'''"""
Tests for {name.replace("-", " ").title()} Skill.

Module: skills/{name.replace("-", "_")}/test_handler.py
"""

import pytest
from .handler import {function_name}


class Test{function_name.title().replace("_", "")}:
    """Test suite for {function_name} skill."""

    def test_{function_name}_basic(self) -> None:
        """Test basic {function_name} execution."""
        # TODO: Add test data
        result = {function_name}(
{chr(10).join(f'            {f["name"]}={_example_value(f["type"])},' for f in input_fields)}
        )

        # TODO: Add assertions
{chr(10).join(f'        assert "{f["name"]}" in result' for f in output_fields)}

    def test_{function_name}_edge_cases(self) -> None:
        """Test edge cases for {function_name}."""
        # TODO: Add edge case tests
        pass

    def test_{function_name}_error_handling(self) -> None:
        """Test error handling for {function_name}."""
        # TODO: Add error handling tests
        pass
'''


def _python_type(json_type: str) -> str:
    """Convert JSON type to Python type hint."""
    mapping = {
        "string": "str",
        "number": "float",
        "boolean": "bool",
        "array": "List[Any]",
        "object": "Dict[str, Any]",
    }
    return mapping.get(json_type, "Any")


def _example_value(json_type: str) -> str:
    """Get example value for JSON type."""
    mapping = {
        "string": '"example"',
        "number": "42.0",
        "boolean": "True",
        "array": "[]",
        "object": "{}",
    }
    return mapping.get(json_type, "None")


# New commands for dual-format support


@skill_cmd.command(name="import")
@click.argument("source")
@click.option("--format", "import_format", type=click.Choice(["anthropic", "native", "hybrid"]), default="hybrid")
@click.option("--safety-flags", multiple=True, help="Safety flags to assign")
@click.option("--requires-approval", is_flag=True, help="Requires human approval")
def skill_import_cmd(source: str, import_format: str, safety_flags: tuple, requires_approval: bool) -> None:
    """Import skill from URL or ZIP file."""
    console.print(f"\n[bold cyan]Importing skill from: {source}[/bold cyan]\n")

    # Download or load ZIP
    try:
        if source.startswith("http://") or source.startswith("https://"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
                console.print(f"[dim]Downloading from {source}...[/dim]")
                urllib.request.urlretrieve(source, tmp.name)
                zip_path = Path(tmp.name)
        else:
            zip_path = Path(source)

        if not zip_path.exists():
            console.print(f"[red]Error: File not found: {zip_path}[/red]")
            return

        # Extract to temp directory
        with tempfile.TemporaryDirectory() as tmpdir:
            skill_dir = SkillPackager.unpack_skill_zip(zip_path, Path(tmpdir))

            # Detect format
            format_info = FormatDetector.detect_format(skill_dir)

            # Import based on format
            target_dir = Path.cwd() / "skills" / skill_dir.name

            if format_info.is_anthropic_only and import_format in ("native", "hybrid"):
                # Convert Anthropic to native
                console.print("[dim]Converting Anthropic format to native...[/dim]")
                safety_flag_list = [getattr(sys.modules["skill_parser"], f.upper()) for f in safety_flags] if safety_flags else None
                SkillConverter.anthropic_to_native(
                    skill_dir, add_safety_flags=safety_flag_list, requires_approval=requires_approval
                )

            # Copy to skills directory
            import shutil
            if target_dir.exists():
                overwrite = questionary.confirm(
                    f"Skill {skill_dir.name} already exists. Overwrite?", default=False
                ).ask()
                if not overwrite:
                    console.print("[yellow]Import cancelled[/yellow]")
                    return
                shutil.rmtree(target_dir)

            shutil.copytree(skill_dir, target_dir)

        console.print(Panel(
            f"[green]✓[/green] Imported skill: {target_dir}\n\nFormat: {import_format}",
            title="[bold green]Skill Imported[/bold green]"
        ))

    except Exception as e:
        console.print(f"[red]Error importing skill: {e}[/red]")


@skill_cmd.command(name="export")
@click.argument("name")
@click.option("--format", "export_format", type=click.Choice(["anthropic", "native"]), default="anthropic")
@click.option("--output", help="Output ZIP path")
def skill_export_cmd(name: str, export_format: str, output: Optional[str]) -> None:
    """Export skill to ZIP for sharing."""
    console.print(f"\n[bold cyan]Exporting skill: {name}[/bold cyan]\n")

    # Use SkillRegistry to find skill by name
    if SkillRegistry is not None:
        registry = SkillRegistry()
        skill_meta = registry.get_skill(name)
        if skill_meta:
            skill_dir = skill_meta.path
        else:
            skill_dir = Path.cwd() / "skills" / name.replace("-", "_")
    else:
        skill_dir = Path.cwd() / "skills" / name.replace("-", "_")

    if not skill_dir.exists():
        console.print(f"[red]Error: Skill not found: {name}[/red]")
        return

    # Convert to Anthropic format if needed
    if export_format == "anthropic":
        format_info = FormatDetector.detect_format(skill_dir)
        if not format_info.has_skill_md:
            console.print("[dim]Generating SKILL.md...[/dim]")
            SkillConverter.native_to_anthropic(skill_dir, include_handler=True)

    # Package as ZIP
    output_path = Path(output) if output else None
    zip_path = SkillPackager.package_skill_zip(skill_dir, output_path)

    console.print(Panel(
        f"[green]✓[/green] Exported to: {zip_path}\n\nReady for Anthropic marketplace",
        title="[bold green]Skill Exported[/bold green]"
    ))


@skill_cmd.command(name="convert")
@click.argument("name")
@click.option("--to", "to_format", type=click.Choice(["anthropic", "native"]), required=True)
@click.option("--include-handler", is_flag=True, default=True, help="Include handler.py as resource")
def skill_convert_cmd(name: str, to_format: str, include_handler: bool) -> None:
    """Convert skill between formats."""
    console.print(f"\n[bold cyan]Converting skill: {name}[/bold cyan]\n")

    # Use SkillRegistry to find skill by name
    if SkillRegistry is not None:
        registry = SkillRegistry()
        skill_meta = registry.get_skill(name)
        if skill_meta:
            skill_dir = skill_meta.path
        else:
            skill_dir = Path.cwd() / "skills" / name.replace("-", "_")
    else:
        skill_dir = Path.cwd() / "skills" / name.replace("-", "_")

    if not skill_dir.exists():
        console.print(f"[red]Error: Skill not found: {name}[/red]")
        return

    format_info = FormatDetector.detect_format(skill_dir)

    if to_format == "anthropic":
        if format_info.has_skill_md:
            console.print("[yellow]Skill already has SKILL.md format[/yellow]")
            return
        SkillConverter.native_to_anthropic(skill_dir, include_handler=include_handler)
        console.print(Panel(
            f"[green]✓[/green] Added Anthropic format (SKILL.md)\n\nSkill is now hybrid format",
            title="[bold green]Conversion Complete[/bold green]"
        ))
    else:  # to native
        if not format_info.has_skill_md:
            console.print("[red]Error: No SKILL.md found to convert[/red]")
            return
        if format_info.has_skill_yaml:
            console.print("[yellow]Skill already has native format[/yellow]")
            return
        SkillConverter.anthropic_to_native(skill_dir)
        console.print(Panel(
            f"[green]✓[/green] Added native format (skill.yaml + schema.json)\n\n"
            "[yellow]Note:[/yellow] Handler implementation required in handler.py",
            title="[bold green]Conversion Complete[/bold green]"
        ))


@skill_cmd.command(name="list")
@click.option("--format", "format_filter", type=click.Choice(["all", "hybrid", "anthropic", "native"]), default="all", help="Filter by format")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed information")
def skill_list_cmd(format_filter: str, verbose: bool) -> None:
    """List all skills from all configured paths."""
    if SkillRegistry is None:
        console.print("[red]Error: SkillRegistry not available[/red]")
        return

    # Create registry and discover skills
    registry = SkillRegistry()
    skills = registry.discover_all()

    if not skills:
        console.print("[yellow]No skills found in any configured path[/yellow]")
        console.print(f"[dim]Searched paths:[/dim]")
        for p in registry.skill_paths:
            console.print(f"  [dim]- {p}[/dim]")
        return

    # Filter by format if specified
    if format_filter != "all":
        skills = registry.list_skills_by_format(format_filter)
        if not skills:
            console.print(f"[yellow]No {format_filter} format skills found[/yellow]")
            return

    # Create table
    table = Table(title=f"Available Skills ({len(skills)} found)")
    table.add_column("Name", style="cyan")
    table.add_column("Format", style="magenta")
    table.add_column("Version", style="green")
    table.add_column("Location", style="dim")
    table.add_column("Description")

    if verbose:
        table.add_column("Handler", style="yellow")
        table.add_column("Resources", style="blue")

    for skill in sorted(skills, key=lambda s: s.name):
        row = [
            skill.name,
            skill.format_label,
            skill.version,
            skill.source_path.name,  # Show which path it came from
            skill.description[:45] + "..." if len(skill.description) > 45 else skill.description,
        ]

        if verbose:
            handler = f"{skill.handler_function}" if skill.handler_function else "-"
            resources = []
            if skill.resources:
                if skill.resources.scripts:
                    resources.append(f"{len(skill.resources.scripts)}S")
                if skill.resources.references:
                    resources.append(f"{len(skill.resources.references)}R")
                if skill.resources.assets:
                    resources.append(f"{len(skill.resources.assets)}A")
            row.append(handler)
            row.append(" ".join(resources) if resources else "-")

        table.add_row(*row)

    console.print(table)

    # Show skill paths
    console.print()
    console.print("[dim]Skill paths searched:[/dim]")
    for p in registry.skill_paths:
        skill_count = len([s for s in skills if s.source_path == p])
        console.print(f"  [dim]- {p} ({skill_count} skills)[/dim]")


@skill_cmd.command(name="validate")
@click.argument("name")
def skill_validate_cmd(name: str) -> None:
    """Validate skill format and schemas."""
    console.print(f"\n[bold cyan]Validating skill: {name}[/bold cyan]\n")

    # Use SkillRegistry to find skill by name
    if SkillRegistry is not None:
        registry = SkillRegistry()
        skill_meta = registry.get_skill(name)
        if skill_meta:
            skill_dir = skill_meta.path
        else:
            # Fallback to local path
            skill_dir = Path.cwd() / "skills" / name.replace("-", "_")
    else:
        skill_dir = Path.cwd() / "skills" / name.replace("-", "_")

    if not skill_dir.exists():
        console.print(f"[red]Error: Skill not found: {name}[/red]")
        if SkillRegistry is not None:
            console.print("[dim]Searched paths:[/dim]")
            for p in registry.skill_paths:
                console.print(f"  [dim]- {p}[/dim]")
        return

    errors = []
    warnings = []

    # Detect format
    format_info = FormatDetector.detect_format(skill_dir)

    if not (format_info.has_skill_md or format_info.has_skill_yaml):
        errors.append("No valid skill format found (missing both SKILL.md and skill.yaml)")

    # Validate SKILL.md if present
    if format_info.has_skill_md:
        try:
            skill_md_path = skill_dir / "SKILL.md"
            metadata, markdown = SkillParser.parse_skill_md(skill_md_path)
            console.print("[green]✓[/green] SKILL.md format is valid")
        except Exception as e:
            errors.append(f"SKILL.md validation failed: {e}")

    # Validate native format if present
    if format_info.has_skill_yaml and format_info.has_schema_json:
        try:
            from jsonschema import Draft7Validator, ValidationError

            skill_yaml_path = skill_dir / "skill.yaml"
            schema_json_path = skill_dir / "schema.json"

            with open(skill_yaml_path, "r") as f:
                skill_config = yaml.safe_load(f)

            with open(schema_json_path, "r") as f:
                schemas = json.load(f)

            # Validate schemas
            Draft7Validator.check_schema(schemas.get("input", {}))
            Draft7Validator.check_schema(schemas.get("output", {}))

            console.print("[green]✓[/green] Native format (skill.yaml + schema.json) is valid")

            # Check handler
            if not format_info.has_handler_py:
                warnings.append("handler.py not found")

        except Exception as e:
            errors.append(f"Native format validation failed: {e}")

    # Show results
    if errors:
        console.print("\n[red]Errors:[/red]")
        for err in errors:
            console.print(f"  [red]✗[/red] {err}")

    if warnings:
        console.print("\n[yellow]Warnings:[/yellow]")
        for warn in warnings:
            console.print(f"  [yellow]![/yellow] {warn}")

    if not errors and not warnings:
        console.print(Panel(
            "[green]✓[/green] All validations passed",
            title="[bold green]Skill Valid[/bold green]"
        ))
    elif not errors:
        console.print("\n[green]Validation passed with warnings[/green]")
    else:
        console.print("\n[red]Validation failed[/red]")


@skill_cmd.command(name="package")
@click.argument("name")
@click.option("--output", help="Output ZIP path")
def skill_package_cmd(name: str, output: Optional[str]) -> None:
    """Package skill as ZIP for distribution."""
    console.print(f"\n[bold cyan]Packaging skill: {name}[/bold cyan]\n")

    # Use SkillRegistry to find skill by name
    if SkillRegistry is not None:
        registry = SkillRegistry()
        skill_meta = registry.get_skill(name)
        if skill_meta:
            skill_dir = skill_meta.path
        else:
            skill_dir = Path.cwd() / "skills" / name.replace("-", "_")
    else:
        skill_dir = Path.cwd() / "skills" / name.replace("-", "_")

    if not skill_dir.exists():
        console.print(f"[red]Error: Skill not found: {name}[/red]")
        return

    output_path = Path(output) if output else None
    zip_path = SkillPackager.package_skill_zip(skill_dir, output_path)

    console.print(Panel(
        f"[green]✓[/green] Packaged to: {zip_path}",
        title="[bold green]Skill Packaged[/bold green]"
    ))


def create_skill_programmatic(
    name: str,
    description: str = "",
    input_fields: Optional[List[str]] = None,
    output_fields: Optional[List[str]] = None,
    safety_flags: Optional[List[str]] = None,
    config_dir: str = ".kautilya",
) -> dict:
    """
    Programmatic interface for creating skills.

    Args:
        name: Skill name
        description: Skill description
        input_fields: Input field names
        output_fields: Output field names
        safety_flags: Safety flags
        config_dir: Configuration directory

    Returns:
        Dictionary with skill details
    """
    input_fields = input_fields or ["input"]
    output_fields = output_fields or ["result"]
    safety_flags = safety_flags or []

    # Create skill directory
    skill_name_underscore = name.replace("-", "_")
    skill_dir = Path.cwd() / "skills" / skill_name_underscore
    skill_dir.mkdir(parents=True, exist_ok=True)

    # Build schema
    input_schema = {
        "type": "object",
        "properties": {field: {"type": "string", "description": f"{field}"} for field in input_fields},
        "required": input_fields[:1],  # First field is required
    }

    output_schema = {
        "type": "object",
        "properties": {field: {"type": "string", "description": f"{field}"} for field in output_fields},
        "required": output_fields[:1],
    }

    schema_doc = {
        "input": input_schema,
        "output": output_schema,
    }

    # Create skill.yaml
    skill_config = {
        "name": name,
        "version": "1.0.0",
        "description": description or f"Skill: {name}",
        "safety_flags": safety_flags,
        "requires_approval": len(safety_flags) > 0,
        "input_schema": "./schema.json#/input",
        "output_schema": "./schema.json#/output",
        "handler": f"handler.{skill_name_underscore}",
    }

    files_created = []

    with open(skill_dir / "skill.yaml", "w") as f:
        yaml.dump(skill_config, f, default_flow_style=False)
    files_created.append(str(skill_dir / "skill.yaml"))

    with open(skill_dir / "schema.json", "w") as f:
        json.dump(schema_doc, f, indent=2)
    files_created.append(str(skill_dir / "schema.json"))

    # Create handler
    handler_content = f'''"""
{name.replace("-", " ").title()} Skill Handler.

Module: skills/{skill_name_underscore}/handler.py
"""

from typing import Dict, Any, List, Optional


def {skill_name_underscore}({", ".join(f"{field}: str" for field in input_fields)}) -> Dict[str, Any]:
    """
    Execute {name} operation.

    Args:
{chr(10).join(f"        {field}: {field} (string)" for field in input_fields)}

    Returns:
        Result dictionary with:
{chr(10).join(f"        - {field}: {field} (string)" for field in output_fields)}
    """
    # TODO: Implement skill logic here

    # Example implementation (replace with actual logic)
    result = {{
{chr(10).join(f'        "{field}": None,  # TODO: Compute {field}' for field in output_fields)}
    }}

    return result
'''

    with open(skill_dir / "handler.py", "w") as f:
        f.write(handler_content)
    files_created.append(str(skill_dir / "handler.py"))

    return {
        "skill_path": str(skill_dir),
        "name": name,
        "description": description,
        "input_fields": input_fields,
        "output_fields": output_fields,
        "safety_flags": safety_flags,
        "files_created": files_created,
    }


def list_skills_programmatic(config_dir: str = ".kautilya") -> List[dict]:
    """
    Programmatic interface for listing skills.

    Uses SkillRegistry to discover skills from all configured paths.

    Args:
        config_dir: Configuration directory

    Returns:
        List of skill dictionaries
    """
    # Use SkillRegistry if available
    if SkillRegistry is not None:
        registry = SkillRegistry()
        discovered_skills = registry.discover_all()
        return [
            {
                "name": skill.name,
                "version": skill.version,
                "description": skill.description,
                "format": skill.format_label,
                "path": str(skill.path),
                "source_path": str(skill.source_path),
                "handler": skill.handler_function,
                "safety_flags": skill.safety_flags,
                "requires_approval": skill.requires_approval,
            }
            for skill in discovered_skills
        ]

    # Fallback to legacy behavior
    skills_dir = Path.cwd() / "skills"
    skills = []

    if not skills_dir.exists():
        return skills

    for skill_dir in skills_dir.iterdir():
        if skill_dir.is_dir():
            skill_yaml = skill_dir / "skill.yaml"
            skill_md = skill_dir / "SKILL.md"

            if skill_yaml.exists():
                with open(skill_yaml) as f:
                    config = yaml.safe_load(f)
                    skills.append({
                        "name": config.get("name", skill_dir.name),
                        "version": config.get("version", "1.0.0"),
                        "description": config.get("description", ""),
                        "format": "native",
                        "path": str(skill_dir),
                    })
            elif skill_md.exists():
                skills.append({
                    "name": skill_dir.name,
                    "version": "1.0.0",
                    "description": "Anthropic format skill",
                    "format": "anthropic",
                    "path": str(skill_dir),
                })

    return skills
