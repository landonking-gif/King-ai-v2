"""
Agent Management Commands.

Module: kautilya/commands/agent.py

Supports configuration of:
- Role-based capabilities
- Skills (from SkillRegistry)
- MCP tools (from MCP Gateway)
"""

from typing import Optional, List, Dict, Any
from pathlib import Path
import sys
import importlib.util
import click
import questionary
import yaml
import json
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

# Load SkillRegistry for skill discovery
_project_root = Path(__file__).parents[4]
_skill_registry_path = _project_root / "code-exec" / "service" / "skill_registry.py"

SkillRegistry = None
if _skill_registry_path.exists():
    try:
        # Load dependencies
        _models_path = _project_root / "code-exec" / "service" / "models.py"
        if _models_path.exists():
            spec = importlib.util.spec_from_file_location("models", _models_path)
            models_module = importlib.util.module_from_spec(spec)
            if "models" not in sys.modules:
                sys.modules["models"] = models_module
            spec.loader.exec_module(models_module)

        _skill_parser_path = _project_root / "code-exec" / "service" / "skill_parser.py"
        if _skill_parser_path.exists():
            spec = importlib.util.spec_from_file_location("skill_parser", _skill_parser_path)
            skill_parser_module = importlib.util.module_from_spec(spec)
            if "skill_parser" not in sys.modules:
                sys.modules["skill_parser"] = skill_parser_module
            spec.loader.exec_module(skill_parser_module)

        spec = importlib.util.spec_from_file_location("skill_registry", _skill_registry_path)
        skill_registry_module = importlib.util.module_from_spec(spec)
        sys.modules["skill_registry"] = skill_registry_module
        spec.loader.exec_module(skill_registry_module)
        SkillRegistry = skill_registry_module.SkillRegistry
    except Exception:
        pass  # SkillRegistry will be None


@click.group(name="agent")
def agent_cmd() -> None:
    """Manage subagents."""
    pass


@agent_cmd.command(name="new")
@click.argument("name")
@click.option("--role", help="Agent role (research/verify/code/synthesis/custom)")
@click.option("--capabilities", help="Comma-separated list of capabilities")
@click.option("--skills", help="Comma-separated list of skills to bind")
@click.option("--mcp-tools", help="Comma-separated list of MCP tools (format: server_id.tool_name or server_id:*)")
@click.option("--output-type", help="Output artifact type")
@click.pass_context
def agent_new_cmd(
    ctx: click.Context,
    name: str,
    role: Optional[str],
    capabilities: Optional[str],
    skills: Optional[str],
    mcp_tools: Optional[str],
    output_type: Optional[str],
) -> None:
    """Generate new subagent with role template, skills, and MCP tools."""
    config_dir = ctx.obj.get("config_dir", ".kautilya")
    create_agent(
        name, config_dir, role, capabilities, skills, mcp_tools, output_type, interactive=False
    )


def create_agent(
    name: str,
    config_dir: str,
    role: Optional[str] = None,
    capabilities: Optional[str] = None,
    skills: Optional[str] = None,
    mcp_tools: Optional[str] = None,
    output_type: Optional[str] = None,
    interactive: bool = True,
) -> None:
    """
    Create a new subagent.

    Args:
        name: Agent name
        config_dir: Configuration directory
        role: Agent role
        capabilities: Comma-separated capabilities
        skills: Comma-separated skill names
        mcp_tools: Comma-separated MCP tools (server_id.tool_name or server_id:*)
        output_type: Output artifact type
        interactive: Use interactive prompts
    """
    console.print(f"\n[bold cyan]Creating subagent: {name}[/bold cyan]\n")

    # Get role
    if not role and interactive:
        role = questionary.select(
            "Agent role:",
            choices=["research", "verify", "code", "synthesis", "custom"],
        ).ask()
    elif not role:
        role = "custom"

    # Get capabilities
    if not capabilities and interactive:
        capability_suggestions = _get_capability_suggestions(role)
        selected_caps = questionary.checkbox(
            "Select capabilities (space to select, enter to confirm):",
            choices=capability_suggestions,
        ).ask()
        capabilities = ",".join(selected_caps) if selected_caps else ""
    elif not capabilities:
        capabilities = ""

    caps_list = [c.strip() for c in capabilities.split(",") if c.strip()]

    # Get skills from SkillRegistry
    skills_list: List[str] = []
    skills_config: List[Dict[str, Any]] = []
    if not skills and interactive:
        available_skills = _get_available_skills()
        if available_skills:
            console.print("\n[bold]Available Skills:[/bold]")
            selected_skills = questionary.checkbox(
                "Select skills to bind (space to select, enter to confirm):",
                choices=[
                    questionary.Choice(
                        f"{s['name']} ({s['format']}) - {s['description'][:40]}...",
                        value=s["name"],
                    )
                    for s in available_skills
                ],
            ).ask()
            skills_list = selected_skills if selected_skills else []
        else:
            console.print("[dim]No skills available in SkillRegistry[/dim]")
    elif skills:
        skills_list = [s.strip() for s in skills.split(",") if s.strip()]

    # Build skills config with metadata
    if skills_list:
        available_skills = _get_available_skills()
        skill_map = {s["name"]: s for s in available_skills}
        for skill_name in skills_list:
            if skill_name in skill_map:
                skill_info = skill_map[skill_name]
                skills_config.append({
                    "name": skill_name,
                    "path": skill_info.get("path", ""),
                    "handler": skill_info.get("handler", ""),
                    "requires_approval": skill_info.get("requires_approval", False),
                })
            else:
                skills_config.append({"name": skill_name})

    # Get MCP tools
    mcp_tools_list: List[str] = []
    mcp_tools_config: List[Dict[str, Any]] = []
    if not mcp_tools and interactive:
        available_mcp = _get_available_mcp_tools()
        if available_mcp:
            console.print("\n[bold]Available MCP Tools:[/bold]")
            selected_mcp = questionary.checkbox(
                "Select MCP tools to bind (space to select, enter to confirm):",
                choices=[
                    questionary.Choice(
                        f"{t['server_id']}.{t['tool_name']} - {t['description'][:40]}...",
                        value=f"{t['server_id']}.{t['tool_name']}",
                    )
                    for t in available_mcp
                ],
            ).ask()
            mcp_tools_list = selected_mcp if selected_mcp else []
        else:
            console.print("[dim]No MCP tools available (MCP Gateway may be offline)[/dim]")
    elif mcp_tools:
        mcp_tools_list = [t.strip() for t in mcp_tools.split(",") if t.strip()]

    # Build MCP tools config
    for tool_spec in mcp_tools_list:
        if ":" in tool_spec:
            # Format: server_id:* (all tools from server)
            server_id, tool_pattern = tool_spec.split(":", 1)
            mcp_tools_config.append({
                "server_id": server_id,
                "tool_pattern": tool_pattern,
                "all_tools": tool_pattern == "*",
            })
        elif "." in tool_spec:
            # Format: server_id.tool_name
            server_id, tool_name = tool_spec.split(".", 1)
            mcp_tools_config.append({
                "server_id": server_id,
                "tool_name": tool_name,
            })
        else:
            # Assume it's a server_id with all tools
            mcp_tools_config.append({
                "server_id": tool_spec,
                "all_tools": True,
            })

    # Get output type
    if not output_type and interactive:
        output_type = questionary.select(
            "Output artifact type:",
            choices=["research_snippet", "claim_verification", "code_patch", "generic"],
        ).ask()
    elif not output_type:
        output_type = "generic"

    # Create agent directory
    agent_dir = Path.cwd() / "agents" / name
    agent_dir.mkdir(parents=True, exist_ok=True)

    # Create config.yaml with skills and mcp_tools
    config: Dict[str, Any] = {
        "name": name,
        "role": role,
        "capabilities": caps_list,
        "output_type": output_type,
        "timeout_seconds": 30,
        "retry_on_failure": True,
    }

    # Add skills section if any skills selected
    if skills_config:
        config["skills"] = skills_config

    # Add mcp_tools section if any MCP tools selected
    if mcp_tools_config:
        config["mcp_tools"] = mcp_tools_config

    with open(agent_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    # Create capabilities.json (enhanced with skills and tools)
    capabilities_doc = {
        "agent_id": name,
        "role": role,
        "capabilities": [{"name": cap, "description": f"{cap} capability"} for cap in caps_list],
        "skills": skills_config,
        "mcp_tools": mcp_tools_config,
        "restrictions": [],
    }

    with open(agent_dir / "capabilities.json", "w") as f:
        json.dump(capabilities_doc, f, indent=2)

    # Create prompts directory and system prompt
    prompts_dir = agent_dir / "prompts"
    prompts_dir.mkdir(exist_ok=True)

    system_prompt = _generate_system_prompt(
        name, role, caps_list, output_type, skills_config, mcp_tools_config
    )
    (prompts_dir / "system.txt").write_text(system_prompt)

    # Show success
    skills_display = ", ".join(skills_list) if skills_list else "none"
    mcp_display = ", ".join(mcp_tools_list) if mcp_tools_list else "none"

    success_message = f"""
[green]✓[/green] Generated agent: agents/{name}/
  ├── config.yaml
  ├── capabilities.json
  └── prompts/system.txt

[bold]Configuration:[/bold]
  Role: {role}
  Capabilities: {', '.join(caps_list) if caps_list else 'none'}
  Skills: {skills_display}
  MCP Tools: {mcp_display}
  Output type: {output_type}
    """

    console.print(Panel(success_message.strip(), title="[bold green]Agent Created[/bold green]"))


def _get_capability_suggestions(role: str) -> List[str]:
    """Get capability suggestions based on role."""
    capabilities_by_role = {
        "research": [
            "web_search",
            "document_read",
            "summarize",
            "extract_entities",
            "semantic_search",
        ],
        "verify": [
            "fact_check",
            "cross_reference",
            "evidence_collection",
            "confidence_scoring",
        ],
        "code": [
            "code_generation",
            "code_review",
            "test_creation",
            "refactoring",
            "documentation",
        ],
        "synthesis": [
            "content_generation",
            "summarization",
            "formatting",
            "quality_check",
        ],
        "custom": [
            "web_search",
            "document_read",
            "code_generation",
            "summarize",
        ],
    }

    return capabilities_by_role.get(role, capabilities_by_role["custom"])


def _get_available_skills() -> List[Dict[str, Any]]:
    """Get available skills from SkillRegistry."""
    if SkillRegistry is None:
        return []

    try:
        registry = SkillRegistry()
        skills = registry.discover_all()
        return [
            {
                "name": skill.name,
                "format": skill.format_label,
                "description": skill.description,
                "path": str(skill.path),
                "handler": skill.handler_function,
                "requires_approval": skill.requires_approval,
                "safety_flags": skill.safety_flags,
            }
            for skill in skills
        ]
    except Exception:
        return []


def _get_available_mcp_tools() -> List[Dict[str, Any]]:
    """Get available MCP tools from gateway."""
    import os

    gateway_url = os.getenv("MCP_GATEWAY_URL", "http://localhost:8080")

    try:
        import httpx

        with httpx.Client(timeout=5.0) as client:
            response = client.get(
                f"{gateway_url}/catalog/tools",
                params={"enabled_only": True},
            )
            response.raise_for_status()
            data = response.json()

        tools = []
        for server in data.get("servers", []):
            server_id = server.get("server_id", "")
            for tool in server.get("tools", []):
                tools.append({
                    "server_id": server_id,
                    "tool_name": tool.get("name", ""),
                    "description": tool.get("description", "")[:80],
                })
        return tools
    except Exception:
        return []


def _generate_system_prompt(
    name: str,
    role: str,
    capabilities: List[str],
    output_type: str,
    skills: Optional[List[Dict[str, Any]]] = None,
    mcp_tools: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Generate system prompt for agent with skills and MCP tools."""
    skills = skills or []
    mcp_tools = mcp_tools or []

    # Build skills section
    skills_section = ""
    if skills:
        skills_list = "\n".join(f"- {s['name']}" for s in skills)
        skills_section = f"""
## Bound Skills
You have access to the following deterministic skills:
{skills_list}

Skills are executed via the Code Executor service. Each skill:
- Has defined input/output schemas
- Runs in a sandboxed environment
- May require approval based on safety flags
"""

    # Build MCP tools section
    mcp_section = ""
    if mcp_tools:
        mcp_list = []
        for t in mcp_tools:
            if t.get("all_tools"):
                mcp_list.append(f"- {t['server_id']}:* (all tools)")
            elif t.get("tool_name"):
                mcp_list.append(f"- {t['server_id']}.{t['tool_name']}")
            else:
                mcp_list.append(f"- {t['server_id']}")
        mcp_section = f"""
## Bound MCP Tools
You have access to the following MCP (Model Context Protocol) tools:
{chr(10).join(mcp_list)}

MCP tools are executed via the MCP Gateway. Each tool call:
- Is rate-limited and authenticated
- Has provenance tracking
- May have PII/security restrictions
"""

    return f"""# {name} Agent - {role.capitalize()} Role

You are a {role} agent in an enterprise agentic framework.

## Your Role
{_get_role_description(role)}

## Capabilities
You have access to the following capabilities:
{chr(10).join(f'- {cap}' for cap in capabilities) if capabilities else '- No specific capabilities'}
{skills_section}{mcp_section}
## Output Requirements
You must produce output in the following format: {output_type}

Ensure all outputs are:
1. JSON Schema validated
2. Include provenance information
3. Have appropriate confidence scores
4. Follow safety classifications

## Constraints
- Stay within your defined capabilities, skills, and tools
- Request approval for sensitive operations
- Log all tool invocations with provenance
- Handle errors gracefully with informative messages
"""


def _get_role_description(role: str) -> str:
    """Get description for role."""
    descriptions = {
        "research": "Gather information from various sources, synthesize findings, and produce research snippets.",
        "verify": "Validate claims, check facts, and assess evidence quality with confidence scoring.",
        "code": "Generate, review, and refactor code with tests and documentation.",
        "synthesis": "Combine information from multiple sources into coherent outputs.",
        "custom": "Perform custom tasks as defined by your capabilities.",
    }
    return descriptions.get(role, descriptions["custom"])


def create_agent_programmatic(
    name: str,
    role: str = "custom",
    capabilities: Optional[List[str]] = None,
    skills: Optional[List[str]] = None,
    mcp_tools: Optional[List[str]] = None,
    output_type: Optional[str] = None,
    config_dir: str = ".kautilya",
) -> dict:
    """
    Programmatic interface for creating agents.

    Args:
        name: Agent name
        role: Agent role
        capabilities: List of capabilities
        skills: List of skill names to bind
        mcp_tools: List of MCP tools (format: server_id.tool_name or server_id:*)
        output_type: Output artifact type
        config_dir: Configuration directory

    Returns:
        Dictionary with agent details
    """
    caps_list = capabilities or []
    skills_list = skills or []
    mcp_tools_list = mcp_tools or []
    output_type = output_type or "generic"

    # Build skills config with metadata from registry
    skills_config: List[Dict[str, Any]] = []
    if skills_list:
        available_skills = _get_available_skills()
        skill_map = {s["name"]: s for s in available_skills}
        for skill_name in skills_list:
            if skill_name in skill_map:
                skill_info = skill_map[skill_name]
                skills_config.append({
                    "name": skill_name,
                    "path": skill_info.get("path", ""),
                    "handler": skill_info.get("handler", ""),
                    "requires_approval": skill_info.get("requires_approval", False),
                })
            else:
                skills_config.append({"name": skill_name})

    # Build MCP tools config
    mcp_tools_config: List[Dict[str, Any]] = []
    for tool_spec in mcp_tools_list:
        if ":" in tool_spec:
            server_id, tool_pattern = tool_spec.split(":", 1)
            mcp_tools_config.append({
                "server_id": server_id,
                "tool_pattern": tool_pattern,
                "all_tools": tool_pattern == "*",
            })
        elif "." in tool_spec:
            server_id, tool_name = tool_spec.split(".", 1)
            mcp_tools_config.append({
                "server_id": server_id,
                "tool_name": tool_name,
            })
        else:
            mcp_tools_config.append({
                "server_id": tool_spec,
                "all_tools": True,
            })

    # Create agent directory
    agent_dir = Path.cwd() / "agents" / name
    agent_dir.mkdir(parents=True, exist_ok=True)

    # Create config.yaml with skills and mcp_tools
    config: Dict[str, Any] = {
        "name": name,
        "role": role,
        "capabilities": caps_list,
        "output_type": output_type,
        "timeout_seconds": 30,
        "retry_on_failure": True,
    }

    if skills_config:
        config["skills"] = skills_config

    if mcp_tools_config:
        config["mcp_tools"] = mcp_tools_config

    with open(agent_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    # Create capabilities.json (enhanced with skills and tools)
    capabilities_doc = {
        "agent_id": name,
        "role": role,
        "capabilities": [{"name": cap, "description": f"{cap} capability"} for cap in caps_list],
        "skills": skills_config,
        "mcp_tools": mcp_tools_config,
        "restrictions": [],
    }

    with open(agent_dir / "capabilities.json", "w") as f:
        json.dump(capabilities_doc, f, indent=2)

    # Create prompts directory and system prompt
    prompts_dir = agent_dir / "prompts"
    prompts_dir.mkdir(exist_ok=True)

    system_prompt = _generate_system_prompt(
        name, role, caps_list, output_type, skills_config, mcp_tools_config
    )
    (prompts_dir / "system.txt").write_text(system_prompt)

    return {
        "agent_path": str(agent_dir),
        "role": role,
        "capabilities": caps_list,
        "skills": skills_list,
        "mcp_tools": mcp_tools_list,
        "output_type": output_type,
        "files_created": [
            str(agent_dir / "config.yaml"),
            str(agent_dir / "capabilities.json"),
            str(prompts_dir / "system.txt"),
        ],
    }
