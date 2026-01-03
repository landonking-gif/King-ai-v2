"""
Tool Executor for Kautilya LLM Integration.

Module: kautilya/tool_executor.py

Executes Kautilya commands based on LLM tool calls.
Includes MCP Gateway integration for external tool invocation.
Tracks sources for transparency and attribution.
"""

import os
import sys
import json
import logging
import time
import importlib.util
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pathlib import Path
from rich.console import Console

console = Console()
logger = logging.getLogger(__name__)

# Load SkillRegistry for dynamic skill discovery
# Path: tool_executor.py -> kautilya -> kautilya -> tools -> agent-framework
_project_root = Path(__file__).parents[3]  # kautilya(dir) -> tools -> agent-framework
_skill_registry_path = _project_root / "code-exec" / "service" / "skill_registry.py"

# Ensure skills directory is in Python path for relative imports
_skills_dir = _project_root / "code-exec" / "skills"
if _skills_dir.exists() and str(_skills_dir) not in sys.path:
    sys.path.insert(0, str(_skills_dir))
    logger.debug(f"Added skills directory to sys.path: {_skills_dir}")

SkillRegistry = None
if _skill_registry_path.exists():
    try:
        # Load models.py first (dependency)
        _models_path = _project_root / "code-exec" / "service" / "models.py"
        if _models_path.exists():
            spec = importlib.util.spec_from_file_location("models", _models_path)
            models_module = importlib.util.module_from_spec(spec)
            if "models" not in sys.modules:
                sys.modules["models"] = models_module
            spec.loader.exec_module(models_module)

        # Load skill_parser.py (dependency)
        _skill_parser_path = _project_root / "code-exec" / "service" / "skill_parser.py"
        if _skill_parser_path.exists():
            spec = importlib.util.spec_from_file_location("skill_parser", _skill_parser_path)
            skill_parser_module = importlib.util.module_from_spec(spec)
            if "skill_parser" not in sys.modules:
                sys.modules["skill_parser"] = skill_parser_module
            spec.loader.exec_module(skill_parser_module)

        # Load skill_registry.py
        spec = importlib.util.spec_from_file_location("skill_registry", _skill_registry_path)
        skill_registry_module = importlib.util.module_from_spec(spec)
        sys.modules["skill_registry"] = skill_registry_module
        spec.loader.exec_module(skill_registry_module)
        SkillRegistry = skill_registry_module.SkillRegistry
        logger.debug("SkillRegistry loaded successfully")
    except Exception as e:
        logger.warning(f"Failed to load SkillRegistry: {e}")


class SourceType(str, Enum):
    """Types of sources that can be tracked."""
    FILE_READ = "file_read"
    FILE_SEARCH = "file_search"
    WEB_FETCH = "web_fetch"
    WEB_SEARCH = "web_search"
    MCP_CALL = "mcp_call"
    BASH_EXEC = "bash_exec"
    CONFIG_READ = "config_read"


@dataclass
class SourceEntry:
    """A single source entry for attribution."""
    source_type: SourceType
    location: str  # file path, URL, or command
    description: str  # what was found/done
    timestamp: datetime = field(default_factory=datetime.now)
    line_range: Optional[str] = None  # e.g., "10-25" for file reads
    snippet: Optional[str] = None  # brief excerpt if relevant

    def to_display(self, include_snippet: bool = False) -> str:
        """Format for display."""
        if self.source_type == SourceType.FILE_READ:
            loc = self.location
            if self.line_range:
                loc = f"{self.location}:{self.line_range}"
            return f"📄 {loc}"
        elif self.source_type == SourceType.FILE_SEARCH:
            return f"🔍 Search: {self.description} → {self.location}"
        elif self.source_type == SourceType.WEB_FETCH:
            return f"🌐 {self.location}"
        elif self.source_type == SourceType.WEB_SEARCH:
            return f"🔎 Web search: {self.description}"
        elif self.source_type == SourceType.MCP_CALL:
            return f"🔌 MCP: {self.location} → {self.description}"
        elif self.source_type == SourceType.BASH_EXEC:
            return f"⚡ Command: {self.location}"
        elif self.source_type == SourceType.CONFIG_READ:
            return f"⚙️ Config: {self.location}"
        return f"{self.source_type.value}: {self.location}"


class SourceTracker:
    """Tracks sources consulted during tool execution for attribution."""

    def __init__(self):
        """Initialize source tracker."""
        self._sources: List[SourceEntry] = []
        self._assumptions: List[str] = []

    def clear(self) -> None:
        """Clear all tracked sources and assumptions."""
        self._sources.clear()
        self._assumptions.clear()

    def add_source(
        self,
        source_type: SourceType,
        location: str,
        description: str,
        line_range: Optional[str] = None,
        snippet: Optional[str] = None,
    ) -> None:
        """Add a source entry."""
        self._sources.append(SourceEntry(
            source_type=source_type,
            location=location,
            description=description,
            line_range=line_range,
            snippet=snippet,
        ))

    def add_assumption(self, assumption: str) -> None:
        """Track an assumption made during execution."""
        if assumption not in self._assumptions:
            self._assumptions.append(assumption)

    def get_sources(self) -> List[SourceEntry]:
        """Get all tracked sources."""
        return self._sources.copy()

    def get_assumptions(self) -> List[str]:
        """Get all tracked assumptions."""
        return self._assumptions.copy()

    def has_sources(self) -> bool:
        """Check if any sources were tracked."""
        return len(self._sources) > 0

    def format_sources_summary(self, max_items: int = 10) -> str:
        """
        Format sources as a summary block.

        Args:
            max_items: Maximum number of items to show

        Returns:
            Formatted string for display
        """
        if not self._sources and not self._assumptions:
            return ""

        lines = []

        # Group sources by type
        by_type: Dict[SourceType, List[SourceEntry]] = {}
        for source in self._sources:
            if source.source_type not in by_type:
                by_type[source.source_type] = []
            by_type[source.source_type].append(source)

        # Format sources
        if self._sources:
            lines.append("📋 **Sources consulted:**")
            count = 0
            for source_type, entries in by_type.items():
                for entry in entries:
                    if count >= max_items:
                        remaining = len(self._sources) - count
                        lines.append(f"  ... and {remaining} more")
                        break
                    lines.append(f"  • {entry.to_display()}")
                    count += 1
                if count >= max_items:
                    break

        # Format assumptions
        if self._assumptions:
            lines.append("")
            lines.append("💡 **Assumptions made:**")
            for assumption in self._assumptions[:5]:
                lines.append(f"  • {assumption}")
            if len(self._assumptions) > 5:
                lines.append(f"  ... and {len(self._assumptions) - 5} more")

        return "\n".join(lines)

    def format_inline_citations(self) -> Dict[str, str]:
        """
        Generate citation mapping for inline references.

        Returns:
            Dict mapping citation key [1], [2] to source description
        """
        citations = {}
        for i, source in enumerate(self._sources, 1):
            key = f"[{i}]"
            citations[key] = source.to_display()
        return citations


# Global source tracker instance
_source_tracker: Optional[SourceTracker] = None


def get_source_tracker() -> SourceTracker:
    """Get the global source tracker instance."""
    global _source_tracker
    if _source_tracker is None:
        _source_tracker = SourceTracker()
    return _source_tracker


def clear_source_tracker() -> None:
    """Clear the global source tracker."""
    global _source_tracker
    if _source_tracker is not None:
        _source_tracker.clear()


class ToolExecutor:
    """Executes Kautilya tools from LLM function calls."""

    # Cache for MCP tools registry (tool_id.tool_name -> server info)
    _mcp_tools_cache: Optional[Dict[str, Dict[str, Any]]] = None
    _mcp_cache_timestamp: float = 0
    _MCP_CACHE_TTL: float = 300  # 5 minutes

    def __init__(self, config_dir: str = ".kautilya"):
        """
        Initialize tool executor.

        Args:
            config_dir: Configuration directory
        """
        self.config_dir = config_dir
        self.project_root = os.getcwd()
        self.gateway_url = os.getenv("MCP_GATEWAY_URL", "http://localhost:8080")
        self._skill_registry = None

    def _get_skill_path(self, skill_name: str, fallback_path: Optional[Path] = None) -> Optional[Path]:
        """
        Get skill path using SkillRegistry or fallback.

        Args:
            skill_name: Name of the skill to find
            fallback_path: Fallback path if registry lookup fails

        Returns:
            Path to the skill directory, or None if not found
        """
        if SkillRegistry is not None:
            try:
                if self._skill_registry is None:
                    self._skill_registry = SkillRegistry()
                skill_meta = self._skill_registry.get_skill(skill_name)
                if skill_meta:
                    return skill_meta.path
            except Exception as e:
                logger.warning(f"SkillRegistry lookup failed for {skill_name}: {e}")

        # Return fallback if provided
        if fallback_path and fallback_path.exists():
            return fallback_path

        return None

    def execute(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool by name with given arguments.

        Args:
            tool_name: Name of the tool to execute
            args: Tool arguments

        Returns:
            Execution result
        """
        method = getattr(self, f"_exec_{tool_name}", None)

        if method is None:
            # Try to execute as a skill from skills directory
            skill_result = self._try_execute_skill(tool_name, args)
            if skill_result is not None:
                return skill_result

            return {
                "success": False,
                "error": f"Unknown tool: {tool_name}",
            }

        try:
            return method(**args)
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

    def _try_execute_skill(
        self,
        skill_name: str,
        args: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """
        Try to execute a skill from the skills directory.

        Skills are dynamically discovered and executed from code-exec/skills/.

        Args:
            skill_name: Name of the skill (e.g., "document_qa", "deep_research")
            args: Skill arguments

        Returns:
            Execution result or None if skill not found
        """
        # Normalize skill name (replace hyphens with underscores)
        normalized_name = skill_name.replace("-", "_")

        # Find skills directory
        skills_dir = _project_root / "code-exec" / "skills"
        if not skills_dir.exists():
            logger.debug(f"Skills directory not found: {skills_dir}")
            return None

        # Try to find the skill - check multiple locations
        skill_path = None
        possible_paths = [
            skills_dir / normalized_name,  # e.g., skills/document_qa
            skills_dir / skill_name,  # e.g., skills/document-qa
        ]

        # Also search in subdirectories (e.g., skills/file_operations/file_write)
        for subdir in skills_dir.iterdir():
            if subdir.is_dir() and not subdir.name.startswith(("_", ".")):
                possible_paths.append(subdir / normalized_name)
                possible_paths.append(subdir / skill_name)

        for path in possible_paths:
            if path.exists() and path.is_dir():
                skill_path = path
                break

        if not skill_path:
            logger.debug(f"Skill not found: {skill_name}")
            return None

        # Load and execute the handler
        handler_path = skill_path / "handler.py"
        if not handler_path.exists():
            logger.warning(f"Skill handler not found: {handler_path}")
            return {
                "success": False,
                "error": f"Skill handler not found: {handler_path}",
                "suggestion": f"Check if {skill_name}/handler.py exists in skills directory",
            }

        try:
            import importlib

            # Add skill directory and parent to path for imports
            skill_parent = skill_path.parent
            paths_to_add = [str(skill_parent), str(skill_path)]
            for p in paths_to_add:
                if p not in sys.path:
                    sys.path.insert(0, p)

            # For skills with subpackages (like document_qa with components/),
            # we need to set up proper package structure for relative imports
            package_name = normalized_name

            # Check if skill has an __init__.py (is a package)
            init_path = skill_path / "__init__.py"
            has_init = init_path.exists()

            # Create a proper package module first
            if has_init or (skill_path / "components").exists() or (skill_path / "pipelines").exists():
                # This is a package-style skill - import as a package

                # Make sure parent is in path
                if str(skill_parent) not in sys.path:
                    sys.path.insert(0, str(skill_parent))

                # Import the package and its handler
                try:
                    # Try importing as a package first
                    pkg = importlib.import_module(package_name)
                    # Now import the handler from within the package
                    handler_module = importlib.import_module(f"{package_name}.handler")
                except ImportError as ie:
                    # If that fails, create the package structure manually
                    pkg_spec = importlib.util.spec_from_file_location(
                        package_name,
                        init_path if has_init else handler_path,
                        submodule_search_locations=[str(skill_path)],
                    )
                    pkg = importlib.util.module_from_spec(pkg_spec)
                    pkg.__path__ = [str(skill_path)]
                    sys.modules[package_name] = pkg
                    if has_init:
                        pkg_spec.loader.exec_module(pkg)

                    # Now load the handler
                    handler_spec = importlib.util.spec_from_file_location(
                        f"{package_name}.handler",
                        handler_path,
                    )
                    handler_module = importlib.util.module_from_spec(handler_spec)
                    handler_module.__package__ = package_name
                    sys.modules[f"{package_name}.handler"] = handler_module
                    handler_spec.loader.exec_module(handler_module)
            else:
                # Simple skill - load directly
                spec = importlib.util.spec_from_file_location(
                    f"skill_{normalized_name}_handler",
                    handler_path
                )
                handler_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(handler_module)

            # Find the main function (same name as skill or first callable)
            import inspect

            handler_func = getattr(handler_module, normalized_name, None)
            if handler_func is None:
                # Try common handler function names
                common_names = [
                    normalized_name.replace("_", ""),  # file_write -> filewrite
                    "run", "execute", "main", "handler",
                    # Also try removing common prefixes/suffixes
                    normalized_name.split("_")[-1],  # file_write -> write
                ]
                for name in common_names:
                    if hasattr(handler_module, name):
                        obj = getattr(handler_module, name)
                        if inspect.isfunction(obj):
                            handler_func = obj
                            break

            if handler_func is None:
                # Find the first actual function (not types or imports)
                for name in dir(handler_module):
                    if not name.startswith("_"):
                        obj = getattr(handler_module, name)
                        if inspect.isfunction(obj):
                            handler_func = obj
                            break

            if handler_func is None:
                logger.warning(f"No handler function found in {handler_path}")
                return None

            logger.info(f"Executing skill: {skill_name} with handler: {handler_func.__name__}")

            # Execute the handler
            result = handler_func(**args)

            # Ensure result is a dict
            if not isinstance(result, dict):
                result = {"success": True, "result": result}

            # Track skill execution as a source
            try:
                tracker = get_source_tracker()
                # Build description from args
                desc_parts = []
                if 'query' in args:
                    desc_parts.append(f"Query: {args['query'][:50]}...")
                if 'documents' in args:
                    docs = args['documents']
                    if isinstance(docs, list):
                        desc_parts.append(f"Documents: {', '.join(str(d)[:30] for d in docs[:3])}")
                desc = " | ".join(desc_parts) if desc_parts else f"Skill: {skill_name}"

                # Track sources from skill result if available
                if result.get('sources'):
                    for src in result['sources'][:5]:  # Limit to 5 sources
                        src_file = src.get('file', src.get('document', 'unknown'))
                        src_page = src.get('page', src.get('chunk_id', ''))
                        tracker.add_source(
                            source_type=SourceType.FILE_READ,
                            location=str(src_file),
                            description=f"Page {src_page}" if src_page else "Document content",
                        )
            except Exception:
                pass  # Don't fail if source tracking fails

            return result

        except Exception as e:
            logger.error(f"Failed to execute skill {skill_name}: {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": f"Skill execution failed: {str(e)}",
            }

    # =========================================================================
    # MCP Gateway Integration - External Tool Execution
    # =========================================================================

    def _refresh_mcp_tools_cache(self) -> Dict[str, Dict[str, Any]]:
        """
        Refresh the MCP tools cache from the gateway.

        Returns:
            Dict mapping tool identifiers to their metadata
        """
        current_time = time.time()

        # Return cached data if still valid
        if (
            ToolExecutor._mcp_tools_cache is not None
            and current_time - ToolExecutor._mcp_cache_timestamp < ToolExecutor._MCP_CACHE_TTL
        ):
            return ToolExecutor._mcp_tools_cache

        try:
            import httpx

            with httpx.Client(timeout=10.0) as client:
                response = client.get(
                    f"{self.gateway_url}/catalog/tools",
                    params={"enabled_only": True},
                )
                response.raise_for_status()
                data = response.json()

            tools_cache: Dict[str, Dict[str, Any]] = {}
            servers = data.get("servers", [])

            for server_entry in servers:
                registration = server_entry.get("registration", {})
                tool_id = registration.get("tool_id", "")
                server_name = registration.get("name", "")
                endpoint = registration.get("endpoint", "")
                auth_flow = registration.get("auth_flow", "none")
                metadata = registration.get("metadata", {})

                for tool in registration.get("tools", []):
                    tool_name = tool.get("name", "")
                    full_key = f"{tool_id}.{tool_name}"

                    tools_cache[full_key] = {
                        "tool_id": tool_id,
                        "tool_name": tool_name,
                        "server_name": server_name,
                        "description": tool.get("description", ""),
                        "parameters": tool.get("parameters", []),
                        "returns": tool.get("returns", ""),
                        "endpoint": endpoint,
                        "auth_flow": auth_flow,
                        "api_key_env": metadata.get("api_key_env"),
                    }

            ToolExecutor._mcp_tools_cache = tools_cache
            ToolExecutor._mcp_cache_timestamp = current_time
            logger.debug(f"Refreshed MCP tools cache: {len(tools_cache)} tools")
            return tools_cache

        except Exception as e:
            logger.warning(f"Failed to refresh MCP tools cache: {e}")
            return ToolExecutor._mcp_tools_cache or {}

    def get_available_mcp_tools(self) -> List[Dict[str, Any]]:
        """
        Get list of available MCP tools from registered servers.

        Returns:
            List of tool definitions with id, name, description, parameters
        """
        cache = self._refresh_mcp_tools_cache()
        return [
            {
                "tool_id": info["tool_id"],
                "tool_name": info["tool_name"],
                "full_name": f"{info['tool_id']}.{info['tool_name']}",
                "server_name": info["server_name"],
                "description": info["description"],
                "parameters": info["parameters"],
            }
            for info in cache.values()
        ]

    def is_mcp_tool(self, tool_id: str, tool_name: str) -> bool:
        """
        Check if a tool is available via MCP gateway.

        Args:
            tool_id: MCP server ID (e.g., "firecrawl_mcp")
            tool_name: Tool name (e.g., "scrape")

        Returns:
            True if tool exists in MCP registry
        """
        cache = self._refresh_mcp_tools_cache()
        full_key = f"{tool_id}.{tool_name}"
        return full_key in cache

    def _exec_mcp_call(
        self,
        tool_id: str,
        tool_name: str,
        arguments: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute an MCP tool through the gateway.

        This is the universal bridge for calling any external MCP server tool.
        The gateway handles authentication, rate limiting, and proxying.

        Args:
            tool_id: MCP server ID (e.g., "firecrawl_mcp", "github_mcp")
            tool_name: Specific tool name (e.g., "scrape", "crawl", "search")
            arguments: Tool-specific arguments

        Returns:
            Execution result from the MCP server
        """
        import httpx

        # Validate tool exists
        cache = self._refresh_mcp_tools_cache()
        full_key = f"{tool_id}.{tool_name}"

        if full_key not in cache:
            available_tools = list(cache.keys())[:10]
            return {
                "success": False,
                "error": f"Unknown MCP tool: {full_key}",
                "hint": f"Available tools: {available_tools}",
                "suggestion": "Use /mcp list to see all available MCP servers and their tools",
            }

        tool_info = cache[full_key]
        arguments = arguments or {}

        # Build invocation request
        invocation_request = {
            "tool_id": tool_id,
            "tool_name": tool_name,
            "arguments": arguments,
            "actor_id": "kautilya_cli",
            "actor_type": "lead_agent",
            "runtime_mode": "orchestrated",
        }

        try:
            with httpx.Client(timeout=60.0) as client:
                response = client.post(
                    f"{self.gateway_url}/tools/invoke",
                    json=invocation_request,
                )

                if response.status_code == 200:
                    result = response.json()

                    # Track source for attribution with meaningful description
                    tracker = get_source_tracker()

                    # Extract the most useful info from arguments for display
                    source_detail = ""
                    if arguments:
                        # Priority order for common argument names
                        for key in ["url", "query", "search", "path", "file", "repo", "id", "name"]:
                            if key in arguments:
                                val = str(arguments[key])
                                # Truncate long values
                                if len(val) > 50:
                                    val = val[:47] + "..."
                                source_detail = val
                                break
                        if not source_detail:
                            # Fallback: use first argument value
                            first_val = str(list(arguments.values())[0])
                            if len(first_val) > 50:
                                first_val = first_val[:47] + "..."
                            source_detail = first_val

                    tracker.add_source(
                        source_type=SourceType.MCP_CALL,
                        location=f"{tool_id}.{tool_name}",
                        description=source_detail or f"Called with {len(arguments)} args",
                    )

                    return {
                        "success": result.get("success", True),
                        "result": result.get("result"),
                        "invocation_id": result.get("invocation_id"),
                        "execution_time_ms": result.get("execution_time_ms"),
                        "tool_id": tool_id,
                        "tool_name": tool_name,
                    }
                else:
                    error_detail = response.json().get("detail", response.text)
                    return {
                        "success": False,
                        "error": f"Gateway error ({response.status_code}): {error_detail}",
                        "tool_id": tool_id,
                        "tool_name": tool_name,
                    }

        except httpx.TimeoutException:
            return {
                "success": False,
                "error": f"Timeout calling {tool_id}.{tool_name}",
                "hint": "The external MCP server took too long to respond",
            }
        except httpx.ConnectError:
            return {
                "success": False,
                "error": "Cannot connect to MCP Gateway",
                "hint": "Ensure gateway is running: check with /status or start with kautilya",
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"MCP call failed: {str(e)}",
                "tool_id": tool_id,
                "tool_name": tool_name,
            }

    def _exec_init_project(
        self,
        name: str,
        provider: str = "openai",
        enable_mcp: bool = True,
    ) -> Dict[str, Any]:
        """Initialize a new agent project."""
        from .commands.init import initialize_project_programmatic

        try:
            result = initialize_project_programmatic(
                name=name,
                provider=provider,
                enable_mcp=enable_mcp,
                config_dir=self.config_dir,
            )
            return {
                "success": True,
                "message": f"Created project '{name}' with {provider} provider",
                "project_path": result.get("project_path", name),
                "files_created": result.get("files_created", []),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _exec_create_agent(
        self,
        name: str,
        role: str = "custom",
        capabilities: Optional[list] = None,
        output_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a new agent."""
        from .commands.agent import create_agent_programmatic

        try:
            result = create_agent_programmatic(
                name=name,
                role=role,
                capabilities=capabilities or [],
                output_type=output_type,
                config_dir=self.config_dir,
            )
            return {
                "success": True,
                "message": f"Created agent '{name}' with role '{role}'",
                "agent_path": result.get("agent_path", f"agents/{name}"),
                "capabilities": capabilities or [],
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _exec_create_skill(
        self,
        name: str,
        description: Optional[str] = None,
        input_fields: Optional[list] = None,
        output_fields: Optional[list] = None,
        safety_flags: Optional[list] = None,
    ) -> Dict[str, Any]:
        """Create a new skill."""
        from .commands.skill import create_skill_programmatic

        try:
            result = create_skill_programmatic(
                name=name,
                description=description or f"Skill: {name}",
                input_fields=input_fields or ["input"],
                output_fields=output_fields or ["result"],
                safety_flags=safety_flags or [],
                config_dir=self.config_dir,
            )
            return {
                "success": True,
                "message": f"Created skill '{name}'",
                "skill_path": result.get("skill_path", f"skills/{name.replace('-', '_')}"),
                "files_created": result.get("files_created", []),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _exec_list_skills(self) -> Dict[str, Any]:
        """List all available skills."""
        from .commands.skill import list_skills_programmatic

        try:
            skills = list_skills_programmatic(config_dir=self.config_dir)
            return {
                "success": True,
                "skills": skills,
                "count": len(skills),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _exec_configure_llm(
        self,
        provider: str,
        model: Optional[str] = None,
        set_default: bool = False,
    ) -> Dict[str, Any]:
        """Configure LLM provider."""
        from .commands.llm import configure_llm_programmatic

        try:
            result = configure_llm_programmatic(
                provider=provider,
                model=model,
                set_default=set_default,
                config_dir=self.config_dir,
            )
            return {
                "success": True,
                "message": f"Configured {provider} provider" + (" as default" if set_default else ""),
                "provider": provider,
                "model": model or result.get("model"),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _exec_list_llm_providers(self) -> Dict[str, Any]:
        """
        List available LLM providers (same as command-line).

        Returns all 6 supported LLM providers with their configurations.
        This matches the command-line 'kautilya llm list' output.
        """
        # Keep this in sync with commands/llm.py:list_llm_providers()
        providers = [
            {
                "name": "anthropic",
                "default_model": "claude-sonnet-4-20250514",
                "fallback_model": "claude-haiku-4-20250514",
                "api_key_env": "ANTHROPIC_API_KEY",
                "endpoint": "API",
                "status": "✅ Production",
                "description": "Anthropic Claude API",
            },
            {
                "name": "openai",
                "default_model": "gpt-4o",
                "fallback_model": "gpt-4o-mini",
                "api_key_env": "OPENAI_API_KEY",
                "endpoint": "API",
                "status": "✅ Production",
                "description": "OpenAI GPT models",
            },
            {
                "name": "azure",
                "default_model": "gpt-4o",
                "fallback_model": "gpt-4o-mini",
                "api_key_env": "AZURE_OPENAI_KEY",
                "endpoint": "Custom",
                "status": "✅ Production",
                "description": "Azure OpenAI services",
            },
            {
                "name": "gemini",
                "default_model": "gemini-2.0-flash",
                "fallback_model": "gemini-1.5-pro",
                "api_key_env": "GEMINI_API_KEY",
                "endpoint": "API",
                "status": "✅ Production",
                "description": "Google Gemini API",
            },
            {
                "name": "local",
                "default_model": "llama3.1:70b",
                "fallback_model": "llama3.1:8b",
                "api_key_env": "N/A",
                "endpoint": "localhost:11434",
                "status": "⚠️ Requires Ollama server",
                "description": "Ollama (local inference)",
            },
            {
                "name": "vllm",
                "default_model": "meta-llama/Llama-2-70b",
                "fallback_model": "meta-llama/Llama-2-13b",
                "api_key_env": "N/A",
                "endpoint": "localhost:8000",
                "status": "⚠️ Requires vLLM server",
                "description": "vLLM (optimized local inference)",
            },
        ]
        return {
            "success": True,
            "providers": providers,
            "count": len(providers),
            "message": "All 6 LLM providers available. Configure with: /llm config",
        }

    def _exec_test_llm_connection(self) -> Dict[str, Any]:
        """Test LLM connection."""
        from .commands.llm import test_llm_connection_programmatic

        try:
            result = test_llm_connection_programmatic(config_dir=self.config_dir)
            return {
                "success": True,
                "message": "LLM connection successful",
                "provider": result.get("provider"),
                "model": result.get("model"),
                "response_time_ms": result.get("response_time_ms"),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _exec_set_llm_params(self, **kwargs) -> Dict[str, Any]:
        """Set LLM hyperparameters."""
        from .commands.llm import set_hyperparameters

        try:
            provider = kwargs.get("provider")
            temperature = kwargs.get("temperature")
            max_tokens = kwargs.get("max_tokens")
            top_p = kwargs.get("top_p")
            top_k = kwargs.get("top_k")
            frequency_penalty = kwargs.get("frequency_penalty")
            presence_penalty = kwargs.get("presence_penalty")
            max_retries = kwargs.get("max_retries")

            set_hyperparameters(
                self.config_dir,
                provider=provider,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                top_k=top_k,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                max_retries=max_retries,
                interactive=False,
            )

            return {
                "success": True,
                "message": f"Hyperparameters updated for {provider or 'default provider'}",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _exec_show_llm_params(self, **kwargs) -> Dict[str, Any]:
        """Show LLM hyperparameters."""
        from .config import load_llm_config

        try:
            provider = kwargs.get("provider")

            llm_config = load_llm_config(self.config_dir)

            if not llm_config or "providers" not in llm_config:
                return {
                    "success": False,
                    "error": "No LLM providers configured. Run /llm config first.",
                }

            if not provider:
                provider = llm_config.get("default_provider", "anthropic")

            if provider not in llm_config["providers"]:
                return {"success": False, "error": f"Provider {provider} not configured"}

            provider_config = llm_config["providers"][provider]
            hyperparams = provider_config.get("hyperparameters", {})

            return {
                "success": True,
                "provider": provider,
                "hyperparameters": hyperparams,
                "message": f"Hyperparameters for {provider}",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _exec_add_mcp_server(
        self,
        server_name: str,
        scopes: Optional[list] = None,
    ) -> Dict[str, Any]:
        """Add an MCP server."""
        from .commands.mcp import add_mcp_server_programmatic

        try:
            result = add_mcp_server_programmatic(
                server_name=server_name,
                scopes=scopes or [],
                config_dir=self.config_dir,
            )
            return {
                "success": True,
                "message": f"Added MCP server '{server_name}'",
                "server": server_name,
                "scopes": scopes or [],
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _exec_list_mcp_servers(self) -> Dict[str, Any]:
        """
        List available MCP servers (same as command-line).

        Connects to MCP Gateway to get real-time list of all registered servers.
        Falls back to cached list if gateway is unavailable.
        """
        from .mcp_gateway_client import MCPGatewayClient

        try:
            with MCPGatewayClient() as client:
                # Test connection first
                if not client.test_connection_sync():
                    # Gateway not reachable, return fallback list
                    return self._get_fallback_mcp_servers(
                        message="MCP Gateway not reachable. Showing cached list."
                    )

                # Get live server list from gateway
                servers = client.list_servers_sync(enabled_only=False)

                if not servers:
                    return {
                        "success": True,
                        "servers": [],
                        "count": 0,
                        "message": "No MCP servers registered in gateway.",
                    }

                # Format servers for LLM-friendly output
                formatted_servers = []
                for server in servers:
                    reg = server.get("registration", {})
                    formatted_servers.append({
                        "tool_id": reg.get("tool_id", ""),
                        "name": reg.get("name", ""),
                        "version": reg.get("version", ""),
                        "endpoint": reg.get("endpoint", "N/A"),
                        "enabled": server.get("enabled", False),
                        "call_count": server.get("call_count", 0),
                        "description": reg.get("tools", [{}])[0].get("description", "") if reg.get("tools") else "",
                        "requires_approval": "requires_approval" in reg.get("classification", []),
                    })

                return {
                    "success": True,
                    "servers": formatted_servers,
                    "count": len(formatted_servers),
                    "enabled_count": sum(1 for s in servers if s.get("enabled")),
                    "source": "mcp_gateway",
                }

        except Exception as e:
            logger.error(f"Error connecting to MCP Gateway: {e}")
            return self._get_fallback_mcp_servers(
                message=f"Error connecting to MCP Gateway: {str(e)}. Showing cached list."
            )

    def _get_fallback_mcp_servers(self, message: str = "") -> Dict[str, Any]:
        """
        Get fallback MCP server list when gateway is unavailable.

        This is a minimal cached list of common servers.
        For accurate data, ensure MCP Gateway is running.
        """
        fallback_servers = [
            {"name": "filesystem", "description": "Local file operations", "requires_approval": False},
            {"name": "github", "description": "GitHub API integration", "requires_approval": False},
            {"name": "postgres", "description": "Database queries", "requires_approval": False},
            {"name": "web-search", "description": "Web search capabilities", "requires_approval": False},
            {"name": "rag", "description": "Document retrieval", "requires_approval": False},
        ]
        return {
            "success": True,
            "servers": fallback_servers,
            "count": len(fallback_servers),
            "source": "cached_fallback",
            "warning": message or "Using cached server list. Start MCP Gateway for real-time data.",
        }

    def _exec_create_manifest(
        self,
        name: str,
        description: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a workflow manifest."""
        from .commands.manifest import create_manifest_programmatic

        try:
            result = create_manifest_programmatic(
                name=name,
                description=description or f"Workflow: {name}",
                config_dir=self.config_dir,
            )
            return {
                "success": True,
                "message": f"Created manifest '{name}'",
                "manifest_path": result.get("manifest_path", f"manifests/{name}.yaml"),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _exec_validate_manifest(self, manifest_file: str) -> Dict[str, Any]:
        """Validate a workflow manifest."""
        from .commands.manifest import validate_manifest_programmatic

        try:
            result = validate_manifest_programmatic(
                manifest_file=manifest_file,
                config_dir=self.config_dir,
            )
            return {
                "success": True,
                "message": f"Manifest '{manifest_file}' is valid",
                "manifest_file": manifest_file,
                "warnings": result.get("warnings", []),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _exec_run_project(self) -> Dict[str, Any]:
        """Run the current project."""
        from .commands.runtime import run_project_programmatic

        try:
            result = run_project_programmatic(config_dir=self.config_dir)
            return {
                "success": True,
                "message": "Project is running in development mode",
                "services": result.get("services", []),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _exec_show_status(self) -> Dict[str, Any]:
        """Show project status."""
        from .commands.runtime import show_status_programmatic

        try:
            result = show_status_programmatic(config_dir=self.config_dir)
            return {
                "success": True,
                "services": result.get("services", []),
                "agents_running": result.get("agents_running", 0),
                "memory_usage": result.get("memory_usage", "N/A"),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _exec_show_logs(self, agent_name: Optional[str] = None) -> Dict[str, Any]:
        """Show logs."""
        from .commands.runtime import show_logs_programmatic

        try:
            result = show_logs_programmatic(
                agent_name=agent_name,
                config_dir=self.config_dir,
            )
            return {
                "success": True,
                "logs": result.get("logs", []),
                "agent": agent_name or "all",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _exec_show_help(self, topic: Optional[str] = None) -> Dict[str, Any]:
        """Show help information."""
        help_content = {
            "agents": """
## Agents
Agents are specialized LLM-powered workers. Types:
- **research**: Gathers information from various sources
- **verify**: Validates claims and cross-references data
- **code**: Generates and modifies code
- **synthesis**: Combines and summarizes information

Create with: /agent new <name> --role <type>
            """,
            "skills": """
## Skills
Skills are deterministic, reusable operations with defined I/O schemas.
- Must have input and output JSON schemas
- Can have safety flags (pii_risk, external_call, side_effect)
- Executed in sandboxed Code Executor

Create with: /skill new <name>
            """,
            "manifests": """
## Manifests
YAML workflow definitions that orchestrate multiple agents:
- Define steps with roles and capabilities
- Configure memory persistence
- Set compaction strategies
- Define tool access

Create with: /manifest new
Validate with: /manifest validate <file>
            """,
            "mcp": """
## MCP (Model Context Protocol)
Integration system for external tools:
- filesystem: Local file operations
- github: GitHub API integration
- postgres: Database queries
- slack: Messaging
- jira: Ticket management

Add with: /mcp add <server>
List with: /mcp list
            """,
        }

        if topic and topic.lower() in help_content:
            return {
                "success": True,
                "topic": topic,
                "content": help_content[topic.lower()],
            }

        return {
            "success": True,
            "topics": list(help_content.keys()),
            "commands": [
                "/init - Initialize new project",
                "/agent new <name> - Create agent",
                "/skill new <name> - Create skill",
                "/llm config|list|test - LLM configuration",
                "/llm set-params|show-params - Hyperparameters",
                "/mcp add|list <server> - MCP servers",
                "/manifest new|validate - Workflows",
                "/run - Run project",
                "/status - Show status",
                "/logs [agent] - View logs",
                "/help - This help",
                "/exit - Exit",
            ],
            "file_operations": [
                "file_read - Read file contents",
                "file_glob - Find files by pattern",
                "file_grep - Search for text in files",
                "file_write - Write file contents",
                "file_edit - Edit file (replace text)",
            ],
        }

    # ========================================
    # File Operation Tools (Claude Code-like)
    # ========================================

    def _exec_file_read(
        self,
        file_path: str,
        offset: Optional[int] = None,
        limit: int = 2000,
    ) -> Dict[str, Any]:
        """Read file contents with line numbers."""
        # Import the handler from prepackaged skills
        try:
            import sys
            skills_path = Path(self.project_root).parent / "code-exec" / "skills" / "prepackaged"
            if str(skills_path) not in sys.path:
                sys.path.insert(0, str(skills_path))

            from file_operations.file_read.handler import read_file

            result = read_file(file_path=file_path, offset=offset, limit=limit)

            # Track source for attribution
            if result.get("file_exists", False):
                tracker = get_source_tracker()
                line_range = None
                if offset:
                    end_line = offset + result.get("lines_read", limit)
                    line_range = f"{offset}-{end_line}"
                tracker.add_source(
                    source_type=SourceType.FILE_READ,
                    location=file_path,
                    description=f"Read {result.get('lines_read', 0)} lines",
                    line_range=line_range,
                )

            return {"success": result.get("file_exists", False), **result}
        except ImportError:
            # Fallback: direct implementation
            return self._file_read_fallback(file_path, offset, limit)

    def _file_read_fallback(
        self,
        file_path: str,
        offset: Optional[int] = None,
        limit: int = 2000,
    ) -> Dict[str, Any]:
        """Fallback file read implementation."""
        path = Path(file_path)

        if not path.exists():
            return {
                "success": False,
                "content": f"File not found: {file_path}",
                "file_exists": False,
            }

        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()

            total_lines = len(lines)
            start_idx = (offset - 1) if offset else 0
            start_idx = max(0, min(start_idx, total_lines))
            end_idx = min(start_idx + limit, total_lines)

            selected_lines = lines[start_idx:end_idx]
            max_width = len(str(end_idx))

            numbered = ""
            for i, line in enumerate(selected_lines, start=start_idx + 1):
                if len(line) > 2000:
                    line = line[:2000] + "...[truncated]\n"
                numbered += f"{i:>{max_width + 1}}->{line}"

            # Track source for attribution
            tracker = get_source_tracker()
            line_range = f"{start_idx + 1}-{end_idx}" if offset else None
            tracker.add_source(
                source_type=SourceType.FILE_READ,
                location=file_path,
                description=f"Read {len(selected_lines)} lines",
                line_range=line_range,
            )

            return {
                "success": True,
                "content": numbered,
                "file_exists": True,
                "total_lines": total_lines,
                "lines_read": len(selected_lines),
                "file_size_bytes": path.stat().st_size,
            }
        except Exception as e:
            return {"success": False, "error": str(e), "file_exists": True}

    def _exec_file_glob(
        self,
        pattern: str,
        path: Optional[str] = None,
        max_results: int = 100,
    ) -> Dict[str, Any]:
        """Find files matching glob pattern."""
        base_path = Path(path) if path else Path.cwd()

        if not base_path.exists():
            return {"success": False, "files": [], "total_matches": 0, "truncated": False}

        try:
            matches = list(base_path.glob(pattern))
            matches.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)

            total = len(matches)
            truncated = total > max_results
            files = [str(m.absolute()) for m in matches[:max_results]]

            # Track source for attribution
            tracker = get_source_tracker()
            tracker.add_source(
                source_type=SourceType.FILE_SEARCH,
                location=str(base_path),
                description=f"Glob '{pattern}' → {total} files",
            )

            return {
                "success": True,
                "files": files,
                "total_matches": total,
                "truncated": truncated,
            }
        except Exception as e:
            return {"success": False, "error": str(e), "files": [], "total_matches": 0}

    def _exec_file_grep(
        self,
        pattern: str,
        path: Optional[str] = None,
        glob: Optional[str] = None,
        case_insensitive: bool = False,
        context_lines: int = 0,
        max_results: int = 100,
        output_mode: str = "files_with_matches",
    ) -> Dict[str, Any]:
        """Search for text pattern in files."""
        import re

        base_path = Path(path) if path else Path.cwd()

        if not base_path.exists():
            return {
                "success": False,
                "matches": [],
                "files_with_matches": [],
                "total_matches": 0,
                "files_searched": 0,
            }

        flags = re.IGNORECASE if case_insensitive else 0
        try:
            regex = re.compile(pattern, flags)
        except re.error as e:
            return {"success": False, "error": f"Invalid regex: {e}"}

        matches = []
        files_with_matches = []
        total_matches = 0
        files_searched = 0

        if base_path.is_file():
            files_to_search = [base_path]
        else:
            glob_pattern = glob if glob else "**/*"
            files_to_search = [f for f in base_path.glob(glob_pattern) if f.is_file()]

        for file_path in files_to_search:
            if file_path.stat().st_size > 10 * 1024 * 1024:
                continue

            try:
                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    lines = f.readlines()
            except (PermissionError, OSError):
                continue

            files_searched += 1
            file_has_match = False

            for i, line in enumerate(lines):
                if regex.search(line):
                    file_has_match = True
                    total_matches += 1

                    if output_mode == "content" and len(matches) < max_results:
                        ctx_before = []
                        ctx_after = []
                        if context_lines > 0:
                            start = max(0, i - context_lines)
                            ctx_before = [lines[j].rstrip() for j in range(start, i)]
                            end = min(len(lines), i + context_lines + 1)
                            ctx_after = [lines[j].rstrip() for j in range(i + 1, end)]

                        matches.append({
                            "file": str(file_path.absolute()),
                            "line_number": i + 1,
                            "content": line.rstrip(),
                            "context_before": ctx_before,
                            "context_after": ctx_after,
                        })

            if file_has_match:
                files_with_matches.append(str(file_path.absolute()))

        # Track source for attribution
        tracker = get_source_tracker()
        tracker.add_source(
            source_type=SourceType.FILE_SEARCH,
            location=str(base_path),
            description=f"Grep '{pattern}' → {total_matches} matches in {len(files_with_matches)} files",
        )

        return {
            "success": True,
            "matches": matches[:max_results] if output_mode == "content" else [],
            "files_with_matches": files_with_matches[:max_results],
            "total_matches": total_matches,
            "files_searched": files_searched,
            "truncated": total_matches > max_results,
        }

    def _exec_file_write(
        self,
        file_path: str,
        content: str,
        create_directories: bool = False,
    ) -> Dict[str, Any]:
        """Write content to a file."""
        path = Path(file_path)
        created_dirs = []

        try:
            if create_directories and not path.parent.exists():
                dirs_to_create = []
                current = path.parent
                while not current.exists():
                    dirs_to_create.append(current)
                    current = current.parent
                path.parent.mkdir(parents=True, exist_ok=True)
                created_dirs = [str(d) for d in reversed(dirs_to_create)]
            elif not path.parent.exists():
                return {
                    "success": False,
                    "error": f"Parent directory does not exist: {path.parent}",
                    "file_path": file_path,
                }

            with open(path, "w", encoding="utf-8") as f:
                f.write(content)

            # Track source for attribution
            tracker = get_source_tracker()
            tracker.add_source(
                source_type=SourceType.FILE_READ,  # Using FILE_READ for write ops too
                location=str(path.absolute()),
                description=f"Wrote {path.stat().st_size} bytes",
            )

            return {
                "success": True,
                "file_path": str(path.absolute()),
                "bytes_written": path.stat().st_size,
                "created_directories": created_dirs,
            }
        except Exception as e:
            return {"success": False, "error": str(e), "file_path": file_path}

    def _exec_file_edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> Dict[str, Any]:
        """Edit a file by replacing text."""
        import difflib

        path = Path(file_path)

        if not path.exists():
            return {
                "success": False,
                "error": f"File not found: {file_path}",
                "file_path": file_path,
            }

        try:
            with open(path, "r", encoding="utf-8") as f:
                original = f.read()

            occurrences = original.count(old_string)

            if occurrences == 0:
                return {
                    "success": False,
                    "error": f"String not found in file",
                    "file_path": file_path,
                }

            if not replace_all and occurrences > 1:
                return {
                    "success": False,
                    "error": f"String found {occurrences} times. Use replace_all=true or provide more context.",
                    "file_path": file_path,
                }

            if replace_all:
                new_content = original.replace(old_string, new_string)
                replacements = occurrences
            else:
                new_content = original.replace(old_string, new_string, 1)
                replacements = 1

            # Generate diff
            diff = difflib.unified_diff(
                original.splitlines(keepends=True),
                new_content.splitlines(keepends=True),
                fromfile=f"a/{path.name}",
                tofile=f"b/{path.name}",
            )
            diff_preview = "".join(list(diff)[:30])

            with open(path, "w", encoding="utf-8") as f:
                f.write(new_content)

            # Track source for attribution
            tracker = get_source_tracker()
            tracker.add_source(
                source_type=SourceType.FILE_READ,  # Using FILE_READ for edit ops too
                location=str(path.absolute()),
                description=f"Edited: {replacements} replacement(s)",
            )

            return {
                "success": True,
                "file_path": str(path.absolute()),
                "replacements_made": replacements,
                "diff_preview": diff_preview,
            }
        except Exception as e:
            return {"success": False, "error": str(e), "file_path": file_path}

    # ========================================
    # Code Execution Tools
    # ========================================

    def _exec_bash_exec(
        self,
        command: str,
        timeout_ms: int = 120000,
        working_directory: Optional[str] = None,
        environment: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Execute a bash command."""
        import subprocess
        import time

        timeout_seconds = timeout_ms / 1000.0
        env = os.environ.copy()
        if environment:
            env.update(environment)

        cwd = working_directory
        if cwd and not os.path.isdir(cwd):
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Working directory does not exist: {cwd}",
                "exit_code": 1,
            }

        start_time = time.time()

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                cwd=cwd,
                env=env,
            )

            execution_time_ms = (time.time() - start_time) * 1000

            stdout = result.stdout
            stderr = result.stderr

            # Truncate if too long
            if len(stdout) > 30000:
                stdout = stdout[:30000] + "\n... [output truncated]"
            if len(stderr) > 30000:
                stderr = stderr[:30000] + "\n... [output truncated]"

            # Track source for attribution
            tracker = get_source_tracker()
            # Truncate long commands for display
            cmd_display = command[:60] + "..." if len(command) > 60 else command
            tracker.add_source(
                source_type=SourceType.BASH_EXEC,
                location=cmd_display,
                description=f"Exit {result.returncode}, {round(execution_time_ms)}ms",
            )

            return {
                "success": result.returncode == 0,
                "stdout": stdout,
                "stderr": stderr,
                "exit_code": result.returncode,
                "execution_time_ms": round(execution_time_ms, 2),
                "timed_out": False,
            }

        except subprocess.TimeoutExpired:
            execution_time_ms = (time.time() - start_time) * 1000
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Command timed out after {timeout_seconds}s",
                "exit_code": 124,
                "execution_time_ms": round(execution_time_ms, 2),
                "timed_out": True,
            }
        except Exception as e:
            return {
                "success": False,
                "stdout": "",
                "stderr": str(e),
                "exit_code": 1,
            }

    def _exec_python_exec(
        self,
        code: str,
        timeout_seconds: int = 30,
        auto_install_packages: bool = True,
        allow_all_packages: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute Python code with safe package auto-installation.

        Args:
            code: Python code to execute
            timeout_seconds: Maximum execution time
            auto_install_packages: If True, auto-install missing packages
            allow_all_packages: If True, allow all packages (use with caution in production)

        Returns:
            Execution result with output, error, and package installation details
        """
        import subprocess
        import sys
        import tempfile
        import time
        import re

        start_time = time.time()
        packages_installed = []
        max_retries = 2  # Maximum attempts to install packages and retry

        try:
            # Initialize safe package manager
            from .safe_package_manager import SafePackageManager

            # Check environment variable for package installation policy
            auto_install_env = os.getenv("KAUTILYA_AUTO_INSTALL_PACKAGES", "true").lower()
            auto_install_packages = auto_install_packages and (auto_install_env == "true")

            package_manager = SafePackageManager(
                allow_all=allow_all_packages,
                max_install_time_seconds=60,
                audit_log_path=Path(self.config_dir) / "package_installs.log" if auto_install_packages else None,
            )

            # Attempt execution with retry on import errors
            for attempt in range(max_retries):
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    suffix=".py",
                    delete=False,
                    encoding="utf-8",
                ) as f:
                    f.write(code)
                    temp_file = f.name

                result = subprocess.run(
                    [sys.executable, temp_file],
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds,
                    env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                )

                execution_time_ms = (time.time() - start_time) * 1000

                output = result.stdout
                error = result.stderr

                # Clean up temp file
                try:
                    os.unlink(temp_file)
                except OSError:
                    pass

                # Check for import errors
                if result.returncode != 0 and error and "ModuleNotFoundError" in error:
                    # Extract missing module name
                    match = re.search(r"No module named ['\"]([^'\"]+)['\"]", error)
                    if match and auto_install_packages and attempt < max_retries - 1:
                        missing_module = match.group(1)

                        console.print(f"\n[yellow]⚠ Missing package detected:[/yellow] {missing_module}")
                        console.print(f"[cyan]→ Attempting safe installation...[/cyan]")

                        # Install the missing package
                        install_result = package_manager.install_package(missing_module)

                        if install_result.success:
                            packages_installed.append({
                                "package": install_result.package_name,
                                "version": install_result.version,
                                "time_ms": install_result.install_time_ms,
                            })
                            console.print(
                                f"[green]✓ Installed {install_result.package_name}=={install_result.version}[/green]"
                            )
                            console.print(f"[cyan]→ Retrying code execution...[/cyan]\n")
                            continue  # Retry execution
                        else:
                            console.print(f"[red]✗ Failed to install:[/red] {install_result.message}")

                            if len(output) > 30000:
                                output = output[:30000] + "\n... [output truncated]"

                            return {
                                "success": False,
                                "output": output,
                                "error": error,
                                "execution_time_ms": round(execution_time_ms, 2),
                                "packages_installed": packages_installed,
                                "package_install_failed": {
                                    "package": install_result.package_name,
                                    "reason": install_result.message,
                                },
                            }

                # Success or non-import error
                if len(output) > 30000:
                    output = output[:30000] + "\n... [output truncated]"

                return {
                    "success": result.returncode == 0,
                    "output": output,
                    "error": error if error else None,
                    "execution_time_ms": round(execution_time_ms, 2),
                    "packages_installed": packages_installed if packages_installed else None,
                }

            # Max retries exceeded
            return {
                "success": False,
                "output": output if 'output' in locals() else "",
                "error": "Maximum package installation retries exceeded",
                "execution_time_ms": round((time.time() - start_time) * 1000, 2),
                "packages_installed": packages_installed,
            }

        except subprocess.TimeoutExpired:
            execution_time_ms = (time.time() - start_time) * 1000
            return {
                "success": False,
                "output": "",
                "error": f"Execution timed out after {timeout_seconds}s",
                "execution_time_ms": round(execution_time_ms, 2),
                "packages_installed": packages_installed if packages_installed else None,
            }
        except Exception as e:
            return {
                "success": False,
                "output": "",
                "error": str(e),
                "packages_installed": packages_installed if packages_installed else None,
            }
        finally:
            try:
                if "temp_file" in locals():
                    os.unlink(temp_file)
            except OSError:
                pass

    def _exec_notebook_edit(
        self,
        notebook_path: str,
        new_source: str,
        cell_index: Optional[int] = None,
        cell_type: str = "code",
        edit_mode: str = "replace",
    ) -> Dict[str, Any]:
        """Edit a Jupyter notebook cell."""
        path = Path(notebook_path)

        if not path.exists():
            return {
                "success": False,
                "error": f"Notebook not found: {notebook_path}",
            }

        if path.suffix != ".ipynb":
            return {
                "success": False,
                "error": "File is not a Jupyter notebook (.ipynb)",
            }

        try:
            with open(path, "r", encoding="utf-8") as f:
                notebook = json.load(f)

            cells = notebook.get("cells", [])

            new_cell = {
                "cell_type": cell_type,
                "metadata": {},
                "source": new_source.split("\n") if new_source else [],
            }

            if cell_type == "code":
                new_cell["execution_count"] = None
                new_cell["outputs"] = []

            if edit_mode == "insert":
                insert_idx = cell_index if cell_index is not None else len(cells)
                insert_idx = min(insert_idx, len(cells))
                cells.insert(insert_idx, new_cell)
                affected_idx = insert_idx

            elif edit_mode == "delete":
                if cell_index is None or cell_index >= len(cells):
                    return {
                        "success": False,
                        "error": f"Invalid cell index: {cell_index}",
                    }
                del cells[cell_index]
                affected_idx = cell_index

            else:  # replace
                if cell_index is None:
                    cell_index = 0
                if cell_index >= len(cells):
                    return {
                        "success": False,
                        "error": f"Cell index {cell_index} out of range",
                    }
                cells[cell_index] = new_cell
                affected_idx = cell_index

            notebook["cells"] = cells

            with open(path, "w", encoding="utf-8") as f:
                json.dump(notebook, f, indent=1, ensure_ascii=False)

            return {
                "success": True,
                "notebook_path": str(path.absolute()),
                "cell_index": affected_idx,
                "total_cells": len(cells),
            }

        except json.JSONDecodeError as e:
            return {"success": False, "error": f"Invalid notebook JSON: {e}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ========================================
    # Web Search Tools (DuckDuckGo & Tavily)
    # ========================================

    def _exec_web_search(
        self,
        query: str,
        max_results: int = 5,
        provider: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Search the web using configured provider.

        Args:
            query: Search query
            max_results: Maximum number of results
            provider: Force specific provider ('duckduckgo' or 'tavily')
        """
        # Load web search configuration
        config = self._load_websearch_config()

        # Determine provider
        if provider:
            selected_provider = provider.lower()
        else:
            selected_provider = config.get("default_provider", "duckduckgo")

        # Route to appropriate implementation
        if selected_provider == "tavily":
            if not config.get("tavily_api_key"):
                return {
                    "success": False,
                    "error": "Tavily API key not configured. Use /websearch config to set it up.",
                    "results": [],
                }
            return self._exec_web_search_tavily(
                query=query,
                max_results=max_results,
                api_key=config["tavily_api_key"],
            )
        else:
            # Default to DuckDuckGo (free, no API key)
            return self._exec_web_search_duckduckgo(
                query=query,
                max_results=max_results,
            )

    def _exec_web_search_duckduckgo(
        self,
        query: str,
        max_results: int = 5,
    ) -> Dict[str, Any]:
        """Search the web using DuckDuckGo (free, no API key required)."""
        try:
            from ddgs import DDGS
        except ImportError as e:
            import sys
            return {
                "success": False,
                "error": f"DuckDuckGo search package not found. Install it with: uv pip install ddgs\nPython path: {sys.executable}\nImport error: {str(e)}",
                "results": [],
                "fix": "Run: cd /Users/paragpradhan/Projects/Agent\\ framework/agent-framework/tools/kautilya && source .venv/bin/activate && uv pip install ddgs",
            }

        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))

            formatted_results = []
            for r in results:
                formatted_results.append({
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", ""),
                })

            # Track source for transparency - include actual URLs found
            tracker = get_source_tracker()

            # Add main search as a source
            tracker.add_source(
                SourceType.WEB_SEARCH,
                "duckduckgo",
                f"Search: {query[:40]}{'...' if len(query) > 40 else ''}",
            )

            # Add individual result URLs as sources
            for r in formatted_results[:3]:  # Top 3 results
                url = r.get("url", "")
                title = r.get("title", "")[:30]
                if url:
                    # Extract domain for cleaner display
                    try:
                        from urllib.parse import urlparse
                        domain = urlparse(url).netloc
                    except Exception:
                        domain = url[:40]
                    tracker.add_source(
                        SourceType.WEB_FETCH,
                        domain,
                        title or url[:40],
                    )

            return {
                "success": True,
                "query": query,
                "provider": "duckduckgo",
                "results": formatted_results,
                "result_count": len(formatted_results),
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "query": query,
                "provider": "duckduckgo",
                "results": [],
            }

    def _exec_web_search_tavily(
        self,
        query: str,
        max_results: int = 5,
        api_key: str = "",
    ) -> Dict[str, Any]:
        """Search the web using Tavily (requires API key)."""
        try:
            from tavily import TavilyClient
        except ImportError:
            return {
                "success": False,
                "error": "tavily-python not installed. Install with: uv pip install tavily-python",
                "results": [],
            }

        if not api_key:
            return {
                "success": False,
                "error": "Tavily API key required. Configure with /websearch config",
                "results": [],
            }

        try:
            client = TavilyClient(api_key=api_key)
            response = client.search(query, max_results=max_results)

            results = response.get("results", [])
            formatted_results = []

            for r in results:
                formatted_results.append({
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "snippet": r.get("content", ""),
                    "score": r.get("score", 0),
                })

            # Track source for transparency - include actual URLs found
            tracker = get_source_tracker()

            # Add main search as a source
            tracker.add_source(
                SourceType.WEB_SEARCH,
                "tavily",
                f"Search: {query[:40]}{'...' if len(query) > 40 else ''}",
            )

            # Add individual result URLs as sources
            for r in formatted_results[:3]:  # Top 3 results
                url = r.get("url", "")
                title = r.get("title", "")[:30]
                if url:
                    try:
                        from urllib.parse import urlparse
                        domain = urlparse(url).netloc
                    except Exception:
                        domain = url[:40]
                    tracker.add_source(
                        SourceType.WEB_FETCH,
                        domain,
                        title or url[:40],
                    )

            return {
                "success": True,
                "query": query,
                "provider": "tavily",
                "results": formatted_results,
                "result_count": len(formatted_results),
                "answer": response.get("answer"),  # Tavily may provide a direct answer
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "query": query,
                "provider": "tavily",
                "results": [],
            }

    def _load_websearch_config(self) -> Dict[str, Any]:
        """Load web search configuration from .env and config files."""
        config = {
            "default_provider": "duckduckgo",  # Free by default
            "tavily_api_key": None,
        }

        # Load from environment
        tavily_key = os.getenv("TAVILY_API_KEY")
        if tavily_key:
            config["tavily_api_key"] = tavily_key

        # Load from config file
        config_file = Path(self.config_dir) / "websearch.json"
        if config_file.exists():
            try:
                with open(config_file, "r") as f:
                    file_config = json.load(f)
                    config.update(file_config)
            except (json.JSONDecodeError, IOError):
                pass

        # Check environment for default provider override
        env_provider = os.getenv("KAUTILYA_WEBSEARCH_PROVIDER")
        if env_provider:
            config["default_provider"] = env_provider.lower()

        return config

    def _save_websearch_config(self, config: Dict[str, Any]) -> None:
        """Save web search configuration to config file."""
        config_dir_path = Path(self.config_dir)
        config_dir_path.mkdir(parents=True, exist_ok=True)

        config_file = config_dir_path / "websearch.json"
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)

    def _exec_configure_websearch(
        self,
        provider: Optional[str] = None,
        tavily_api_key: Optional[str] = None,
        set_as_default: bool = False,
    ) -> Dict[str, Any]:
        """Configure web search provider."""
        config = self._load_websearch_config()

        # Update configuration
        if tavily_api_key:
            config["tavily_api_key"] = tavily_api_key
            console.print("[green]Tavily API key configured[/green]")

        if set_as_default and provider:
            if provider.lower() in ["duckduckgo", "tavily"]:
                config["default_provider"] = provider.lower()
                console.print(f"[green]Set {provider} as default provider[/green]")
            else:
                return {
                    "success": False,
                    "error": f"Unknown provider: {provider}. Use 'duckduckgo' or 'tavily'",
                }

        # Save configuration
        self._save_websearch_config(config)

        return {
            "success": True,
            "default_provider": config["default_provider"],
            "tavily_configured": bool(config.get("tavily_api_key")),
            "duckduckgo_available": True,  # Always available, no API key needed
        }

    def _exec_list_websearch_providers(self) -> Dict[str, Any]:
        """List available web search providers."""
        config = self._load_websearch_config()

        providers = [
            {
                "name": "duckduckgo",
                "description": "DuckDuckGo web search (free, no API key required)",
                "configured": True,
                "is_default": config["default_provider"] == "duckduckgo",
                "cost": "free",
            },
            {
                "name": "tavily",
                "description": "Tavily AI search (requires API key, more accurate)",
                "configured": bool(config.get("tavily_api_key")),
                "is_default": config["default_provider"] == "tavily",
                "cost": "paid",
            },
        ]

        return {
            "success": True,
            "providers": providers,
            "default_provider": config["default_provider"],
        }

    # ========================================
    # Deep Research Tool - Comprehensive Web Research
    # ========================================

    def _exec_deep_research(
        self,
        query: str,
        min_sources: Optional[int] = None,
        max_sources: Optional[int] = None,
        output_format: str = "markdown",
        search_depth: str = "standard",
        include_raw_content: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute comprehensive deep research on a query.

        This tool:
        1. Generates multiple search queries based on the topic
        2. Searches using web_search tool
        3. Ranks results by relevance
        4. Fetches full content using Firecrawl MCP
        5. Extracts key facts and synthesizes findings
        6. Returns structured data ready for LLM synthesis

        Args:
            query: The research question or topic
            min_sources: Minimum sources to fetch (default from env or 10)
            max_sources: Maximum sources to fetch (default from env or 15)
            output_format: Output format - "markdown", "json", or "summary"
            search_depth: Depth - "quick", "standard", or "thorough"
            include_raw_content: Whether to include raw fetched content

        Returns:
            Dict with success, sources, extracted_facts, synthesis_context
        """
        try:
            # Use SkillRegistry to find the skill dynamically
            skill_path = None
            if SkillRegistry is not None:
                registry = SkillRegistry()
                skill_meta = registry.get_skill("deep-research")
                if skill_meta:
                    skill_path = skill_meta.path.parent
                    logger.debug(f"Found deep-research skill via registry: {skill_path}")

            # Fallback to hardcoded path if not found via registry
            if skill_path is None:
                skill_path = Path(__file__).resolve().parent.parent.parent.parent / "code-exec" / "skills"
                logger.debug(f"Using fallback skill path: {skill_path}")

            # Add skills directory to path if needed
            if str(skill_path) not in sys.path:
                sys.path.insert(0, str(skill_path))

            from deep_research.handler import deep_research

            # Execute the skill, passing self as tool_executor
            result = deep_research(
                query=query,
                min_sources=min_sources,
                max_sources=max_sources,
                output_format=output_format,
                search_depth=search_depth,
                include_raw_content=include_raw_content,
                tool_executor=self,
            )

            # Track sources for transparency
            tracker = get_source_tracker()
            if result.get("success") and result.get("sources"):
                for source in result["sources"][:5]:  # Top 5 sources
                    try:
                        from urllib.parse import urlparse
                        domain = urlparse(source.get("url", "")).netloc
                        tracker.add_source(
                            SourceType.WEB_FETCH,
                            domain or "deep_research",
                            source.get("title", "")[:40] or "Research source",
                        )
                    except Exception:
                        pass

            return result

        except ImportError as e:
            logger.error(f"Failed to import deep_research skill: {e}")
            return {
                "success": False,
                "error": f"Deep research skill not found. Ensure code-exec/skills/deep_research/handler.py exists. Error: {str(e)}",
                "query": query,
                "sources": [],
                "source_count": 0,
            }
        except Exception as e:
            logger.error(f"Deep research failed: {e}")
            return {
                "success": False,
                "error": f"Deep research failed: {str(e)}",
                "query": query,
                "sources": [],
                "source_count": 0,
            }

    # ========================================
    # Reflective Agent Tools (PLAN -> EXECUTE -> VALIDATE -> REFINE)
    # ========================================

    def _exec_execute_with_reflection(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
        constraints: Optional[list] = None,
        max_iterations: int = 3,
        validation_strictness: str = "medium",
    ) -> Dict[str, Any]:
        """
        Execute a task using the reflective agent loop.

        This demonstrates the PLAN -> EXECUTE -> VALIDATE -> REFINE pattern.
        """
        import time
        from datetime import datetime

        start_time = time.time()
        iterations = []

        # Phase tracking
        phases_completed = []

        try:
            # PHASE 1: PLAN
            console.print("[cyan]PLAN[/cyan]: Analyzing task and creating execution plan...")
            phases_completed.append("plan")

            plan = self._create_plan(task, context, constraints)

            if plan["confidence"] < 0.5:
                console.print(
                    f"[yellow]Warning:[/yellow] Low confidence plan ({plan['confidence']:.2f})"
                )

            iteration = 1
            success = False
            final_result = None

            while iteration <= max_iterations and not success:
                console.print(f"\n[bold]Iteration {iteration}/{max_iterations}[/bold]")

                # PHASE 2: EXECUTE
                console.print("[cyan]EXECUTE[/cyan]: Running plan steps...")
                phases_completed.append(f"execute_{iteration}")

                execution_result = self._execute_plan(plan)

                # PHASE 3: VALIDATE
                console.print("[cyan]VALIDATE[/cyan]: Checking results...")
                phases_completed.append(f"validate_{iteration}")

                validation = self._validate_result(
                    task, execution_result, plan, validation_strictness
                )

                iterations.append({
                    "iteration": iteration,
                    "execution_result": execution_result,
                    "validation": validation,
                })

                if validation["is_valid"] and validation["score"] >= 0.7:
                    success = True
                    final_result = execution_result
                    console.print(
                        f"[green]SUCCESS[/green]: Validation passed "
                        f"(score: {validation['score']:.2f})"
                    )
                else:
                    # PHASE 4: REFINE
                    console.print(
                        f"[cyan]REFINE[/cyan]: Validation failed "
                        f"(score: {validation['score']:.2f}), adjusting approach..."
                    )
                    phases_completed.append(f"refine_{iteration}")

                    refinement = self._refine_approach(
                        task, plan, execution_result, validation
                    )

                    if refinement["action"] == "abort":
                        console.print("[red]ABORT[/red]: Cannot proceed further")
                        break

                    # Update plan for next iteration
                    if refinement.get("new_plan"):
                        plan = refinement["new_plan"]

                iteration += 1

            elapsed_time = time.time() - start_time

            return {
                "success": success,
                "task": task,
                "iterations_used": iteration - 1,
                "max_iterations": max_iterations,
                "phases_completed": phases_completed,
                "final_plan": plan,
                "final_result": final_result,
                "iterations_detail": iterations,
                "execution_time_seconds": round(elapsed_time, 2),
                "timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "task": task,
                "phases_completed": phases_completed,
            }

    def _exec_create_execution_plan(
        self,
        task: str,
        available_tools: Optional[list] = None,
        constraints: Optional[list] = None,
    ) -> Dict[str, Any]:
        """Create an execution plan for a task without executing it."""
        return self._create_plan(task, {"available_tools": available_tools}, constraints)

    def _create_plan(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
        constraints: Optional[list] = None,
    ) -> Dict[str, Any]:
        """Create an execution plan for a task."""
        # This is a simplified planning implementation
        # In production, this would use the LLM to generate a proper plan

        # Analyze task to determine approach
        task_lower = task.lower()

        steps = []

        # Detect task type and create appropriate steps
        if "file" in task_lower or "read" in task_lower:
            steps.append({
                "action": "Identify target files",
                "tool": "file_glob",
                "expected_output": "List of matching files",
            })
            steps.append({
                "action": "Read file contents",
                "tool": "file_read",
                "expected_output": "File contents with line numbers",
            })

        elif "search" in task_lower or "find" in task_lower:
            steps.append({
                "action": "Search for pattern in codebase",
                "tool": "file_grep",
                "expected_output": "Matching files and lines",
            })

        elif "run" in task_lower or "execute" in task_lower:
            steps.append({
                "action": "Execute command",
                "tool": "bash_exec",
                "expected_output": "Command output",
            })

        elif "create" in task_lower or "write" in task_lower:
            steps.append({
                "action": "Create or write file",
                "tool": "file_write",
                "expected_output": "File created successfully",
            })

        else:
            # Generic approach
            steps.append({
                "action": "Analyze and execute task",
                "tool": None,
                "expected_output": "Task completed",
            })

        # Calculate confidence based on task clarity
        confidence = 0.8
        if len(steps) == 1 and steps[0]["tool"] is None:
            confidence = 0.5  # Generic task, lower confidence

        return {
            "task_understanding": f"Execute: {task}",
            "approach": f"Using {len(steps)} step(s) to complete the task",
            "steps": steps,
            "success_criteria": [
                "All steps completed without errors",
                "Output matches expected results",
            ],
            "potential_risks": [
                "Task may require clarification",
                "Tools may not be available",
            ],
            "confidence": confidence,
            "constraints": constraints or [],
        }

    def _execute_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a plan and return results."""
        results = {
            "steps_completed": [],
            "steps_failed": [],
            "outputs": {},
            "errors": [],
        }

        for i, step in enumerate(plan.get("steps", [])):
            step_id = f"step_{i + 1}"

            try:
                if step.get("tool"):
                    # Would execute the tool here
                    # For demo, we'll simulate success
                    results["steps_completed"].append(step_id)
                    results["outputs"][step_id] = {
                        "action": step["action"],
                        "status": "simulated",
                    }
                else:
                    results["steps_completed"].append(step_id)
                    results["outputs"][step_id] = {
                        "action": step["action"],
                        "status": "completed",
                    }

            except Exception as e:
                results["steps_failed"].append(step_id)
                results["errors"].append(str(e))

        total_steps = len(plan.get("steps", []))
        results["success_rate"] = (
            len(results["steps_completed"]) / total_steps if total_steps > 0 else 0
        )

        return results

    def _validate_result(
        self,
        task: str,
        execution_result: Dict[str, Any],
        plan: Dict[str, Any],
        strictness: str,
    ) -> Dict[str, Any]:
        """Validate execution results against success criteria."""
        # Calculate validation score
        success_rate = execution_result.get("success_rate", 0)
        has_errors = len(execution_result.get("errors", [])) > 0

        # Base score on success rate
        score = success_rate

        # Adjust based on strictness
        if strictness == "high":
            # High strictness: penalize any failures
            if has_errors:
                score *= 0.5
        elif strictness == "low":
            # Low strictness: more lenient
            score = min(1.0, score + 0.2)

        # Determine validity
        is_valid = score >= 0.7 and not has_errors

        return {
            "is_valid": is_valid,
            "score": round(score, 2),
            "errors": execution_result.get("errors", []),
            "warnings": [],
            "suggestions": [
                "Consider adding more specific success criteria"
            ] if not is_valid else [],
            "criteria_met": {
                "all_steps_completed": execution_result.get("success_rate", 0) == 1.0,
                "no_errors": not has_errors,
            },
        }

    def _refine_approach(
        self,
        task: str,
        plan: Dict[str, Any],
        execution_result: Dict[str, Any],
        validation: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Analyze failures and determine refinement action."""
        errors = validation.get("errors", [])
        score = validation.get("score", 0)

        # Decide on action based on severity
        if score < 0.3:
            # Very low score - need significant change
            action = "modify"
            reasoning = "Low validation score requires different approach"
        elif errors:
            # Has errors - try to fix them
            action = "retry"
            reasoning = f"Retrying to address errors: {errors}"
        else:
            # Partial success - make adjustments
            action = "modify"
            reasoning = "Adjusting approach for better results"

        # Create modified plan if needed
        new_plan = None
        if action == "modify":
            new_plan = plan.copy()
            new_plan["approach"] = f"Modified approach: {reasoning}"
            new_plan["confidence"] = min(0.9, plan.get("confidence", 0.5) + 0.1)

        return {
            "action": action,
            "reasoning": reasoning,
            "new_plan": new_plan,
        }

    # ========================================
    # Enterprise Agent Tools (THINK -> PLAN -> APPROVE -> EXECUTE -> VALIDATE -> REFLECT)
    # ========================================

    def _init_enterprise_components(self) -> None:
        """Initialize enterprise agent components lazily."""
        if not hasattr(self, "_governance"):
            import sys
            service_path = Path(self.project_root).parent / "subagent-manager" / "service"
            if str(service_path) not in sys.path:
                sys.path.insert(0, str(service_path))

            try:
                from governance import GovernanceGate
                from provenance import ProvenanceTracker, ActorType
                from audit_logger import AuditLogger, AuditEvent, AuditPhase, MemoryAuditSink

                self._governance = GovernanceGate()
                self._provenance = ProvenanceTracker()
                self._audit = AuditLogger(sinks=[MemoryAuditSink()])
                self._ActorType = ActorType
                self._AuditEvent = AuditEvent
                self._AuditPhase = AuditPhase
                self._enterprise_initialized = True
            except ImportError as e:
                console.print(f"[yellow]Warning:[/yellow] Enterprise components not available: {e}")
                self._enterprise_initialized = False

    def _exec_enterprise_execute(
        self,
        task: str,
        user_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        constraints: Optional[list] = None,
        enable_governance: bool = True,
        auto_approve_low_risk: bool = True,
        max_iterations: int = 3,
    ) -> Dict[str, Any]:
        """
        Execute a task using the Enterprise Agent with full audit trail.

        THINK -> PLAN -> APPROVE -> EXECUTE -> VALIDATE -> REFLECT
        """
        import time
        from datetime import datetime
        from uuid import uuid4

        self._init_enterprise_components()

        if not getattr(self, "_enterprise_initialized", False):
            return {
                "success": False,
                "error": "Enterprise components not initialized",
            }

        start_time = time.time()
        execution_id = str(uuid4())[:12]
        user_id = user_id or "anonymous"

        # Start provenance chain
        chain = self._provenance.start_chain(execution_id=execution_id)

        # Log START
        self._audit.log_event(self._AuditEvent(
            phase=self._AuditPhase.START,
            execution_id=execution_id,
            task=task,
            user_id=user_id,
        ))

        phases_completed = []
        thinking_trace = None
        plan = None
        approval = None
        execution_result = None
        validation = None
        reflection = None

        try:
            # PHASE 1: THINK
            console.print("[bold cyan]THINK[/bold cyan]: Natural reasoning about the task...")
            phases_completed.append("think")

            thinking_trace = self._enterprise_think_internal(task, context)

            self._provenance.record(
                actor_id=execution_id,
                actor_type=self._ActorType.ENTERPRISE_AGENT,
                action="think",
                inputs_hash=self._provenance.hash_data({"task": task, "context": context}),
                outputs_hash=self._provenance.hash_data(thinking_trace),
            )

            self._audit.log_event(self._AuditEvent(
                phase=self._AuditPhase.THINK,
                execution_id=execution_id,
                reasoning_trace=thinking_trace.get("reasoning", ""),
                duration_ms=thinking_trace.get("duration_ms"),
            ))

            # PHASE 2: PLAN
            console.print("[bold cyan]PLAN[/bold cyan]: Creating structured execution plan...")
            phases_completed.append("plan")

            plan = self._enterprise_plan_internal(
                task, thinking_trace, context, constraints
            )

            self._provenance.record(
                actor_id=execution_id,
                actor_type=self._ActorType.ENTERPRISE_AGENT,
                action="plan",
                inputs_hash=self._provenance.hash_data(thinking_trace),
                outputs_hash=self._provenance.hash_data(plan),
            )

            self._audit.log_event(self._AuditEvent(
                phase=self._AuditPhase.PLAN,
                execution_id=execution_id,
                plan_id=plan.get("plan_id"),
                confidence=plan.get("confidence"),
                risk_level=plan.get("risk_level"),
                steps_count=len(plan.get("steps", [])),
            ))

            # PHASE 3: APPROVE
            if enable_governance:
                console.print("[bold cyan]APPROVE[/bold cyan]: Checking governance policies...")
                phases_completed.append("approve")

                approval = self._enterprise_approve_internal(
                    plan, auto_approve_low_risk, user_id
                )

                self._audit.log_event(self._AuditEvent(
                    phase=self._AuditPhase.APPROVE,
                    execution_id=execution_id,
                    approved=approval.get("approved"),
                    approver=approval.get("approver"),
                    violations=[v.get("description", str(v)) for v in approval.get("violations", [])],
                ))

                if not approval.get("approved"):
                    if approval.get("requires_human"):
                        console.print(
                            "[yellow]BLOCKED[/yellow]: Human approval required. "
                            f"Violations: {approval.get('violations')}"
                        )
                        return {
                            "success": False,
                            "blocked": True,
                            "requires_human_approval": True,
                            "execution_id": execution_id,
                            "phases_completed": phases_completed,
                            "violations": approval.get("violations"),
                            "plan": plan,
                        }
                    else:
                        console.print(f"[red]DENIED[/red]: {approval.get('reason')}")
                        return {
                            "success": False,
                            "blocked": True,
                            "execution_id": execution_id,
                            "phases_completed": phases_completed,
                            "reason": approval.get("reason"),
                        }

                console.print(f"[green]APPROVED[/green] by {approval.get('approver')}")
            else:
                approval = {"approved": True, "approver": "governance_disabled"}

            # PHASE 4: EXECUTE
            iteration = 1
            success = False

            while iteration <= max_iterations and not success:
                console.print(f"\n[bold]Iteration {iteration}/{max_iterations}[/bold]")
                console.print("[bold cyan]EXECUTE[/bold cyan]: Running plan steps...")
                phases_completed.append(f"execute_{iteration}")

                execution_result = self._execute_plan(plan)

                self._provenance.record(
                    actor_id=execution_id,
                    actor_type=self._ActorType.ENTERPRISE_AGENT,
                    action=f"execute_iteration_{iteration}",
                    inputs_hash=self._provenance.hash_data(plan),
                    outputs_hash=self._provenance.hash_data(execution_result),
                )

                # PHASE 5: VALIDATE
                console.print("[bold cyan]VALIDATE[/bold cyan]: Checking results...")
                phases_completed.append(f"validate_{iteration}")

                validation = self._validate_result(task, execution_result, plan, "medium")

                self._audit.log_event(self._AuditEvent(
                    phase=self._AuditPhase.VALIDATE,
                    execution_id=execution_id,
                    is_valid=validation.get("is_valid"),
                    score=validation.get("score"),
                    criteria_met=validation.get("criteria_met"),
                ))

                if validation.get("is_valid") and validation.get("score", 0) >= 0.7:
                    success = True
                    console.print(
                        f"[green]SUCCESS[/green]: Validation passed "
                        f"(score: {validation.get('score', 0):.2f})"
                    )
                else:
                    # PHASE 6: REFLECT
                    console.print("[bold cyan]REFLECT[/bold cyan]: Analyzing results...")
                    phases_completed.append(f"reflect_{iteration}")

                    reflection = self._enterprise_reflect_internal(
                        task, plan, execution_result, validation
                    )

                    self._audit.log_event(self._AuditEvent(
                        phase=self._AuditPhase.REFLECT,
                        execution_id=execution_id,
                        lessons_learned=reflection.get("lessons_learned"),
                        should_retry=reflection.get("should_retry"),
                        refinement_action=reflection.get("action"),
                    ))

                    if not reflection.get("should_retry"):
                        console.print("[yellow]STOPPING[/yellow]: Further retries not recommended")
                        break

                    if reflection.get("new_plan"):
                        plan = reflection["new_plan"]

                iteration += 1

            # Finalize
            elapsed_time = time.time() - start_time
            self._provenance.finalize_chain(chain.chain_id)

            self._audit.log_event(self._AuditEvent(
                phase=self._AuditPhase.COMPLETE,
                execution_id=execution_id,
                success=success,
                iterations_used=iteration - 1,
                total_duration_ms=elapsed_time * 1000,
            ))

            return {
                "success": success,
                "execution_id": execution_id,
                "task": task,
                "user_id": user_id,
                "phases_completed": phases_completed,
                "iterations_used": iteration - 1,
                "thinking_trace": thinking_trace,
                "final_plan": plan,
                "approval": approval,
                "final_result": execution_result,
                "validation": validation,
                "reflection": reflection,
                "execution_time_seconds": round(elapsed_time, 2),
                "provenance_chain_id": chain.chain_id,
                "timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            self._audit.log_event(self._AuditEvent(
                phase=self._AuditPhase.ERROR,
                execution_id=execution_id,
                error=str(e),
                error_type=type(e).__name__,
            ))
            return {
                "success": False,
                "execution_id": execution_id,
                "error": str(e),
                "phases_completed": phases_completed,
            }

    def _enterprise_think_internal(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Internal THINK phase - natural reasoning."""
        import time
        from uuid import uuid4

        start = time.time()

        # Analyze task naturally
        task_lower = task.lower()
        key_insights = []
        identified_risks = []
        proposed_approach = ""

        # Extract insights based on task content
        if "file" in task_lower:
            key_insights.append("Task involves file operations")
            if "write" in task_lower or "create" in task_lower:
                identified_risks.append("File modification may cause data loss")
        if "database" in task_lower or "sql" in task_lower:
            key_insights.append("Task involves database operations")
            identified_risks.append("Database changes may be irreversible")
        if "deploy" in task_lower or "production" in task_lower:
            key_insights.append("Task involves deployment")
            identified_risks.append("Production deployment requires careful review")
        if "delete" in task_lower or "remove" in task_lower:
            key_insights.append("Task involves deletion")
            identified_risks.append("Deletion is irreversible")
        if "search" in task_lower or "find" in task_lower:
            key_insights.append("Task is read-only search operation")

        # Determine approach
        if identified_risks:
            proposed_approach = "Proceed with caution, consider backups"
        else:
            proposed_approach = "Safe to proceed with standard execution"

        if not key_insights:
            key_insights.append("General task - analyze requirements carefully")

        duration_ms = (time.time() - start) * 1000

        return {
            "trace_id": str(uuid4())[:12],
            "reasoning": f"Analyzing task: {task}",
            "key_insights": key_insights,
            "identified_risks": identified_risks,
            "proposed_approach": proposed_approach,
            "duration_ms": round(duration_ms, 2),
        }

    def _enterprise_plan_internal(
        self,
        task: str,
        thinking: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        constraints: Optional[list] = None,
    ) -> Dict[str, Any]:
        """Internal PLAN phase - structured planning with risk assessment."""
        from uuid import uuid4

        # Use base planning
        base_plan = self._create_plan(task, context, constraints)

        # Assess risk level
        risks = thinking.get("identified_risks", [])
        if any("production" in r.lower() or "deploy" in r.lower() for r in risks):
            risk_level = "critical"
        elif any("irreversible" in r.lower() or "delete" in r.lower() for r in risks):
            risk_level = "high"
        elif risks:
            risk_level = "medium"
        else:
            risk_level = "low"

        # Determine if approval required
        requires_approval = risk_level in ("high", "critical")

        return {
            **base_plan,
            "plan_id": str(uuid4())[:12],
            "risk_level": risk_level,
            "requires_approval": requires_approval,
            "thinking_summary": thinking.get("proposed_approach"),
            "identified_risks": risks,
        }

    def _enterprise_approve_internal(
        self,
        plan: Dict[str, Any],
        auto_approve_low_risk: bool,
        user_id: str,
    ) -> Dict[str, Any]:
        """Internal APPROVE phase - governance gate."""
        from unittest.mock import MagicMock

        # Create mock plan object for governance gate
        mock_plan = MagicMock()
        mock_plan.plan_id = plan.get("plan_id", "unknown")
        mock_plan.risk_level = plan.get("risk_level", "medium")
        mock_plan.steps = [MagicMock(tool=s.get("tool"), action=s.get("action"))
                          for s in plan.get("steps", [])]
        mock_plan.model_dump.return_value = plan

        # Create mock config
        mock_config = MagicMock()
        mock_config.auto_approve_low_risk = auto_approve_low_risk
        mock_config.max_risk_level_auto_approve = "low" if auto_approve_low_risk else "none"

        # Run governance check
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        result = loop.run_until_complete(
            self._governance.evaluate(
                plan=mock_plan,
                context={"user_id": user_id},
                config=mock_config,
            )
        )

        return {
            "approved": result.approved,
            "approver": result.approver,
            "requires_human": result.requires_human,
            "violations": [v.model_dump() for v in result.violations],
            "warnings": [w.model_dump() for w in result.warnings],
            "reason": result.reason,
        }

    def _enterprise_reflect_internal(
        self,
        task: str,
        plan: Dict[str, Any],
        execution_result: Dict[str, Any],
        validation: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Internal REFLECT phase - post-execution analysis."""
        what_worked = []
        what_failed = []
        lessons_learned = []

        # Analyze execution
        steps_completed = execution_result.get("steps_completed", [])
        steps_failed = execution_result.get("steps_failed", [])
        errors = validation.get("errors", [])

        if steps_completed:
            what_worked.append(f"Completed {len(steps_completed)} step(s)")
        if steps_failed:
            what_failed.append(f"Failed {len(steps_failed)} step(s)")
        if errors:
            what_failed.extend(errors[:3])

        # Determine lessons
        if validation.get("score", 0) < 0.5:
            lessons_learned.append("Consider breaking task into smaller steps")
        if errors:
            lessons_learned.append("Add error handling for edge cases")

        # Decide on retry
        should_retry = (
            validation.get("score", 0) >= 0.3 and
            len(steps_failed) < len(steps_completed)
        )

        refinement = self._refine_approach(task, plan, execution_result, validation)

        return {
            "what_worked": what_worked,
            "what_failed": what_failed,
            "lessons_learned": lessons_learned,
            "should_retry": should_retry,
            "action": refinement.get("action"),
            "reasoning": refinement.get("reasoning"),
            "new_plan": refinement.get("new_plan"),
        }

    def _exec_enterprise_think(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
        thinking_budget_tokens: int = 1000,
    ) -> Dict[str, Any]:
        """Execute THINK phase standalone."""
        self._init_enterprise_components()

        result = self._enterprise_think_internal(task, context)
        return {
            "success": True,
            **result,
        }

    def _exec_governance_check(
        self,
        plan_description: str,
        tools_used: list,
        risk_level: str = "medium",
    ) -> Dict[str, Any]:
        """Check a plan against governance policies."""
        self._init_enterprise_components()

        if not getattr(self, "_enterprise_initialized", False):
            return {"success": False, "error": "Enterprise components not initialized"}

        plan = {
            "plan_id": "check",
            "description": plan_description,
            "risk_level": risk_level,
            "steps": [{"tool": t, "action": f"use {t}"} for t in tools_used],
        }

        result = self._enterprise_approve_internal(plan, True, "check")

        return {
            "success": True,
            "approved": result.get("approved"),
            "approver": result.get("approver"),
            "requires_human": result.get("requires_human"),
            "violations": result.get("violations"),
            "reason": result.get("reason"),
        }

    def _exec_provenance_record(
        self,
        action: str,
        actor_id: str,
        inputs: Optional[Dict[str, Any]] = None,
        outputs: Optional[Dict[str, Any]] = None,
        tool_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Record a provenance entry."""
        self._init_enterprise_components()

        if not getattr(self, "_enterprise_initialized", False):
            return {"success": False, "error": "Enterprise components not initialized"}

        record = self._provenance.record(
            actor_id=actor_id,
            actor_type=self._ActorType.ENTERPRISE_AGENT,
            action=action,
            inputs_hash=self._provenance.hash_data(inputs or {}),
            outputs_hash=self._provenance.hash_data(outputs or {}),
            tool_id=tool_id,
        )

        return {
            "success": True,
            "provenance_id": record.provenance_id,
            "inputs_hash": record.inputs_hash,
            "outputs_hash": record.outputs_hash,
            "timestamp": record.timestamp.isoformat(),
        }

    def _exec_audit_log(
        self,
        phase: str,
        message: str,
        execution_id: Optional[str] = None,
        severity: str = "info",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Log an audit event."""
        self._init_enterprise_components()

        if not getattr(self, "_enterprise_initialized", False):
            return {"success": False, "error": "Enterprise components not initialized"}

        # Map phase string to enum
        phase_map = {
            "start": self._AuditPhase.START,
            "think": self._AuditPhase.THINK,
            "plan": self._AuditPhase.PLAN,
            "approve": self._AuditPhase.APPROVE,
            "execute": self._AuditPhase.EXECUTE,
            "validate": self._AuditPhase.VALIDATE,
            "reflect": self._AuditPhase.REFLECT,
            "complete": self._AuditPhase.COMPLETE,
            "error": self._AuditPhase.ERROR,
        }

        audit_phase = phase_map.get(phase.lower(), self._AuditPhase.START)

        event = self._AuditEvent(
            phase=audit_phase,
            execution_id=execution_id,
            task=message,
            metadata=metadata or {},
        )

        event_id = self._audit.log_event(event)

        return {
            "success": True,
            "event_id": event_id,
            "phase": phase,
            "message": message,
        }

    def _exec_get_audit_trail(
        self,
        execution_id: str,
        include_provenance: bool = True,
        include_events: bool = True,
    ) -> Dict[str, Any]:
        """Get audit trail for an execution."""
        self._init_enterprise_components()

        if not getattr(self, "_enterprise_initialized", False):
            return {"success": False, "error": "Enterprise components not initialized"}

        result: Dict[str, Any] = {
            "success": True,
            "execution_id": execution_id,
        }

        if include_events:
            events = self._audit.get_events_by_execution(execution_id)
            result["events"] = [
                {
                    "event_id": e.event_id,
                    "phase": e.phase.value,
                    "timestamp": e.timestamp.isoformat(),
                    "task": e.task,
                }
                for e in events
            ]
            result["event_count"] = len(events)

        if include_provenance:
            chain = self._provenance.get_chain_by_execution(execution_id)
            if chain:
                result["provenance"] = {
                    "chain_id": chain.chain_id,
                    "record_count": len(chain.records),
                    "root_hash": chain.root_hash,
                    "finalized": chain.finalized_at is not None,
                }
            else:
                result["provenance"] = None

        return result

    # ========================================
    # Long Content Generation Tools
    # ========================================

    def _exec_smart_content_planner(
        self,
        description: str,
        target_format: str,
        estimated_size: str = "medium",
        output_file: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Plan long content generation by breaking it into manageable sections.

        This tool analyzes a content generation request and returns a structured
        plan with sections to generate separately, avoiding token limit issues.

        Args:
            description: Description of content to generate
            target_format: Target format (html, python, javascript, markdown, etc.)
            estimated_size: Size estimate (small, medium, large, very_large)
            output_file: Optional target output file path

        Returns:
            Content plan with sections, token estimates, and generation strategy
        """
        import uuid
        from datetime import datetime

        plan_id = str(uuid.uuid4())[:12]

        # Analyze content type and create appropriate sections
        format_lower = target_format.lower()
        desc_lower = description.lower()

        sections = []
        total_estimated_tokens = 0

        # HTML Dashboard/Web Page
        if format_lower in ("html", "htm") or "dashboard" in desc_lower or "webpage" in desc_lower:
            sections = [
                {
                    "section_id": "html_structure",
                    "name": "HTML Structure",
                    "description": "DOCTYPE, html, head, meta tags, title, CSS links",
                    "order": 1,
                    "estimated_tokens": 300,
                    "dependencies": [],
                },
                {
                    "section_id": "css_styles",
                    "name": "CSS Styles",
                    "description": "Embedded or linked CSS for layout and styling",
                    "order": 2,
                    "estimated_tokens": 800,
                    "dependencies": ["html_structure"],
                },
                {
                    "section_id": "html_body_structure",
                    "name": "Body Structure",
                    "description": "Main container, header, sidebar, content areas",
                    "order": 3,
                    "estimated_tokens": 500,
                    "dependencies": ["html_structure"],
                },
            ]

            # Add chart sections if Plotly/charts mentioned
            if "plotly" in desc_lower or "chart" in desc_lower or "graph" in desc_lower:
                sections.extend([
                    {
                        "section_id": "plotly_setup",
                        "name": "Plotly Setup",
                        "description": "Plotly.js library include and initialization",
                        "order": 4,
                        "estimated_tokens": 200,
                        "dependencies": ["html_structure"],
                    },
                    {
                        "section_id": "chart_containers",
                        "name": "Chart Containers",
                        "description": "HTML divs for each chart",
                        "order": 5,
                        "estimated_tokens": 300,
                        "dependencies": ["html_body_structure"],
                    },
                    {
                        "section_id": "chart_data",
                        "name": "Chart Data",
                        "description": "JavaScript data arrays for charts",
                        "order": 6,
                        "estimated_tokens": 1500,
                        "dependencies": ["plotly_setup"],
                    },
                    {
                        "section_id": "chart_configs",
                        "name": "Chart Configurations",
                        "description": "Plotly trace and layout configurations",
                        "order": 7,
                        "estimated_tokens": 2000,
                        "dependencies": ["chart_data"],
                    },
                    {
                        "section_id": "chart_render",
                        "name": "Chart Rendering",
                        "description": "Plotly.newPlot calls for each chart",
                        "order": 8,
                        "estimated_tokens": 500,
                        "dependencies": ["chart_configs", "chart_containers"],
                    },
                ])

            # Add table sections if DataTables mentioned
            if "datatable" in desc_lower or "table" in desc_lower:
                sections.extend([
                    {
                        "section_id": "datatables_setup",
                        "name": "DataTables Setup",
                        "description": "DataTables library include and CSS",
                        "order": 9,
                        "estimated_tokens": 200,
                        "dependencies": ["html_structure"],
                    },
                    {
                        "section_id": "table_structure",
                        "name": "Table Structure",
                        "description": "HTML table elements with headers",
                        "order": 10,
                        "estimated_tokens": 400,
                        "dependencies": ["html_body_structure"],
                    },
                    {
                        "section_id": "table_data",
                        "name": "Table Data",
                        "description": "Table row data (tbody content)",
                        "order": 11,
                        "estimated_tokens": 2000,
                        "dependencies": ["table_structure"],
                    },
                    {
                        "section_id": "datatables_init",
                        "name": "DataTables Initialization",
                        "description": "DataTables configuration and initialization",
                        "order": 12,
                        "estimated_tokens": 500,
                        "dependencies": ["datatables_setup", "table_structure"],
                    },
                ])

            # Add closing section
            sections.append({
                "section_id": "html_closing",
                "name": "HTML Closing",
                "description": "Closing tags and final scripts",
                "order": 99,
                "estimated_tokens": 100,
                "dependencies": [],
            })

        # Python Module/Script
        elif format_lower in ("python", "py"):
            sections = [
                {
                    "section_id": "imports",
                    "name": "Imports",
                    "description": "Import statements and dependencies",
                    "order": 1,
                    "estimated_tokens": 300,
                    "dependencies": [],
                },
                {
                    "section_id": "constants",
                    "name": "Constants & Config",
                    "description": "Module-level constants and configuration",
                    "order": 2,
                    "estimated_tokens": 200,
                    "dependencies": [],
                },
                {
                    "section_id": "models",
                    "name": "Data Models",
                    "description": "Pydantic models or dataclasses",
                    "order": 3,
                    "estimated_tokens": 800,
                    "dependencies": ["imports"],
                },
                {
                    "section_id": "utilities",
                    "name": "Utility Functions",
                    "description": "Helper functions and utilities",
                    "order": 4,
                    "estimated_tokens": 1000,
                    "dependencies": ["imports"],
                },
                {
                    "section_id": "core_logic",
                    "name": "Core Logic",
                    "description": "Main business logic and classes",
                    "order": 5,
                    "estimated_tokens": 2000,
                    "dependencies": ["models", "utilities"],
                },
                {
                    "section_id": "main",
                    "name": "Main Entry Point",
                    "description": "Main function and CLI handling",
                    "order": 6,
                    "estimated_tokens": 500,
                    "dependencies": ["core_logic"],
                },
            ]

        # JavaScript/TypeScript
        elif format_lower in ("javascript", "js", "typescript", "ts"):
            sections = [
                {
                    "section_id": "imports",
                    "name": "Imports",
                    "description": "Import/require statements",
                    "order": 1,
                    "estimated_tokens": 200,
                    "dependencies": [],
                },
                {
                    "section_id": "types",
                    "name": "Types & Interfaces",
                    "description": "TypeScript types and interfaces",
                    "order": 2,
                    "estimated_tokens": 500,
                    "dependencies": [],
                },
                {
                    "section_id": "constants",
                    "name": "Constants",
                    "description": "Configuration and constants",
                    "order": 3,
                    "estimated_tokens": 200,
                    "dependencies": [],
                },
                {
                    "section_id": "utilities",
                    "name": "Utility Functions",
                    "description": "Helper functions",
                    "order": 4,
                    "estimated_tokens": 800,
                    "dependencies": ["imports"],
                },
                {
                    "section_id": "components",
                    "name": "Components/Classes",
                    "description": "Main components or classes",
                    "order": 5,
                    "estimated_tokens": 2000,
                    "dependencies": ["utilities"],
                },
                {
                    "section_id": "exports",
                    "name": "Exports",
                    "description": "Module exports",
                    "order": 6,
                    "estimated_tokens": 100,
                    "dependencies": ["components"],
                },
            ]

        # Markdown/Blog Post
        elif format_lower in ("markdown", "md") or "blog" in desc_lower or "article" in desc_lower:
            sections = [
                {
                    "section_id": "frontmatter",
                    "name": "Frontmatter",
                    "description": "YAML frontmatter with metadata",
                    "order": 1,
                    "estimated_tokens": 100,
                    "dependencies": [],
                },
                {
                    "section_id": "introduction",
                    "name": "Introduction",
                    "description": "Title, hook, and introduction",
                    "order": 2,
                    "estimated_tokens": 500,
                    "dependencies": [],
                },
                {
                    "section_id": "body_sections",
                    "name": "Body Sections",
                    "description": "Main content sections",
                    "order": 3,
                    "estimated_tokens": 2000,
                    "dependencies": ["introduction"],
                },
                {
                    "section_id": "examples",
                    "name": "Examples & Code",
                    "description": "Code examples and demonstrations",
                    "order": 4,
                    "estimated_tokens": 1500,
                    "dependencies": ["body_sections"],
                },
                {
                    "section_id": "conclusion",
                    "name": "Conclusion",
                    "description": "Summary and call to action",
                    "order": 5,
                    "estimated_tokens": 300,
                    "dependencies": ["body_sections"],
                },
            ]

        # Generic fallback
        else:
            sections = [
                {
                    "section_id": "header",
                    "name": "Header",
                    "description": "File header and initial content",
                    "order": 1,
                    "estimated_tokens": 300,
                    "dependencies": [],
                },
                {
                    "section_id": "main_content",
                    "name": "Main Content",
                    "description": "Primary content",
                    "order": 2,
                    "estimated_tokens": 2000,
                    "dependencies": ["header"],
                },
                {
                    "section_id": "footer",
                    "name": "Footer",
                    "description": "Closing content",
                    "order": 3,
                    "estimated_tokens": 200,
                    "dependencies": ["main_content"],
                },
            ]

        # Adjust token estimates based on size
        size_multipliers = {
            "small": 0.5,
            "medium": 1.0,
            "large": 2.0,
            "very_large": 3.5,
        }
        multiplier = size_multipliers.get(estimated_size.lower(), 1.0)

        for section in sections:
            section["estimated_tokens"] = int(section["estimated_tokens"] * multiplier)
            total_estimated_tokens += section["estimated_tokens"]

        # Sort sections by order
        sections.sort(key=lambda x: x["order"])

        # Determine generation strategy
        if total_estimated_tokens <= 2000:
            strategy = "single_generation"
            strategy_description = "Content is small enough to generate in one shot"
        elif total_estimated_tokens <= 8000:
            strategy = "chunked_generation"
            strategy_description = "Generate sections sequentially using section_generate tool"
        else:
            strategy = "streaming_generation"
            strategy_description = "Use streaming file writer for very large content"

        # Store plan for reference
        plan = {
            "plan_id": plan_id,
            "description": description,
            "target_format": target_format,
            "output_file": output_file,
            "estimated_size": estimated_size,
            "total_estimated_tokens": total_estimated_tokens,
            "strategy": strategy,
            "strategy_description": strategy_description,
            "sections": sections,
            "section_count": len(sections),
            "created_at": datetime.utcnow().isoformat(),
            "status": "planned",
        }

        # Save plan to temp storage for reference
        plans_dir = Path(self.config_dir) / "content_plans"
        plans_dir.mkdir(parents=True, exist_ok=True)
        plan_file = plans_dir / f"{plan_id}.json"

        with open(plan_file, "w") as f:
            json.dump(plan, f, indent=2)

        return {
            "success": True,
            "plan_id": plan_id,
            "strategy": strategy,
            "strategy_description": strategy_description,
            "total_estimated_tokens": total_estimated_tokens,
            "section_count": len(sections),
            "sections": sections,
            "recommended_approach": (
                f"Use '{strategy}' approach. "
                f"{'Generate all at once.' if strategy == 'single_generation' else ''}"
                f"{'Call section_generate for each section in order.' if strategy == 'chunked_generation' else ''}"
                f"{'Use streaming_file_write for continuous generation.' if strategy == 'streaming_generation' else ''}"
            ),
            "output_file": output_file,
        }

    def _exec_section_generate(
        self,
        plan_id: str,
        section_id: str,
        content: str,
        file_path: Optional[str] = None,
        mode: str = "append",
    ) -> Dict[str, Any]:
        """
        Generate and write a specific section of planned content.

        This tool writes a section of content to a file, tracking progress
        against a content plan created by smart_content_planner.

        Args:
            plan_id: ID of the content plan
            section_id: ID of the section being generated
            content: The generated content for this section
            file_path: Target file path (overrides plan's output_file)
            mode: Write mode - 'append', 'prepend', or 'replace'

        Returns:
            Result with section status and overall progress
        """
        from datetime import datetime

        # Load the plan
        plans_dir = Path(self.config_dir) / "content_plans"
        plan_file = plans_dir / f"{plan_id}.json"

        if not plan_file.exists():
            return {
                "success": False,
                "error": f"Plan {plan_id} not found. Create a plan first with smart_content_planner.",
            }

        with open(plan_file, "r") as f:
            plan = json.load(f)

        # Find the section
        section = None
        section_index = -1
        for i, s in enumerate(plan.get("sections", [])):
            if s["section_id"] == section_id:
                section = s
                section_index = i
                break

        if section is None:
            return {
                "success": False,
                "error": f"Section {section_id} not found in plan {plan_id}",
                "available_sections": [s["section_id"] for s in plan.get("sections", [])],
            }

        # Determine file path
        target_file = file_path or plan.get("output_file")
        if not target_file:
            return {
                "success": False,
                "error": "No file path specified. Provide file_path or set output_file in plan.",
            }

        target_path = Path(target_file)

        # Check dependencies
        dependencies = section.get("dependencies", [])
        completed_sections = [
            s["section_id"] for s in plan.get("sections", [])
            if s.get("status") == "completed"
        ]

        unmet_dependencies = [d for d in dependencies if d not in completed_sections]
        if unmet_dependencies and mode == "append":
            console.print(
                f"[yellow]Warning:[/yellow] Unmet dependencies: {unmet_dependencies}"
            )

        # Write content based on mode
        try:
            if mode == "replace" or not target_path.exists():
                # Create new file or replace
                target_path.parent.mkdir(parents=True, exist_ok=True)
                with open(target_path, "w", encoding="utf-8") as f:
                    f.write(content)
                write_mode_used = "created" if not target_path.exists() else "replaced"

            elif mode == "append":
                # Append to existing file
                with open(target_path, "a", encoding="utf-8") as f:
                    f.write(content)
                write_mode_used = "appended"

            elif mode == "prepend":
                # Prepend to existing file
                existing_content = ""
                if target_path.exists():
                    with open(target_path, "r", encoding="utf-8") as f:
                        existing_content = f.read()

                with open(target_path, "w", encoding="utf-8") as f:
                    f.write(content + existing_content)
                write_mode_used = "prepended"

            else:
                return {
                    "success": False,
                    "error": f"Invalid mode: {mode}. Use 'append', 'prepend', or 'replace'.",
                }

            # Update section status in plan
            plan["sections"][section_index]["status"] = "completed"
            plan["sections"][section_index]["completed_at"] = datetime.utcnow().isoformat()
            plan["sections"][section_index]["bytes_written"] = len(content.encode("utf-8"))

            # Calculate progress
            completed_count = sum(
                1 for s in plan["sections"] if s.get("status") == "completed"
            )
            total_count = len(plan["sections"])
            progress_percent = (completed_count / total_count) * 100

            # Update plan status
            if completed_count == total_count:
                plan["status"] = "completed"
                plan["completed_at"] = datetime.utcnow().isoformat()
            else:
                plan["status"] = "in_progress"

            # Save updated plan
            with open(plan_file, "w") as f:
                json.dump(plan, f, indent=2)

            # Determine next section
            next_section = None
            for s in plan["sections"]:
                if s.get("status") != "completed":
                    # Check if dependencies are met
                    deps = s.get("dependencies", [])
                    if all(d in completed_sections + [section_id] for d in deps):
                        next_section = s
                        break

            result = {
                "success": True,
                "section_id": section_id,
                "section_name": section.get("name"),
                "file_path": str(target_path.absolute()),
                "write_mode": write_mode_used,
                "bytes_written": len(content.encode("utf-8")),
                "progress": {
                    "completed": completed_count,
                    "total": total_count,
                    "percent": round(progress_percent, 1),
                },
                "plan_status": plan["status"],
            }

            if next_section:
                result["next_section"] = {
                    "section_id": next_section["section_id"],
                    "name": next_section.get("name"),
                    "description": next_section.get("description"),
                }
            elif plan["status"] == "completed":
                result["message"] = "All sections completed! Content generation finished."

            return result

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "section_id": section_id,
            }

    def _exec_streaming_file_write(
        self,
        file_path: str,
        content_parts: List[str],
        mode: str = "overwrite",
        add_newlines: bool = True,
        progress_callback: bool = True,
    ) -> Dict[str, Any]:
        """
        Write large content to a file in streaming chunks.

        This tool handles very large content by writing multiple parts
        sequentially, avoiding memory issues and providing progress feedback.

        Args:
            file_path: Target file path
            content_parts: List of content strings to write sequentially
            mode: 'overwrite' to create new file, 'append' to add to existing
            add_newlines: Add newline between parts
            progress_callback: Show progress during writing

        Returns:
            Result with total bytes written and part statistics
        """
        from datetime import datetime

        path = Path(file_path)

        try:
            # Create parent directories if needed
            path.parent.mkdir(parents=True, exist_ok=True)

            # Determine write mode
            file_mode = "w" if mode == "overwrite" else "a"

            total_bytes = 0
            part_stats = []

            with open(path, file_mode, encoding="utf-8") as f:
                for i, part in enumerate(content_parts):
                    # Write part
                    f.write(part)

                    part_bytes = len(part.encode("utf-8"))
                    total_bytes += part_bytes

                    # Add newline separator if requested
                    if add_newlines and i < len(content_parts) - 1:
                        f.write("\n")
                        total_bytes += 1

                    # Track part stats
                    part_stats.append({
                        "part_index": i,
                        "bytes": part_bytes,
                        "cumulative_bytes": total_bytes,
                    })

                    # Show progress
                    if progress_callback:
                        progress_pct = ((i + 1) / len(content_parts)) * 100
                        console.print(
                            f"[dim]Writing part {i + 1}/{len(content_parts)} "
                            f"({progress_pct:.0f}%) - {part_bytes:,} bytes[/dim]"
                        )

                    # Flush to disk periodically for very large writes
                    if total_bytes > 100000:  # Every 100KB
                        f.flush()

            return {
                "success": True,
                "file_path": str(path.absolute()),
                "mode": mode,
                "total_bytes_written": total_bytes,
                "total_parts": len(content_parts),
                "part_statistics": part_stats,
                "file_size_bytes": path.stat().st_size,
                "completed_at": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "file_path": file_path,
                "parts_attempted": len(content_parts),
            }

    def _exec_get_content_plan(
        self,
        plan_id: str,
    ) -> Dict[str, Any]:
        """
        Retrieve a content generation plan by ID.

        Args:
            plan_id: The plan ID to retrieve

        Returns:
            The full plan with current status
        """
        plans_dir = Path(self.config_dir) / "content_plans"
        plan_file = plans_dir / f"{plan_id}.json"

        if not plan_file.exists():
            # List available plans
            available_plans = []
            if plans_dir.exists():
                available_plans = [f.stem for f in plans_dir.glob("*.json")]

            return {
                "success": False,
                "error": f"Plan {plan_id} not found",
                "available_plans": available_plans,
            }

        with open(plan_file, "r") as f:
            plan = json.load(f)

        # Calculate current progress
        completed = sum(1 for s in plan.get("sections", []) if s.get("status") == "completed")
        total = len(plan.get("sections", []))

        return {
            "success": True,
            "plan": plan,
            "progress": {
                "completed": completed,
                "total": total,
                "percent": round((completed / total) * 100, 1) if total > 0 else 0,
            },
        }

    def _exec_list_content_plans(self) -> Dict[str, Any]:
        """
        List all content generation plans.

        Returns:
            List of plans with their status
        """
        plans_dir = Path(self.config_dir) / "content_plans"

        if not plans_dir.exists():
            return {
                "success": True,
                "plans": [],
                "count": 0,
            }

        plans = []
        for plan_file in plans_dir.glob("*.json"):
            try:
                with open(plan_file, "r") as f:
                    plan = json.load(f)

                completed = sum(
                    1 for s in plan.get("sections", [])
                    if s.get("status") == "completed"
                )
                total = len(plan.get("sections", []))

                plans.append({
                    "plan_id": plan.get("plan_id"),
                    "description": plan.get("description", "")[:50] + "...",
                    "target_format": plan.get("target_format"),
                    "status": plan.get("status"),
                    "progress": f"{completed}/{total}",
                    "created_at": plan.get("created_at"),
                })
            except (json.JSONDecodeError, KeyError):
                continue

        return {
            "success": True,
            "plans": plans,
            "count": len(plans),
        }
