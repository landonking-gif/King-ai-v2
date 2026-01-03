"""
Dynamic Capability Registry for Kautilya.

Discovers and manages all available capabilities:
- Skills (document_qa, deep_research, etc.)
- Tools (file_read, bash_exec, etc.)
- MCP Servers (external integrations)

All capabilities are discovered at runtime, no hardcoded mappings.
"""

import importlib.util
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)

# Ensure skills directory is in Python path for relative imports
_project_root = Path(__file__).parents[4]  # capability_registry -> agent -> kautilya -> kautilya -> tools -> agent-framework
_skills_dir = _project_root / "code-exec" / "skills"
if _skills_dir.exists() and str(_skills_dir) not in sys.path:
    sys.path.insert(0, str(_skills_dir))
    logger.debug(f"Added skills directory to sys.path: {_skills_dir}")


@dataclass
class Capability:
    """Represents a single capability (skill, tool, or MCP server)."""

    name: str
    type: str  # "skill", "tool", "mcp"
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)  # JSON Schema for inputs
    when_to_use: str = ""  # Semantic description of when to use
    when_not_to_use: str = ""  # When NOT to use this capability
    handler: Optional[Callable] = None  # Callable handler for skills
    examples: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    requires_approval: bool = False
    safety_flags: List[str] = field(default_factory=list)

    # Tier 1: Lightweight metadata for fast pre-filtering
    short_description: str = ""  # One-line description for quick display
    category: str = ""  # Category (document_processing, file_operations, research, etc.)
    file_types: List[str] = field(default_factory=list)  # Supported file extensions
    intents: List[str] = field(default_factory=list)  # Matching intents
    priority: int = 5  # 1=highest, 10=lowest (for sorting within category)
    schema_path: Optional[str] = None  # Path to schema.json for skill input/output schemas

    def to_dict(self) -> Dict[str, Any]:
        """Convert capability to dictionary."""
        return {
            "name": self.name,
            "type": self.type,
            "description": self.description,
            "parameters": self.parameters,
            "when_to_use": self.when_to_use,
            "tags": self.tags,
        }

    def to_openai_tool(self) -> Dict[str, Any]:
        """Convert to OpenAI function tool format."""
        # Ensure parameters are in proper JSON Schema format
        params = self.parameters
        if params and "properties" not in params and "type" not in params:
            # Wrap raw parameter definitions in JSON Schema object format
            params = {
                "type": "object",
                "properties": params,
                "required": list(params.keys()),
            }
        elif not params:
            params = {"type": "object", "properties": {}}

        return {
            "type": "function",
            "function": {
                "name": f"{self.type}_{self.name}" if self.type == "skill" else self.name,
                "description": f"{self.description}\n\nWhen to use: {self.when_to_use}",
                "parameters": params,
            },
        }

    def matches_task(self, task_description: str) -> float:
        """
        Calculate how well this capability matches a task.

        Returns:
            Score from 0.0 to 1.0
        """
        task_lower = task_description.lower()
        score = 0.0

        # Check tags
        for tag in self.tags:
            if tag.lower() in task_lower:
                score += 0.2

        # Check description keywords
        desc_words = set(self.description.lower().split())
        task_words = set(task_lower.split())
        overlap = len(desc_words & task_words)
        if overlap > 0:
            score += min(0.3, overlap * 0.1)

        # Check when_to_use
        when_words = set(self.when_to_use.lower().split())
        when_overlap = len(when_words & task_words)
        if when_overlap > 0:
            score += min(0.3, when_overlap * 0.1)

        # Check examples
        for example in self.examples:
            if any(word in task_lower for word in example.lower().split()):
                score += 0.1

        return min(1.0, score)


class CapabilityRegistry:
    """
    Dynamic registry for all capabilities.

    Discovers:
    - Skills from skills/ directory
    - Tools from KAUTILYA_TOOLS
    - MCP servers from configuration
    """

    def __init__(
        self,
        skills_dir: Optional[Path] = None,
        tools_list: Optional[List[Dict]] = None,
        mcp_config_path: Optional[Path] = None,
    ):
        """
        Initialize capability registry.

        Args:
            skills_dir: Directory containing skills
            tools_list: List of tool definitions (KAUTILYA_TOOLS format)
            mcp_config_path: Path to MCP configuration
        """
        self.skills_dir = skills_dir or self._detect_skills_dir()
        self.tools_list = tools_list
        self.mcp_config_path = mcp_config_path

        self._capabilities: Dict[str, Capability] = {}
        self._skills_cache: Dict[str, Any] = {}
        self._cleared: bool = False  # Track if explicitly cleared

    def _detect_skills_dir(self) -> Optional[Path]:
        """Detect skills directory from project structure."""
        # Try relative to this file
        current = Path(__file__).resolve()

        # Navigate up to find code-exec/skills
        for _ in range(10):
            skills_path = current / "code-exec" / "skills"
            if skills_path.exists():
                return skills_path

            skills_path = current / "skills"
            if skills_path.exists():
                return skills_path

            current = current.parent

        # Try from environment
        env_path = os.getenv("KAUTILYA_SKILLS_DIR")
        if env_path and Path(env_path).exists():
            return Path(env_path)

        return None

    def discover_all(self) -> List[Capability]:
        """
        Discover all capabilities from all sources.

        Priority: Skills > Tools > MCP (skills with rich metadata take precedence)

        Returns:
            List of all discovered capabilities
        """
        self._capabilities.clear()

        # Track skill names to avoid overwriting with tools
        skill_names = set()

        # Discover skills FIRST (they have rich tier1/tier2 metadata)
        skill_caps = self._discover_skills()
        for cap in skill_caps:
            # Use the skill name directly (not prefixed)
            self._capabilities[cap.name] = cap
            skill_names.add(cap.name)

        # Discover tools (only add if not already covered by a skill)
        tool_caps = self._discover_tools()
        for cap in tool_caps:
            if cap.name not in skill_names:
                self._capabilities[cap.name] = cap
            else:
                # Tool has same name as skill - skip (skill has richer metadata)
                logger.debug(f"Skipping tool '{cap.name}' - skill with same name exists")

        # Discover MCP servers
        mcp_caps = self._discover_mcp_servers()
        for cap in mcp_caps:
            mcp_key = f"mcp_{cap.name}"
            if mcp_key not in self._capabilities:
                self._capabilities[mcp_key] = cap

        logger.info(
            f"Discovered {len(self._capabilities)} capabilities: "
            f"{len(skill_caps)} skills, {len(tool_caps)} tools, {len(mcp_caps)} MCP"
        )

        return list(self._capabilities.values())

    def _discover_skills(self) -> List[Capability]:
        """Discover all skills from skills directory."""
        capabilities = []

        if not self.skills_dir or not self.skills_dir.exists():
            logger.warning(f"Skills directory not found: {self.skills_dir}")
            return capabilities

        # Scan for skill.yaml files
        for skill_yaml in self.skills_dir.rglob("skill.yaml"):
            try:
                cap = self._load_skill(skill_yaml)
                if cap:
                    capabilities.append(cap)
            except Exception as e:
                logger.warning(f"Failed to load skill from {skill_yaml}: {e}")

        return capabilities

    def _load_skill(self, skill_yaml_path: Path) -> Optional[Capability]:
        """Load a skill from its YAML definition."""
        try:
            with open(skill_yaml_path) as f:
                skill_config = yaml.safe_load(f)

            skill_dir = skill_yaml_path.parent
            skill_name = skill_config.get("name", skill_dir.name)

            # Load schema if available
            schema_path = skill_dir / "schema.json"
            parameters = {"type": "object", "properties": {}, "required": []}
            if schema_path.exists():
                with open(schema_path) as f:
                    schema = json.load(f)
                    parameters = schema.get("input", parameters)

            # Load handler if available
            handler = None
            handler_path = skill_dir / "handler.py"
            if handler_path.exists():
                handler = self._load_handler(handler_path, skill_name)

            # Get description (may be multiline)
            description = skill_config.get("description", "")
            if isinstance(description, str):
                description = description.strip()

            tags = skill_config.get("tags", [])

            # Parse Tier 1 metadata (for fast pre-filtering)
            tier1 = skill_config.get("tier1", {})
            short_description = tier1.get("short_description", description[:100] if description else "")
            category = tier1.get("category", "general")
            file_types = tier1.get("file_types", [])
            intents = tier1.get("intents", [])
            priority = tier1.get("priority", 5)

            # Parse Tier 2 metadata (for LLM decision-making)
            when_to_use = skill_config.get("when_to_use", "")
            if not when_to_use:
                when_to_use = self._build_when_to_use(skill_name, description, tags)
            when_not_to_use = skill_config.get("when_not_to_use", "")

            # Build examples from skill.yaml or SKILL.md
            examples = skill_config.get("examples", [])
            if not examples:
                skill_md_path = skill_dir / "SKILL.md"
                if skill_md_path.exists():
                    examples = self._extract_examples_from_skillmd(skill_md_path)

            return Capability(
                name=skill_name,
                type="skill",
                description=description,
                parameters=parameters,
                when_to_use=when_to_use,
                when_not_to_use=when_not_to_use,
                handler=handler,
                examples=examples,
                tags=tags,
                requires_approval=skill_config.get("requires_approval", False),
                safety_flags=skill_config.get("safety_flags", []),
                # Tier 1 metadata
                short_description=short_description,
                category=category,
                file_types=file_types,
                intents=intents,
                priority=priority,
                # Schema path for dynamic tool conversion
                schema_path=str(schema_path) if schema_path.exists() else None,
            )

        except Exception as e:
            logger.error(f"Error loading skill from {skill_yaml_path}: {e}")
            return None

    def _load_handler(self, handler_path: Path, skill_name: str) -> Optional[Callable]:
        """Load skill handler function.

        Handles both simple skills and package-style skills with relative imports.
        """
        import importlib
        import inspect

        skill_path = handler_path.parent
        normalized_name = skill_name.replace("-", "_")

        try:
            # Check if skill has an __init__.py (is a package) or subpackages
            init_path = skill_path / "__init__.py"
            has_init = init_path.exists()
            has_subpackages = (skill_path / "components").exists() or (skill_path / "pipelines").exists()

            if has_init or has_subpackages:
                # Package-style skill - import as a package
                # Make sure parent directory is in path
                skill_parent = skill_path.parent
                if str(skill_parent) not in sys.path:
                    sys.path.insert(0, str(skill_parent))

                try:
                    # Try importing as a package first
                    pkg = importlib.import_module(normalized_name)
                    # Now import the handler from within the package
                    handler_module = importlib.import_module(f"{normalized_name}.handler")
                except ImportError:
                    # If that fails, create the package structure manually
                    pkg_spec = importlib.util.spec_from_file_location(
                        normalized_name,
                        init_path if has_init else handler_path,
                        submodule_search_locations=[str(skill_path)],
                    )
                    pkg = importlib.util.module_from_spec(pkg_spec)
                    pkg.__path__ = [str(skill_path)]
                    sys.modules[normalized_name] = pkg
                    if has_init:
                        pkg_spec.loader.exec_module(pkg)

                    # Now load the handler
                    handler_spec = importlib.util.spec_from_file_location(
                        f"{normalized_name}.handler",
                        handler_path,
                    )
                    handler_module = importlib.util.module_from_spec(handler_spec)
                    handler_module.__package__ = normalized_name
                    sys.modules[f"{normalized_name}.handler"] = handler_module
                    handler_spec.loader.exec_module(handler_module)
            else:
                # Simple skill - load directly
                spec = importlib.util.spec_from_file_location(
                    f"skill_{normalized_name}_handler",
                    handler_path,
                )
                if not spec or not spec.loader:
                    return None
                handler_module = importlib.util.module_from_spec(spec)
                sys.modules[f"skill_{normalized_name}_handler"] = handler_module
                spec.loader.exec_module(handler_module)

            # Look for handler function with skill name or default names
            handler_names = [normalized_name, skill_name, "handler", "run", "execute", "main"]
            for name in handler_names:
                if hasattr(handler_module, name):
                    handler = getattr(handler_module, name)
                    if inspect.isfunction(handler):
                        self._skills_cache[skill_name] = handler
                        return handler

            # Find the first actual function (not types or imports)
            for name in dir(handler_module):
                if not name.startswith("_"):
                    obj = getattr(handler_module, name)
                    if inspect.isfunction(obj):
                        self._skills_cache[skill_name] = obj
                        return obj

            return None

        except Exception as e:
            logger.debug(f"Could not load handler from {handler_path}: {e}")
            return None

    def _build_when_to_use(
        self,
        name: str,
        description: str,
        tags: List[str],
    ) -> str:
        """Build semantic 'when to use' description."""
        # Map skill names to usage scenarios
        skill_usage_map = {
            "document_qa": (
                "Use when user wants to extract information from documents "
                "(PDF, DOCX, XLSX, PPTX), answer questions about file content, "
                "or analyze document data. Look for @file references or mentions "
                "of reading/extracting/analyzing documents."
            ),
            "deep_research": (
                "Use when user wants to research a topic using web search, "
                "find current information, news, or comprehensive analysis. "
                "Look for words like 'research', 'find out', 'latest', 'current'."
            ),
            "text_summarize": (
                "Use when user wants to summarize text, get key points, "
                "or condense long content into shorter form."
            ),
            "extract_entities": (
                "Use when user wants to extract named entities like people, "
                "organizations, locations, dates from text."
            ),
            "code_review": (
                "Use when user wants code reviewed for quality, security, "
                "best practices, or bugs."
            ),
        }

        if name in skill_usage_map:
            return skill_usage_map[name]

        # Build from description and tags
        parts = []
        if description:
            parts.append(f"Use when: {description}")
        if tags:
            parts.append(f"Related to: {', '.join(tags)}")

        return " ".join(parts) if parts else f"Use the {name} skill when needed."

    def _extract_examples_from_skillmd(self, skill_md_path: Path) -> List[str]:
        """Extract usage examples from SKILL.md."""
        examples = []
        try:
            with open(skill_md_path) as f:
                content = f.read()

            # Look for example sections
            import re

            # Find code blocks after "Example" headers
            pattern = r"##.*[Ee]xample.*?\n```.*?\n(.*?)```"
            matches = re.findall(pattern, content, re.DOTALL)
            examples.extend([m.strip()[:200] for m in matches[:3]])

        except Exception:
            pass

        return examples

    def _discover_tools(self) -> List[Capability]:
        """Discover tools from KAUTILYA_TOOLS list."""
        capabilities = []

        # If tools_list not provided, try to import from llm_client
        tools_list = self.tools_list
        if tools_list is None:
            try:
                from ..llm_client import KAUTILYA_TOOLS

                tools_list = KAUTILYA_TOOLS
            except ImportError:
                logger.warning("Could not import KAUTILYA_TOOLS")
                return capabilities

        for tool in tools_list:
            if tool.get("type") != "function":
                continue

            func = tool.get("function", {})
            name = func.get("name", "")
            description = func.get("description", "")
            parameters = func.get("parameters", {})

            # Build when_to_use from tool type
            when_to_use = self._build_tool_when_to_use(name, description)

            # Determine tags from name
            tags = self._infer_tool_tags(name)

            capabilities.append(
                Capability(
                    name=name,
                    type="tool",
                    description=description,
                    parameters=parameters,
                    when_to_use=when_to_use,
                    handler=None,  # Tools are executed by tool_executor
                    tags=tags,
                )
            )

        return capabilities

    def _build_tool_when_to_use(self, name: str, description: str) -> str:
        """Build when_to_use for tools."""
        tool_usage_map = {
            "file_read": "Use to read file contents. Required before editing files.",
            "file_edit": "Use to modify existing files. Always read first.",
            "file_write": "Use to create new files or overwrite existing ones.",
            "file_glob": "Use to find files by pattern (e.g., **/*.py).",
            "file_grep": "Use to search for text patterns in files.",
            "bash_exec": "Use to run shell commands (git, npm, etc.).",
            "python_exec": "Use to execute Python code.",
            "web_search": "Use to search the web for current information.",
            "web_fetch": "Use to fetch content from a URL.",
        }

        return tool_usage_map.get(name, description[:100] if description else "")

    def _infer_tool_tags(self, name: str) -> List[str]:
        """Infer tags from tool name."""
        tag_map = {
            "file": ["file", "filesystem", "io"],
            "bash": ["shell", "command", "terminal"],
            "python": ["code", "execution", "python"],
            "web": ["internet", "search", "http"],
            "mcp": ["external", "integration"],
        }

        tags = []
        for prefix, tool_tags in tag_map.items():
            if name.startswith(prefix):
                tags.extend(tool_tags)

        return tags

    def _discover_mcp_servers(self) -> List[Capability]:
        """Discover MCP servers from configuration."""
        capabilities = []

        # Try to load from MCP config
        if self.mcp_config_path and self.mcp_config_path.exists():
            try:
                with open(self.mcp_config_path) as f:
                    config = yaml.safe_load(f)

                for server in config.get("servers", []):
                    for tool in server.get("tools", []):
                        capabilities.append(
                            Capability(
                                name=f"{server['name']}_{tool['name']}",
                                type="mcp",
                                description=tool.get("description", ""),
                                parameters=tool.get("parameters", {}),
                                when_to_use=tool.get("when_to_use", ""),
                                tags=["mcp", server["name"]],
                            )
                        )
            except Exception as e:
                logger.warning(f"Failed to load MCP config: {e}")

        return capabilities

    def get_capability(self, name: str) -> Optional[Capability]:
        """Get a capability by name."""
        # Ensure capabilities are discovered
        if not self._capabilities:
            self.discover_all()

        return self._capabilities.get(name)

    def get(self, name: str) -> Optional[Capability]:
        """Get a capability by name (alias for get_capability)."""
        # Try direct name first
        cap = self.get_capability(name)
        if cap:
            return cap

        # Try with skill_ prefix
        cap = self.get_capability(f"skill_{name}")
        if cap:
            return cap

        # Try without skill_ prefix
        if name.startswith("skill_"):
            cap = self.get_capability(name[6:])
            if cap:
                return cap

        return None

    def register(self, capability: Capability) -> None:
        """Register a new capability."""
        key = capability.name
        if capability.type == "skill" and not key.startswith("skill_"):
            key = f"skill_{key}"
        self._capabilities[key] = capability

    def clear(self) -> None:
        """Clear all registered capabilities."""
        self._capabilities.clear()
        self._skills_cache.clear()
        self._cleared = True  # Prevent auto-discovery after clear

    def get_all(self) -> List[Capability]:
        """Get all capabilities (alias for get_all_capabilities)."""
        return self.get_all_capabilities()

    def get_by_type(self, cap_type: str) -> List[Capability]:
        """Get all capabilities of a specific type."""
        if not self._capabilities and not self._cleared:
            self.discover_all()
        return [c for c in self._capabilities.values() if c.type == cap_type]

    def get_all_capabilities(self) -> List[Capability]:
        """Get all discovered capabilities."""
        if not self._capabilities and not self._cleared:
            self.discover_all()
        return list(self._capabilities.values())

    def get_skills(self) -> List[Capability]:
        """Get all skill capabilities."""
        return [c for c in self.get_all_capabilities() if c.type == "skill"]

    def get_tools(self) -> List[Capability]:
        """Get all tool capabilities."""
        return [c for c in self.get_all_capabilities() if c.type == "tool"]

    def match_capabilities(
        self,
        task_description: str,
        min_score: float = 0.2,
        max_results: int = 5,
    ) -> List[Capability]:
        """
        Find capabilities that match a task description.

        Args:
            task_description: Description of the task
            min_score: Minimum match score (0.0-1.0)
            max_results: Maximum number of results

        Returns:
            List of matching capabilities, sorted by relevance
        """
        if not self._capabilities:
            self.discover_all()

        scored = []
        for cap in self._capabilities.values():
            score = cap.matches_task(task_description)
            if score >= min_score:
                scored.append((score, cap))

        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)

        return [cap for _, cap in scored[:max_results]]

    def execute_skill(
        self,
        skill_name: str,
        inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Execute a skill by name.

        Args:
            skill_name: Name of the skill
            inputs: Input parameters

        Returns:
            Skill execution result
        """
        # Get capability
        cap_name = f"skill_{skill_name}" if not skill_name.startswith("skill_") else skill_name
        cap = self.get_capability(cap_name)

        if not cap:
            return {"success": False, "error": f"Skill not found: {skill_name}"}

        if not cap.handler:
            return {"success": False, "error": f"Skill has no handler: {skill_name}"}

        try:
            result = cap.handler(**inputs)
            return result if isinstance(result, dict) else {"success": True, "result": result}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def to_openai_tools(self) -> List[Dict[str, Any]]:
        """Convert all capabilities to OpenAI tool format."""
        if not self._capabilities:
            self.discover_all()

        return [cap.to_openai_tool() for cap in self._capabilities.values()]

    def format_capabilities_for_prompt(self) -> str:
        """Format capabilities for inclusion in system prompt."""
        if not self._capabilities:
            self.discover_all()

        lines = ["# Available Capabilities\n"]

        # Group by type
        by_type: Dict[str, List[Capability]] = {}
        for cap in self._capabilities.values():
            by_type.setdefault(cap.type, []).append(cap)

        for cap_type, caps in by_type.items():
            lines.append(f"\n## {cap_type.title()}s\n")
            for cap in caps:
                lines.append(f"### {cap.name}")
                lines.append(f"- Description: {cap.description}")
                lines.append(f"- When to use: {cap.when_to_use}")
                if cap.tags:
                    lines.append(f"- Tags: {', '.join(cap.tags)}")
                lines.append("")

        return "\n".join(lines)

    # ============================================================
    # TWO-STAGE SKILL SELECTION (Pre-filtering + Rich Context)
    # ============================================================

    # Intent patterns for pre-filtering
    INTENT_PATTERNS = {
        "extract": ["extract", "get", "pull", "identify", "find", "list", "retrieve"],
        "analyze": ["analyze", "analyse", "evaluate", "assess", "examine", "review"],
        "summarize": ["summarize", "summarise", "condense", "brief", "overview"],
        "research": ["research", "investigate", "look up", "find out", "search online", "search for", "search the", "web search", "look for", "find information"],
        "compare": ["compare", "contrast", "versus", "vs", "difference", "competitor"],
        "read": ["read", "show", "display", "view", "cat", "open", "see"],
        "write": ["write", "save", "create", "export", "output", "generate", "store"],
        "edit": ["edit", "modify", "change", "update", "fix"],
        "execute": ["run", "execute", "eval", "compute", "calculate"],
        "search": ["search", "query", "lookup", "google", "bing", "duckduckgo"],  # Web search
    }

    # File extension to category mapping
    FILE_TYPE_CATEGORIES = {
        "document_processing": [".pdf", ".docx", ".doc", ".xlsx", ".xls", ".pptx", ".ppt", ".html", ".htm"],
        "file_operations": [".txt", ".md", ".json", ".yaml", ".yml", ".toml", ".env", ".ini", ".cfg"],
        "code_execution": [".py", ".js", ".ts", ".java", ".go", ".rs", ".rb", ".sh", ".bash"],
    }

    def detect_intents(self, query: str) -> List[str]:
        """
        Detect user intents from query text.

        Args:
            query: User's query string

        Returns:
            List of detected intent types
        """
        query_lower = query.lower()
        detected = []

        for intent, patterns in self.INTENT_PATTERNS.items():
            for pattern in patterns:
                if pattern in query_lower:
                    detected.append(intent)
                    break  # One match per intent type is enough

        return detected

    # File type keywords that indicate document types (without extension dots)
    FILE_TYPE_KEYWORDS = {
        ".pdf": ["pdf", "pdf report", "pdf file", "pdf document"],
        ".docx": ["word", "word file", "word document", "docx"],
        ".xlsx": ["excel", "excel file", "spreadsheet", "xlsx"],
        ".pptx": ["powerpoint", "ppt", "presentation", "pptx"],
        ".csv": ["csv", "csv file"],
        ".html": ["html", "webpage", "web page"],
    }

    def detect_file_extensions(self, query: str) -> List[str]:
        """
        Detect file extensions mentioned in query.

        Handles both:
        1. Explicit extensions: ".pdf", "file.xlsx"
        2. File type keywords: "pdf report", "excel file", "word document"

        Args:
            query: User's query string

        Returns:
            List of detected file extensions (e.g., ['.pdf', '.csv'])
        """
        import re

        detected = set()
        query_lower = query.lower()

        # Pattern 1: Match explicit extensions (e.g., .pdf, file.xlsx)
        ext_pattern = r'\.(pdf|docx?|xlsx?|pptx?|csv|txt|md|json|yaml|yml|py|js|ts|html?)\b'
        matches = re.findall(ext_pattern, query_lower)
        for ext in matches:
            detected.add(f".{ext}")

        # Pattern 2: Match file type keywords (e.g., "pdf report", "excel file")
        for extension, keywords in self.FILE_TYPE_KEYWORDS.items():
            for keyword in keywords:
                if keyword in query_lower:
                    detected.add(extension)
                    break

        return list(detected)

    def get_category_for_file_type(self, file_ext: str) -> Optional[str]:
        """Get the category for a file extension."""
        for category, extensions in self.FILE_TYPE_CATEGORIES.items():
            if file_ext.lower() in extensions:
                return category
        return None

    # Intents that indicate document extraction (prefer document_qa over file_read)
    DOCUMENT_EXTRACTION_INTENTS = {"extract", "analyze", "summarize", "find"}

    # Intents that indicate web research (prefer deep_research)
    RESEARCH_INTENTS = {"research", "search", "compare"}

    def prefilter_capabilities(
        self,
        query: str,
        max_candidates: int = 10,
    ) -> List[Capability]:
        """
        Stage 1: Fast pre-filter capabilities using Tier 1 metadata.

        This uses lightweight metadata (intents, file_types, priority) to quickly
        narrow down candidates WITHOUT calling the LLM.

        Args:
            query: User's query string
            max_candidates: Maximum number of candidates to return

        Returns:
            List of candidate capabilities sorted by relevance
        """
        if not self._capabilities and not self._cleared:
            self.discover_all()

        # Detect intents and file types from query
        detected_intents = set(self.detect_intents(query))
        detected_extensions = self.detect_file_extensions(query)

        # Check if this is a document extraction task
        has_document_file = any(
            self.get_category_for_file_type(ext) == "document_processing"
            for ext in detected_extensions
        )
        has_extraction_intent = bool(detected_intents & self.DOCUMENT_EXTRACTION_INTENTS)
        is_document_extraction = has_document_file and has_extraction_intent

        # Check if this is a research/search task
        is_research_task = bool(detected_intents & self.RESEARCH_INTENTS)

        # Score each capability
        scored_caps: List[tuple] = []

        for cap in self._capabilities.values():
            score = 0.0

            # STRONG BOOST: Document extraction tasks should use document_qa
            if is_document_extraction and cap.category == "document_processing":
                score += 0.8  # Strong boost for document processing skills

            # STRONG BOOST: Research/search tasks should use deep_research
            if is_research_task and cap.category == "research":
                score += 0.8  # Strong boost for research skills
            elif is_research_task and cap.name in ("deep_research", "web_search"):
                score += 0.7  # Also boost web_search for search tasks

            # Score based on intent matching
            if cap.intents:
                intent_matches = len(detected_intents & set(cap.intents))
                score += intent_matches * 0.25

            # Score based on file type matching
            if detected_extensions and cap.file_types:
                for ext in detected_extensions:
                    if ext in cap.file_types:
                        score += 0.35  # Strong signal
                    # Check if skill handles document types
                    category = self.get_category_for_file_type(ext)
                    if category and cap.category == category:
                        score += 0.2

            # Score based on category + detected file types
            if detected_extensions:
                for ext in detected_extensions:
                    category = self.get_category_for_file_type(ext)
                    if category == "document_processing" and cap.category == "document_processing":
                        score += 0.25
                    elif category == cap.category:
                        score += 0.12

            # Bonus for keyword matches in description/tags
            query_lower = query.lower()
            for tag in cap.tags:
                if tag.lower() in query_lower:
                    score += 0.08

            # Apply priority (lower priority number = better)
            priority_bonus = (10 - cap.priority) * 0.015
            score += priority_bonus

            # PENALTY: If this is document extraction, penalize file_read
            if is_document_extraction and cap.name in ("file_read", "file-read"):
                score -= 0.5  # Penalize file_read for document tasks

            if score > 0:
                scored_caps.append((score, cap.priority, cap))

        # Sort by score (desc), then by priority (asc)
        scored_caps.sort(key=lambda x: (-x[0], x[1]))

        # Return top candidates
        return [cap for _, _, cap in scored_caps[:max_candidates]]

    def get_relevant_capabilities(
        self,
        query: str,
        max_results: int = 5,
    ) -> List[Capability]:
        """
        Two-stage skill selection: Pre-filter then rank.

        Stage 1: Fast pre-filter using Tier 1 metadata (no LLM)
        Stage 2: Return top candidates with full Tier 2 metadata

        Args:
            query: User's query string
            max_results: Maximum number of relevant capabilities

        Returns:
            List of relevant capabilities with full metadata
        """
        # Stage 1: Pre-filter using Tier 1 metadata
        candidates = self.prefilter_capabilities(query, max_candidates=max_results * 2)

        if not candidates:
            # Fallback to keyword matching if no pre-filter matches
            return self.match_capabilities(query, max_results=max_results)

        # Return top results (already sorted by relevance)
        return candidates[:max_results]

    def format_selected_capabilities_for_prompt(
        self,
        capabilities: List[Capability],
        include_when_not_to_use: bool = True,
    ) -> str:
        """
        Format selected capabilities with FULL Tier 2 metadata for LLM.

        Only called for pre-filtered candidates to keep prompt size small.

        Args:
            capabilities: List of pre-filtered capabilities
            include_when_not_to_use: Whether to include when_not_to_use guidance

        Returns:
            Formatted string for inclusion in system prompt
        """
        lines = ["# Relevant Skills for This Task\n"]
        lines.append("The following skills have been pre-selected as relevant. Choose the BEST one.\n")

        for i, cap in enumerate(capabilities, 1):
            lines.append(f"## {i}. {cap.name} (Priority: {cap.priority})")
            lines.append(f"**Category:** {cap.category or 'general'}")
            lines.append(f"**Description:** {cap.description[:200]}...")
            lines.append("")

            if cap.when_to_use:
                lines.append("**WHEN TO USE:**")
                lines.append(cap.when_to_use.strip())
                lines.append("")

            if include_when_not_to_use and cap.when_not_to_use:
                lines.append("**WHEN NOT TO USE:**")
                lines.append(cap.when_not_to_use.strip())
                lines.append("")

            if cap.examples:
                lines.append("**Examples:**")
                for ex in cap.examples[:3]:
                    lines.append(f"  - {ex}")
                lines.append("")

            if cap.file_types:
                lines.append(f"**Supported file types:** {', '.join(cap.file_types)}")

            lines.append("")
            lines.append("---")
            lines.append("")

        return "\n".join(lines)
