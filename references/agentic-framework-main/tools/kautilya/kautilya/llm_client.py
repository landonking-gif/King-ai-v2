"""
LLM Client for Kautilya Chat Interface.

Module: kautilya/llm_client.py

Provides conversational interface for natural language interaction with Kautilya features.

Configuration:
- Provider and model detection uses the unified LLM adapter factory
- Configuration from .env (OPENAI_MODEL, ANTHROPIC_MODEL, etc.) with runtime overrides
- Currently uses OpenAI SDK for full feature support (streaming, tool calling)
- Future: Full multi-provider support via adapters

Note: For full provider-agnostic LLM usage in skills, use adapters.llm.create_sync_adapter()
"""

import os
import json
import sys
from typing import Any, Dict, List, Optional, Generator
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Add adapters to path for unified configuration
_repo_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

# Import adapter factory for unified configuration
try:
    from adapters.llm.factory import (
        get_default_provider,
        get_default_model,
        get_llm_config,
        LLMProvider,
        PROVIDER_ENV_MAP,
    )
    _ADAPTERS_AVAILABLE = True
except ImportError:
    _ADAPTERS_AVAILABLE = False

# Import OpenAI SDK for actual API calls
# (Kautilya's architecture requires OpenAI-specific features)
from openai import OpenAI


def _find_project_env() -> Optional[Path]:
    """
    Find .env file by searching upward from this file's location.

    Looks for .env file in directories containing project markers
    (.git, pyproject.toml, setup.py) or the .env file itself.

    Returns:
        Path to .env file if found, None otherwise
    """
    current = Path(__file__).resolve().parent

    # Search upward up to 10 levels
    for _ in range(10):
        env_path = current / ".env"
        if env_path.exists():
            return env_path

        # Check for project root markers
        markers = [".git", "pyproject.toml", "setup.py", "CLAUDE.md"]
        for marker in markers:
            if (current / marker).exists():
                # Found project root, check for .env here
                if env_path.exists():
                    return env_path
                # Also check parent (in case .env is one level up)
                parent_env = current.parent / ".env"
                if parent_env.exists():
                    return parent_env

        parent = current.parent
        if parent == current:
            break  # Reached filesystem root
        current = parent

    return None


# Load .env with override=True so .env values take precedence
_env_path = _find_project_env()
if _env_path:
    load_dotenv(_env_path, override=True)
else:
    # Fallback: try default load_dotenv behavior (CWD)
    load_dotenv(override=True)


@dataclass
class Message:
    """Chat message."""

    role: str  # "system", "user", "assistant", "tool"
    content: str
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to OpenAI message format."""
        msg: Dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_call_id:
            msg["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            msg["tool_calls"] = self.tool_calls
        if self.name:
            msg["name"] = self.name
        return msg


@dataclass
class ChatHistory:
    """Manages conversation history."""

    messages: List[Message] = field(default_factory=list)
    max_messages: int = 50

    def add(self, message: Message) -> None:
        """Add message to history."""
        self.messages.append(message)
        # Trim old messages if exceeds max (keep system message)
        if len(self.messages) > self.max_messages:
            self._trim_history()

    def _trim_history(self) -> None:
        """
        Trim history while preserving tool_calls/tool response pairs.

        This ensures we never have orphaned tool messages (tool messages without
        their corresponding assistant message with tool_calls).
        """
        # Separate system messages from others
        system_msgs = [m for m in self.messages if m.role == "system"]
        other_msgs = [m for m in self.messages if m.role != "system"]

        # Calculate how many non-system messages we can keep
        max_other = self.max_messages - len(system_msgs)

        if len(other_msgs) <= max_other:
            return  # No trimming needed

        # Strategy: Remove oldest messages, but never break tool_calls/tool pairs
        # Build a map of tool_call_ids to their assistant message index
        tool_call_to_assistant: Dict[str, int] = {}
        for i, msg in enumerate(other_msgs):
            if msg.role == "assistant" and msg.tool_calls:
                for tc in msg.tool_calls:
                    if tc.get("id"):
                        tool_call_to_assistant[tc["id"]] = i

        # Mark messages to keep (working backwards from the end)
        messages_to_keep = set()
        messages_added = 0
        target_count = max_other

        # Work backwards through messages
        for i in range(len(other_msgs) - 1, -1, -1):
            if messages_added >= target_count:
                break

            msg = other_msgs[i]

            # Check if this message should be kept
            if i in messages_to_keep:
                continue  # Already marked for keeping

            # If this is a tool message, also mark its assistant message
            if msg.role == "tool" and msg.tool_call_id:
                assistant_idx = tool_call_to_assistant.get(msg.tool_call_id)
                if assistant_idx is not None and assistant_idx < i:
                    # Mark the assistant message for keeping
                    if assistant_idx not in messages_to_keep:
                        messages_to_keep.add(assistant_idx)
                        messages_added += 1

                    # Also mark all other tool messages for this assistant
                    assistant_msg = other_msgs[assistant_idx]
                    if assistant_msg.tool_calls:
                        all_tool_ids = {tc.get("id") for tc in assistant_msg.tool_calls if tc.get("id")}
                        for j in range(assistant_idx + 1, len(other_msgs)):
                            other = other_msgs[j]
                            if other.role == "tool" and other.tool_call_id in all_tool_ids:
                                if j not in messages_to_keep:
                                    messages_to_keep.add(j)
                                    messages_added += 1

            # Mark this message for keeping
            if i not in messages_to_keep:
                messages_to_keep.add(i)
                messages_added += 1

        # Build the final list of messages to keep (in order)
        kept_messages = [other_msgs[i] for i in sorted(messages_to_keep)]

        # If we're still over the limit, remove oldest messages
        # (but respect tool_calls/tool pairs)
        while len(kept_messages) > max_other:
            # Find the first message we can safely remove
            removed = False
            for i in range(len(kept_messages)):
                msg = kept_messages[i]

                # Don't remove assistant messages with tool_calls if their tool responses are still present
                if msg.role == "assistant" and msg.tool_calls:
                    tool_ids = {tc.get("id") for tc in msg.tool_calls if tc.get("id")}
                    # Check if any tool responses for this assistant are still in kept_messages
                    has_tool_responses = any(
                        m.role == "tool" and m.tool_call_id in tool_ids
                        for m in kept_messages[i+1:]
                    )
                    if has_tool_responses:
                        continue  # Can't remove this

                # Don't remove tool messages (they should be removed with their assistant message)
                if msg.role == "tool":
                    continue

                # Safe to remove
                kept_messages.pop(i)
                removed = True
                break

            if not removed:
                # Can't safely remove any more messages
                # Remove the oldest tool_calls group
                for i in range(len(kept_messages)):
                    msg = kept_messages[i]
                    if msg.role == "assistant" and msg.tool_calls:
                        # Remove this assistant and all its tool messages
                        tool_ids = {tc.get("id") for tc in msg.tool_calls if tc.get("id")}
                        kept_messages = [
                            m for j, m in enumerate(kept_messages)
                            if j != i and not (m.role == "tool" and m.tool_call_id in tool_ids)
                        ]
                        break
                else:
                    # No tool_calls groups found, just remove oldest
                    kept_messages.pop(0)

        # Reconstruct messages: system + trimmed others
        self.messages = system_msgs + kept_messages

    def to_list(self) -> List[Dict[str, Any]]:
        """Convert to OpenAI messages format."""
        return [m.to_dict() for m in self.messages]

    def clear(self) -> None:
        """Clear history (keep system message)."""
        self.messages = [m for m in self.messages if m.role == "system"]


# Define tools for Kautilya commands
KAUTILYA_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "init_project",
            "description": "Initialize a new agent project with the Agentic Framework. Creates project structure with agents/, skills/, manifests/ directories and configuration files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name of the project to create",
                    },
                    "provider": {
                        "type": "string",
                        "enum": ["openai", "anthropic", "azure", "local"],
                        "description": "LLM provider to use (default: openai)",
                    },
                    "enable_mcp": {
                        "type": "boolean",
                        "description": "Enable MCP (Model Context Protocol) integration",
                    },
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_agent",
            "description": "Create a new subagent with a specific role and capabilities. Agents are specialized workers that perform specific tasks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name of the agent (e.g., research-agent, code-agent)",
                    },
                    "role": {
                        "type": "string",
                        "enum": ["research", "verify", "code", "synthesis", "custom"],
                        "description": "Role of the agent determining its behavior",
                    },
                    "capabilities": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of capabilities (e.g., web_search, document_read, summarize, code_gen)",
                    },
                    "output_type": {
                        "type": "string",
                        "description": "Type of artifact this agent produces (e.g., research_snippet, code_patch)",
                    },
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_skill",
            "description": "Create a new skill with defined input/output schema. Skills are reusable, deterministic operations that agents can execute.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name of the skill (e.g., extract-entities, summarize-text)",
                    },
                    "description": {
                        "type": "string",
                        "description": "Description of what the skill does",
                    },
                    "input_fields": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Input field names (e.g., text, entity_types)",
                    },
                    "output_fields": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Output field names (e.g., entities, summary)",
                    },
                    "safety_flags": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["pii_risk", "external_call", "side_effect"],
                        },
                        "description": "Safety flags for the skill",
                    },
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_skills",
            "description": "List all available skills in the current project or framework.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "configure_llm",
            "description": "Configure LLM provider settings including API keys and default models.",
            "parameters": {
                "type": "object",
                "properties": {
                    "provider": {
                        "type": "string",
                        "enum": ["openai", "anthropic", "azure", "local"],
                        "description": "LLM provider to configure",
                    },
                    "model": {
                        "type": "string",
                        "description": "Model name to use (e.g., gpt-4o, claude-sonnet-4)",
                    },
                    "set_default": {
                        "type": "boolean",
                        "description": "Set this provider as the default",
                    },
                },
                "required": ["provider"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_llm_providers",
            "description": "List all available LLM providers and their configurations.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "test_llm_connection",
            "description": "Test the connection to the configured LLM provider.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_mcp_server",
            "description": "Add an MCP (Model Context Protocol) server to the project for tool integration.",
            "parameters": {
                "type": "object",
                "properties": {
                    "server_name": {
                        "type": "string",
                        "description": "Name of the MCP server (e.g., github, filesystem, postgres, slack)",
                    },
                    "scopes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Permission scopes for the server",
                    },
                },
                "required": ["server_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_mcp_servers",
            "description": "List all available MCP servers from the catalog.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mcp_call",
            "description": "Call any external MCP server tool through the gateway. Use this to invoke tools from registered MCP servers like Firecrawl (web scraping), GitHub, Slack, databases, etc. The gateway handles authentication and rate limiting.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tool_id": {
                        "type": "string",
                        "description": "MCP server ID (e.g., 'firecrawl_mcp', 'github_mcp', 'web_search')",
                    },
                    "tool_name": {
                        "type": "string",
                        "description": "Specific tool name within the server (e.g., 'scrape', 'crawl', 'search')",
                    },
                    "arguments": {
                        "type": "object",
                        "description": "Tool-specific arguments as key-value pairs",
                        "additionalProperties": True,
                    },
                },
                "required": ["tool_id", "tool_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_manifest",
            "description": "Create a new workflow manifest that defines a multi-step agent workflow.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name of the manifest/workflow",
                    },
                    "description": {
                        "type": "string",
                        "description": "Description of what the workflow does",
                    },
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "validate_manifest",
            "description": "Validate a workflow manifest against the schema.",
            "parameters": {
                "type": "object",
                "properties": {
                    "manifest_file": {
                        "type": "string",
                        "description": "Path to the manifest file to validate",
                    },
                },
                "required": ["manifest_file"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_project",
            "description": "Run the current agent project in development mode.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "show_status",
            "description": "Show status of running agents, memory usage, and service connections.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "show_logs",
            "description": "Show logs for agents or all services.",
            "parameters": {
                "type": "object",
                "properties": {
                    "agent_name": {
                        "type": "string",
                        "description": "Name of specific agent to show logs for (optional, shows all if not specified)",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "show_help",
            "description": "Show help information about Kautilya commands and capabilities.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Specific topic to get help on (e.g., agents, skills, manifests)",
                    },
                },
            },
        },
    },
]

# Pre-packaged file operation tools (Claude Code-like capabilities)
FILE_OPERATION_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "file_read",
            "description": "Read contents of a file with line numbers. Use for viewing code, configs, or any text file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Absolute path to the file to read",
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Starting line number (1-based)",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum lines to read (default 2000)",
                    },
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "file_glob",
            "description": "Find files matching glob patterns (e.g., **/*.py, src/**/*.ts). Use for file discovery.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern to match files",
                    },
                    "path": {
                        "type": "string",
                        "description": "Base directory to search in",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum results to return (default 100)",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "file_grep",
            "description": "Search for text patterns in files using regex. Use for finding code, functions, or text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regex pattern to search for",
                    },
                    "path": {
                        "type": "string",
                        "description": "File or directory to search in",
                    },
                    "glob": {
                        "type": "string",
                        "description": "Filter files by glob pattern (e.g., *.py)",
                    },
                    "case_insensitive": {
                        "type": "boolean",
                        "description": "Case insensitive search",
                    },
                    "context_lines": {
                        "type": "integer",
                        "description": "Context lines before/after match",
                    },
                    "output_mode": {
                        "type": "string",
                        "enum": ["content", "files_with_matches", "count"],
                        "description": "Output mode",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "file_write",
            "description": "Write or create a file with content. Overwrites existing files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Absolute path to the file",
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write",
                    },
                    "create_directories": {
                        "type": "boolean",
                        "description": "Create parent directories if needed",
                    },
                },
                "required": ["file_path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "file_edit",
            "description": "Edit a file by replacing specific text. Use for surgical code changes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Absolute path to the file",
                    },
                    "old_string": {
                        "type": "string",
                        "description": "Text to find and replace",
                    },
                    "new_string": {
                        "type": "string",
                        "description": "Replacement text",
                    },
                    "replace_all": {
                        "type": "boolean",
                        "description": "Replace all occurrences",
                    },
                },
                "required": ["file_path", "old_string", "new_string"],
            },
        },
    },
]

# Code execution tools
CODE_EXECUTION_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash_exec",
            "description": "Execute a bash command in the shell. Use for running scripts, git commands, npm, pip, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Bash command to execute",
                    },
                    "timeout_ms": {
                        "type": "integer",
                        "description": "Timeout in milliseconds (default 120000 = 2 min)",
                    },
                    "working_directory": {
                        "type": "string",
                        "description": "Working directory for command",
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "python_exec",
            "description": "Execute Python code. Use for running scripts, calculations, or testing code snippets.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute",
                    },
                    "timeout_seconds": {
                        "type": "integer",
                        "description": "Timeout in seconds (default 30)",
                    },
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "notebook_edit",
            "description": "Edit a Jupyter notebook cell. Use for modifying .ipynb files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "notebook_path": {
                        "type": "string",
                        "description": "Path to the Jupyter notebook",
                    },
                    "new_source": {
                        "type": "string",
                        "description": "New source content for the cell",
                    },
                    "cell_index": {
                        "type": "integer",
                        "description": "Cell index (0-based)",
                    },
                    "cell_type": {
                        "type": "string",
                        "enum": ["code", "markdown"],
                        "description": "Type of cell",
                    },
                    "edit_mode": {
                        "type": "string",
                        "enum": ["replace", "insert", "delete"],
                        "description": "Edit mode",
                    },
                },
                "required": ["notebook_path", "new_source"],
            },
        },
    },
]

# Web Search Tools (DuckDuckGo & Tavily)
WEB_SEARCH_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for real-time information using DuckDuckGo (free) or Tavily (API key). "
            "Use this to find current information, latest news, documentation, or answers to questions that require web research. "
            "Returns titles, URLs, and snippets from search results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (e.g., 'latest GPT models 2025', 'Python asyncio best practices')",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default 5, max 10)",
                    },
                    "provider": {
                        "type": "string",
                        "enum": ["duckduckgo", "tavily"],
                        "description": "Search provider to use. DuckDuckGo is free (default), Tavily requires API key but more accurate.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "configure_websearch",
            "description": "Configure web search provider settings. Set Tavily API key or change default provider.",
            "parameters": {
                "type": "object",
                "properties": {
                    "provider": {
                        "type": "string",
                        "enum": ["duckduckgo", "tavily"],
                        "description": "Provider to configure",
                    },
                    "tavily_api_key": {
                        "type": "string",
                        "description": "Tavily API key (required for Tavily provider)",
                    },
                    "set_as_default": {
                        "type": "boolean",
                        "description": "Set this provider as default for future searches",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_websearch_providers",
            "description": "List available web search providers and their configuration status.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "deep_research",
            "description": "Conduct comprehensive web research with source fetching and content extraction. "
            "Use this for in-depth research requiring multiple sources, qualitative analysis, and detailed reports. "
            "Searches web, fetches full page content via Firecrawl, extracts data, and prepares synthesis context. "
            "Default minimum 10 sources (configurable via DEEP_RESEARCH_MIN_SOURCES env var).",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Research question or topic (e.g., 'silver prices India last 30 days analysis')",
                    },
                    "min_sources": {
                        "type": "integer",
                        "description": "Minimum number of sources to fetch (default: 10, from DEEP_RESEARCH_MIN_SOURCES env)",
                    },
                    "max_sources": {
                        "type": "integer",
                        "description": "Maximum number of sources to fetch (default: 15, from DEEP_RESEARCH_MAX_SOURCES env)",
                    },
                    "output_format": {
                        "type": "string",
                        "enum": ["markdown", "json", "summary"],
                        "description": "Output format for the research report (default: markdown)",
                    },
                    "search_depth": {
                        "type": "string",
                        "enum": ["quick", "standard", "thorough"],
                        "description": "Search depth - affects number of search queries (default: standard)",
                    },
                },
                "required": ["query"],
            },
        },
    },
]

# Reflective agent tools (PLAN -> EXECUTE -> VALIDATE -> REFINE)
REFLECTIVE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "execute_with_reflection",
            "description": "Execute a task using the reflective agent loop: PLAN -> EXECUTE -> VALIDATE -> REFINE. Use for complex tasks that benefit from planning and self-correction.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "Task to execute with reflection",
                    },
                    "context": {
                        "type": "object",
                        "description": "Additional context for the task",
                    },
                    "constraints": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Constraints to follow during execution",
                    },
                    "max_iterations": {
                        "type": "integer",
                        "description": "Maximum refinement iterations (default 3)",
                    },
                    "validation_strictness": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                        "description": "How strict validation should be",
                    },
                },
                "required": ["task"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_execution_plan",
            "description": "Create an execution plan for a task without executing it. Returns a structured plan with steps, success criteria, and confidence score.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "Task to plan",
                    },
                    "available_tools": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tools available for execution",
                    },
                    "constraints": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Constraints to consider",
                    },
                },
                "required": ["task"],
            },
        },
    },
]

# Enterprise Agent tools (THINK -> PLAN -> APPROVE -> EXECUTE -> VALIDATE -> REFLECT)
# Hybrid approach: natural reasoning + structured auditability
ENTERPRISE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "enterprise_execute",
            "description": "Execute a task using the Enterprise Agent with full audit trail. "
            "Uses THINK -> PLAN -> APPROVE -> EXECUTE -> VALIDATE -> REFLECT loop. "
            "Best for enterprise tasks requiring governance, compliance, and auditability.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "Task to execute",
                    },
                    "user_id": {
                        "type": "string",
                        "description": "User ID for attribution (required for compliance)",
                    },
                    "context": {
                        "type": "object",
                        "description": "Additional context for the task",
                    },
                    "constraints": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Constraints to follow",
                    },
                    "enable_governance": {
                        "type": "boolean",
                        "description": "Enable governance gate for policy enforcement (default true)",
                    },
                    "auto_approve_low_risk": {
                        "type": "boolean",
                        "description": "Auto-approve low-risk actions (default true)",
                    },
                    "max_iterations": {
                        "type": "integer",
                        "description": "Maximum refinement iterations (default 3)",
                    },
                },
                "required": ["task"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "enterprise_think",
            "description": "Use the THINK phase to reason about a task naturally before planning. "
            "Captures reasoning traces for audit trail. Returns insights, risks, and proposed approach.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "Task to think about",
                    },
                    "context": {
                        "type": "object",
                        "description": "Additional context",
                    },
                    "thinking_budget_tokens": {
                        "type": "integer",
                        "description": "Token budget for thinking (default 1000)",
                    },
                },
                "required": ["task"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "governance_check",
            "description": "Check a plan against governance policies. Returns approval status, "
            "violations, and whether human approval is required. Use before executing high-risk actions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "plan_description": {
                        "type": "string",
                        "description": "Description of the plan to check",
                    },
                    "tools_used": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of tools the plan will use",
                    },
                    "risk_level": {
                        "type": "string",
                        "enum": ["low", "medium", "high", "critical"],
                        "description": "Assessed risk level of the plan",
                    },
                },
                "required": ["plan_description", "tools_used"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "provenance_record",
            "description": "Record a provenance entry for an action. Creates cryptographic hash-based "
            "audit trail. Use for compliance and forensic analysis.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": "Action being recorded",
                    },
                    "actor_id": {
                        "type": "string",
                        "description": "ID of entity performing action",
                    },
                    "inputs": {
                        "type": "object",
                        "description": "Input data (will be hashed)",
                    },
                    "outputs": {
                        "type": "object",
                        "description": "Output data (will be hashed)",
                    },
                    "tool_id": {
                        "type": "string",
                        "description": "Tool used for action (optional)",
                    },
                },
                "required": ["action", "actor_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "audit_log",
            "description": "Log an audit event. Use for compliance logging and monitoring. "
            "Supports all enterprise agent phases (THINK, PLAN, APPROVE, EXECUTE, VALIDATE, REFLECT).",
            "parameters": {
                "type": "object",
                "properties": {
                    "phase": {
                        "type": "string",
                        "enum": [
                            "start", "think", "plan", "approve",
                            "execute", "validate", "reflect", "complete", "error"
                        ],
                        "description": "Phase of execution",
                    },
                    "execution_id": {
                        "type": "string",
                        "description": "Execution context ID",
                    },
                    "message": {
                        "type": "string",
                        "description": "Log message",
                    },
                    "severity": {
                        "type": "string",
                        "enum": ["debug", "info", "warning", "error", "critical"],
                        "description": "Severity level",
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Additional metadata",
                    },
                },
                "required": ["phase", "message"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_audit_trail",
            "description": "Get the audit trail for an execution. Returns all events, provenance records, "
            "and governance decisions. Use for compliance review and debugging.",
            "parameters": {
                "type": "object",
                "properties": {
                    "execution_id": {
                        "type": "string",
                        "description": "Execution ID to get trail for",
                    },
                    "include_provenance": {
                        "type": "boolean",
                        "description": "Include provenance chain (default true)",
                    },
                    "include_events": {
                        "type": "boolean",
                        "description": "Include audit events (default true)",
                    },
                },
                "required": ["execution_id"],
            },
        },
    },
]

# Long Content Generation Tools (for handling large files that exceed token limits)
LONG_CONTENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "smart_content_planner",
            "description": (
                "Plan long content generation by breaking it into manageable sections. "
                "Use this BEFORE generating large files (HTML dashboards, long code, blog posts). "
                "Returns a plan with sections to generate separately, avoiding token limit issues. "
                "ALWAYS use this for: dashboards with charts/tables, Python modules > 200 lines, "
                "long blog posts, multi-component web pages."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "Description of content to generate",
                    },
                    "target_format": {
                        "type": "string",
                        "description": "Target format: html, python, javascript, markdown, etc.",
                    },
                    "estimated_size": {
                        "type": "string",
                        "enum": ["small", "medium", "large", "very_large"],
                        "description": "Estimated size of content",
                    },
                    "output_file": {
                        "type": "string",
                        "description": "Target output file path",
                    },
                },
                "required": ["description", "target_format"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "section_generate",
            "description": (
                "Generate and write a specific section of planned content. "
                "Use after smart_content_planner to generate each section. "
                "Tracks progress and provides next section to generate. "
                "Call this for each section in the plan, in order."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "plan_id": {
                        "type": "string",
                        "description": "ID of the content plan from smart_content_planner",
                    },
                    "section_id": {
                        "type": "string",
                        "description": "ID of the section to generate",
                    },
                    "content": {
                        "type": "string",
                        "description": "The generated content for this section",
                    },
                    "file_path": {
                        "type": "string",
                        "description": "Target file path (optional, uses plan's output_file if not specified)",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["append", "prepend", "replace"],
                        "description": "Write mode - usually 'append' for sequential generation",
                    },
                },
                "required": ["plan_id", "section_id", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "streaming_file_write",
            "description": (
                "Write large content to a file in streaming chunks. "
                "Use for very large content that needs to be written in parts. "
                "Handles memory efficiently and provides progress feedback. "
                "Best for: large data files, generated reports, concatenated outputs."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Target file path",
                    },
                    "content_parts": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of content strings to write sequentially",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["overwrite", "append"],
                        "description": "Write mode",
                    },
                    "add_newlines": {
                        "type": "boolean",
                        "description": "Add newline between parts",
                    },
                },
                "required": ["file_path", "content_parts"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_content_plan",
            "description": "Retrieve a content generation plan by ID to check progress or resume.",
            "parameters": {
                "type": "object",
                "properties": {
                    "plan_id": {
                        "type": "string",
                        "description": "The plan ID to retrieve",
                    },
                },
                "required": ["plan_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_content_plans",
            "description": "List all content generation plans with their status.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
]

# Combine all tools
KAUTILYA_TOOLS = (
    KAUTILYA_TOOLS
    + FILE_OPERATION_TOOLS
    + CODE_EXECUTION_TOOLS
    + WEB_SEARCH_TOOLS
    + REFLECTIVE_TOOLS
    + ENTERPRISE_TOOLS
    + LONG_CONTENT_TOOLS
)

SYSTEM_PROMPT = """You are Kautilya, a CLI assistant for the Enterprise Agentic Framework.

## Core Principle
UNDERSTAND -> CLASSIFY -> EXECUTE -> VERIFY

**Bias toward action.** Execute with sensible defaults, state your assumptions, let user refine.
Avoid over-questioning - users prefer results they can adjust over answering questions upfront.

---

## PHASE 1: UNDERSTAND & DECIDE

### Step 1: Classify the Request

| User Says | Intent | First Action |
|-----------|--------|--------------|
| "What is...", "How does...", "Explain..." | EXPLAIN | Answer directly |
| "Where is...", "Find...", "Search for..." | LOCATE | file_grep or file_glob |
| "Change...", "Update...", "Fix...", "Add..." | MODIFY | file_read -> file_edit |
| "Run...", "Execute...", "Test..." | EXECUTE | bash_exec |
| "Configure...", "Set up...", "Enable..." | CONFIGURE | Check current state first |
| "Scrape...", "Fetch from web...", "Query API..." | INTEGRATE | Route to MCP |
| Current events, news, market data, recent developments | RESEARCH | web_search first |

### Step 1b: When to Use Web Search

**ALWAYS use `web_search` tool FIRST when the query involves:**
- Current events, news, or recent developments (politics, markets, world events)
- Real-time data (stock prices, weather, sports scores)
- Information that changes frequently (documentation updates, release notes)
- Topics where your training data may be outdated
- Questions containing words like: "latest", "current", "today", "recent", "now", "2024", "2025"

**Example patterns that REQUIRE web_search:**
- "What's happening in [country/region]?" -> web_search for current news
- "Analyse the [situation/crisis/event]" -> web_search for latest updates
- "Impact on [market/economy/sector]" -> web_search for current analysis
- "Latest developments in..." -> web_search required
- Any geopolitical, economic, or market-related analysis -> web_search first

### Step 1c: Source Citations in Responses

**ALWAYS cite sources using inline numbered references [1], [2], [3] when:**
- Using information from web searches
- Referencing specific files or documents
- Using data from MCP tool calls
- Making claims based on external sources

**Citation Format:**
- Use [1], [2], [3] immediately after the relevant claim or statement
- Multiple sources for same claim: "This trend is accelerating [1][2]"
- Sources are numbered in order of first use in your response
- Be specific: cite the exact source, not just "according to sources"

**Examples:**
- "The market dropped 2% today [1] following news of the policy change [2]."
- "According to the configuration file [1], the default model is claude-sonnet-4."
- "TechCrunch reports [1] that AI adoption increased by 40% in 2024."

**Citation Rules:**
- Cite web sources by their domain or title
- Cite files by their path
- Cite MCP calls by the tool and key argument (URL, query, etc.)
- Every factual claim from an external source MUST have a citation

### Step 2: Risk-Based Decision - Ask or Execute?

**EXECUTE IMMEDIATELY (Low Risk):**
- Read-only operations (file_read, file_glob, file_grep, list commands)
- Reversible changes (file_edit with clear intent)
- Standard conventions apply (YAML for config, src/ for code)
- User said "just do it", "go ahead", "proceed"

**EXECUTE + STATE ASSUMPTIONS (Medium Risk):**
- Creating new files (state where and why)
- Choosing between valid options (state which and why)
- Using defaults for optional parameters
- Example: "I created `UserService` in `src/services/` following existing patterns. Adjust?"

**ASK BEFORE PROCEEDING (High Risk):**
- Destructive operations (delete, overwrite, drop)
- Irreversible changes (production deployments, database migrations)
- Costly operations (paid API calls, long-running processes)
- 2+ equally valid interpretations with significant differences
- Security/compliance decisions

### Decision Checklist (Before Asking)
```
[ ] Is this destructive/irreversible? → If YES, ASK
[ ] Is this costly (API credits, compute time)? → If YES, ASK
[ ] Are there 2+ VERY different valid paths? → If YES, ASK
[ ] Is it read-only or easily reversible? → If YES, EXECUTE
[ ] Does context/convention make intent clear? → If YES, EXECUTE
```

### Anti-Patterns to AVOID
- Asking about file format when convention is clear
- Asking about location when project structure implies it
- Asking sequential questions (batch if truly needed)
- Asking for confirmation on read-only operations
- Re-confirming after user already approved

---

## PHASE 2: TASK-SPECIFIC PROTOCOLS

### Protocol A: Code Operations (LOCATE/MODIFY)

LOCATE phase:
- Specific file mentioned? -> file_read directly
- Pattern search needed? -> file_glob for files, file_grep for content
- Never use bash grep/find - use native tools

MODIFY phase:
- ALWAYS read file first (file_read)
- Small change (<10 lines)? -> file_edit with exact match
- Large change or new file? -> file_write
- Verify change with file_read after

**Examples:**
- User: "Fix the typo in config.py"
  -> Intent: MODIFY
  -> Action: file_read config.py -> identify typo -> file_edit with correction

- User: "Find all files using Redis"
  -> Intent: LOCATE
  -> Action: file_grep pattern="redis|Redis|REDIS" -> return file list

### Protocol B: Configuration Tasks (CONFIGURE)

1. Check current state first:
   - LLM config? -> Read .kautilya/llm.yaml
   - MCP servers? -> /mcp list
   - Project config? -> Read .kautilya/config.yaml

2. Make minimal changes:
   - Use existing commands when available (/llm config, /mcp enable)
   - Only edit files directly if no command exists

3. Verify after change:
   - Re-read config or run test command

**Examples:**
- User: "Change default model to gpt-4o"
  -> Intent: CONFIGURE
  -> Action: Read current llm.yaml -> file_edit to update default_model -> verify

- User: "Enable the Firecrawl server"
  -> Intent: CONFIGURE
  -> Action: /mcp list -> /mcp enable firecrawl_mcp -> confirm enabled

### Protocol C: Execution Tasks (EXECUTE)

1. Validate before running:
   - Does command exist?
   - Are dependencies installed?
   - Is path correct?

2. Execute with awareness:
   - Set reasonable timeout
   - Capture both stdout and stderr
   - Don't run destructive commands without confirmation

3. Interpret results:
   - Success? Report concisely
   - Failure? Diagnose and suggest fix

**Examples:**
- User: "Run the tests"
  -> Intent: EXECUTE
  -> Action: Check if pytest/test framework exists -> bash_exec "pytest" -> interpret results

- User: "Build the project"
  -> Intent: EXECUTE
  -> Action: Check for build tool (npm, make, etc.) -> run appropriate command

### Protocol D: External Integration (INTEGRATE) - MCP Tool Execution

{MCP_SERVER_TABLE}

**HOW TO CALL MCP TOOLS:**
Use the `mcp_call` tool to invoke any registered MCP server tool:

```
mcp_call(
    tool_id="firecrawl_mcp",     # Server ID from table above
    tool_name="scrape",          # Tool name from server's tools list
    arguments={"url": "https://example.com"}  # Tool-specific args
)
```

**WORKFLOW:**
1. Check MCP table above to find the right server and tool
2. Use `mcp_call` with correct tool_id, tool_name, and arguments
3. The gateway handles auth (API keys from env vars) and rate limiting

**Examples:**
- User: "Scrape the pricing page from example.com"
  -> Use mcp_call(tool_id="firecrawl_mcp", tool_name="scrape", arguments={"url": "https://example.com/pricing"})

- User: "Crawl the documentation site"
  -> Use mcp_call(tool_id="firecrawl_mcp", tool_name="crawl", arguments={"url": "https://docs.example.com", "limit": 50})

- User: "Search the web for Python tutorials"
  -> Use mcp_call(tool_id="web_search", tool_name="search", arguments={"query": "Python tutorials", "max_results": 10})

**IMPORTANT:** If a tool returns an error about missing API key, inform the user which environment variable needs to be set (check the metadata.api_key_env field).

### Protocol E: Long Content Generation (GENERATE)

For generating large files (dashboards, long code, blog posts), use chunked generation:

**DETECTION - Use long content tools when:**
- HTML with Plotly, DataTables, or multiple components
- Python modules > 200 lines
- Blog posts > 2000 words
- Any file likely to exceed 2000 tokens

**WORKFLOW:**
1. smart_content_planner -> Creates plan with sections
2. section_generate -> Generate each section in order
3. Repeat until all sections complete

**TOKEN LIMITS:**
- Default max_tokens: 2048 (configurable via /llm set-params)
- Large HTML dashboard: 10,000+ tokens -> MUST use chunked generation
- Long Python module: 5,000+ tokens -> SHOULD use chunked generation

**Examples:**
- User: "Create a Plotly dashboard with charts and DataTables"
  -> Intent: GENERATE (large content)
  -> Action: smart_content_planner(description="...", target_format="html", estimated_size="large")
  -> Then: section_generate for each section in plan

- User: "Write a comprehensive Python module for data processing"
  -> Intent: GENERATE (large content)
  -> Action: smart_content_planner(description="...", target_format="python", estimated_size="large")
  -> Then: section_generate for each section

- User: "Write a 3000-word blog post about AI agents"
  -> Intent: GENERATE (large content)
  -> Action: smart_content_planner(description="...", target_format="markdown", estimated_size="large")
  -> Then: section_generate for each section

**NEVER try to generate large files in one file_write call - it will be truncated!**

---

## PHASE 3: EXECUTION RULES

### Tool Selection Priority

**For Reading/Finding:**
- Specific file path known     -> file_read
- Find files by name/pattern   -> file_glob
- Find content in files        -> file_grep
- Need command output          -> bash_exec
- Need external data           -> MCP (verify first)

**For Writing/Changing:**
- Small precise edit           -> file_edit (requires prior file_read)
- New file or full rewrite     -> file_write
- System command               -> bash_exec
- External action              -> MCP (verify first)

### Parallel vs Sequential

**Run in PARALLEL when:**
- Reading multiple independent files
- Searching in different directories
- Gathering unrelated information

**Run SEQUENTIALLY when:**
- Output of one tool feeds into another
- Making changes that depend on current state
- Verification steps after modifications

### Iteration Management
- Track iteration count mentally
- If approaching limit (check KAUTILYA_MAX_ITERATIONS env var):
  - Summarize progress so far
  - Ask user: continue, pivot, or stop?
- Never silently abandon a task

---

## PHASE 4: RESPONSE FORMAT

{OUTPUT_MODE_INSTRUCTIONS}

### For Errors:
[What failed]
[Why it likely failed]
[1-2 specific alternatives]
[Ask: which approach to try?]

---

## GUARDRAILS

**ALWAYS:**
- Verify file exists before editing
- Read before write
- Check MCP availability before use
- Explain failures clearly
- Ask when uncertain

**NEVER:**
- Guess file paths
- Run destructive commands without confirmation
- Ignore error messages
- Repeat failed approaches
- Use bash for file operations (use native tools)
- Assume MCP servers are enabled

---

## SLASH COMMAND REFERENCE

| Command | Action |
|---------|--------|
| /init | Initialize new agent project |
| /agent new <name> | Create new agent |
| /skill new <name> | Create new skill |
| /llm config | Configure LLM provider |
| /llm list | List LLM providers |
| /llm test | Test LLM connection |
| /llm set-params | Set hyperparameters |
| /llm show-params | Show hyperparameters |
| /mcp list | List MCP servers |
| /mcp enable <id> | Enable MCP server |
| /mcp disable <id> | Disable MCP server |
| /manifest new | Create workflow manifest |
| /manifest validate | Validate manifest |
| /run | Run project |
| /status | Show status |
| /logs | View logs |
| /help | Show all commands |

---

## SUCCESS CRITERIA

You succeed when:
1. User accomplishes their goal
2. With minimal back-and-forth
3. Using the most direct path
4. With clear communication throughout

Think step-by-step. Act precisely. Verify results."""


class KautilyaLLMClient:
    """LLM client for Kautilya chat interface."""

    # Reasoning models that don't support temperature, top_p, etc.
    REASONING_MODEL_PREFIXES = ("o1", "o3", "o4", "gpt-5")

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_iterations: Optional[int] = None,
        custom_system_prompt: Optional[str] = None,
        config_dir: str = ".kautilya",
    ):
        """
        Initialize LLM client.

        Loading Priority:
            1. Constructor parameters (api_key, model)
            2. Environment variables from .env (OPENAI_API_KEY, OPENAI_MODEL)
            3. Config file (.kautilya/llm.yaml)
            4. Hardcoded defaults

        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: Model to use (defaults to OPENAI_MODEL env var, then llm.yaml, then gpt-4o-mini)
            max_iterations: Max tool iterations (defaults to KAUTILYA_MAX_ITERATIONS or 5)
            custom_system_prompt: Custom system prompt (overrides default)
            config_dir: Configuration directory for loading hyperparameters
        """
        self.config_dir = config_dir

        # Load llm.yaml config for fallback values
        llm_config = self._load_llm_config_safe()

        # API Key Priority: parameter -> .env (OPENAI_API_KEY) -> llm.yaml -> error
        self.api_key = self._resolve_api_key(api_key, llm_config)

        # Model Priority: parameter -> .env (OPENAI_MODEL) -> llm.yaml (default_model) -> "gpt-4o-mini"
        self.model = self._resolve_model(model, llm_config)

        # Load max iterations from env or use default
        default_max_iter = int(os.getenv("KAUTILYA_MAX_ITERATIONS", "5"))
        self.default_max_iterations = max_iterations or default_max_iter

        # Load hyperparameters from config
        self.hyperparameters = self._load_hyperparameters()

        # Load custom system prompt from file or parameter
        system_prompt = self._load_system_prompt(custom_system_prompt)

        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY in .env file, "
                "configure in .kautilya/llm.yaml, or pass api_key parameter."
            )

        self.client = OpenAI(api_key=self.api_key)
        self.history = ChatHistory()
        self._is_reasoning_model = self._check_reasoning_model()

        # Initialize with system prompt
        self.history.add(Message(role="system", content=system_prompt))

    def _load_llm_config_safe(self) -> Optional[Dict[str, Any]]:
        """
        Safely load llm.yaml config without raising exceptions.

        Returns:
            Config dictionary or None if loading fails
        """
        try:
            from .config import load_llm_config
            return load_llm_config(self.config_dir)
        except Exception:
            return None

    def _resolve_api_key(
        self, param_key: Optional[str], llm_config: Optional[Dict[str, Any]]
    ) -> Optional[str]:
        """
        Resolve API key with priority: parameter -> .env -> llm.yaml.

        Args:
            param_key: API key passed as constructor parameter
            llm_config: Loaded llm.yaml config

        Returns:
            Resolved API key or None
        """
        # Priority 1: Constructor parameter
        if param_key:
            return param_key

        # Priority 2: Environment variable from .env (OPENAI_API_KEY)
        env_key = os.getenv("OPENAI_API_KEY")
        if env_key:
            return env_key

        # Priority 3: llm.yaml - check api_key_env reference
        if llm_config and "providers" in llm_config:
            default_provider = llm_config.get("default_provider", "openai")
            if default_provider in llm_config["providers"]:
                provider_config = llm_config["providers"][default_provider]
                # Check if there's a different env var name specified
                api_key_env = provider_config.get("api_key_env", "OPENAI_API_KEY")
                env_key = os.getenv(api_key_env)
                if env_key:
                    return env_key

        return None

    def _resolve_model(
        self, param_model: Optional[str], llm_config: Optional[Dict[str, Any]]
    ) -> str:
        """
        Resolve model with priority: parameter -> adapter factory -> .env -> llm.yaml -> default.

        Uses the unified adapter factory for consistent configuration across the framework.

        Args:
            param_model: Model passed as constructor parameter
            llm_config: Loaded llm.yaml config

        Returns:
            Resolved model name
        """
        # Priority 1: Constructor parameter
        if param_model:
            return param_model

        # Priority 2: Use adapter factory (reads from .env with proper parsing)
        if _ADAPTERS_AVAILABLE:
            try:
                adapter_model = get_default_model()
                if adapter_model:
                    return adapter_model
            except Exception:
                pass  # Fall through to legacy resolution

        # Priority 3: Environment variable from .env (OPENAI_MODEL) - legacy fallback
        env_model = os.getenv("OPENAI_MODEL")
        if env_model:
            # Clean up potential whitespace from .env
            return env_model.strip().replace(" ", "")

        # Priority 4: llm.yaml default_model
        if llm_config and "providers" in llm_config:
            default_provider = llm_config.get("default_provider", "openai")
            if default_provider in llm_config["providers"]:
                provider_config = llm_config["providers"][default_provider]
                yaml_model = provider_config.get("default_model")
                if yaml_model:
                    return yaml_model

        # Priority 5: Hardcoded default
        return "gpt-4o-mini"

    def _try_fix_incomplete_json(self, raw_json: str) -> Optional[Dict[str, Any]]:
        """
        Attempt to fix incomplete JSON from streaming responses.

        Common issues:
        - Unterminated strings (missing closing quote)
        - Missing closing braces/brackets
        - Truncated content

        Args:
            raw_json: The incomplete/malformed JSON string

        Returns:
            Parsed dict if fixable, None otherwise
        """
        if not raw_json or raw_json.strip() == "":
            return {}

        # Try original first
        try:
            return json.loads(raw_json)
        except json.JSONDecodeError:
            pass

        # Common fixes for streaming issues
        fixes_to_try = [
            # Fix: missing closing quote and brace
            lambda s: s + '"}',
            # Fix: missing closing brace only
            lambda s: s + '}',
            # Fix: missing closing bracket and brace
            lambda s: s + ']}',
            # Fix: truncated in middle of value - add quote and braces
            lambda s: s.rstrip(',') + '"}' if s.rstrip().endswith(',') else None,
            # Fix: truncated string value
            lambda s: s + '"' + '}' * (s.count('{') - s.count('}')),
        ]

        for fix in fixes_to_try:
            try:
                fixed = fix(raw_json)
                if fixed:
                    result = json.loads(fixed)
                    # Verify it's a dict (expected format for tool args)
                    if isinstance(result, dict):
                        return result
            except (json.JSONDecodeError, TypeError):
                continue

        return None

    def _check_reasoning_model(self) -> bool:
        """Check if current model is a reasoning model (no temperature support)."""
        model_lower = self.model.lower()
        return any(model_lower.startswith(prefix) for prefix in self.REASONING_MODEL_PREFIXES)

    def _load_hyperparameters(self) -> Dict[str, Any]:
        """
        Load hyperparameters from config file and environment variables.

        Priority (highest to lowest):
        1. Environment variables (KAUTILYA_MAX_TOKENS)
        2. Config file (.kautilya/llm.yaml hyperparameters)
        3. Defaults

        Returns:
            Dictionary of hyperparameters with defaults
        """
        from .config import load_llm_config

        hyperparams = self._get_default_hyperparameters()

        try:
            llm_config = load_llm_config(self.config_dir)

            if llm_config and "providers" in llm_config:
                # Try to find the current provider based on the model
                # First, check if there's a default provider
                default_provider = llm_config.get("default_provider")

                if default_provider and default_provider in llm_config["providers"]:
                    provider_config = llm_config["providers"][default_provider]
                    config_hyperparams = provider_config.get("hyperparameters", {})
                    hyperparams = {**hyperparams, **config_hyperparams}

        except Exception as e:
            print(f"Warning: Failed to load hyperparameters from config: {e}")

        # Override with environment variables (highest priority)
        env_max_tokens = os.getenv("KAUTILYA_MAX_TOKENS")
        if env_max_tokens:
            try:
                hyperparams["max_tokens"] = int(env_max_tokens)
            except ValueError:
                print(f"Warning: Invalid KAUTILYA_MAX_TOKENS value: {env_max_tokens}")

        return hyperparams

    def _get_default_hyperparameters(self) -> Dict[str, Any]:
        """Get default hyperparameters."""
        return {
            "temperature": 0.7,
            "max_tokens": None,
            "top_p": None,
            "top_k": None,
            "frequency_penalty": None,
            "presence_penalty": None,
            "max_retries": 3,
        }

    def _build_mcp_server_table(self) -> str:
        """
        Build dynamic MCP server routing table from registered servers.

        Fetches the current list of MCP servers from the gateway and formats
        them as a markdown table for the system prompt.

        Returns:
            Markdown formatted table of MCP servers
        """
        try:
            from kautilya.mcp_gateway_client import MCPGatewayClient

            with MCPGatewayClient() as client:
                # Get all servers (both enabled and disabled)
                servers = client.list_servers_sync(enabled_only=False)

            if not servers:
                return """MCP SERVER ROUTING TABLE:
| Server ID | Name | Status | Description |
|-----------|------|--------|-------------|
| (none)    | No MCP servers registered | - | Run /mcp list for details |

**Note:** No MCP servers are currently registered. Import servers from YAML files."""

            # Build the table with description column
            lines = [
                "MCP SERVER ROUTING TABLE (dynamically loaded):",
                "| Server ID | Name | Description | Status | Tools |",
                "|-----------|------|-------------|--------|-------|",
            ]

            for server in servers:
                reg = server.get("registration", server)
                tool_id = reg.get("tool_id", "unknown")
                name = reg.get("name", "Unknown")
                enabled = server.get("enabled", True)
                status = "✓" if enabled else "○"

                # Get description - check multiple sources
                description = reg.get("description", "")

                # If no explicit description, derive from tools
                tools = reg.get("tools", [])
                if not description and tools:
                    # Use first tool's description or combine tool purposes
                    tool_descriptions = [t.get("description", "") for t in tools[:2] if t.get("description")]
                    if tool_descriptions:
                        description = tool_descriptions[0]

                # Truncate long descriptions
                if len(description) > 50:
                    description = description[:47] + "..."

                # If still no description, use a generic one based on name
                if not description:
                    description = f"{name} operations"

                # Get tool names
                if tools:
                    tool_names = [t.get("name", "") for t in tools[:3]]
                    tools_str = ", ".join(tool_names)
                    if len(tools) > 3:
                        tools_str += f" +{len(tools) - 3}"
                else:
                    tools_str = "-"

                lines.append(f"| {tool_id} | {name} | {description} | {status} | {tools_str} |")

            # Add usage guidance
            lines.append("")
            lines.append("**Legend:** ✓ = enabled, ○ = disabled")
            lines.append("**To use:** Check /mcp list, verify enabled, then call tools.")

            return "\n".join(lines)

        except Exception as e:
            # Fallback if gateway unavailable
            return f"""MCP SERVER ROUTING TABLE:
| Status | Message |
|--------|---------|
| ⚠ | Could not fetch MCP servers: {str(e)[:50]} |

**Fallback:** Use /mcp list to see available servers."""

    def _build_output_mode_instructions(self) -> str:
        """
        Build output mode instructions based on current setting.

        Returns:
            Instructions for response formatting based on concise/verbose mode
        """
        from kautilya.iteration_display import get_output_mode, OutputMode

        mode = get_output_mode()

        if mode == OutputMode.CONCISE:
            return """### Response Style: CONCISE (Active)

**CRITICAL: Give direct answers only. Do NOT explain your process.**

✓ DO:
- State the result/answer directly
- Be brief and to the point
- Only mention what changed or what the answer is

✗ DO NOT:
- Explain which tools you used
- Describe your step-by-step process
- Say "I analyzed...", "I searched...", "I found..."
- Narrate what you did to arrive at the answer
- Include phrases like "Let me...", "First I...", "After examining..."

**Good example:**
"The database connection has been updated to use new-server.example.com."

**Bad example:**
"I read the config file, searched for the database section, and updated the connection string. The database connection has been updated to use new-server.example.com."

Only include process details if the user explicitly asks "how did you...", "explain your process", or "what steps did you take"."""
        else:
            return """### Response Style: VERBOSE (Active)

**CRITICAL: Always START with the answer or result. Never start by explaining your process.**

For ALL responses:
- Lead with the answer, finding, or result
- Put the most important information first
- Include helpful context and examples after the main answer

✗ NEVER start with:
- "Action:", "Let me...", "I will...", "First, I..."
- "I'm going to...", "I'll search...", "I'll analyze..."
- Explanations of what tools you're using or why
- Descriptions of your approach or strategy

✓ ALWAYS start with:
- The direct answer to the question
- The result or finding
- The key information the user asked for

**Good example:**
"The Nifty 50 index rose 2.3% over the last month, driven primarily by IT and banking sectors. [Analysis follows...]"

**Bad example:**
"Action: I'll use web search to find Nifty 50 data, then analyze the trends. Let me search for recent market news..."

Tool execution details are shown separately in the UI - do not narrate them in your response."""

    def _inject_dynamic_content(self, prompt: str) -> str:
        """
        Inject dynamic content into system prompt template.

        Replaces placeholders with current values:
        - {MCP_SERVER_TABLE}: Current MCP server list
        - {OUTPUT_MODE_INSTRUCTIONS}: Current output verbosity instructions

        Args:
            prompt: System prompt template with placeholders

        Returns:
            System prompt with placeholders replaced
        """
        if "{MCP_SERVER_TABLE}" in prompt:
            mcp_table = self._build_mcp_server_table()
            prompt = prompt.replace("{MCP_SERVER_TABLE}", mcp_table)

        if "{OUTPUT_MODE_INSTRUCTIONS}" in prompt:
            output_instructions = self._build_output_mode_instructions()
            prompt = prompt.replace("{OUTPUT_MODE_INSTRUCTIONS}", output_instructions)

        return prompt

    def _load_system_prompt(self, custom_prompt: Optional[str] = None) -> str:
        """
        Load system prompt from custom source or use default.

        Args:
            custom_prompt: Optional custom prompt text or path to prompt file

        Returns:
            System prompt string with dynamic content injected
        """
        base_prompt = None

        # Priority 1: Explicit custom prompt parameter
        if custom_prompt:
            # Check if it's a file path
            if custom_prompt.endswith('.txt') or custom_prompt.endswith('.md'):
                try:
                    with open(custom_prompt, 'r') as f:
                        base_prompt = f.read()
                except FileNotFoundError:
                    # Treat as literal prompt text
                    base_prompt = custom_prompt
            else:
                base_prompt = custom_prompt

        # Priority 2: Custom prompt from env var
        if base_prompt is None:
            prompt_path = os.getenv("KAUTILYA_CUSTOM_SYSTEM_PROMPT_PATH")
            if prompt_path and Path(prompt_path).exists():
                try:
                    with open(prompt_path, 'r') as f:
                        base_prompt = f.read()
                except Exception:
                    pass  # Fall back to default

        # Priority 3: Default system prompt
        if base_prompt is None:
            base_prompt = SYSTEM_PROMPT

        # Inject dynamic content (MCP servers, etc.)
        return self._inject_dynamic_content(base_prompt)

    def _get_chat_params(
        self,
        include_tools: bool = True,
        stream: bool = True,
        additional_tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Get chat completion parameters based on model type.

        Reasoning models (o1, o3, o4, gpt-5) don't support temperature,
        top_p, presence_penalty, frequency_penalty, etc.

        Args:
            include_tools: Whether to include tool definitions
            stream: Whether to stream the response
            additional_tools: Extra tools to add (e.g., dynamic skill tools)
        """
        params: Dict[str, Any] = {
            "model": self.model,
            "messages": self.history.to_list(),
        }

        # Add max_tokens if configured
        if self.hyperparameters.get("max_tokens"):
            params["max_completion_tokens"] = self.hyperparameters["max_tokens"]
        else:
            params["max_completion_tokens"] = 2048  # Default

        if stream:
            params["stream"] = True
            params["stream_options"] = {"include_usage": True}  # Get token usage in streaming mode

        if include_tools:
            tools = KAUTILYA_TOOLS + CODE_EXECUTION_TOOLS
            if additional_tools:
                tools = tools + additional_tools
            params["tools"] = tools
            params["tool_choice"] = "auto"

        # Only add sampling parameters for non-reasoning models
        if not self._is_reasoning_model:
            # Temperature
            params["temperature"] = self.hyperparameters.get("temperature", 0.7)

            # Top-p (nucleus sampling)
            if self.hyperparameters.get("top_p") is not None:
                params["top_p"] = self.hyperparameters["top_p"]

            # Frequency penalty
            if self.hyperparameters.get("frequency_penalty") is not None:
                params["frequency_penalty"] = self.hyperparameters["frequency_penalty"]

            # Presence penalty
            if self.hyperparameters.get("presence_penalty") is not None:
                params["presence_penalty"] = self.hyperparameters["presence_penalty"]

        return params

    def chat(
        self,
        user_message: str,
        tool_executor: Optional[Any] = None,
        stream: bool = True,
        max_tool_iterations: Optional[int] = None,
        additional_tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Generator[str, None, Optional[Dict[str, Any]]]:
        """
        Send a chat message and get response with agentic loop support.

        The agent can now execute multiple rounds of tool calls, reviewing results
        and deciding next actions until the task is complete or max iterations reached.

        Args:
            user_message: User's message
            tool_executor: Optional executor for tool calls
            stream: Whether to stream the response
            max_tool_iterations: Maximum tool execution iterations (defaults to config or 5)
            additional_tools: Extra tools to add dynamically (e.g., skill tools)

        Yields:
            Response chunks (if streaming)

        Returns:
            Tool call results if any tools were executed
        """
        # Use default max iterations if not specified
        max_tool_iterations = max_tool_iterations or self.default_max_iterations
        # Add user message to history
        self.history.add(Message(role="user", content=user_message))

        all_tool_results = []
        iteration = 0
        total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        try:
            if stream:
                # Agentic loop - continue until no more tool calls or max iterations
                while iteration < max_tool_iterations:
                    iteration += 1

                    if iteration > 1:
                        yield f"\n\n[Iteration {iteration}/{max_tool_iterations}]\n"

                    # Clean up any orphaned tool messages before API call
                    self.cleanup_orphaned_tool_messages()

                    # Validate history before making API call
                    if not self.validate_history():
                        raise ValueError(
                            "Conversation history is invalid: some tool_calls have no matching responses. "
                            "This is a bug - please report with your query."
                        )

                    # Get response WITH TOOLS ENABLED (critical for agentic loop)
                    params = self._get_chat_params(include_tools=True, stream=True, additional_tools=additional_tools)
                    response = self.client.chat.completions.create(**params)

                    collected_content = ""
                    collected_tool_calls: List[Dict[str, Any]] = []
                    usage_info = None  # Track token usage
                    finish_reason = None  # Track finish reason for continuation

                    for chunk in response:
                        # Capture usage information from the final chunk
                        if hasattr(chunk, 'usage') and chunk.usage:
                            usage_info = {
                                "prompt_tokens": chunk.usage.prompt_tokens,
                                "completion_tokens": chunk.usage.completion_tokens,
                                "total_tokens": chunk.usage.total_tokens,
                            }

                        # Capture finish_reason
                        if chunk.choices and chunk.choices[0].finish_reason:
                            finish_reason = chunk.choices[0].finish_reason

                        delta = chunk.choices[0].delta if chunk.choices else None

                        if delta is None:
                            continue

                        # Handle content streaming
                        if delta.content:
                            collected_content += delta.content
                            yield delta.content

                        # Handle tool calls
                        if delta.tool_calls:
                            for tool_call in delta.tool_calls:
                                if tool_call.index is not None:
                                    # Start new tool call or continue existing
                                    while len(collected_tool_calls) <= tool_call.index:
                                        collected_tool_calls.append(
                                            {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}
                                        )

                                    tc = collected_tool_calls[tool_call.index]

                                    if tool_call.id:
                                        tc["id"] = tool_call.id
                                    if tool_call.function:
                                        if tool_call.function.name:
                                            tc["function"]["name"] = tool_call.function.name
                                        if tool_call.function.arguments:
                                            tc["function"]["arguments"] += tool_call.function.arguments

                    # Accumulate token usage from this iteration
                    if usage_info:
                        total_usage["prompt_tokens"] += usage_info.get("prompt_tokens", 0)
                        total_usage["completion_tokens"] += usage_info.get("completion_tokens", 0)
                        total_usage["total_tokens"] += usage_info.get("total_tokens", 0)

                    # Filter out invalid tool calls (empty id or name)
                    valid_tool_calls = [
                        tc for tc in collected_tool_calls
                        if tc.get("id") and tc.get("function", {}).get("name")
                    ]

                    # If no valid tool calls, check for continuation or complete
                    if not valid_tool_calls or not tool_executor:
                        # Handle continuation for truncated responses
                        if finish_reason == "length" and collected_content:
                            # Response was truncated - continue generation
                            max_continuations = int(os.getenv("KAUTILYA_MAX_CONTINUATIONS", "5"))
                            continuation_count = 0
                            full_content = collected_content

                            while finish_reason == "length" and continuation_count < max_continuations:
                                continuation_count += 1
                                yield f"\n\n[Continuing... ({continuation_count}/{max_continuations})]\n\n"

                                # Add partial response to history
                                self.history.add(Message(role="assistant", content=full_content))

                                # Request continuation
                                self.history.add(Message(
                                    role="user",
                                    content="Continue exactly where you left off. Do not repeat any content. Continue seamlessly:"
                                ))

                                # Make continuation API call (no tools to avoid interruption)
                                cont_params = self._get_chat_params(include_tools=False, stream=True)
                                cont_response = self.client.chat.completions.create(**cont_params)

                                cont_content = ""
                                finish_reason = None

                                for chunk in cont_response:
                                    if hasattr(chunk, 'usage') and chunk.usage:
                                        total_usage["prompt_tokens"] += chunk.usage.prompt_tokens or 0
                                        total_usage["completion_tokens"] += chunk.usage.completion_tokens or 0
                                        total_usage["total_tokens"] += chunk.usage.total_tokens or 0

                                    if chunk.choices and chunk.choices[0].finish_reason:
                                        finish_reason = chunk.choices[0].finish_reason

                                    delta = chunk.choices[0].delta if chunk.choices else None
                                    if delta and delta.content:
                                        cont_content += delta.content
                                        yield delta.content

                                # Update full content for next iteration
                                full_content = cont_content

                                # Remove the continuation request from history (clean up)
                                if self.history.messages and self.history.messages[-1].role == "user":
                                    self.history.messages.pop()
                                if self.history.messages and self.history.messages[-1].role == "assistant":
                                    self.history.messages.pop()

                            # Add final complete response to history
                            if full_content:
                                self.history.add(Message(role="assistant", content=full_content))

                            if finish_reason == "length":
                                yield f"\n\n[Max continuations reached. Response may be incomplete.]\n"
                        elif collected_content:
                            self.history.add(Message(role="assistant", content=collected_content))
                        break

                    # Add assistant message with ONLY valid tool calls
                    self.history.add(
                        Message(
                            role="assistant",
                            content=collected_content or "",
                            tool_calls=valid_tool_calls,
                        )
                    )

                    # Execute tools and collect results
                    for tool_call in valid_tool_calls:
                        yield f"\n\n> Executing: {tool_call['function']['name']}...\n"

                        try:
                            # Parse tool arguments with error handling for streaming issues
                            raw_args = tool_call["function"]["arguments"] or "{}"
                            try:
                                parsed_args = json.loads(raw_args)
                            except json.JSONDecodeError as json_err:
                                # Handle incomplete/malformed JSON from streaming
                                error_msg = str(json_err)
                                if "Unterminated string" in error_msg or "Expecting" in error_msg:
                                    # Try to fix common streaming issues
                                    fixed_args = self._try_fix_incomplete_json(raw_args)
                                    if fixed_args:
                                        parsed_args = fixed_args
                                    else:
                                        raise ValueError(
                                            f"Failed to parse tool arguments (streaming issue): {error_msg}. "
                                            f"This is a transient error - please retry your query."
                                        )
                                else:
                                    raise

                            result = tool_executor.execute(
                                tool_call["function"]["name"],
                                parsed_args,
                            )

                            # Ensure result is serializable
                            if result is None:
                                result = {"success": True, "message": "Tool executed successfully"}

                        except Exception as tool_error:
                            # If tool execution fails, still add a response to match the tool_call_id
                            error_str = str(tool_error)
                            # Check for streaming-related errors
                            is_transient = any(x in error_str.lower() for x in [
                                "unterminated", "streaming", "incomplete", "truncated"
                            ])
                            result = {
                                "success": False,
                                "error": error_str,
                                "message": f"Tool execution failed: {error_str}",
                                "is_transient": is_transient,
                                "suggestion": "Please retry your query" if is_transient else None,
                            }

                        all_tool_results.append(
                            {"tool_call_id": tool_call["id"], "result": result, "iteration": iteration}
                        )

                        # Add tool result to history - ALWAYS add a response for each tool_call_id
                        self.history.add(
                            Message(
                                role="tool",
                                content=json.dumps(result),
                                tool_call_id=tool_call["id"],
                            )
                        )

                    # Continue loop - agent will review results and decide next action

                # If we exited loop due to max iterations and have tool results,
                # make one final call to get the synthesis/response
                if iteration >= max_tool_iterations and all_tool_results:
                    yield f"\n\n[Max iterations reached - getting final response]\n"

                    # Clean up any orphaned tool messages before final call
                    self.cleanup_orphaned_tool_messages()

                    # Validate history before final API call
                    if not self.validate_history():
                        raise ValueError(
                            "Conversation history is invalid before final call. "
                            "This is a bug - please report with your query."
                        )

                    # Make final API call WITHOUT tools to force a text response
                    params = self._get_chat_params(include_tools=False, stream=True)
                    response = self.client.chat.completions.create(**params)

                    final_content = ""
                    usage_info = None

                    for chunk in response:
                        # Capture usage from final chunk
                        if hasattr(chunk, 'usage') and chunk.usage:
                            usage_info = {
                                "prompt_tokens": chunk.usage.prompt_tokens,
                                "completion_tokens": chunk.usage.completion_tokens,
                                "total_tokens": chunk.usage.total_tokens,
                            }

                        delta = chunk.choices[0].delta if chunk.choices else None
                        if delta and delta.content:
                            final_content += delta.content
                            yield delta.content

                    # Accumulate final token usage
                    if usage_info:
                        total_usage["prompt_tokens"] += usage_info.get("prompt_tokens", 0)
                        total_usage["completion_tokens"] += usage_info.get("completion_tokens", 0)
                        total_usage["total_tokens"] += usage_info.get("total_tokens", 0)

                    # Add final response to history
                    if final_content:
                        self.history.add(Message(role="assistant", content=final_content))

                # Return all tool results with iteration count and token usage
                if all_tool_results:
                    return {
                        "tool_results": all_tool_results,
                        "total_iterations": iteration,
                        "usage": total_usage if total_usage["total_tokens"] > 0 else None
                    }
                return {"usage": total_usage if total_usage["total_tokens"] > 0 else None}

            else:
                # Non-streaming agentic loop
                while iteration < max_tool_iterations:
                    iteration += 1

                    if iteration > 1:
                        yield f"\n[Iteration {iteration}/{max_tool_iterations}]\n"

                    # Clean up any orphaned tool messages before API call
                    self.cleanup_orphaned_tool_messages()

                    # Validate history before making API call
                    if not self.validate_history():
                        raise ValueError(
                            "Conversation history is invalid: some tool_calls have no matching responses. "
                            "This is a bug - please report with your query."
                        )

                    params = self._get_chat_params(include_tools=True, stream=False, additional_tools=additional_tools)
                    response = self.client.chat.completions.create(**params)

                    message = response.choices[0].message

                    # Filter valid tool calls
                    valid_tool_calls = [
                        tc for tc in (message.tool_calls or [])
                        if tc.id and tc.function and tc.function.name
                    ]

                    # If no valid tool calls, we're done
                    if not valid_tool_calls or not tool_executor:
                        content = message.content or ""
                        if content:
                            self.history.add(Message(role="assistant", content=content))
                            yield content
                        break

                    # Add assistant message with ONLY valid tool calls
                    self.history.add(
                        Message(
                            role="assistant",
                            content=message.content or "",
                            tool_calls=[tc.model_dump() for tc in valid_tool_calls],
                        )
                    )

                    # Execute tools
                    for tool_call in valid_tool_calls:
                        try:
                            # Parse tool arguments with error handling for streaming issues
                            raw_args = tool_call.function.arguments or "{}"
                            try:
                                parsed_args = json.loads(raw_args)
                            except json.JSONDecodeError as json_err:
                                # Handle incomplete/malformed JSON from streaming
                                error_msg = str(json_err)
                                if "Unterminated string" in error_msg or "Expecting" in error_msg:
                                    # Try to fix common streaming issues
                                    fixed_args = self._try_fix_incomplete_json(raw_args)
                                    if fixed_args:
                                        parsed_args = fixed_args
                                    else:
                                        raise ValueError(
                                            f"Failed to parse tool arguments (streaming issue): {error_msg}. "
                                            f"This is a transient error - please retry your query."
                                        )
                                else:
                                    raise

                            result = tool_executor.execute(
                                tool_call.function.name,
                                parsed_args,
                            )

                            # Ensure result is serializable
                            if result is None:
                                result = {"success": True, "message": "Tool executed successfully"}

                        except Exception as tool_error:
                            # If tool execution fails, still add a response to match the tool_call_id
                            error_str = str(tool_error)
                            # Check for streaming-related errors
                            is_transient = any(x in error_str.lower() for x in [
                                "unterminated", "streaming", "incomplete", "truncated"
                            ])
                            result = {
                                "success": False,
                                "error": error_str,
                                "message": f"Tool execution failed: {error_str}",
                                "is_transient": is_transient,
                                "suggestion": "Please retry your query" if is_transient else None,
                            }

                        all_tool_results.append(
                            {"tool_call_id": tool_call.id, "result": result, "iteration": iteration}
                        )

                        # ALWAYS add a response for each tool_call_id
                        self.history.add(
                            Message(
                                role="tool",
                                content=json.dumps(result),
                                tool_call_id=tool_call.id,
                            )
                        )

                # If we exited loop due to max iterations and have tool results,
                # make one final call to get the synthesis/response
                if iteration >= max_tool_iterations and all_tool_results:
                    yield f"\n[Max iterations reached - getting final response]\n"

                    # Clean up any orphaned tool messages before final call
                    self.cleanup_orphaned_tool_messages()

                    # Validate history before final API call
                    if not self.validate_history():
                        raise ValueError(
                            "Conversation history is invalid before final call. "
                            "This is a bug - please report with your query."
                        )

                    # Make final API call WITHOUT tools to force a text response
                    params = self._get_chat_params(include_tools=False, stream=False)
                    response = self.client.chat.completions.create(**params)

                    message = response.choices[0].message
                    final_content = message.content or ""

                    if final_content:
                        self.history.add(Message(role="assistant", content=final_content))
                        yield final_content

                # Return results
                if all_tool_results:
                    return {"tool_results": all_tool_results, "total_iterations": iteration}
                return None

        except Exception as e:
            error_str = str(e)

            # Provide helpful error message for tool call mismatches
            if "tool_call_id" in error_str and "did not have response messages" in error_str:
                error_msg = (
                    f"Error: Tool call history mismatch. This is usually a bug in the agentic loop.\n"
                    f"Please report this error with your query.\n"
                    f"Details: {error_str}"
                )
            else:
                error_msg = f"Error communicating with LLM: {error_str}"

            yield error_msg
            return None

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.history.clear()
        # Re-add system prompt
        system_prompt = self._load_system_prompt()
        self.history.add(Message(role="system", content=system_prompt))

    def get_history(self) -> List[Dict[str, Any]]:
        """Get conversation history."""
        return self.history.to_list()

    def validate_history(self) -> bool:
        """
        Validate conversation history for tool call/response matching.

        Checks both directions:
        1. Every assistant message with tool_calls has matching tool responses
        2. Every tool message has a preceding assistant message with tool_calls

        Returns:
            True if history is valid, False otherwise
        """
        messages = self.history.to_list()

        # Track all tool_call_ids we've seen from assistant messages
        seen_tool_call_ids = set()

        for i, msg in enumerate(messages):
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                # Add all tool_call_ids to seen set
                for tc in msg["tool_calls"]:
                    if "id" in tc:
                        seen_tool_call_ids.add(tc["id"])

            elif msg.get("role") == "tool":
                # CRITICAL CHECK: Every tool message MUST have a tool_call_id
                # that was previously declared in an assistant message
                tool_call_id = msg.get("tool_call_id")

                if not tool_call_id:
                    # Tool message has no tool_call_id - INVALID
                    print(f"ERROR: Tool message at position {i} has no tool_call_id")
                    return False

                if tool_call_id not in seen_tool_call_ids:
                    # Tool message references a tool_call_id we've never seen - INVALID
                    # This is the "orphaned tool message" bug
                    print(f"ERROR: Tool message at position {i} references unknown tool_call_id: {tool_call_id}")
                    print(f"  Known tool_call_ids: {seen_tool_call_ids}")
                    return False

        # Now check if all tool_calls got responses
        pending_tool_calls = set()

        for msg in messages:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    if "id" in tc:
                        pending_tool_calls.add(tc["id"])

            elif msg.get("role") == "tool":
                tool_call_id = msg.get("tool_call_id")
                if tool_call_id in pending_tool_calls:
                    pending_tool_calls.remove(tool_call_id)

        # If any pending_tool_calls remain, some tool_calls have no responses
        if pending_tool_calls:
            print(f"ERROR: {len(pending_tool_calls)} tool_calls have no responses: {pending_tool_calls}")
            return False

        return True

    def cleanup_orphaned_tool_messages(self) -> int:
        """
        Remove orphaned tool messages from history.

        An orphaned tool message is one that has a tool_call_id that doesn't
        match any assistant message with tool_calls that precedes it.

        Returns:
            Number of orphaned messages removed
        """
        # Track all tool_call_ids we've seen from assistant messages
        seen_tool_call_ids = set()
        messages_to_keep = []
        removed_count = 0

        for i, msg in enumerate(self.history.messages):
            if msg.role == "assistant" and msg.tool_calls:
                # Add all tool_call_ids to seen set
                for tc in msg.tool_calls:
                    if tc.get("id"):
                        seen_tool_call_ids.add(tc["id"])
                messages_to_keep.append(msg)

            elif msg.role == "tool":
                # Check if this tool message is orphaned
                tool_call_id = msg.tool_call_id

                if not tool_call_id or tool_call_id not in seen_tool_call_ids:
                    # Orphaned tool message - remove it
                    print(f"WARNING: Removing orphaned tool message at position {i} (tool_call_id: {tool_call_id})")
                    removed_count += 1
                else:
                    messages_to_keep.append(msg)

            else:
                # Keep all non-tool messages
                messages_to_keep.append(msg)

        # Update history with cleaned messages
        self.history.messages = messages_to_keep

        if removed_count > 0:
            print(f"Cleaned up {removed_count} orphaned tool message(s)")

        return removed_count
