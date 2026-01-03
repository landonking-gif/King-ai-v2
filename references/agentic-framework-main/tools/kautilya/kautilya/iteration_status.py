"""
Dynamic iteration status extraction for better UX transparency.

Module: kautilya/iteration_status.py

Extracts meaningful status information from LLM responses and tool executions.
"""

import re
from typing import Dict, Optional, Any, List


class IterationStatusExtractor:
    """Extracts dynamic status information from agent execution."""

    # Tool purpose mappings for better descriptions
    TOOL_PURPOSES = {
        "read_file": "Reading file to understand current code",
        "glob_files": "Searching for relevant files in the codebase",
        "grep_code": "Searching codebase for specific patterns",
        "bash_execute": "Running shell command",
        "write_file": "Creating or updating file",
        "edit_file": "Modifying existing file",
        "llm_config": "Configuring LLM provider settings",
        "llm_test": "Testing LLM connection",
        "llm_list": "Listing available LLM providers",
        "llm_set_params": "Updating LLM hyperparameters",
        "llm_show_params": "Checking current LLM hyperparameters",
        "web_search": "Searching the web for information",
        "web_fetch": "Fetching content from URL",
        "kautilya_init": "Initializing new agent project",
        "skill_new": "Creating new skill",
        "skill_import": "Importing skill from external source",
        "mcp_add": "Adding MCP server",
        "mcp_list": "Listing available MCP servers",
    }

    # Reasoning indicators - patterns that suggest the LLM is thinking
    REASONING_PATTERNS = [
        r"(?:Let me|I'll|I will|I need to|First,|Next,|Then,)\s+(.{10,80})",
        r"(?:To solve this|To accomplish|To help with)\s+(.{10,80})",
        r"(?:I can|I should|I'm going to)\s+(.{10,80})",
        r"(?:The best approach|My plan|The strategy)\s+(?:is|would be)\s+(.{10,80})",
    ]

    @classmethod
    def extract_planning_info(cls, content: str, iteration: int) -> str:
        """
        Extract planning/reasoning information from LLM content.

        Args:
            content: The LLM's response content
            iteration: Current iteration number

        Returns:
            Descriptive status message
        """
        if not content or len(content.strip()) < 10:
            return f"Starting iteration {iteration} - analyzing the request"

        # Try to extract reasoning from patterns
        for pattern in cls.REASONING_PATTERNS:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                extracted = match.group(1).strip()
                # Clean up and truncate
                if extracted.endswith(('.', ',')):
                    extracted = extracted[:-1]
                if len(extracted) > 70:
                    extracted = extracted[:67] + "..."
                return f"Planning: {extracted}"

        # Extract first meaningful sentence
        sentences = re.split(r'[.!?]\s+', content.strip())
        for sentence in sentences:
            if len(sentence) > 15 and not sentence.startswith('>'):
                sentence = sentence.strip()
                if len(sentence) > 70:
                    sentence = sentence[:67] + "..."
                return f"Strategy: {sentence}"

        return f"Iteration {iteration} - formulating approach"

    @classmethod
    def extract_tool_purpose(cls, tool_name: str, args: Optional[Dict[str, Any]] = None) -> str:
        """
        Extract the purpose of a tool execution with context from arguments.

        Args:
            tool_name: Name of the tool being executed
            args: Optional tool arguments

        Returns:
            Descriptive status message
        """
        # Get base purpose
        base_purpose = cls.TOOL_PURPOSES.get(tool_name, f"Executing {tool_name}")

        # Enhance with argument context if available
        if args:
            if tool_name in ["read_file", "write_file", "edit_file"]:
                if "file_path" in args:
                    file_path = args["file_path"]
                    # Extract just the filename
                    filename = file_path.split('/')[-1]
                    return f"{base_purpose}: {filename}"

            elif tool_name == "glob_files":
                if "pattern" in args:
                    return f"Searching for files matching: {args['pattern']}"

            elif tool_name == "grep_code":
                if "pattern" in args:
                    pattern = args["pattern"]
                    if len(pattern) > 40:
                        pattern = pattern[:37] + "..."
                    return f"Searching code for: {pattern}"

            elif tool_name == "bash_execute":
                if "command" in args:
                    cmd = args["command"]
                    # Get first part of command
                    cmd_parts = cmd.split()
                    if cmd_parts:
                        base_cmd = cmd_parts[0]
                        if base_cmd in ["git", "npm", "pip", "pytest", "mypy", "black"]:
                            return f"Running {base_cmd} command"
                    if len(cmd) > 40:
                        cmd = cmd[:37] + "..."
                    return f"Running: {cmd}"

            elif tool_name == "llm_set_params":
                if "provider" in args:
                    return f"Configuring {args['provider']} hyperparameters"

            elif tool_name == "llm_config":
                if "provider" in args:
                    return f"Setting up {args['provider']} provider"

            elif tool_name == "web_search":
                if "query" in args:
                    query = args["query"]
                    if len(query) > 40:
                        query = query[:37] + "..."
                    return f"Searching web for: {query}"

            elif tool_name == "skill_new":
                if "name" in args:
                    return f"Creating skill: {args['name']}"

        return base_purpose

    @classmethod
    def extract_review_info(cls, content: str, tool_name: Optional[str] = None) -> str:
        """
        Extract what the agent is doing after tool execution.

        Args:
            content: The LLM's response content after tool execution
            tool_name: Name of the tool that was just executed

        Returns:
            Descriptive status message
        """
        if not content or len(content.strip()) < 10:
            if tool_name:
                return f"Analyzing {tool_name} results"
            return "Processing results and planning next steps"

        # Look for action keywords in recent content
        action_patterns = [
            (r"(?:Now|Next)\s+(?:I'll|I will|let me)\s+(.{10,60})", "Next: "),
            (r"(?:Based on|After reviewing|Looking at)\s+(.{10,60})", "Reviewing: "),
            (r"(?:I found|I see|I notice)\s+(.{10,60})", "Found: "),
            (r"(?:I need to|I should)\s+(.{10,60})", "Next step: "),
        ]

        for pattern, prefix in action_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                extracted = match.group(1).strip()
                if extracted.endswith(('.', ',')):
                    extracted = extracted[:-1]
                if len(extracted) > 55:
                    extracted = extracted[:52] + "..."
                return f"{prefix}{extracted}"

        # Default based on tool
        if tool_name:
            return f"Analyzing {tool_name} results and determining next action"

        return "Evaluating progress and planning next steps"

    @classmethod
    def extract_completion_summary(cls, content: str, tools_used: List[str]) -> str:
        """
        Extract a summary of what was accomplished.

        Args:
            content: Final response content
            tools_used: List of tools that were used

        Returns:
            Summary message
        """
        tool_count = len(tools_used)

        if tool_count == 0:
            return "Task completed with direct response"

        # Categorize tools
        read_tools = [t for t in tools_used if t in ["read_file", "glob_files", "grep_code"]]
        write_tools = [t for t in tools_used if t in ["write_file", "edit_file"]]
        exec_tools = [t for t in tools_used if t in ["bash_execute"]]
        config_tools = [t for t in tools_used if t.startswith("llm_") or t.startswith("mcp_")]

        summary_parts = []
        if read_tools:
            summary_parts.append(f"analyzed {len(read_tools)} file(s)")
        if write_tools:
            summary_parts.append(f"modified {len(write_tools)} file(s)")
        if exec_tools:
            summary_parts.append(f"executed {len(exec_tools)} command(s)")
        if config_tools:
            summary_parts.append(f"updated {len(config_tools)} configuration(s)")

        if summary_parts:
            return f"Completed: {', '.join(summary_parts)}"

        return f"Task completed using {tool_count} tool(s)"


class ResponseBuffer:
    """Buffers LLM response content for analysis."""

    def __init__(self, max_size: int = 500):
        """Initialize response buffer.

        Args:
            max_size: Maximum characters to buffer
        """
        self.buffer = ""
        self.max_size = max_size

    def add(self, chunk: str) -> None:
        """Add a chunk to the buffer."""
        self.buffer += chunk
        # Keep only the most recent content
        if len(self.buffer) > self.max_size:
            self.buffer = self.buffer[-self.max_size:]

    def get(self) -> str:
        """Get current buffer content."""
        return self.buffer

    def clear(self) -> None:
        """Clear the buffer."""
        self.buffer = ""

    def get_last_sentence(self) -> str:
        """Get the last complete sentence from buffer."""
        # Find the last sentence boundary
        sentences = re.split(r'[.!?]\s+', self.buffer.strip())
        if len(sentences) > 0:
            return sentences[-1].strip()
        return self.buffer.strip()
