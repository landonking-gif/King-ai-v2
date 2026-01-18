"""
Subagent lifecycle management.

Handles creation, execution, and cleanup of isolated subagent contexts.

Supports:
- Skill bindings (from SkillRegistry) - just-in-time loading
- MCP tool bindings (from MCP Gateway) - on-demand invocation
"""

import asyncio
import importlib
import importlib.util
import logging
import os
import sys
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import anyio
from anyio import create_task_group, fail_after

from adapters.llm import LLMAdapter, LLMMessage, MessageRole, MockLLMAdapter

from service.config import SubagentManagerConfig
from service.models import (
    MCPToolBinding,
    SkillBinding,
    SubagentExecuteRequest,
    SubagentInfo,
    SubagentResponse,
    SubagentRole,
    SubagentSpawnRequest,
    SubagentStatus,
)
from service.validator import SchemaValidator

logger = logging.getLogger(__name__)


class SubagentContext:
    """
    Isolated context for a single subagent.

    Each subagent maintains its own conversation history, capabilities,
    skill bindings, MCP tool bindings, and execution state.

    Skills are loaded just-in-time when first invoked.
    MCP tools are called on-demand via the gateway.
    """

    def __init__(
        self,
        subagent_id: str,
        role: SubagentRole,
        capabilities: List[str],
        skills: List[SkillBinding],
        mcp_tools: List[MCPToolBinding],
        system_prompt: str,
        llm_adapter: LLMAdapter,
        timeout: int,
        max_iterations: int,
        metadata: Dict[str, Any],
        mcp_gateway_url: Optional[str] = None,
    ) -> None:
        """
        Initialize subagent context.

        Args:
            subagent_id: Unique identifier
            role: Subagent role
            capabilities: List of enabled capabilities
            skills: Skill bindings from SkillRegistry
            mcp_tools: MCP tool bindings from gateway
            system_prompt: System prompt with instructions
            llm_adapter: LLM adapter instance
            timeout: Timeout in seconds
            max_iterations: Maximum task iterations
            metadata: Additional metadata
            mcp_gateway_url: URL for MCP gateway (for tool calls)
        """
        self.subagent_id = subagent_id
        self.role = role
        self.capabilities = capabilities
        self.skills = skills
        self.mcp_tools = mcp_tools
        self.system_prompt = system_prompt
        self.llm_adapter = llm_adapter
        self.timeout = timeout
        self.max_iterations = max_iterations
        self.metadata = metadata
        self.mcp_gateway_url = mcp_gateway_url or os.getenv(
            "MCP_GATEWAY_URL", "http://localhost:8080"
        )

        # State
        self.status = SubagentStatus.INITIALIZING
        self.created_at = datetime.utcnow()
        self.last_active = datetime.utcnow()
        self.executions = 0
        self.total_tokens = 0

        # Skill cache - stores loaded handler functions (just-in-time)
        self._skill_handlers: Dict[str, Callable] = {}
        self._skill_paths_added: set = set()

        # Conversation history
        self.conversation_history: List[LLMMessage] = []
        self._initialize_conversation()

    def _initialize_conversation(self) -> None:
        """Initialize conversation with system prompt."""
        # Format system prompt with capabilities
        formatted_prompt = self.llm_adapter.format_system_prompt(
            self.system_prompt, self.capabilities
        )
        self.conversation_history.append(
            LLMMessage(role=MessageRole.SYSTEM, content=formatted_prompt)
        )
        self.status = SubagentStatus.READY

    # =========================================================================
    # Skill Loading and Execution (Just-in-Time)
    # =========================================================================

    def get_skill_binding(self, skill_name: str) -> Optional[SkillBinding]:
        """
        Get skill binding by name.

        Args:
            skill_name: Name of the skill

        Returns:
            SkillBinding or None if not bound
        """
        for skill in self.skills:
            if skill.name == skill_name:
                return skill
        return None

    def is_skill_bound(self, skill_name: str) -> bool:
        """Check if a skill is bound to this subagent."""
        return self.get_skill_binding(skill_name) is not None

    def _load_skill_handler(self, skill: SkillBinding) -> Optional[Callable]:
        """
        Load a skill handler just-in-time.

        Uses lazy loading - handler is only imported when first needed.

        Args:
            skill: Skill binding with path and handler info

        Returns:
            Handler function or None if loading fails
        """
        if skill.name in self._skill_handlers:
            return self._skill_handlers[skill.name]

        if not skill.path or not skill.handler:
            logger.warning(f"Skill {skill.name} missing path or handler info")
            return None

        try:
            skill_path = Path(skill.path)
            parent_path = skill_path.parent

            # Add to Python path if not already added
            if str(parent_path) not in sys.path and str(parent_path) not in self._skill_paths_added:
                sys.path.insert(0, str(parent_path))
                self._skill_paths_added.add(str(parent_path))
                logger.debug(f"Added skill path to sys.path: {parent_path}")

            # Parse handler string (e.g., "handler.deep_research" or "deep_research")
            handler_str = skill.handler
            if "." in handler_str:
                module_name, func_name = handler_str.rsplit(".", 1)
            else:
                module_name = "handler"
                func_name = handler_str

            # Build full module name
            skill_dir_name = skill_path.name
            full_module_name = f"{skill_dir_name}.{module_name}"

            # Import the module
            module = importlib.import_module(full_module_name)

            # Get the handler function
            handler_func = getattr(module, func_name, None)
            if handler_func is None:
                logger.error(f"Handler function {func_name} not found in {full_module_name}")
                return None

            # Cache the handler
            self._skill_handlers[skill.name] = handler_func
            logger.info(f"Loaded skill handler: {skill.name} -> {full_module_name}.{func_name}")

            return handler_func

        except ImportError as e:
            logger.error(f"Failed to import skill {skill.name}: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to load skill handler {skill.name}: {e}")
            return None

    async def execute_skill(
        self,
        skill_name: str,
        args: Dict[str, Any],
        require_approval: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute a bound skill.

        Uses just-in-time loading - handler is only imported when first invoked.

        Args:
            skill_name: Name of the skill to execute
            args: Arguments to pass to the skill handler
            require_approval: Whether to check requires_approval flag

        Returns:
            Skill execution result
        """
        # Check if skill is bound
        skill = self.get_skill_binding(skill_name)
        if not skill:
            return {
                "success": False,
                "error": f"Skill '{skill_name}' is not bound to this subagent",
                "available_skills": [s.name for s in self.skills],
            }

        # Check approval requirement
        if require_approval and skill.requires_approval:
            return {
                "success": False,
                "error": f"Skill '{skill_name}' requires approval before execution",
                "requires_approval": True,
                "safety_flags": skill.safety_flags,
            }

        # Load handler (just-in-time)
        handler = self._load_skill_handler(skill)
        if handler is None:
            return {
                "success": False,
                "error": f"Failed to load handler for skill '{skill_name}'",
            }

        # Execute the skill
        try:
            start_time = time.time()

            # Check if handler is async
            if asyncio.iscoroutinefunction(handler):
                result = await handler(**args)
            else:
                # Run sync handler in thread pool
                result = await anyio.to_thread.run_sync(lambda: handler(**args))

            execution_time_ms = int((time.time() - start_time) * 1000)

            return {
                "success": True,
                "skill": skill_name,
                "result": result,
                "execution_time_ms": execution_time_ms,
            }

        except Exception as e:
            logger.error(f"Skill execution failed for {skill_name}: {e}")
            return {
                "success": False,
                "skill": skill_name,
                "error": str(e),
            }

    def list_bound_skills(self) -> List[Dict[str, Any]]:
        """
        List all skills bound to this subagent.

        Returns:
            List of skill info dictionaries
        """
        return [
            {
                "name": skill.name,
                "path": skill.path,
                "handler": skill.handler,
                "requires_approval": skill.requires_approval,
                "safety_flags": skill.safety_flags,
                "loaded": skill.name in self._skill_handlers,
            }
            for skill in self.skills
        ]

    # =========================================================================
    # MCP Tool Invocation (On-Demand via Gateway)
    # =========================================================================

    def get_mcp_tool_binding(
        self, server_id: str, tool_name: Optional[str] = None
    ) -> Optional[MCPToolBinding]:
        """
        Get MCP tool binding by server and tool name.

        Args:
            server_id: MCP server identifier
            tool_name: Specific tool name (optional)

        Returns:
            MCPToolBinding or None if not bound
        """
        for binding in self.mcp_tools:
            if binding.server_id != server_id:
                continue

            # Check for exact match, pattern match, or all_tools
            if binding.all_tools:
                return binding
            if binding.tool_name == tool_name:
                return binding
            if binding.tool_pattern and tool_name:
                import fnmatch
                if fnmatch.fnmatch(tool_name, binding.tool_pattern):
                    return binding

        return None

    def is_mcp_tool_bound(self, server_id: str, tool_name: str) -> bool:
        """Check if an MCP tool is bound to this subagent."""
        return self.get_mcp_tool_binding(server_id, tool_name) is not None

    async def invoke_mcp_tool(
        self,
        server_id: str,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Invoke an MCP tool via the gateway.

        Args:
            server_id: MCP server identifier
            tool_name: Tool name to invoke
            arguments: Tool arguments

        Returns:
            Tool invocation result
        """
        # Check if tool is bound
        binding = self.get_mcp_tool_binding(server_id, tool_name)
        if not binding:
            return {
                "success": False,
                "error": f"MCP tool '{server_id}.{tool_name}' is not bound to this subagent",
                "available_tools": [
                    f"{t.server_id}.{t.tool_name or '*'}" for t in self.mcp_tools
                ],
            }

        # Call MCP gateway
        try:
            import httpx

            start_time = time.time()

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.mcp_gateway_url}/tools/{server_id}/{tool_name}",
                    json={
                        "arguments": arguments,
                        "caller_id": self.subagent_id,
                        "scopes": binding.scopes,
                    },
                )
                response.raise_for_status()
                result = response.json()

            execution_time_ms = int((time.time() - start_time) * 1000)

            return {
                "success": True,
                "server_id": server_id,
                "tool_name": tool_name,
                "result": result,
                "execution_time_ms": execution_time_ms,
            }

        except httpx.HTTPStatusError as e:
            logger.error(f"MCP tool call failed: {e}")
            return {
                "success": False,
                "server_id": server_id,
                "tool_name": tool_name,
                "error": f"HTTP {e.response.status_code}: {e.response.text}",
            }
        except Exception as e:
            logger.error(f"MCP tool invocation failed: {e}")
            return {
                "success": False,
                "server_id": server_id,
                "tool_name": tool_name,
                "error": str(e),
            }

    def list_bound_mcp_tools(self) -> List[Dict[str, Any]]:
        """
        List all MCP tools bound to this subagent.

        Returns:
            List of MCP tool info dictionaries
        """
        return [
            {
                "server_id": tool.server_id,
                "tool_name": tool.tool_name,
                "tool_pattern": tool.tool_pattern,
                "all_tools": tool.all_tools,
                "scopes": tool.scopes,
                "rate_limit": tool.rate_limit,
            }
            for tool in self.mcp_tools
        ]

    def is_expired(self, max_lifetime: int) -> bool:
        """
        Check if subagent has exceeded its maximum lifetime.

        Args:
            max_lifetime: Maximum lifetime in seconds

        Returns:
            True if expired
        """
        age = (datetime.utcnow() - self.created_at).total_seconds()
        return age > max_lifetime

    def is_idle(self, idle_timeout: int) -> bool:
        """
        Check if subagent has been idle too long.

        Args:
            idle_timeout: Idle timeout in seconds

        Returns:
            True if idle
        """
        idle_time = (datetime.utcnow() - self.last_active).total_seconds()
        return idle_time > idle_timeout

    def get_info(self) -> SubagentInfo:
        """
        Get information about this subagent.

        Returns:
            SubagentInfo object
        """
        return SubagentInfo(
            subagent_id=self.subagent_id,
            role=self.role,
            status=self.status,
            capabilities=self.capabilities,
            skills=self.skills,
            mcp_tools=self.mcp_tools,
            created_at=self.created_at,
            last_active=self.last_active,
            timeout=self.timeout,
            executions=self.executions,
            total_tokens=self.total_tokens,
            metadata=self.metadata,
        )


class SubagentLifecycleManager:
    """
    Manages lifecycle of all subagent instances.

    Handles spawning, execution, timeout enforcement, and cleanup.
    """

    def __init__(self, config: SubagentManagerConfig, validator: SchemaValidator) -> None:
        """
        Initialize lifecycle manager.

        Args:
            config: Configuration instance
            validator: Schema validator instance
        """
        self.config = config
        self.validator = validator
        self.subagents: Dict[str, SubagentContext] = {}
        self._cleanup_task: Optional[Any] = None

    async def start(self) -> None:
        """Start the lifecycle manager and background tasks."""
        # Start cleanup task
        async with create_task_group() as tg:
            self._cleanup_task = tg.start_soon(self._cleanup_loop)

    async def _cleanup_loop(self) -> None:
        """Background task to cleanup expired subagents."""
        while True:
            await anyio.sleep(self.config.cleanup_interval)
            await self._cleanup_expired()

    async def _cleanup_expired(self) -> None:
        """Remove expired and idle subagents."""
        to_remove = []

        for subagent_id, context in self.subagents.items():
            if context.is_expired(self.config.max_lifetime):
                context.status = SubagentStatus.TIMEOUT
                to_remove.append(subagent_id)
            elif context.status in (
                SubagentStatus.COMPLETED,
                SubagentStatus.FAILED,
                SubagentStatus.DESTROYED,
            ):
                # Cleanup finished subagents after some time
                if context.is_idle(300):  # 5 minutes
                    to_remove.append(subagent_id)

        for subagent_id in to_remove:
            del self.subagents[subagent_id]

    def _create_llm_adapter(self) -> LLMAdapter:
        """
        Create an LLM adapter based on configuration.

        Returns:
            LLM adapter instance
        """
        provider = self.config.llm_provider.lower()

        if provider == "mock":
            return MockLLMAdapter(model=self.config.llm_model)

        elif provider == "anthropic":
            from adapters.llm.anthropic import AnthropicAdapter

            return AnthropicAdapter(
                model=self.config.llm_model,
                api_key=self.config.anthropic_api_key,
            )

        elif provider == "openai":
            from adapters.llm.openai import OpenAIAdapter

            return OpenAIAdapter(
                model=self.config.llm_model,
                api_key=self.config.openai_api_key,
            )

        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

    async def spawn_subagent(self, request: SubagentSpawnRequest) -> SubagentInfo:
        """
        Spawn a new subagent instance.

        Args:
            request: Spawn request with configuration

        Returns:
            Information about the created subagent

        Raises:
            ValueError: If max concurrent subagents exceeded
        """
        # Check resource limits
        if len(self.subagents) >= self.config.max_concurrent_subagents:
            raise ValueError(
                f"Maximum concurrent subagents ({self.config.max_concurrent_subagents}) exceeded"
            )

        # Generate unique ID
        subagent_id = f"{request.role.value}-{uuid.uuid4().hex[:8]}"

        # Create LLM adapter
        llm_adapter = self._create_llm_adapter()

        # Create subagent context with skills and MCP tools
        timeout = request.timeout or self.config.default_timeout
        context = SubagentContext(
            subagent_id=subagent_id,
            role=request.role,
            capabilities=request.capabilities,
            skills=request.skills,
            mcp_tools=request.mcp_tools,
            system_prompt=request.system_prompt,
            llm_adapter=llm_adapter,
            timeout=timeout,
            max_iterations=request.max_iterations,
            metadata=request.metadata,
            mcp_gateway_url=self.config.mcp_gateway_url,
        )

        # Log skill/tool bindings
        if request.skills:
            logger.info(
                f"Subagent {subagent_id} spawned with {len(request.skills)} skills: "
                f"{[s.name for s in request.skills]}"
            )
        if request.mcp_tools:
            logger.info(
                f"Subagent {subagent_id} spawned with {len(request.mcp_tools)} MCP tool bindings"
            )

        # Store context
        self.subagents[subagent_id] = context

        return context.get_info()

    def get_context(self, subagent_id: str) -> Optional[SubagentContext]:
        """
        Get context for a specific subagent.

        Args:
            subagent_id: The subagent identifier

        Returns:
            SubagentContext if found, None otherwise
        """
        return self.subagents.get(subagent_id)

    async def execute_task(self, request: SubagentExecuteRequest) -> SubagentResponse:
        """
        Execute a task with a subagent.

        Args:
            request: Execution request

        Returns:
            Subagent response with validated output

        Raises:
            ValueError: If subagent not found
            TimeoutError: If execution times out
        """
        # Get subagent context
        if request.subagent_id not in self.subagents:
            raise ValueError(f"Subagent {request.subagent_id} not found")

        context = self.subagents[request.subagent_id]

        # Update status
        context.status = SubagentStatus.EXECUTING
        context.last_active = datetime.utcnow()

        start_time = time.time()
        timeout = request.timeout or context.timeout

        try:
            # Execute with timeout
            async with fail_after(timeout):
                response = await self._execute_with_llm(context, request)

            context.status = SubagentStatus.COMPLETED
            return response

        except TimeoutError:
            context.status = SubagentStatus.TIMEOUT
            return SubagentResponse(
                subagent_id=request.subagent_id,
                status=SubagentStatus.TIMEOUT,
                error=f"Execution timed out after {timeout} seconds",
                execution_time_ms=int((time.time() - start_time) * 1000),
            )

        except Exception as e:
            context.status = SubagentStatus.FAILED
            return SubagentResponse(
                subagent_id=request.subagent_id,
                status=SubagentStatus.FAILED,
                error=str(e),
                execution_time_ms=int((time.time() - start_time) * 1000),
            )

    async def _execute_with_llm(
        self, context: SubagentContext, request: SubagentExecuteRequest
    ) -> SubagentResponse:
        """
        Execute task using LLM adapter.

        Args:
            context: Subagent context
            request: Execution request

        Returns:
            Validated response
        """
        start_time = time.time()

        # Add user message to conversation
        user_message = self._format_task_message(request.task, request.inputs)
        context.conversation_history.append(
            LLMMessage(role=MessageRole.USER, content=user_message)
        )

        # Call LLM
        llm_response = await context.llm_adapter.complete(
            messages=context.conversation_history,
            temperature=self.config.llm_temperature,
            max_tokens=self.config.llm_max_tokens,
        )

        # Add assistant response to history
        context.conversation_history.append(
            LLMMessage(role=MessageRole.ASSISTANT, content=llm_response.content)
        )

        # Update token usage
        context.total_tokens += llm_response.usage.get("total_tokens", 0)
        context.executions += 1

        # Parse and validate output
        output = None
        error = None

        if request.expected_output_schema:
            output, error = await self._validate_output(
                llm_response.content, request.expected_output_schema
            )

        execution_time_ms = int((time.time() - start_time) * 1000)

        return SubagentResponse(
            subagent_id=request.subagent_id,
            status=SubagentStatus.COMPLETED if output else SubagentStatus.FAILED,
            output=output,
            raw_response=llm_response.content,
            error=error,
            iterations=1,
            tokens_used=llm_response.usage,
            execution_time_ms=execution_time_ms,
        )

    def _format_task_message(self, task: str, inputs: Dict[str, Any]) -> str:
        """
        Format task and inputs into a user message.

        Args:
            task: Task description
            inputs: Task inputs

        Returns:
            Formatted message
        """
        if not inputs:
            return task

        inputs_str = "\n".join(f"- {k}: {v}" for k, v in inputs.items())
        return f"{task}\n\nInputs:\n{inputs_str}"

    async def _validate_output(
        self, content: str, schema_name: str
    ) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Parse and validate LLM output against schema.

        Args:
            content: LLM response content
            schema_name: Expected schema name

        Returns:
            Tuple of (validated_output, error_message)
        """
        try:
            # Try to extract JSON from the response
            import json
            import re

            # Look for JSON in code blocks or raw JSON
            json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find raw JSON
                json_match = re.search(r"\{.*\}", content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    return None, "No JSON found in response"

            # Parse JSON
            output = json.loads(json_str)

            # Validate against schema
            is_valid, error = self.validator.validate(output, schema_name)
            if is_valid:
                return output, None
            else:
                return None, f"Schema validation failed: {error}"

        except json.JSONDecodeError as e:
            return None, f"Failed to parse JSON: {str(e)}"
        except Exception as e:
            return None, f"Validation error: {str(e)}"

    async def destroy_subagent(self, subagent_id: str) -> bool:
        """
        Destroy a subagent and cleanup its context.

        Args:
            subagent_id: Subagent identifier

        Returns:
            True if destroyed, False if not found
        """
        if subagent_id not in self.subagents:
            return False

        context = self.subagents[subagent_id]
        context.status = SubagentStatus.DESTROYED
        context.conversation_history.clear()

        # Remove from active subagents
        del self.subagents[subagent_id]

        return True

    async def get_subagent_status(self, subagent_id: str) -> Optional[SubagentInfo]:
        """
        Get status of a subagent.

        Args:
            subagent_id: Subagent identifier

        Returns:
            Subagent info or None if not found
        """
        if subagent_id not in self.subagents:
            return None

        return self.subagents[subagent_id].get_info()

    async def list_subagents(self) -> List[SubagentInfo]:
        """
        List all active subagents.

        Returns:
            List of subagent info
        """
        return [context.get_info() for context in self.subagents.values()]
