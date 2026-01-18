"""
Pydantic models for Subagent Manager API.

Supports:
- Role-based capabilities
- Skill bindings (from SkillRegistry)
- MCP tool bindings (from MCP Gateway)
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class SubagentRole(str, Enum):
    """Predefined subagent roles."""

    RESEARCH = "research"
    VERIFY = "verify"
    CODE = "code"
    SYNTHESIS = "synthesis"
    CUSTOM = "custom"


class SubagentStatus(str, Enum):
    """Subagent lifecycle status."""

    INITIALIZING = "initializing"
    READY = "ready"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    DESTROYED = "destroyed"


class SkillBinding(BaseModel):
    """Binding to a deterministic skill from SkillRegistry."""

    name: str = Field(..., description="Skill name (from SkillRegistry)")
    path: Optional[str] = Field(default=None, description="Path to skill directory")
    handler: Optional[str] = Field(
        default=None, description="Handler function (module.function)"
    )
    requires_approval: bool = Field(
        default=False, description="Requires human approval before execution"
    )
    safety_flags: List[str] = Field(
        default_factory=list,
        description="Safety flags: pii_risk, external_call, side_effect, etc.",
    )


class MCPToolBinding(BaseModel):
    """Binding to an MCP tool from MCP Gateway."""

    server_id: str = Field(..., description="MCP server identifier")
    tool_name: Optional[str] = Field(
        default=None, description="Specific tool name (None for pattern/all)"
    )
    tool_pattern: Optional[str] = Field(
        default=None, description="Tool name pattern (e.g., 'read_*')"
    )
    all_tools: bool = Field(
        default=False, description="Bind all tools from this server"
    )
    scopes: List[str] = Field(
        default_factory=list, description="Required scopes for this tool"
    )
    rate_limit: Optional[int] = Field(
        default=None, description="Rate limit override (calls/minute)"
    )


class SubagentSpawnRequest(BaseModel):
    """Request to spawn a new subagent."""

    role: SubagentRole = Field(..., description="Subagent role")
    capabilities: List[str] = Field(
        default_factory=list, description="List of enabled capabilities/tools"
    )
    skills: List[SkillBinding] = Field(
        default_factory=list,
        description="Skills to bind (from SkillRegistry)",
    )
    mcp_tools: List[MCPToolBinding] = Field(
        default_factory=list,
        description="MCP tools to bind (from MCP Gateway)",
    )
    system_prompt: str = Field(..., description="System prompt for the subagent")
    timeout: Optional[int] = Field(
        default=None, description="Timeout in seconds (None for default)"
    )
    max_iterations: int = Field(default=10, description="Maximum task iterations")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class SubagentExecuteRequest(BaseModel):
    """Request to execute a task with a subagent."""

    subagent_id: str = Field(..., description="Subagent identifier")
    task: str = Field(..., description="Task description or prompt")
    inputs: Dict[str, Any] = Field(
        default_factory=dict, description="Task inputs/context"
    )
    expected_output_schema: Optional[str] = Field(
        default=None, description="Expected artifact schema name"
    )
    timeout: Optional[int] = Field(
        default=None, description="Override timeout for this execution"
    )


class SubagentResponse(BaseModel):
    """Response from a subagent execution."""

    subagent_id: str = Field(..., description="Subagent identifier")
    status: SubagentStatus = Field(..., description="Execution status")
    output: Optional[Dict[str, Any]] = Field(
        default=None, description="Validated output artifact"
    )
    raw_response: Optional[str] = Field(
        default=None, description="Raw LLM response"
    )
    error: Optional[str] = Field(default=None, description="Error message if failed")
    iterations: int = Field(default=0, description="Number of iterations performed")
    tokens_used: Dict[str, int] = Field(
        default_factory=dict, description="Token usage statistics"
    )
    execution_time_ms: int = Field(default=0, description="Execution time in milliseconds")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class SubagentInfo(BaseModel):
    """Information about a subagent instance."""

    subagent_id: str = Field(..., description="Subagent identifier")
    role: SubagentRole = Field(..., description="Subagent role")
    status: SubagentStatus = Field(..., description="Current status")
    capabilities: List[str] = Field(..., description="Enabled capabilities")
    skills: List[SkillBinding] = Field(
        default_factory=list, description="Bound skills"
    )
    mcp_tools: List[MCPToolBinding] = Field(
        default_factory=list, description="Bound MCP tools"
    )
    created_at: datetime = Field(..., description="Creation timestamp")
    last_active: datetime = Field(..., description="Last activity timestamp")
    timeout: int = Field(..., description="Timeout in seconds")
    executions: int = Field(default=0, description="Number of executions")
    total_tokens: int = Field(default=0, description="Total tokens used")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class SubagentDestroyRequest(BaseModel):
    """Request to destroy a subagent."""

    subagent_id: str = Field(..., description="Subagent identifier")
    reason: Optional[str] = Field(default=None, description="Destruction reason")


class SubagentListResponse(BaseModel):
    """Response listing all subagents."""

    subagents: List[SubagentInfo] = Field(..., description="List of subagent instances")
    total: int = Field(..., description="Total number of subagents")
    active: int = Field(..., description="Number of active subagents")
