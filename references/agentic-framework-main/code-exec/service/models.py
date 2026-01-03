"""
Pydantic models for Code Executor Service.
Module: code-exec/service/models.py
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class SafetyFlag(str, Enum):
    """Safety flags for skill classification."""

    NONE = "none"
    PII_RISK = "pii_risk"
    EXTERNAL_CALL = "external_call"
    SIDE_EFFECT = "side_effect"
    FILE_SYSTEM = "file_system"
    NETWORK_ACCESS = "network_access"


class SkillMetadata(BaseModel):
    """Metadata for a registered skill."""

    name: str = Field(..., description="Unique skill identifier")
    version: str = Field(..., description="Skill version (semver)")
    description: str = Field(..., description="Human-readable description")
    safety_flags: List[SafetyFlag] = Field(
        default_factory=list, description="Safety classifications"
    )
    requires_approval: bool = Field(
        default=False, description="Requires policy approval before execution"
    )
    input_schema: Dict[str, Any] = Field(..., description="JSON Schema for input validation")
    output_schema: Dict[str, Any] = Field(
        ..., description="JSON Schema for output validation"
    )
    handler_path: str = Field(..., description="Python module path to handler function")
    tags: List[str] = Field(default_factory=list, description="Searchable tags")
    created_at: datetime = Field(default_factory=datetime.utcnow)

    @field_validator("version")
    @classmethod
    def validate_version(cls, v: str) -> str:
        """Validate semantic versioning format."""
        parts = v.split(".")
        if len(parts) != 3 or not all(p.isdigit() for p in parts):
            raise ValueError("Version must follow semver format (e.g., 1.0.0)")
        return v


class ExecutionRequest(BaseModel):
    """Request to execute a skill."""

    skill: str = Field(..., description="Skill name to execute")
    args: Dict[str, Any] = Field(..., description="Skill input arguments")
    request_id: Optional[str] = Field(default=None, description="Optional request tracking ID")
    actor_id: Optional[str] = Field(default=None, description="ID of requesting actor/agent")
    actor_type: Optional[str] = Field(
        default=None, description="Type of actor (subagent, orchestrator, etc.)"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional request metadata"
    )


class ProvenanceRecord(BaseModel):
    """Provenance tracking for skill execution."""

    execution_id: str = Field(..., description="Unique execution identifier")
    skill_name: str = Field(..., description="Executed skill name")
    skill_version: str = Field(..., description="Executed skill version")
    actor_id: Optional[str] = Field(default=None, description="Requesting actor ID")
    actor_type: Optional[str] = Field(default=None, description="Requesting actor type")
    inputs_hash: str = Field(..., description="SHA-256 hash of input arguments")
    outputs_hash: str = Field(..., description="SHA-256 hash of output result")
    tool_ids: List[str] = Field(default_factory=list, description="External tools called")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    execution_time_ms: float = Field(..., description="Execution duration in milliseconds")
    success: bool = Field(..., description="Whether execution succeeded")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")


class ExecutionLog(BaseModel):
    """Execution log entry."""

    timestamp: datetime = Field(default_factory=datetime.utcnow)
    level: str = Field(..., description="Log level (INFO, WARNING, ERROR)")
    message: str = Field(..., description="Log message")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context")


class ExecutionResult(BaseModel):
    """Result of skill execution."""

    execution_id: str = Field(..., description="Unique execution identifier")
    skill: str = Field(..., description="Executed skill name")
    result: Any = Field(..., description="Skill output (validated against output schema)")
    logs: List[ExecutionLog] = Field(default_factory=list, description="Execution logs")
    inputs_hash: str = Field(..., description="SHA-256 hash of inputs")
    outputs_hash: str = Field(..., description="SHA-256 hash of outputs")
    provenance: ProvenanceRecord = Field(..., description="Full provenance record")
    execution_time_ms: float = Field(..., description="Execution time in milliseconds")
    success: bool = Field(default=True, description="Whether execution succeeded")


class SkillRegistrationRequest(BaseModel):
    """Request to register a new skill."""

    name: str = Field(..., description="Skill name")
    version: str = Field(..., description="Skill version (semver)")
    description: str = Field(..., description="Skill description")
    safety_flags: List[SafetyFlag] = Field(default_factory=list)
    requires_approval: bool = Field(default=False)
    input_schema: Dict[str, Any] = Field(..., description="JSON Schema for inputs")
    output_schema: Dict[str, Any] = Field(..., description="JSON Schema for outputs")
    handler_path: str = Field(..., description="Path to handler function")
    tags: Optional[List[str]] = Field(default_factory=list)


class SkillListResponse(BaseModel):
    """Response containing list of available skills."""

    skills: List[SkillMetadata] = Field(..., description="List of registered skills")
    total: int = Field(..., description="Total number of skills")


class SkillSchemaResponse(BaseModel):
    """Response containing skill I/O schemas."""

    skill_name: str = Field(..., description="Skill name")
    version: str = Field(..., description="Skill version")
    input_schema: Dict[str, Any] = Field(..., description="Input JSON Schema")
    output_schema: Dict[str, Any] = Field(..., description="Output JSON Schema")


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional details")
    execution_id: Optional[str] = Field(default=None, description="Execution ID if applicable")
