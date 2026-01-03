"""
Pydantic models for the Orchestrator service.

Defines all data structures for workflows, artifacts, subagent requests, and API contracts.
All models use strict type hints and validation per team standards.
"""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator, ConfigDict


# ============================================================================
# Enums
# ============================================================================


class SubagentRole(str, Enum):
    """Supported subagent roles."""

    RESEARCH = "research"
    VERIFY = "verify"
    CODE = "code"
    SYNTHESIS = "synthesis"


class ArtifactType(str, Enum):
    """Supported artifact types."""

    RESEARCH_SNIPPET = "research_snippet"
    CLAIM_VERIFICATION = "claim_verification"
    CODE_PATCH = "code_patch"
    SYNTHESIS_RESULT = "synthesis_result"


class WorkflowStatus(str, Enum):
    """Workflow execution status."""

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SafetyClass(str, Enum):
    """Safety classification for artifacts."""

    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    PII_RISK = "pii_risk"


class CompactionStrategy(str, Enum):
    """Memory compaction strategies."""

    SUMMARIZE = "summarize"
    TRUNCATE = "truncate"
    NONE = "none"


class Verdict(str, Enum):
    """Claim verification verdict."""

    VERIFIED = "verified"
    REFUTED = "refuted"
    INCONCLUSIVE = "inconclusive"
    NEEDS_MORE_INFO = "needs_more_info"


# ============================================================================
# Provenance Models
# ============================================================================


class ProvenanceRecord(BaseModel):
    """Provenance tracking for all operations."""

    model_config = ConfigDict(frozen=True)

    actor_id: str = Field(..., description="ID of the actor (subagent, user, system)")
    actor_type: str = Field(
        ..., description="Type of actor (subagent, user, system, skill)"
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    inputs_hash: str = Field(..., description="SHA256 hash of inputs")
    outputs_hash: str = Field(..., description="SHA256 hash of outputs")
    tool_ids: List[str] = Field(
        default_factory=list, description="List of tools/skills used"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


# ============================================================================
# Artifact Models
# ============================================================================


class ArtifactBase(BaseModel):
    """Base model for all typed artifacts."""

    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique artifact ID")
    artifact_type: ArtifactType = Field(..., description="Type of artifact")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: str = Field(..., description="ID of creator (subagent or user)")
    provenance: ProvenanceRecord = Field(..., description="Provenance record")
    safety_class: SafetyClass = Field(
        default=SafetyClass.INTERNAL, description="Safety classification"
    )


class ResearchSource(BaseModel):
    """Source information for research snippets."""

    url: Optional[str] = Field(default=None, description="Source URL if web-based")
    doc_id: Optional[str] = Field(default=None, description="Document ID if internal")
    title: Optional[str] = Field(default=None, description="Source title")


class ResearchSnippet(ArtifactBase):
    """Research snippet artifact produced by research subagents."""

    artifact_type: ArtifactType = Field(default=ArtifactType.RESEARCH_SNIPPET)
    source: ResearchSource = Field(..., description="Source of the research")
    text: str = Field(..., min_length=1, description="Full text of research snippet")
    summary: str = Field(..., min_length=1, description="Concise summary")
    tags: List[str] = Field(default_factory=list, description="Categorical tags")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score (0.0-1.0)"
    )
    provenance_refs: List[str] = Field(
        default_factory=list, description="References to other artifacts"
    )
    embedding_ref: Optional[str] = Field(
        default=None, description="Reference to vector embedding"
    )


class ClaimVerification(ArtifactBase):
    """Claim verification artifact produced by verify subagents."""

    artifact_type: ArtifactType = Field(default=ArtifactType.CLAIM_VERIFICATION)
    claim_text: str = Field(..., min_length=1, description="The claim being verified")
    verdict: Verdict = Field(..., description="Verification verdict")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence in verdict (0.0-1.0)"
    )
    evidence_refs: List[str] = Field(
        ..., description="References to supporting evidence artifacts"
    )
    disagreement_notes: Optional[str] = Field(
        default=None, description="Notes on conflicting evidence"
    )
    method: str = Field(..., description="Verification method used")
    verifier: str = Field(..., description="ID of verifying subagent")
    action_suggestion: Optional[str] = Field(
        default=None, description="Suggested next action"
    )


class FileChange(BaseModel):
    """Individual file change in a code patch."""

    file_path: str = Field(..., description="Path to file")
    change_type: str = Field(
        ..., description="Type of change (added, modified, deleted)"
    )
    diff: str = Field(..., description="Unified diff format")
    lines_added: int = Field(default=0, ge=0)
    lines_removed: int = Field(default=0, ge=0)


class ValidationResult(BaseModel):
    """Result of code validation checks."""

    check_name: str = Field(..., description="Name of validation check")
    passed: bool = Field(..., description="Whether check passed")
    message: str = Field(..., description="Check result message")
    details: Optional[Dict[str, Any]] = Field(default=None)


class CodePatch(ArtifactBase):
    """Code patch artifact produced by code subagents."""

    artifact_type: ArtifactType = Field(default=ArtifactType.CODE_PATCH)
    repo: str = Field(..., description="Repository identifier")
    base_commit: str = Field(..., description="Base commit SHA")
    files_changed: List[FileChange] = Field(..., description="List of file changes")
    patch_summary: str = Field(..., description="Summary of changes")
    tests: List[str] = Field(default_factory=list, description="Test commands to run")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence in patch correctness"
    )
    risks: List[str] = Field(default_factory=list, description="Identified risks")
    authoring_subagent: str = Field(..., description="Subagent that created patch")
    validation_results: List[ValidationResult] = Field(
        default_factory=list, description="Validation check results"
    )
    approved_by: Optional[str] = Field(
        default=None, description="Approver ID if human approval given"
    )
    merge_ready: bool = Field(
        default=False, description="Whether patch is ready to merge"
    )


class SynthesisResult(ArtifactBase):
    """Final synthesis result from synthesis subagent."""

    artifact_type: ArtifactType = Field(default=ArtifactType.SYNTHESIS_RESULT)
    input_artifact_refs: List[str] = Field(
        ..., description="References to input artifacts"
    )
    synthesized_output: str = Field(
        ..., min_length=1, description="Final synthesized output"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence in synthesis"
    )
    reasoning_trace: Optional[str] = Field(
        default=None, description="Explanation of synthesis reasoning"
    )


# Union type for all artifacts
Artifact = Union[ResearchSnippet, ClaimVerification, CodePatch, SynthesisResult]


# ============================================================================
# Workflow Models
# ============================================================================


class WorkflowStepInput(BaseModel):
    """Input specification for a workflow step."""

    name: str = Field(..., description="Input parameter name")
    source: str = Field(
        ...,
        description="Source of input (user_input, previous_step, artifact:<id>, memory)",
    )
    required: bool = Field(default=True, description="Whether input is required")


class WorkflowStepOutput(BaseModel):
    """Output specification for a workflow step."""

    name: str = Field(..., description="Output parameter name")
    artifact_type: ArtifactType = Field(..., description="Expected artifact type")


class WorkflowStep(BaseModel):
    """Individual step in a workflow."""

    id: str = Field(..., description="Unique step identifier")
    role: SubagentRole = Field(..., description="Subagent role for this step")
    capabilities: List[str] = Field(..., description="Required capabilities")
    inputs: List[WorkflowStepInput] = Field(..., description="Input specifications")
    outputs: List[WorkflowStepOutput] = Field(..., description="Output specifications")
    timeout: int = Field(
        default=300, ge=1, le=3600, description="Step timeout in seconds"
    )
    retry_on_failure: bool = Field(
        default=True, description="Whether to retry on failure"
    )
    max_retries: int = Field(default=3, ge=0, le=10, description="Maximum retry attempts")


class MemoryConfig(BaseModel):
    """Memory configuration for workflow."""

    persist_on: List[str] = Field(
        default_factory=lambda: ["on_complete"],
        description="Events that trigger memory persistence",
    )
    compaction: Dict[str, Any] = Field(
        default_factory=lambda: {"strategy": "summarize", "max_tokens": 8000},
        description="Memory compaction configuration",
    )


class PolicyConfig(BaseModel):
    """Policy configuration for workflow."""

    requires_human_approval: bool = Field(
        default=False, description="Whether workflow requires human approval"
    )
    allowed_tool_categories: List[str] = Field(
        default_factory=list, description="Allowed tool categories from MCP"
    )
    max_tool_calls_per_step: int = Field(
        default=10, ge=1, description="Maximum tool calls per step"
    )


class ReflectionConfig(BaseModel):
    """Configuration for reflective agent behavior (PLAN -> EXECUTE -> VALIDATE -> REFINE)."""

    # Enable/disable reflection phases
    enable_planning: bool = Field(
        default=True, description="Enable planning phase before execution"
    )
    enable_validation: bool = Field(
        default=True, description="Enable self-validation of outputs"
    )
    enable_refinement: bool = Field(
        default=True, description="Enable refinement loop on validation failure"
    )
    enable_reflection: bool = Field(
        default=False, description="Enable post-execution reflection for learning"
    )

    # Planning configuration
    require_explicit_plan: bool = Field(
        default=False, description="Always generate formal plan before executing"
    )
    min_confidence_to_execute: float = Field(
        default=0.6, ge=0.0, le=1.0, description="Minimum plan confidence to proceed"
    )

    # Validation configuration
    validation_strictness: str = Field(
        default="medium", description="Validation strictness: low, medium, high"
    )
    min_quality_score: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Minimum quality score to accept"
    )

    # Refinement configuration
    max_refinement_iterations: int = Field(
        default=3, ge=1, le=10, description="Maximum refinement iterations"
    )
    refinement_strategy: str = Field(
        default="adaptive",
        description="Refinement strategy: conservative, adaptive, aggressive",
    )
    max_retries_per_step: int = Field(
        default=2, ge=0, le=5, description="Maximum retries per individual step"
    )

    # Timeouts
    planning_timeout_seconds: int = Field(
        default=30, ge=5, le=300, description="Timeout for planning phase"
    )
    validation_timeout_seconds: int = Field(
        default=15, ge=5, le=120, description="Timeout for validation phase"
    )
    refinement_timeout_seconds: int = Field(
        default=30, ge=5, le=300, description="Timeout for refinement phase"
    )


class EnterpriseConfig(BaseModel):
    """Configuration for enterprise agent with full auditability.

    Extends ReflectionConfig with:
    - THINK phase for natural reasoning
    - APPROVE phase for governance gates
    - REFLECT phase for learning
    - Full provenance tracking
    - Audit logging
    """

    # Enable enterprise features
    enable_enterprise_mode: bool = Field(
        default=False, description="Enable full enterprise agent mode"
    )

    # THINK phase (natural reasoning)
    enable_thinking: bool = Field(
        default=True, description="Enable natural reasoning (THINK) phase"
    )
    thinking_budget_tokens: int = Field(
        default=1000, ge=100, le=10000, description="Token budget for thinking"
    )
    log_thinking_traces: bool = Field(
        default=True, description="Log thinking traces for audit"
    )

    # APPROVE phase (governance)
    enable_governance: bool = Field(
        default=True, description="Enable governance gate (APPROVE) phase"
    )
    auto_approve_low_risk: bool = Field(
        default=True, description="Auto-approve low risk actions"
    )
    max_risk_level_auto_approve: str = Field(
        default="low", description="Maximum risk level for auto-approval: low, medium, high"
    )
    require_human_approval_for: List[str] = Field(
        default_factory=lambda: ["file_write", "file_delete", "bash_exec", "deploy"],
        description="Tool names requiring human approval",
    )
    human_approval_timeout_seconds: int = Field(
        default=3600, ge=60, le=86400, description="Timeout for human approval requests"
    )

    # REFLECT phase (learning)
    enable_reflection_learning: bool = Field(
        default=True, description="Enable post-execution reflection"
    )
    store_lessons_learned: bool = Field(
        default=True, description="Store lessons for future runs"
    )

    # Audit & Provenance
    enable_full_audit: bool = Field(
        default=True, description="Enable comprehensive audit logging"
    )
    enable_provenance_tracking: bool = Field(
        default=True, description="Enable cryptographic provenance tracking"
    )
    hash_algorithm: str = Field(
        default="sha256", description="Hash algorithm for provenance: sha256, sha384, sha512"
    )
    audit_log_path: Optional[str] = Field(
        default=None, description="Path for audit log file (None for memory only)"
    )
    provenance_retention_days: int = Field(
        default=365, ge=30, le=3650, description="Provenance record retention in days"
    )

    # Compliance
    compliance_mode: str = Field(
        default="standard", description="Compliance mode: standard, strict, audit_only"
    )
    require_user_attribution: bool = Field(
        default=True, description="Require user_id for all operations"
    )
    redact_sensitive_logs: bool = Field(
        default=True, description="Redact sensitive data in audit logs"
    )


class WorkflowManifest(BaseModel):
    """Complete workflow manifest structure."""

    manifest_id: str = Field(
        default_factory=lambda: str(uuid4()), description="Unique manifest ID"
    )
    name: str = Field(..., min_length=1, description="Workflow name")
    version: str = Field(..., description="Workflow version (semver)")
    description: Optional[str] = Field(default=None, description="Workflow description")
    steps: List[WorkflowStep] = Field(..., min_items=1, description="Workflow steps")
    memory: MemoryConfig = Field(
        default_factory=MemoryConfig, description="Memory configuration"
    )
    tools: Dict[str, Any] = Field(
        default_factory=dict, description="Tool catalog configuration"
    )
    policies: PolicyConfig = Field(
        default_factory=PolicyConfig, description="Policy configuration"
    )
    reflection: ReflectionConfig = Field(
        default_factory=ReflectionConfig,
        description="Reflective agent configuration (PLAN -> EXECUTE -> VALIDATE -> REFINE)",
    )
    enterprise: EnterpriseConfig = Field(
        default_factory=EnterpriseConfig,
        description="Enterprise agent configuration (THINK -> PLAN -> APPROVE -> EXECUTE -> VALIDATE -> REFLECT)",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    @field_validator("version")
    @classmethod
    def validate_version(cls, v: str) -> str:
        """Validate version follows semver pattern."""
        parts = v.split(".")
        if len(parts) != 3:
            raise ValueError("Version must follow semver format (x.y.z)")
        for part in parts:
            if not part.isdigit():
                raise ValueError("Version parts must be numeric")
        return v


# ============================================================================
# Workflow Execution Models
# ============================================================================


class WorkflowContext(BaseModel):
    """Runtime context for workflow execution."""

    workflow_id: str = Field(..., description="Unique workflow execution ID")
    manifest: WorkflowManifest = Field(..., description="Workflow manifest")
    status: WorkflowStatus = Field(default=WorkflowStatus.PENDING)
    current_step_index: int = Field(default=0, ge=0)
    user_input: Dict[str, Any] = Field(
        default_factory=dict, description="User-provided input"
    )
    step_artifacts: Dict[str, List[str]] = Field(
        default_factory=dict, description="Artifacts produced by each step (step_id -> artifact_ids)"
    )
    error_messages: List[str] = Field(
        default_factory=list, description="Error messages encountered"
    )
    started_at: Optional[datetime] = Field(default=None)
    completed_at: Optional[datetime] = Field(default=None)


# ============================================================================
# API Request/Response Models
# ============================================================================


class WorkflowStartRequest(BaseModel):
    """Request to start a new workflow."""

    manifest_name: Optional[str] = Field(
        default=None, description="Name of registered manifest"
    )
    manifest_yaml: Optional[str] = Field(
        default=None, description="Inline YAML manifest"
    )
    user_input: Dict[str, Any] = Field(
        default_factory=dict, description="User-provided input data"
    )
    llm_provider: Optional[str] = Field(
        default=None, description="Override default LLM provider"
    )

    @field_validator("manifest_name", "manifest_yaml")
    @classmethod
    def validate_manifest_source(cls, v: Optional[str], info: Any) -> Optional[str]:
        """Ensure at least one manifest source is provided."""
        values = info.data
        if not values.get("manifest_name") and not values.get("manifest_yaml"):
            raise ValueError("Either manifest_name or manifest_yaml must be provided")
        return v


class WorkflowStartResponse(BaseModel):
    """Response from workflow start request."""

    workflow_id: str = Field(..., description="Unique workflow execution ID")
    status: WorkflowStatus = Field(..., description="Current workflow status")
    message: str = Field(..., description="Human-readable message")
    estimated_duration_seconds: Optional[int] = Field(
        default=None, description="Estimated execution time"
    )


class SubagentRequest(BaseModel):
    """Request to execute a subagent."""

    workflow_id: str = Field(..., description="Associated workflow ID")
    step_id: str = Field(..., description="Workflow step ID")
    role: SubagentRole = Field(..., description="Subagent role")
    capabilities: List[str] = Field(..., description="Required capabilities")
    inputs: Dict[str, Any] = Field(..., description="Input data for subagent")
    system_prompt: Optional[str] = Field(
        default=None, description="Custom system prompt"
    )
    timeout: int = Field(default=300, ge=1, le=3600, description="Execution timeout")
    llm_provider: Optional[str] = Field(
        default=None, description="Override LLM provider"
    )


class SubagentResponse(BaseModel):
    """Response from subagent execution."""

    subagent_id: str = Field(..., description="Unique subagent execution ID")
    workflow_id: str = Field(..., description="Associated workflow ID")
    step_id: str = Field(..., description="Workflow step ID")
    status: str = Field(..., description="Execution status (success, failed, timeout)")
    artifacts: List[str] = Field(
        default_factory=list, description="IDs of produced artifacts"
    )
    error_message: Optional[str] = Field(default=None, description="Error if failed")
    execution_time_seconds: float = Field(
        ..., ge=0.0, description="Execution time in seconds"
    )
    token_usage: Dict[str, int] = Field(
        default_factory=dict, description="Token usage statistics"
    )


class ArtifactHandleRequest(BaseModel):
    """Request to validate and handle an artifact."""

    artifact_data: Dict[str, Any] = Field(..., description="Raw artifact data")
    artifact_type: ArtifactType = Field(..., description="Type of artifact")
    validate_only: bool = Field(
        default=False, description="Only validate, don't persist"
    )
    workflow_id: Optional[str] = Field(
        default=None, description="Associated workflow ID"
    )


class ArtifactHandleResponse(BaseModel):
    """Response from artifact handling."""

    artifact_id: str = Field(..., description="Unique artifact ID")
    valid: bool = Field(..., description="Whether artifact is valid")
    validation_errors: List[str] = Field(
        default_factory=list, description="Validation errors if any"
    )
    persisted: bool = Field(..., description="Whether artifact was persisted")
    memory_ref: Optional[str] = Field(
        default=None, description="Memory service reference"
    )


class HealthCheckResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status (healthy, degraded, unhealthy)")
    version: str = Field(..., description="Service version")
    dependencies: Dict[str, str] = Field(
        ..., description="Status of dependent services"
    )
    uptime_seconds: float = Field(..., ge=0.0, description="Service uptime")


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional details")
    request_id: Optional[str] = Field(default=None, description="Request ID for tracing")
