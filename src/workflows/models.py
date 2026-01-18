"""
Workflow Models.

Pydantic models for YAML workflow definitions.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from enum import Enum
from pydantic import BaseModel, Field


class RiskLevel(str, Enum):
    """Risk level for workflow steps."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ApprovalType(str, Enum):
    """Types of approval gates."""
    HUMAN = "human"
    AUTO = "auto"
    CONDITIONAL = "conditional"


class StepStatus(str, Enum):
    """Status of a workflow step."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    WAITING_APPROVAL = "waiting_approval"


class WorkflowVariable(BaseModel):
    """A variable in the workflow."""
    name: str
    description: str = ""
    type: str = "string"  # string, number, boolean, list, object
    default: Optional[Any] = None
    required: bool = False


class StepDependency(BaseModel):
    """Dependency on another step."""
    step_id: str
    condition: str = "completed"  # completed, success, any


class ApprovalGate(BaseModel):
    """Approval gate configuration."""
    type: ApprovalType = ApprovalType.AUTO
    required_approvers: int = 1
    timeout_seconds: int = 3600
    auto_approve_conditions: List[str] = Field(default_factory=list)
    message: str = ""


class WorkflowStep(BaseModel):
    """A single step in a workflow."""
    
    id: str
    name: str
    description: str = ""
    
    # Agent configuration
    agent: str  # Agent type to use (research, code, content, etc.)
    inputs: Dict[str, Any] = Field(default_factory=dict)
    
    # Outputs
    outputs: List[str] = Field(default_factory=list)  # Names of output artifacts
    output_artifact_type: str = "generic"
    
    # Dependencies
    depends_on: List[Union[str, StepDependency]] = Field(default_factory=list)
    
    # Approval
    require_approval: bool = False
    approval: Optional[ApprovalGate] = None
    
    # Risk
    risk: RiskLevel = RiskLevel.LOW
    
    # Conditions
    condition: Optional[str] = None  # Expression to evaluate
    skip_on_failure: bool = False
    
    # Timeout
    timeout_seconds: int = 300
    max_retries: int = 2
    
    # Metadata
    tags: List[str] = Field(default_factory=list)
    
    def get_dependency_ids(self) -> List[str]:
        """Get list of dependency step IDs."""
        deps = []
        for dep in self.depends_on:
            if isinstance(dep, str):
                deps.append(dep)
            else:
                deps.append(dep.step_id)
        return deps


class WorkflowManifest(BaseModel):
    """Complete workflow definition."""
    
    # Identity
    id: str
    name: str
    description: str = ""
    version: str = "1.0.0"
    
    # Variables (user inputs)
    variables: List[WorkflowVariable] = Field(default_factory=list)
    
    # Steps
    steps: List[WorkflowStep] = Field(default_factory=list)
    
    # Configuration
    parallel_execution: bool = True
    max_parallel_steps: int = 5
    fail_fast: bool = False
    
    # Outputs
    final_outputs: List[str] = Field(default_factory=list)
    
    # Metadata
    author: str = ""
    created_at: datetime = Field(default_factory=datetime.utcnow)
    tags: List[str] = Field(default_factory=list)
    category: str = ""  # saas, ecommerce, consulting, etc.
    
    def get_step(self, step_id: str) -> Optional[WorkflowStep]:
        """Get a step by ID."""
        for step in self.steps:
            if step.id == step_id:
                return step
        return None
    
    def get_execution_order(self) -> List[List[str]]:
        """
        Get steps organized by execution order.
        
        Returns list of lists, where each inner list contains
        step IDs that can be executed in parallel.
        """
        # Build dependency graph
        remaining = {s.id for s in self.steps}
        completed: set = set()
        order: List[List[str]] = []
        
        while remaining:
            # Find steps with all dependencies satisfied
            ready = []
            for step_id in remaining:
                step = self.get_step(step_id)
                if step:
                    deps = set(step.get_dependency_ids())
                    if deps.issubset(completed):
                        ready.append(step_id)
            
            if not ready:
                # Circular dependency or missing step
                break
            
            order.append(ready)
            completed.update(ready)
            remaining -= set(ready)
        
        return order
    
    def validate_workflow(self) -> List[str]:
        """
        Validate the workflow definition.
        
        Returns list of validation errors.
        """
        errors = []
        step_ids = {s.id for s in self.steps}
        
        # Check for duplicate IDs
        if len(step_ids) != len(self.steps):
            errors.append("Duplicate step IDs found")
        
        # Check dependencies exist
        for step in self.steps:
            for dep_id in step.get_dependency_ids():
                if dep_id not in step_ids:
                    errors.append(f"Step '{step.id}' depends on unknown step '{dep_id}'")
        
        # Check for circular dependencies
        order = self.get_execution_order()
        executed = set()
        for batch in order:
            executed.update(batch)
        
        if executed != step_ids:
            errors.append("Circular dependency detected in workflow")
        
        # Check required variables
        for var in self.variables:
            if var.required and var.default is None:
                errors.append(f"Required variable '{var.name}' has no default")
        
        return errors
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for YAML serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "variables": [v.model_dump() for v in self.variables],
            "steps": [s.model_dump() for s in self.steps],
            "parallel_execution": self.parallel_execution,
            "max_parallel_steps": self.max_parallel_steps,
            "fail_fast": self.fail_fast,
            "final_outputs": self.final_outputs,
            "author": self.author,
            "tags": self.tags,
            "category": self.category,
        }


class WorkflowRun(BaseModel):
    """A running instance of a workflow."""
    
    id: str
    workflow_id: str
    workflow_version: str
    
    # Status
    status: str = "created"  # created, running, completed, failed, cancelled
    
    # Variables (resolved values)
    variables: Dict[str, Any] = Field(default_factory=dict)
    
    # Step states
    step_states: Dict[str, StepStatus] = Field(default_factory=dict)
    step_results: Dict[str, Any] = Field(default_factory=dict)
    step_artifacts: Dict[str, List[str]] = Field(default_factory=dict)
    
    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Context
    business_id: Optional[str] = None
    user_id: Optional[str] = None
    
    # Error tracking
    error: Optional[str] = None
    failed_step: Optional[str] = None
