"""
Data models for the planning and execution system.
Supports ReAct pattern and dependency-aware execution.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any, Literal
from enum import Enum
from datetime import datetime
from uuid import uuid4


class TaskStatus(str, Enum):
    """Status of a planned task."""
    PENDING = "pending"
    READY = "ready"           # Dependencies satisfied
    IN_PROGRESS = "in_progress"
    WAITING_APPROVAL = "waiting_approval"
    APPROVED = "approved"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


class RiskLevel(str, Enum):
    """Risk classification for tasks."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ReActStep(BaseModel):
    """A single step in the ReAct loop."""
    step_number: int
    thought: str              # Reasoning about what to do
    action: Optional[str] = None     # The action to take
    action_input: Optional[Dict[str, Any]] = None  # Parameters for the action
    observation: Optional[str] = None  # Result of the action
    timestamp: datetime = Field(default_factory=datetime.now)


class PlanTask(BaseModel):
    """A single task within an execution plan."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    description: str
    agent: str
    
    # Execution control
    status: TaskStatus = TaskStatus.PENDING
    priority: int = 5  # 1 = highest, 10 = lowest
    
    # Dependencies
    depends_on: List[str] = Field(default_factory=list)  # Task IDs
    blocks: List[str] = Field(default_factory=list)      # Tasks that depend on this
    
    # Risk and approval
    risk_level: RiskLevel = RiskLevel.LOW
    requires_approval: bool = False
    approval_reason: Optional[str] = None
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None
    
    # Execution data
    input_data: Dict[str, Any] = Field(default_factory=dict)
    output_data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    
    # Timing
    estimated_duration_minutes: int = 5
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # ReAct trace
    react_steps: List[ReActStep] = Field(default_factory=list)
    
    @property
    def duration_ms(self) -> Optional[float]:
        """Calculate actual duration if completed."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds() * 1000
        return None
    
    def can_execute(self, completed_tasks: set) -> bool:
        """Check if all dependencies are satisfied."""
        return all(dep in completed_tasks for dep in self.depends_on)


class ExecutionPlan(BaseModel):
    """Complete execution plan for achieving a goal."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    goal: str
    context: Optional[str] = None
    
    # Tasks
    tasks: List[PlanTask] = Field(default_factory=list)
    
    # Status
    status: Literal["planning", "ready", "executing", "paused", "completed", "failed"] = "planning"
    current_task_id: Optional[str] = None
    
    # Metrics
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    
    # Risk assessment
    overall_risk: RiskLevel = RiskLevel.LOW
    requires_human_review: bool = False
    
    # Timing
    created_at: datetime = Field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    estimated_duration_minutes: int = 0
    
    # ReAct planning trace
    planning_steps: List[ReActStep] = Field(default_factory=list)
    
    def get_task(self, task_id: str) -> Optional[PlanTask]:
        """Get a task by ID."""
        for task in self.tasks:
            if task.id == task_id:
                return task
        return None
    
    def get_ready_tasks(self) -> List[PlanTask]:
        """Get tasks that are ready to execute."""
        completed = {t.id for t in self.tasks if t.status == TaskStatus.COMPLETED}
        return [
            t for t in self.tasks
            if t.status == TaskStatus.PENDING and t.can_execute(completed)
        ]
    
    def get_next_task(self) -> Optional[PlanTask]:
        """Get the highest priority ready task."""
        ready = self.get_ready_tasks()
        if not ready:
            return None
        return min(ready, key=lambda t: t.priority)
    
    def update_metrics(self):
        """Recalculate plan metrics."""
        self.total_tasks = len(self.tasks)
        self.completed_tasks = len([t for t in self.tasks if t.status == TaskStatus.COMPLETED])
        self.failed_tasks = len([t for t in self.tasks if t.status == TaskStatus.FAILED])
        
        # Update status
        if self.failed_tasks > 0 and self.status == "executing":
            # Check if we can continue
            ready = self.get_ready_tasks()
            if not ready and self.completed_tasks < self.total_tasks:
                self.status = "failed"
        elif self.completed_tasks == self.total_tasks:
            self.status = "completed"
            self.completed_at = datetime.now()


class PlanningContext(BaseModel):
    """Context provided to the planner."""
    goal: str
    user_input: str
    empire_state: str
    risk_profile: str
    available_agents: List[str]
    constraints: List[str] = Field(default_factory=list)
    preferences: Dict[str, Any] = Field(default_factory=dict)


class ReplanRequest(BaseModel):
    """Request to replan after a failure."""
    plan_id: str
    failed_task_id: str
    failure_reason: str
    context: str
    attempt_number: int = 1
    max_attempts: int = 3
