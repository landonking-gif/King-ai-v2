# King AI v2 - Implementation Plan Part 4 of 20
## Master AI Brain - Planning & ReAct Implementation

**Target Timeline:** Week 3-4
**Objective:** Implement sophisticated planning capabilities using the ReAct (Reason-Act-Think) pattern for multi-step goal decomposition and execution.

---

## Table of Contents
1. [Overview of 20-Part Plan](#overview-of-20-part-plan)
2. [Part 4 Scope](#part-4-scope)
3. [Current State Analysis](#current-state-analysis)
4. [Implementation Tasks](#implementation-tasks)
5. [File-by-File Instructions](#file-by-file-instructions)
6. [Testing Requirements](#testing-requirements)
7. [Acceptance Criteria](#acceptance-criteria)

---

## Overview of 20-Part Plan

| Part | Title | Status |
|------|-------|--------|
| 1 | Infrastructure Layer & Core System Hardening | ✅ Complete |
| 2 | Master AI Brain - Core Enhancements | ✅ Complete |
| 3 | Master AI Brain - Context & Memory System | ✅ Complete |
| **4** | **Master AI Brain - Planning & ReAct Implementation** | 🔄 Current |
| 5 | Evolution Engine - Code Modification System | ⏳ Pending |
| 6 | Evolution Engine - ML Retraining Pipeline | ⏳ Pending |
| 7 | Evolution Engine - Sandbox & Testing | ⏳ Pending |
| 8 | Sub-Agents - Research Agent Enhancement | ⏳ Pending |
| 9 | Sub-Agents - Code Generator Agent | ⏳ Pending |
| 10 | Sub-Agents - Content Agent | ⏳ Pending |
| 11 | Sub-Agents - Commerce Agent (Shopify/AliExpress) | ⏳ Pending |
| 12 | Sub-Agents - Finance Agent (Stripe/Plaid) | ⏳ Pending |
| 13 | Sub-Agents - Analytics Agent | ⏳ Pending |
| 14 | Sub-Agents - Legal Agent | ⏳ Pending |
| 15 | Business Units - Lifecycle Engine | ⏳ Pending |
| 16 | Business Units - Playbook System | ⏳ Pending |
| 17 | Business Units - Portfolio Management | ⏳ Pending |
| 18 | Dashboard - React UI Components | ⏳ Pending |
| 19 | Dashboard - Approval Workflows & Risk Engine | ⏳ Pending |
| 20 | Dashboard - Real-time Monitoring & WebSocket + Final Integration | ⏳ Pending |

---

## Part 4 Scope

This part focuses on:
1. ReAct (Reason-Act-Think) pattern implementation
2. Multi-step goal decomposition with dependency graphs
3. Dynamic task scheduling with priority queues
4. Execution monitoring and progress tracking
5. Automatic replanning on failure
6. Risk-aware planning with approval gates
7. Plan persistence and resumption

---

## Current State Analysis

### What Exists in `src/master_ai/planner.py`
| Feature | Status | Issue |
|---------|--------|-------|
| Basic Planner class | ✅ Exists | Very simple implementation |
| create_plan method | ✅ Works | No dependency handling |
| LLM-based planning | ✅ Basic | No ReAct pattern |
| Error handling | ⚠️ Minimal | Fallback only |

### What Needs to Be Added
1. ReAct loop implementation (Thought → Action → Observation)
2. Dependency graph for task ordering
3. Plan persistence to database
4. Execution state machine
5. Dynamic replanning capabilities
6. Risk assessment integration
7. Progress tracking and reporting

---

## Implementation Tasks

### Task 4.1: Create Planning Data Models
**Priority:** 🔴 Critical
**Estimated Time:** 2 hours
**Dependencies:** Part 2 complete

#### File: `src/master_ai/planning_models.py` (CREATE NEW FILE)
```python
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
    action: Optional[str]     # The action to take
    action_input: Optional[Dict[str, Any]]  # Parameters for the action
    observation: Optional[str]  # Result of the action
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
```

---

### Task 4.2: Implement ReAct Planning Engine
**Priority:** 🔴 Critical
**Estimated Time:** 4 hours
**Dependencies:** Task 4.1

#### File: `src/master_ai/react_planner.py` (CREATE NEW FILE)
```python
"""
ReAct (Reason-Act-Think) Planning Engine.
Implements iterative planning with observation feedback.
"""

import json
import asyncio
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime

from src.master_ai.planning_models import (
    ExecutionPlan, PlanTask, PlanningContext, ReActStep,
    TaskStatus, RiskLevel, ReplanRequest
)
from src.master_ai.prompts import REACT_PLANNING_PROMPT, TASK_DECOMPOSITION_PROMPT
from src.utils.llm_router import LLMRouter, TaskContext
from src.utils.structured_logging import get_logger
from src.utils.retry import with_retry, LLM_RETRY_CONFIG
from config.settings import settings

logger = get_logger("react_planner")


# Risk thresholds by profile
RISK_THRESHOLDS = {
    "conservative": {
        "max_spend_auto": 50,
        "approval_required": [RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL],
    },
    "moderate": {
        "max_spend_auto": 500,
        "approval_required": [RiskLevel.HIGH, RiskLevel.CRITICAL],
    },
    "aggressive": {
        "max_spend_auto": 5000,
        "approval_required": [RiskLevel.CRITICAL],
    }
}


class ReActPlanner:
    """
    Implements the ReAct planning pattern for goal decomposition.
    
    ReAct Loop:
    1. THOUGHT: Reason about the current state and what to do next
    2. ACTION: Decide on an action to take
    3. OBSERVATION: Execute and observe the result
    4. Repeat until goal is achieved or plan is complete
    """
    
    def __init__(self, llm_router: LLMRouter):
        """
        Initialize the planner.
        
        Args:
            llm_router: LLM router for inference
        """
        self.llm = llm_router
        self.max_planning_steps = 10
        self.available_agents = [
            "research", "code_generator", "content", "commerce",
            "finance", "analytics", "legal"
        ]
    
    async def create_plan(
        self,
        goal: str,
        context: str,
        action: str = None,
        parameters: Dict[str, Any] = None
    ) -> ExecutionPlan:
        """
        Create an execution plan for a goal using ReAct pattern.
        
        Args:
            goal: The user's goal to achieve
            context: Current empire context
            action: Specific action type (optional)
            parameters: Extracted parameters (optional)
            
        Returns:
            Complete execution plan with tasks
        """
        logger.info("Creating plan", goal=goal[:100])
        
        plan = ExecutionPlan(
            goal=goal,
            context=context[:2000],
            status="planning"
        )
        
        # Build planning context
        planning_context = PlanningContext(
            goal=goal,
            user_input=goal,
            empire_state=context,
            risk_profile=settings.risk_profile,
            available_agents=self.available_agents
        )
        
        try:
            # Phase 1: High-level task decomposition
            tasks = await self._decompose_goal(planning_context, parameters)
            
            # Phase 2: Build dependency graph
            tasks = self._build_dependencies(tasks)
            
            # Phase 3: Assess risks and set approval requirements
            tasks = self._assess_risks(tasks)
            
            # Phase 4: Optimize execution order
            tasks = self._optimize_order(tasks)
            
            plan.tasks = tasks
            plan.status = "ready"
            plan.update_metrics()
            
            # Calculate total estimated time
            plan.estimated_duration_minutes = sum(t.estimated_duration_minutes for t in tasks)
            
            # Set overall risk
            risk_levels = [t.risk_level for t in tasks]
            if RiskLevel.CRITICAL in risk_levels:
                plan.overall_risk = RiskLevel.CRITICAL
            elif RiskLevel.HIGH in risk_levels:
                plan.overall_risk = RiskLevel.HIGH
            elif RiskLevel.MEDIUM in risk_levels:
                plan.overall_risk = RiskLevel.MEDIUM
            else:
                plan.overall_risk = RiskLevel.LOW
            
            # Check if human review needed
            plan.requires_human_review = any(t.requires_approval for t in tasks)
            
            logger.info(
                "Plan created",
                plan_id=plan.id,
                tasks=len(plan.tasks),
                risk=plan.overall_risk.value
            )
            
            return plan
            
        except Exception as e:
            logger.error("Planning failed", error=str(e), exc_info=True)
            plan.status = "failed"
            raise
    
    @with_retry(LLM_RETRY_CONFIG)
    async def _decompose_goal(
        self,
        context: PlanningContext,
        parameters: Dict[str, Any] = None
    ) -> List[PlanTask]:
        """
        Decompose a goal into concrete tasks using LLM.
        """
        prompt = TASK_DECOMPOSITION_PROMPT.format(
            goal=context.goal,
            context=context.empire_state[:3000],
            risk_profile=context.risk_profile,
            available_agents=", ".join(context.available_agents),
            parameters=json.dumps(parameters or {})
        )
        
        llm_context = TaskContext(
            task_type="planning",
            risk_level="low",
            requires_accuracy=True,
            token_estimate=2000,
            priority="high"
        )
        
        response = await self.llm.complete(prompt, context=llm_context)
        
        # Parse response
        tasks_data = self._parse_json_response(response)
        
        tasks = []
        for i, task_data in enumerate(tasks_data.get("tasks", [])):
            task = PlanTask(
                name=task_data.get("name", f"Task {i+1}"),
                description=task_data.get("description", ""),
                agent=task_data.get("agent", "research"),
                priority=task_data.get("priority", 5),
                input_data=task_data.get("input", {}),
                estimated_duration_minutes=task_data.get("duration_minutes", 5),
                risk_level=RiskLevel(task_data.get("risk_level", "low")),
            )
            tasks.append(task)
        
        return tasks
    
    def _build_dependencies(self, tasks: List[PlanTask]) -> List[PlanTask]:
        """
        Build dependency graph between tasks.
        Uses heuristics and task descriptions to infer dependencies.
        """
        if not tasks:
            return tasks
        
        # Simple heuristic: tasks depend on previous tasks of certain types
        research_task_ids = []
        setup_task_ids = []
        
        for task in tasks:
            # Research tasks typically come first
            if task.agent == "research":
                research_task_ids.append(task.id)
            
            # Setup/commerce tasks depend on research
            elif task.agent in ["commerce", "code_generator"]:
                task.depends_on = research_task_ids.copy()
                setup_task_ids.append(task.id)
            
            # Content depends on setup
            elif task.agent == "content":
                task.depends_on = setup_task_ids.copy() if setup_task_ids else research_task_ids.copy()
            
            # Analytics/finance depend on operations being set up
            elif task.agent in ["analytics", "finance"]:
                task.depends_on = setup_task_ids.copy()
            
            # Legal reviews depend on having something to review
            elif task.agent == "legal":
                task.depends_on = [t.id for t in tasks if t.id != task.id][:3]
        
        # Build reverse mapping (blocks)
        for task in tasks:
            for dep_id in task.depends_on:
                dep_task = next((t for t in tasks if t.id == dep_id), None)
                if dep_task and task.id not in dep_task.blocks:
                    dep_task.blocks.append(task.id)
        
        return tasks
    
    def _assess_risks(self, tasks: List[PlanTask]) -> List[PlanTask]:
        """
        Assess risks and set approval requirements based on risk profile.
        """
        thresholds = RISK_THRESHOLDS.get(settings.risk_profile, RISK_THRESHOLDS["moderate"])
        
        for task in tasks:
            # High-risk agents always need review
            if task.agent in ["legal", "finance"]:
                task.risk_level = max(task.risk_level, RiskLevel.MEDIUM, key=lambda x: list(RiskLevel).index(x))
            
            # Check if approval needed based on risk
            if task.risk_level in thresholds["approval_required"]:
                task.requires_approval = True
                task.approval_reason = f"Task risk level ({task.risk_level.value}) requires approval under {settings.risk_profile} profile"
            
            # Commerce tasks with potential spending
            if task.agent == "commerce":
                spend = task.input_data.get("estimated_spend", 0)
                if spend > thresholds["max_spend_auto"]:
                    task.requires_approval = True
                    task.approval_reason = f"Estimated spend ${spend} exceeds auto-approval limit ${thresholds['max_spend_auto']}"
                    task.risk_level = RiskLevel.HIGH
        
        return tasks
    
    def _optimize_order(self, tasks: List[PlanTask]) -> List[PlanTask]:
        """
        Optimize task execution order using topological sort.
        """
        if not tasks:
            return tasks
        
        # Topological sort with priority consideration
        task_map = {t.id: t for t in tasks}
        in_degree = {t.id: len(t.depends_on) for t in tasks}
        
        # Start with tasks that have no dependencies
        queue = [(t.priority, t.id) for t in tasks if in_degree[t.id] == 0]
        queue.sort()
        
        sorted_tasks = []
        while queue:
            _, task_id = queue.pop(0)
            task = task_map[task_id]
            sorted_tasks.append(task)
            
            # Update dependent tasks
            for blocked_id in task.blocks:
                in_degree[blocked_id] -= 1
                if in_degree[blocked_id] == 0:
                    blocked_task = task_map[blocked_id]
                    queue.append((blocked_task.priority, blocked_id))
                    queue.sort()
        
        # Check for cycles (tasks not in sorted list)
        if len(sorted_tasks) != len(tasks):
            logger.warning("Dependency cycle detected, using original order")
            return tasks
        
        return sorted_tasks
    
    async def replan(self, request: ReplanRequest, context: str) -> ExecutionPlan:
        """
        Create a new plan after a task failure.
        """
        logger.info(
            "Replanning after failure",
            plan_id=request.plan_id,
            failed_task=request.failed_task_id,
            attempt=request.attempt_number
        )
        
        if request.attempt_number > request.max_attempts:
            raise RuntimeError(f"Max replan attempts ({request.max_attempts}) exceeded")
        
        prompt = f"""A task in the plan failed. Create an alternative approach.

ORIGINAL GOAL: (retrieve from context)
FAILED TASK: {request.failed_task_id}
FAILURE REASON: {request.failure_reason}

CONTEXT:
{context[:2000]}

Create an alternative plan that:
1. Avoids the failed approach
2. Achieves the same goal differently
3. Is more conservative if previous attempt was risky

Respond with the same JSON task format.
"""
        
        response = await self.llm.complete(prompt)
        tasks_data = self._parse_json_response(response)
        
        plan = ExecutionPlan(
            goal=f"Replan (attempt {request.attempt_number})",
            context=context[:2000],
            status="ready"
        )
        
        for task_data in tasks_data.get("tasks", []):
            task = PlanTask(
                name=task_data.get("name", "Replanned task"),
                description=task_data.get("description", ""),
                agent=task_data.get("agent", "research"),
                input_data=task_data.get("input", {}),
                risk_level=RiskLevel.MEDIUM,  # Conservative for replanning
                requires_approval=True,  # Require approval for replanned tasks
                approval_reason="Replanned task requires verification"
            )
            plan.tasks.append(task)
        
        plan.update_metrics()
        return plan
    
    async def run_react_loop(
        self,
        task: PlanTask,
        execute_fn,
        max_iterations: int = 5
    ) -> PlanTask:
        """
        Run the ReAct loop for a single task.
        
        Args:
            task: The task to execute
            execute_fn: Async function to execute actions
            max_iterations: Maximum ReAct iterations
            
        Returns:
            Updated task with results
        """
        task.status = TaskStatus.IN_PROGRESS
        task.started_at = datetime.now()
        
        for i in range(max_iterations):
            # THOUGHT: Reason about current state
            thought = await self._generate_thought(task, i)
            
            step = ReActStep(
                step_number=i + 1,
                thought=thought
            )
            
            # Check if we're done
            if "FINAL ANSWER" in thought or "COMPLETE" in thought:
                step.observation = "Task completed successfully"
                task.react_steps.append(step)
                task.status = TaskStatus.COMPLETED
                task.completed_at = datetime.now()
                break
            
            # ACTION: Decide on action
            action, action_input = await self._decide_action(task, thought)
            step.action = action
            step.action_input = action_input
            
            # OBSERVATION: Execute and observe
            try:
                result = await execute_fn(task, action, action_input)
                step.observation = str(result)[:500]
                task.output_data = result
            except Exception as e:
                step.observation = f"ERROR: {str(e)}"
                task.error = str(e)
                task.status = TaskStatus.FAILED
                task.completed_at = datetime.now()
                task.react_steps.append(step)
                break
            
            task.react_steps.append(step)
        
        if task.status == TaskStatus.IN_PROGRESS:
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
        
        return task
    
    async def _generate_thought(self, task: PlanTask, iteration: int) -> str:
        """Generate a thought for the current ReAct step."""
        previous_steps = "\n".join([
            f"Step {s.step_number}: {s.thought[:100]}... -> {s.observation[:100] if s.observation else 'pending'}..."
            for s in task.react_steps
        ])
        
        prompt = f"""You are executing a task using the ReAct pattern.

TASK: {task.name}
DESCRIPTION: {task.description}
AGENT: {task.agent}
INPUT: {json.dumps(task.input_data)}

PREVIOUS STEPS:
{previous_steps or "None yet"}

ITERATION: {iteration + 1}

Think about:
1. What has been done so far?
2. What remains to be done?
3. What is the next best action?

If the task is complete, include "FINAL ANSWER" in your thought.

YOUR THOUGHT:"""
        
        return await self.llm.complete(prompt)
    
    async def _decide_action(
        self,
        task: PlanTask,
        thought: str
    ) -> Tuple[str, Dict[str, Any]]:
        """Decide on the next action based on the thought."""
        prompt = f"""Based on this thought, decide on the next action.

THOUGHT: {thought}
TASK: {task.name}
AGENT: {task.agent}

Available actions for {task.agent} agent:
- execute: Run the main task logic
- query: Get more information
- validate: Validate current results
- finalize: Complete the task

Respond with JSON:
{{"action": "action_name", "input": {{...parameters...}}}}
"""
        
        response = await self.llm.complete(prompt)
        data = self._parse_json_response(response)
        
        return data.get("action", "execute"), data.get("input", {})
    
    def _parse_json_response(self, response: str) -> dict:
        """Parse JSON from LLM response."""
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            response = response.split("```")[1].split("```")[0].strip()
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON response", response=response[:200])
            return {"tasks": []}
```

---

### Task 4.3: Create Plan Executor
**Priority:** 🔴 Critical
**Estimated Time:** 3 hours
**Dependencies:** Tasks 4.1, 4.2

#### File: `src/master_ai/plan_executor.py` (CREATE NEW FILE)
```python
"""
Plan Executor - Manages execution of plans with state tracking.
Handles task scheduling, approval gates, and failure recovery.
"""

import asyncio
from typing import Optional, Callable, Dict, Any, List
from datetime import datetime
from uuid import uuid4

from src.master_ai.planning_models import (
    ExecutionPlan, PlanTask, TaskStatus, ReplanRequest
)
from src.master_ai.react_planner import ReActPlanner
from src.agents.router import AgentRouter
from src.database.connection import get_db
from src.database.models import Task as DBTask
from src.utils.structured_logging import get_logger
from src.utils.monitoring import monitor

logger = get_logger("plan_executor")


class PlanExecutor:
    """
    Executes plans with state management and failure handling.
    """
    
    def __init__(
        self,
        planner: ReActPlanner,
        agent_router: AgentRouter,
        on_approval_needed: Callable = None,
        on_task_complete: Callable = None,
        on_plan_complete: Callable = None
    ):
        """
        Initialize the executor.
        
        Args:
            planner: ReAct planner for replanning
            agent_router: Router to dispatch tasks to agents
            on_approval_needed: Callback when approval is needed
            on_task_complete: Callback when a task completes
            on_plan_complete: Callback when plan completes
        """
        self.planner = planner
        self.agent_router = agent_router
        self.on_approval_needed = on_approval_needed
        self.on_task_complete = on_task_complete
        self.on_plan_complete = on_plan_complete
        
        self._active_plans: Dict[str, ExecutionPlan] = {}
        self._paused_plans: Dict[str, ExecutionPlan] = {}
    
    async def execute_plan(
        self,
        plan: ExecutionPlan,
        auto_continue: bool = True
    ) -> ExecutionPlan:
        """
        Execute a plan, handling approvals and failures.
        
        Args:
            plan: The plan to execute
            auto_continue: Whether to auto-execute ready tasks
            
        Returns:
            Updated plan with results
        """
        logger.info("Starting plan execution", plan_id=plan.id, tasks=len(plan.tasks))
        monitor.increment("plan_executor.plans_started")
        
        plan.status = "executing"
        plan.started_at = datetime.now()
        self._active_plans[plan.id] = plan
        
        try:
            while plan.status == "executing":
                # Get next ready task
                next_task = plan.get_next_task()
                
                if next_task is None:
                    # Check if we're done or blocked
                    if plan.completed_tasks == plan.total_tasks:
                        plan.status = "completed"
                        plan.completed_at = datetime.now()
                        break
                    
                    # Check for pending approvals
                    waiting = [t for t in plan.tasks if t.status == TaskStatus.WAITING_APPROVAL]
                    if waiting:
                        logger.info("Plan waiting for approvals", count=len(waiting))
                        plan.status = "paused"
                        self._paused_plans[plan.id] = plan
                        break
                    
                    # Check for failures blocking progress
                    failed = [t for t in plan.tasks if t.status == TaskStatus.FAILED]
                    if failed:
                        logger.warning("Plan blocked by failures", count=len(failed))
                        plan.status = "failed"
                        break
                    
                    # Shouldn't reach here
                    logger.error("Plan in unexpected state")
                    break
                
                # Check if approval needed
                if next_task.requires_approval and next_task.status != TaskStatus.APPROVED:
                    next_task.status = TaskStatus.WAITING_APPROVAL
                    await self._request_approval(plan, next_task)
                    
                    if not auto_continue:
                        plan.status = "paused"
                        break
                    continue
                
                # Execute the task
                plan.current_task_id = next_task.id
                await self._execute_task(plan, next_task)
                
                plan.update_metrics()
                
                if self.on_task_complete:
                    await self.on_task_complete(plan, next_task)
            
            # Cleanup
            if plan.id in self._active_plans:
                del self._active_plans[plan.id]
            
            if self.on_plan_complete:
                await self.on_plan_complete(plan)
            
            monitor.increment(f"plan_executor.plans_{plan.status}")
            logger.info(
                "Plan execution finished",
                plan_id=plan.id,
                status=plan.status,
                completed=plan.completed_tasks,
                failed=plan.failed_tasks
            )
            
            return plan
            
        except Exception as e:
            logger.error("Plan execution error", error=str(e), exc_info=True)
            plan.status = "failed"
            raise
    
    async def _execute_task(self, plan: ExecutionPlan, task: PlanTask):
        """Execute a single task."""
        logger.info("Executing task", task_id=task.id, name=task.name, agent=task.agent)
        monitor.increment("plan_executor.tasks_started", tags={"agent": task.agent})
        
        task.status = TaskStatus.IN_PROGRESS
        task.started_at = datetime.now()
        
        try:
            # Use ReAct loop for complex tasks, direct execution for simple ones
            if self._should_use_react(task):
                task = await self.planner.run_react_loop(
                    task,
                    self._execute_action
                )
            else:
                result = await self.agent_router.execute({
                    "name": task.name,
                    "description": task.description,
                    "agent": task.agent,
                    "input": task.input_data
                })
                
                task.output_data = result
                if result.get("success"):
                    task.status = TaskStatus.COMPLETED
                else:
                    task.status = TaskStatus.FAILED
                    task.error = result.get("error", "Unknown error")
                
                task.completed_at = datetime.now()
            
            # Persist to database
            await self._persist_task_result(task)
            
            monitor.increment(
                f"plan_executor.tasks_{task.status.value}",
                tags={"agent": task.agent}
            )
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = datetime.now()
            
            logger.error("Task execution failed", task_id=task.id, error=str(e))
            monitor.increment("plan_executor.tasks_failed", tags={"agent": task.agent})
            
            # Attempt replanning if enabled
            await self._handle_failure(plan, task)
    
    def _should_use_react(self, task: PlanTask) -> bool:
        """Determine if a task should use the ReAct loop."""
        # Use ReAct for complex, multi-step tasks
        complex_agents = ["research", "code_generator", "analytics"]
        return task.agent in complex_agents and task.estimated_duration_minutes > 10
    
    async def _execute_action(
        self,
        task: PlanTask,
        action: str,
        action_input: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute an action as part of ReAct loop."""
        return await self.agent_router.execute({
            "name": f"{task.name}:{action}",
            "description": task.description,
            "agent": task.agent,
            "input": {**task.input_data, **action_input}
        })
    
    async def _request_approval(self, plan: ExecutionPlan, task: PlanTask):
        """Request approval for a task."""
        logger.info(
            "Requesting approval",
            task_id=task.id,
            reason=task.approval_reason
        )
        
        # Persist to database for dashboard
        async with get_db() as db:
            db_task = DBTask(
                id=task.id,
                name=task.name,
                description=task.description,
                type=task.agent,
                status="pending_approval",
                agent=task.agent,
                input_data=task.input_data,
                requires_approval=True
            )
            db.add(db_task)
            await db.commit()
        
        if self.on_approval_needed:
            await self.on_approval_needed(plan, task)
    
    async def approve_task(self, plan_id: str, task_id: str, approver: str = "admin"):
        """Approve a pending task and resume execution."""
        plan = self._paused_plans.get(plan_id) or self._active_plans.get(plan_id)
        
        if not plan:
            raise ValueError(f"Plan {plan_id} not found")
        
        task = plan.get_task(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found in plan")
        
        task.status = TaskStatus.APPROVED
        task.approved_by = approver
        task.approved_at = datetime.now()
        
        logger.info("Task approved", task_id=task_id, approver=approver)
        
        # Update database
        async with get_db() as db:
            db_task = await db.get(DBTask, task_id)
            if db_task:
                db_task.status = "approved"
                db_task.approved_by = approver
                db_task.approved_at = datetime.now()
                await db.commit()
        
        # Resume plan if it was paused
        if plan.status == "paused":
            plan.status = "executing"
            if plan.id in self._paused_plans:
                del self._paused_plans[plan.id]
            asyncio.create_task(self.execute_plan(plan))
    
    async def reject_task(self, plan_id: str, task_id: str, reason: str = None):
        """Reject a pending task."""
        plan = self._paused_plans.get(plan_id) or self._active_plans.get(plan_id)
        
        if not plan:
            raise ValueError(f"Plan {plan_id} not found")
        
        task = plan.get_task(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found in plan")
        
        task.status = TaskStatus.SKIPPED
        task.error = f"Rejected: {reason}" if reason else "Rejected by user"
        
        logger.info("Task rejected", task_id=task_id, reason=reason)
        
        # Update metrics and potentially fail the plan
        plan.update_metrics()
    
    async def _handle_failure(self, plan: ExecutionPlan, failed_task: PlanTask):
        """Handle task failure with potential replanning."""
        # Check if task has dependents that should be skipped
        for dependent_id in failed_task.blocks:
            dependent = plan.get_task(dependent_id)
            if dependent:
                dependent.status = TaskStatus.SKIPPED
                dependent.error = f"Skipped due to failure of dependency: {failed_task.name}"
        
        # TODO: Implement automatic replanning based on settings
        # For now, just log the failure
        logger.warning(
            "Task failed, dependents skipped",
            task_id=failed_task.id,
            dependents=len(failed_task.blocks)
        )
    
    async def _persist_task_result(self, task: PlanTask):
        """Save task result to database."""
        async with get_db() as db:
            db_task = await db.get(DBTask, task.id)
            if db_task:
                db_task.status = task.status.value
                db_task.output_data = task.output_data
                db_task.completed_at = task.completed_at
                await db.commit()
            else:
                # Create new record
                db_task = DBTask(
                    id=task.id,
                    name=task.name,
                    description=task.description,
                    type=task.agent,
                    status=task.status.value,
                    agent=task.agent,
                    input_data=task.input_data,
                    output_data=task.output_data,
                    requires_approval=task.requires_approval,
                    completed_at=task.completed_at
                )
                db.add(db_task)
                await db.commit()
    
    def get_plan_status(self, plan_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of a plan."""
        plan = self._active_plans.get(plan_id) or self._paused_plans.get(plan_id)
        
        if not plan:
            return None
        
        return {
            "id": plan.id,
            "goal": plan.goal,
            "status": plan.status,
            "progress": {
                "total": plan.total_tasks,
                "completed": plan.completed_tasks,
                "failed": plan.failed_tasks,
                "percentage": (plan.completed_tasks / plan.total_tasks * 100) if plan.total_tasks > 0 else 0
            },
            "current_task": plan.current_task_id,
            "pending_approvals": [
                {"id": t.id, "name": t.name, "reason": t.approval_reason}
                for t in plan.tasks if t.status == TaskStatus.WAITING_APPROVAL
            ]
        }
```

---

### Task 4.4: Update Prompts for Planning
**Priority:** 🟡 High
**Estimated Time:** 1 hour
**Dependencies:** Task 4.2

#### File: `src/master_ai/prompts.py` (ADD TO EXISTING FILE)
Add these new prompts after the existing ones:

```python
# Add after existing prompts

TASK_DECOMPOSITION_PROMPT = """Decompose this goal into concrete, executable tasks.

GOAL: {goal}

CURRENT EMPIRE STATE:
{context}

RISK PROFILE: {risk_profile}
AVAILABLE AGENTS: {available_agents}
EXTRACTED PARAMETERS: {parameters}

Create a plan with specific tasks. For each task specify:
- name: Clear, action-oriented name
- description: What exactly needs to be done
- agent: Which agent handles it ({available_agents})
- priority: 1-10 (1 = highest priority)
- duration_minutes: Estimated time
- risk_level: low, medium, high, or critical
- input: Parameters needed for the task

Consider:
1. What research is needed first?
2. What setup/infrastructure is required?
3. What content or assets need creation?
4. What financial/legal review is needed?
5. How should tasks be ordered for efficiency?

Respond with JSON:
{{
    "goal_summary": "Brief summary of the goal",
    "tasks": [
        {{
            "name": "Task name",
            "description": "Detailed description",
            "agent": "agent_name",
            "priority": 1-10,
            "duration_minutes": 5-60,
            "risk_level": "low|medium|high|critical",
            "input": {{...task parameters...}}
        }}
    ],
    "total_estimated_minutes": 60,
    "risk_assessment": "Overall risk assessment"
}}
"""


REACT_PLANNING_PROMPT = """You are using the ReAct (Reason-Act-Think) pattern to plan.

GOAL: {goal}
CONTEXT: {context}

Previous steps:
{previous_steps}

Current iteration: {iteration}

Think step by step:
1. THOUGHT: What do I know? What do I need to find out?
2. ACTION: What specific action should I take next?
3. (After action) OBSERVATION: What did I learn?

If you have enough information to create the full plan, respond with:
THOUGHT: I have enough information to create the plan.
FINAL PLAN:
[Your complete plan in JSON format]

Otherwise, respond with:
THOUGHT: [Your reasoning]
ACTION: [The action to take]
ACTION_INPUT: [Parameters for the action as JSON]
"""


REPLAN_PROMPT = """A task in your plan failed. Create an alternative approach.

ORIGINAL GOAL: {goal}
FAILED TASK: {failed_task}
FAILURE REASON: {failure_reason}
PREVIOUS PLAN: {previous_plan}

CONTEXT:
{context}

Create an alternative plan that:
1. Avoids the approach that failed
2. Achieves the same goal through a different method
3. Is more conservative to reduce risk of another failure
4. May break the failed step into smaller, safer steps

Respond with the same JSON task format as before.
"""
```

---

### Task 4.5: Update Planner to Use ReAct
**Priority:** 🔴 Critical
**Estimated Time:** 2 hours
**Dependencies:** Tasks 4.2, 4.3

#### File: `src/master_ai/planner.py` (REPLACE ENTIRE FILE)
```python
"""
Planner - High-level planning interface.
Wraps ReActPlanner for backward compatibility and provides simple API.
"""

import json
from typing import Dict, Any, Optional

from src.master_ai.react_planner import ReActPlanner
from src.master_ai.planning_models import ExecutionPlan, PlanTask
from src.utils.llm_router import LLMRouter
from src.utils.structured_logging import get_logger

logger = get_logger("planner")


class Planner:
    """
    High-level planner that translates user goals into execution plans.
    Uses ReActPlanner internally for sophisticated planning.
    """
    
    def __init__(self, llm: LLMRouter):
        """
        Initialize the planner.
        
        Args:
            llm: LLM router for inference
        """
        self.llm = llm
        self.react_planner = ReActPlanner(llm)
    
    async def create_plan(
        self,
        goal: str,
        action: str = None,
        parameters: Dict[str, Any] = None,
        context: str = ""
    ) -> Dict[str, Any]:
        """
        Create an execution plan for a goal.
        
        Args:
            goal: The user's goal
            action: Specific action type
            parameters: Extracted parameters
            context: Current empire context
            
        Returns:
            Plan dictionary with steps (backward compatible format)
        """
        try:
            # Use ReAct planner for sophisticated planning
            plan = await self.react_planner.create_plan(
                goal=goal,
                context=context,
                action=action,
                parameters=parameters
            )
            
            # Convert to backward-compatible format
            return self._to_legacy_format(plan)
            
        except Exception as e:
            logger.error("Planning failed", error=str(e))
            # Return fallback plan
            return self._create_fallback_plan(goal, str(e))
    
    def _to_legacy_format(self, plan: ExecutionPlan) -> Dict[str, Any]:
        """Convert ExecutionPlan to legacy dictionary format."""
        return {
            "id": plan.id,
            "goal": plan.goal,
            "status": plan.status,
            "steps": [
                {
                    "id": task.id,
                    "name": task.name,
                    "description": task.description,
                    "agent": task.agent,
                    "requires_approval": task.requires_approval,
                    "dependencies": task.depends_on,
                    "estimated_duration": f"{task.estimated_duration_minutes} minutes",
                    "risk_level": task.risk_level.value,
                    "input": task.input_data,
                    "type": task.agent  # Alias for backward compatibility
                }
                for task in plan.tasks
            ],
            "total_estimated_duration": f"{plan.estimated_duration_minutes} minutes",
            "requires_human_review": plan.requires_human_review,
            "overall_risk": plan.overall_risk.value
        }
    
    def _create_fallback_plan(self, goal: str, error: str) -> Dict[str, Any]:
        """Create a fallback plan when planning fails."""
        return {
            "goal": goal,
            "status": "fallback",
            "steps": [
                {
                    "name": "Manual Review Required",
                    "description": f"Automatic planning failed: {error}. Please review and create plan manually.",
                    "agent": "research",
                    "requires_approval": True,
                    "dependencies": [],
                    "estimated_duration": "unknown",
                    "risk_level": "high",
                    "input": {"original_goal": goal, "error": error}
                }
            ],
            "requires_human_review": True,
            "overall_risk": "high"
        }
    
    async def get_plan_model(
        self,
        goal: str,
        action: str = None,
        parameters: Dict[str, Any] = None,
        context: str = ""
    ) -> ExecutionPlan:
        """
        Get an ExecutionPlan model directly.
        Use this for new code that wants the full model.
        """
        return await self.react_planner.create_plan(
            goal=goal,
            context=context,
            action=action,
            parameters=parameters
        )
```

---

## Testing Requirements

### Unit Tests

#### File: `tests/test_planning.py` (CREATE NEW FILE)
```python
"""
Tests for planning and execution system.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime

from src.master_ai.planning_models import (
    ExecutionPlan, PlanTask, TaskStatus, RiskLevel, PlanningContext
)
from src.master_ai.react_planner import ReActPlanner
from src.master_ai.plan_executor import PlanExecutor
from src.master_ai.planner import Planner


class TestPlanTask:
    """Tests for PlanTask model."""
    
    def test_can_execute_no_dependencies(self):
        """Task with no dependencies can execute."""
        task = PlanTask(name="Test", description="Test", agent="research")
        assert task.can_execute(set())
    
    def test_can_execute_with_satisfied_dependencies(self):
        """Task executes when dependencies are met."""
        task = PlanTask(
            name="Test",
            description="Test",
            agent="research",
            depends_on=["task-1", "task-2"]
        )
        assert task.can_execute({"task-1", "task-2", "task-3"})
    
    def test_cannot_execute_with_unsatisfied_dependencies(self):
        """Task blocks when dependencies aren't met."""
        task = PlanTask(
            name="Test",
            description="Test",
            agent="research",
            depends_on=["task-1", "task-2"]
        )
        assert not task.can_execute({"task-1"})


class TestExecutionPlan:
    """Tests for ExecutionPlan model."""
    
    def test_get_ready_tasks(self):
        """Get tasks ready for execution."""
        plan = ExecutionPlan(goal="Test")
        
        task1 = PlanTask(id="1", name="First", description="", agent="research")
        task2 = PlanTask(id="2", name="Second", description="", agent="commerce", depends_on=["1"])
        
        plan.tasks = [task1, task2]
        
        ready = plan.get_ready_tasks()
        assert len(ready) == 1
        assert ready[0].id == "1"
    
    def test_get_next_task_priority(self):
        """Next task respects priority."""
        plan = ExecutionPlan(goal="Test")
        
        task1 = PlanTask(id="1", name="Low", description="", agent="research", priority=10)
        task2 = PlanTask(id="2", name="High", description="", agent="research", priority=1)
        
        plan.tasks = [task1, task2]
        
        next_task = plan.get_next_task()
        assert next_task.id == "2"
    
    def test_update_metrics(self):
        """Metrics update correctly."""
        plan = ExecutionPlan(goal="Test")
        
        task1 = PlanTask(id="1", name="Done", description="", agent="research", status=TaskStatus.COMPLETED)
        task2 = PlanTask(id="2", name="Failed", description="", agent="commerce", status=TaskStatus.FAILED)
        task3 = PlanTask(id="3", name="Pending", description="", agent="content")
        
        plan.tasks = [task1, task2, task3]
        plan.update_metrics()
        
        assert plan.total_tasks == 3
        assert plan.completed_tasks == 1
        assert plan.failed_tasks == 1


class TestReActPlanner:
    """Tests for ReAct planner."""
    
    @pytest.fixture
    def planner(self):
        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock(return_value='''
        {
            "tasks": [
                {"name": "Research", "description": "Do research", "agent": "research", "priority": 1},
                {"name": "Setup", "description": "Setup store", "agent": "commerce", "priority": 2}
            ]
        }
        ''')
        return ReActPlanner(mock_llm)
    
    @pytest.mark.asyncio
    async def test_create_plan(self, planner):
        """Test plan creation."""
        plan = await planner.create_plan(
            goal="Start a business",
            context="Empire context"
        )
        
        assert plan.status == "ready"
        assert len(plan.tasks) == 2
    
    @pytest.mark.asyncio
    async def test_dependency_building(self, planner):
        """Test that dependencies are built."""
        plan = await planner.create_plan(
            goal="Start a business",
            context="Empire context"
        )
        
        # Commerce should depend on research
        commerce_task = next(t for t in plan.tasks if t.agent == "commerce")
        research_task = next(t for t in plan.tasks if t.agent == "research")
        
        assert research_task.id in commerce_task.depends_on


class TestPlanExecutor:
    """Tests for plan executor."""
    
    @pytest.fixture
    def executor(self):
        mock_planner = MagicMock()
        mock_router = MagicMock()
        mock_router.execute = AsyncMock(return_value={"success": True, "output": "Done"})
        
        return PlanExecutor(
            planner=mock_planner,
            agent_router=mock_router
        )
    
    @pytest.mark.asyncio
    async def test_execute_simple_plan(self, executor):
        """Test executing a simple plan."""
        plan = ExecutionPlan(goal="Test")
        task = PlanTask(
            id="1",
            name="Simple task",
            description="Test",
            agent="research"
        )
        plan.tasks = [task]
        plan.update_metrics()
        
        with patch('src.master_ai.plan_executor.get_db') as mock_db:
            mock_session = AsyncMock()
            mock_session.get.return_value = None
            mock_db.return_value.__aenter__.return_value = mock_session
            
            result = await executor.execute_plan(plan)
        
        assert result.status == "completed"
        assert result.completed_tasks == 1
    
    @pytest.mark.asyncio
    async def test_approval_pauses_plan(self, executor):
        """Test that approval requirement pauses plan."""
        plan = ExecutionPlan(goal="Test")
        task = PlanTask(
            id="1",
            name="Needs approval",
            description="Test",
            agent="finance",
            requires_approval=True
        )
        plan.tasks = [task]
        plan.update_metrics()
        
        with patch('src.master_ai.plan_executor.get_db') as mock_db:
            mock_session = AsyncMock()
            mock_db.return_value.__aenter__.return_value = mock_session
            
            result = await executor.execute_plan(plan, auto_continue=False)
        
        assert result.status == "paused"
        assert task.status == TaskStatus.WAITING_APPROVAL
```

---

## Acceptance Criteria

### Part 4 Completion Checklist

- [ ] **Planning Models**
  - [ ] `src/master_ai/planning_models.py` created
  - [ ] PlanTask with dependencies working
  - [ ] ExecutionPlan with metrics working
  - [ ] ReActStep for tracing

- [ ] **ReAct Planner**
  - [ ] `src/master_ai/react_planner.py` created
  - [ ] Goal decomposition working
  - [ ] Dependency graph building
  - [ ] Risk assessment working
  - [ ] Topological ordering working

- [ ] **Plan Executor**
  - [ ] `src/master_ai/plan_executor.py` created
  - [ ] Task execution working
  - [ ] Approval gates working
  - [ ] Failure handling working
  - [ ] State persistence working

- [ ] **Planner Updated**
  - [ ] `src/master_ai/planner.py` updated
  - [ ] Backward compatible API
  - [ ] Uses ReActPlanner internally

- [ ] **Tests Passing**
  - [ ] All unit tests pass
  - [ ] Integration tests pass

---

## File Summary

| Action | File Path |
|--------|-----------|
| CREATE | `src/master_ai/planning_models.py` |
| CREATE | `src/master_ai/react_planner.py` |
| CREATE | `src/master_ai/plan_executor.py` |
| REPLACE | `src/master_ai/planner.py` |
| MODIFY | `src/master_ai/prompts.py` |
| CREATE | `tests/test_planning.py` |

---

## Next Part Preview

**Part 5: Evolution Engine - Code Modification System** will cover:
- Enhanced EvolutionEngine for self-modification
- Code generation and patching system
- AST-based code analysis
- Safe modification proposals
- Git integration for versioning

---

*End of Part 4 - Master AI Brain - Planning & ReAct Implementation*
