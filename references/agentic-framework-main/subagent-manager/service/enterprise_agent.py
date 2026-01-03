"""
Enterprise Agent with Natural Reasoning + Structured Auditability.

Module: subagent-manager/service/enterprise_agent.py

Implements THINK -> PLAN -> APPROVE -> EXECUTE -> VALIDATE -> REFLECT loop
with full provenance tracking for enterprise compliance.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Callable
from uuid import uuid4

from pydantic import BaseModel, Field

from .reflective_agent import (
    ExecutionPlan,
    PlanStep,
    ReflectiveAgentConfig,
    ValidationResult,
    RefinementAction,
)
from .governance import GovernanceGate, ApprovalResult, PolicyViolation
from .provenance import ProvenanceTracker, ProvenanceRecord
from .audit_logger import AuditLogger, AuditEvent, AuditPhase


logger = logging.getLogger(__name__)


class EnterprisePhase(str, Enum):
    """Phases in enterprise agent execution."""

    THINK = "think"           # Natural reasoning
    PLAN = "plan"             # Structured planning
    APPROVE = "approve"       # Governance gate
    EXECUTE = "execute"       # Tracked execution
    VALIDATE = "validate"     # Measurable validation
    REFLECT = "reflect"       # Learning & refinement
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"       # Blocked by governance


class ThinkingTrace(BaseModel):
    """Captured natural language reasoning."""

    trace_id: str = Field(default_factory=lambda: str(uuid4())[:12])
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    reasoning: str = Field(..., description="Natural language reasoning")
    key_insights: List[str] = Field(default_factory=list)
    identified_risks: List[str] = Field(default_factory=list)
    proposed_approach: str = Field(default="")
    confidence_assessment: str = Field(default="")
    duration_ms: float = Field(0.0)


class EnterpriseExecutionPlan(ExecutionPlan):
    """Extended execution plan with enterprise fields."""

    thinking_trace: Optional[ThinkingTrace] = Field(
        None, description="Natural reasoning that led to this plan"
    )
    risk_level: str = Field(
        default="medium", description="low/medium/high/critical"
    )
    requires_approval: bool = Field(default=False)
    approval_reason: Optional[str] = Field(None)
    estimated_impact: str = Field(default="")
    rollback_plan: Optional[str] = Field(None)


class ExecutionStepResult(BaseModel):
    """Result of executing a single step."""

    step_id: str
    action: str
    tool_used: Optional[str] = None
    inputs_hash: str = Field(default="")
    outputs_hash: str = Field(default="")
    status: str = Field(default="pending")  # pending/running/success/failed
    output: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_ms: float = Field(0.0)
    provenance_id: Optional[str] = None


class EnterpriseExecutionResult(BaseModel):
    """Complete execution result with full audit trail."""

    execution_id: str = Field(default_factory=lambda: str(uuid4()))
    plan_id: str
    steps: List[ExecutionStepResult] = Field(default_factory=list)
    overall_status: str = Field(default="pending")
    success_rate: float = Field(0.0)
    total_duration_ms: float = Field(0.0)
    provenance_chain: List[str] = Field(default_factory=list)


class EnterpriseValidationResult(ValidationResult):
    """Extended validation with evidence."""

    validation_id: str = Field(default_factory=lambda: str(uuid4())[:12])
    evidence: List[Dict[str, Any]] = Field(default_factory=list)
    validator: str = Field(default="self")  # self/peer/human
    validation_method: str = Field(default="criteria_check")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ReflectionSummary(BaseModel):
    """Post-execution reflection for learning."""

    reflection_id: str = Field(default_factory=lambda: str(uuid4())[:12])
    what_worked: List[str] = Field(default_factory=list)
    what_failed: List[str] = Field(default_factory=list)
    lessons_learned: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    should_retry: bool = Field(default=False)
    refinement_action: Optional[RefinementAction] = None


@dataclass
class EnterpriseAgentState:
    """Complete state of enterprise agent execution."""

    # Identifiers
    execution_id: str = field(default_factory=lambda: str(uuid4()))
    task_id: str = ""
    original_task: str = ""

    # Phase tracking
    current_phase: EnterprisePhase = EnterprisePhase.THINK
    phases_completed: List[str] = field(default_factory=list)
    iteration: int = 0
    max_iterations: int = 3

    # THINK phase
    thinking_traces: List[ThinkingTrace] = field(default_factory=list)

    # PLAN phase
    current_plan: Optional[EnterpriseExecutionPlan] = None
    plan_history: List[EnterpriseExecutionPlan] = field(default_factory=list)

    # APPROVE phase
    approval_results: List[ApprovalResult] = field(default_factory=list)
    policy_violations: List[PolicyViolation] = field(default_factory=list)

    # EXECUTE phase
    execution_results: List[EnterpriseExecutionResult] = field(default_factory=list)

    # VALIDATE phase
    validation_results: List[EnterpriseValidationResult] = field(default_factory=list)

    # REFLECT phase
    reflections: List[ReflectionSummary] = field(default_factory=list)

    # Outcome
    final_output: Optional[Dict[str, Any]] = None
    success: bool = False
    blocked: bool = False
    error_message: Optional[str] = None

    # Metrics
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    total_llm_calls: int = 0
    total_tool_calls: int = 0
    total_approvals_required: int = 0
    total_approvals_granted: int = 0

    # Audit
    audit_trail: List[str] = field(default_factory=list)  # List of AuditEvent IDs


class EnterpriseAgentConfig(ReflectiveAgentConfig):
    """Configuration for enterprise agent."""

    # Governance
    enable_governance: bool = Field(True, description="Enable governance gates")
    auto_approve_low_risk: bool = Field(True, description="Auto-approve low risk actions")
    require_human_approval_for: List[str] = Field(
        default_factory=lambda: ["file_write", "file_delete", "bash_exec", "deploy"],
        description="Actions requiring human approval",
    )
    max_risk_level_auto_approve: str = Field(
        "low", description="Max risk level for auto-approval"
    )

    # Audit
    enable_full_audit: bool = Field(True, description="Enable comprehensive audit logging")
    log_reasoning_traces: bool = Field(True, description="Log natural language reasoning")
    log_all_tool_calls: bool = Field(True, description="Log all tool invocations")

    # Provenance
    enable_provenance: bool = Field(True, description="Track provenance for all artifacts")
    hash_algorithm: str = Field("sha256", description="Hash algorithm for provenance")

    # Thinking
    enable_extended_thinking: bool = Field(True, description="Enable extended reasoning")
    thinking_budget_tokens: int = Field(1000, description="Token budget for thinking")


class EnterpriseAgent:
    """
    Enterprise-grade agent with natural reasoning and full auditability.

    Implements: THINK -> PLAN -> APPROVE -> EXECUTE -> VALIDATE -> REFLECT

    Features:
    - Natural language reasoning (THINK phase)
    - Structured execution plans (PLAN phase)
    - Governance gates with policy enforcement (APPROVE phase)
    - Fully tracked execution with provenance (EXECUTE phase)
    - Measurable validation with evidence (VALIDATE phase)
    - Learning and refinement (REFLECT phase)
    - Complete audit trail for compliance
    """

    def __init__(
        self,
        llm_adapter: Any,
        config: Optional[EnterpriseAgentConfig] = None,
        tool_executor: Optional[Any] = None,
        governance_gate: Optional[GovernanceGate] = None,
        provenance_tracker: Optional[ProvenanceTracker] = None,
        audit_logger: Optional[AuditLogger] = None,
    ):
        """
        Initialize enterprise agent.

        Args:
            llm_adapter: LLM adapter for reasoning
            config: Agent configuration
            tool_executor: Tool/skill executor
            governance_gate: Policy enforcement gate
            provenance_tracker: Provenance tracking
            audit_logger: Audit logging
        """
        self.llm = llm_adapter
        self.config = config or EnterpriseAgentConfig()
        self.tool_executor = tool_executor
        self.governance = governance_gate or GovernanceGate()
        self.provenance = provenance_tracker or ProvenanceTracker()
        self.audit = audit_logger or AuditLogger()

        # Import prompts
        from .planning_prompts import PlanningPrompts
        self.prompts = PlanningPrompts()

    async def execute_task(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
        constraints: Optional[List[str]] = None,
        user_id: Optional[str] = None,
    ) -> EnterpriseAgentState:
        """
        Execute a task with full enterprise controls.

        Args:
            task: Task description
            context: Additional context
            constraints: Constraints to follow
            user_id: ID of requesting user (for audit)

        Returns:
            EnterpriseAgentState with complete execution history
        """
        state = EnterpriseAgentState(
            task_id=str(uuid4()),
            original_task=task,
            max_iterations=self.config.max_iterations,
        )

        # Log execution start
        self.audit.log_event(AuditEvent(
            phase=AuditPhase.START,
            execution_id=state.execution_id,
            task=task,
            user_id=user_id,
            metadata={"context": context, "constraints": constraints},
        ))

        logger.info(f"Starting enterprise execution: {task[:100]}...")

        try:
            while state.iteration < state.max_iterations:
                state.iteration += 1
                logger.info(f"=== Iteration {state.iteration}/{state.max_iterations} ===")

                # PHASE 1: THINK (Natural Reasoning)
                if self.config.enable_extended_thinking:
                    state.current_phase = EnterprisePhase.THINK
                    thinking = await self._think(task, context, constraints, state)
                    state.thinking_traces.append(thinking)
                    state.phases_completed.append(f"think_{state.iteration}")

                    self.audit.log_event(AuditEvent(
                        phase=AuditPhase.THINK,
                        execution_id=state.execution_id,
                        reasoning_trace=thinking.reasoning,
                        duration_ms=thinking.duration_ms,
                    ))

                # PHASE 2: PLAN (Structured)
                state.current_phase = EnterprisePhase.PLAN
                plan = await self._plan(task, context, constraints, state)
                state.current_plan = plan
                state.plan_history.append(plan)
                state.phases_completed.append(f"plan_{state.iteration}")

                self.audit.log_event(AuditEvent(
                    phase=AuditPhase.PLAN,
                    execution_id=state.execution_id,
                    plan_id=plan.plan_id,
                    confidence=plan.confidence,
                    risk_level=plan.risk_level,
                    steps_count=len(plan.steps),
                ))

                # PHASE 3: APPROVE (Governance Gate)
                if self.config.enable_governance:
                    state.current_phase = EnterprisePhase.APPROVE
                    approval = await self._approve(plan, state, user_id)
                    state.approval_results.append(approval)
                    state.phases_completed.append(f"approve_{state.iteration}")

                    self.audit.log_event(AuditEvent(
                        phase=AuditPhase.APPROVE,
                        execution_id=state.execution_id,
                        approved=approval.approved,
                        approver=approval.approver,
                        violations=[v.description for v in approval.violations],
                    ))

                    if not approval.approved:
                        if approval.requires_human:
                            state.current_phase = EnterprisePhase.BLOCKED
                            state.blocked = True
                            state.error_message = "Blocked: Requires human approval"
                            logger.warning("Execution blocked - requires human approval")
                            break
                        else:
                            # Try to refine plan to address violations
                            logger.info("Plan not approved, refining...")
                            continue

                # PHASE 4: EXECUTE (Tracked)
                state.current_phase = EnterprisePhase.EXECUTE
                execution_result = await self._execute(plan, state, user_id)
                state.execution_results.append(execution_result)
                state.phases_completed.append(f"execute_{state.iteration}")

                self.audit.log_event(AuditEvent(
                    phase=AuditPhase.EXECUTE,
                    execution_id=state.execution_id,
                    steps_completed=len([s for s in execution_result.steps if s.status == "success"]),
                    steps_failed=len([s for s in execution_result.steps if s.status == "failed"]),
                    success_rate=execution_result.success_rate,
                    provenance_chain=execution_result.provenance_chain,
                ))

                # PHASE 5: VALIDATE (Measurable)
                state.current_phase = EnterprisePhase.VALIDATE
                validation = await self._validate(execution_result, plan, state)
                state.validation_results.append(validation)
                state.phases_completed.append(f"validate_{state.iteration}")

                self.audit.log_event(AuditEvent(
                    phase=AuditPhase.VALIDATE,
                    execution_id=state.execution_id,
                    is_valid=validation.is_valid,
                    score=validation.score,
                    criteria_met=validation.criteria_met,
                    evidence=validation.evidence,
                ))

                if validation.is_valid and validation.score >= self.config.min_quality_score:
                    # Success!
                    state.current_phase = EnterprisePhase.COMPLETED
                    state.success = True
                    state.final_output = {
                        "execution_result": execution_result.model_dump(),
                        "validation": validation.model_dump(),
                    }
                    logger.info(f"SUCCESS: Score {validation.score:.2f}")
                    break

                # PHASE 6: REFLECT & REFINE
                state.current_phase = EnterprisePhase.REFLECT
                reflection = await self._reflect(execution_result, validation, state)
                state.reflections.append(reflection)
                state.phases_completed.append(f"reflect_{state.iteration}")

                self.audit.log_event(AuditEvent(
                    phase=AuditPhase.REFLECT,
                    execution_id=state.execution_id,
                    lessons_learned=reflection.lessons_learned,
                    should_retry=reflection.should_retry,
                    refinement_action=reflection.refinement_action.action_type if reflection.refinement_action else None,
                ))

                if not reflection.should_retry:
                    state.current_phase = EnterprisePhase.FAILED
                    state.error_message = "Decided not to retry after reflection"
                    break

            # Check if exhausted iterations
            if state.iteration >= state.max_iterations and not state.success:
                state.current_phase = EnterprisePhase.FAILED
                state.error_message = f"Max iterations ({state.max_iterations}) exhausted"

        except Exception as e:
            logger.exception(f"Enterprise execution failed: {e}")
            state.current_phase = EnterprisePhase.FAILED
            state.error_message = str(e)

            self.audit.log_event(AuditEvent(
                phase=AuditPhase.ERROR,
                execution_id=state.execution_id,
                error=str(e),
            ))

        finally:
            state.completed_at = datetime.utcnow()

            # Final audit log
            self.audit.log_event(AuditEvent(
                phase=AuditPhase.COMPLETE,
                execution_id=state.execution_id,
                success=state.success,
                blocked=state.blocked,
                iterations_used=state.iteration,
                total_duration_ms=(state.completed_at - state.started_at).total_seconds() * 1000,
                final_phase=state.current_phase.value,
            ))

        return state

    async def _think(
        self,
        task: str,
        context: Optional[Dict[str, Any]],
        constraints: Optional[List[str]],
        state: EnterpriseAgentState,
    ) -> ThinkingTrace:
        """
        Natural reasoning phase - think before planning.

        This captures the agent's reasoning process in natural language,
        which is crucial for auditability and explainability.
        """
        import time
        start_time = time.time()

        logger.info("THINK: Reasoning about the task...")

        prompt = f"""You are an expert agent. Think carefully about this task before acting.

## Task
{task}

## Context
{self._format_context(context)}

## Constraints
{self._format_list(constraints)}

## Previous Attempts
{self._format_previous_attempts(state)}

## Instructions
Think through this task step by step:

1. **Understanding**: What exactly is being asked?
2. **Approach**: What's the best way to accomplish this?
3. **Risks**: What could go wrong?
4. **Dependencies**: What do I need to succeed?
5. **Success**: How will I know I've succeeded?

Be thorough but concise. This reasoning will be logged for audit purposes."""

        response = await self._call_llm(prompt, state)
        state.total_llm_calls += 1

        duration_ms = (time.time() - start_time) * 1000

        # Parse reasoning into structured trace
        trace = ThinkingTrace(
            reasoning=response,
            key_insights=self._extract_insights(response),
            identified_risks=self._extract_risks(response),
            proposed_approach=self._extract_approach(response),
            duration_ms=duration_ms,
        )

        logger.info(f"THINK complete: {len(trace.key_insights)} insights, {len(trace.identified_risks)} risks identified")

        return trace

    async def _plan(
        self,
        task: str,
        context: Optional[Dict[str, Any]],
        constraints: Optional[List[str]],
        state: EnterpriseAgentState,
    ) -> EnterpriseExecutionPlan:
        """Create structured execution plan."""
        logger.info("PLAN: Creating execution plan...")

        # Use thinking trace if available
        thinking = state.thinking_traces[-1] if state.thinking_traces else None

        prompt = self.prompts.build_planning_prompt(
            task=task,
            context=context,
            constraints=constraints,
            previous_attempts=state.plan_history,
            available_tools=self._get_available_tools(),
        )

        if thinking:
            prompt = f"""Based on this reasoning:

{thinking.reasoning}

{prompt}"""

        response = await self._call_llm(prompt, state)
        state.total_llm_calls += 1

        plan = self._parse_enterprise_plan(response, thinking)

        # Assess risk level
        plan.risk_level = self._assess_risk_level(plan)
        plan.requires_approval = self._requires_approval(plan)

        logger.info(f"PLAN complete: {len(plan.steps)} steps, risk={plan.risk_level}, confidence={plan.confidence:.2f}")

        return plan

    async def _approve(
        self,
        plan: EnterpriseExecutionPlan,
        state: EnterpriseAgentState,
        user_id: Optional[str],
    ) -> ApprovalResult:
        """Governance gate - check policies and get approval."""
        logger.info("APPROVE: Checking governance policies...")

        result = await self.governance.evaluate(
            plan=plan,
            context={
                "user_id": user_id,
                "iteration": state.iteration,
                "previous_violations": state.policy_violations,
            },
            config=self.config,
        )

        if result.approved:
            state.total_approvals_granted += 1
            logger.info(f"APPROVE: Approved by {result.approver}")
        else:
            state.policy_violations.extend(result.violations)
            logger.warning(f"APPROVE: Denied - {len(result.violations)} violations")

        state.total_approvals_required += 1

        return result

    async def _execute(
        self,
        plan: EnterpriseExecutionPlan,
        state: EnterpriseAgentState,
        user_id: Optional[str],
    ) -> EnterpriseExecutionResult:
        """Execute plan with full tracking."""
        import time

        logger.info("EXECUTE: Running plan steps...")

        result = EnterpriseExecutionResult(plan_id=plan.plan_id)
        start_time = time.time()

        for i, step in enumerate(plan.steps):
            step_start = time.time()
            logger.info(f"  Step {i+1}/{len(plan.steps)}: {step.action}")

            step_result = ExecutionStepResult(
                step_id=step.step_id,
                action=step.action,
                tool_used=step.tool,
                started_at=datetime.utcnow(),
            )

            try:
                # Calculate input hash for provenance
                step_result.inputs_hash = self.provenance.hash_data(step.inputs)

                # Execute the step
                if step.tool and self.tool_executor:
                    output = self.tool_executor.execute(step.tool, step.inputs)
                    state.total_tool_calls += 1
                else:
                    output = await self._execute_step_with_llm(step, state)

                # Calculate output hash
                step_result.outputs_hash = self.provenance.hash_data(output)
                step_result.output = output
                step_result.status = "success"

                # Record provenance
                prov_record = self.provenance.record(
                    actor_id=state.execution_id,
                    actor_type="enterprise_agent",
                    action=step.action,
                    inputs_hash=step_result.inputs_hash,
                    outputs_hash=step_result.outputs_hash,
                    tool_id=step.tool,
                )
                step_result.provenance_id = prov_record.provenance_id
                result.provenance_chain.append(prov_record.provenance_id)

            except Exception as e:
                step_result.status = "failed"
                step_result.error = str(e)
                logger.warning(f"  Step failed: {e}")

            step_result.completed_at = datetime.utcnow()
            step_result.duration_ms = (time.time() - step_start) * 1000
            result.steps.append(step_result)

        # Calculate overall metrics
        successful = len([s for s in result.steps if s.status == "success"])
        result.success_rate = successful / len(result.steps) if result.steps else 0
        result.overall_status = "success" if result.success_rate == 1.0 else "partial" if result.success_rate > 0 else "failed"
        result.total_duration_ms = (time.time() - start_time) * 1000

        logger.info(f"EXECUTE complete: {successful}/{len(result.steps)} steps succeeded")

        return result

    async def _validate(
        self,
        execution_result: EnterpriseExecutionResult,
        plan: EnterpriseExecutionPlan,
        state: EnterpriseAgentState,
    ) -> EnterpriseValidationResult:
        """Validate results with evidence."""
        logger.info("VALIDATE: Checking results against criteria...")

        prompt = self.prompts.build_validation_prompt(
            task=state.original_task,
            plan=plan,
            execution_result=execution_result.model_dump(),
            success_criteria=plan.success_criteria,
            strictness=self.config.validation_strictness,
        )

        response = await self._call_llm(prompt, state)
        state.total_llm_calls += 1

        validation = self._parse_enterprise_validation(response, plan, execution_result)

        logger.info(f"VALIDATE complete: valid={validation.is_valid}, score={validation.score:.2f}")

        return validation

    async def _reflect(
        self,
        execution_result: EnterpriseExecutionResult,
        validation: EnterpriseValidationResult,
        state: EnterpriseAgentState,
    ) -> ReflectionSummary:
        """Reflect on execution for learning."""
        logger.info("REFLECT: Analyzing execution for improvements...")

        prompt = f"""Reflect on this execution attempt:

## Task
{state.original_task}

## Execution Result
- Success rate: {execution_result.success_rate:.2%}
- Steps completed: {len([s for s in execution_result.steps if s.status == 'success'])}/{len(execution_result.steps)}
- Errors: {[s.error for s in execution_result.steps if s.error]}

## Validation
- Valid: {validation.is_valid}
- Score: {validation.score:.2f}
- Errors: {validation.errors}

## Questions
1. What worked well?
2. What failed and why?
3. What lessons can be learned?
4. Should we retry with a different approach?
5. If retrying, what should change?

Provide a structured reflection."""

        response = await self._call_llm(prompt, state)
        state.total_llm_calls += 1

        reflection = self._parse_reflection(response, validation)

        logger.info(f"REFLECT complete: should_retry={reflection.should_retry}")

        return reflection

    # Helper methods

    async def _call_llm(self, prompt: str, state: EnterpriseAgentState) -> str:
        """Call LLM with prompt."""
        if hasattr(self.llm, "generate"):
            return await self.llm.generate(prompt)
        elif hasattr(self.llm, "chat"):
            return await self.llm.chat(prompt)
        else:
            # Fallback for testing
            return f"[LLM Response to: {prompt[:100]}...]"

    async def _execute_step_with_llm(
        self, step: PlanStep, state: EnterpriseAgentState
    ) -> Dict[str, Any]:
        """Execute a step using LLM when no tool is specified."""
        prompt = f"Execute: {step.action}\nInputs: {step.inputs}\nExpected: {step.expected_output}"
        response = await self._call_llm(prompt, state)
        return {"response": response, "status": "completed"}

    def _get_available_tools(self) -> List[str]:
        """Get list of available tools."""
        if self.tool_executor and hasattr(self.tool_executor, "list_tools"):
            return self.tool_executor.list_tools()
        return ["file_read", "file_write", "file_edit", "file_glob", "file_grep", "bash_exec", "python_exec"]

    def _assess_risk_level(self, plan: EnterpriseExecutionPlan) -> str:
        """Assess risk level of a plan."""
        high_risk_tools = {"bash_exec", "file_write", "file_delete", "deploy"}
        medium_risk_tools = {"file_edit", "python_exec"}

        tools_used = {step.tool for step in plan.steps if step.tool}

        if tools_used & high_risk_tools:
            return "high"
        elif tools_used & medium_risk_tools:
            return "medium"
        else:
            return "low"

    def _requires_approval(self, plan: EnterpriseExecutionPlan) -> bool:
        """Check if plan requires approval."""
        if plan.risk_level in ["high", "critical"]:
            return True

        tools_used = {step.tool for step in plan.steps if step.tool}
        requires_approval_tools = set(self.config.require_human_approval_for)

        return bool(tools_used & requires_approval_tools)

    def _format_context(self, context: Optional[Dict[str, Any]]) -> str:
        """Format context for prompts."""
        if not context:
            return "None provided"
        return "\n".join(f"- {k}: {v}" for k, v in context.items())

    def _format_list(self, items: Optional[List[str]]) -> str:
        """Format list for prompts."""
        if not items:
            return "None"
        return "\n".join(f"- {item}" for item in items)

    def _format_previous_attempts(self, state: EnterpriseAgentState) -> str:
        """Format previous attempts for context."""
        if not state.plan_history:
            return "None"

        lines = []
        for i, plan in enumerate(state.plan_history, 1):
            lines.append(f"Attempt {i}: {plan.approach} (confidence: {plan.confidence:.2f})")
        return "\n".join(lines)

    def _extract_insights(self, reasoning: str) -> List[str]:
        """Extract key insights from reasoning."""
        # Simple extraction - in production, use NLP
        insights = []
        for line in reasoning.split("\n"):
            if any(kw in line.lower() for kw in ["key", "important", "note", "insight"]):
                insights.append(line.strip())
        return insights[:5]  # Limit to 5

    def _extract_risks(self, reasoning: str) -> List[str]:
        """Extract identified risks from reasoning."""
        risks = []
        for line in reasoning.split("\n"):
            if any(kw in line.lower() for kw in ["risk", "danger", "careful", "warning", "could fail"]):
                risks.append(line.strip())
        return risks[:5]

    def _extract_approach(self, reasoning: str) -> str:
        """Extract proposed approach from reasoning."""
        for line in reasoning.split("\n"):
            if any(kw in line.lower() for kw in ["approach", "strategy", "plan", "will"]):
                return line.strip()
        return ""

    def _parse_enterprise_plan(
        self, response: str, thinking: Optional[ThinkingTrace]
    ) -> EnterpriseExecutionPlan:
        """Parse LLM response into enterprise plan."""
        import json
        import re

        # Try to find JSON
        json_match = re.search(r"\{[\s\S]*\}", response)
        if json_match:
            try:
                data = json.loads(json_match.group())
                plan = EnterpriseExecutionPlan(**data)
                plan.thinking_trace = thinking
                return plan
            except (json.JSONDecodeError, ValueError):
                pass

        # Fallback
        return EnterpriseExecutionPlan(
            task_understanding=response[:200],
            approach=response[:500],
            steps=[PlanStep(action="Execute task", expected_output="Task completed")],
            success_criteria=["Task completed successfully"],
            confidence=0.6,
            thinking_trace=thinking,
        )

    def _parse_enterprise_validation(
        self,
        response: str,
        plan: EnterpriseExecutionPlan,
        execution_result: EnterpriseExecutionResult,
    ) -> EnterpriseValidationResult:
        """Parse validation response."""
        import json
        import re

        json_match = re.search(r"\{[\s\S]*\}", response)
        if json_match:
            try:
                data = json.loads(json_match.group())
                return EnterpriseValidationResult(**data)
            except (json.JSONDecodeError, ValueError):
                pass

        # Infer from execution result
        is_valid = execution_result.success_rate >= 0.8
        score = execution_result.success_rate

        return EnterpriseValidationResult(
            is_valid=is_valid,
            score=score,
            errors=[s.error for s in execution_result.steps if s.error],
            evidence=[{"execution_success_rate": execution_result.success_rate}],
        )

    def _parse_reflection(
        self, response: str, validation: EnterpriseValidationResult
    ) -> ReflectionSummary:
        """Parse reflection response."""
        should_retry = validation.score < 0.9 and validation.score > 0.3

        return ReflectionSummary(
            what_worked=self._extract_insights(response),
            what_failed=[e for e in validation.errors],
            lessons_learned=["Captured from reflection"],
            recommendations=validation.suggestions,
            should_retry=should_retry,
            refinement_action=RefinementAction(
                action_type="modify" if should_retry else "abort",
                reasoning=response[:200],
            ) if not validation.is_valid else None,
        )
