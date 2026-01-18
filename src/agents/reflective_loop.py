"""
Reflective Agent Loop.

Self-validating agent execution: PLAN -> EXECUTE -> VALIDATE -> REFINE.
Based on agentic-framework reflective agent patterns.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
from uuid import uuid4
import asyncio

from src.utils.structured_logging import get_logger

logger = get_logger("reflective_agent")


class AgentPhase(str, Enum):
    """Phases of reflective execution."""
    PLAN = "plan"
    EXECUTE = "execute"
    VALIDATE = "validate"
    REFINE = "refine"
    COMPLETE = "complete"
    FAILED = "failed"


class ValidationResult(str, Enum):
    """Result of validation."""
    PASS = "pass"
    PARTIAL = "partial"
    FAIL = "fail"


@dataclass
class ExecutionPlan:
    """Plan for execution."""
    id: str = field(default_factory=lambda: f"plan_{uuid4().hex[:8]}")
    objective: str = ""
    steps: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    estimated_duration_ms: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "objective": self.objective,
            "steps": self.steps,
            "success_criteria": self.success_criteria,
            "constraints": self.constraints,
            "estimated_duration_ms": self.estimated_duration_ms,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class ExecutionResult:
    """Result of execution."""
    id: str = field(default_factory=lambda: f"exec_{uuid4().hex[:8]}")
    plan_id: str = ""
    success: bool = False
    output: Any = None
    error: Optional[str] = None
    steps_completed: int = 0
    duration_ms: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "plan_id": self.plan_id,
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "steps_completed": self.steps_completed,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
        }


@dataclass
class ValidationReport:
    """Report from validation phase."""
    id: str = field(default_factory=lambda: f"val_{uuid4().hex[:8]}")
    execution_id: str = ""
    result: ValidationResult = ValidationResult.FAIL
    quality_score: float = 0.0
    criteria_results: Dict[str, bool] = field(default_factory=dict)
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    
    @property
    def passed(self) -> bool:
        return self.result == ValidationResult.PASS
    
    @property
    def criteria_pass_rate(self) -> float:
        if not self.criteria_results:
            return 0.0
        passed = sum(1 for v in self.criteria_results.values() if v)
        return passed / len(self.criteria_results)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "execution_id": self.execution_id,
            "result": self.result.value,
            "quality_score": self.quality_score,
            "criteria_results": self.criteria_results,
            "issues": self.issues,
            "suggestions": self.suggestions,
            "passed": self.passed,
        }


@dataclass
class RefinementAction:
    """Action to take during refinement."""
    action_type: str  # "modify", "retry", "expand", "simplify"
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReflectionEntry:
    """Entry in the reflection log."""
    phase: AgentPhase
    timestamp: datetime = field(default_factory=datetime.utcnow)
    input_summary: str = ""
    output_summary: str = ""
    decisions_made: List[str] = field(default_factory=list)
    lessons: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "phase": self.phase.value,
            "timestamp": self.timestamp.isoformat(),
            "input_summary": self.input_summary,
            "output_summary": self.output_summary,
            "decisions_made": self.decisions_made,
            "lessons": self.lessons,
        }


@dataclass
class ReflectiveLoopResult:
    """Complete result of a reflective loop execution."""
    id: str = field(default_factory=lambda: f"loop_{uuid4().hex[:8]}")
    objective: str = ""
    success: bool = False
    final_output: Any = None
    
    # Phase artifacts
    plan: Optional[ExecutionPlan] = None
    execution_result: Optional[ExecutionResult] = None
    validation_report: Optional[ValidationReport] = None
    
    # Iterations
    iteration_count: int = 0
    max_iterations: int = 3
    
    # Timing
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    
    # Reflection
    reflection_log: List[ReflectionEntry] = field(default_factory=list)
    
    @property
    def duration_ms(self) -> int:
        if not self.completed_at:
            return 0
        return int((self.completed_at - self.started_at).total_seconds() * 1000)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "objective": self.objective,
            "success": self.success,
            "final_output": self.final_output,
            "plan": self.plan.to_dict() if self.plan else None,
            "execution_result": self.execution_result.to_dict() if self.execution_result else None,
            "validation_report": self.validation_report.to_dict() if self.validation_report else None,
            "iteration_count": self.iteration_count,
            "duration_ms": self.duration_ms,
            "reflection_log": [r.to_dict() for r in self.reflection_log],
        }


class ReflectiveAgentLoop:
    """
    Self-validating agent execution loop.
    
    Phases:
    1. PLAN - Create execution plan with success criteria
    2. EXECUTE - Execute the plan
    3. VALIDATE - Validate output against criteria
    4. REFINE - If validation fails, refine and retry
    
    Features:
    - Automatic retry with refinement
    - Quality scoring
    - Reflection and learning
    - Configurable strictness
    """
    
    def __init__(
        self,
        llm_router=None,
        max_iterations: int = 3,
        min_quality_score: float = 0.7,
        strict_validation: bool = False,
    ):
        """
        Initialize the loop.
        
        Args:
            llm_router: Router for LLM calls
            max_iterations: Maximum refinement iterations
            min_quality_score: Minimum quality score to pass
            strict_validation: If True, all criteria must pass
        """
        self.llm_router = llm_router
        self.max_iterations = max_iterations
        self.min_quality_score = min_quality_score
        self.strict_validation = strict_validation
        
        # Custom handlers
        self._planner: Optional[Callable] = None
        self._executor: Optional[Callable] = None
        self._validator: Optional[Callable] = None
        self._refiner: Optional[Callable] = None
    
    def set_planner(self, planner: Callable[[str, Dict[str, Any]], ExecutionPlan]) -> None:
        """Set custom planner function."""
        self._planner = planner
    
    def set_executor(self, executor: Callable[[ExecutionPlan], ExecutionResult]) -> None:
        """Set custom executor function."""
        self._executor = executor
    
    def set_validator(self, validator: Callable[[ExecutionResult, ExecutionPlan], ValidationReport]) -> None:
        """Set custom validator function."""
        self._validator = validator
    
    def set_refiner(self, refiner: Callable[[ValidationReport, ExecutionPlan], RefinementAction]) -> None:
        """Set custom refiner function."""
        self._refiner = refiner
    
    async def run(
        self,
        objective: str,
        context: Dict[str, Any] = None,
        success_criteria: List[str] = None,
    ) -> ReflectiveLoopResult:
        """
        Run the reflective loop.
        
        Args:
            objective: What to accomplish
            context: Additional context
            success_criteria: Criteria for success
            
        Returns:
            Complete loop result
        """
        result = ReflectiveLoopResult(
            objective=objective,
            max_iterations=self.max_iterations,
        )
        
        context = context or {}
        success_criteria = success_criteria or []
        
        logger.info(f"Starting reflective loop for: {objective[:50]}...")
        
        try:
            # PLAN phase
            plan = await self._plan_phase(objective, context, success_criteria)
            result.plan = plan
            result.reflection_log.append(ReflectionEntry(
                phase=AgentPhase.PLAN,
                input_summary=objective[:100],
                output_summary=f"{len(plan.steps)} steps planned",
                decisions_made=[f"Created plan with {len(plan.steps)} steps"],
            ))
            
            # Iterate EXECUTE -> VALIDATE -> REFINE
            current_plan = plan
            for iteration in range(self.max_iterations):
                result.iteration_count = iteration + 1
                
                # EXECUTE phase
                exec_result = await self._execute_phase(current_plan, context)
                result.execution_result = exec_result
                result.reflection_log.append(ReflectionEntry(
                    phase=AgentPhase.EXECUTE,
                    input_summary=f"Plan: {current_plan.objective}",
                    output_summary=f"Success: {exec_result.success}, Steps: {exec_result.steps_completed}",
                ))
                
                if not exec_result.success:
                    logger.warning(f"Execution failed: {exec_result.error}")
                    # Try to refine and retry
                    if iteration < self.max_iterations - 1:
                        current_plan = await self._refine_plan_after_failure(
                            current_plan, exec_result.error
                        )
                        continue
                    else:
                        result.success = False
                        break
                
                # VALIDATE phase
                validation = await self._validate_phase(exec_result, current_plan)
                result.validation_report = validation
                result.reflection_log.append(ReflectionEntry(
                    phase=AgentPhase.VALIDATE,
                    input_summary=f"Execution result",
                    output_summary=f"Result: {validation.result.value}, Score: {validation.quality_score:.2f}",
                    lessons=validation.suggestions[:3],
                ))
                
                # Check if validation passed
                if validation.passed or (
                    not self.strict_validation and 
                    validation.quality_score >= self.min_quality_score
                ):
                    result.success = True
                    result.final_output = exec_result.output
                    break
                
                # REFINE phase
                if iteration < self.max_iterations - 1:
                    refinement = await self._refine_phase(validation, current_plan)
                    result.reflection_log.append(ReflectionEntry(
                        phase=AgentPhase.REFINE,
                        input_summary=f"Issues: {len(validation.issues)}",
                        output_summary=f"Action: {refinement.action_type}",
                        decisions_made=[refinement.description],
                    ))
                    
                    # Update plan based on refinement
                    current_plan = await self._apply_refinement(
                        current_plan, refinement, context
                    )
                else:
                    # Last iteration failed validation
                    result.success = False
                    result.final_output = exec_result.output
            
            result.completed_at = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"Reflective loop failed: {e}")
            result.success = False
            result.completed_at = datetime.utcnow()
            result.reflection_log.append(ReflectionEntry(
                phase=AgentPhase.FAILED,
                input_summary="Exception occurred",
                output_summary=str(e)[:100],
            ))
        
        logger.info(
            f"Reflective loop complete: success={result.success}, "
            f"iterations={result.iteration_count}, "
            f"duration={result.duration_ms}ms"
        )
        
        return result
    
    # Phase implementations
    
    async def _plan_phase(
        self,
        objective: str,
        context: Dict[str, Any],
        success_criteria: List[str],
    ) -> ExecutionPlan:
        """Create execution plan."""
        if self._planner:
            return await self._planner(objective, context)
        
        # Default planning with LLM
        if self.llm_router:
            try:
                prompt = f"""Create an execution plan for the following objective:

Objective: {objective}

Context: {str(context)[:500]}

Provide:
1. A list of concrete steps to accomplish this
2. Success criteria to validate the result
3. Any constraints or considerations

Format as a structured plan."""

                result = await self.llm_router.route(
                    model="llama3.2",
                    messages=[{"role": "user", "content": prompt}],
                )
                
                # Parse result into plan (simplified)
                plan = ExecutionPlan(
                    objective=objective,
                    steps=["Execute objective based on context"],
                    success_criteria=success_criteria or ["Objective completed"],
                )
                
                return plan
                
            except Exception as e:
                logger.warning(f"LLM planning failed: {e}")
        
        # Fallback plan
        return ExecutionPlan(
            objective=objective,
            steps=["Execute the objective"],
            success_criteria=success_criteria or ["Objective completed successfully"],
        )
    
    async def _execute_phase(
        self,
        plan: ExecutionPlan,
        context: Dict[str, Any],
    ) -> ExecutionResult:
        """Execute the plan."""
        start_time = datetime.utcnow()
        
        if self._executor:
            return await self._executor(plan)
        
        # Default execution with LLM
        if self.llm_router:
            try:
                prompt = f"""Execute the following plan:

Objective: {plan.objective}

Steps:
{chr(10).join(f'- {step}' for step in plan.steps)}

Context: {str(context)[:500]}

Provide the result of executing this plan."""

                result = await self.llm_router.route(
                    model="llama3.2",
                    messages=[{"role": "user", "content": prompt}],
                )
                
                duration = int((datetime.utcnow() - start_time).total_seconds() * 1000)
                
                return ExecutionResult(
                    plan_id=plan.id,
                    success=True,
                    output=result,
                    steps_completed=len(plan.steps),
                    duration_ms=duration,
                )
                
            except Exception as e:
                duration = int((datetime.utcnow() - start_time).total_seconds() * 1000)
                return ExecutionResult(
                    plan_id=plan.id,
                    success=False,
                    error=str(e),
                    duration_ms=duration,
                )
        
        # Fallback - simulate execution
        duration = int((datetime.utcnow() - start_time).total_seconds() * 1000)
        return ExecutionResult(
            plan_id=plan.id,
            success=True,
            output=f"Executed: {plan.objective}",
            steps_completed=len(plan.steps),
            duration_ms=duration,
        )
    
    async def _validate_phase(
        self,
        execution: ExecutionResult,
        plan: ExecutionPlan,
    ) -> ValidationReport:
        """Validate execution result."""
        if self._validator:
            return await self._validator(execution, plan)
        
        # Default validation
        criteria_results = {}
        issues = []
        
        for criterion in plan.success_criteria:
            # Simple check - would use LLM for real validation
            passed = execution.success and execution.output is not None
            criteria_results[criterion] = passed
            if not passed:
                issues.append(f"Criterion not met: {criterion}")
        
        # Calculate quality score
        if criteria_results:
            pass_rate = sum(1 for v in criteria_results.values() if v) / len(criteria_results)
        else:
            pass_rate = 1.0 if execution.success else 0.0
        
        # Determine result
        if all(criteria_results.values()):
            result = ValidationResult.PASS
        elif any(criteria_results.values()):
            result = ValidationResult.PARTIAL
        else:
            result = ValidationResult.FAIL
        
        return ValidationReport(
            execution_id=execution.id,
            result=result,
            quality_score=pass_rate,
            criteria_results=criteria_results,
            issues=issues,
            suggestions=["Review and refine approach"] if issues else [],
        )
    
    async def _refine_phase(
        self,
        validation: ValidationReport,
        plan: ExecutionPlan,
    ) -> RefinementAction:
        """Determine refinement action."""
        if self._refiner:
            return await self._refiner(validation, plan)
        
        # Default refinement logic
        if validation.quality_score < 0.3:
            return RefinementAction(
                action_type="simplify",
                description="Simplify the approach and try again",
                parameters={"reduce_scope": True},
            )
        elif validation.quality_score < 0.7:
            return RefinementAction(
                action_type="modify",
                description="Modify approach based on feedback",
                parameters={"issues": validation.issues},
            )
        else:
            return RefinementAction(
                action_type="retry",
                description="Minor issues, retry with same approach",
                parameters={},
            )
    
    async def _apply_refinement(
        self,
        plan: ExecutionPlan,
        refinement: RefinementAction,
        context: Dict[str, Any],
    ) -> ExecutionPlan:
        """Apply refinement to create new plan."""
        if refinement.action_type == "simplify":
            # Reduce number of steps
            new_steps = plan.steps[:max(1, len(plan.steps) // 2)]
            return ExecutionPlan(
                objective=plan.objective,
                steps=new_steps,
                success_criteria=plan.success_criteria,
                constraints=plan.constraints + ["Simplified approach"],
            )
        
        elif refinement.action_type == "expand":
            # Add more steps
            return ExecutionPlan(
                objective=plan.objective,
                steps=plan.steps + ["Additional verification step"],
                success_criteria=plan.success_criteria,
                constraints=plan.constraints,
            )
        
        else:
            # Modify - keep same structure
            return ExecutionPlan(
                objective=plan.objective,
                steps=plan.steps,
                success_criteria=plan.success_criteria,
                constraints=plan.constraints + [f"Refined due to: {refinement.description}"],
            )
    
    async def _refine_plan_after_failure(
        self,
        plan: ExecutionPlan,
        error: str,
    ) -> ExecutionPlan:
        """Refine plan after execution failure."""
        return ExecutionPlan(
            objective=plan.objective,
            steps=plan.steps,
            success_criteria=plan.success_criteria,
            constraints=plan.constraints + [f"Previous attempt failed: {error[:50]}"],
        )


# Global instance
_reflective_loop: Optional[ReflectiveAgentLoop] = None


def get_reflective_agent_loop() -> ReflectiveAgentLoop:
    """Get or create the global reflective agent loop."""
    global _reflective_loop
    if _reflective_loop is None:
        _reflective_loop = ReflectiveAgentLoop()
    return _reflective_loop
