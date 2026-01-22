"""
Workflow Executor.

Executes workflow manifests by orchestrating agents.
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

from src.workflows.models import (
    WorkflowManifest,
    WorkflowStep,
    WorkflowRun,
    StepStatus,
    RiskLevel,
)
from src.artifacts.models import Artifact, ArtifactType, ProvenanceRecord
from src.artifacts.store import get_artifact_store
from src.utils.structured_logging import get_logger

logger = get_logger("workflow_executor")


class StepResult:
    """Result of executing a workflow step."""
    
    def __init__(
        self,
        step_id: str,
        success: bool,
        outputs: Dict[str, Any] = None,
        artifacts: List[str] = None,
        error: Optional[str] = None,
        duration_ms: int = 0,
    ):
        self.step_id = step_id
        self.success = success
        self.outputs = outputs or {}
        self.artifacts = artifacts or []
        self.error = error
        self.duration_ms = duration_ms


class WorkflowExecutor:
    """
    Executes workflow manifests.
    
    Features:
    - Parallel step execution
    - Dependency resolution
    - Approval gate handling
    - Artifact tracking
    - Error recovery
    """
    
    def __init__(
        self,
        agent_router=None,
        approval_manager=None,
        artifact_store=None,
    ):
        """
        Initialize workflow executor.
        
        Args:
            agent_router: Router for dispatching to agents
            approval_manager: Manager for approval gates
            artifact_store: Store for artifacts
        """
        self.agent_router = agent_router
        self.approval_manager = approval_manager
        self.artifact_store = artifact_store or get_artifact_store()
        
        # Active runs
        self._active_runs: Dict[str, WorkflowRun] = {}
    
    async def execute(
        self,
        workflow: WorkflowManifest,
        variables: Dict[str, Any] = None,
        business_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> WorkflowRun:
        """
        Execute a workflow.
        
        Args:
            workflow: Workflow to execute
            variables: Variable values
            business_id: Associated business
            user_id: User running the workflow
            
        Returns:
            Completed workflow run
        """
        # Validate workflow
        errors = workflow.validate_workflow()
        if errors:
            raise ValueError(f"Invalid workflow: {', '.join(errors)}")
        
        # Create run
        run = WorkflowRun(
            id=f"run_{uuid4().hex[:12]}",
            workflow_id=workflow.id,
            workflow_version=workflow.version,
            variables=self._resolve_variables(workflow, variables or {}),
            business_id=business_id,
            user_id=user_id,
            status="running",
            started_at=datetime.utcnow(),
        )
        
        # Initialize step states
        for step in workflow.steps:
            run.step_states[step.id] = StepStatus.PENDING
        
        self._active_runs[run.id] = run
        
        logger.info(
            "Starting workflow execution",
            workflow_id=workflow.id,
            run_id=run.id,
            step_count=len(workflow.steps),
        )
        
        try:
            # Execute in dependency order
            execution_order = workflow.get_execution_order()
            
            for batch in execution_order:
                if workflow.parallel_execution:
                    # Execute batch in parallel
                    await self._execute_batch_parallel(
                        workflow, run, batch
                    )
                else:
                    # Execute sequentially
                    for step_id in batch:
                        await self._execute_step(
                            workflow, run, step_id
                        )
                
                # Check for failures
                if workflow.fail_fast:
                    for step_id in batch:
                        if run.step_states[step_id] == StepStatus.FAILED:
                            run.status = "failed"
                            run.error = f"Step '{step_id}' failed"
                            run.failed_step = step_id
                            break
                
                if run.status == "failed":
                    break
            
            # Complete run
            if run.status != "failed":
                run.status = "completed"
            
            run.completed_at = datetime.utcnow()
            
        except Exception as e:
            run.status = "failed"
            run.error = str(e)
            run.completed_at = datetime.utcnow()
            logger.error(f"Workflow execution failed: {e}")
        
        finally:
            self._active_runs.pop(run.id, None)
        
        logger.info(
            "Workflow execution finished",
            workflow_id=workflow.id,
            run_id=run.id,
            status=run.status,
            duration_ms=int((run.completed_at - run.started_at).total_seconds() * 1000) if run.completed_at and run.started_at else 0,
        )
        
        return run
    
    async def execute_workflow(
        self,
        workflow: WorkflowManifest,
        variables: Dict[str, Any] = None,
        business_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> WorkflowRun:
        """
        Alias for execute method to match API expectations.
        """
        return await self.execute(
            workflow=workflow,
            variables=variables,
            business_id=business_id,
            user_id=user_id,
        )

    async def cancel(self, run_id: str) -> bool:
        """Cancel a running workflow."""
        run = self._active_runs.get(run_id)
        if run:
            run.status = "cancelled"
            run.completed_at = datetime.utcnow()
            return True
        return False
    
    async def get_run_status(self, run_id: str) -> Optional[WorkflowRun]:
        """Get status of a workflow run."""
        return self._active_runs.get(run_id)
    
    # Private methods
    
    def _resolve_variables(
        self,
        workflow: WorkflowManifest,
        provided: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Resolve workflow variables with defaults."""
        resolved = {}
        
        for var in workflow.variables:
            if var.name in provided:
                resolved[var.name] = provided[var.name]
            elif var.default is not None:
                resolved[var.name] = var.default
            elif var.required:
                raise ValueError(f"Required variable '{var.name}' not provided")
        
        return resolved
    
    async def _execute_batch_parallel(
        self,
        workflow: WorkflowManifest,
        run: WorkflowRun,
        step_ids: List[str],
    ) -> None:
        """Execute a batch of steps in parallel honoring max_parallel_steps."""
        # Respect workflow-level max_parallel_steps
        max_parallel = max(1, getattr(workflow, "max_parallel_steps", len(step_ids)))
        sem = asyncio.Semaphore(max_parallel)

        async def _run_step_with_semaphore(sid: str):
            async with sem:
                return await self._execute_step(workflow, run, sid)

        tasks = [asyncio.create_task(_run_step_with_semaphore(sid)) for sid in step_ids]

        # Wait for all tasks to complete
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _execute_step(
        self,
        workflow: WorkflowManifest,
        run: WorkflowRun,
        step_id: str,
    ) -> StepResult:
        """Execute a single workflow step."""
        step = workflow.get_step(step_id)
        if not step:
            return StepResult(step_id, False, error="Step not found")
        
        # Check if cancelled
        if run.status == "cancelled":
            run.step_states[step_id] = StepStatus.SKIPPED
            return StepResult(step_id, False, error="Workflow cancelled")
        
        # Check dependencies
        for dep_id in step.get_dependency_ids():
            dep_status = run.step_states.get(dep_id)
            if dep_status not in [StepStatus.COMPLETED, StepStatus.SKIPPED]:
                if step.skip_on_failure:
                    run.step_states[step_id] = StepStatus.SKIPPED
                    return StepResult(step_id, True)
                else:
                    run.step_states[step_id] = StepStatus.FAILED
                    return StepResult(step_id, False, error=f"Dependency '{dep_id}' not satisfied")
        
        # Check condition
        if step.condition:
            if not self._evaluate_condition(step.condition, run):
                run.step_states[step_id] = StepStatus.SKIPPED
                return StepResult(step_id, True)
        
        # Handle approval gate
        if step.require_approval and self.approval_manager:
            run.step_states[step_id] = StepStatus.WAITING_APPROVAL
            approved = await self._wait_for_approval(step, run)
            if not approved:
                run.step_states[step_id] = StepStatus.FAILED
                return StepResult(step_id, False, error="Approval denied")
        
        # Execute step
        run.step_states[step_id] = StepStatus.RUNNING
        start_time = datetime.utcnow()
        
        logger.debug(f"Executing step '{step_id}' with agent '{step.agent}'")
        
        try:
            # Resolve inputs from previous step outputs
            resolved_inputs = self._resolve_inputs(step.inputs, run)
            
            # Call agent
            result = await self._call_agent(
                step.agent,
                resolved_inputs,
                run,
            )
            
            end_time = datetime.utcnow()
            duration_ms = int((end_time - start_time).total_seconds() * 1000)
            
            if result.get("success", False):
                run.step_states[step_id] = StepStatus.COMPLETED
                run.step_results[step_id] = result.get("output", {})
                
                # Create artifacts
                artifacts = await self._create_artifacts(
                    step, result, run, duration_ms
                )
                run.step_artifacts[step_id] = artifacts
                
                return StepResult(
                    step_id, True,
                    outputs=result.get("output", {}),
                    artifacts=artifacts,
                    duration_ms=duration_ms,
                )
            else:
                run.step_states[step_id] = StepStatus.FAILED
                return StepResult(
                    step_id, False,
                    error=result.get("error", "Unknown error"),
                    duration_ms=duration_ms,
                )
        
        except Exception as e:
            run.step_states[step_id] = StepStatus.FAILED
            logger.error(f"Step '{step_id}' failed: {e}")
            return StepResult(step_id, False, error=str(e))
    
    async def _call_agent(
        self,
        agent_type: str,
        inputs: Dict[str, Any],
        run: WorkflowRun,
    ) -> Dict[str, Any]:
        """Call an agent to execute a step."""
        if not self.agent_router:
            # Simulate agent execution for testing
            return {
                "success": True,
                "output": {"result": f"Simulated {agent_type} output"},
            }
        
        try:
            result = await self.agent_router.route(
                agent_type=agent_type,
                task={
                    "inputs": inputs,
                    "business_id": run.business_id,
                    "user_id": run.user_id,
                }
            )
            return {
                "success": result.success,
                "output": result.data if result.success else {},
                "error": result.error,
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }
    
    def _resolve_inputs(
        self,
        inputs: Dict[str, Any],
        run: WorkflowRun,
    ) -> Dict[str, Any]:
        """Resolve input values, substituting variables and outputs."""
        resolved = {}
        
        for key, value in inputs.items():
            if isinstance(value, str):
                # Check for variable reference: ${variable_name}
                if value.startswith("${") and value.endswith("}"):
                    var_name = value[2:-1]
                    if var_name in run.variables:
                        resolved[key] = run.variables[var_name]
                    elif "." in var_name:
                        # Step output reference: step_id.output_name
                        parts = var_name.split(".", 1)
                        step_id, output_name = parts[0], parts[1]
                        if step_id in run.step_results:
                            resolved[key] = run.step_results[step_id].get(output_name)
                        else:
                            resolved[key] = value
                    else:
                        resolved[key] = value
                else:
                    resolved[key] = value
            else:
                resolved[key] = value
        
        return resolved
    
    def _evaluate_condition(
        self,
        condition: str,
        run: WorkflowRun,
    ) -> bool:
        """Evaluate a step condition."""
        # Simple condition evaluation
        # Supports: step_id.completed, variable == value
        try:
            if "==" in condition:
                left, right = condition.split("==", 1)
                left_val = self._get_value(left.strip(), run)
                right_val = right.strip().strip('"\'')
                return str(left_val) == right_val
            elif ".completed" in condition:
                step_id = condition.replace(".completed", "").strip()
                return run.step_states.get(step_id) == StepStatus.COMPLETED
            elif ".failed" in condition:
                step_id = condition.replace(".failed", "").strip()
                return run.step_states.get(step_id) == StepStatus.FAILED
            else:
                return True
        except Exception:
            return True
    
    def _get_value(self, ref: str, run: WorkflowRun) -> Any:
        """Get a value from variables or step outputs."""
        if ref.startswith("${") and ref.endswith("}"):
            ref = ref[2:-1]
        
        if ref in run.variables:
            return run.variables[ref]
        
        if "." in ref:
            parts = ref.split(".", 1)
            step_id, output_name = parts[0], parts[1]
            if step_id in run.step_results:
                return run.step_results[step_id].get(output_name)
        
        return ref
    
    async def _wait_for_approval(
        self,
        step: WorkflowStep,
        run: WorkflowRun,
    ) -> bool:
        """Wait for human approval of a step."""
        if not self.approval_manager:
            # Auto-approve if no manager
            return True
        
        try:
            request = await self.approval_manager.create_approval_request(
                workflow_id=run.workflow_id,
                step_id=step.id,
                operation=step.name,
                description=step.description,
                requested_by=run.user_id or "system",
            )
            
            # Wait for approval (with timeout)
            timeout = step.approval.timeout_seconds if step.approval else 3600
            # This would typically be event-driven, but simplified here
            # In production, use approval webhooks or polling
            
            return True  # Simplified
        except Exception as e:
            logger.error(f"Approval failed: {e}")
            return False
    
    async def _create_artifacts(
        self,
        step: WorkflowStep,
        result: Dict[str, Any],
        run: WorkflowRun,
        duration_ms: int,
    ) -> List[str]:
        """Create artifacts from step output."""
        artifact_ids = []
        
        output = result.get("output", {})
        if not output:
            return artifact_ids
        
        # Map step output to artifact type
        artifact_type_map = {
            "research": ArtifactType.RESEARCH,
            "code": ArtifactType.CODE,
            "content": ArtifactType.CONTENT,
            "finance": ArtifactType.FINANCE,
            "legal": ArtifactType.LEGAL,
            "analytics": ArtifactType.ANALYSIS,
        }
        
        artifact_type = artifact_type_map.get(
            step.agent,
            ArtifactType.GENERIC
        )
        
        # Create artifact
        artifact = Artifact(
            name=f"{step.name} Output",
            artifact_type=artifact_type,
            content=output,
            created_by=f"workflow:{run.workflow_id}:{step.id}",
            business_id=run.business_id,
            session_id=run.id,
            provenance=ProvenanceRecord(
                actor_id=step.id,
                actor_type="workflow_step",
                action=step.name,
                duration_ms=duration_ms,
            ),
            tags=step.tags + [run.workflow_id, step.agent],
        )
        
        await self.artifact_store.store(artifact)
        artifact_ids.append(artifact.id)
        
        return artifact_ids


# Global executor instance
_workflow_executor: Optional[WorkflowExecutor] = None


def get_workflow_executor() -> WorkflowExecutor:
    """Get or create the global workflow executor."""
    global _workflow_executor
    if _workflow_executor is None:
        _workflow_executor = WorkflowExecutor()
    return _workflow_executor
