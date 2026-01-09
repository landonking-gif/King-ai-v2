"""
ReAct (Reason-Act-Think) Planning Engine.
Implements iterative planning with observation feedback.
"""

import json
from typing import List, Dict, Any, Tuple
from datetime import datetime

from src.master_ai.planning_models import (
    ExecutionPlan, PlanTask, PlanningContext, ReActStep,
    TaskStatus, RiskLevel, ReplanRequest
)
from src.master_ai.prompts import TASK_DECOMPOSITION_PROMPT
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
            context=context[:2000] if context else "",
            status="planning"
        )
        
        # Build planning context
        risk_profile = getattr(settings, "risk_profile", "moderate")
        if not isinstance(risk_profile, str):
            risk_profile = "moderate"

        planning_context = PlanningContext(
            goal=goal,
            user_input=goal,
            empire_state=context or "",
            risk_profile=risk_profile,
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
        """Parse JSON from LLM response with robust fallback extraction."""
        import re
        
        # Try to extract JSON from code blocks first
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            parts = response.split("```")
            if len(parts) > 1:
                response = parts[1].split("```")[0].strip()
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        
        # Try to find JSON object in response
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        # Try to extract task names from markdown-style lists
        tasks = []
        task_patterns = [
            r'\*\*(?:Task\s*\d*:?\s*)?([^*]+)\*\*[:\s]*([^\n*]+)?',  # **Task: Name**: description
            r'(?:^|\n)\d+\.\s*\*\*([^*]+)\*\*[:\s]*([^\n]+)?',  # 1. **Name**: description
            r'(?:^|\n)-\s*\*\*([^*]+)\*\*[:\s]*([^\n]+)?',  # - **Name**: description
        ]
        
        for pattern in task_patterns:
            matches = re.findall(pattern, response)
            for match in matches:
                name = match[0].strip() if isinstance(match, tuple) else match.strip()
                description = match[1].strip() if isinstance(match, tuple) and len(match) > 1 and match[1] else ""
                if name and len(name) > 2:
                    tasks.append({
                        "name": name[:100],
                        "description": description[:500] if description else f"Execute: {name}",
                        "agent": self._infer_agent_from_text(name + " " + description),
                        "risk_level": "low",
                        "duration_minutes": 5
                    })
        
        if tasks:
            logger.info(f"Extracted {len(tasks)} tasks from non-JSON response")
            return {"tasks": tasks}
        
        logger.warning("Failed to parse JSON response", response=response[:200])
        return {"tasks": []}
    
    def _infer_agent_from_text(self, text: str) -> str:
        """Infer the appropriate agent based on task text."""
        text_lower = text.lower()
        
        agent_keywords = {
            "research": ["research", "search", "find", "look up", "investigate", "analyze market", "competitor"],
            "code_generator": ["code", "implement", "develop", "program", "script", "api", "integrate"],
            "content": ["write", "create content", "blog", "article", "copy", "marketing material"],
            "commerce": ["shopify", "store", "e-commerce", "product", "catalog", "shop"],
            "finance": ["payment", "stripe", "invoice", "financial", "revenue", "cost"],
            "analytics": ["analytics", "metrics", "report", "dashboard", "data", "statistics"],
            "legal": ["contract", "legal", "terms", "policy", "compliance", "agreement"],
        }
        
        for agent, keywords in agent_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return agent
        
        return "research"  # Default to research agent

