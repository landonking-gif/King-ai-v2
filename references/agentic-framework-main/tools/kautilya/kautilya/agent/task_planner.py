"""
Task Planner for Kautilya.

Provides intelligent task decomposition and planning:
- Analyzes complex queries to identify subtasks
- Determines capability requirements for each step
- Creates execution plans with dependencies
- Handles multi-step workflows
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class StepStatus(Enum):
    """Status of a plan step."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class PlanStep:
    """A single step in an execution plan."""

    id: str
    description: str
    capability_type: str  # "skill", "tool", "mcp", "llm"
    capability_name: Optional[str] = None  # Specific capability to use
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)  # Step IDs this depends on
    status: StepStatus = StepStatus.PENDING
    error: Optional[str] = None
    retries: int = 0
    max_retries: int = 3

    def is_ready(self, completed_steps: Set[str]) -> bool:
        """Check if this step is ready to execute (all dependencies met)."""
        return all(dep in completed_steps for dep in self.dependencies)

    def can_retry(self) -> bool:
        """Check if this step can be retried."""
        return self.retries < self.max_retries


@dataclass
class ExecutionPlan:
    """An execution plan with multiple steps."""

    task_description: str
    steps: List[PlanStep] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    completed_steps: Set[str] = field(default_factory=set)
    failed_steps: Set[str] = field(default_factory=set)

    def add_step(self, step: PlanStep) -> None:
        """Add a step to the plan."""
        self.steps.append(step)

    def get_ready_steps(self) -> List[PlanStep]:
        """Get all steps that are ready to execute."""
        return [
            step
            for step in self.steps
            if step.status == StepStatus.PENDING and step.is_ready(self.completed_steps)
        ]

    def mark_completed(self, step_id: str, outputs: Dict[str, Any]) -> None:
        """Mark a step as completed."""
        for step in self.steps:
            if step.id == step_id:
                step.status = StepStatus.COMPLETED
                step.outputs = outputs
                self.completed_steps.add(step_id)
                # Store outputs in context for dependent steps
                self.context[step_id] = outputs
                break

    def mark_failed(self, step_id: str, error: str) -> None:
        """Mark a step as failed."""
        for step in self.steps:
            if step.id == step_id:
                step.status = StepStatus.FAILED
                step.error = error
                step.retries += 1
                if not step.can_retry():
                    self.failed_steps.add(step_id)
                break

    def is_complete(self) -> bool:
        """Check if the plan is complete (all steps done or failed)."""
        return all(
            step.status in (StepStatus.COMPLETED, StepStatus.FAILED, StepStatus.SKIPPED)
            for step in self.steps
        )

    def is_successful(self) -> bool:
        """Check if the plan completed successfully."""
        return self.is_complete() and len(self.failed_steps) == 0

    def get_final_output(self) -> Dict[str, Any]:
        """Get the final output from the last completed step."""
        for step in reversed(self.steps):
            if step.status == StepStatus.COMPLETED:
                return step.outputs
        return {}


class TaskPlanner:
    """
    Intelligent task planner for complex queries.

    Analyzes user queries and creates execution plans:
    - Identifies required capabilities (skills, tools, MCP servers)
    - Breaks complex tasks into subtasks
    - Determines step dependencies
    - Supports parallel execution where possible
    """

    # Task patterns and their typical decomposition
    TASK_PATTERNS = {
        "document_extraction": {
            "keywords": ["extract", "read", "parse", "document", "pdf", "file", "report"],
            "steps": ["resolve_file", "extract_content", "process_data"],
        },
        "research": {
            "keywords": ["research", "find", "search", "look up", "investigate"],
            "steps": ["search", "retrieve", "synthesize"],
        },
        "code_analysis": {
            "keywords": ["analyze", "review", "debug", "fix", "refactor", "code"],
            "steps": ["read_code", "analyze", "suggest_fixes"],
        },
        "qa": {
            "keywords": ["question", "answer", "what", "why", "how", "explain"],
            "steps": ["understand_query", "retrieve_context", "generate_answer"],
        },
        "summarization": {
            "keywords": ["summarize", "summary", "brief", "tldr", "overview"],
            "steps": ["retrieve_content", "analyze", "summarize"],
        },
        "data_processing": {
            "keywords": ["process", "transform", "convert", "calculate", "compute"],
            "steps": ["load_data", "process", "output_result"],
        },
    }

    def __init__(
        self,
        llm_client: Optional[Any] = None,
        capability_registry: Optional[Any] = None,
    ):
        """
        Initialize task planner.

        Args:
            llm_client: LLM client for complex planning (optional)
            capability_registry: Registry of available capabilities
        """
        self.llm_client = llm_client
        self.capability_registry = capability_registry

    def analyze_task(self, task: str) -> Dict[str, Any]:
        """
        Analyze a task to understand its requirements.

        Args:
            task: User's task description

        Returns:
            Analysis with task type, required capabilities, complexity
        """
        task_lower = task.lower()

        # Detect task type from patterns
        task_type = "general"
        matched_keywords = []

        for pattern_name, pattern_info in self.TASK_PATTERNS.items():
            keywords = pattern_info["keywords"]
            matches = [kw for kw in keywords if kw in task_lower]
            if len(matches) > len(matched_keywords):
                task_type = pattern_name
                matched_keywords = matches

        # Detect file references
        has_file_reference = "@" in task or any(
            ext in task_lower
            for ext in [".pdf", ".docx", ".xlsx", ".csv", ".txt", ".py", ".json"]
        )

        # Detect complexity
        complexity = "simple"
        if len(task.split()) > 20:
            complexity = "moderate"
        if len(task.split()) > 50 or "and" in task_lower or "then" in task_lower:
            complexity = "complex"

        # Identify required capability types
        required_capabilities = []
        if has_file_reference:
            required_capabilities.append("file_resolution")
        if task_type == "document_extraction":
            required_capabilities.append("document_processing")
        if task_type == "research":
            required_capabilities.append("web_search")
        if task_type == "code_analysis":
            required_capabilities.append("code_execution")

        return {
            "task_type": task_type,
            "matched_keywords": matched_keywords,
            "has_file_reference": has_file_reference,
            "complexity": complexity,
            "required_capabilities": required_capabilities,
            "original_task": task,
        }

    def create_plan(
        self,
        task: str,
        available_capabilities: Optional[List[Any]] = None,
    ) -> ExecutionPlan:
        """
        Create an execution plan for a task.

        Args:
            task: User's task description
            available_capabilities: List of available capabilities

        Returns:
            ExecutionPlan with steps to execute
        """
        analysis = self.analyze_task(task)
        plan = ExecutionPlan(task_description=task)

        # Step 1: File resolution (if needed)
        if analysis["has_file_reference"]:
            plan.add_step(
                PlanStep(
                    id="resolve_files",
                    description="Resolve file references to actual paths",
                    capability_type="tool",
                    capability_name="file_resolver",
                    inputs={"task": task},
                )
            )

        # Step 2: Capability selection
        plan.add_step(
            PlanStep(
                id="select_capability",
                description="Select the best capability for this task",
                capability_type="tool",
                capability_name="capability_matcher",
                inputs={"task": task, "analysis": analysis},
                dependencies=["resolve_files"] if analysis["has_file_reference"] else [],
            )
        )

        # Step 3: Execute primary capability
        plan.add_step(
            PlanStep(
                id="execute_capability",
                description=f"Execute the selected capability for: {analysis['task_type']}",
                capability_type="dynamic",  # Will be determined at runtime
                inputs={"task": task},
                dependencies=["select_capability"],
            )
        )

        # Step 4: Validate output
        plan.add_step(
            PlanStep(
                id="validate_output",
                description="Validate and format the output",
                capability_type="tool",
                capability_name="output_validator",
                dependencies=["execute_capability"],
            )
        )

        logger.info(f"Created plan with {len(plan.steps)} steps for task type: {analysis['task_type']}")
        return plan

    async def create_plan_with_llm(
        self,
        task: str,
        available_capabilities: List[Dict[str, Any]],
    ) -> ExecutionPlan:
        """
        Create an execution plan using LLM for complex tasks.

        Args:
            task: User's task description
            available_capabilities: List of available capabilities

        Returns:
            ExecutionPlan with steps to execute
        """
        if not self.llm_client:
            # Fall back to rule-based planning
            return self.create_plan(task, available_capabilities)

        # Format capabilities for LLM
        cap_descriptions = []
        for cap in available_capabilities:
            cap_descriptions.append(f"- {cap['name']}: {cap.get('description', 'No description')}")

        capabilities_text = "\n".join(cap_descriptions)

        planning_prompt = f"""You are a task planner. Analyze the following task and create an execution plan.

TASK: {task}

AVAILABLE CAPABILITIES:
{capabilities_text}

Create a step-by-step plan. For each step, specify:
1. Step description
2. Which capability to use (from the list above)
3. What inputs are needed
4. What outputs are expected

Respond in this JSON format:
{{
    "steps": [
        {{
            "id": "step_1",
            "description": "Description of what this step does",
            "capability": "capability_name",
            "inputs": {{}},
            "depends_on": []
        }}
    ]
}}
"""

        try:
            response = await self.llm_client.chat(planning_prompt)
            # Parse LLM response and create plan
            plan = self._parse_llm_plan(task, response)
            return plan
        except Exception as e:
            logger.warning(f"LLM planning failed, falling back to rule-based: {e}")
            return self.create_plan(task, available_capabilities)

    def _parse_llm_plan(self, task: str, llm_response: str) -> ExecutionPlan:
        """Parse LLM response into an ExecutionPlan."""
        import json
        import re

        plan = ExecutionPlan(task_description=task)

        # Try to extract JSON from response
        json_match = re.search(r"\{[\s\S]*\}", llm_response)
        if json_match:
            try:
                data = json.loads(json_match.group())
                steps = data.get("steps", [])

                for step_data in steps:
                    plan.add_step(
                        PlanStep(
                            id=step_data.get("id", f"step_{len(plan.steps)}"),
                            description=step_data.get("description", ""),
                            capability_type="skill",
                            capability_name=step_data.get("capability"),
                            inputs=step_data.get("inputs", {}),
                            dependencies=step_data.get("depends_on", []),
                        )
                    )

                return plan
            except json.JSONDecodeError:
                pass

        # If JSON parsing fails, create a simple plan
        return self.create_plan(task, None)

    def decompose_complex_task(self, task: str) -> List[str]:
        """
        Decompose a complex task into subtasks.

        Args:
            task: Complex task description

        Returns:
            List of subtask descriptions
        """
        subtasks = []

        # Split on common conjunctions
        parts = task.replace(" and ", "|").replace(" then ", "|").replace(", ", "|").split("|")

        for part in parts:
            part = part.strip()
            if part and len(part) > 5:  # Ignore very short fragments
                subtasks.append(part)

        # If no decomposition found, return original task
        if not subtasks:
            subtasks = [task]

        return subtasks

    def estimate_complexity(self, plan: ExecutionPlan) -> Dict[str, Any]:
        """
        Estimate the complexity and resource requirements of a plan.

        Args:
            plan: Execution plan to analyze

        Returns:
            Complexity metrics
        """
        total_steps = len(plan.steps)
        parallel_steps = 0
        sequential_chains = []

        # Analyze dependency graph
        deps_count = sum(len(step.dependencies) for step in plan.steps)

        # Find steps that can run in parallel (no dependencies on each other)
        for i, step1 in enumerate(plan.steps):
            for step2 in plan.steps[i + 1 :]:
                if step1.id not in step2.dependencies and step2.id not in step1.dependencies:
                    parallel_steps += 1

        return {
            "total_steps": total_steps,
            "dependency_count": deps_count,
            "potential_parallelism": parallel_steps,
            "estimated_complexity": "high" if total_steps > 5 else "medium" if total_steps > 2 else "low",
        }
