"""
ReAct Loop for Kautilya.

Implements the Reason-Act-Observe-Reflect pattern:
- Reason: Analyze the task and decide what to do
- Act: Execute the chosen action (skill, tool, or LLM call)
- Observe: Process the result
- Reflect: Evaluate success and decide next steps
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ActionType(Enum):
    """Types of actions the agent can take."""

    THINK = "think"  # Internal reasoning
    SKILL = "skill"  # Execute a skill
    TOOL = "tool"  # Use a tool
    MCP = "mcp"  # Call MCP server
    LLM = "llm"  # Direct LLM call
    RESPOND = "respond"  # Final response to user
    ASK_USER = "ask_user"  # Request clarification


class LoopStatus(Enum):
    """Status of the ReAct loop."""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    WAITING_USER = "waiting_user"
    MAX_ITERATIONS = "max_iterations"


@dataclass
class ThoughtAction:
    """A thought-action pair in the ReAct loop."""

    thought: str  # What the agent is thinking
    action_type: ActionType
    action_name: Optional[str] = None  # Name of skill/tool/etc
    action_input: Dict[str, Any] = field(default_factory=dict)
    observation: Optional[str] = None  # Result of action
    reflection: Optional[str] = None  # Evaluation of result
    success: bool = False
    duration: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_prompt_format(self) -> str:
        """Format as prompt text."""
        lines = [f"Thought: {self.thought}"]
        lines.append(f"Action: {self.action_type.value}")
        if self.action_name:
            lines.append(f"Action Input: {self.action_name}")
            if self.action_input:
                lines.append(f"Parameters: {self.action_input}")
        if self.observation:
            lines.append(f"Observation: {self.observation}")
        if self.reflection:
            lines.append(f"Reflection: {self.reflection}")
        return "\n".join(lines)


@dataclass
class LoopResult:
    """Result of a ReAct loop execution."""

    status: LoopStatus
    final_answer: Optional[str] = None
    steps: List[ThoughtAction] = field(default_factory=list)
    total_duration: float = 0.0
    iterations: int = 0
    error: Optional[str] = None

    def get_trace(self) -> str:
        """Get a human-readable trace of the loop."""
        lines = [f"ReAct Loop Trace ({self.status.value})"]
        lines.append(f"Iterations: {self.iterations}")
        lines.append(f"Duration: {self.total_duration:.2f}s")
        lines.append("-" * 40)

        for i, step in enumerate(self.steps, 1):
            lines.append(f"\n[Step {i}]")
            lines.append(step.to_prompt_format())

        if self.final_answer:
            lines.append(f"\n[Final Answer]\n{self.final_answer}")
        if self.error:
            lines.append(f"\n[Error]\n{self.error}")

        return "\n".join(lines)


class ReActLoop:
    """
    ReAct (Reasoning + Acting) Loop implementation.

    Implements the iterative reasoning pattern:
    1. Reason about the current state and task
    2. Choose an action (skill, tool, or direct response)
    3. Execute the action
    4. Observe the result
    5. Reflect on whether the task is complete
    6. Repeat or respond
    """

    REACT_SYSTEM_PROMPT = """You are an intelligent agent using the ReAct (Reasoning and Acting) pattern.

For each step, you will:
1. THINK: Analyze the current situation and decide what to do next
2. ACT: Choose and execute an action
3. OBSERVE: Note the result
4. REFLECT: Evaluate if you've achieved the goal

Available action types:
- skill: Execute a registered skill (e.g., document_qa, deep_research)
- tool: Use a tool (e.g., file operations, web search)
- respond: Provide final answer to the user
- ask_user: Request clarification from the user

## CRITICAL SKILL SELECTION GUIDELINES

When selecting a skill, CAREFULLY read the "WHEN TO USE" and "WHEN NOT TO USE" sections for each skill:

### Document Processing Tasks:
- For PDF, DOCX, XLSX, PPTX files → Use `document_qa` (has RAG pipeline, semantic search, page citations)
- For extracting, analyzing, summarizing document content → Use `document_qa`
- NEVER use `file_read` for documents - it returns raw binary/text without semantic understanding

### Research Tasks:
- For web research, competitor analysis, market trends → Use `deep_research`
- For comparing document content with external sources → Use BOTH `document_qa` + `deep_research`

### File Operations:
- For reading source code, config files (.py, .yaml, .json) → Use `file_read`
- For saving/exporting results to files → Use `file_write`

### Multi-Step Tasks:
If the task requires multiple skills (e.g., "extract from PDF, research competitors, save to CSV"):
1. Execute skills in logical order
2. Pass outputs between steps
3. Only use `respond` when ALL steps are complete

When you have enough information to answer, use action_type: "respond" with your final answer.

Always think step by step. If something fails, try an alternative approach."""

    REACT_STEP_PROMPT = """Current task: {task}

Previous steps:
{history}

Available capabilities:
{capabilities}

Based on the above, determine your next action.

Respond in this exact format:
THOUGHT: <your reasoning about what to do next>
ACTION_TYPE: <skill|tool|respond|ask_user>
ACTION_NAME: <name of skill/tool if applicable, or empty for respond/ask_user>
ACTION_INPUT: <JSON input for the action>

If you're ready to give the final answer, use:
ACTION_TYPE: respond
ACTION_INPUT: {{"answer": "your final answer here"}}"""

    def __init__(
        self,
        llm_client: Any,
        capability_registry: Optional[Any] = None,
        error_recovery: Optional[Any] = None,
        session_memory: Optional[Any] = None,
        max_iterations: int = 10,
        verbose: bool = False,
    ):
        """
        Initialize ReAct loop.

        Args:
            llm_client: LLM client for reasoning
            capability_registry: Registry of capabilities
            error_recovery: Error recovery engine
            session_memory: Session memory for context
            max_iterations: Maximum loop iterations
            verbose: Enable verbose logging
        """
        self.llm_client = llm_client
        self.capability_registry = capability_registry
        self.error_recovery = error_recovery
        self.session_memory = session_memory
        self.max_iterations = max_iterations
        self.verbose = verbose

    async def run(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> LoopResult:
        """
        Run the ReAct loop for a task.

        Args:
            task: The task to complete
            context: Additional context

        Returns:
            LoopResult with final answer and trace
        """
        start_time = time.time()
        steps: List[ThoughtAction] = []
        context = context or {}

        # Build available capabilities description (task-aware pre-filtering)
        capabilities_desc = await self._get_capabilities_description(task=task)

        for iteration in range(self.max_iterations):
            logger.info(f"ReAct iteration {iteration + 1}/{self.max_iterations}")

            # Build history from previous steps
            history = self._format_history(steps)

            # Generate next thought/action
            step_prompt = self.REACT_STEP_PROMPT.format(
                task=task,
                history=history or "No previous steps",
                capabilities=capabilities_desc,
            )

            try:
                # Get LLM decision
                response = await self.llm_client.chat(
                    step_prompt,
                    system_prompt=self.REACT_SYSTEM_PROMPT,
                )

                # Parse the response
                thought_action = self._parse_response(response)
                step_start = time.time()

                if self.verbose:
                    logger.info(f"Thought: {thought_action.thought}")
                    logger.info(f"Action: {thought_action.action_type.value} - {thought_action.action_name}")

                # Check if ready to respond
                if thought_action.action_type == ActionType.RESPOND:
                    answer = thought_action.action_input.get("answer", response)
                    thought_action.success = True
                    thought_action.duration = time.time() - step_start
                    steps.append(thought_action)

                    return LoopResult(
                        status=LoopStatus.COMPLETED,
                        final_answer=answer,
                        steps=steps,
                        total_duration=time.time() - start_time,
                        iterations=iteration + 1,
                    )

                # Check if asking user
                if thought_action.action_type == ActionType.ASK_USER:
                    question = thought_action.action_input.get("question", "Could you clarify?")
                    thought_action.observation = f"Waiting for user input: {question}"
                    thought_action.duration = time.time() - step_start
                    steps.append(thought_action)

                    return LoopResult(
                        status=LoopStatus.WAITING_USER,
                        final_answer=question,
                        steps=steps,
                        total_duration=time.time() - start_time,
                        iterations=iteration + 1,
                    )

                # Execute the action
                observation, success = await self._execute_action(thought_action)
                thought_action.observation = observation
                thought_action.success = success
                thought_action.duration = time.time() - step_start

                # Reflect on the result
                thought_action.reflection = await self._reflect(
                    task, thought_action, steps
                )

                steps.append(thought_action)

                # Record in session memory
                if self.session_memory:
                    self.session_memory.record_capability_usage(
                        thought_action.action_name or "unknown",
                        success,
                        thought_action.duration,
                        None if success else observation,
                    )

            except Exception as e:
                logger.error(f"ReAct iteration failed: {e}")

                # Try error recovery
                if self.error_recovery:
                    analysis = self.error_recovery.analyze_error(e, task, context)
                    if analysis.is_recoverable:
                        best_action = analysis.get_best_action()
                        if best_action:
                            logger.info(f"Attempting recovery: {best_action}")
                            # Add error step and continue
                            error_step = ThoughtAction(
                                thought=f"Error occurred: {str(e)}. Attempting recovery.",
                                action_type=ActionType.THINK,
                                observation=f"Recovery strategy: {best_action.description}",
                            )
                            steps.append(error_step)
                            continue

                # Cannot recover
                return LoopResult(
                    status=LoopStatus.FAILED,
                    steps=steps,
                    total_duration=time.time() - start_time,
                    iterations=iteration + 1,
                    error=str(e),
                )

        # Max iterations reached
        return LoopResult(
            status=LoopStatus.MAX_ITERATIONS,
            final_answer=self._synthesize_best_answer(steps),
            steps=steps,
            total_duration=time.time() - start_time,
            iterations=self.max_iterations,
        )

    async def _get_capabilities_description(self, task: str = "") -> str:
        """
        Get description of relevant capabilities for a task.

        Uses two-stage skill selection:
        1. Stage 1: Fast pre-filter using Tier 1 metadata (intents, file types)
        2. Stage 2: Format top candidates with full Tier 2 metadata

        Args:
            task: The user's task description for relevance filtering

        Returns:
            Formatted capability descriptions for LLM context
        """
        if not self.capability_registry:
            return "No specific capabilities registered. Use general reasoning."

        # Use two-stage skill selection if task is provided
        if task:
            # Stage 1: Pre-filter using lightweight Tier 1 metadata
            relevant_caps = self.capability_registry.get_relevant_capabilities(
                query=task,
                max_results=5,  # Limit to top 5 most relevant
            )

            if relevant_caps:
                # Stage 2: Format with full Tier 2 metadata for LLM decision
                return self.capability_registry.format_selected_capabilities_for_prompt(
                    capabilities=relevant_caps,
                    include_when_not_to_use=True,
                )

        # Fallback: Get all capabilities (limited)
        capabilities = self.capability_registry.get_all()
        lines = []
        for cap in capabilities[:10]:  # Limit to top 10
            lines.append(f"- {cap.name}: {cap.description[:100]}")
            if cap.when_to_use:
                lines.append(f"  When to use: {cap.when_to_use[:80]}...")

        return "\n".join(lines) if lines else "No capabilities available"

    def _format_history(self, steps: List[ThoughtAction]) -> str:
        """Format previous steps as history."""
        if not steps:
            return ""

        lines = []
        for i, step in enumerate(steps[-5:], 1):  # Last 5 steps
            lines.append(f"Step {i}:")
            lines.append(f"  Thought: {step.thought[:100]}")
            lines.append(f"  Action: {step.action_type.value}")
            if step.observation:
                lines.append(f"  Observation: {step.observation[:200]}")
            if step.reflection:
                lines.append(f"  Reflection: {step.reflection[:100]}")

        return "\n".join(lines)

    def _parse_response(self, response: str) -> ThoughtAction:
        """Parse LLM response into ThoughtAction."""
        import json
        import re

        thought = ""
        action_type = ActionType.THINK
        action_name = None
        action_input: Dict[str, Any] = {}

        # Extract thought
        thought_match = re.search(r"THOUGHT:\s*(.+?)(?=ACTION_TYPE:|$)", response, re.DOTALL | re.IGNORECASE)
        if thought_match:
            thought = thought_match.group(1).strip()

        # Extract action type
        type_match = re.search(r"ACTION_TYPE:\s*(\w+)", response, re.IGNORECASE)
        if type_match:
            type_str = type_match.group(1).lower()
            type_map = {
                "skill": ActionType.SKILL,
                "tool": ActionType.TOOL,
                "respond": ActionType.RESPOND,
                "ask_user": ActionType.ASK_USER,
                "think": ActionType.THINK,
                "llm": ActionType.LLM,
                "mcp": ActionType.MCP,
            }
            action_type = type_map.get(type_str, ActionType.THINK)

        # Extract action name
        name_match = re.search(r"ACTION_NAME:\s*(\S+)", response, re.IGNORECASE)
        if name_match:
            action_name = name_match.group(1).strip()

        # Extract action input (JSON)
        input_match = re.search(r"ACTION_INPUT:\s*({[\s\S]*?})", response, re.IGNORECASE)
        if input_match:
            try:
                action_input = json.loads(input_match.group(1))
            except json.JSONDecodeError:
                # Try to extract key-value pairs
                action_input = {"raw": input_match.group(1)}

        return ThoughtAction(
            thought=thought or "Processing...",
            action_type=action_type,
            action_name=action_name,
            action_input=action_input,
        )

    async def _execute_action(self, action: ThoughtAction) -> Tuple[str, bool]:
        """
        Execute an action and return observation.

        Args:
            action: The action to execute

        Returns:
            Tuple of (observation, success)
        """
        if action.action_type == ActionType.SKILL:
            return await self._execute_skill(action)
        elif action.action_type == ActionType.TOOL:
            return await self._execute_tool(action)
        elif action.action_type == ActionType.LLM:
            return await self._execute_llm(action)
        elif action.action_type == ActionType.THINK:
            return "Thought processed", True
        else:
            return f"Unknown action type: {action.action_type}", False

    async def _execute_skill(self, action: ThoughtAction) -> Tuple[str, bool]:
        """Execute a skill."""
        if not self.capability_registry:
            return "No capability registry available", False

        skill_name = action.action_name
        if not skill_name:
            return "No skill name specified", False

        try:
            result = await self.capability_registry.execute_skill(
                skill_name, action.action_input
            )
            return str(result)[:1000], True
        except Exception as e:
            return f"Skill execution failed: {str(e)}", False

    async def _execute_tool(self, action: ThoughtAction) -> Tuple[str, bool]:
        """Execute a tool."""
        tool_name = action.action_name
        if not tool_name:
            return "No tool name specified", False

        # Handle built-in tools
        if tool_name == "file_resolver":
            # Use FileResolver
            from .file_resolver import FileResolver
            resolver = FileResolver()
            try:
                ref = action.action_input.get("reference", "")
                match = resolver.resolve(ref)
                return f"Resolved: {match.path} ({match.match_type})", True
            except FileNotFoundError as e:
                return str(e), False

        # Try capability registry for tools
        if self.capability_registry:
            try:
                result = await self.capability_registry.execute_skill(
                    tool_name, action.action_input
                )
                return str(result)[:1000], True
            except Exception as e:
                return f"Tool execution failed: {str(e)}", False

        return f"Unknown tool: {tool_name}", False

    async def _execute_llm(self, action: ThoughtAction) -> Tuple[str, bool]:
        """Execute a direct LLM call."""
        prompt = action.action_input.get("prompt", str(action.action_input))
        try:
            response = await self.llm_client.chat(prompt)
            return response[:1000], True
        except Exception as e:
            return f"LLM call failed: {str(e)}", False

    async def _reflect(
        self,
        task: str,
        action: ThoughtAction,
        previous_steps: List[ThoughtAction],
    ) -> str:
        """Generate reflection on the action result."""
        if not action.observation:
            return "No observation to reflect on"

        reflection_prompt = f"""Task: {task}

Action taken: {action.action_type.value} - {action.action_name}
Observation: {action.observation[:500]}

Briefly evaluate:
1. Did this action make progress toward the goal?
2. What should be done next?

Keep reflection to 1-2 sentences."""

        try:
            response = await self.llm_client.chat(reflection_prompt)
            return response[:200]
        except Exception:
            return "Proceeding to next step" if action.success else "Need to try different approach"

    def _synthesize_best_answer(self, steps: List[ThoughtAction]) -> str:
        """Synthesize best possible answer from steps taken."""
        if not steps:
            return "Unable to complete the task."

        # Look for successful observations
        successful_obs = [
            step.observation
            for step in steps
            if step.success and step.observation
        ]

        if successful_obs:
            return f"Based on my analysis: {successful_obs[-1]}"

        # Return last thought
        return f"I attempted to help but encountered issues. Last thought: {steps[-1].thought}"
