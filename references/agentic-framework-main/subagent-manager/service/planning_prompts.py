"""
Planning Prompts for Reflective Agent.

Module: subagent-manager/service/planning_prompts.py

Structured prompts for PLAN, VALIDATE, and REFINE phases.
"""

from typing import Any, Dict, List, Optional


class PlanningPrompts:
    """Prompts for reflective agent planning and refinement."""

    def build_planning_prompt(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
        constraints: Optional[List[str]] = None,
        previous_attempts: Optional[List[Any]] = None,
        available_tools: Optional[List[str]] = None,
    ) -> str:
        """
        Build prompt for planning phase.

        Args:
            task: Task description
            context: Additional context
            constraints: Constraints to follow
            previous_attempts: Previous planning attempts
            available_tools: Available tools/skills

        Returns:
            Planning prompt
        """
        prompt = f"""You are a planning agent. Create a detailed execution plan for the following task.

## Task
{task}

"""
        if context:
            prompt += f"""## Context
{self._format_dict(context)}

"""

        if constraints:
            prompt += f"""## Constraints
{self._format_list(constraints)}

"""

        if available_tools:
            prompt += f"""## Available Tools
{self._format_list(available_tools)}

"""

        if previous_attempts:
            prompt += f"""## Previous Attempts
The following plans were tried but did not fully succeed:
{self._format_previous_attempts(previous_attempts)}

Learn from these attempts and create an improved plan.

"""

        prompt += """## Instructions
Create a detailed execution plan with the following structure:

1. **Task Understanding**: Summarize what needs to be accomplished
2. **Approach**: Describe your high-level strategy
3. **Steps**: List specific actions to take (use available tools when applicable)
4. **Success Criteria**: Define what success looks like
5. **Potential Risks**: Identify things that could go wrong
6. **Confidence**: Rate your confidence (0.0 to 1.0)

## Output Format
Respond with a JSON object:
```json
{
    "task_understanding": "Your understanding of the task",
    "approach": "High-level approach",
    "steps": [
        {
            "action": "Specific action to take",
            "tool": "tool_name or null",
            "inputs": {"key": "value"},
            "expected_output": "What success looks like",
            "fallback": "Alternative if this fails or null"
        }
    ],
    "success_criteria": ["Criterion 1", "Criterion 2"],
    "potential_risks": ["Risk 1", "Risk 2"],
    "confidence": 0.8
}
```

Think step by step and be thorough."""

        return prompt

    def build_validation_prompt(
        self,
        task: str,
        plan: Optional[Any],
        execution_result: Dict[str, Any],
        success_criteria: List[str],
        strictness: str = "medium",
    ) -> str:
        """
        Build prompt for validation phase.

        Args:
            task: Original task
            plan: Execution plan
            execution_result: Results from execution
            success_criteria: Criteria to validate against
            strictness: Validation strictness (low/medium/high)

        Returns:
            Validation prompt
        """
        strictness_guidance = {
            "low": "Be lenient - accept results that partially meet criteria",
            "medium": "Be balanced - require most criteria to be met",
            "high": "Be strict - all criteria must be fully met",
        }

        prompt = f"""You are a validation agent. Evaluate whether the execution results meet the success criteria.

## Original Task
{task}

## Execution Plan
{self._format_plan(plan) if plan else "No formal plan was used"}

## Execution Results
{self._format_dict(execution_result)}

## Success Criteria
{self._format_list(success_criteria) if success_criteria else "Task should be completed successfully"}

## Validation Strictness
{strictness.upper()}: {strictness_guidance.get(strictness, strictness_guidance['medium'])}

## Instructions
Evaluate the execution results against each success criterion. Consider:
1. Were all required outputs produced?
2. Is the quality acceptable?
3. Were there any errors that affect the outcome?
4. Does the result actually solve the original task?

## Output Format
Respond with a JSON object:
```json
{{
    "is_valid": true/false,
    "score": 0.0 to 1.0,
    "errors": ["Error 1", "Error 2"],
    "warnings": ["Warning 1"],
    "suggestions": ["Suggestion for improvement"],
    "criteria_met": {{
        "criterion_1": true/false,
        "criterion_2": true/false
    }}
}}
```

Be objective and thorough in your assessment."""

        return prompt

    def build_refinement_prompt(
        self,
        task: str,
        plan: Optional[Any],
        execution_result: Optional[Dict[str, Any]],
        validation_result: Optional[Any],
        iteration: int,
        max_iterations: int,
        previous_refinements: Optional[List[Any]] = None,
        strategy: str = "adaptive",
    ) -> str:
        """
        Build prompt for refinement phase.

        Args:
            task: Original task
            plan: Current execution plan
            execution_result: Results from execution
            validation_result: Validation outcome
            iteration: Current iteration number
            max_iterations: Maximum allowed iterations
            previous_refinements: Previous refinement attempts
            strategy: Refinement strategy

        Returns:
            Refinement prompt
        """
        strategy_guidance = {
            "conservative": "Make minimal changes - only fix what's clearly broken",
            "adaptive": "Balance between fixing issues and trying new approaches",
            "aggressive": "Consider significantly different approaches if needed",
        }

        prompt = f"""You are a refinement agent. The previous execution attempt did not fully succeed. Analyze what went wrong and determine the best course of action.

## Original Task
{task}

## Current Iteration
{iteration} of {max_iterations} maximum iterations

## Current Plan
{self._format_plan(plan) if plan else "No formal plan"}

## Execution Result
{self._format_dict(execution_result) if execution_result else "No results"}

## Validation Feedback
{self._format_validation(validation_result) if validation_result else "No validation performed"}

"""

        if previous_refinements:
            prompt += f"""## Previous Refinement Attempts
{self._format_refinements(previous_refinements)}

"""

        prompt += f"""## Refinement Strategy
{strategy.upper()}: {strategy_guidance.get(strategy, strategy_guidance['adaptive'])}

## Instructions
Analyze the failure and decide on one of these actions:

1. **retry**: Try the same approach again (useful for transient failures)
2. **modify**: Change the approach based on learnings
3. **escalate**: Ask for human input (not implemented yet)
4. **abort**: Give up if the task seems impossible

Consider:
- What specifically went wrong?
- Is this a fixable issue or a fundamental problem?
- Have we tried similar approaches before?
- Do we have iterations remaining to try something different?

## Output Format
Respond with a JSON object:
```json
{{
    "action_type": "retry|modify|escalate|abort",
    "reasoning": "Detailed explanation of why this action was chosen",
    "modifications": {{
        "key": "value for any changes to make"
    }},
    "new_approach": "If modifying, describe the new approach (null otherwise)"
}}
```

Think critically about what will actually lead to success."""

        return prompt

    def build_plan_modification_prompt(
        self,
        original_plan: Optional[Any],
        refinement: Any,
        available_tools: Optional[List[str]] = None,
    ) -> str:
        """
        Build prompt for modifying a plan based on refinement feedback.

        Args:
            original_plan: Original execution plan
            refinement: Refinement action with modifications
            available_tools: Available tools

        Returns:
            Plan modification prompt
        """
        prompt = f"""You are a planning agent. Modify the existing plan based on refinement feedback.

## Original Plan
{self._format_plan(original_plan) if original_plan else "No original plan"}

## Refinement Feedback
Action: {refinement.action_type}
Reasoning: {refinement.reasoning}
New Approach: {refinement.new_approach or "Not specified"}
Modifications: {self._format_dict(refinement.modifications) if refinement.modifications else "None"}

"""

        if available_tools:
            prompt += f"""## Available Tools
{self._format_list(available_tools)}

"""

        prompt += """## Instructions
Create a modified plan that addresses the issues identified. The new plan should:
1. Incorporate the suggested modifications
2. Avoid repeating previous mistakes
3. Be more likely to succeed

## Output Format
Respond with a JSON object matching the plan schema:
```json
{
    "task_understanding": "Updated understanding",
    "approach": "Modified approach",
    "steps": [...],
    "success_criteria": [...],
    "potential_risks": [...],
    "confidence": 0.0 to 1.0
}
```"""

        return prompt

    def build_reflection_prompt(
        self,
        task: str,
        final_state: Any,
    ) -> str:
        """
        Build prompt for post-execution reflection.

        Args:
            task: Original task
            final_state: Final reflection state

        Returns:
            Reflection prompt
        """
        return f"""You are a learning agent. Reflect on the completed task execution to extract learnings.

## Task
{task}

## Execution Summary
- Iterations used: {final_state.iteration}
- Success: {final_state.success}
- Total LLM calls: {final_state.total_llm_calls}
- Total tool calls: {final_state.total_tool_calls}

## Questions to Consider
1. What approach worked well?
2. What could have been done more efficiently?
3. Were there any patterns that led to failure?
4. What should be remembered for similar tasks?

Provide a brief reflection summary."""

    # Helper methods

    def _format_dict(self, d: Dict[str, Any]) -> str:
        """Format dictionary for prompt."""
        if not d:
            return "None"
        lines = []
        for k, v in d.items():
            if isinstance(v, dict):
                lines.append(f"- {k}:")
                for k2, v2 in v.items():
                    lines.append(f"    - {k2}: {v2}")
            elif isinstance(v, list):
                lines.append(f"- {k}: {', '.join(str(x) for x in v)}")
            else:
                lines.append(f"- {k}: {v}")
        return "\n".join(lines)

    def _format_list(self, items: List[str]) -> str:
        """Format list for prompt."""
        if not items:
            return "None"
        return "\n".join(f"- {item}" for item in items)

    def _format_plan(self, plan: Any) -> str:
        """Format execution plan for prompt."""
        if not plan:
            return "None"

        lines = [
            f"Understanding: {plan.task_understanding}",
            f"Approach: {plan.approach}",
            f"Confidence: {plan.confidence}",
            "Steps:",
        ]

        for i, step in enumerate(plan.steps, 1):
            lines.append(f"  {i}. {step.action}")
            if step.tool:
                lines.append(f"     Tool: {step.tool}")

        lines.append("Success Criteria:")
        for criterion in plan.success_criteria:
            lines.append(f"  - {criterion}")

        return "\n".join(lines)

    def _format_validation(self, validation: Any) -> str:
        """Format validation result for prompt."""
        if not validation:
            return "None"

        lines = [
            f"Valid: {validation.is_valid}",
            f"Score: {validation.score}",
        ]

        if validation.errors:
            lines.append("Errors:")
            for error in validation.errors:
                lines.append(f"  - {error}")

        if validation.suggestions:
            lines.append("Suggestions:")
            for suggestion in validation.suggestions:
                lines.append(f"  - {suggestion}")

        return "\n".join(lines)

    def _format_previous_attempts(self, attempts: List[Any]) -> str:
        """Format previous attempts for prompt."""
        if not attempts:
            return "None"

        lines = []
        for i, attempt in enumerate(attempts, 1):
            lines.append(f"Attempt {i}:")
            lines.append(f"  Approach: {attempt.approach}")
            lines.append(f"  Confidence: {attempt.confidence}")
            lines.append("")

        return "\n".join(lines)

    def _format_refinements(self, refinements: List[Any]) -> str:
        """Format previous refinements for prompt."""
        if not refinements:
            return "None"

        lines = []
        for i, ref in enumerate(refinements, 1):
            lines.append(f"Refinement {i}:")
            lines.append(f"  Action: {ref.action_type}")
            lines.append(f"  Reasoning: {ref.reasoning[:100]}...")
            lines.append("")

        return "\n".join(lines)
