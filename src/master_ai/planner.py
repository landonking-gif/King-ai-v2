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
