"""
LangGraph-based Workflow Orchestration for Master AI.

Implements stateful workflow management using LangGraph's StateGraph
for complex multi-step task orchestration with persistence.
"""
from typing import TypedDict, Optional, List, Any, Annotated
from datetime import datetime
from enum import Enum
import operator
import asyncio

try:
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import MemorySaver
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    StateGraph = None
    END = "END"
    MemorySaver = None

from src.utils.logging import get_logger

logger = get_logger(__name__)


class WorkflowStage(str, Enum):
    """Stages in the Master AI workflow."""
    CLASSIFY = "classify"
    PLAN = "plan"
    EXECUTE = "execute"
    REVIEW = "review"
    APPROVE = "approve"
    COMPLETE = "complete"
    ERROR = "error"


class WorkflowState(TypedDict, total=False):
    """State object for LangGraph workflow."""
    # Input
    user_input: str
    business_id: Optional[str]
    
    # Classification
    intent: Optional[str]
    confidence: float
    requires_approval: bool
    risk_level: str
    
    # Planning
    plan_id: Optional[str]
    tasks: List[dict]
    current_task_index: int
    
    # Execution
    task_results: Annotated[List[dict], operator.add]
    partial_output: str
    
    # Control flow
    stage: WorkflowStage
    error: Optional[str]
    retry_count: int
    
    # Metadata
    started_at: str
    completed_at: Optional[str]


class WorkflowGraph:
    """
    LangGraph-based workflow orchestrator for Master AI.
    
    Provides stateful workflow management with:
    - Automatic checkpointing
    - Conditional routing based on task type and risk
    - Human-in-the-loop approval gates
    - Error handling and retry logic
    """

    def __init__(self, master_ai: Any = None):
        """
        Initialize the workflow graph.
        
        Args:
            master_ai: Reference to the MasterAI instance for delegation
        """
        self.master_ai = master_ai
        self.checkpointer = MemorySaver() if LANGGRAPH_AVAILABLE and MemorySaver else None
        self.graph = None
        
        if LANGGRAPH_AVAILABLE:
            self._build_graph()
        else:
            logger.warning("LangGraph not available - using fallback workflow")

    def _build_graph(self):
        """Build the LangGraph state graph."""
        if not LANGGRAPH_AVAILABLE:
            return
            
        # Create the graph with our state schema
        workflow = StateGraph(WorkflowState)
        
        # Add nodes
        workflow.add_node("classify", self._classify_node)
        workflow.add_node("plan", self._plan_node)
        workflow.add_node("check_approval", self._check_approval_node)
        workflow.add_node("execute", self._execute_node)
        workflow.add_node("review", self._review_node)
        workflow.add_node("handle_error", self._handle_error_node)
        
        # Set entry point
        workflow.set_entry_point("classify")
        
        # Add edges
        workflow.add_edge("classify", "plan")
        
        # Conditional edge after planning
        workflow.add_conditional_edges(
            "plan",
            self._should_require_approval,
            {
                "approve": "check_approval",
                "execute": "execute",
                "error": "handle_error",
            }
        )
        
        # Conditional edge after approval check
        workflow.add_conditional_edges(
            "check_approval",
            self._approval_result,
            {
                "approved": "execute",
                "pending": END,  # Wait for human
                "rejected": END,
            }
        )
        
        # Execution to review
        workflow.add_edge("execute", "review")
        
        # Conditional edge after review
        workflow.add_conditional_edges(
            "review",
            self._should_continue,
            {
                "continue": "execute",
                "complete": END,
                "error": "handle_error",
            }
        )
        
        # Error handling back to execute or end
        workflow.add_conditional_edges(
            "handle_error",
            self._should_retry,
            {
                "retry": "execute",
                "abort": END,
            }
        )
        
        # Compile the graph
        self.graph = workflow.compile(checkpointer=self.checkpointer)
        logger.info("LangGraph workflow compiled successfully")

    async def _classify_node(self, state: WorkflowState) -> dict:
        """Classify the user input and determine intent."""
        logger.info(f"Classifying input: {state.get('user_input', '')[:50]}...")
        
        intent = "command"  # Default
        confidence = 0.8
        requires_approval = False
        risk_level = "low"
        
        if self.master_ai:
            try:
                # Use Master AI's classification if available
                result = await self.master_ai._classify_intent(state.get("user_input", ""))
                intent = result.get("intent", "command")
                confidence = result.get("confidence", 0.8)
            except Exception as e:
                logger.error(f"Classification error: {e}")
        
        # Determine risk level and approval needs
        high_risk_keywords = ["financial", "legal", "external", "deploy", "production"]
        user_input = state.get("user_input", "").lower()
        
        for keyword in high_risk_keywords:
            if keyword in user_input:
                risk_level = "high"
                requires_approval = True
                break
        
        return {
            "intent": intent,
            "confidence": confidence,
            "requires_approval": requires_approval,
            "risk_level": risk_level,
            "stage": WorkflowStage.PLAN,
        }

    async def _plan_node(self, state: WorkflowState) -> dict:
        """Create execution plan for the task."""
        logger.info("Creating execution plan...")
        
        tasks = []
        plan_id = None
        
        if self.master_ai:
            try:
                # Use Master AI's planner
                plan = await self.master_ai._create_plan(
                    state.get("user_input", ""),
                    state.get("business_id")
                )
                tasks = plan.get("tasks", [])
                plan_id = plan.get("id")
            except Exception as e:
                logger.error(f"Planning error: {e}")
                return {"error": str(e), "stage": WorkflowStage.ERROR}
        else:
            # Simple fallback plan
            tasks = [{
                "id": "task_1",
                "action": "process",
                "input": state.get("user_input", ""),
            }]
        
        return {
            "plan_id": plan_id,
            "tasks": tasks,
            "current_task_index": 0,
            "stage": WorkflowStage.EXECUTE if not state.get("requires_approval") else WorkflowStage.APPROVE,
        }

    async def _check_approval_node(self, state: WorkflowState) -> dict:
        """Check if approval has been granted."""
        logger.info("Checking approval status...")
        
        # In production, this would check the approval manager
        # For now, we simulate pending approval
        return {
            "stage": WorkflowStage.APPROVE,
        }

    async def _execute_node(self, state: WorkflowState) -> dict:
        """Execute the current task in the plan."""
        tasks = state.get("tasks", [])
        current_index = state.get("current_task_index", 0)
        
        if current_index >= len(tasks):
            return {"stage": WorkflowStage.COMPLETE}
        
        task = tasks[current_index]
        logger.info(f"Executing task {current_index + 1}/{len(tasks)}: {task.get('id', 'unknown')}")
        
        result = {"task_id": task.get("id"), "success": False, "output": None}
        
        if self.master_ai:
            try:
                output = await self.master_ai._execute_task(task)
                result = {
                    "task_id": task.get("id"),
                    "success": True,
                    "output": output,
                }
            except Exception as e:
                logger.error(f"Task execution error: {e}")
                result = {
                    "task_id": task.get("id"),
                    "success": False,
                    "error": str(e),
                }
        else:
            # Simulated execution
            result = {
                "task_id": task.get("id"),
                "success": True,
                "output": f"Executed: {task.get('action', 'unknown')}",
            }
        
        return {
            "task_results": [result],
            "current_task_index": current_index + 1,
            "stage": WorkflowStage.REVIEW,
        }

    async def _review_node(self, state: WorkflowState) -> dict:
        """Review execution results and determine next steps."""
        task_results = state.get("task_results", [])
        current_index = state.get("current_task_index", 0)
        total_tasks = len(state.get("tasks", []))
        
        # Check last result
        if task_results:
            last_result = task_results[-1]
            if not last_result.get("success", False):
                return {"stage": WorkflowStage.ERROR, "error": last_result.get("error", "Unknown error")}
        
        # Check if more tasks to execute
        if current_index < total_tasks:
            return {"stage": WorkflowStage.EXECUTE}
        
        # All tasks complete
        return {
            "stage": WorkflowStage.COMPLETE,
            "completed_at": datetime.utcnow().isoformat(),
        }

    async def _handle_error_node(self, state: WorkflowState) -> dict:
        """Handle errors and determine retry strategy."""
        error = state.get("error", "Unknown error")
        retry_count = state.get("retry_count", 0)
        
        logger.error(f"Handling error (retry {retry_count}): {error}")
        
        return {
            "retry_count": retry_count + 1,
        }

    def _should_require_approval(self, state: WorkflowState) -> str:
        """Determine if approval is required."""
        if state.get("error"):
            return "error"
        if state.get("requires_approval"):
            return "approve"
        return "execute"

    def _approval_result(self, state: WorkflowState) -> str:
        """Determine approval result."""
        # In production, check actual approval status
        stage = state.get("stage")
        if stage == WorkflowStage.APPROVE:
            return "pending"
        return "approved"

    def _should_continue(self, state: WorkflowState) -> str:
        """Determine if execution should continue."""
        stage = state.get("stage")
        if stage == WorkflowStage.ERROR:
            return "error"
        if stage == WorkflowStage.COMPLETE:
            return "complete"
        return "continue"

    def _should_retry(self, state: WorkflowState) -> str:
        """Determine if retry should be attempted."""
        retry_count = state.get("retry_count", 0)
        max_retries = 3
        
        if retry_count < max_retries:
            return "retry"
        return "abort"

    async def run(
        self,
        user_input: str,
        business_id: Optional[str] = None,
        thread_id: Optional[str] = None,
    ) -> dict:
        """
        Run the workflow for a user input.
        
        Args:
            user_input: The user's input/command
            business_id: Optional business context
            thread_id: Optional thread ID for conversation continuity
            
        Returns:
            Final workflow state with results
        """
        initial_state: WorkflowState = {
            "user_input": user_input,
            "business_id": business_id,
            "stage": WorkflowStage.CLASSIFY,
            "task_results": [],
            "retry_count": 0,
            "started_at": datetime.utcnow().isoformat(),
        }
        
        if not LANGGRAPH_AVAILABLE or not self.graph:
            # Fallback execution without LangGraph
            logger.info("Running fallback workflow (LangGraph not available)")
            return await self._fallback_run(initial_state)
        
        config = {"configurable": {"thread_id": thread_id or "default"}}
        
        try:
            # Run the graph
            final_state = await self.graph.ainvoke(initial_state, config)
            return dict(final_state)
        except Exception as e:
            logger.error(f"Workflow execution error: {e}")
            return {
                **initial_state,
                "stage": WorkflowStage.ERROR,
                "error": str(e),
            }

    async def _fallback_run(self, state: WorkflowState) -> dict:
        """Fallback workflow when LangGraph is not available."""
        # Simple sequential execution
        state = {**state, **await self._classify_node(state)}
        state = {**state, **await self._plan_node(state)}
        
        if state.get("error"):
            return state
        
        while state.get("current_task_index", 0) < len(state.get("tasks", [])):
            state = {**state, **await self._execute_node(state)}
            state = {**state, **await self._review_node(state)}
            
            if state.get("stage") in (WorkflowStage.ERROR, WorkflowStage.COMPLETE):
                break
        
        return state

    async def get_state(self, thread_id: str) -> Optional[dict]:
        """Get the current state for a thread."""
        if not LANGGRAPH_AVAILABLE or not self.checkpointer:
            return None
        
        try:
            config = {"configurable": {"thread_id": thread_id}}
            state = self.graph.get_state(config)
            return dict(state.values) if state else None
        except Exception as e:
            logger.error(f"Error getting workflow state: {e}")
            return None

    async def resume(self, thread_id: str, updates: dict) -> dict:
        """
        Resume a paused workflow with updates (e.g., after approval).
        
        Args:
            thread_id: The thread ID to resume
            updates: State updates to apply before resuming
            
        Returns:
            Final workflow state
        """
        if not LANGGRAPH_AVAILABLE or not self.graph:
            return {"error": "LangGraph not available"}
        
        try:
            config = {"configurable": {"thread_id": thread_id}}
            
            # Update state
            self.graph.update_state(config, updates)
            
            # Resume execution
            final_state = await self.graph.ainvoke(None, config)
            return dict(final_state)
        except Exception as e:
            logger.error(f"Error resuming workflow: {e}")
            return {"error": str(e)}


# Global instance
workflow_graph: Optional[WorkflowGraph] = None


def get_workflow_graph(master_ai: Any = None) -> WorkflowGraph:
    """Get or create the global workflow graph instance."""
    global workflow_graph
    if workflow_graph is None:
        workflow_graph = WorkflowGraph(master_ai)
    return workflow_graph
