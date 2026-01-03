"""
Agentic Core Package for Kautilya.

This package provides intelligent agentic capabilities:
- Dynamic capability discovery (skills, tools, MCP servers)
- Task planning and decomposition
- ReAct reasoning loop (Reason -> Act -> Observe -> Reflect)
- Error recovery and self-correction
- Output validation and self-critique
- Session memory for learning from failures

Usage:
    from kautilya.agent import AgentCore

    agent = AgentCore(llm_client, tool_executor)
    result = agent.process("Extract ESG metrics from @reports/sample.pdf")
"""

from .core import AgentCore, ProcessingResult
from .file_resolver import FileResolver, FileMatch
from .capability_registry import CapabilityRegistry, Capability
from .task_planner import TaskPlanner, ExecutionPlan, PlanStep
from .error_recovery import ErrorRecoveryEngine, RecoveryAction, ErrorCategory, RecoveryStrategy
from .session_memory import SessionMemory, MemoryType
from .react_loop import ReActLoop, ThoughtAction, LoopResult, LoopStatus
from .output_validator import OutputValidator, ValidationResult, ValidationLevel
from .integration import AgentIntegration, create_agent_integration

__all__ = [
    # Core
    "AgentCore",
    "ProcessingResult",
    # Integration
    "AgentIntegration",
    "create_agent_integration",
    # File Resolution
    "FileResolver",
    "FileMatch",
    # Capability Discovery
    "CapabilityRegistry",
    "Capability",
    # Task Planning
    "TaskPlanner",
    "ExecutionPlan",
    "PlanStep",
    # Error Recovery
    "ErrorRecoveryEngine",
    "RecoveryAction",
    "RecoveryStrategy",
    "ErrorCategory",
    # Session Memory
    "SessionMemory",
    "MemoryType",
    # ReAct Loop
    "ReActLoop",
    "ThoughtAction",
    "LoopResult",
    "LoopStatus",
    # Output Validation
    "OutputValidator",
    "ValidationResult",
    "ValidationLevel",
]
