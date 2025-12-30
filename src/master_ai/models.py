"""
Pydantic models for Master AI structured outputs.
Ensures type-safe communication between components.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any, Literal
from enum import Enum
from datetime import datetime


class IntentType(str, Enum):
    """Types of user intents the Master AI can handle."""
    CONVERSATION = "conversation"  # General chat, greetings
    COMMAND = "command"            # Action requests (start/stop/modify)
    QUERY = "query"                # Data/status requests


class ActionType(str, Enum):
    """Specific actions the Master AI can execute."""
    START_BUSINESS = "start_business"
    STOP_BUSINESS = "stop_business"
    ANALYZE_BUSINESS = "analyze_business"
    OPTIMIZE_BUSINESS = "optimize_business"
    CREATE_CONTENT = "create_content"
    RESEARCH_MARKET = "research_market"
    GENERATE_REPORT = "generate_report"
    PROPOSE_EVOLUTION = "propose_evolution"
    UNKNOWN = "unknown"


class ClassifiedIntent(BaseModel):
    """Structured output from intent classification."""
    type: IntentType
    action: Optional[ActionType] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    reasoning: Optional[str] = None
    
    @validator('action', pre=True, always=True)
    def validate_action(cls, v, values):
        """Ensure action is set for command intents."""
        if values.get('type') == IntentType.COMMAND and v is None:
            return ActionType.UNKNOWN
        return v


class PlanStep(BaseModel):
    """A single step in an execution plan."""
    name: str
    description: str
    agent: str  # research, commerce, finance, etc.
    requires_approval: bool = False
    dependencies: List[str] = Field(default_factory=list)
    estimated_duration: str = "unknown"
    input_data: Dict[str, Any] = Field(default_factory=dict)
    risk_level: Literal["low", "medium", "high"] = "low"


class ExecutionPlan(BaseModel):
    """Complete execution plan for a user goal."""
    goal: str
    steps: List[PlanStep]
    total_estimated_duration: str = "unknown"
    requires_human_approval: bool = False
    risk_assessment: str = ""


class ActionResult(BaseModel):
    """Result from executing an action."""
    step_name: str
    agent: str
    success: bool
    output: Any = None
    error: Optional[str] = None
    duration_ms: float = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MasterAIResponse(BaseModel):
    """Standard response format from Master AI."""
    type: Literal["conversation", "action", "error"]
    response: str
    actions_taken: List[ActionResult] = Field(default_factory=list)
    pending_approvals: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)


class TokenUsage(BaseModel):
    """Track token usage for cost management."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    
    def add(self, other: "TokenUsage") -> "TokenUsage":
        """Combine token usage from multiple calls."""
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            estimated_cost_usd=self.estimated_cost_usd + other.estimated_cost_usd
        )


class ConversationTurn(BaseModel):
    """A single turn in a conversation."""
    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    token_usage: Optional[TokenUsage] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
