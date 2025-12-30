# King AI v2 - Implementation Plan Part 2 of 20
## Master AI Brain - Core Enhancements

**Target Timeline:** Week 2-3
**Objective:** Enhance the Master AI brain with production-ready LLM integration, improved intent classification, and robust error handling.

---

## Table of Contents
1. [Overview of 20-Part Plan](#overview-of-20-part-plan)
2. [Part 2 Scope](#part-2-scope)
3. [Current State Analysis](#current-state-analysis)
4. [Implementation Tasks](#implementation-tasks)
5. [File-by-File Instructions](#file-by-file-instructions)
6. [Testing Requirements](#testing-requirements)
7. [Acceptance Criteria](#acceptance-criteria)

---

## Overview of 20-Part Plan

| Part | Title | Status |
|------|-------|--------|
| 1 | Infrastructure Layer & Core System Hardening | ✅ Complete |
| **2** | **Master AI Brain - Core Enhancements** | 🔄 Current |
| 3 | Master AI Brain - Context & Memory System | ⏳ Pending |
| 4 | Master AI Brain - Planning & ReAct Implementation | ⏳ Pending |
| 5 | Evolution Engine - Code Modification System | ⏳ Pending |
| 6 | Evolution Engine - ML Retraining Pipeline | ⏳ Pending |
| 7 | Evolution Engine - Sandbox & Testing | ⏳ Pending |
| 8 | Sub-Agents - Research Agent Enhancement | ⏳ Pending |
| 9 | Sub-Agents - Code Generator Agent | ⏳ Pending |
| 10 | Sub-Agents - Content Agent | ⏳ Pending |
| 11 | Sub-Agents - Commerce Agent (Shopify/AliExpress) | ⏳ Pending |
| 12 | Sub-Agents - Finance Agent (Stripe/Plaid) | ⏳ Pending |
| 13 | Sub-Agents - Analytics Agent | ⏳ Pending |
| 14 | Sub-Agents - Legal Agent | ⏳ Pending |
| 15 | Business Units - Lifecycle Engine | ⏳ Pending |
| 16 | Business Units - Playbook System | ⏳ Pending |
| 17 | Business Units - Portfolio Management | ⏳ Pending |
| 18 | Dashboard - React UI Components | ⏳ Pending |
| 19 | Dashboard - Approval Workflows & Risk Engine | ⏳ Pending |
| 20 | Dashboard - Real-time Monitoring & WebSocket + Final Integration | ⏳ Pending |

---

## Part 2 Scope

This part focuses on:
1. Integrating the LLM Router into the Master AI brain
2. Enhancing intent classification with structured outputs
3. Adding retry logic and error handling
4. Implementing rate limiting and token management
5. Adding comprehensive logging and metrics

---

## Current State Analysis

### What Exists in `src/master_ai/brain.py`
| Feature | Status | Issue |
|---------|--------|-------|
| Basic MasterAI class | ✅ Exists | Needs enhancement |
| Intent classification | ✅ Basic | No structured validation |
| LLM calls via Ollama | ✅ Works | No fallback routing |
| Gemini fallback | ✅ Basic | Not integrated with router |
| Error handling | ⚠️ Minimal | No retry logic |
| Logging | ❌ Missing | No structured logging |
| Metrics | ❌ Missing | No performance tracking |

### What Needs to Change
1. Replace direct Ollama calls with LLMRouter
2. Add Pydantic models for intent classification
3. Implement exponential backoff retry
4. Add structured logging with context
5. Integrate Datadog metrics
6. Add token counting and budget management

---

## Implementation Tasks

### Task 2.1: Create Intent Classification Models
**Priority:** 🔴 Critical
**Estimated Time:** 2 hours
**Dependencies:** Part 1 complete

#### File: `src/master_ai/models.py` (CREATE NEW FILE)
```python
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
```

---

### Task 2.2: Create Structured Logging Utility
**Priority:** 🟡 High
**Estimated Time:** 2 hours
**Dependencies:** None

#### File: `src/utils/structured_logging.py` (CREATE NEW FILE)
```python
"""
Structured logging for the Master AI system.
Provides context-aware logging with JSON output for aggregation.
"""

import logging
import json
import sys
from datetime import datetime
from typing import Any, Dict, Optional
from contextvars import ContextVar
from functools import wraps
import traceback

# Context variable for request tracking
request_id_var: ContextVar[str] = ContextVar('request_id', default='no-request-id')
user_id_var: ContextVar[str] = ContextVar('user_id', default='anonymous')
business_id_var: ContextVar[str] = ContextVar('business_id', default='none')


class JSONFormatter(logging.Formatter):
    """Format log records as JSON for easy parsing."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "request_id": request_id_var.get(),
            "user_id": user_id_var.get(),
            "business_id": business_id_var.get(),
        }
        
        # Add extra fields if present
        if hasattr(record, 'extra_data'):
            log_data["data"] = record.extra_data
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": traceback.format_exception(*record.exc_info)
            }
        
        return json.dumps(log_data)


class StructuredLogger:
    """
    Wrapper around Python's logging with structured output.
    Automatically includes context from ContextVars.
    """
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # Only add handler if not already present
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(JSONFormatter())
            self.logger.addHandler(handler)
    
    def _log(self, level: int, message: str, data: Dict[str, Any] = None, exc_info=None):
        """Internal logging method."""
        extra = {}
        if data:
            extra['extra_data'] = data
        
        self.logger.log(level, message, extra=extra, exc_info=exc_info)
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self._log(logging.DEBUG, message, kwargs if kwargs else None)
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        self._log(logging.INFO, message, kwargs if kwargs else None)
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self._log(logging.WARNING, message, kwargs if kwargs else None)
    
    def error(self, message: str, exc_info=False, **kwargs):
        """Log error message."""
        self._log(logging.ERROR, message, kwargs if kwargs else None, exc_info=exc_info)
    
    def critical(self, message: str, exc_info=False, **kwargs):
        """Log critical message."""
        self._log(logging.CRITICAL, message, kwargs if kwargs else None, exc_info=exc_info)
    
    def llm_call(
        self,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float,
        success: bool,
        error: str = None
    ):
        """Log an LLM API call."""
        self.info(
            "LLM call completed",
            provider=provider,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            latency_ms=latency_ms,
            success=success,
            error=error
        )
    
    def agent_execution(
        self,
        agent: str,
        task: str,
        duration_ms: float,
        success: bool,
        error: str = None
    ):
        """Log an agent task execution."""
        self.info(
            "Agent execution completed",
            agent=agent,
            task=task,
            duration_ms=duration_ms,
            success=success,
            error=error
        )
    
    def business_event(
        self,
        event_type: str,
        business_id: str,
        details: Dict[str, Any] = None
    ):
        """Log a business-related event."""
        self.info(
            f"Business event: {event_type}",
            event_type=event_type,
            business_id=business_id,
            details=details or {}
        )


def set_request_context(request_id: str, user_id: str = None, business_id: str = None):
    """Set context variables for the current request."""
    request_id_var.set(request_id)
    if user_id:
        user_id_var.set(user_id)
    if business_id:
        business_id_var.set(business_id)


def log_function_call(logger: StructuredLogger):
    """Decorator to log function entry and exit."""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            logger.debug(f"Entering {func.__name__}", args_count=len(args), kwargs_keys=list(kwargs.keys()))
            try:
                result = await func(*args, **kwargs)
                logger.debug(f"Exiting {func.__name__}", success=True)
                return result
            except Exception as e:
                logger.error(f"Error in {func.__name__}", exc_info=True, error=str(e))
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            logger.debug(f"Entering {func.__name__}", args_count=len(args), kwargs_keys=list(kwargs.keys()))
            try:
                result = func(*args, **kwargs)
                logger.debug(f"Exiting {func.__name__}", success=True)
                return result
            except Exception as e:
                logger.error(f"Error in {func.__name__}", exc_info=True, error=str(e))
                raise
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    return decorator


# Create default loggers for each module
def get_logger(name: str) -> StructuredLogger:
    """Get a structured logger for a module."""
    return StructuredLogger(f"king_ai.{name}")
```

---

### Task 2.3: Implement Retry Logic with Exponential Backoff
**Priority:** 🔴 Critical
**Estimated Time:** 2 hours
**Dependencies:** None

#### File: `src/utils/retry.py` (CREATE NEW FILE)
```python
"""
Retry utilities with exponential backoff.
Handles transient failures gracefully.
"""

import asyncio
import random
from typing import TypeVar, Callable, Any, Type, Tuple
from functools import wraps
from dataclasses import dataclass

from src.utils.structured_logging import get_logger

logger = get_logger("retry")

T = TypeVar('T')


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    base_delay: float = 1.0  # seconds
    max_delay: float = 60.0  # seconds
    exponential_base: float = 2.0
    jitter: bool = True
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,)


def calculate_delay(
    attempt: int,
    config: RetryConfig
) -> float:
    """Calculate delay for next retry with exponential backoff."""
    delay = min(
        config.base_delay * (config.exponential_base ** attempt),
        config.max_delay
    )
    
    if config.jitter:
        # Add random jitter (±25%)
        jitter_range = delay * 0.25
        delay += random.uniform(-jitter_range, jitter_range)
    
    return max(0, delay)


async def retry_async(
    func: Callable[..., Any],
    *args,
    config: RetryConfig = None,
    **kwargs
) -> Any:
    """
    Execute an async function with retry logic.
    
    Args:
        func: The async function to execute
        *args: Positional arguments for the function
        config: Retry configuration
        **kwargs: Keyword arguments for the function
        
    Returns:
        The function result
        
    Raises:
        The last exception if all retries fail
    """
    if config is None:
        config = RetryConfig()
    
    last_exception = None
    
    for attempt in range(config.max_attempts):
        try:
            return await func(*args, **kwargs)
        except config.retryable_exceptions as e:
            last_exception = e
            
            if attempt < config.max_attempts - 1:
                delay = calculate_delay(attempt, config)
                logger.warning(
                    f"Retry attempt {attempt + 1}/{config.max_attempts}",
                    function=func.__name__,
                    error=str(e),
                    delay_seconds=delay
                )
                await asyncio.sleep(delay)
            else:
                logger.error(
                    f"All retry attempts exhausted",
                    function=func.__name__,
                    error=str(e),
                    attempts=config.max_attempts
                )
    
    raise last_exception


def with_retry(config: RetryConfig = None):
    """
    Decorator to add retry logic to async functions.
    
    Usage:
        @with_retry(RetryConfig(max_attempts=5))
        async def flaky_function():
            ...
    """
    if config is None:
        config = RetryConfig()
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await retry_async(func, *args, config=config, **kwargs)
        return wrapper
    return decorator


class RetryableError(Exception):
    """Base class for errors that should trigger retry."""
    pass


class RateLimitError(RetryableError):
    """Raised when API rate limit is hit."""
    def __init__(self, retry_after: float = None):
        self.retry_after = retry_after
        super().__init__(f"Rate limit exceeded. Retry after: {retry_after}s")


class TransientError(RetryableError):
    """Raised for temporary failures (network, timeout, etc.)."""
    pass


class PermanentError(Exception):
    """Raised for errors that should not be retried."""
    pass


# Pre-configured retry configs for common scenarios
LLM_RETRY_CONFIG = RetryConfig(
    max_attempts=3,
    base_delay=2.0,
    max_delay=30.0,
    retryable_exceptions=(TransientError, RateLimitError, TimeoutError, ConnectionError)
)

DB_RETRY_CONFIG = RetryConfig(
    max_attempts=5,
    base_delay=0.5,
    max_delay=10.0,
    retryable_exceptions=(TransientError, ConnectionError)
)

API_RETRY_CONFIG = RetryConfig(
    max_attempts=3,
    base_delay=1.0,
    max_delay=15.0,
    retryable_exceptions=(TransientError, RateLimitError, TimeoutError)
)
```

---

### Task 2.4: Update Master AI Brain with Router Integration
**Priority:** 🔴 Critical
**Estimated Time:** 4 hours
**Dependencies:** Tasks 2.1, 2.2, 2.3, Part 1

#### File: `src/master_ai/brain.py` (REPLACE ENTIRE FILE)
```python
"""
Master AI Brain - The central orchestrator for the autonomous empire.

This is the ONLY component that directly interfaces with the LLM (via LLMRouter).
All strategic decisions, operational planning, and agent delegation flow through this class.
It serves as the "CEO" of the business empire.
"""

import json
import asyncio
import time
from datetime import datetime
from typing import Optional, Any
from uuid import uuid4

# Internal project imports
from src.master_ai.context import ContextManager
from src.master_ai.planner import Planner
from src.master_ai.evolution import EvolutionEngine
from src.master_ai.prompts import SYSTEM_PROMPT, INTENT_CLASSIFICATION_PROMPT
from src.master_ai.models import (
    ClassifiedIntent, IntentType, ActionType, MasterAIResponse,
    ActionResult, TokenUsage, PlanStep
)
from src.agents.router import AgentRouter
from src.database.connection import get_db
from src.database.models import ConversationMessage, Task, EvolutionProposal
from src.utils.llm_router import LLMRouter, TaskContext
from src.utils.structured_logging import get_logger, set_request_context
from src.utils.retry import with_retry, LLM_RETRY_CONFIG, TransientError
from src.utils.monitoring import monitor, trace_llm
from config.settings import settings


logger = get_logger("master_ai")


class MasterAI:
    """
    The Master AI core that manages the autonomous lifecycle of businesses.
    
    Key Responsibilities:
    1. Intent Classification: Deciding if a user wants to chat, query data, or issue commands.
    2. Planning: Breaking down complex user goals into executable subtasks.
    3. Delegation: Routing subtasks to specialized agents (e.g., ResearchAgent).
    4. Autonomous Optimization: Periodic background checks to improve business performance.
    5. Evolution: Proposing source code or structural changes to itself.
    """
    
    def __init__(self):
        """Initializes all sub-components and management engines."""
        logger.info("Initializing Master AI brain")
        
        # Primary LLM interface - handles all provider routing and fallback
        self.llm_router = LLMRouter()
        
        # Specialized manager components
        self.context = ContextManager()
        self.planner = Planner(self.llm_router)
        self.evolution = EvolutionEngine(self.llm_router)
        self.agent_router = AgentRouter()
        
        # State and rate limiting for autonomous features
        self.autonomous_mode = settings.enable_autonomous_mode
        self._evolution_count_this_hour = 0
        self._hour_start = datetime.now()
        self._total_tokens_today = 0
        self._token_budget_daily = 1_000_000  # 1M tokens per day
        
        logger.info(
            "Master AI initialized",
            autonomous_mode=self.autonomous_mode,
            risk_profile=settings.risk_profile
        )
    
    async def process_input(self, user_input: str, request_id: str = None) -> MasterAIResponse:
        """
        Primary entry point for any user interaction (API or CLI).
        
        Logic:
        1. Build the current global context (financials, active tasks, history).
        2. Use the LLM to classify the user's intent.
        3. Dispatch to the appropriate handler (conversation, command, or query).
        
        Args:
            user_input: The user's message
            request_id: Optional request ID for tracing
            
        Returns:
            Structured MasterAIResponse
        """
        request_id = request_id or str(uuid4())
        set_request_context(request_id)
        start_time = time.time()
        
        logger.info("Processing user input", input_length=len(user_input))
        monitor.increment("master_ai.requests")
        
        try:
            # 1. Gather all relevant data for the LLM to make informed decisions
            context = await self.context.build_context()
            
            # 2. Classify the user's intent with structured output
            intent = await self._classify_intent(user_input, context)
            logger.info(
                "Intent classified",
                intent_type=intent.type.value,
                action=intent.action.value if intent.action else None,
                confidence=intent.confidence
            )
            
            # 3. Route to appropriate handler based on intent
            if intent.type == IntentType.CONVERSATION:
                response = await self._handle_conversation(user_input, context)
                result = MasterAIResponse(
                    type="conversation",
                    response=response,
                    metadata={"intent_confidence": intent.confidence}
                )
                
            elif intent.type == IntentType.COMMAND:
                result = await self._handle_command(user_input, intent, context)
                
            elif intent.type == IntentType.QUERY:
                response = await self._handle_query(user_input, context)
                result = MasterAIResponse(
                    type="conversation",
                    response=response,
                    metadata={"intent_confidence": intent.confidence}
                )
            else:
                result = MasterAIResponse(
                    type="error",
                    response="Unknown intent type"
                )
            
            # Record metrics
            duration_ms = (time.time() - start_time) * 1000
            monitor.timing("master_ai.request_duration", duration_ms)
            monitor.increment("master_ai.requests_success")
            
            logger.info(
                "Request completed",
                duration_ms=duration_ms,
                response_type=result.type
            )
            
            return result
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            monitor.increment("master_ai.requests_error")
            logger.error("Request failed", exc_info=True, error=str(e))
            
            return MasterAIResponse(
                type="error",
                response=f"I encountered an error processing your request: {str(e)}",
                metadata={"error": str(e), "duration_ms": duration_ms}
            )

    @with_retry(LLM_RETRY_CONFIG)
    async def _classify_intent(self, user_input: str, context: str) -> ClassifiedIntent:
        """
        Uses the LLM as a router to interpret natural language into a structured intent.
        
        Returns a validated ClassifiedIntent object.
        """
        prompt = INTENT_CLASSIFICATION_PROMPT.format(
            user_input=user_input,
            context=context[:2000]  # Limit context for classification
        )
        
        # Use low-risk task context for classification
        task_context = TaskContext(
            task_type="classification",
            risk_level="low",
            requires_accuracy=False,
            token_estimate=500,
            priority="normal"
        )
        
        response = await self._call_llm(prompt, task_context=task_context)
        
        try:
            # Parse and validate the response
            parsed = self._parse_json_response(response)
            
            return ClassifiedIntent(
                type=IntentType(parsed.get("type", "conversation")),
                action=ActionType(parsed.get("action")) if parsed.get("action") else None,
                parameters=parsed.get("parameters", {}),
                confidence=parsed.get("confidence", 0.5),
                reasoning=parsed.get("reasoning")
            )
        except Exception as e:
            logger.warning(
                "Failed to parse intent, defaulting to conversation",
                error=str(e),
                response=response[:200]
            )
            return ClassifiedIntent(type=IntentType.CONVERSATION, confidence=0.3)
    
    async def _handle_conversation(self, user_input: str, context: str) -> str:
        """Handles chit-chat while maintaining awareness of the empire's state."""
        prompt = f"""{SYSTEM_PROMPT}

CURRENT CONTEXT OF THE EMPIRE:
{context}

USER MESSAGE:
{user_input}

Respond naturally as King AI, the autonomous CEO. Be professional, data-aware, and concise.
"""
        task_context = TaskContext(
            task_type="conversation",
            risk_level="low",
            requires_accuracy=False,
            token_estimate=1000,
            priority="normal"
        )
        
        return await self._call_llm(prompt, task_context=task_context)
    
    async def _handle_command(
        self,
        user_input: str,
        intent: ClassifiedIntent,
        context: str
    ) -> MasterAIResponse:
        """
        Orchestrates complex actions:
        1. Breaks goal into steps via the Planner.
        2. Checks for approval requirements based on Risk Profile.
        3. Executes immediate tasks or queues for user review.
        """
        logger.info(
            "Handling command",
            action=intent.action.value if intent.action else "unknown",
            parameters=intent.parameters
        )
        
        # Create a detailed multi-step plan
        plan = await self.planner.create_plan(
            goal=user_input,
            action=intent.action.value if intent.action else None,
            parameters=intent.parameters,
            context=context
        )
        
        actions_taken = []
        pending_approvals = []
        
        # Iterate through planned steps
        for step in plan.get("steps", []):
            step_model = PlanStep(**step) if isinstance(step, dict) else step
            
            if step_model.requires_approval:
                # If high risk, save to DB for user review
                task_info = await self._create_approval_task(step_model)
                pending_approvals.append(task_info)
                logger.info("Task queued for approval", task_name=step_model.name)
            else:
                # If low risk, execute immediately via router
                start = time.time()
                result = await self.agent_router.execute(step)
                duration_ms = (time.time() - start) * 1000
                
                action_result = ActionResult(
                    step_name=step_model.name,
                    agent=step_model.agent,
                    success=result.get("success", False),
                    output=result.get("output"),
                    error=result.get("error"),
                    duration_ms=duration_ms
                )
                actions_taken.append(action_result)
                
                logger.agent_execution(
                    agent=step_model.agent,
                    task=step_model.name,
                    duration_ms=duration_ms,
                    success=result.get("success", False)
                )
        
        # Generate a final confirmation or status report for the user
        response = await self._generate_action_summary(
            user_input,
            [a.dict() for a in actions_taken],
            pending_approvals,
            context
        )
        
        return MasterAIResponse(
            type="action",
            response=response,
            actions_taken=actions_taken,
            pending_approvals=pending_approvals,
            metadata={
                "plan_steps": len(plan.get("steps", [])),
                "executed": len(actions_taken),
                "pending": len(pending_approvals)
            }
        )
    
    async def _generate_action_summary(
        self,
        user_input: str,
        actions: list,
        approvals: list,
        context: str
    ) -> str:
        """Uses LLM to summarize what was just done and what is pending."""
        prompt = f"""Summarize the actions taken and items pending approval in response to: "{user_input}"
        
Actions Completed: {json.dumps(actions, default=str)}
Pending Review: {json.dumps(approvals, default=str)}

Provide a concise confirmation message to the user. Be specific about what was done and what needs approval.
"""
        task_context = TaskContext(
            task_type="summary",
            risk_level="low",
            requires_accuracy=False,
            token_estimate=500,
            priority="normal"
        )
        
        return await self._call_llm(prompt, task_context=task_context)

    async def _handle_query(self, user_input: str, context: str) -> str:
        """Answers data queries by synthesizing context gathered from the database."""
        prompt = f"""{SYSTEM_PROMPT}

SYSTEM STATE & FINANCIALS:
{context}

USER DATA QUERY:
{user_input}

Provide a data-driven response. Use specific numbers (revenue, costs, status) found in the context.
If the data is not available, say so clearly.
"""
        task_context = TaskContext(
            task_type="query",
            risk_level="low",
            requires_accuracy=True,
            token_estimate=1000,
            priority="normal"
        )
        
        return await self._call_llm(prompt, task_context=task_context)

    @trace_llm
    async def _call_llm(
        self,
        prompt: str,
        system: str = None,
        task_context: TaskContext = None
    ) -> str:
        """
        Central LLM calling method with routing, retry, and metrics.
        All LLM calls should go through this method.
        """
        start_time = time.time()
        
        # Check token budget
        if self._total_tokens_today >= self._token_budget_daily:
            logger.warning("Daily token budget exhausted")
            raise TransientError("Daily token budget exhausted. Try again tomorrow.")
        
        try:
            response = await self.llm_router.complete(
                prompt=prompt,
                system=system,
                context=task_context
            )
            
            # Estimate tokens (rough: 4 chars per token)
            estimated_tokens = (len(prompt) + len(response)) // 4
            self._total_tokens_today += estimated_tokens
            
            duration_ms = (time.time() - start_time) * 1000
            monitor.timing("llm.call_duration", duration_ms)
            monitor.increment("llm.calls_success")
            
            logger.llm_call(
                provider="router",
                model=settings.ollama_model,
                prompt_tokens=len(prompt) // 4,
                completion_tokens=len(response) // 4,
                latency_ms=duration_ms,
                success=True
            )
            
            return response
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            monitor.increment("llm.calls_error")
            
            logger.llm_call(
                provider="router",
                model=settings.ollama_model,
                prompt_tokens=len(prompt) // 4,
                completion_tokens=0,
                latency_ms=duration_ms,
                success=False,
                error=str(e)
            )
            
            raise
    
    def _parse_json_response(self, response: str) -> dict:
        """Parse JSON from LLM response, handling common formatting issues."""
        # Clean up markdown code blocks
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            response = response.split("```")[1].split("```")[0].strip()
        
        return json.loads(response)
    
    async def run_autonomous_loop(self):
        """
        Background optimization loop.
        Runs every 6 hours to analyze business units and propose evolutions.
        """
        logger.info("Starting autonomous loop")
        
        while self.autonomous_mode:
            try:
                context = await self.context.build_context()
                
                # Step 1: Self-improvement analysis
                await self._consider_evolution(context)
                
                # Step 2: Business unit health check
                await self._check_business_health(context)
                
                logger.info("Autonomous loop iteration complete")
                
            except Exception as e:
                logger.error("Autonomous loop error", exc_info=True)
            
            # Wait for the next optimization cycle (6 hours)
            await asyncio.sleep(6 * 60 * 60)
    
    async def _check_business_health(self, context: str):
        """Analyze business unit performance and suggest optimizations."""
        # TODO: Implement in Part 3
        pass
    
    async def _consider_evolution(self, context: str):
        """
        Calls the EvolutionEngine to see if any system improvements are needed.
        Rate limited to prevent runaway self-modifications.
        """
        if not settings.enable_self_modification:
            return
        
        # Periodic reset of the hourly evolution counter
        now = datetime.now()
        if (now - self._hour_start).seconds >= 3600:
            self._evolution_count_this_hour = 0
            self._hour_start = now
        
        # Enforce rate limit from settings
        if self._evolution_count_this_hour >= settings.max_evolutions_per_hour:
            logger.debug("Evolution rate limit reached")
            return
        
        # Ask LLM for a proposal
        proposal = await self.evolution.propose_improvement(context)
        
        if proposal and proposal.get("is_beneficial"):
            self._evolution_count_this_hour += 1
            
            # Validate confidence threshold
            confidence = proposal.get("confidence", 0)
            if confidence < settings.evolution_confidence_threshold:
                logger.info(
                    "Evolution proposal rejected due to low confidence",
                    confidence=confidence,
                    threshold=settings.evolution_confidence_threshold
                )
                return
            
            # Process in sandbox
            await self._process_evolution_proposal(proposal)
    
    async def _process_evolution_proposal(self, proposal: dict):
        """Process a validated evolution proposal through sandbox testing."""
        from src.utils.sandbox import Sandbox
        
        changes = proposal.get("changes", {})
        if not changes:
            return
        
        logger.info(
            "Processing evolution proposal",
            type=proposal.get("type"),
            files=list(changes.keys())
        )
        
        # Create sandbox and test
        sandbox = Sandbox()
        sandbox.create_sandbox(list(changes.keys()))
        
        try:
            all_patches_applied = True
            for file_path, new_code in changes.items():
                if not sandbox.apply_patch(file_path, new_code):
                    all_patches_applied = False
                    break
            
            if all_patches_applied:
                test_result = sandbox.run_tests()
                if test_result["success"]:
                    proposal["confidence"] = min(0.95, proposal.get("confidence", 0) + 0.1)
                    await self._save_evolution_proposal(proposal)
                    logger.info("Evolution proposal saved for approval")
                else:
                    logger.warning(
                        "Evolution proposal failed sandbox tests",
                        output=test_result.get("output", "")[:500]
                    )
        finally:
            sandbox.cleanup()
    
    async def _create_approval_task(self, step: PlanStep) -> dict:
        """Persists a high-risk task that requires human intervention."""
        async with get_db() as db:
            task = Task(
                id=str(uuid4()),
                name=step.name,
                description=step.description,
                type=step.agent,
                status="pending_approval",
                agent=step.agent,
                input_data=step.input_data,
                requires_approval=True
            )
            db.add(task)
            await db.commit()
            
            logger.info("Created approval task", task_id=task.id, name=task.name)
            
            return {
                "id": task.id,
                "name": task.name,
                "description": task.description,
                "agent": task.agent
            }
    
    async def _save_evolution_proposal(self, proposal: dict):
        """Saves a self-modification proposal for user approval."""
        async with get_db() as db:
            prop = EvolutionProposal(
                id=str(uuid4()),
                type=proposal["type"],
                description=proposal["description"],
                rationale=proposal["rationale"],
                proposed_changes=proposal["changes"],
                expected_impact=proposal.get("expected_impact"),
                confidence_score=proposal.get("confidence", 0.0)
            )
            db.add(prop)
            await db.commit()
            
            logger.info(
                "Saved evolution proposal",
                proposal_id=prop.id,
                type=prop.type
            )
```

---

### Task 2.5: Update Prompts with Intent Classification
**Priority:** 🟡 High
**Estimated Time:** 1 hour
**Dependencies:** Task 2.1

#### File: `src/master_ai/prompts.py` (ADD TO EXISTING FILE)
Add this new prompt constant after the existing `SYSTEM_PROMPT`:

```python
# Add this after SYSTEM_PROMPT in the file

INTENT_CLASSIFICATION_PROMPT = """Classify the user's intent and extract relevant parameters.

USER INPUT: "{user_input}"

CURRENT CONTEXT (for reference):
{context}

Analyze the input and respond with JSON:
{{
    "type": "conversation" | "command" | "query",
    "action": null | "start_business" | "stop_business" | "analyze_business" | "optimize_business" | "create_content" | "research_market" | "generate_report" | "propose_evolution",
    "parameters": {{
        // Extracted parameters relevant to the action
        // e.g., "business_type": "dropshipping", "niche": "pet products"
    }},
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation of why this classification"
}}

DEFINITIONS:
- "conversation": General interaction, greetings, philosophical discussion, or unclear intent
- "command": Request to CREATE, START, STOP, MODIFY, or OPTIMIZE something in the empire
- "query": Request for DATA, STATUS, REPORTS, or INFORMATION about existing state

EXAMPLES:
- "Hello, how are you?" -> type: conversation
- "Start a new dropshipping store for pet products" -> type: command, action: start_business
- "How much revenue did we make yesterday?" -> type: query
- "Optimize our marketing spend" -> type: command, action: optimize_business
- "What businesses are currently active?" -> type: query

Respond with valid JSON only.
"""
```

---

## Testing Requirements

### Unit Tests

#### File: `tests/test_master_ai_enhanced.py` (CREATE NEW FILE)
```python
"""
Tests for enhanced Master AI functionality.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime

from src.master_ai.brain import MasterAI
from src.master_ai.models import (
    ClassifiedIntent, IntentType, ActionType, MasterAIResponse
)


class TestIntentClassification:
    """Tests for intent classification."""
    
    @pytest.fixture
    def master_ai(self):
        with patch('src.master_ai.brain.LLMRouter'):
            with patch('src.master_ai.brain.ContextManager'):
                with patch('src.master_ai.brain.Planner'):
                    with patch('src.master_ai.brain.EvolutionEngine'):
                        with patch('src.master_ai.brain.AgentRouter'):
                            return MasterAI()
    
    @pytest.mark.asyncio
    async def test_conversation_intent(self, master_ai):
        """Test classification of conversation intent."""
        master_ai.llm_router.complete = AsyncMock(return_value='''
        {"type": "conversation", "action": null, "parameters": {}, "confidence": 0.9}
        ''')
        master_ai.context.build_context = AsyncMock(return_value="test context")
        
        intent = await master_ai._classify_intent("Hello there", "context")
        assert intent.type == IntentType.CONVERSATION
    
    @pytest.mark.asyncio
    async def test_command_intent(self, master_ai):
        """Test classification of command intent."""
        master_ai.llm_router.complete = AsyncMock(return_value='''
        {"type": "command", "action": "start_business", "parameters": {"niche": "pets"}, "confidence": 0.85}
        ''')
        
        intent = await master_ai._classify_intent("Start a pet store", "context")
        assert intent.type == IntentType.COMMAND
        assert intent.action == ActionType.START_BUSINESS
    
    @pytest.mark.asyncio
    async def test_fallback_on_parse_error(self, master_ai):
        """Test fallback to conversation on parse error."""
        master_ai.llm_router.complete = AsyncMock(return_value="invalid json")
        
        intent = await master_ai._classify_intent("Test input", "context")
        assert intent.type == IntentType.CONVERSATION
        assert intent.confidence < 0.5


class TestMasterAIResponse:
    """Tests for response handling."""
    
    @pytest.fixture
    def master_ai(self):
        with patch('src.master_ai.brain.LLMRouter'):
            with patch('src.master_ai.brain.ContextManager'):
                with patch('src.master_ai.brain.Planner'):
                    with patch('src.master_ai.brain.EvolutionEngine'):
                        with patch('src.master_ai.brain.AgentRouter'):
                            return MasterAI()
    
    @pytest.mark.asyncio
    async def test_process_input_returns_response(self, master_ai):
        """Test that process_input returns structured response."""
        master_ai.context.build_context = AsyncMock(return_value="context")
        master_ai._classify_intent = AsyncMock(return_value=ClassifiedIntent(
            type=IntentType.CONVERSATION
        ))
        master_ai._handle_conversation = AsyncMock(return_value="Hello!")
        
        response = await master_ai.process_input("Hi")
        
        assert isinstance(response, MasterAIResponse)
        assert response.type == "conversation"
        assert response.response == "Hello!"


class TestTokenBudget:
    """Tests for token budget management."""
    
    @pytest.fixture
    def master_ai(self):
        with patch('src.master_ai.brain.LLMRouter'):
            with patch('src.master_ai.brain.ContextManager'):
                with patch('src.master_ai.brain.Planner'):
                    with patch('src.master_ai.brain.EvolutionEngine'):
                        with patch('src.master_ai.brain.AgentRouter'):
                            return MasterAI()
    
    @pytest.mark.asyncio
    async def test_token_budget_enforcement(self, master_ai):
        """Test that token budget is enforced."""
        master_ai._total_tokens_today = master_ai._token_budget_daily
        master_ai.llm_router.complete = AsyncMock(return_value="response")
        
        from src.utils.retry import TransientError
        with pytest.raises(TransientError, match="token budget"):
            await master_ai._call_llm("test prompt")
```

---

## Acceptance Criteria

### Part 2 Completion Checklist

- [ ] **Models Created**
  - [ ] `src/master_ai/models.py` with all Pydantic models
  - [ ] Intent classification models working
  - [ ] Response models validated

- [ ] **Structured Logging**
  - [ ] `src/utils/structured_logging.py` implemented
  - [ ] JSON log format working
  - [ ] Context variables propagating

- [ ] **Retry Logic**
  - [ ] `src/utils/retry.py` implemented
  - [ ] Exponential backoff working
  - [ ] Circuit breaker logic tested

- [ ] **Master AI Brain Updated**
  - [ ] LLMRouter integrated
  - [ ] Intent classification with structured output
  - [ ] Error handling improved
  - [ ] Metrics and logging added
  - [ ] Token budget management working

- [ ] **Tests Passing**
  - [ ] All unit tests pass
  - [ ] Integration tests pass

---

## File Summary

| Action | File Path |
|--------|-----------|
| CREATE | `src/master_ai/models.py` |
| CREATE | `src/utils/structured_logging.py` |
| CREATE | `src/utils/retry.py` |
| REPLACE | `src/master_ai/brain.py` |
| MODIFY | `src/master_ai/prompts.py` |
| CREATE | `tests/test_master_ai_enhanced.py` |

---

## Next Part Preview

**Part 3: Master AI Brain - Context & Memory System** will cover:
- Enhanced ContextManager with RAG integration
- Conversation history management
- Token-aware context truncation
- Semantic search integration
- Long-term memory storage

---

*End of Part 2 - Master AI Brain - Core Enhancements*
