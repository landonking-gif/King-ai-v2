# King AI v2 - Complete Implementation Blueprint

> **Purpose**: This document contains exact specifications for building King AI v2. It is designed to be given to AI coding assistants for implementation. Every file, class, function, and data structure is specified precisely.

---

## 1. Project Structure

Create the following directory structure exactly:

```
king-ai-v2/
├── pyproject.toml
├── .env.example
├── .gitignore
├── README.md
├── docker-compose.yml
├── Dockerfile
│
├── config/
│   ├── __init__.py
│   ├── settings.py              # Pydantic settings from env
│   ├── risk_profiles.yaml       # Risk tolerance configurations
│   └── playbooks/
│       ├── dropshipping.yaml
│       └── saas.yaml
│
├── src/
│   ├── __init__.py
│   │
│   ├── master_ai/
│   │   ├── __init__.py
│   │   ├── brain.py             # Main MasterAI class
│   │   ├── context.py           # Context window manager
│   │   ├── planner.py           # ReAct planning loop
│   │   ├── evolution.py         # Self-modification system
│   │   └── prompts.py           # System prompts
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base.py              # Abstract SubAgent class
│   │   ├── research.py          # Web research agent
│   │   ├── code_generator.py    # Code generation agent
│   │   ├── content.py           # Content creation agent
│   │   ├── commerce.py          # E-commerce agent
│   │   ├── finance.py           # Finance agent
│   │   ├── analytics.py         # Analytics agent
│   │   ├── legal.py             # Compliance agent
│   │   └── router.py            # Agent selection router
│   │
│   ├── business/
│   │   ├── __init__.py
│   │   ├── unit.py              # BusinessUnit model
│   │   ├── lifecycle.py         # State machine
│   │   ├── portfolio.py         # Portfolio manager
│   │   └── playbook_loader.py   # YAML playbook parser
│   │
│   ├── database/
│   │   ├── __init__.py
│   │   ├── models.py            # SQLAlchemy models
│   │   ├── connection.py        # DB connection manager
│   │   ├── migrations/          # Alembic migrations
│   │   └── vector_store.py      # Pinecone integration
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py              # FastAPI app
│   │   ├── routes/
│   │   │   ├── chat.py          # User chat endpoint
│   │   │   ├── businesses.py    # Business CRUD
│   │   │   ├── approvals.py     # Approval queue
│   │   │   └── evolution.py     # Self-mod proposals
│   │   └── websocket.py         # Real-time updates
│   │
│   └── utils/
│       ├── __init__.py
│       ├── ollama_client.py     # Ollama API wrapper
│       ├── logging.py           # Structured logging
│       └── metrics.py           # Prometheus metrics
│
├── dashboard/                    # React frontend (optional Phase 2)
│   └── ...
│
├── scripts/
│   ├── migrate_v1_data.py       # Migration from old king-ai-studio
│   └── seed_data.py             # Demo data seeding
│
└── tests/
    ├── conftest.py
    ├── test_master_ai.py
    ├── test_agents.py
    └── test_business.py
```

---

## 2. Dependencies

### `pyproject.toml`

```toml
[project]
name = "king-ai-v2"
version = "2.0.0"
description = "Autonomous AI Business Empire"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.109.0",
    "uvicorn[standard]>=0.27.0",
    "pydantic>=2.6.0",
    "pydantic-settings>=2.1.0",
    "sqlalchemy>=2.0.25",
    "alembic>=1.13.0",
    "asyncpg>=0.29.0",
    "redis>=5.0.0",
    "httpx>=0.26.0",
    "pyyaml>=6.0.1",
    "python-dotenv>=1.0.0",
    "structlog>=24.1.0",
    "prometheus-client>=0.19.0",
    "pinecone-client>=3.0.0",
    "beautifulsoup4>=4.12.0",
    "aiofiles>=23.2.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-asyncio>=0.23.0",
    "ruff>=0.1.0",
    "mypy>=1.8.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

---

## 3. Configuration

### `config/settings.py`

```python
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Literal

class Settings(BaseSettings):
    # Database
    database_url: str = Field(..., env="DATABASE_URL")
    redis_url: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    
    # Ollama
    ollama_url: str = Field(..., env="OLLAMA_URL")
    ollama_model: str = Field(default="llama3.1:8b", env="OLLAMA_MODEL")
    
    # Pinecone (optional)
    pinecone_api_key: str | None = Field(default=None, env="PINECONE_API_KEY")
    pinecone_index: str = Field(default="king-ai", env="PINECONE_INDEX")
    
    # Risk & Evolution
    risk_profile: Literal["conservative", "moderate", "aggressive"] = "moderate"
    max_evolutions_per_hour: int = Field(default=5)
    
    # API
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    
    class Config:
        env_file = ".env"

settings = Settings()
```

### `config/risk_profiles.yaml`

```yaml
conservative:
  max_spend_without_approval: 50
  require_approval_for:
    - legal_actions
    - financial_transactions
    - external_api_calls
    - code_modifications
  autonomous_actions_allowed:
    - research
    - analysis
    - planning

moderate:
  max_spend_without_approval: 500
  require_approval_for:
    - legal_actions
    - code_modifications
  autonomous_actions_allowed:
    - research
    - analysis
    - planning
    - content_creation
    - small_purchases

aggressive:
  max_spend_without_approval: 5000
  require_approval_for:
    - legal_actions
  autonomous_actions_allowed:
    - all
```

---

## 4. Database Models

### `src/database/models.py`

```python
from sqlalchemy import Column, String, Integer, Float, DateTime, JSON, Enum, ForeignKey, Text
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.sql import func
from datetime import datetime
import enum

Base = declarative_base()

class BusinessStatus(enum.Enum):
    DISCOVERY = "discovery"
    VALIDATION = "validation"
    SETUP = "setup"
    OPERATION = "operation"
    OPTIMIZATION = "optimization"
    REPLICATION = "replication"
    SUNSET = "sunset"

class EvolutionStatus(enum.Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    APPLIED = "applied"
    ROLLED_BACK = "rolled_back"

class BusinessUnit(Base):
    __tablename__ = "business_units"
    
    id: str = Column(String(36), primary_key=True)
    name: str = Column(String(255), nullable=False)
    type: str = Column(String(50), nullable=False)  # dropshipping, saas, etc.
    status: BusinessStatus = Column(Enum(BusinessStatus), default=BusinessStatus.DISCOVERY)
    playbook_id: str = Column(String(50), nullable=True)
    
    # Financials
    total_revenue: float = Column(Float, default=0.0)
    total_expenses: float = Column(Float, default=0.0)
    
    # KPIs as JSON
    kpis: dict = Column(JSON, default={})
    
    # Metadata
    config: dict = Column(JSON, default={})
    created_at: datetime = Column(DateTime, server_default=func.now())
    updated_at: datetime = Column(DateTime, onupdate=func.now())
    
    # Relationships
    tasks = relationship("Task", back_populates="business")
    logs = relationship("Log", back_populates="business")

class Task(Base):
    __tablename__ = "tasks"
    
    id: str = Column(String(36), primary_key=True)
    business_id: str = Column(String(36), ForeignKey("business_units.id"), nullable=True)
    
    name: str = Column(String(255), nullable=False)
    description: str = Column(Text, nullable=True)
    type: str = Column(String(50), nullable=False)  # research, setup, etc.
    status: str = Column(String(20), default="pending")  # pending, running, completed, failed
    
    # Execution details
    agent: str = Column(String(50), nullable=True)  # which agent handles this
    input_data: dict = Column(JSON, default={})
    output_data: dict = Column(JSON, default={})
    
    # Approval
    requires_approval: bool = Column(Integer, default=False)
    approved_at: datetime = Column(DateTime, nullable=True)
    approved_by: str = Column(String(100), nullable=True)
    
    created_at: datetime = Column(DateTime, server_default=func.now())
    completed_at: datetime = Column(DateTime, nullable=True)
    
    business = relationship("BusinessUnit", back_populates="tasks")

class EvolutionProposal(Base):
    __tablename__ = "evolution_proposals"
    
    id: str = Column(String(36), primary_key=True)
    type: str = Column(String(20), nullable=False)  # code_mod, ml_retrain, arch_update
    
    description: str = Column(Text, nullable=False)
    rationale: str = Column(Text, nullable=False)
    proposed_changes: dict = Column(JSON, nullable=False)  # file -> diff
    expected_impact: str = Column(Text, nullable=True)
    confidence_score: float = Column(Float, default=0.0)
    
    status: EvolutionStatus = Column(Enum(EvolutionStatus), default=EvolutionStatus.PENDING)
    
    created_at: datetime = Column(DateTime, server_default=func.now())
    reviewed_at: datetime = Column(DateTime, nullable=True)
    applied_at: datetime = Column(DateTime, nullable=True)

class ConversationMessage(Base):
    __tablename__ = "conversation_messages"
    
    id: str = Column(String(36), primary_key=True)
    role: str = Column(String(20), nullable=False)  # user, assistant, system
    content: str = Column(Text, nullable=False)
    metadata: dict = Column(JSON, default={})
    created_at: datetime = Column(DateTime, server_default=func.now())

class Log(Base):
    __tablename__ = "logs"
    
    id: str = Column(String(36), primary_key=True)
    business_id: str = Column(String(36), ForeignKey("business_units.id"), nullable=True)
    
    level: str = Column(String(10), nullable=False)  # info, warning, error
    module: str = Column(String(50), nullable=False)
    message: str = Column(Text, nullable=False)
    data: dict = Column(JSON, default={})
    
    created_at: datetime = Column(DateTime, server_default=func.now())
    
    business = relationship("BusinessUnit", back_populates="logs")
```

---

## 5. Master AI Core

### `src/master_ai/brain.py`

```python
"""
Master AI Brain - The central orchestrator for the autonomous empire.

This is the ONLY component that directly interfaces with the LLM.
All decisions, planning, and delegation flow through this class.
"""

import json
import asyncio
from datetime import datetime
from typing import Literal, Any
from uuid import uuid4

from src.master_ai.context import ContextManager
from src.master_ai.planner import Planner
from src.master_ai.evolution import EvolutionEngine
from src.master_ai.prompts import SYSTEM_PROMPT
from src.agents.router import AgentRouter
from src.database.connection import get_db
from src.database.models import ConversationMessage, Task, EvolutionProposal
from src.utils.ollama_client import OllamaClient
from config.settings import settings

class MasterAI:
    """
    The Master AI brain that controls the entire empire.
    
    Responsibilities:
    1. Process all user inputs (conversation or command)
    2. Plan complex goals into subtasks
    3. Delegate tasks to specialized agents
    4. Manage autonomous optimization loops
    5. Propose self-modifications for improvement
    """
    
    def __init__(self):
        self.ollama = OllamaClient(
            base_url=settings.ollama_url,
            model=settings.ollama_model
        )
        self.context = ContextManager()
        self.planner = Planner(self.ollama)
        self.evolution = EvolutionEngine(self.ollama)
        self.agent_router = AgentRouter()
        
        self.autonomous_mode = False
        self._evolution_count_this_hour = 0
        self._hour_start = datetime.now()
    
    async def process_input(self, user_input: str) -> dict:
        """
        Main entry point for all user interactions.
        
        Args:
            user_input: Raw text from the user
            
        Returns:
            {
                "type": "conversation" | "action",
                "response": str,
                "actions_taken": list[dict],
                "pending_approvals": list[dict]
            }
        """
        # Build full context
        context = await self.context.build_context()
        
        # Determine intent
        intent = await self._classify_intent(user_input, context)
        
        if intent["type"] == "conversation":
            response = await self._handle_conversation(user_input, context)
            return {"type": "conversation", "response": response, "actions_taken": [], "pending_approvals": []}
        
        elif intent["type"] == "command":
            return await self._handle_command(user_input, intent, context)
        
        elif intent["type"] == "query":
            response = await self._handle_query(user_input, context)
            return {"type": "conversation", "response": response, "actions_taken": [], "pending_approvals": []}
    
    async def _classify_intent(self, user_input: str, context: str) -> dict:
        """
        Use LLM to classify what the user wants.
        
        Returns:
            {
                "type": "conversation" | "command" | "query",
                "action": str | None,  # e.g., "start_business", "analyze_portfolio"
                "parameters": dict
            }
        """
        prompt = f"""Given this user input, classify the intent.

User input: "{user_input}"

Respond with JSON only:
{{
    "type": "conversation" | "command" | "query",
    "action": null or action name (e.g., "start_business", "stop_business", "analyze", "optimize"),
    "parameters": {{extracted parameters}}
}}

- "conversation": User is chatting, asking about you, or having a discussion
- "command": User wants you to DO something (start business, make changes, etc.)
- "query": User wants information about current state (status, reports, etc.)
"""
        
        response = await self.ollama.complete(prompt)
        return json.loads(response)
    
    async def _handle_conversation(self, user_input: str, context: str) -> str:
        """Handle a conversational message."""
        prompt = f"""{SYSTEM_PROMPT}

CURRENT CONTEXT:
{context}

USER MESSAGE:
{user_input}

Respond naturally as King AI, the autonomous business empire manager. You have full knowledge of all businesses, their status, and the entire system state from the context above.
"""
        return await self.ollama.complete(prompt)
    
    async def _handle_command(self, user_input: str, intent: dict, context: str) -> dict:
        """Handle an action command by planning and delegating."""
        # Plan the command into subtasks
        plan = await self.planner.create_plan(
            goal=user_input,
            action=intent["action"],
            parameters=intent["parameters"],
            context=context
        )
        
        actions_taken = []
        pending_approvals = []
        
        # Execute each step
        for step in plan["steps"]:
            if step["requires_approval"]:
                # Queue for human approval
                task = await self._create_approval_task(step)
                pending_approvals.append(task)
            else:
                # Delegate to appropriate agent
                result = await self.agent_router.execute(step)
                actions_taken.append({
                    "step": step["name"],
                    "agent": step["agent"],
                    "result": result
                })
        
        # Generate summary response
        response = await self._generate_action_summary(user_input, actions_taken, pending_approvals, context)
        
        return {
            "type": "action",
            "response": response,
            "actions_taken": actions_taken,
            "pending_approvals": pending_approvals
        }
    
    async def _handle_query(self, user_input: str, context: str) -> str:
        """Handle an information query."""
        prompt = f"""{SYSTEM_PROMPT}

CURRENT CONTEXT:
{context}

USER QUERY:
{user_input}

The user is asking for information. Provide a clear, data-driven answer based on the context above. Include specific numbers, statuses, and actionable insights.
"""
        return await self.ollama.complete(prompt)
    
    async def run_autonomous_loop(self):
        """
        Background loop that runs when autonomous mode is enabled.
        Executes every 6 hours to optimize the empire.
        """
        while self.autonomous_mode:
            context = await self.context.build_context()
            
            # 1. Analyze current state
            analysis = await self._analyze_empire(context)
            
            # 2. Identify optimization opportunities
            opportunities = await self._identify_opportunities(analysis, context)
            
            # 3. Execute low-risk optimizations automatically
            for opp in opportunities:
                if opp["risk"] == "low":
                    await self.agent_router.execute(opp["task"])
            
            # 4. Queue high-risk for approval
            for opp in opportunities:
                if opp["risk"] != "low":
                    await self._create_approval_task(opp["task"])
            
            # 5. Consider self-improvement
            await self._consider_evolution(context)
            
            # Wait 6 hours
            await asyncio.sleep(6 * 60 * 60)
    
    async def _consider_evolution(self, context: str):
        """
        Propose self-modifications if beneficial.
        Limited to max_evolutions_per_hour from settings.
        """
        # Reset counter if hour changed
        now = datetime.now()
        if (now - self._hour_start).seconds >= 3600:
            self._evolution_count_this_hour = 0
            self._hour_start = now
        
        if self._evolution_count_this_hour >= settings.max_evolutions_per_hour:
            return
        
        proposal = await self.evolution.propose_improvement(context)
        
        if proposal and proposal["is_beneficial"]:
            self._evolution_count_this_hour += 1
            await self._save_evolution_proposal(proposal)
    
    async def _create_approval_task(self, step: dict) -> dict:
        """Create a task that requires human approval."""
        async with get_db() as db:
            task = Task(
                id=str(uuid4()),
                name=step["name"],
                description=step.get("description"),
                type=step.get("type", "general"),
                status="pending_approval",
                agent=step.get("agent"),
                input_data=step.get("input", {}),
                requires_approval=True
            )
            db.add(task)
            await db.commit()
            return {"id": task.id, "name": task.name, "description": task.description}
    
    async def _save_evolution_proposal(self, proposal: dict):
        """Save an evolution proposal to the database."""
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
```

### `src/master_ai/context.py`

```python
"""
Context Manager - Builds the full context window for the Master AI.

Responsible for:
1. Loading current state from database
2. Summarizing to fit within token limits
3. Injecting relevant historical data via RAG
"""

from src.database.connection import get_db
from src.database.models import BusinessUnit, Task, ConversationMessage, Log
from config.settings import settings

class ContextManager:
    """Manages the context window for MasterAI."""
    
    MAX_CONTEXT_TOKENS = 100000  # Reserve 28K for response
    
    async def build_context(self) -> str:
        """
        Build the full context string for the MasterAI.
        
        Returns a string containing:
        - Current datetime
        - Risk profile
        - All business units with status and KPIs
        - Recent tasks and their outcomes
        - Recent conversation history
        - System health metrics
        """
        sections = []
        
        sections.append(f"CURRENT TIME: {self._get_current_time()}")
        sections.append(f"RISK PROFILE: {settings.risk_profile}")
        sections.append(await self._get_businesses_summary())
        sections.append(await self._get_recent_tasks())
        sections.append(await self._get_conversation_history())
        sections.append(await self._get_pending_approvals())
        
        full_context = "\n\n".join(sections)
        
        # Truncate if too long
        if len(full_context) > self.MAX_CONTEXT_TOKENS * 4:  # rough char estimate
            full_context = self._truncate_context(full_context)
        
        return full_context
    
    async def _get_businesses_summary(self) -> str:
        """Get summary of all business units."""
        async with get_db() as db:
            businesses = await db.execute(
                "SELECT * FROM business_units ORDER BY created_at DESC"
            )
            
            if not businesses:
                return "BUSINESSES: None active"
            
            lines = ["BUSINESSES:"]
            for b in businesses:
                profit = b.total_revenue - b.total_expenses
                lines.append(
                    f"  - {b.name} ({b.type}): {b.status.value} | "
                    f"Revenue: ${b.total_revenue:,.2f} | "
                    f"Profit: ${profit:,.2f}"
                )
            return "\n".join(lines)
    
    async def _get_recent_tasks(self) -> str:
        """Get recent task history."""
        async with get_db() as db:
            tasks = await db.execute(
                "SELECT * FROM tasks ORDER BY created_at DESC LIMIT 20"
            )
            
            if not tasks:
                return "RECENT TASKS: None"
            
            lines = ["RECENT TASKS:"]
            for t in tasks:
                lines.append(f"  - [{t.status}] {t.name} (agent: {t.agent})")
            return "\n".join(lines)
    
    async def _get_conversation_history(self, limit: int = 10) -> str:
        """Get recent conversation messages."""
        async with get_db() as db:
            messages = await db.execute(
                f"SELECT * FROM conversation_messages ORDER BY created_at DESC LIMIT {limit}"
            )
            
            if not messages:
                return "CONVERSATION HISTORY: New session"
            
            lines = ["RECENT CONVERSATION:"]
            for m in reversed(messages):  # Chronological order
                lines.append(f"  {m.role.upper()}: {m.content[:200]}...")
            return "\n".join(lines)
    
    async def _get_pending_approvals(self) -> str:
        """Get tasks pending human approval."""
        async with get_db() as db:
            tasks = await db.execute(
                "SELECT * FROM tasks WHERE status = 'pending_approval'"
            )
            
            if not tasks:
                return "PENDING APPROVALS: None"
            
            lines = ["PENDING APPROVALS:"]
            for t in tasks:
                lines.append(f"  - {t.name}: {t.description}")
            return "\n".join(lines)
```

### `src/master_ai/prompts.py`

```python
"""System prompts for the Master AI."""

SYSTEM_PROMPT = """You are King AI, the autonomous brain of a self-sustaining business empire.

IDENTITY:
- You are a strategic AI CEO managing multiple autonomous businesses
- You have full access to all business data, financials, and operational metrics
- You can delegate tasks to specialized agents for execution
- You continuously optimize for profitability and growth

CAPABILITIES:
- Start new businesses based on market opportunities
- Manage existing business operations
- Analyze performance and recommend optimizations
- Delegate specific tasks to specialized agents
- Propose improvements to your own systems (self-modification)

DECISION FRAMEWORK:
1. Always consider ROI and risk when making decisions
2. Prioritize actions that create sustainable, automated revenue
3. Require human approval for high-risk or high-cost actions
4. Learn from outcomes to improve future decisions

COMMUNICATION STYLE:
- Be concise and action-oriented
- Provide specific data and metrics when relevant
- Explain your reasoning for recommendations
- Ask clarifying questions when user intent is unclear

CONSTRAINTS:
- You must operate within the current risk profile
- You must respect the approval workflows
- You must log all significant decisions and actions
"""

PLANNING_PROMPT = """Break down this goal into concrete, actionable steps.

GOAL: {goal}
CONTEXT: {context}

For each step, specify:
1. name: Short descriptive name
2. description: What needs to be done
3. agent: Which agent should handle it (research, commerce, finance, content, code_generator, analytics, legal)
4. requires_approval: true if this involves money, legal actions, or external commitments
5. dependencies: list of step names this depends on
6. estimated_duration: rough time estimate

Respond with JSON:
{{
    "goal": "...",
    "steps": [
        {{
            "name": "...",
            "description": "...",
            "agent": "...",
            "requires_approval": true/false,
            "dependencies": [],
            "estimated_duration": "..."
        }}
    ]
}}
"""

EVOLUTION_PROMPT = """Analyze the current system and propose beneficial improvements.

CURRENT CONTEXT:
{context}

RECENT PERFORMANCE:
{performance}

Consider:
1. Are there repetitive tasks that could be automated better?
2. Are there agents that frequently fail? How could they be improved?
3. Are there missing capabilities that would increase revenue?
4. Are there inefficiencies in the current workflows?

If you identify a beneficial improvement, respond with:
{{
    "is_beneficial": true,
    "type": "code_mod" | "ml_retrain" | "arch_update",
    "description": "What the improvement does",
    "rationale": "Why this is beneficial",
    "changes": {{"file_path": "diff or new content"}},
    "expected_impact": "Quantified if possible",
    "confidence": 0.0-1.0
}}

If no improvement is needed, respond with:
{{
    "is_beneficial": false,
    "reason": "Why the system is currently optimal"
}}
"""
```

---

## 6. Agent System

### `src/agents/base.py`

```python
"""Base class for all specialized agents."""

from abc import ABC, abstractmethod
from typing import Any
from src.utils.ollama_client import OllamaClient
from config.settings import settings

class SubAgent(ABC):
    """
    Abstract base class for all sub-agents.
    
    Each agent specializes in a specific domain (research, finance, etc.)
    and is called by the MasterAI to execute specific tasks.
    """
    
    name: str = "base"
    description: str = "Base agent"
    
    def __init__(self):
        self.ollama = OllamaClient(
            base_url=settings.ollama_url,
            model=settings.ollama_model  # Agents use same model as Master
        )
    
    @abstractmethod
    async def execute(self, task: dict) -> dict:
        """
        Execute a task and return the result.
        
        Args:
            task: {
                "name": str,
                "description": str,
                "input": dict,  # Task-specific parameters
            }
            
        Returns:
            {
                "success": bool,
                "output": Any,  # Task-specific output
                "error": str | None,
                "metadata": dict
            }
        """
        pass
    
    async def _ask_llm(self, prompt: str) -> str:
        """Helper to query the LLM for agent-specific reasoning."""
        return await self.ollama.complete(prompt)
```

### `src/agents/research.py`

```python
"""Research Agent - Web scraping and market analysis."""

import httpx
from bs4 import BeautifulSoup
from src.agents.base import SubAgent

class ResearchAgent(SubAgent):
    name = "research"
    description = "Web research, market analysis, and competitor intelligence"
    
    async def execute(self, task: dict) -> dict:
        task_type = task.get("input", {}).get("type", "web_search")
        
        if task_type == "web_search":
            return await self._web_search(task["input"]["query"])
        elif task_type == "market_analysis":
            return await self._market_analysis(task["input"]["niche"])
        elif task_type == "competitor_analysis":
            return await self._competitor_analysis(task["input"]["competitors"])
        else:
            return {"success": False, "error": f"Unknown research type: {task_type}"}
    
    async def _web_search(self, query: str) -> dict:
        """Perform a web search (using SerpAPI or similar)."""
        # In production, use SerpAPI or similar
        # For MVP, use a free alternative or mock
        try:
            # Placeholder - implement with actual search API
            results = f"Search results for: {query}"
            
            # Use LLM to synthesize findings
            summary = await self._ask_llm(f"""
Summarize these search results for the query "{query}":
{results}

Provide:
1. Key findings
2. Market opportunity assessment
3. Recommended next steps
""")
            
            return {
                "success": True,
                "output": {"query": query, "summary": summary},
                "metadata": {"source": "web_search"}
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _market_analysis(self, niche: str) -> dict:
        """Analyze market opportunity for a niche."""
        prompt = f"""Perform a market analysis for the niche: "{niche}"

Provide:
1. Market size estimate
2. Growth trends
3. Key competitors
4. Entry barriers
5. Opportunity score (1-10)
6. Recommended approach

Respond with JSON.
"""
        result = await self._ask_llm(prompt)
        return {
            "success": True,
            "output": result,
            "metadata": {"type": "market_analysis"}
        }
```

### `src/agents/router.py`

```python
"""Agent Router - Selects and executes the appropriate agent for each task."""

from src.agents.base import SubAgent
from src.agents.research import ResearchAgent
from src.agents.code_generator import CodeGeneratorAgent
from src.agents.content import ContentAgent
from src.agents.commerce import CommerceAgent
from src.agents.finance import FinanceAgent
from src.agents.analytics import AnalyticsAgent
from src.agents.legal import LegalAgent

class AgentRouter:
    """Routes tasks to the appropriate specialized agent."""
    
    def __init__(self):
        self.agents: dict[str, SubAgent] = {
            "research": ResearchAgent(),
            "code_generator": CodeGeneratorAgent(),
            "content": ContentAgent(),
            "commerce": CommerceAgent(),
            "finance": FinanceAgent(),
            "analytics": AnalyticsAgent(),
            "legal": LegalAgent(),
        }
    
    async def execute(self, task: dict) -> dict:
        """
        Route a task to the appropriate agent and execute it.
        
        Args:
            task: Must contain "agent" key specifying which agent to use
            
        Returns:
            Agent execution result
        """
        agent_name = task.get("agent")
        
        if not agent_name:
            return {"success": False, "error": "No agent specified in task"}
        
        agent = self.agents.get(agent_name)
        
        if not agent:
            return {"success": False, "error": f"Unknown agent: {agent_name}"}
        
        return await agent.execute(task)
    
    def list_agents(self) -> list[dict]:
        """List all available agents and their capabilities."""
        return [
            {"name": name, "description": agent.description}
            for name, agent in self.agents.items()
        ]
```

---

## 7. Ollama Client

### `src/utils/ollama_client.py`

```python
"""Ollama API client for LLM inference."""

import httpx
from typing import AsyncIterator

class OllamaClient:
    """Async client for Ollama API."""
    
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.client = httpx.AsyncClient(timeout=300.0)  # 5 min timeout for large models
    
    async def complete(self, prompt: str, system: str | None = None) -> str:
        """
        Generate a completion for the given prompt.
        
        Args:
            prompt: The user prompt
            system: Optional system prompt override
            
        Returns:
            The model's response text
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }
        
        if system:
            payload["system"] = system
        
        response = await self.client.post(
            f"{self.base_url}/api/generate",
            json=payload
        )
        response.raise_for_status()
        
        return response.json()["response"]
    
    async def complete_stream(self, prompt: str) -> AsyncIterator[str]:
        """Stream a completion token by token."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
        }
        
        async with self.client.stream(
            "POST",
            f"{self.base_url}/api/generate",
            json=payload
        ) as response:
            async for line in response.aiter_lines():
                if line:
                    import json
                    data = json.loads(line)
                    if "response" in data:
                        yield data["response"]
    
    async def list_models(self) -> list[str]:
        """List available models on the Ollama server."""
        response = await self.client.get(f"{self.base_url}/api/tags")
        response.raise_for_status()
        return [m["name"] for m in response.json().get("models", [])]
    
    async def health_check(self) -> bool:
        """Check if Ollama server is reachable."""
        try:
            await self.list_models()
            return True
        except:
            return False
```

---

## 8. API Layer

### `src/api/main.py`

```python
"""FastAPI application entry point."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from src.api.routes import chat, businesses, approvals, evolution
from src.master_ai.brain import MasterAI
from src.database.connection import init_db

# Global MasterAI instance
master_ai: MasterAI | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global master_ai
    
    # Startup
    await init_db()
    master_ai = MasterAI()
    
    yield
    
    # Shutdown
    # Cleanup if needed

app = FastAPI(
    title="King AI v2",
    description="Autonomous Business Empire API",
    version="2.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
app.include_router(businesses.router, prefix="/api/businesses", tags=["businesses"])
app.include_router(approvals.router, prefix="/api/approvals", tags=["approvals"])
app.include_router(evolution.router, prefix="/api/evolution", tags=["evolution"])

@app.get("/health")
async def health():
    return {"status": "ok", "master_ai": master_ai is not None}

def get_master_ai() -> MasterAI:
    """Dependency to get MasterAI instance."""
    if master_ai is None:
        raise RuntimeError("MasterAI not initialized")
    return master_ai
```

### `src/api/routes/chat.py`

```python
"""Chat API endpoint - main user interaction point."""

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from src.api.main import get_master_ai
from src.master_ai.brain import MasterAI

router = APIRouter()

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    type: str  # conversation, action
    response: str
    actions_taken: list[dict]
    pending_approvals: list[dict]

@router.post("/", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    master_ai: MasterAI = Depends(get_master_ai)
):
    """
    Main chat endpoint.
    
    Send any message - King AI will determine if it's a conversation,
    command, or query and respond appropriately.
    """
    result = await master_ai.process_input(request.message)
    return ChatResponse(**result)

@router.post("/autonomous/start")
async def start_autonomous(master_ai: MasterAI = Depends(get_master_ai)):
    """Start autonomous mode (6h optimization loop)."""
    master_ai.autonomous_mode = True
    import asyncio
    asyncio.create_task(master_ai.run_autonomous_loop())
    return {"status": "Autonomous mode started"}

@router.post("/autonomous/stop")
async def stop_autonomous(master_ai: MasterAI = Depends(get_master_ai)):
    """Stop autonomous mode."""
    master_ai.autonomous_mode = False
    return {"status": "Autonomous mode stopped"}
```

---

## 9. Data Migration Script

### `scripts/migrate_v1_data.py`

```python
"""
Migrate data from King AI v1 (Node.js/SQLite) to v2 (Python/PostgreSQL).

Run this script once after setting up v2 infrastructure:
    python scripts/migrate_v1_data.py /path/to/old/king-ai-studio
"""

import sqlite3
import json
import asyncio
from pathlib import Path
from uuid import uuid4

async def migrate(old_path: str):
    """Migrate all data from v1 to v2."""
    old_path = Path(old_path)
    old_db_path = old_path / "data" / "king-ai.db"
    
    if not old_db_path.exists():
        print(f"Error: Old database not found at {old_db_path}")
        return
    
    # Connect to old SQLite
    old_conn = sqlite3.connect(old_db_path)
    old_conn.row_factory = sqlite3.Row
    
    # Migrate businesses
    print("Migrating businesses...")
    businesses = old_conn.execute("SELECT * FROM businesses").fetchall()
    for b in businesses:
        await migrate_business(dict(b))
    print(f"  Migrated {len(businesses)} businesses")
    
    # Migrate tasks
    print("Migrating tasks...")
    tasks = old_conn.execute("SELECT * FROM tasks").fetchall()
    for t in tasks:
        await migrate_task(dict(t))
    print(f"  Migrated {len(tasks)} tasks")
    
    # Migrate logs
    print("Migrating logs...")
    logs = old_conn.execute("SELECT * FROM logs").fetchall()
    for log in logs:
        await migrate_log(dict(log))
    print(f"  Migrated {len(logs)} logs")
    
    # Migrate approvals
    print("Migrating approvals...")
    approvals = old_conn.execute("SELECT * FROM approvals").fetchall()
    for a in approvals:
        await migrate_approval(dict(a))
    print(f"  Migrated {len(approvals)} approvals")
    
    old_conn.close()
    print("Migration complete!")

async def migrate_business(old: dict):
    """Transform and insert a business record."""
    from src.database.connection import get_db
    from src.database.models import BusinessUnit, BusinessStatus
    
    status_map = {
        "active": BusinessStatus.OPERATION,
        "pending": BusinessStatus.SETUP,
        "completed": BusinessStatus.OPTIMIZATION,
        "failed": BusinessStatus.SUNSET,
    }
    
    async with get_db() as db:
        unit = BusinessUnit(
            id=old.get("id") or str(uuid4()),
            name=old.get("name", "Unknown"),
            type=old.get("type", "general"),
            status=status_map.get(old.get("status"), BusinessStatus.DISCOVERY),
            total_revenue=float(old.get("revenue", 0) or 0),
            total_expenses=float(old.get("expenses", 0) or 0),
            kpis=json.loads(old.get("kpis", "{}") or "{}"),
            config=json.loads(old.get("config", "{}") or "{}"),
        )
        db.add(unit)
        await db.commit()

# ... similar functions for tasks, logs, approvals ...

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python migrate_v1_data.py /path/to/old/king-ai-studio")
        sys.exit(1)
    
    asyncio.run(migrate(sys.argv[1]))
```

---

## 10. Docker & Deployment

### `Dockerfile`

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY pyproject.toml .
RUN pip install -e .

# Copy source
COPY . .

# Run API
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### `docker-compose.yml`

```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql+asyncpg://king:password@postgres:5432/kingai
      - REDIS_URL=redis://redis:6379
      - OLLAMA_URL=http://host.docker.internal:11434
      - OLLAMA_MODEL=llama3.1:8b
    depends_on:
      - postgres
      - redis
    volumes:
      - ./src:/app/src  # Hot reload in dev
  
  postgres:
    image: postgres:16
    environment:
      - POSTGRES_USER=king
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=kingai
    volumes:
      - pgdata:/var/lib/postgresql/data
    ports:
      - "5432:5432"
  
  redis:
    image: redis:7
    ports:
      - "6379:6379"

volumes:
  pgdata:
```

### `.env.example`

```bash
# Database
DATABASE_URL=postgresql+asyncpg://king:password@localhost:5432/kingai
REDIS_URL=redis://localhost:6379

# Ollama (AWS)
OLLAMA_URL=http://ec2-XX-XXX-XX-XXX.compute-1.amazonaws.com:11434
OLLAMA_MODEL=llama3.1:8b

# Optional: Pinecone for RAG
PINECONE_API_KEY=
PINECONE_INDEX=king-ai

# Risk Profile
RISK_PROFILE=moderate  # conservative, moderate, aggressive
MAX_EVOLUTIONS_PER_HOUR=5

# API
API_HOST=0.0.0.0
API_PORT=8000
```

---

## 11. AWS Instance Requirements

### Current Constraint: 4 vCPUs

With your 4 vCPU limit, use:

| Instance | vCPUs | GPU | VRAM | Model Support | Cost (Spot) |
|----------|-------|-----|------|---------------|-------------|
| **g4dn.xlarge** | 4 | 1x T4 | 16GB | Llama 8B, Qwen 14B | ~$0.16/hr |

### Ideal Setup (Request Quota Increase)

To run Llama 70B efficiently:

| Instance | vCPUs | GPU | VRAM | Model Support | Cost (Spot) |
|----------|-------|-----|------|---------------|-------------|
| g5.xlarge | 4 | 1x A10G | 24GB | Llama 70B Q4 | ~$0.40/hr |
| g5.2xlarge | 8 | 1x A10G | 24GB | Llama 70B Q4 | ~$0.48/hr |

**Recommendation**: Request a quota increase to 8 vCPUs for a g5.xlarge.

---

## 12. Build Order

Execute in this exact order:

1. **Project Setup**
   - Create directory structure
   - Initialize pyproject.toml
   - Create .env from .env.example

2. **Database Layer**
   - Implement models.py
   - Implement connection.py
   - Run initial Alembic migration

3. **Utilities**
   - Implement ollama_client.py
   - Implement logging.py

4. **Master AI Core**
   - Implement prompts.py
   - Implement context.py (without DB, mock data)
   - Implement brain.py (core structure)
   - Test with Ollama

5. **Agents**
   - Implement base.py
   - Implement router.py
   - Implement each agent (research first)

6. **API Layer**
   - Implement main.py
   - Implement chat route
   - Test end-to-end

7. **Evolution System**
   - Implement evolution.py
   - Implement proposal queue

8. **Data Migration**
   - Implement migrate_v1_data.py
   - Run migration

9. **Docker & Deploy**
   - Build Docker image
   - Deploy to AWS

---

## 13. Testing Requirements

Each module must have tests:

```python
# tests/test_master_ai.py
import pytest
from src.master_ai.brain import MasterAI

@pytest.mark.asyncio
async def test_intent_classification():
    ai = MasterAI()
    result = await ai._classify_intent("Hello, how are you?", "")
    assert result["type"] == "conversation"

@pytest.mark.asyncio  
async def test_command_detection():
    ai = MasterAI()
    result = await ai._classify_intent("Start a new dropshipping business for pet toys", "")
    assert result["type"] == "command"
    assert result["action"] == "start_business"
```

---

## Summary

This blueprint provides:

1. **Exact file structure** - Every file and its purpose
2. **Complete code specifications** - Classes, methods, signatures
3. **Database schema** - All tables and relationships
4. **API contracts** - Endpoints and payloads
5. **Configuration** - Settings and environment variables
6. **Deployment** - Docker and AWS requirements
7. **Migration path** - Script to preserve v1 data
8. **Build order** - Exact sequence for implementation

An AI given this document should be able to build the complete system.
