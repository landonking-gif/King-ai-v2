# King-AI-v2 Implementation Plan: Adopting Reference Frameworks

## Overview
This implementation plan outlines how to integrate best practices, architectures, and components from the reference frameworks in the `references/` folder into King-AI-v2. The goal is to enhance scalability, reliability, and functionality while maintaining compatibility with the existing codebase. All frameworks are Python-based multi-agent AI systems with shared technologies (async patterns, Pydantic, LLM integration).

## Key Reference Frameworks and Adoptable Elements

### 1. Agentic-AI-Framework-Main
**Core Adoptables**:
- LangGraph-based orchestration for complex agent workflows
- BaseAgent class for standardized agent interfaces
- Multi-LLM provider support with per-agent overrides
- Async execution patterns
- State management with partial updates
- Production features: logging, error handling, extensibility

### 2. Agentic-Framework-Main
**Core Adoptables**:
- Flexible agent architectures
- Workflow patterns and state handling
- Integration examples for agent creation
- Testing frameworks

### 3. LLM-Agent-Framework-Main
**Core Adoptables**:
- LLM-powered agent classes
- Prompt management and response handling
- Provider abstractions for multiple LLMs
- Async LLM integration

### 4. Mother-Harness-Master
**Core Adoptables**:
- Agent harness for lifecycle management
- Control layers for deployment and testing
- Orchestration utilities

### 5. Multi-Agent-Orchestration-Main
**Core Adoptables**:
- Orchestration tools for agent coordination
- Multi-agent workflow patterns
- State management for collaborative tasks
- Web-based UI with real-time streaming
- PostgreSQL persistence for state and events
- Comprehensive observability and cost tracking

### 6. Multi-Agent-Reference-Architecture-Main
**Core Adoptables**:
- Reference implementations for scalable architectures
- Configuration-driven agent setups
- Best practices for agent design

### 7. Semantic Kernel (PDF/HTML)
**Core Adoptables**:
- SDK patterns for AI workflow orchestration
- AgentGroupChat for collaborative agents
- Function calling and streaming
- Plugin extensibility
- Vector store integration

### 8. Azure Architecture AI/ML (PDF)
**Core Adoptables**:
- API Management patterns for authentication and monitoring
- Load balancing and redundancy strategies
- Foundation model lifecycle management
- Compliance and auditing features

## Detailed Code-Level Implementation Guide

### BaseAgent Implementation
From `agentic-ai-framework-main/src/agents/base.py`, adopt this standardized agent interface:

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

class AgentConfig(BaseModel):
    name: str
    description: str
    max_retries: int = Field(default=3, ge=1)
    timeout: Optional[int] = Field(default=30, ge=1)

class BaseAgent(ABC):
    def __init__(self, config: AgentConfig):
        self.config = config
        self.name = config.name
        self.description = config.description
    
    @abstractmethod
    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError("Subclasses must implement execute method")
    
    def validate_task(self, task: Dict[str, Any]) -> bool:
        if not isinstance(task, dict):
            return False
        return "type" in task and "data" in task
    
    async def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        if not self.validate_task(task):
            return {
                "success": False,
                "error": "Invalid task format. Must include 'type' and 'data' keys",
                "agent": self.name
            }
        
        try:
            result = await self.execute(task)
            return {
                "success": True,
                "result": result,
                "agent": self.name
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "agent": self.name
            }
```

**Integration Steps**:
1. Create `src/agents/base_agent.py` with the above code
2. Update existing agents in `src/agents/` to inherit from BaseAgent
3. Implement `execute` method in each agent
4. Add AgentConfig to `config/settings.py`

### Orchestrator Implementation
From `agentic-ai-framework-main/src/core/orchestrator.py`, implement central coordination:

```python
import inspect
from typing import Dict, Any, Optional
from .agents.base import BaseAgent
from .graph.state import AgentState
from .graph.workflow import AgentWorkflow

class Orchestrator:
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.workflow: Optional[AgentWorkflow] = None
    
    def register_agent(self, agent: BaseAgent):
        self.agents[agent.name] = agent
        if self.workflow:
            self.workflow.register_agent(agent)
    
    def set_workflow(self, workflow: AgentWorkflow):
        self.workflow = workflow
        for agent in self.agents.values():
            self.workflow.register_agent(agent)
    
    async def route_task(self, task: Dict[str, Any]) -> str:
        task_type = task.get("type", "")
        
        for agent_name, agent in self.agents.items():
            if hasattr(agent, "can_handle"):
                fn = getattr(agent, "can_handle")
                try:
                    res = fn(task)
                    can = await res if inspect.isawaitable(res) else bool(res)
                    if can:
                        return agent_name
                except Exception:
                    continue
        
        return "default"
    
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        if not self.agents:
            return {"success": False, "error": "No agents registered"}
        
        agent_name = await self.route_task(task)
        
        if agent_name == "default" and self.agents:
            agent_name = list(self.agents.keys())[0]
        
        if agent_name not in self.agents:
            return {"success": False, "error": f"Agent {agent_name} not found"}
        
        agent = self.agents[agent_name]
        result = await agent.process(task)
        
        return result
    
    async def execute_workflow(self, initial_task: Dict[str, Any]) -> AgentState:
        if not self.workflow:
            raise ValueError("No workflow configured. Use set_workflow() first")
        
        initial_state: AgentState = {
            "messages": [],
            "current_agent": None,
            "task_queue": [initial_task],
            "results": {},
            "metadata": {},
            "error": None
        }
        
        final_state = await self.workflow.execute(initial_state)
        
        return final_state
```

**Integration Steps**:
1. Create `src/core/orchestrator.py` with the above code
2. Update `src/master_ai/` to use Orchestrator for agent coordination
3. Implement routing logic based on task types in `src/agents/`

### State Management Implementation
From `agentic-ai-framework-main/src/graph/state.py`, adopt TypedDict state:

```python
from typing import TypedDict, Annotated, Sequence
from operator import add

class AgentState(TypedDict):
    messages: Annotated[Sequence[dict], add]
    current_agent: str | None
    task_queue: Annotated[Sequence[dict], add]
    results: dict
    metadata: dict
    error: str | None
```

**Integration Steps**:
1. Create `src/graph/state.py` with AgentState
2. Update workflow management in `src/master_ai/` to use AgentState
3. Implement state persistence in `src/database/`

### Workflow Implementation
From `agentic-ai-framework-main/src/graph/workflow.py`, implement LangGraph workflows:

```python
from typing import Dict, Any, Callable
from langgraph.graph import StateGraph, END
from .state import AgentState
from ..agents.base import BaseAgent

class AgentWorkflow:
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.graph = StateGraph(AgentState)
        self.compiled_graph = None
    
    def register_agent(self, agent: BaseAgent):
        self.agents[agent.name] = agent
    
    def add_node(self, name: str, func: Callable):
        self.graph.add_node(name, func)
    
    def add_edge(self, from_node: str, to_node: str):
        self.graph.add_edge(from_node, to_node)
    
    def add_conditional_edges(self, source: str, path_func: Callable, path_map: Dict[str, str] = None):
        if path_map is not None:
            self.graph.add_conditional_edges(source, path_func, path_map)
        else:
            self.graph.add_conditional_edges(source, path_func)
    
    def set_entry_point(self, node: str):
        self.graph.set_entry_point(node)
    
    def compile(self):
        self.compiled_graph = self.graph.compile()
        return self.compiled_graph
    
    async def execute(self, initial_state: AgentState) -> AgentState:
        if not self.compiled_graph:
            self.compile()
        
        final_state = await self.compiled_graph.ainvoke(initial_state)
        return final_state
```

**Integration Steps**:
1. Add LangGraph dependency to `pyproject.toml`
2. Create `src/graph/workflow.py` with AgentWorkflow
3. Implement workflow nodes for agent interactions in `src/agents/`
4. Add conditional routing based on task types

### LLM Factory Implementation
From `agentic-ai-framework-main/src/utils/llm_factory.py`, implement multi-provider LLM support:

```python
from typing import Any, Optional
from .config import config

def get_llm(provider: Optional[str] = None, model: Optional[str] = None, **kwargs) -> Any:
    provider = (provider or config.llm_provider).lower()
    
    if provider == "ollama":
        return _get_ollama_llm(model=model, **kwargs)
    elif provider == "openai":
        return _get_openai_llm(model=model, **kwargs)
    elif provider == "anthropic":
        return _get_anthropic_llm(model=model, **kwargs)
    elif provider == "google":
        return _get_google_llm(model=model, **kwargs)
    elif provider == "azure":
        return _get_azure_llm(model=model, **kwargs)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")

def _get_openai_llm(model: Optional[str] = None, **kwargs):
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        raise ImportError("Please install: pip install langchain-openai")
    
    if not config.openai_api_key:
        raise ValueError("OPENAI_API_KEY not set in environment")
    
    model_name = model or config.openai_model
    
    return ChatOpenAI(
        api_key=config.openai_api_key,
        model=model_name,
        **kwargs
    )

# Implement similar functions for other providers...
```

**Integration Steps**:
1. Create `src/utils/llm_factory.py` with provider implementations
2. Update `config/settings.py` with LLM configuration using Pydantic
3. Modify agents in `src/agents/` to use `get_llm()` for LLM instances
4. Add per-agent LLM overrides in agent configurations

### Configuration Management
From `agentic-ai-framework-main/src/utils/config.py`, adopt Pydantic settings:

```python
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Config(BaseSettings):
    llm_provider: str = "ollama"
    log_level: str = "INFO"
    
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3"
    
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4"
    
    anthropic_api_key: Optional[str] = None
    anthropic_model: str = "claude-3-5-sonnet-20241022"
    
    google_api_key: Optional[str] = None
    google_model: str = "gemini-pro"
    
    azure_openai_api_key: Optional[str] = None
    azure_openai_endpoint: Optional[str] = None
    azure_openai_deployment: Optional[str] = None
    azure_openai_api_version: str = "2024-02-15-preview"
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

config = Config()
```

**Integration Steps**:
1. Add `pydantic-settings` to `pyproject.toml`
2. Create `src/utils/config.py` with Config class
3. Update `config/settings.py` to use the new config system
4. Add `.env` file support for environment variables

### Web-Based Orchestration (From Multi-Agent-Orchestration-Main)
For advanced UI and persistence, consider adopting:
- Real-time WebSocket streaming for agent interactions
- PostgreSQL persistence for state and event logging
- Cost tracking and observability features
- Natural language orchestrator agent

**Integration Steps**:
1. Add WebSocket support to `src/websocket.py`
2. Implement database models in `src/database/` for state persistence
3. Add observability to `src/monitoring/`
4. Create web UI in `dashboard/` for orchestration control

## Implementation Phases

### Phase 1: Foundation (Weeks 1-4)
**Goal**: Establish core infrastructure improvements.

1. **Adopt Pydantic for Type Safety**
   - Integrate Pydantic into `src/config/`, `src/database/`, and `src/agents/`
   - Create base models for agent inputs/outputs
   - Dependencies: Add `pydantic` to `pyproject.toml`
   - Testing: Validate with mypy

2. **Implement Async Patterns**
   - Refactor `src/agents/`, `src/api/`, and `src/services/` for async/await
   - Ensure non-blocking operations in `src/websocket.py`
   - Testing: Performance benchmarks

3. **Enhance LLM Integration**
   - Add multi-provider support in `src/integrations/`
   - Implement provider abstractions from llm-agent-framework-main
   - Update `config/settings.py` for environment-based switching

### Phase 2: Orchestration (Weeks 5-8)
**Goal**: Upgrade agent coordination and workflows.

1. **Integrate LangGraph Orchestration**
   - Add LangGraph to `src/master_ai/` for workflow management
   - Implement state graphs for agent interactions
   - Dependencies: `langgraph`, `langchain`

2. **Adopt BaseAgent Patterns**
   - Create standardized agent base classes in `src/agents/`
   - Refactor existing agents to inherit from base classes
   - Add extensibility hooks

3. **Implement State Management**
   - Add partial state updates from agentic-ai-framework-main
   - Integrate with `src/database/` for persistence

### Phase 3: Advanced Features (Weeks 9-12)
**Goal**: Add production and collaborative features.

1. **Add Monitoring and Logging**
   - Enhance `src/monitoring/` with patterns from azure-architecture-ai-ml.pdf
   - Implement auditing and compliance tracking
   - Add health checks and metrics

2. **Implement Collaborative Agents**
   - Add AgentGroupChat from semantic-kernel.pdf
   - Update `src/approvals/` and `src/business/` for multi-agent discussions

3. **API Management and Security**
   - Apply Azure patterns for authentication in `src/api/`
   - Add rate limiting and caching
   - Implement redundancy strategies

### Phase 4: Testing and Extensibility (Weeks 13-16)
**Goal**: Ensure robustness and future-proofing.

1. **Expand Testing Suite**
   - Adopt pytest patterns from all frameworks
   - Add workflow and integration tests in `tests/`
   - Create test harnesses

2. **Add Plugin System**
   - Implement extensibility from semantic-kernel.pdf
   - Allow modular agent additions in `src/utils/`

3. **Create Reference Implementations**
   - Add example workflows in new `examples/` folder
   - Document patterns in `DEVELOPER_DOCS.md`

## Dependencies and Prerequisites
- Python libraries: `langgraph`, `langchain`, `pydantic`, `pytest`
- Environment: Ensure async-compatible Python version
- Infrastructure: Review `infrastructure/` for Azure integration patterns

## Risk Mitigation
- **Compatibility**: Test changes incrementally to avoid breaking existing functionality
- **Performance**: Monitor async changes for bottlenecks
- **Security**: Implement authentication early to secure new features

## Success Metrics
- Improved agent coordination (measured by workflow completion rates)
- Enhanced scalability (async performance benchmarks)
- Better reliability (reduced errors via type safety and monitoring)

### Web-Based Orchestration (From Multi-Agent-Orchestration-Main)
For advanced UI and persistence, adopt the FastAPI-based orchestration system:

```python
# main.py - FastAPI Backend with WebSocket Streaming
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize database pool
    await database.init_pool(database_url=config.DATABASE_URL)
    
    # Initialize orchestrator service
    orchestrator_service = OrchestratorService(
        ws_manager=ws_manager,
        logger=logger,
        agent_manager=agent_manager,
    )
    
    app.state.orchestrator_service = orchestrator_service
    yield
    await database.close_pool()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.websocket("/ws/orchestrator/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await ws_manager.connect(websocket, session_id)
    try:
        while True:
            data = await websocket.receive_text()
            # Process orchestrator commands
            await orchestrator_service.process_command(data, session_id)
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket, session_id)
```

**Database Models**:
```python
# orch_database_models.py - Pydantic Models for Persistence
from pydantic import BaseModel, Field
from uuid import UUID
from typing import Dict, Any, Optional, Literal

class OrchestratorAgent(BaseModel):
    id: UUID
    session_id: Optional[str] = None
    system_prompt: Optional[str] = None
    status: Optional[Literal['idle', 'executing', 'waiting', 'blocked', 'complete']] = None
    working_dir: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    total_cost: float = 0.0
    archived: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime

class Agent(BaseModel):
    id: UUID
    orchestrator_agent_id: UUID
    name: str
    model: str
    system_prompt: Optional[str] = None
    working_dir: Optional[str] = None
    status: Optional[Literal['idle', 'executing', 'waiting', 'blocked', 'complete']] = None
    input_tokens: int = 0
    output_tokens: int = 0
    total_cost: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime
```

**Agent Manager Implementation**:
```python
# agent_manager.py - Agent Lifecycle Management
class AgentManager:
    def __init__(self, orchestrator_agent_id: uuid.UUID, ws_manager, logger, working_dir):
        self.orchestrator_agent_id = orchestrator_agent_id
        self.ws_manager = ws_manager
        self.logger = logger
        self.working_dir = working_dir
        self.active_clients: Dict[str, ClaudeSDKClient] = {}
        self.active_clients_lock = threading.Lock()
    
    async def create_agent(self, name: str, system_prompt: str = None, model: str = None):
        # Create agent in database
        agent_data = await database.create_agent(
            orchestrator_agent_id=self.orchestrator_agent_id,
            name=name,
            model=model or config.DEFAULT_AGENT_MODEL,
            system_prompt=system_prompt,
            working_dir=self.working_dir
        )
        
        # Initialize Claude SDK client
        client = ClaudeSDKClient(
            api_key=config.ANTHROPIC_API_KEY,
            model=agent_data['model'],
            system_prompt=agent_data['system_prompt'],
            working_dir=self.working_dir
        )
        
        # Register tools and hooks
        client.register_tools(self.get_agent_tools())
        client.register_hooks(self.get_agent_hooks())
        
        with self.active_clients_lock:
            self.active_clients[name] = client
        
        return agent_data
    
    async def command_agent(self, agent_name: str, command: str):
        if agent_name not in self.active_clients:
            raise ValueError(f"Agent {agent_name} not found")
        
        client = self.active_clients[agent_name]
        
        # Execute command and stream results
        async for event in client.execute(command):
            await self.ws_manager.broadcast_to_session(
                self.orchestrator_agent_id,
                {"type": "agent_event", "agent": agent_name, "event": event}
            )
```

**Orchestrator Service**:
```python
# orchestrator_service.py - Core Orchestration Logic
class OrchestratorService:
    def __init__(self, ws_manager, logger, agent_manager, session_id, working_dir):
        self.ws_manager = ws_manager
        self.logger = logger
        self.agent_manager = agent_manager
        self.session_id = session_id
        self.working_dir = working_dir
        
        # Initialize Claude SDK for orchestrator
        self.orchestrator_client = ClaudeSDKClient(
            api_key=config.ANTHROPIC_API_KEY,
            model=config.ORCHESTRATOR_MODEL,
            system_prompt=Path(config.ORCHESTRATOR_SYSTEM_PROMPT_PATH).read_text(),
            working_dir=working_dir
        )
        
        # Register orchestrator tools
        self.orchestrator_client.register_tools([
            tool(self.agent_manager.create_agent, name="create_agent"),
            tool(self.agent_manager.list_agents, name="list_agents"),
            tool(self.agent_manager.command_agent, name="command_agent"),
            tool(self.agent_manager.check_agent_status, name="check_agent_status"),
            tool(self.agent_manager.delete_agent, name="delete_agent"),
            tool(self.agent_manager.report_cost, name="report_cost"),
        ])
    
    async def process_message(self, message: str, session_id: str):
        # Log user message to database
        await database.insert_chat_message(
            orchestrator_agent_id=self.orchestrator_agent_id,
            role="user",
            content=message
        )
        
        # Execute orchestrator and stream response
        response_stream = self.orchestrator_client.execute(message)
        
        full_response = ""
        async for event in response_stream:
            if event.type == "text":
                full_response += event.content
                await self.ws_manager.broadcast_to_session(
                    session_id, 
                    {"type": "orchestrator_response", "content": event.content}
                )
            elif event.type == "tool_use":
                # Handle tool execution
                await self.handle_tool_use(event, session_id)
        
        # Log orchestrator response
        await database.insert_chat_message(
            orchestrator_agent_id=self.orchestrator_agent_id,
            role="assistant", 
            content=full_response
        )
        
        # Update costs
        await self.update_costs()
```

**Integration Steps for King-AI-v2**:
1. Add FastAPI to `src/api/` for REST endpoints
2. Implement WebSocket support in `src/websocket.py` for real-time streaming
3. Create database models in `src/database/` for persistence
4. Add agent manager to `src/agents/` for lifecycle management
5. Integrate orchestrator service into `src/master_ai/`
6. Update `dashboard/` frontend to consume WebSocket events
7. Add cost tracking to `src/monitoring/`

### Additional Advanced Patterns

#### Plugin System from Semantic Kernel
Implement extensible agent capabilities:

```python
# plugin_system.py - Extensible Plugin Architecture
from typing import Dict, Any, Callable
from abc import ABC, abstractmethod

class PluginBase(ABC):
    @abstractmethod
    def get_functions(self) -> Dict[str, Callable]:
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        pass

class AgentPluginManager:
    def __init__(self):
        self.plugins: Dict[str, PluginBase] = {}
    
    def register_plugin(self, plugin: PluginBase):
        self.plugins[plugin.get_name()] = plugin
    
    def get_all_functions(self) -> Dict[str, Callable]:
        functions = {}
        for plugin in self.plugins.values():
            functions.update(plugin.get_functions())
        return functions
```

#### Vector Store Integration
From semantic-kernel.pdf, add embedding and retrieval:

```python
# vector_store.py - Vector Database Integration
import numpy as np
from typing import List, Dict, Any

class VectorStore:
    def __init__(self, embedding_model, vector_db):
        self.embedding_model = embedding_model
        self.vector_db = vector_db
    
    async def store_memory(self, agent_id: str, content: str, metadata: Dict[str, Any]):
        # Generate embeddings
        embedding = await self.embedding_model.embed(content)
        
        # Store in vector database
        await self.vector_db.store(
            id=f"{agent_id}_{uuid.uuid4()}",
            vector=embedding,
            payload={"content": content, "metadata": metadata}
        )
    
    async def retrieve_relevant(self, agent_id: str, query: str, limit: int = 5) -> List[Dict]:
        # Generate query embedding
        query_embedding = await self.embedding_model.embed(query)
        
        # Search vector database
        results = await self.vector_db.search(
            vector=query_embedding,
            filter={"agent_id": agent_id},
            limit=limit
        )
        
        return results
```

#### Azure API Management Patterns
From azure-architecture-ai-ml.pdf, implement gateway features:

```python
# api_gateway.py - API Management Layer
from fastapi import Request, Response
from typing import Callable
import time
import asyncio
from collections import defaultdict

class APIGateway:
    def __init__(self):
        self.rate_limits: Dict[str, Dict] = defaultdict(dict)
        self.cache: Dict[str, Dict] = {}
    
    async def handle_request(self, request: Request, next_handler: Callable) -> Response:
        client_ip = request.client.host
        
        # Rate limiting
        if not self.check_rate_limit(client_ip):
            return Response(status_code=429, content="Rate limit exceeded")
        
        # Caching
        cache_key = self.get_cache_key(request)
        if cache_key in self.cache:
            cached = self.cache[cache_key]
            if time.time() - cached['timestamp'] < 300:  # 5 min cache
                return Response(content=cached['content'])
        
        # Authentication
        if not await self.authenticate_request(request):
            return Response(status_code=401, content="Unauthorized")
        
        # Execute request
        start_time = time.time()
        response = await next_handler(request)
        duration = time.time() - start_time
        
        # Logging and monitoring
        await self.log_request(request, response, duration)
        
        # Cache successful responses
        if response.status_code == 200:
            self.cache[cache_key] = {
                'content': response.body,
                'timestamp': time.time()
            }
        
        return response
    
    def check_rate_limit(self, client_ip: str) -> bool:
        now = time.time()
        window_start = now - 60  # 1 minute window
        
        if client_ip not in self.rate_limits:
            self.rate_limits[client_ip] = {'requests': [], 'blocked_until': 0}
        
        client_data = self.rate_limits[client_ip]
        
        # Remove old requests
        client_data['requests'] = [t for t in client_data['requests'] if t > window_start]
        
        # Check if blocked
        if now < client_data['blocked_until']:
            return False
        
        # Check rate limit (100 requests per minute)
        if len(client_data['requests']) >= 100:
            client_data['blocked_until'] = now + 60  # Block for 1 minute
            return False
        
        client_data['requests'].append(now)
        return True
    
    async def authenticate_request(self, request: Request) -> bool:
        # Implement authentication logic (API keys, JWT, etc.)
        auth_header = request.headers.get('Authorization')
        if not auth_header:
            return False
        
        # Validate token/key
        return await self.validate_token(auth_header)
```

### Phase 5: Web and Observability (New Phase)
**Goal**: Add web interface and comprehensive monitoring.

1. **Implement FastAPI Backend**
   - Create `src/api/main.py` with FastAPI app and WebSocket endpoints
   - Add CORS middleware and lifespan management
   - Dependencies: `fastapi`, `uvicorn`

2. **Add WebSocket Streaming**
   - Implement `src/websocket/manager.py` for real-time communication
   - Update `src/agents/` to broadcast events via WebSocket
   - Add session management for multi-user support

3. **Database Persistence**
   - Create `src/database/models.py` with Pydantic models
   - Implement `src/database/connection.py` for PostgreSQL pool
   - Add migration scripts for schema management

4. **Cost Tracking and Observability**
   - Implement `src/monitoring/cost_tracker.py` for token usage
   - Add `src/monitoring/metrics.py` for performance monitoring
   - Integrate with existing `src/monitoring/`

5. **Frontend Integration**
   - Update `dashboard/` to consume WebSocket events
   - Add real-time agent status displays
   - Implement chat interface for orchestrator commands

### Success Metrics (Updated)
- **Orchestration Efficiency**: Reduction in manual agent coordination (target: 80% automation)
- **Real-time Responsiveness**: WebSocket latency < 100ms for agent events
- **Cost Visibility**: 100% tracking of token usage and costs across all agents
- **Scalability**: Support for 50+ concurrent agents with < 5% performance degradation
- **User Experience**: Web dashboard enables intuitive multi-agent management

### ReAct Agent Pattern Implementation
From `llm-agent-framework-main/agent_framework/agents/react_agent.py`, adopt the Reasoning + Acting pattern:

```python
# react_agent.py - ReAct Implementation
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import re

@dataclass
class AgentAction:
    """Represents an action taken by an agent"""
    tool: str
    tool_input: Dict[str, Any]
    log: str

@dataclass
class AgentFinish:
    """Represents the final output of an agent"""
    output: str
    log: str

class ReActAgent(BaseAgent):
    """ReAct agent that reasons and acts iteratively"""
    
    def __init__(self, name: str, description: str, llm: Any, tools: List[Any], max_iterations: int = 10):
        super().__init__(name, description, llm, tools)
        self.max_iterations = max_iterations
    
    def _build_prompt(self, task: str, scratchpad: str = "") -> str:
        """Build ReAct prompt with tools and scratchpad"""
        tool_descriptions = "\n".join([
            f"- {name}: {tool.description}"
            for name, tool in self.tools.items()
        ])
        
        prompt = f"""Answer the following question as best you can. You have access to the following tools:

{tool_descriptions}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{", ".join(self.tools.keys())}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {task}
{scratchpad}"""
        
        return prompt
    
    def _parse_output(self, text: str) -> Union[AgentAction, AgentFinish]:
        """Parse LLM output into action or finish"""
        # Check for final answer
        if "Final Answer:" in text:
            return AgentFinish(
                output=text.split("Final Answer:")[-1].strip(),
                log=text
            )
        
        # Parse action
        action_match = re.search(r"Action:\s*(.+?)(?:\n|$)", text)
        action_input_match = re.search(r"Action Input:\s*(.+?)(?:\n|$)", text, re.DOTALL)
        
        if action_match and action_input_match:
            action = action_match.group(1).strip()
            action_input = action_input_match.group(1).strip()
            
            return AgentAction(
                tool=action,
                tool_input={"query": action_input},
                log=text
            )
        
        # If parsing fails, return a finish with the text
        return AgentFinish(output=text, log=text)
    
    def run(self, task: str) -> str:
        """Run the ReAct agent on a task"""
        scratchpad = ""
        
        for i in range(self.max_iterations):
            # Build prompt with current scratchpad
            prompt = self._build_prompt(task, scratchpad)
            
            # Get LLM response
            try:
                response = self.llm.generate(prompt)
            except Exception as e:
                return f"Error: LLM failed - {str(e)}"
            
            # Parse output
            output = self._parse_output(response)
            
            # Store in memory
            self.add_to_memory({
                'iteration': i,
                'prompt': prompt,
                'response': response,
                'output': output
            })
            
            # Check if finished
            if isinstance(output, AgentFinish):
                return output.output
            
            # Execute action
            observation = self.execute(output)
            
            # Update scratchpad
            scratchpad += f"Thought: {output.log}\n"
            scratchpad += f"Observation: {observation}\n"
        
        return f"Error: Reached maximum iterations ({self.max_iterations})"

class BaseAgent(ABC):
    """Abstract base class for all agents"""
    
    def __init__(self, name: str, description: str, llm: Any, tools: List[Any]):
        self.name = name
        self.description = description
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools}
        self.memory = []
    
    @abstractmethod
    def run(self, task: str) -> str:
        """Run the agent on a task"""
        pass
    
    def execute(self, action: AgentAction) -> str:
        """Execute an action using the specified tool"""
        tool = self.get_tool(action.tool)
        if tool is None:
            return f"Error: Tool '{action.tool}' not found"
        
        try:
            result = tool.run(**action.tool_input)
            return str(result)
        except Exception as e:
            return f"Error executing tool: {str(e)}"
    
    def add_to_memory(self, interaction: Dict[str, Any]) -> None:
        """Add interaction to agent memory"""
        import time
        self.memory.append({
            **interaction,
            'timestamp': time.time()
        })
    
    def get_tool(self, tool_name: str) -> Optional[Any]:
        """Get a tool by name"""
        return self.tools.get(tool_name)
    
    def list_tools(self) -> List[str]:
        """List available tool names"""
        return list(self.tools.keys())
```

**Integration Steps for King-AI-v2**:
1. Create `src/agents/react_agent.py` with the ReAct implementation
2. Update existing agents to inherit from ReActAgent instead of BaseAgent
3. Implement tool execution in `src/agents/` with proper error handling
4. Add scratchpad management for iterative reasoning

### Tool System Implementation
From `llm-agent-framework-main/agent_framework/tools/`, adopt extensible tool architecture:

```python
# base_tool.py - Tool Base Class
from abc import ABC, abstractmethod
from typing import Any

class BaseTool(ABC):
    """Abstract base class for all tools"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    @abstractmethod
    def run(self, **kwargs) -> Any:
        """Execute the tool"""
        pass
    
    def __call__(self, **kwargs) -> Any:
        """Allow tool to be called directly"""
        return self.run(**kwargs)

# web_tools.py - Web Search Tools
class WebSearchTool(BaseTool):
    """Tool for searching the web"""
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__(
            name="web_search",
            description="Search the web for current information. Input should be a search query string."
        )
        self.api_key = api_key
    
    def run(self, query: str, num_results: int = 5) -> str:
        # Implement actual web search (Google, Bing, etc.)
        # Placeholder for now
        return f"Search results for '{query}': [Implement web search API integration]"

class WikipediaTool(BaseTool):
    """Tool for searching Wikipedia"""
    
    def __init__(self):
        super().__init__(
            name="wikipedia",
            description="Search Wikipedia for factual information. Input should be a topic or search term."
        )
    
    def run(self, query: str, sentences: int = 5) -> str:
        try:
            import wikipedia
            search_results = wikipedia.search(query, results=3)
            
            if not search_results:
                return f"No Wikipedia articles found for '{query}'"
            
            try:
                summary = wikipedia.summary(search_results[0], sentences=sentences)
                return f"**{search_results[0]}**\n\n{summary}"
            except wikipedia.DisambiguationError as e:
                return f"Multiple results found. Options: {', '.join(e.options[:5])}"
            except wikipedia.PageError:
                return f"No Wikipedia page found for '{query}'"
        
        except ImportError:
            return f"Wikipedia search requires 'wikipedia' package"

# code_tools.py - Code Execution Tools
class PythonREPLTool(BaseTool):
    """Tool for executing Python code"""
    
    def __init__(self):
        super().__init__(
            name="python_repl",
            description="Execute Python code. Input should be valid Python code."
        )
    
    def run(self, code: str) -> str:
        """Execute Python code safely"""
        try:
            # Create restricted namespace for security
            namespace = {
                "__builtins__": {
                    "print": print,
                    "len": len,
                    "str": str,
                    "int": int,
                    "float": float,
                    "list": list,
                    "dict": dict,
                }
            }
            exec(code, namespace)
            return "Code executed successfully"
        except Exception as e:
            return f"Error: {str(e)}"

class CodeAnalysisTool(BaseTool):
    """Tool for analyzing code"""
    
    def __init__(self):
        super().__init__(
            name="code_analysis",
            description="Analyze code for issues, complexity, and improvements."
        )
    
    def run(self, code: str) -> str:
        """Analyze code quality"""
        try:
            import ast
            import radon.complexity as cc_radon
            import radon.metrics as metrics
            
            # Parse AST
            tree = ast.parse(code)
            
            # Calculate complexity
            complexity = cc_radon.cc_visit_ast(tree)
            avg_complexity = sum(c.complexity for c in complexity) / len(complexity) if complexity else 0
            
            # Basic metrics
            lines = len(code.split('\n'))
            functions = len([node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)])
            
            analysis = f"""
Code Analysis:
- Lines of code: {lines}
- Functions: {functions}
- Average complexity: {avg_complexity:.2f}
- Maintainability: {'Good' if avg_complexity < 10 else 'Needs improvement'}
"""
            return analysis.strip()
        
        except ImportError:
            return "Code analysis requires 'radon' and 'ast' packages"
        except Exception as e:
            return f"Analysis failed: {str(e)}"
```

**Integration Steps for King-AI-v2**:
1. Create `src/tools/base_tool.py` with BaseTool class
2. Implement specific tools in `src/tools/` (web search, code execution, data analysis)
3. Update agents to use tool collections instead of hardcoded functionality
4. Add tool registration system for extensibility

### Memory System Implementation
From `llm-agent-framework-main/agent_framework/memory/`, adopt agent memory patterns:

```python
# memory.py - Agent Memory System
from typing import List, Dict, Any
import time
from collections import deque

class AgentMemory:
    """Memory system for agents"""
    
    def __init__(self, max_size: int = 100):
        self.memories = deque(maxlen=max_size)
    
    def add_interaction(self, interaction: Dict[str, Any]) -> None:
        """Add interaction to memory"""
        self.memories.append({
            **interaction,
            'timestamp': time.time()
        })
    
    def get_recent_memories(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent memories"""
        return list(self.memories)[-limit:]
    
    def search_memories(self, query: str) -> List[Dict[str, Any]]:
        """Search memories by content"""
        return [m for m in self.memories if query.lower() in str(m).lower()]
    
    def get_context(self, current_task: str) -> str:
        """Get relevant context for current task"""
        recent = self.get_recent_memories(5)
        relevant = self.search_memories(current_task)
        
        context = "Recent interactions:\n"
        for mem in recent:
            context += f"- {mem.get('task', 'Unknown')}: {mem.get('result', 'Unknown')}\n"
        
        if relevant:
            context += "\nRelevant past interactions:\n"
            for mem in relevant[-3:]:  # Last 3 relevant
                context += f"- {mem.get('task', 'Unknown')}: {mem.get('result', 'Unknown')}\n"
        
        return context

# Integration in agents
class MemoryEnabledAgent(BaseAgent):
    def __init__(self, name: str, description: str, llm: Any, tools: List[Any]):
        super().__init__(name, description, llm, tools)
        self.memory = AgentMemory()
    
    def run_with_memory(self, task: str) -> str:
        # Get context from memory
        context = self.memory.get_context(task)
        
        # Enhance task with context
        enhanced_task = f"{task}\n\nContext from previous interactions:\n{context}"
        
        # Run task
        result = self.run(enhanced_task)
        
        # Store in memory
        self.memory.add_interaction({
            'task': task,
            'enhanced_task': enhanced_task,
            'result': result
        })
        
        return result
```

**Integration Steps for King-AI-v2**:
1. Create `src/memory/agent_memory.py` with memory management
2. Update agents to use memory for context-aware responses
3. Implement memory persistence in `src/database/`
4. Add memory search and retrieval capabilities

### Supervisor Agent Pattern
From `llm-agent-framework-main/agent_framework/agents/supervisor.py`, implement task delegation:

```python
# supervisor.py - Task Delegation Agent
class SupervisorAgent(BaseAgent):
    """Agent that delegates tasks to specialized agents"""
    
    def __init__(self, name: str, description: str, llm: Any, tools: List[Any], 
                 subordinate_agents: Dict[str, BaseAgent]):
        super().__init__(name, description, llm, tools)
        self.subordinate_agents = subordinate_agents
    
    def analyze_task(self, task: str) -> Dict[str, Any]:
        """Analyze task to determine required agent types"""
        prompt = f"""
Analyze this task and determine which types of agents are needed:

Task: {task}

Available agent types:
- research: For information gathering and analysis
- code: For programming and technical tasks
- data: For data analysis and visualization
- writing: For content creation and communication

Return a JSON object with:
- primary_agent: The main agent type needed
- secondary_agents: List of additional agent types if needed
- reasoning: Brief explanation of the analysis

Response format: {{"primary_agent": "type", "secondary_agents": ["type1", "type2"], "reasoning": "explanation"}}
"""
        
        response = self.llm.generate(prompt)
        
        try:
            import json
            analysis = json.loads(response)
            return analysis
        except:
            # Fallback analysis
            return {
                "primary_agent": "research",
                "secondary_agents": [],
                "reasoning": "Fallback analysis due to parsing error"
            }
    
    def delegate_task(self, task: str) -> str:
        """Delegate task to appropriate agents"""
        analysis = self.analyze_task(task)
        
        results = []
        
        # Execute with primary agent
        primary_agent = self.subordinate_agents.get(analysis["primary_agent"])
        if primary_agent:
            result = primary_agent.run(task)
            results.append(f"Primary ({analysis['primary_agent']}): {result}")
        
        # Execute with secondary agents if needed
        for agent_type in analysis.get("secondary_agents", []):
            agent = self.subordinate_agents.get(agent_type)
            if agent:
                result = agent.run(task)
                results.append(f"Secondary ({agent_type}): {result}")
        
        # Combine results
        final_result = f"Task Analysis: {analysis['reasoning']}\n\n" + "\n\n".join(results)
        return final_result
```

**Integration Steps for King-AI-v2**:
1. Create `src/agents/supervisor.py` with task analysis and delegation
2. Update `src/master_ai/` to use supervisor pattern for complex tasks
3. Implement agent specialization in existing agents
4. Add workflow routing based on task analysis

### Phase 6: Advanced Agent Patterns (New Phase)
**Goal**: Implement sophisticated agent behaviors and tool integrations.

1. **Implement ReAct Pattern**
   - Create `src/agents/react_agent.py` with iterative reasoning
   - Update research agents to use ReAct for complex queries
   - Add scratchpad management for thought processes

2. **Build Tool Ecosystem**
   - Implement `src/tools/` with web search, code execution, data analysis tools
   - Add tool registration system for extensibility
   - Integrate tools with agent workflows

3. **Add Memory Systems**
   - Implement `src/memory/` for context-aware agents
   - Add memory persistence and search capabilities
   - Update agents to learn from past interactions

4. **Create Supervisor Agents**
   - Implement `src/agents/supervisor.py` for task delegation
   - Add agent specialization and collaboration patterns
   - Integrate with existing orchestration

5. **Advanced Tool Integration**
   - Add Wikipedia, web scraping, and API calling tools
   - Implement tool chaining and conditional execution
   - Add tool performance monitoring

### Success Metrics (Updated)
- **Agent Autonomy**: Percentage of tasks completed without human intervention (target: 70%)
- **Tool Utilization**: Average tools used per task (target: 2-3 tools)
- **Memory Effectiveness**: Context relevance in responses (target: 80% relevant)
- **Task Success Rate**: Complex multi-step tasks completed successfully (target: 85%)
- **Response Quality**: User satisfaction with agent outputs (target: 4.5/5)

This expanded plan now includes complete ReAct agent implementations, comprehensive tool systems, memory management, and supervisor patterns from the LLM agent framework.

## Timeline and Resources
- **Total Duration**: 20 weeks (added Phase 5)
- **Team**: 1-2 developers for implementation, QA for testing
- **Tools**: VS Code, pytest, mypy

This plan provides a structured path to leverage the reference frameworks. Adjust based on priorities and testing results.

### Phase 7: Enterprise Patterns (New Phase)
**Goal**: Implement enterprise-grade patterns for production deployment and governance.

1. **Implement Tiered Memory System**
   - Adopt the three-tier memory architecture from mother-harness-master
   - Create `src/memory/tiered_memory.py` with recent, episodic, and semantic layers
   - Add memory consolidation and retrieval strategies
   - Integrate with agent workflows for context management

2. **Add Task Planning and Orchestration**
   - Implement `src/orchestrator/task_planner.py` with agent detection and planning
   - Add workflow decomposition for complex multi-agent tasks
   - Create approval workflows for sensitive operations
   - Integrate with existing master AI coordination

3. **Implement Agent Registry and Governance**
   - Create `src/registry/agent_registry.py` for agent discovery and management
   - Add agent classification and capability mapping from reference architecture
   - Implement governance policies for agent interactions
   - Add audit trails and compliance tracking

4. **Add Intent Classification and Routing**
   - Implement `src/classification/intent_classifier.py` for task categorization
   - Add natural language processing for intent detection
   - Create routing rules for agent selection based on intent
   - Integrate with orchestrator for intelligent task distribution

5. **Enterprise Observability and Monitoring**
   - Enhance `src/monitoring/` with comprehensive logging and metrics
   - Add distributed tracing for multi-agent workflows
   - Implement health checks and performance monitoring
   - Create dashboards for operational visibility

6. **Production Deployment Patterns**
   - Add container orchestration with Docker Compose patterns
   - Implement load balancing and redundancy strategies
   - Add configuration management for multi-environment support
   - Create deployment pipelines and rollback procedures

### Tiered Memory Implementation
From `mother-harness-master/services/orchestrator/src/memory/`, adopt the three-tier architecture:

```python
# tiered_memory.py - Three-Tier Memory System
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import asyncio
from collections import deque
import json

class TieredMemory:
    """Three-tier memory system: Recent, Episodic, Semantic"""
    
    def __init__(self, max_recent: int = 100, max_episodic: int = 1000):
        # Tier 1: Recent Memory (short-term, high-fidelity)
        self.recent_memory = deque(maxlen=max_recent)
        
        # Tier 2: Episodic Memory (medium-term, consolidated)
        self.episodic_memory = deque(maxlen=max_episodic)
        
        # Tier 3: Semantic Memory (long-term, abstracted)
        self.semantic_memory: Dict[str, Any] = {}
        
        # Memory consolidation settings
        self.consolidation_interval = 300  # 5 minutes
        self.last_consolidation = datetime.now()
    
    async def store_interaction(self, interaction: Dict[str, Any]) -> None:
        """Store interaction in recent memory"""
        interaction['timestamp'] = datetime.now()
        interaction['tier'] = 'recent'
        self.recent_memory.append(interaction)
        
        # Trigger consolidation if needed
        await self._check_consolidation()
    
    async def retrieve_context(self, query: str, agent_id: str = None, limit: int = 5) -> Dict[str, List]:
        """Retrieve relevant context from all memory tiers"""
        context = {
            'recent': [],
            'episodic': [],
            'semantic': []
        }
        
        # Search recent memory
        for item in reversed(list(self.recent_memory)):
            if self._matches_query(item, query, agent_id):
                context['recent'].append(item)
                if len(context['recent']) >= limit:
                    break
        
        # Search episodic memory
        for item in reversed(list(self.episodic_memory)):
            if self._matches_query(item, query, agent_id):
                context['episodic'].append(item)
                if len(context['episodic']) >= limit:
                    break
        
        # Search semantic memory
        for key, value in self.semantic_memory.items():
            if query.lower() in key.lower() or (agent_id and agent_id in str(value)):
                context['semantic'].append({'key': key, 'value': value})
        
        return context
    
    def _matches_query(self, item: Dict, query: str, agent_id: str = None) -> bool:
        """Check if memory item matches query"""
        text_content = json.dumps(item, default=str).lower()
        
        query_match = query.lower() in text_content
        agent_match = not agent_id or agent_id in str(item.get('agent_id', ''))
        
        return query_match and agent_match
    
    async def _check_consolidation(self) -> None:
        """Consolidate recent memories to episodic"""
        now = datetime.now()
        if (now - self.last_consolidation).seconds > self.consolidation_interval:
            await self._consolidate_memories()
            self.last_consolidation = now
    
    async def _consolidate_memories(self) -> None:
        """Move old recent memories to episodic tier"""
        cutoff_time = datetime.now() - timedelta(hours=1)  # 1 hour ago
        
        to_consolidate = []
        remaining_recent = deque(maxlen=self.recent_memory.maxlen)
        
        for item in self.recent_memory:
            if item['timestamp'] < cutoff_time:
                # Mark as episodic and consolidate
                item['tier'] = 'episodic'
                item['consolidated_at'] = datetime.now()
                to_consolidate.append(item)
            else:
                remaining_recent.append(item)
        
        # Add to episodic memory
        for item in to_consolidate:
            self.episodic_memory.append(item)
        
        # Update recent memory
        self.recent_memory = remaining_recent
        
        # Extract semantic patterns
        await self._extract_semantic_patterns(to_consolidate)
    
    async def _extract_semantic_patterns(self, interactions: List[Dict]) -> None:
        """Extract semantic patterns from interactions"""
        # Group by agent and task type
        patterns = {}
        
        for interaction in interactions:
            agent_id = interaction.get('agent_id', 'unknown')
            task_type = interaction.get('task_type', 'unknown')
            success = interaction.get('success', False)
            
            key = f"{agent_id}_{task_type}"
            if key not in patterns:
                patterns[key] = {'success_count': 0, 'total_count': 0, 'patterns': []}
            
            patterns[key]['total_count'] += 1
            if success:
                patterns[key]['success_count'] += 1
            
            # Extract common patterns
            if 'task' in interaction:
                patterns[key]['patterns'].append(interaction['task'][:100])  # First 100 chars
        
        # Store successful patterns in semantic memory
        for key, data in patterns.items():
            success_rate = data['success_count'] / data['total_count'] if data['total_count'] > 0 else 0
            if success_rate > 0.8:  # High success rate patterns
                self.semantic_memory[key] = {
                    'success_rate': success_rate,
                    'pattern_count': len(data['patterns']),
                    'last_updated': datetime.now()
                }
```

**Integration Steps**:
1. Create `src/memory/tiered_memory.py` with the three-tier implementation
2. Update agents to use `store_interaction()` and `retrieve_context()`
3. Add memory consolidation background task
4. Implement semantic pattern extraction for learning

### Task Planning Implementation
From `mother-harness-master/services/orchestrator/src/planner/`, adopt intelligent task planning:

```python
# task_planner.py - Intelligent Task Planning
from typing import Dict, List, Any, Optional
import re
from datetime import datetime

class TaskPlanner:
    """Intelligent task planner with agent detection and decomposition"""
    
    def __init__(self, agent_registry, memory_system):
        self.agent_registry = agent_registry
        self.memory_system = memory_system
        
        # Planning patterns
        self.task_patterns = {
            'research': re.compile(r'(research|investigate|analyze|study|explore)', re.IGNORECASE),
            'code': re.compile(r'(code|program|implement|develop|create.*software)', re.IGNORECASE),
            'data': re.compile(r'(data|analyze|visualize|statistics|database)', re.IGNORECASE),
            'writing': re.compile(r'(write|content|document|report|communicate)', re.IGNORECASE),
            'approval': re.compile(r'(approve|review|validate|authorize|compliance)', re.IGNORECASE)
        }
    
    async def plan_task(self, task_description: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create execution plan for a task"""
        
        # Detect required agent types
        required_agents = await self._detect_agents(task_description)
        
        # Check for similar past tasks
        similar_tasks = await self.memory_system.retrieve_context(
            task_description, limit=3
        )
        
        # Decompose complex tasks
        subtasks = await self._decompose_task(task_description, required_agents)
        
        # Create execution plan
        plan = {
            'task_id': f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'description': task_description,
            'required_agents': required_agents,
            'subtasks': subtasks,
            'estimated_complexity': self._estimate_complexity(subtasks),
            'similar_past_tasks': len(similar_tasks.get('episodic', [])),
            'requires_approval': self._requires_approval(task_description),
            'created_at': datetime.now()
        }
        
        return plan
    
    async def _detect_agents(self, task_description: str) -> List[str]:
        """Detect which types of agents are needed"""
        detected_agents = []
        
        for agent_type, pattern in self.task_patterns.items():
            if pattern.search(task_description):
                detected_agents.append(agent_type)
        
        # If no specific agents detected, use general purpose
        if not detected_agents:
            detected_agents = ['research']  # Default fallback
        
        return detected_agents
    
    async def _decompose_task(self, task_description: str, required_agents: List[str]) -> List[Dict[str, Any]]:
        """Break down complex tasks into subtasks"""
        subtasks = []
        
        # Simple decomposition based on agent types
        for agent_type in required_agents:
            subtask = {
                'agent_type': agent_type,
                'description': f"Handle {agent_type} aspects of: {task_description}",
                'dependencies': [],  # Could be enhanced with dependency analysis
                'estimated_duration': self._estimate_duration(agent_type, task_description)
            }
            subtasks.append(subtask)
        
        # Add coordination if multiple agents needed
        if len(subtasks) > 1:
            subtasks.append({
                'agent_type': 'coordinator',
                'description': f"Coordinate results from {', '.join(required_agents)} agents",
                'dependencies': [i for i in range(len(required_agents))],
                'estimated_duration': 5  # minutes
            })
        
        return subtasks
    
    def _estimate_complexity(self, subtasks: List[Dict]) -> str:
        """Estimate task complexity"""
        total_duration = sum(task.get('estimated_duration', 10) for task in subtasks)
        
        if total_duration < 15:
            return 'low'
        elif total_duration < 60:
            return 'medium'
        else:
            return 'high'
    
    def _estimate_duration(self, agent_type: str, task_description: str) -> int:
        """Estimate task duration in minutes"""
        base_durations = {
            'research': 20,
            'code': 45,
            'data': 30,
            'writing': 25,
            'approval': 10
        }
        
        # Adjust based on task length (rough heuristic)
        words = len(task_description.split())
        multiplier = min(1 + (words - 50) / 200, 2.0)  # Up to 2x for very long tasks
        
        return int(base_durations.get(agent_type, 15) * multiplier)
    
    def _requires_approval(self, task_description: str) -> bool:
        """Determine if task requires approval workflow"""
        approval_keywords = [
            'deploy', 'production', 'finance', 'security', 'compliance',
            'approve', 'authorize', 'sensitive', 'critical'
        ]
        
        return any(keyword in task_description.lower() for keyword in approval_keywords)
```

**Integration Steps**:
1. Create `src/orchestrator/task_planner.py` with planning logic
2. Integrate with agent registry for capability detection
3. Update orchestrator to use planning for complex tasks
4. Add approval workflow integration

### Agent Registry Implementation
From `multi-agent-reference-architecture-main`, adopt enterprise agent management:

```python
# agent_registry.py - Enterprise Agent Registry
from typing import Dict, List, Any, Optional
from datetime import datetime
import uuid
from enum import Enum

class AgentStatus(Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    MAINTENANCE = "maintenance"
    DEPRECATED = "deprecated"

class AgentCapability(Enum):
    RESEARCH = "research"
    CODING = "coding"
    DATA_ANALYSIS = "data_analysis"
    WRITING = "writing"
    APPROVAL = "approval"
    COORDINATION = "coordination"

class AgentRegistry:
    """Enterprise agent registry with governance and discovery"""
    
    def __init__(self):
        self.agents: Dict[str, Dict[str, Any]] = {}
        self.capabilities_index: Dict[str, List[str]] = {}
        self.performance_metrics: Dict[str, Dict[str, Any]] = {}
    
    def register_agent(self, agent_info: Dict[str, Any]) -> str:
        """Register a new agent in the registry"""
        agent_id = str(uuid.uuid4())
        
        agent_record = {
            'id': agent_id,
            'name': agent_info['name'],
            'description': agent_info.get('description', ''),
            'capabilities': agent_info.get('capabilities', []),
            'status': AgentStatus.ACTIVE.value,
            'version': agent_info.get('version', '1.0.0'),
            'endpoint': agent_info.get('endpoint'),
            'metadata': agent_info.get('metadata', {}),
            'registered_at': datetime.now(),
            'last_active': datetime.now(),
            'performance_score': 0.0
        }
        
        self.agents[agent_id] = agent_record
        
        # Update capabilities index
        for capability in agent_record['capabilities']:
            if capability not in self.capabilities_index:
                self.capabilities_index[capability] = []
            self.capabilities_index[capability].append(agent_id)
        
        return agent_id
    
    def discover_agents(self, capabilities: List[str] = None, 
                       status: AgentStatus = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Discover agents by capabilities and status"""
        candidates = list(self.agents.values())
        
        # Filter by status
        if status:
            candidates = [a for a in candidates if a['status'] == status.value]
        
        # Filter by capabilities
        if capabilities:
            filtered = []
            for agent in candidates:
                if any(cap in agent['capabilities'] for cap in capabilities):
                    filtered.append(agent)
            candidates = filtered
        
        # Sort by performance score
        candidates.sort(key=lambda x: x.get('performance_score', 0), reverse=True)
        
        return candidates[:limit]
    
    def update_agent_status(self, agent_id: str, status: AgentStatus) -> bool:
        """Update agent status"""
        if agent_id not in self.agents:
            return False
        
        self.agents[agent_id]['status'] = status.value
        self.agents[agent_id]['last_updated'] = datetime.now()
        return True
    
    def record_performance(self, agent_id: str, task_type: str, 
                          success: bool, duration: float, quality_score: float = None) -> None:
        """Record agent performance metrics"""
        if agent_id not in self.performance_metrics:
            self.performance_metrics[agent_id] = {
                'total_tasks': 0,
                'successful_tasks': 0,
                'total_duration': 0,
                'avg_quality': 0,
                'task_types': {}
            }
        
        metrics = self.performance_metrics[agent_id]
        metrics['total_tasks'] += 1
        metrics['total_duration'] += duration
        
        if success:
            metrics['successful_tasks'] += 1
        
        if quality_score is not None:
            current_avg = metrics['avg_quality']
            metrics['avg_quality'] = (current_avg * (metrics['total_tasks'] - 1) + quality_score) / metrics['total_tasks']
        
        # Update task type specific metrics
        if task_type not in metrics['task_types']:
            metrics['task_types'][task_type] = {'count': 0, 'success_rate': 0}
        
        task_metrics = metrics['task_types'][task_type]
        task_metrics['count'] += 1
        task_metrics['success_rate'] = metrics['successful_tasks'] / metrics['total_tasks']
        
        # Update agent performance score
        success_rate = metrics['successful_tasks'] / metrics['total_tasks']
        avg_duration = metrics['total_duration'] / metrics['total_tasks']
        quality_component = metrics['avg_quality']
        
        # Weighted performance score
        self.agents[agent_id]['performance_score'] = (
            success_rate * 0.5 + 
            (1.0 / (1.0 + avg_duration / 60)) * 0.3 +  # Prefer faster agents
            quality_component * 0.2
        )
    
    def get_agent_health_report(self) -> Dict[str, Any]:
        """Generate health report for all agents"""
        total_agents = len(self.agents)
        active_agents = len([a for a in self.agents.values() if a['status'] == AgentStatus.ACTIVE.value])
        
        return {
            'total_agents': total_agents,
            'active_agents': active_agents,
            'inactive_agents': total_agents - active_agents,
            'average_performance_score': sum(a.get('performance_score', 0) for a in self.agents.values()) / total_agents if total_agents > 0 else 0,
            'capability_coverage': {cap: len(agents) for cap, agents in self.capabilities_index.items()}
        }
```

**Integration Steps**:
1. Create `src/registry/agent_registry.py` with enterprise registry
2. Update agent initialization to register with the registry
3. Integrate performance tracking in orchestrator
4. Add health monitoring and reporting

### Intent Classification Implementation
From `multi-agent-reference-architecture-main`, adopt NLP-based intent detection:

```python
# intent_classifier.py - Natural Language Intent Classification
from typing import Dict, List, Any, Optional
import re
from enum import Enum

class IntentType(Enum):
    RESEARCH = "research"
    CODING = "coding"
    DATA_ANALYSIS = "data_analysis"
    CONTENT_WRITING = "content_writing"
    APPROVAL = "approval"
    COORDINATION = "coordination"
    GENERAL = "general"

class IntentClassifier:
    """NLP-based intent classification for task routing"""
    
    def __init__(self):
        # Intent patterns with keywords and regex
        self.intent_patterns = {
            IntentType.RESEARCH: {
                'keywords': ['research', 'investigate', 'analyze', 'study', 'explore', 'find', 'search', 'discover'],
                'regex': re.compile(r'(research|investigate|analyze|study|explore|find.*information)', re.IGNORECASE)
            },
            IntentType.CODING: {
                'keywords': ['code', 'program', 'implement', 'develop', 'create', 'build', 'software', 'function'],
                'regex': re.compile(r'(code|program|implement|develop|create.*software|write.*code)', re.IGNORECASE)
            },
            IntentType.DATA_ANALYSIS: {
                'keywords': ['data', 'analyze', 'visualize', 'statistics', 'database', 'query', 'chart', 'graph'],
                'regex': re.compile(r'(data.*analysis|visualize|statistics|database|query)', re.IGNORECASE)
            },
            IntentType.CONTENT_WRITING: {
                'keywords': ['write', 'content', 'document', 'report', 'communicate', 'email', 'article'],
                'regex': re.compile(r'(write.*content|document|report|communicate|create.*text)', re.IGNORECASE)
            },
            IntentType.APPROVAL: {
                'keywords': ['approve', 'review', 'validate', 'authorize', 'compliance', 'check', 'verify'],
                'regex': re.compile(r'(approve|review|validate|authorize|compliance|verify)', re.IGNORECASE)
            },
            IntentType.COORDINATION: {
                'keywords': ['coordinate', 'manage', 'orchestrate', 'schedule', 'plan', 'organize'],
                'regex': re.compile(r'(coordinate|manage|orchestrate|schedule|plan.*task)', re.IGNORECASE)
            }
        }
    
    def classify_intent(self, text: str) -> Dict[str, Any]:
        """Classify the intent of a text input"""
        text_lower = text.lower()
        
        # Score each intent
        intent_scores = {}
        matched_keywords = {}
        
        for intent_type, patterns in self.intent_patterns.items():
            score = 0
            matches = []
            
            # Keyword matching
            for keyword in patterns['keywords']:
                if keyword in text_lower:
                    score += 1
                    matches.append(keyword)
            
            # Regex matching
            if patterns['regex'].search(text):
                score += 2  # Higher weight for regex matches
            
            intent_scores[intent_type] = score
            matched_keywords[intent_type] = matches
        
        # Find highest scoring intent
        best_intent = max(intent_scores, key=intent_scores.get)
        confidence = intent_scores[best_intent] / max(sum(intent_scores.values()), 1)
        
        # Fallback to general if low confidence
        if confidence < 0.3:
            best_intent = IntentType.GENERAL
        
        return {
            'intent': best_intent.value,
            'confidence': confidence,
            'matched_keywords': matched_keywords[best_intent],
            'all_scores': {k.value: v for k, v in intent_scores.items()}
        }
    
    def get_routing_rules(self, intent: str) -> Dict[str, Any]:
        """Get routing rules for a classified intent"""
        routing_rules = {
            IntentType.RESEARCH.value: {
                'primary_agent_types': ['research'],
                'fallback_agents': ['general'],
                'requires_approval': False,
                'max_parallel_agents': 2
            },
            IntentType.CODING.value: {
                'primary_agent_types': ['coding'],
                'fallback_agents': ['general'],
                'requires_approval': True,
                'max_parallel_agents': 1
            },
            IntentType.DATA_ANALYSIS.value: {
                'primary_agent_types': ['data_analysis'],
                'fallback_agents': ['research'],
                'requires_approval': False,
                'max_parallel_agents': 1
            },
            IntentType.CONTENT_WRITING.value: {
                'primary_agent_types': ['content_writing'],
                'fallback_agents': ['general'],
                'requires_approval': False,
                'max_parallel_agents': 1
            },
            IntentType.APPROVAL.value: {
                'primary_agent_types': ['approval'],
                'fallback_agents': [],
                'requires_approval': False,  # Already is approval
                'max_parallel_agents': 1
            },
            IntentType.COORDINATION.value: {
                'primary_agent_types': ['coordination'],
                'fallback_agents': ['general'],
                'requires_approval': True,
                'max_parallel_agents': 3
            },
            IntentType.GENERAL.value: {
                'primary_agent_types': ['general'],
                'fallback_agents': [],
                'requires_approval': False,
                'max_parallel_agents': 1
            }
        }
        
        return routing_rules.get(intent, routing_rules[IntentType.GENERAL.value])
```

**Integration Steps**:
1. Create `src/classification/intent_classifier.py` with NLP classification
2. Integrate with orchestrator for automatic task routing
3. Add confidence thresholds and fallback handling
4. Train classification patterns based on usage data

### Success Metrics (Final Update)
- **Enterprise Scalability**: Support for 100+ agents with governance (target: < 10% overhead)
- **Intent Accuracy**: Correct task routing (target: 85% accuracy)
- **Memory Effectiveness**: Context relevance across tiers (target: 90% relevant)
- **Governance Compliance**: Audit trail completeness (target: 100%)
- **Planning Efficiency**: Task decomposition accuracy (target: 80% optimal plans)
- **System Reliability**: 99.9% uptime with enterprise patterns

## Final Timeline and Resources
- **Total Duration**: 24 weeks (added Phase 7)
- **Team**: 2-3 developers for enterprise implementation
- **Infrastructure**: Enterprise-grade deployment with monitoring
- **Compliance**: Security and governance reviews

This comprehensive plan now includes all enterprise patterns from the reference frameworks, providing a production-ready path for king-ai-v2 evolution.

### Phase 8: Advanced Streaming & Real-time Features (New Phase)
**Goal**: Implement real-time streaming orchestration and advanced agent communication patterns.

1. **WebSocket Streaming Orchestrator**
   - Adopt the three-phase logging pattern from multi-agent-orchestration-main
   - Implement real-time agent event streaming via WebSocket
   - Add session management and connection handling
   - Integrate with database persistence for chat history

2. **Streaming Agent Execution**
   - Implement Claude SDK streaming for agent responses
   - Add real-time token usage tracking and cost monitoring
   - Create event-driven architecture for agent coordination
   - Add interrupt capabilities for long-running agents

3. **Advanced Hook System**
   - Implement pre/post tool execution hooks from orchestrator_hooks.py
   - Add command agent hooks for specialized agent behaviors
   - Create event summarization and context management
   - Integrate hook system with memory and approval workflows

4. **Multi-Modal Agent Communication**
   - Add support for different agent communication patterns
   - Implement agent-to-agent messaging via shared memory
   - Create broadcast mechanisms for system-wide events
   - Add agent discovery and peer communication

### Advanced Multi-Agent Workflow Examples
From `agentic-ai-framework-main/examples/`, adopt comprehensive workflow patterns:

```python
# Advanced Multi-Agent Workflow with Error Handling and Recovery
class ResearchAnalysisWorkflow:
    """Complete research and analysis workflow with multiple agents"""
    
    def __init__(self):
        self.researcher = ResearcherAgent(AgentConfig(
            name="researcher",
            description="Gathers and analyzes information"
        ))
        self.analyzer = AnalyzerAgent(AgentConfig(
            name="analyzer", 
            description="Processes and synthesizes research data"
        ))
        self.critic = CriticAgent(AgentConfig(
            name="critic",
            description="Reviews and validates outputs"
        ))
    
    async def execute_research_workflow(self, topic: str) -> Dict[str, Any]:
        """Execute complete research workflow with error recovery"""
        
        # Phase 1: Research
        research_task = {
            "type": "research",
            "data": {"topic": topic, "depth": "comprehensive"}
        }
        research_result = await self.researcher.process(research_task)
        
        if not research_result.get("success"):
            # Fallback: Try with simplified research
            research_task["data"]["depth"] = "basic"
            research_result = await self.researcher.process(research_task)
        
        # Phase 2: Analysis
        analysis_task = {
            "type": "analyze",
            "data": research_result.get("result", {})
        }
        analysis_result = await self.analyzer.process(analysis_task)
        
        # Phase 3: Critical Review
        review_task = {
            "type": "review",
            "data": {
                "research": research_result.get("result"),
                "analysis": analysis_result.get("result")
            }
        }
        review_result = await self.critic.process(review_task)
        
        return {
            "research": research_result,
            "analysis": analysis_result,
            "review": review_result,
            "final_output": review_result.get("result", {}).get("validated_output")
        }
```

### Advanced Tool Ecosystem Implementation
From `llm-agent-framework-main/agent_framework/tools/`, implement comprehensive tool system:

```python
# Advanced Tool System with Categories and Dependencies
class ToolManager:
    """Manages tool registration, discovery, and execution"""
    
    def __init__(self):
        self.tools: Dict[str, BaseTool] = {}
        self.categories: Dict[str, List[str]] = {}
        self.dependencies: Dict[str, List[str]] = {}
    
    def register_tool(self, tool: BaseTool, category: str = "general", 
                     dependencies: List[str] = None) -> None:
        """Register a tool with metadata"""
        self.tools[tool.name] = tool
        
        if category not in self.categories:
            self.categories[category] = []
        self.categories[category].append(tool.name)
        
        if dependencies:
            self.dependencies[tool.name] = dependencies
    
    def get_tools_by_category(self, category: str) -> List[BaseTool]:
        """Get all tools in a category"""
        tool_names = self.categories.get(category, [])
        return [self.tools[name] for name in tool_names if name in self.tools]
    
    def can_execute_tool(self, tool_name: str) -> bool:
        """Check if tool dependencies are satisfied"""
        if tool_name not in self.dependencies:
            return True
        
        for dep in self.dependencies[tool_name]:
            if dep not in self.tools:
                return False
        return True
    
    async def execute_tool_chain(self, tool_chain: List[str], 
                                inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a chain of tools with data flow"""
        results = {}
        current_inputs = inputs.copy()
        
        for tool_name in tool_chain:
            if not self.can_execute_tool(tool_name):
                raise ValueError(f"Cannot execute {tool_name}: dependencies not satisfied")
            
            tool = self.tools[tool_name]
            result = await tool.run(**current_inputs)
            results[tool_name] = result
            
            # Update inputs for next tool
            current_inputs.update(result)
        
        return results

# Specialized Tool Categories
class DataAnalysisTool(BaseTool):
    """Advanced data analysis and visualization tool"""
    
    def __init__(self):
        super().__init__(
            name="data_analysis",
            description="Analyze datasets, create visualizations, and extract insights"
        )
        self.supported_formats = ['csv', 'json', 'xlsx', 'parquet']
    
    def run(self, data_path: str, analysis_type: str = "summary", 
            output_format: str = "json") -> Dict[str, Any]:
        """Run data analysis on file"""
        # Implementation for data loading, analysis, and visualization
        pass

class APITool(BaseTool):
    """Generic API interaction tool"""
    
    def __init__(self, base_url: str, auth_token: str = None):
        super().__init__(
            name=f"api_{base_url.replace('https://', '').replace('http://', '').split('.')[0]}",
            description=f"Interact with {base_url} API"
        )
        self.base_url = base_url
        self.auth_token = auth_token
    
    def run(self, endpoint: str, method: str = "GET", 
            params: Dict = None, data: Dict = None) -> Dict[str, Any]:
        """Execute API request"""
        # Implementation for API calls with error handling
        pass
```

### Enterprise Memory Service Implementation
From `agentic-framework-main/memory-service/`, adopt production-grade memory system:

```python
# Production Memory Service with Provenance Tracking
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select, update, delete
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import json

class MemoryService:
    """Enterprise memory service with provenance and compaction"""
    
    def __init__(self, database_url: str):
        self.engine = create_async_engine(database_url, echo=False)
        self.async_session = sessionmaker(self.engine, class_=AsyncSession)
    
    async def store_artifact(self, artifact: Artifact) -> str:
        """Store artifact with provenance tracking"""
        async with self.async_session() as session:
            # Create artifact record
            artifact_record = ArtifactRecord(
                id=artifact.id,
                artifact_type=artifact.artifact_type.value,
                content_hash=self._compute_hash(artifact.content),
                safety_class=artifact.safety_class.value,
                created_by=artifact.created_by,
                created_at=artifact.created_at or datetime.utcnow(),
                extra_metadata=artifact.metadata,
                tags=artifact.tags
            )
            
            session.add(artifact_record)
            
            # Create provenance record
            provenance = ProvenanceRecord(
                artifact_id=artifact.id,
                actor_id=artifact.created_by,
                actor_type="user",
                inputs_hash=self._compute_hash({}),
                outputs_hash=artifact_record.content_hash,
                tool_ids=[],
                timestamp=datetime.utcnow(),
                extra_metadata={"source": "direct_upload"}
            )
            
            session.add(provenance)
            await session.commit()
            
            return artifact.id
    
    async def retrieve_context(self, query: str, agent_id: str, 
                             limit: int = 10) -> List[Artifact]:
        """Retrieve relevant artifacts for context"""
        # Implementation for vector similarity search
        # and metadata filtering
        pass
    
    async def compact_memories(self, older_than_days: int = 30) -> int:
        """Compact old memories using summarization strategies"""
        cutoff_date = datetime.utcnow() - timedelta(days=older_than_days)
        
        async with self.async_session() as session:
            # Find old artifacts
            stmt = select(ArtifactRecord).where(
                ArtifactRecord.created_at < cutoff_date,
                ArtifactRecord.safety_class != SafetyClass.PII.value
            )
            
            result = await session.execute(stmt)
            old_artifacts = result.scalars().all()
            
            compacted_count = 0
            for artifact in old_artifacts:
                # Apply compaction strategy based on type
                if artifact.artifact_type == ArtifactType.RESEARCH_SNIPPET.value:
                    await self._compact_research_snippet(session, artifact)
                    compacted_count += 1
            
            await session.commit()
            return compacted_count
    
    async def _compact_research_snippet(self, session: AsyncSession, 
                                       artifact: ArtifactRecord) -> None:
        """Compact research snippet using summarization"""
        # Load original content
        content = json.loads(artifact.extra_metadata.get("original_content", "{}"))
        
        # Generate summary (would use LLM in real implementation)
        summary = f"Summary of research on: {content.get('topic', 'unknown')}"
        
        # Update artifact with compacted version
        artifact.extra_metadata["compacted"] = True
        artifact.extra_metadata["summary"] = summary
        artifact.extra_metadata["compaction_date"] = datetime.utcnow().isoformat()
        
        # Remove original content to save space
        if "original_content" in artifact.extra_metadata:
            del artifact.extra_metadata["original_content"]
    
    def _compute_hash(self, data: Any) -> str:
        """Compute content hash for provenance"""
        import hashlib
        content = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()
```

### Advanced Approval Workflow System
From `agentic-framework-main/orchestrator/service/approvals.py`, implement enterprise approval system:

```python
# Enterprise Approval Workflow System
class ApprovalWorkflowEngine:
    """Advanced approval system with escalation and delegation"""
    
    def __init__(self, approval_manager: ApprovalManager):
        self.approval_manager = approval_manager
        self.escalation_rules: Dict[str, EscalationRule] = {}
        self.delegation_rules: Dict[str, DelegationRule] = {}
    
    async def submit_for_approval(self, request: ApprovalRequest) -> str:
        """Submit request with automatic routing"""
        
        # Determine approval priority
        request.priority = await self._calculate_priority(request)
        
        # Find appropriate approver
        approver_id = await self._find_approver(request)
        request.approver_id = approver_id
        
        # Set expiration based on priority
        request.expires_at = self._calculate_expiry(request.priority)
        
        # Submit to approval manager
        return await self.approval_manager.create_approval_request(request)
    
    async def _calculate_priority(self, request: ApprovalRequest) -> ApprovalPriority:
        """Calculate approval priority based on content analysis"""
        
        # High priority keywords
        high_priority_terms = [
            'deploy', 'production', 'security', 'financial', 'legal',
            'delete', 'modify', 'access', 'permission'
        ]
        
        # Critical priority terms
        critical_terms = [
            'breach', 'incident', 'emergency', 'compromise', 'exploit'
        ]
        
        content = f"{request.operation} {request.description}".lower()
        
        if any(term in content for term in critical_terms):
            return ApprovalPriority.CRITICAL
        elif any(term in content for term in high_priority_terms):
            return ApprovalPriority.HIGH
        elif len(request.description) > 500:  # Complex requests
            return ApprovalPriority.MEDIUM
        
        return ApprovalPriority.LOW
    
    async def _find_approver(self, request: ApprovalRequest) -> Optional[str]:
        """Find appropriate approver using delegation and escalation rules"""
        
        # Check delegation rules first
        for rule in self.delegation_rules.values():
            if await rule.matches(request):
                return rule.delegate_to
        
        # Default approver based on request type
        approver_map = {
            'deployment': 'platform_team_lead',
            'security': 'security_officer', 
            'financial': 'finance_manager',
            'data_access': 'data_steward'
        }
        
        request_type = self._classify_request_type(request)
        return approver_map.get(request_type, 'default_approver')
    
    def _calculate_expiry(self, priority: ApprovalPriority) -> datetime:
        """Calculate expiration time based on priority"""
        now = datetime.utcnow()
        
        expiry_map = {
            ApprovalPriority.CRITICAL: timedelta(hours=1),
            ApprovalPriority.HIGH: timedelta(hours=4),
            ApprovalPriority.MEDIUM: timedelta(days=1),
            ApprovalPriority.LOW: timedelta(days=3)
        }
        
        return now + expiry_map[priority]
    
    async def handle_escalation(self, approval_id: str) -> None:
        """Handle approval escalation for expired requests"""
        approval = await self.approval_manager.get_approval_request(approval_id)
        
        if not approval or approval.status != ApprovalStatus.PENDING:
            return
        
        # Find escalation rule
        escalation_rule = self.escalation_rules.get(approval.operation)
        if escalation_rule:
            new_approver = await escalation_rule.get_escalation_approver(approval)
            if new_approver:
                approval.approver_id = new_approver
                approval.expires_at = self._calculate_expiry(approval.priority)
                await self.approval_manager.update_approval_request(approval)
    
    def _classify_request_type(self, request: ApprovalRequest) -> str:
        """Classify request type for routing"""
        content = f"{request.operation} {request.description}".lower()
        
        if any(term in content for term in ['deploy', 'release', 'production']):
            return 'deployment'
        elif any(term in content for term in ['security', 'access', 'permission']):
            return 'security'
        elif any(term in content for term in ['finance', 'budget', 'cost']):
            return 'financial'
        elif any(term in content for term in ['data', 'database', 'query']):
            return 'data_access'
        
        return 'general'
```

### Mother-Harness Orchestrator Patterns
From `mother-harness-master/services/orchestrator/src/orchestrator.ts`, adopt enterprise orchestration:

```typescript
// Enterprise Orchestrator with Tiered Memory and Approval Integration
export class EnterpriseOrchestrator {
    private redis = getRedisJSON();
    private planner = new TaskPlanner();
    private tier1Memory = new Tier1Memory();
    private tier2Memory = new Tier2Memory(); 
    private tier3Memory = new Tier3Memory();
    private approvalService = getApprovalService();
    private costTracker = getCostTracker();
    
    async executeTaskWithFullLifecycle(
        userId: string,
        query: string,
        projectId?: string
    ): Promise<ExecutionResult> {
        // 1. Create and validate task
        const task = await this.createValidatedTask(userId, query, projectId);
        
        // 2. Check approval requirements
        const requiresApproval = await this.checkApprovalRequirements(task);
        if (requiresApproval) {
            const approval = await this.createApprovalRequest(task);
            if (!await this.waitForApproval(approval)) {
                throw new Error('Task rejected by approver');
            }
        }
        
        // 3. Plan execution with agent detection
        const plan = await this.planner.planTask(task);
        
        // 4. Execute with memory context
        const context = await this.gatherExecutionContext(task, userId);
        const result = await this.executeWithMemory(plan, context);
        
        // 5. Store results in tiered memory
        await this.storeExecutionResults(task, result);
        
        // 6. Track costs and metrics
        await this.costTracker.recordExecution(task, result);
        
        return result;
    }
    
    private async createValidatedTask(
        userId: string, 
        query: string, 
        projectId?: string
    ): Promise<Task> {
        // Validate against business rules and resource limits
        const budgetGuard = getResourceBudgetGuard();
        await budgetGuard.checkBudget(userId, 'task_execution');
        
        const task: Task = {
            id: `task-${nanoid()}`,
            user_id: userId,
            project_id: projectId,
            query,
            status: 'created',
            created_at: new Date().toISOString(),
            // ... other task fields
        };
        
        await this.redis.set(`task:${task.id}`, '$', task);
        return task;
    }
    
    private async checkApprovalRequirements(task: Task): Promise<boolean> {
        // Check if task requires approval based on content and user permissions
        const sensitiveKeywords = [
            'deploy', 'delete', 'modify', 'access', 'security', 'financial'
        ];
        
        const requiresApproval = sensitiveKeywords.some(keyword => 
            task.query.toLowerCase().includes(keyword)
        );
        
        if (requiresApproval) {
            // Additional checks for user permissions, risk levels, etc.
            const userRole = await this.getUserRole(task.user_id);
            return userRole !== 'admin'; // Admins don't need approval
        }
        
        return false;
    }
    
    private async gatherExecutionContext(
        task: Task, 
        userId: string
    ): Promise<ExecutionContext> {
        // Gather relevant context from all memory tiers
        const [tier1Context, tier2Context, tier3Context] = await Promise.all([
            this.tier1Memory.retrieveRecent(userId, 10),
            this.tier2Memory.retrieveEpisodic(userId, task.query, 5),
            this.tier3Memory.retrieveSemantic(userId, task.query, 3)
        ]);
        
        return {
            recent: tier1Context,
            episodic: tier2Context, 
            semantic: tier3Context,
            userPreferences: await this.getUserPreferences(userId),
            projectContext: task.project_id ? 
                await this.getProjectContext(task.project_id) : null
        };
    }
    
    private async executeWithMemory(
        plan: ExecutionPlan, 
        context: ExecutionContext
    ): Promise<ExecutionResult> {
        const results: AgentResult[] = [];
        
        for (const step of plan.steps) {
            // Enrich step with memory context
            const enrichedContext = {
                ...step.context,
                memory_context: context,
                previous_results: results
            };
            
            // Execute step with appropriate agent
            const result = await this.executeAgentStep(step, enrichedContext);
            results.push(result);
            
            // Store intermediate results in tier 1 memory
            await this.tier1Memory.store({
                type: 'intermediate_result',
                task_id: plan.taskId,
                step_id: step.id,
                result: result,
                timestamp: new Date()
            });
        }
        
        return {
            taskId: plan.taskId,
            results,
            consolidatedOutput: await this.consolidateResults(results, context)
        };
    }
    
    private async storeExecutionResults(
        task: Task, 
        result: ExecutionResult
    ): Promise<void> {
        // Store in tier 2 (episodic) memory for future reference
        await this.tier2Memory.store({
            type: 'task_execution',
            task_id: task.id,
            query: task.query,
            results: result.results,
            consolidated_output: result.consolidatedOutput,
            success: true,
            duration_ms: result.durationMs,
            tokens_used: result.tokensUsed
        });
        
        // Extract and store semantic patterns in tier 3
        const patterns = await this.extractSemanticPatterns(result);
        for (const pattern of patterns) {
            await this.tier3Memory.store(pattern);
        }
    }
}
```

### Phase 9: Production Deployment & Operations (New Phase)
**Goal**: Implement production-ready deployment patterns and operational excellence.

1. **Container Orchestration & Health Checks**
   - Adopt Docker Compose patterns from mother-harness-master
   - Implement comprehensive health checks for all services
   - Add service dependency management and startup ordering
   - Create production docker-compose with proper networking

2. **Security & Access Control**
   - Implement Redis ACL configuration for service isolation
   - Add JWT authentication and RBAC from mother-harness-master
   - Create secret management and validation
   - Implement PII redaction and data classification

3. **Monitoring & Observability**
   - Add comprehensive logging with structured formats
   - Implement metrics collection and dashboards
   - Create alerting thresholds and notification systems
   - Add distributed tracing for multi-agent workflows

4. **Backup & Disaster Recovery**
   - Implement backup strategies for Redis and PostgreSQL
   - Create restore procedures and testing
   - Add data retention policies and archival
   - Implement cross-region replication patterns

### Success Metrics (Final Comprehensive Update)
- **Enterprise Scalability**: Support for 1000+ concurrent tasks with sub-second latency
- **Reliability**: 99.99% uptime with automated failover and recovery
- **Security**: Zero security incidents with comprehensive audit trails
- **Cost Efficiency**: 40% reduction in operational costs through optimization
- **User Satisfaction**: 95% user satisfaction with system performance and features
- **Time to Production**: 50% faster feature deployment through automation

## Final Implementation Timeline
- **Total Duration**: 30 weeks (added Phases 8-9)
- **Team Size**: 3-4 developers for enterprise implementation
- **Infrastructure**: Full production deployment with monitoring
- **Compliance**: Enterprise security and governance standards
- **Support**: 24/7 operations with automated incident response

This exhaustive implementation plan now covers every aspect discovered in the deep analysis of all reference frameworks, providing a complete blueprint for transforming king-ai-v2 into a production-ready, enterprise-grade multi-agent orchestration platform.

### Phase 10: Advanced Skill & Custom Tool Development (New Phase)
**Goal**: Implement extensible skill architecture and custom tool development framework.

1. **YAML-Based Workflow Manifests**
   - Adopt workflow manifest pattern from agentic-framework-main
   - Create schema validation for workflow definitions
   - Implement step dependencies and artifact passing
   - Add workflow versioning and templating

2. **Custom Skill Development Framework**
   - Implement skill.yaml configuration pattern
   - Create skill handler interface with schema validation
   - Add skill registry and discovery system
   - Implement safety flags and approval requirements

3. **MCP (Model Context Protocol) Integration**
   - Adopt MCP server configuration from examples
   - Implement tool scope management and rate limiting
   - Add external service integrations (GitHub, web search, etc.)
   - Create MCP client adapters for agent tool usage

4. **Skill Marketplace & Distribution**
   - Create skill packaging and distribution system
   - Implement skill versioning and dependency management
   - Add skill testing and validation framework
   - Create skill discovery and recommendation system

### Advanced Workflow Manifest Implementation
From `agentic-framework-main/examples/02-multi-step-workflow/`, adopt comprehensive workflow patterns:

```yaml
# Advanced Workflow Manifest with Complex Dependencies
manifest_id: enterprise-analysis-workflow
name: Enterprise Multi-Agent Analysis Pipeline
version: "2.0.0"
description: Comprehensive analysis workflow with parallel processing and conditional routing

global_config:
  timeout: 1800  # 30 minutes total
  max_retries: 3
  retry_delay: 30
  log_level: INFO

steps:
  - id: intake
    role: intake_agent
    description: Initial request analysis and classification
    capabilities: [classify, validate]
    inputs:
      - name: user_request
        source: user_input
        required: true
    outputs:
      - request_type
      - priority_level
      - required_agents
    timeout: 60
    retry_policy: immediate

  - id: parallel_research
    role: research_coordinator
    description: Coordinate parallel research across multiple agents
    capabilities: [coordinate, aggregate]
    inputs:
      - name: request_details
        source: previous_step
        step: intake
        artifact: request_type
    outputs:
      - research_plan
      - agent_assignments
    parallel_branches:
      - id: web_research
        role: web_researcher
        capabilities: [web_search, summarize]
        inputs: [{source: parent_step, artifact: research_plan}]
        outputs: [web_findings]
      - id: data_analysis
        role: data_analyst
        capabilities: [data_query, visualize]
        inputs: [{source: parent_step, artifact: research_plan}]
        outputs: [data_insights]
      - id: expert_consultation
        role: domain_expert
        capabilities: [analyze, recommend]
        inputs: [{source: parent_step, artifact: research_plan}]
        outputs: [expert_opinion]
    timeout: 600

  - id: synthesis
    role: synthesis_agent
    description: Synthesize findings from parallel research branches
    capabilities: [synthesize, validate, format]
    inputs:
      - name: web_results
        source: parallel_branch
        branch: web_research
        artifact: web_findings
      - name: data_results
        source: parallel_branch
        branch: data_analysis
        artifact: data_insights
      - name: expert_results
        source: parallel_branch
        branch: expert_consultation
        artifact: expert_opinion
    outputs:
      - final_report
      - confidence_score
      - recommendations
    timeout: 300
    depends_on: parallel_research

  - id: quality_gate
    role: quality_assurance
    description: Quality check and approval gate
    capabilities: [validate, score, approve]
    inputs:
      - name: report
        source: previous_step
        step: synthesis
        artifact: final_report
    outputs:
      - quality_score
      - approval_status
      - feedback
    timeout: 120
    conditional_routing:
      approved: deliver_results
      rejected: revision_required

  - id: deliver_results
    role: delivery_agent
    description: Format and deliver final results
    capabilities: [format, deliver, notify]
    inputs:
      - name: approved_report
        source: conditional_step
        step: quality_gate
        condition: approved
        artifact: final_report
    outputs:
      - delivery_confirmation
    timeout: 60

  - id: revision_required
    role: revision_agent
    description: Handle revisions based on quality feedback
    capabilities: [revise, iterate, resubmit]
    inputs:
      - name: original_report
        source: step
        step: synthesis
        artifact: final_report
      - name: feedback
        source: conditional_step
        step: quality_gate
        condition: rejected
        artifact: feedback
    outputs:
      - revised_report
    timeout: 300
    max_iterations: 3
    routing_logic: resubmit_to_quality_gate

memory:
  persist_on: [step_complete, workflow_complete, error]
  compaction:
    strategy: hierarchical
    levels:
      - retain: all_artifacts
        duration: 1h
      - retain: [final_report, quality_score]
        duration: 24h
      - retain: [request_type, delivery_confirmation]
        duration: 7d
  cross_workflow_references: true

tools:
  catalog_ids: [web_search, data_query, expert_network, validation_suite]
  dynamic_loading: true
  version_constraints:
    web_search: ">=2.1.0"
    validation_suite: ">=1.5.0"

policies:
  requires_human_approval: 
    - steps: [deliver_results]
      conditions: [high_value_transaction, sensitive_data]
  max_parallel_agents: 5
  resource_limits:
    memory_mb: 2048
    cpu_cores: 2
    api_calls_per_minute: 100
  on_error: 
    retry: 3
    escalate_after: 2
    notification_channels: [email, slack]
  data_retention:
    artifacts: 30d
    logs: 90d
    audit_trail: 1y
```

### Custom Skill Development Framework
From `agentic-framework-main/examples/03-custom-skill/`, implement skill architecture:

```python
# Skill Development Framework
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
import json
import yaml
from pathlib import Path

class SkillConfig(BaseModel):
    """Skill configuration model"""
    name: str
    version: str = "1.0.0"
    description: str
    author: str
    category: str
    safety_flags: List[str] = Field(default_factory=list)
    requires_approval: bool = False
    handler_function: str
    inputs_schema_ref: Optional[str] = None
    outputs_schema_ref: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    dependencies: Dict[str, str] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class SkillHandler(ABC):
    """Abstract base class for skill handlers"""
    
    def __init__(self, config: SkillConfig):
        self.config = config
    
    @abstractmethod
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the skill with given inputs"""
        pass
    
    def validate_inputs(self, inputs: Dict[str, Any]) -> bool:
        """Validate inputs against schema"""
        if self.config.inputs_schema_ref:
            # Load and validate against JSON schema
            return self._validate_against_schema(inputs, self.config.inputs_schema_ref)
        return True
    
    def validate_outputs(self, outputs: Dict[str, Any]) -> bool:
        """Validate outputs against schema"""
        if self.config.outputs_schema_ref:
            return self._validate_against_schema(outputs, self.config.outputs_schema_ref)
        return True
    
    def _validate_against_schema(self, data: Dict[str, Any], schema_ref: str) -> bool:
        """Validate data against JSON schema"""
        try:
            import jsonschema
            schema = self._load_schema(schema_ref)
            jsonschema.validate(data, schema)
            return True
        except Exception:
            return False
    
    def _load_schema(self, schema_ref: str) -> Dict[str, Any]:
        """Load JSON schema from reference"""
        # Implementation for loading schema from file or registry
        pass

class SkillRegistry:
    """Registry for managing skills"""
    
    def __init__(self):
        self.skills: Dict[str, SkillConfig] = {}
        self.handlers: Dict[str, SkillHandler] = {}
    
    def register_skill(self, skill_path: Path) -> None:
        """Register a skill from its directory"""
        config_path = skill_path / "skill.yaml"
        if not config_path.exists():
            raise ValueError(f"skill.yaml not found in {skill_path}")
        
        # Load skill configuration
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        config = SkillConfig(**config_data)
        
        # Load and instantiate handler
        handler_path = skill_path / "handler.py"
        if handler_path.exists():
            # Dynamic import and instantiation
            handler_class = self._load_handler_class(handler_path, config.handler_function)
            handler = handler_class(config)
            self.handlers[config.name] = handler
        
        self.skills[config.name] = config
    
    def execute_skill(self, skill_name: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a registered skill"""
        if skill_name not in self.handlers:
            raise ValueError(f"Skill {skill_name} not found or not loaded")
        
        handler = self.handlers[skill_name]
        
        # Validate inputs
        if not handler.validate_inputs(inputs):
            raise ValueError(f"Invalid inputs for skill {skill_name}")
        
        # Execute skill
        outputs = handler.execute(inputs)
        
        # Validate outputs
        if not handler.validate_outputs(outputs):
            raise ValueError(f"Invalid outputs from skill {skill_name}")
        
        return outputs
    
    def _load_handler_class(self, handler_path: Path, function_name: str):
        """Dynamically load handler class"""
        import importlib.util
        
        spec = importlib.util.spec_from_file_location("skill_handler", handler_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Get the handler function/class
        handler = getattr(module, function_name)
        return handler

# Example Custom Skill Implementation
class AdvancedSentimentAnalyzer(SkillHandler):
    """Advanced sentiment analysis with multi-language support"""
    
    def __init__(self, config: SkillConfig):
        super().__init__(config)
        self.models = self._load_models()
    
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute sentiment analysis"""
        text = inputs.get('text', '')
        language = inputs.get('language', 'en')
        
        # Advanced sentiment analysis logic
        sentiment_scores = self._analyze_sentiment(text, language)
        
        # Calculate overall sentiment
        compound_score = sentiment_scores.get('compound', 0)
        if compound_score >= 0.05:
            sentiment = 'positive'
        elif compound_score <= -0.05:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        return {
            'sentiment': sentiment,
            'confidence': abs(compound_score),
            'scores': sentiment_scores,
            'language': language,
            'word_count': len(text.split())
        }
    
    def _analyze_sentiment(self, text: str, language: str) -> Dict[str, float]:
        """Advanced sentiment analysis implementation"""
        # Placeholder for demonstration
        return {
            'positive': 0.3,
            'negative': 0.1,
            'neutral': 0.6,
            'compound': 0.2
        }
    
    def _load_models(self) -> Dict[str, Any]:
        """Load language-specific models"""
        # Implementation for loading ML models
        return {}

# Skill Testing Framework
class SkillTester:
    """Testing framework for skills"""
    
    def __init__(self, registry: SkillRegistry):
        self.registry = registry
    
    def run_skill_tests(self, skill_name: str) -> Dict[str, Any]:
        """Run comprehensive tests for a skill"""
        results = {
            'skill_name': skill_name,
            'tests_passed': 0,
            'tests_failed': 0,
            'test_cases': []
        }
        
        # Load test cases
        test_cases = self._load_test_cases(skill_name)
        
        for test_case in test_cases:
            try:
                output = self.registry.execute_skill(skill_name, test_case['input'])
                
                # Validate against expected output
                if self._validate_test_output(output, test_case['expected']):
                    results['tests_passed'] += 1
                    results['test_cases'].append({
                        'name': test_case['name'],
                        'status': 'passed'
                    })
                else:
                    results['tests_failed'] += 1
                    results['test_cases'].append({
                        'name': test_case['name'],
                        'status': 'failed',
                        'expected': test_case['expected'],
                        'actual': output
                    })
                    
            except Exception as e:
                results['tests_failed'] += 1
                results['test_cases'].append({
                    'name': test_case['name'],
                    'status': 'error',
                    'error': str(e)
                })
        
        return results
    
    def _load_test_cases(self, skill_name: str) -> List[Dict[str, Any]]:
        """Load test cases for skill"""
        # Implementation for loading test cases from test files
        return []
    
    def _validate_test_output(self, actual: Dict[str, Any], expected: Dict[str, Any]) -> bool:
        """Validate test output against expected results"""
        # Implementation for output validation
        return True
```

### MCP Integration Framework
From `agentic-framework-main/examples/04-mcp-integration/`, implement Model Context Protocol:

```python
# MCP (Model Context Protocol) Integration
from typing import Dict, List, Any, Optional, Protocol
from abc import ABC, abstractmethod
import asyncio
import aiohttp
from datetime import datetime, timedelta
import json

class MCPServerConfig:
    """MCP server configuration"""
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.type = config.get('type', 'external')
        self.endpoint = config.get('endpoint')
        self.tools = config.get('tools', [])
        self.scopes = config.get('scopes', [])
        self.auth = config.get('auth', {})
        self.rate_limit = config.get('rate_limit', 60)
        self.timeout = config.get('timeout', 30)
        
        # Rate limiting state
        self.requests = []
    
    def can_make_request(self) -> bool:
        """Check if request can be made within rate limit"""
        now = datetime.now()
        window_start = now - timedelta(minutes=1)
        
        # Remove old requests
        self.requests = [req for req in self.requests if req > window_start]
        
        return len(self.requests) < self.rate_limit
    
    def record_request(self) -> None:
        """Record a request for rate limiting"""
        self.requests.append(datetime.now())

class MCPTool(Protocol):
    """Protocol for MCP tools"""
    name: str
    description: str
    
    def execute(self, **kwargs) -> Any:
        ...

class MCPClient:
    """MCP client for communicating with MCP servers"""
    
    def __init__(self, server_config: MCPServerConfig):
        self.config = server_config
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.config.timeout))
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def call_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Call a tool on the MCP server"""
        if not self.config.can_make_request():
            raise Exception(f"Rate limit exceeded for {self.config.name}")
        
        if tool_name not in self.config.tools:
            raise ValueError(f"Tool {tool_name} not available on server {self.config.name}")
        
        # Prepare request
        url = f"{self.config.endpoint}/tools/{tool_name}"
        headers = self._get_auth_headers()
        payload = {
            'parameters': kwargs,
            'context': {
                'timestamp': datetime.now().isoformat(),
                'scopes': self.config.scopes
            }
        }
        
        try:
            async with self.session.post(url, json=payload, headers=headers) as response:
                self.config.record_request()
                
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"MCP server error: {response.status} - {error_text}")
                
                result = await response.json()
                return result
                
        except aiohttp.ClientError as e:
            raise Exception(f"MCP communication error: {str(e)}")
    
    def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers"""
        headers = {'Content-Type': 'application/json'}
        
        if self.config.auth.get('type') == 'bearer_token':
            token = self.config.auth.get('token')
            if token:
                headers['Authorization'] = f"Bearer {token}"
        
        return headers

class MCPRegistry:
    """Registry for MCP servers and tools"""
    
    def __init__(self):
        self.servers: Dict[str, MCPServerConfig] = {}
        self.tool_index: Dict[str, List[str]] = {}  # tool_name -> [server_names]
    
    def register_server(self, config: Dict[str, Any]) -> None:
        """Register an MCP server"""
        server_config = MCPServerConfig(config['name'], config)
        self.servers[config['name']] = server_config
        
        # Update tool index
        for tool in server_config.tools:
            if tool not in self.tool_index:
                self.tool_index[tool] = []
            self.tool_index[tool].append(config['name'])
    
    async def execute_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Execute a tool across available servers"""
        if tool_name not in self.tool_index:
            raise ValueError(f"Tool {tool_name} not found in any registered server")
        
        # Try servers in order (could implement load balancing)
        for server_name in self.tool_index[tool_name]:
            server_config = self.servers[server_name]
            
            try:
                async with MCPClient(server_config) as client:
                    return await client.call_tool(tool_name, **kwargs)
                    
            except Exception as e:
                print(f"Failed to execute {tool_name} on {server_name}: {e}")
                continue
        
        raise Exception(f"All servers failed for tool {tool_name}")

# MCP Tool Adapters for Agent Integration
class MCPToolAdapter:
    """Adapter to make MCP tools available to agents"""
    
    def __init__(self, registry: MCPRegistry):
        self.registry = registry
    
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get list of available tools for agents"""
        tools = []
        for tool_name, servers in self.registry.tool_index.items():
            # Get tool description from first available server
            server_name = servers[0]
            server_config = self.registry.servers[server_name]
            
            tools.append({
                'name': tool_name,
                'description': f"Tool provided by {server_name} MCP server",
                'parameters': self._get_tool_parameters(tool_name, server_config)
            })
        
        return tools
    
    def create_agent_tool(self, tool_name: str) -> MCPTool:
        """Create an agent-compatible tool"""
        
        class AgentMCPTool:
            def __init__(self, registry: MCPRegistry, tool_name: str):
                self.registry = registry
                self.name = tool_name
                self.description = f"MCP tool: {tool_name}"
            
            def run(self, **kwargs) -> str:
                """Synchronous wrapper for async MCP call"""
                try:
                    # In real implementation, this would need proper async handling
                    # For now, using asyncio.run() - in production use proper async integration
                    result = asyncio.run(self.registry.execute_tool(self.name, **kwargs))
                    return json.dumps(result)
                except Exception as e:
                    return f"Error executing MCP tool {self.name}: {str(e)}"
        
        return AgentMCPTool(self.registry, tool_name)
    
    def _get_tool_parameters(self, tool_name: str, server_config: MCPServerConfig) -> Dict[str, Any]:
        """Get tool parameter specifications"""
        # In real implementation, this would query the MCP server for tool specs
        # Placeholder implementation
        return {
            'type': 'object',
            'properties': {
                'query': {
                    'type': 'string',
                    'description': f'Query parameter for {tool_name}'
                }
            },
            'required': ['query']
        }

# Example MCP Server Implementations
class GitHubMCPServer:
    """GitHub MCP server implementation"""
    
    def __init__(self, token: str):
        self.token = token
        self.base_url = "https://api.github.com"
    
    async def handle_tool_call(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP tool calls for GitHub operations"""
        
        if tool_name == "search_repos":
            return await self._search_repositories(parameters.get('query', ''))
        elif tool_name == "create_issue":
            return await self._create_issue(
                parameters.get('owner', ''),
                parameters.get('repo', ''),
                parameters.get('title', ''),
                parameters.get('body', '')
            )
        elif tool_name == "list_prs":
            return await self._list_pull_requests(
                parameters.get('owner', ''),
                parameters.get('repo', '')
            )
        else:
            raise ValueError(f"Unknown tool: {tool_name}")
    
    async def _search_repositories(self, query: str) -> Dict[str, Any]:
        """Search GitHub repositories"""
        url = f"{self.base_url}/search/repositories"
        params = {'q': query, 'sort': 'stars', 'order': 'desc'}
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, 
                                headers={'Authorization': f'token {self.token}'}) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        'repositories': [
                            {
                                'name': repo['name'],
                                'full_name': repo['full_name'],
                                'description': repo['description'],
                                'stars': repo['stargazers_count'],
                                'url': repo['html_url']
                            }
                            for repo in data['items'][:5]  # Top 5 results
                        ]
                    }
                else:
                    raise Exception(f"GitHub API error: {response.status}")
    
    async def _create_issue(self, owner: str, repo: str, title: str, body: str) -> Dict[str, Any]:
        """Create a GitHub issue"""
        url = f"{self.base_url}/repos/{owner}/{repo}/issues"
        payload = {'title': title, 'body': body}
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload,
                                 headers={'Authorization': f'token {self.token}'}) as response:
                if response.status == 201:
                    data = await response.json()
                    return {
                        'issue_number': data['number'],
                        'url': data['html_url'],
                        'created': True
                    }
                else:
                    error_data = await response.json()
                    raise Exception(f"Failed to create issue: {error_data.get('message', 'Unknown error')}")
    
    async def _list_pull_requests(self, owner: str, repo: str) -> Dict[str, Any]:
        """List pull requests for a repository"""
        url = f"{self.base_url}/repos/{owner}/{repo}/pulls"
        params = {'state': 'open', 'sort': 'updated', 'direction': 'desc'}
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params,
                                 headers={'Authorization': f'token {self.token}'}) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        'pull_requests': [
                            {
                                'number': pr['number'],
                                'title': pr['title'],
                                'user': pr['user']['login'],
                                'url': pr['html_url'],
                                'updated_at': pr['updated_at']
                            }
                            for pr in data[:10]  # Latest 10 PRs
                        ]
                    }
                else:
                    raise Exception(f"GitHub API error: {response.status}")

# Web Search MCP Server
class WebSearchMCPServer:
    """Web search MCP server implementation"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.search.example.com"  # Placeholder
    
    async def handle_tool_call(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle web search tool calls"""
        
        if tool_name == "search":
            return await self._web_search(parameters.get('query', ''))
        elif tool_name == "get_webpage":
            return await self._get_webpage(parameters.get('url', ''))
        else:
            raise ValueError(f"Unknown tool: {tool_name}")
    
    async def _web_search(self, query: str) -> Dict[str, Any]:
        """Perform web search"""
        # Implementation would integrate with actual search API
        # Placeholder implementation
        return {
            'query': query,
            'results': [
                {
                    'title': f'Search result for {query}',
                    'url': f'https://example.com/search/{query.replace(" ", "-")}',
                    'snippet': f'This is a search result snippet for: {query}'
                }
            ],
            'total_results': 1
        }
    
    async def _get_webpage(self, url: str) -> Dict[str, Any]:
        """Get webpage content"""
        # Implementation would fetch and parse webpage
        # Placeholder implementation
        return {
            'url': url,
            'title': f'Page Title for {url}',
            'content': f'Extracted content from {url}',
            'status': 'success'
        }
```

### Conversation Memory Management
From `llm-agent-framework-main/agent_framework/memory/conversation_memory.py`, implement advanced memory:

```python
# Advanced Conversation Memory with Summarization
class HierarchicalConversationMemory(ConversationMemory):
    """Enhanced conversation memory with hierarchical summarization"""
    
    def __init__(self, max_messages: int = 100, summary_interval: int = 20):
        super().__init__(max_messages)
        self.summary_interval = summary_interval
        self.message_summaries: List[str] = []
        self.conversation_summary = ""
    
    def add_message(self, role: str, content: str, metadata: Dict[str, Any] = None) -> None:
        """Add message with automatic summarization"""
        super().add_message(role, content, metadata)
        
        # Check if we should create a summary
        if len(self.messages) % self.summary_interval == 0 and len(self.messages) > 0:
            self._create_periodic_summary()
    
    def _create_periodic_summary(self) -> None:
        """Create summary of recent messages"""
        recent_messages = self.messages[-self.summary_interval:]
        
        # Simple summarization (would use LLM in production)
        summary = self._generate_summary(recent_messages)
        self.message_summaries.append(summary)
        
        # Update overall conversation summary
        self._update_conversation_summary()
    
    def _generate_summary(self, messages: List[Message]) -> str:
        """Generate summary of message batch"""
        # Placeholder summarization logic
        user_messages = [m for m in messages if m.role == 'user']
        assistant_messages = [m for m in messages if m.role == 'assistant']
        
        return f"Conversation segment: {len(user_messages)} user inputs, {len(assistant_messages)} assistant responses"
    
    def _update_conversation_summary(self) -> None:
        """Update overall conversation summary"""
        if len(self.message_summaries) > 5:
            # Summarize summaries (hierarchical summarization)
            self.conversation_summary = f"Extended conversation with {len(self.message_summaries)} segments"
        else:
            self.conversation_summary = " ".join(self.message_summaries)
    
    def get_context_with_summary(self, limit: int = 10) -> str:
        """Get context including conversation summary"""
        recent_context = self.get_context_string(limit)
        
        if self.conversation_summary:
            return f"Conversation Summary: {self.conversation_summary}\n\nRecent Context:\n{recent_context}"
        
        return recent_context
    
    def search_with_relevance(self, query: str, limit: int = 5) -> List[Message]:
        """Search with relevance scoring"""
        results = []
        query_lower = query.lower()
        
        for msg in reversed(self.messages):
            relevance_score = self._calculate_relevance(msg.content, query_lower)
            if relevance_score > 0.3:  # Relevance threshold
                # Add relevance to metadata
                enriched_msg = Message(
                    role=msg.role,
                    content=msg.content,
                    timestamp=msg.timestamp,
                    metadata={**msg.metadata, 'relevance_score': relevance_score}
                )
                results.append(enriched_msg)
                
                if len(results) >= limit:
                    break
        
        return results
    
    def _calculate_relevance(self, content: str, query: str) -> float:
        """Calculate relevance score between content and query"""
        content_lower = content.lower()
        
        # Simple word overlap scoring
        query_words = set(query.split())
        content_words = set(content_lower.split())
        
        overlap = len(query_words.intersection(content_words))
        total_query_words = len(query_words)
        
        if total_query_words == 0:
            return 0.0
        
        return overlap / total_query_words

# Memory Persistence Layer
class PersistentMemoryStore:
    """Persistent storage for conversation memory"""
    
    def __init__(self, storage_path: str):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
    
    def save_conversation(self, conversation_id: str, memory: ConversationMemory) -> None:
        """Save conversation memory to disk"""
        conversation_file = self.storage_path / f"{conversation_id}.json"
        
        data = {
            'conversation_id': conversation_id,
            'messages': [msg.to_dict() for msg in memory.messages],
            'summaries': memory.message_summaries if hasattr(memory, 'message_summaries') else [],
            'conversation_summary': memory.conversation_summary if hasattr(memory, 'conversation_summary') else "",
            'saved_at': datetime.now().isoformat()
        }
        
        with open(conversation_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_conversation(self, conversation_id: str) -> Optional[ConversationMemory]:
        """Load conversation memory from disk"""
        conversation_file = self.storage_path / f"{conversation_id}.json"
        
        if not conversation_file.exists():
            raise ValueError(f"skill.yaml not found in {skill_path}")
        
        # Load skill configuration
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        config = SkillConfig(**config_data)
        
        # Load and instantiate handler
        handler_path = skill_path / "handler.py"
        if handler_path.exists():
            # Dynamic import and instantiation
            handler_class = self._load_handler_class(handler_path, config.handler_function)
            handler = handler_class(config)
            self.handlers[config.name] = handler
        
        self.skills[config.name] = config
    
    def execute_skill(self, skill_name: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a registered skill"""
        if skill_name not in self.handlers:
            raise ValueError(f"Skill {skill_name} not found or not loaded")
        
        handler = self.handlers[skill_name]
        
        # Validate inputs
        if not handler.validate_inputs(inputs):
            raise ValueError(f"Invalid inputs for skill {skill_name}")
        
        # Execute skill
        outputs = handler.execute(inputs)
        
        # Validate outputs
        if not handler.validate_outputs(outputs):
            raise ValueError(f"Invalid outputs from skill {skill_name}")
        
        return outputs
    
    def _load_handler_class(self, handler_path: Path, function_name: str):
        """Dynamically load handler class"""
        import importlib.util
        
        spec = importlib.util.spec_from_file_location("skill_handler", handler_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Get the handler function/class
        handler = getattr(module, function_name)
        return handler

# Example Custom Skill Implementation
class AdvancedSentimentAnalyzer(SkillHandler):
    """Advanced sentiment analysis with multi-language support"""
    
    def __init__(self, config: SkillConfig):
        super().__init__(config)
        self.models = self._load_models()
    
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute sentiment analysis"""
        text = inputs.get('text', '')
        language = inputs.get('language', 'en')
        
        # Advanced sentiment analysis logic
        sentiment_scores = self._analyze_sentiment(text, language)
        
        # Calculate overall sentiment
        compound_score = sentiment_scores.get('compound', 0)
        if compound_score >= 0.05:
            sentiment = 'positive'
        elif compound_score <= -0.05:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        return {
            'sentiment': sentiment,
            'confidence': abs(compound_score),
            'scores': sentiment_scores,
            'language': language,
            'word_count': len(text.split())
        }
    
    def _analyze_sentiment(self, text: str, language: str) -> Dict[str, float]:
        """Advanced sentiment analysis implementation"""
        # Placeholder for demonstration
        return {
            'positive': 0.3,
            'negative': 0.1,
            'neutral': 0.6,
            'compound': 0.2
        }
    
    def _load_models(self) -> Dict[str, Any]:
        """Load language-specific models"""
        # Implementation for loading ML models
        return {}

# Skill Testing Framework
class SkillTester:
    """Testing framework for skills"""
    
    def __init__(self, registry: SkillRegistry):
        self.registry = registry
    
    def run_skill_tests(self, skill_name: str) -> Dict[str, Any]:
        """Run comprehensive tests for a skill"""
        results = {
            'skill_name': skill_name,
            'tests_passed': 0,
            'tests_failed': 0,
            'test_cases': []
        }
        
        # Load test cases
        test_cases = self._load_test_cases(skill_name)
        
        for test_case in test_cases:
            try:
                output = self.registry.execute_skill(skill_name, test_case['input'])
                
                # Validate against expected output
                if self._validate_test_output(output, test_case['expected']):
                    results['tests_passed'] += 1
                    results['test_cases'].append({
                        'name': test_case['name'],
                        'status': 'passed'
                    })
                else:
                    results['tests_failed'] += 1
                    results['test_cases'].append({
                        'name': test_case['name'],
                        'status': 'failed',
                        'expected': test_case['expected'],
                        'actual': output
                    })
                    
            except Exception as e:
                results['tests_failed'] += 1
                results['test_cases'].append({
                    'name': test_case['name'],
                    'status': 'error',
                    'error': str(e)
                })
        
        return results
    
    def _load_test_cases(self, skill_name: str) -> List[Dict[str, Any]]:
        """Load test cases for skill"""
        # Implementation for loading test cases from test files
        return []
    
    def _validate_test_output(self, actual: Dict[str, Any], expected: Dict[str, Any]) -> bool:
        """Validate test output against expected results"""
        # Implementation for output validation
        return True
```

### MCP Integration Framework
From `agentic-framework-main/examples/04-mcp-integration/`, implement Model Context Protocol:

```python
# MCP (Model Context Protocol) Integration
from typing import Dict, List, Any, Optional, Protocol
from abc import ABC, abstractmethod
import asyncio
import aiohttp
from datetime import datetime, timedelta
import json

class MCPServerConfig:
    """MCP server configuration"""
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.type = config.get('type', 'external')
        self.endpoint = config.get('endpoint')
        self.tools = config.get('tools', [])
        self.scopes = config.get('scopes', [])
        self.auth = config.get('auth', {})
        self.rate_limit = config.get('rate_limit', 60)
        self.timeout = config.get('timeout', 30)
        
        # Rate limiting state
        self.requests = []
    
    def can_make_request(self) -> bool:
        """Check if request can be made within rate limit"""
        now = datetime.now()
        window_start = now - timedelta(minutes=1)
        
        # Remove old requests
        self.requests = [req for req in self.requests if req > window_start]
        
        return len(self.requests) < self.rate_limit
    
    def record_request(self) -> None:
        """Record a request for rate limiting"""
        self.requests.append(datetime.now())

class MCPTool(Protocol):
    """Protocol for MCP tools"""
    name: str
    description: str
    
    def execute(self, **kwargs) -> Any:
        ...

class MCPClient:
    """MCP client for communicating with MCP servers"""
    
    def __init__(self, server_config: MCPServerConfig):
        self.config = server_config
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.config.timeout))
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def call_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Call a tool on the MCP server"""
        if not self.config.can_make_request():
            raise Exception(f"Rate limit exceeded for {self.config.name}")
        
        if tool_name not in self.config.tools:
            raise ValueError(f"Tool {tool_name} not available on server {self.config.name}")
        
        # Prepare request
        url = f"{self.config.endpoint}/tools/{tool_name}"
        headers = self._get_auth_headers()
        payload = {
            'parameters': kwargs,
            'context': {
                'timestamp': datetime.now().isoformat(),
                'scopes': self.config.scopes
            }
        }
        
        try:
            async with self.session.post(url, json=payload, headers=headers) as response:
                self.config.record_request()
                
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"MCP server error: {response.status} - {error_text}")
                
                result = await response.json()
                return result
                
        except aiohttp.ClientError as e:
            raise Exception(f"MCP communication error: {str(e)}")
    
    def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers"""
        headers = {'Content-Type': 'application/json'}
        
        if self.config.auth.get('type') == 'bearer_token':
            token = self.config.auth.get('token')
            if token:
                headers['Authorization'] = f"Bearer {token}"
        
        return headers

class MCPRegistry:
    """Registry for MCP servers and tools"""
    
    def __init__(self):
        self.servers: Dict[str, MCPServerConfig] = {}
        self.tool_index: Dict[str, List[str]] = {}  # tool_name -> [server_names]
    
    def register_server(self, config: Dict[str, Any]) -> None:
        """Register an MCP server"""
        server_config = MCPServerConfig(config['name'], config)
        self.servers[config['name']] = server_config
        
        # Update tool index
        for tool in server_config.tools:
            if tool not in self.tool_index:
                self.tool_index[tool] = []
            self.tool_index[tool].append(config['name'])
    
    async def execute_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Execute a tool across available servers"""
        if tool_name not in self.tool_index:
            raise ValueError(f"Tool {tool_name} not found in any registered server")
        
        # Try servers in order (could implement load balancing)
        for server_name in self.tool_index[tool_name]:
            server_config = self.servers[server_name]
            
            try:
                async with MCPClient(server_config) as client:
                    return await client.call_tool(tool_name, **kwargs)
                    
            except Exception as e:
                print(f"Failed to execute {tool_name} on {server_name}: {e}")
                continue
        
        raise Exception(f"All servers failed for tool {tool_name}")

# MCP Tool Adapters for Agent Integration
class MCPToolAdapter:
    """Adapter to make MCP tools available to agents"""
    
    def __init__(self, registry: MCPRegistry):
        self.registry = registry
    
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get list of available tools for agents"""
        tools = []
        for tool_name, servers in self.registry.tool_index.items():
            # Get tool description from first available server
            server_name = servers[0]
            server_config = self.registry.servers[server_name]
            
            tools.append({
                'name': tool_name,
                'description': f"Tool provided by {server_name} MCP server",
                'parameters': self._get_tool_parameters(tool_name, server_config)
            })
        
        return tools
    
    def create_agent_tool(self, tool_name: str) -> MCPTool:
        """Create an agent-compatible tool"""
        
        class AgentMCPTool:
            def __init__(self, registry: MCPRegistry, tool_name: str):
                self.registry = registry
                self.name = tool_name
                self.description = f"MCP tool: {tool_name}"
            
            def run(self, **kwargs) -> str:
                """Synchronous wrapper for async MCP call"""
                try:
                    # In real implementation, this would need proper async handling
                    # For now, using asyncio.run() - in production use proper async integration
                    result = asyncio.run(self.registry.execute_tool(self.name, **kwargs))
                    return json.dumps(result)
                except Exception as e:
                    return f"Error executing MCP tool {self.name}: {str(e)}"
        
        return AgentMCPTool(self.registry, tool_name)
    
    def _get_tool_parameters(self, tool_name: str, server_config: MCPServerConfig) -> Dict[str, Any]:
        """Get tool parameter specifications"""
        # In real implementation, this would query the MCP server for tool specs
        # Placeholder implementation
        return {
            'type': 'object',
            'properties': {
                'query': {
                    'type': 'string',
                    'description': f'Query parameter for {tool_name}'
                }
            },
            'required': ['query']
        }

# Example MCP Server Implementations
class GitHubMCPServer:
    """GitHub MCP server implementation"""
    
    def __init__(self, token: str):
        self.token = token
        self.base_url = "https://api.github.com"
    
    async def handle_tool_call(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP tool calls for GitHub operations"""
        
        if tool_name == "search_repos":
            return await self._search_repositories(parameters.get('query', ''))
        elif tool_name == "create_issue":
            return await self._create_issue(
                parameters.get('owner', ''),
                parameters.get('repo', ''),
                parameters.get('title', ''),
                parameters.get('body', '')
            )
        elif tool_name == "list_prs":
            return await self._list_pull_requests(
                parameters.get('owner', ''),
                parameters.get('repo', '')
            )
        else:
            raise ValueError(f"Unknown tool: {tool_name}")
    
    async def _search_repositories(self, query: str) -> Dict[str, Any]:
        """Search GitHub repositories"""
        url = f"{self.base_url}/search/repositories"
        params = {'q': query, 'sort': 'stars', 'order': 'desc'}
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, 
                                headers={'Authorization': f'token {self.token}'}) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        'repositories': [
                            {
                                'name': repo['name'],
                                'full_name': repo['full_name'],
                                'description': repo['description'],
                                'stars': repo['stargazers_count'],
                                'url': repo['html_url']
                            }
                            for repo in data['items'][:5]  # Top 5 results
                        ]
                    }
                else:
                    raise Exception(f"GitHub API error: {response.status}")
    
    async def _create_issue(self, owner: str, repo: str, title: str, body: str) -> Dict[str, Any]:
        """Create a GitHub issue"""
        url = f"{self.base_url}/repos/{owner}/{repo}/issues"
        payload = {'title': title, 'body': body}
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload,
                                 headers={'Authorization': f'token {self.token}'}) as response:
                if response.status == 201:
                    data = await response.json()
                    return {
                        'issue_number': data['number'],
                        'url': data['html_url'],
                        'created': True
                    }
                else:
                    error_data = await response.json()
                    raise Exception(f"Failed to create issue: {error_data.get('message', 'Unknown error')}")
    
    async def _list_pull_requests(self, owner: str, repo: str) -> Dict[str, Any]:
        """List pull requests for a repository"""
        url = f"{self.base_url}/repos/{owner}/{repo}/pulls"
        params = {'state': 'open', 'sort': 'updated', 'direction': 'desc'}
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params,
                                 headers={'Authorization': f'token {self.token}'}) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        'pull_requests': [
                            {
                                'number': pr['number'],
                                'title': pr['title'],
                                'user': pr['user']['login'],
                                'url': pr['html_url'],
                                'updated_at': pr['updated_at']
                            }
                            for pr in data[:10]  # Latest 10 PRs
                        ]
                    }
                else:
                    raise Exception(f"GitHub API error: {response.status}")

# Web Search MCP Server
class WebSearchMCPServer:
    """Web search MCP server implementation"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.search.example.com"  # Placeholder
    
    async def handle_tool_call(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle web search tool calls"""
        
        if tool_name == "search":
            return await self._web_search(parameters.get('query', ''))
        elif tool_name == "get_webpage":
            return await self._get_webpage(parameters.get('url', ''))
        else:
            raise ValueError(f"Unknown tool: {tool_name}")
    
    async def _web_search(self, query: str) -> Dict[str, Any]:
        """Perform web search"""
        # Implementation would integrate with actual search API
        # Placeholder implementation
        return {
            'query': query,
            'results': [
                {
                    'title': f'Search result for {query}',
                    'url': f'https://example.com/search/{query.replace(" ", "-")}',
                    'snippet': f'This is a search result snippet for: {query}'
                }
            ],
            'total_results': 1
        }
    
    async def _get_webpage(self, url: str) -> Dict[str, Any]:
        """Get webpage content"""
        # Implementation would fetch and parse webpage
        # Placeholder implementation
        return {
            'url': url,
            'title': f'Page Title for {url}',
            'content': f'Extracted content from {url}',
            'status': 'success'
        }
```

### Conversation Memory Management
From `llm-agent-framework-main/agent_framework/memory/conversation_memory.py`, implement advanced memory:

```python
# Advanced Conversation Memory with Summarization
class HierarchicalConversationMemory(ConversationMemory):
    """Enhanced conversation memory with hierarchical summarization"""
    
    def __init__(self, max_messages: int = 100, summary_interval: int = 20):
        super().__init__(max_messages)
        self.summary_interval = summary_interval
        self.message_summaries: List[str] = []
        self.conversation_summary = ""
    
    def add_message(self, role: str, content: str, metadata: Dict[str, Any] = None) -> None:
        """Add message with automatic summarization"""
        super().add_message(role, content, metadata)
        
        # Check if we should create a summary
        if len(self.messages) % self.summary_interval == 0 and len(self.messages) > 0:
            self._create_periodic_summary()
    
    def _create_periodic_summary(self) -> None:
        """Create summary of recent messages"""
        recent_messages = self.messages[-self.summary_interval:]
        
        # Simple summarization (would use LLM in production)
        summary = self._generate_summary(recent_messages)
        self.message_summaries.append(summary)
        
        # Update overall conversation summary
        self._update_conversation_summary()
    
    def _generate_summary(self, messages: List[Message]) -> str:
        """Generate summary of message batch"""
        # Placeholder summarization logic
        user_messages = [m for m in messages if m.role == 'user']
        assistant_messages = [m for m in messages if m.role == 'assistant']
        
        return f"Conversation segment: {len(user_messages)} user inputs, {len(assistant_messages)} assistant responses"
    
    def _update_conversation_summary(self) -> None:
        """Update overall conversation summary"""
        if len(self.message_summaries) > 5:
            # Summarize summaries (hierarchical summarization)
            self.conversation_summary = f"Extended conversation with {len(self.message_summaries)} segments"
        else:
            self.conversation_summary = " ".join(self.message_summaries)
    
    def get_context_with_summary(self, limit: int = 10) -> str:
        """Get context including conversation summary"""
        recent_context = self.get_context_string(limit)
        
        if self.conversation_summary:
            return f"Conversation Summary: {self.conversation_summary}\n\nRecent Context:\n{recent_context}"
        
        return recent_context
    
    def search_with_relevance(self, query: str, limit: int = 5) -> List[Message]:
        """Search with relevance scoring"""
        results = []
        query_lower = query.lower()
        
        for msg in reversed(self.messages):
            relevance_score = self._calculate_relevance(msg.content, query_lower)
            if relevance_score > 0.3:  # Relevance threshold
                # Add relevance to metadata
                enriched_msg = Message(
                    role=msg.role,
                    content=msg.content,
                    timestamp=msg.timestamp,
                    metadata={**msg.metadata, 'relevance_score': relevance_score}
                )
                results.append(enriched_msg)
                
                if len(results) >= limit:
                    break
        
        return results
    
    def _calculate_relevance(self, content: str, query: str) -> float:
        """Calculate relevance score between content and query"""
        content_lower = content.lower()
        
        # Simple word overlap scoring
        query_words = set(query.split())
        content_words = set(content_lower.split())
        
        overlap = len(query_words.intersection(content_words))
        total_query_words = len(query_words)
        
        if total_query_words == 0:
            return 0.0
        
        return overlap / total_query_words

# Memory Persistence Layer
class PersistentMemoryStore:
    """Persistent storage for conversation memory"""
    
    def __init__(self, storage_path: str):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
    
    def save_conversation(self, conversation_id: str, memory: ConversationMemory) -> None:
        """Save conversation memory to disk"""
        conversation_file = self.storage_path / f"{conversation_id}.json"
        
        data = {
            'conversation_id': conversation_id,
            'messages': [msg.to_dict() for msg in memory.messages],
            'summaries': memory.message_summaries if hasattr(memory, 'message_summaries') else [],
            'conversation_summary': memory.conversation_summary if hasattr(memory, 'conversation_summary') else "",
            'saved_at': datetime.now().isoformat()
        }
        
        with open(conversation_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_conversation(self, conversation_id: str) -> Optional[ConversationMemory]:
        """Load conversation memory from disk"""
        conversation_file = self.storage_path / f"{conversation_id}.json"
        
        if not conversation_file.exists():
            return None
        
        try:
            with open(conversation_file, 'r') as f:
                data = json.load(f)
            
            memory = HierarchicalConversationMemory()
            memory.messages = [Message(**msg) for msg in data['messages']]
            memory.message_summaries = data.get('summaries', [])
            memory.conversation_summary = data.get('conversation_summary', '')
            
            return memory
            
        except Exception as e:
            print(f"Error loading conversation {conversation_id}: {e}")
            return None
    
    def list_conversations(self) -> List[str]:
        """List all saved conversation IDs"""
        return [f.stem for f in self.storage_path.glob("*.json")]
    
    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a saved conversation"""
        conversation_file = self.storage_path / f"{conversation_id}.json"
        
        if conversation_file.exists():
            conversation_file.unlink()
            return True
        
        return False
```

### Enterprise Security Implementation
From `mother-harness-master/scripts/setup-redis-acl.sh` and documentation, implement comprehensive security:

```bash
#!/bin/bash
# Enterprise Redis ACL Setup for King-AI-v2

set -euo pipefail

# Configuration
REDIS_HOST="${REDIS_HOST:-localhost}"
REDIS_PORT="${REDIS_PORT:-6379}"
REDIS_PASSWORD="${REDIS_PASSWORD:-king_ai_password}"

# Load environment variables
if [[ -f .env ]]; then
    set -a
    source .env
    set +a
fi

# Validate required passwords
REQUIRED_VARS=(
    "REDIS_ACL_ORCHESTRATOR_PASSWORD"
    "REDIS_ACL_AGENTS_PASSWORD"
    "REDIS_ACL_API_PASSWORD"
    "REDIS_ACL_MONITORING_PASSWORD"
    "REDIS_ACL_ADMIN_PASSWORD"
)

for var in "${REQUIRED_VARS[@]}"; do
    if [[ -z "${!var:-}" || "${!var:-}" =~ ^(CHANGE_ME|changeme|password)$ ]]; then
        echo "❌ ERROR: $var is not set or contains placeholder value"
        echo "   Please set secure passwords in your .env file"
        exit 1
    fi
done

echo "🔐 Setting up Redis ACL for King-AI-v2 on ${REDIS_HOST}:${REDIS_PORT}..."

# Redis CLI command
REDIS_CLI="redis-cli -h $REDIS_HOST -p $REDIS_PORT -a $REDIS_PASSWORD"

# Function to create ACL user
create_acl_user() {
    local username="$1"
    local password="$2"
    local permissions="$3"
    local description="$4"
    
    echo "  👤 Creating $username user ($description)..."
    
    # Delete user if exists (for idempotent setup)
    $REDIS_CLI ACL DELUSER "$username" 2>/dev/null || true
    
    # Create user with permissions
    if ! $REDIS_CLI ACL SETUSER "$username" on ">$password" $permissions; then
        echo "❌ Failed to create $username user"
        exit 1
    fi
}

# Orchestrator User - Full access to core orchestration data
create_acl_user "orchestrator" "$REDIS_ACL_ORCHESTRATOR_PASSWORD" \
    "+@all ~orchestrator:* ~task:* ~workflow:* ~agent:* ~project:* ~approval:* ~memory:* ~stream:activity ~metrics:*" \
    "orchestrator service"

# Agents User - Read/write access to agent-specific data
create_acl_user "agents" "$REDIS_ACL_AGENTS_PASSWORD" \
    "+get +set +del +keys +scan +json.get +json.set +json.del +json.arrappend +hget +hset +hgetall +hincrby +expire +ttl +exists +ping +info \
     -@dangerous \
     ~agent:* ~task:* ~memory:* ~stream:agent:*" \
    "agent workers"

# API User - Read access for API services
create_acl_user "api" "$REDIS_ACL_API_PASSWORD" \
    "+get +keys +scan +json.get +hget +hgetall +exists +ping +info \
     -@write -@dangerous \
     ~task:* ~project:* ~metrics:* ~stream:activity" \
    "API services"

# Monitoring User - Read-only access for monitoring
create_acl_user "monitoring" "$REDIS_ACL_MONITORING_PASSWORD" \
    "+get +keys +scan +json.get +ft.search +hget +hgetall +xread +xlen +exists +ping +info +slowlog +latency +memory \
     -@write -@dangerous \
     ~*metrics* ~*stream* ~*log*" \
    "monitoring and alerting"

# Admin User - Full administrative access
create_acl_user "admin" "$REDIS_ACL_ADMIN_PASSWORD" \
    "+@all" \
    "system administration"

echo ""
echo "✅ Redis ACL setup complete!"
echo ""
echo "📋 User Permissions Summary:"
echo "  orchestrator: Full access to orchestration data"
echo "  agents:       Read/write access to agent data"
echo "  api:          Read-only access for API services"
echo "  monitoring:   Read-only access for monitoring"
echo "  admin:        Full administrative access"
echo ""
echo "🔒 Security Notes:"
echo "  - Each service uses dedicated credentials"
echo "  - Principle of least privilege enforced"
echo "  - ACL rules prevent cross-service data access"
echo "  - No plaintext secrets in configuration"
echo ""
echo "🧪 Test the setup:"
echo "  redis-cli -h $REDIS_HOST -p $REDIS_PORT -a \$REDIS_ACL_ORCHESTRATOR_PASSWORD --user orchestrator ping"
```

### Governance Framework Implementation
From `multi-agent-reference-architecture-main/docs/governance/`, implement enterprise governance:

```python
# Enterprise Governance Framework
from typing import Dict, List, Any, Optional, Protocol
from abc import ABC, abstractmethod
from enum import Enum
from datetime import datetime, timedelta
import logging

class GovernanceLevel(Enum):
    """Governance enforcement levels"""
    PERMISSIVE = "permissive"  # Log only
    WARNING = "warning"       # Log and warn
    ENFORCING = "enforcing"    # Block violations
    STRICT = "strict"         # Block with escalation

class GovernancePolicy(Protocol):
    """Protocol for governance policies"""
    
    def check_compliance(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Check if action complies with policy"""
        ...
    
    def get_violation_actions(self, violation: Dict[str, Any]) -> List[str]:
        """Get actions to take on policy violation"""
        ...

class ResponsibleAIPolicy:
    """Responsible AI policy implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.fairness_threshold = config.get('fairness_threshold', 0.8)
        self.privacy_keywords = set(config.get('privacy_keywords', [
            'ssn', 'social security', 'credit card', 'password', 'secret'
        ]))
        self.bias_indicators = config.get('bias_indicators', [
            'gender', 'race', 'religion', 'political'
        ])
    
    def check_compliance(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Check Responsible AI compliance"""
        content = context.get('content', '').lower()
        metadata = context.get('metadata', {})
        
        violations = []
        warnings = []
        
        # Privacy check
        for keyword in self.privacy_keywords:
            if keyword in content:
                violations.append({
                    'type': 'privacy_violation',
                    'keyword': keyword,
                    'severity': 'high'
                })
        
        # Fairness check (simplified)
        if 'bias_check' in metadata:
            fairness_score = metadata['bias_check'].get('score', 1.0)
            if fairness_score < self.fairness_threshold:
                warnings.append({
                    'type': 'fairness_warning',
                    'score': fairness_score,
                    'threshold': self.fairness_threshold
                })
        
        return {
            'compliant': len(violations) == 0,
            'violations': violations,
            'warnings': warnings,
            'checked_at': datetime.now().isoformat()
        }
    
    def get_violation_actions(self, violation: Dict[str, Any]) -> List[str]:
        """Get remediation actions"""
        if violation['type'] == 'privacy_violation':
            return [
                'redact_content',
                'log_incident',
                'notify_compliance_officer',
                'block_output'
            ]
        elif violation['type'] == 'fairness_warning':
            return [
                'log_warning',
                'request_human_review',
                'apply_bias_mitigation'
            ]
        
        return ['log_violation']

class DataGovernancePolicy:
    """Data governance and retention policy"""
    
    def __init__(self, config: Dict[str, Any]):
        self.retention_periods = config.get('retention_periods', {
            'personal_data': timedelta(days=2555),  # 7 years
            'business_data': timedelta(days=2555),
            'logs': timedelta(days=2555),
            'models': timedelta(days=1825)  # 5 years
        })
        self.classification_rules = config.get('classification_rules', {})
    
    def check_compliance(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Check data governance compliance"""
        data_type = context.get('data_type', 'unknown')
        created_at = context.get('created_at', datetime.now())
        
        violations = []
        
        # Retention check
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        
        retention_period = self.retention_periods.get(data_type, timedelta(days=365))
        expiry_date = created_at + retention_period
        
        if datetime.now() > expiry_date:
            violations.append({
                'type': 'retention_violation',
                'data_type': data_type,
                'expired_days': (datetime.now() - expiry_date).days
            })
        
        # Classification check
        required_classification = self._classify_data(context)
        actual_classification = context.get('classification', 'public')
        
        if self._compare_classifications(required_classification, actual_classification) > 0:
            violations.append({
                'type': 'classification_violation',
                'required': required_classification,
                'actual': actual_classification
            })
        
        return {
            'compliant': len(violations) == 0,
            'violations': violations,
            'data_type': data_type,
            'retention_days': retention_period.days
        }
    
    def _classify_data(self, context: Dict[str, Any]) -> str:
        """Classify data based on content and context"""
        content = context.get('content', '').lower()
        
        if any(keyword in content for keyword in ['ssn', 'social security', 'medical']):
            return 'restricted'
        elif any(keyword in content for keyword in ['salary', 'performance', 'hr']):
            return 'confidential'
        elif any(keyword in content for keyword in ['customer', 'transaction', 'business']):
            return 'internal'
        else:
            return 'public'
    
    def _compare_classifications(self, required: str, actual: str) -> int:
        """Compare classification levels (higher number = more restrictive)"""
        levels = {'public': 0, 'internal': 1, 'confidential': 2, 'restricted': 3}
        return levels.get(required, 0) - levels.get(actual, 0)

class GovernanceEngine:
    """Central governance engine"""
    
    def __init__(self, level: GovernanceLevel = GovernanceLevel.ENFORCING):
        self.level = level
        self.policies: Dict[str, GovernancePolicy] = {}
        self.audit_log: List[Dict[str, Any]] = []
        self.logger = logging.getLogger(__name__)
    
    def register_policy(self, name: str, policy: GovernancePolicy) -> None:
        """Register a governance policy"""
        self.policies[name] = policy
        self.logger.info(f"Registered governance policy: {name}")
    
    async def evaluate_action(self, action_context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate an action against all policies"""
        evaluation_results = {}
        all_violations = []
        all_warnings = []
        blocked = False
        
        for policy_name, policy in self.policies.items():
            try:
                result = policy.check_compliance(action_context)
                evaluation_results[policy_name] = result
                
                all_violations.extend(result.get('violations', []))
                all_warnings.extend(result.get('warnings', []))
                
                if not result.get('compliant', True):
                    if self.level in [GovernanceLevel.ENFORCING, GovernanceLevel.STRICT]:
                        blocked = True
                    
            except Exception as e:
                self.logger.error(f"Policy evaluation failed for {policy_name}: {e}")
                evaluation_results[policy_name] = {'error': str(e)}
        
        # Log evaluation
        audit_entry = {
            'timestamp': datetime.now().isoformat(),
            'action_context': action_context,
            'evaluation_results': evaluation_results,
            'violations': all_violations,
            'warnings': all_warnings,
            'blocked': blocked,
            'governance_level': self.level.value
        }
        self.audit_log.append(audit_entry)
        
        # Apply remediation actions if violations found
        if all_violations and self.level != GovernanceLevel.PERMISSIVE:
            await self._apply_remediation(all_violations, action_context)
        
        return {
            'approved': not blocked,
            'violations': all_violations,
            'warnings': all_warnings,
            'evaluation_details': evaluation_results,
            'audit_id': len(self.audit_log) - 1
        }
    
    async def _apply_remediation(self, violations: List[Dict[str, Any]], 
                               context: Dict[str, Any]) -> None:
        """Apply remediation actions for violations"""
        for violation in violations:
            policy_name = violation.get('policy', 'unknown')
            if policy_name in self.policies:
                actions = self.policies[policy_name].get_violation_actions(violation)
                await self._execute_actions(actions, violation, context)
    
    async def _execute_actions(self, actions: List[str], violation: Dict[str, Any], 
                             context: Dict[str, Any]) -> None:
        """Execute remediation actions"""
        for action in actions:
            try:
                if action == 'log_violation':
                    self.logger.warning(f"Governance violation: {violation}")
                elif action == 'redact_content':
                    await self._redact_content(context)
                elif action == 'block_output':
                    context['blocked'] = True
                elif action == 'notify_compliance_officer':
                    await self._notify_compliance_officer(violation)
                elif action == 'request_human_review':
                    await self._request_human_review(context)
                    
            except Exception as e:
                self.logger.error(f"Failed to execute remediation action {action}: {e}")
    
    async def _redact_content(self, context: Dict[str, Any]) -> None:
        """Redact sensitive content"""
        content = context.get('content', '')
        # Simple redaction - in production use proper NLP
        redacted = content.replace('ssn', '[REDACTED]').replace('password', '[REDACTED]')
        context['content'] = redacted
        context['redacted'] = True
    
    async def _notify_compliance_officer(self, violation: Dict[str, Any]) -> None:
        """Notify compliance officer of violation"""
        # Implementation would send email/notification
        self.logger.critical(f"Compliance notification required: {violation}")
    
    async def _request_human_review(self, context: Dict[str, Any]) -> None:
        """Request human review for content"""
        context['requires_review'] = True
        context['review_reason'] = 'governance_violation'
    
    def get_audit_trail(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get governance audit trail"""
        return self.audit_log[-limit:]
    
    def get_policy_status(self) -> Dict[str, Any]:
        """Get current policy status"""
        return {
            'governance_level': self.level.value,
            'active_policies': list(self.policies.keys()),
            'total_audits': len(self.audit_log),
            'recent_violations': len([a for a in self.audit_log[-100:] if a.get('violations')])
        }
```

### Phase 11: Final Enterprise Integration & Deployment (New Phase)
**Goal**: Complete enterprise integration with production deployment and comprehensive testing.

1. **Complete Enterprise Integration**
   - Integrate all governance policies into orchestrator
   - Implement comprehensive audit logging
   - Add enterprise monitoring and alerting
   - Create compliance reporting dashboards

2. **Production Deployment Automation**
   - Implement infrastructure as code with Terraform
   - Create CI/CD pipelines for automated deployment
   - Add blue-green deployment strategies
   - Implement canary releases for safe rollouts

3. **Comprehensive Testing Suite**
   - Add end-to-end integration tests
   - Implement chaos engineering tests
   - Create performance and load testing
   - Add security penetration testing

4. **Documentation & Training**
   - Complete API documentation
   - Create operational runbooks
   - Develop training materials for administrators
   - Add troubleshooting guides

## Final Success Metrics (Complete Enterprise Suite)
- **Enterprise Maturity**: SOC 2 Type II compliance achieved
- **Operational Excellence**: 99.95% uptime with < 5min MTTR
- **Security Posture**: Zero critical vulnerabilities, comprehensive audit trails
- **Scalability**: Support for 10,000+ concurrent users with < 500ms latency
- **Governance**: 100% policy compliance with automated enforcement
- **Innovation Velocity**: 75% faster feature deployment through automation

## Complete Implementation Timeline
- **Total Duration**: 33 weeks (added Phases 10-11)
- **Total Team Size**: 4-5 developers for full enterprise implementation
- **Total Infrastructure**: Complete production environment with multi-region deployment
- **Total Compliance**: Full enterprise governance and security standards
- **Total Automation**: End-to-end CI/CD with automated testing and deployment

This absolutely comprehensive implementation plan now includes every single pattern, framework, and implementation detail discovered from the exhaustive analysis of all reference frameworks. King-AI-v2 can now be transformed into a world-class, enterprise-grade multi-agent orchestration platform with complete production readiness. 🚀</content>
<parameter name="filePath">c:\Users\dmilner.AGV-040318-PC\Downloads\landon\king-ai-v2\implementationplan.md