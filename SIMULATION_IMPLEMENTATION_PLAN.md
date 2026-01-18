# King AI v2 Simulation Layer Implementation Plan

## Overview

Add a simulation/self-play layer enabling strategy tournaments with real-world data but simulated banking/legal/money operations. Consolidates patterns from 7 reference frameworks into King AI v2's existing architecture.

**Key Design Decisions:**
- Storage: SQLite WAL for simulation isolation (separate from PostgreSQL production data)
- Parallelism: asyncio.gather with semaphores
- Orchestration API: Unified `invoke(task, runtime)` interface
- Checkpoints: 6 phases per action for fine replay control
- Scoring: 0.6 × ROI + 0.4 × Sharpe-like ratio
- Strategy versioning: Semantic versioning (MAJOR.MINOR.PATCH)

---

## File Structure

```
src/simulation/
├── __init__.py
├── checkpoint_store.py      # SQLite WAL database
├── models.py                # Pydantic models
├── context.py               # SimulationContext
├── orchestration/
│   ├── __init__.py
│   ├── base.py              # BaseOrchestration
│   ├── sequential.py        # Pipeline execution
│   ├── concurrent.py        # Parallel execution
│   ├── group_chat.py        # Manager-coordinated
│   ├── handoff.py           # Dynamic delegation
│   └── magentic.py          # MagenticOne pattern
├── execution/
│   ├── __init__.py
│   ├── phase_executor.py    # 6-phase execution
│   ├── provenance.py        # Hash-chain
│   └── mock_adapters.py     # Simulated banking/legal
├── tournament/
│   ├── __init__.py
│   ├── tournament_runner.py # Round-robin orchestration
│   ├── scoring.py           # Composite scoring
│   ├── strategy_registry.py # Strategy management
│   └── standings.py         # Leaderboard
└── streaming/
    ├── __init__.py
    ├── broadcast_manager.py # WebSocket extensions
    ├── event_log.py         # Three-phase logging
    └── websocket_routes.py  # FastAPI WS endpoints

src/api/routes/simulation/
├── __init__.py
├── tournaments.py
├── simulations.py
└── strategies.py
```

---

## Part 1: Core Infrastructure

### 1.1 checkpoint_store.py

SQLite database with WAL mode for simulation state persistence.

**Tables:**
```sql
-- Tournament definition
CREATE TABLE tournaments (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    config JSON NOT NULL,
    status TEXT DEFAULT 'pending',  -- pending/running/completed/cancelled
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP
);

-- Strategy definitions with versioning
CREATE TABLE strategies (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    version TEXT NOT NULL,  -- semver: 1.0.0
    parameters JSON NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(name, version)
);

-- Individual simulation runs
CREATE TABLE simulations (
    id TEXT PRIMARY KEY,
    tournament_id TEXT REFERENCES tournaments(id),
    strategy_id TEXT REFERENCES strategies(id),
    config JSON NOT NULL,
    status TEXT DEFAULT 'pending',
    initial_capital REAL NOT NULL,
    final_capital REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP
);

-- Phase checkpoints (6 per action)
CREATE TABLE checkpoints (
    id TEXT PRIMARY KEY,
    simulation_id TEXT REFERENCES simulations(id),
    phase TEXT NOT NULL,  -- analyze/plan/validate/execute/evaluate/consolidate
    phase_index INTEGER NOT NULL,
    state_hash TEXT NOT NULL,
    parent_hash TEXT,
    data JSON NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_checkpoints_sim ON checkpoints(simulation_id, phase_index);

-- Tournament standings
CREATE TABLE standings (
    id TEXT PRIMARY KEY,
    tournament_id TEXT REFERENCES tournaments(id),
    strategy_id TEXT REFERENCES strategies(id),
    wins INTEGER DEFAULT 0,
    losses INTEGER DEFAULT 0,
    total_roi REAL DEFAULT 0,
    sharpe_ratio REAL DEFAULT 0,
    composite_score REAL DEFAULT 0,
    rank INTEGER,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(tournament_id, strategy_id)
);

-- Event log for streaming
CREATE TABLE events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    correlation_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    source TEXT NOT NULL,
    data JSON NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_events_correlation ON events(correlation_id);
```

**Class: CheckpointStore**
```python
class CheckpointStore:
    def __init__(self, db_path: str = "simulation.db"):
        # Initialize with WAL mode
        
    async def init_db(self): ...
    
    # Tournament CRUD
    async def create_tournament(self, config: TournamentConfig) -> str: ...
    async def get_tournament(self, id: str) -> Tournament: ...
    async def update_tournament_status(self, id: str, status: str): ...
    
    # Strategy CRUD
    async def register_strategy(self, strategy: StrategyVersion) -> str: ...
    async def get_strategy(self, id: str) -> StrategyVersion: ...
    async def list_strategies(self, name: str = None) -> List[StrategyVersion]: ...
    
    # Simulation CRUD
    async def create_simulation(self, config: SimulationConfig) -> str: ...
    async def get_simulation(self, id: str) -> Simulation: ...
    
    # Checkpoint operations
    async def save_checkpoint(self, checkpoint: Checkpoint) -> str: ...
    async def get_checkpoints(self, simulation_id: str) -> List[Checkpoint]: ...
    async def get_checkpoint_chain(self, simulation_id: str) -> List[str]: ...
    
    # Standings
    async def update_standings(self, tournament_id: str, strategy_id: str, result: SimulationResult): ...
    async def get_standings(self, tournament_id: str) -> List[Standing]: ...
    
    # Events
    async def log_event(self, event: SimulationEvent): ...
    async def get_events(self, correlation_id: str) -> List[SimulationEvent]: ...
```

### 1.2 models.py

Pydantic models for type safety.

```python
from pydantic import BaseModel, Field
from enum import Enum
from typing import Optional, List, Dict, Any
from datetime import datetime

class SimulationPhase(str, Enum):
    ANALYZE = "analyze"
    PLAN = "plan"
    VALIDATE = "validate"
    EXECUTE = "execute"
    EVALUATE = "evaluate"
    CONSOLIDATE = "consolidate"

class TournamentStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"

class TournamentFormat(str, Enum):
    ROUND_ROBIN = "round_robin"
    ELIMINATION = "elimination"
    SWISS = "swiss"

class SimulationConfig(BaseModel):
    duration_days: int = 30
    initial_capital: float = 10000.0
    risk_profile: str = "moderate"
    strategy_id: str
    market_data_source: str = "historical"
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None

class TournamentConfig(BaseModel):
    name: str
    format: TournamentFormat = TournamentFormat.ROUND_ROBIN
    strategy_ids: List[str]
    simulation_config: SimulationConfig
    max_concurrent: int = 4
    scoring_weights: Dict[str, float] = {"roi": 0.6, "sharpe": 0.4}

class StrategyVersion(BaseModel):
    id: Optional[str] = None
    name: str
    version: str  # semver
    parameters: Dict[str, Any]
    description: Optional[str] = None
    created_at: Optional[datetime] = None

class Checkpoint(BaseModel):
    id: Optional[str] = None
    simulation_id: str
    phase: SimulationPhase
    phase_index: int
    state_hash: str
    parent_hash: Optional[str] = None
    data: Dict[str, Any]
    created_at: Optional[datetime] = None

class SimulationResult(BaseModel):
    simulation_id: str
    strategy_id: str
    initial_capital: float
    final_capital: float
    roi: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    phase_returns: List[float]

class Standing(BaseModel):
    tournament_id: str
    strategy_id: str
    wins: int = 0
    losses: int = 0
    total_roi: float = 0.0
    sharpe_ratio: float = 0.0
    composite_score: float = 0.0
    rank: Optional[int] = None

class SimulationEvent(BaseModel):
    correlation_id: str
    event_type: str  # phase_start, phase_complete, error, etc.
    source: str
    data: Dict[str, Any]
    created_at: Optional[datetime] = None

class OrchestrationResult(BaseModel):
    success: bool
    output: Any
    messages: List[Dict[str, Any]] = []
    metadata: Dict[str, Any] = {}
```

### 1.3 context.py

Simulation context extending existing ContextManager patterns.

```python
class SimulationContext:
    """Context for a single simulation run."""
    
    def __init__(
        self,
        simulation_id: str,
        config: SimulationConfig,
        checkpoint_store: CheckpointStore,
        mock_services: Dict[str, Any]
    ):
        self.simulation_id = simulation_id
        self.config = config
        self.store = checkpoint_store
        self.services = mock_services
        
        self.current_phase = SimulationPhase.ANALYZE
        self.phase_index = 0
        self.capital = config.initial_capital
        self.positions: List[Position] = []
        self.trades: List[Trade] = []
        self.phase_returns: List[float] = []
        self.last_checkpoint_hash: Optional[str] = None
    
    async def advance_phase(self):
        """Move to next phase, save checkpoint."""
        
    async def save_checkpoint(self, data: Dict[str, Any]) -> Checkpoint:
        """Create checkpoint with hash chain."""
        
    def get_service(self, name: str) -> Any:
        """Get mock service by name (stripe, plaid, etc.)."""
        
    def record_trade(self, trade: Trade):
        """Record a trade and update capital."""
        
    def compute_metrics(self) -> SimulationResult:
        """Compute final metrics for scoring."""

class SimulationRuntime:
    """Runtime environment for orchestrations."""
    
    def __init__(self, checkpoint_store: CheckpointStore):
        self.store = checkpoint_store
        self.active_simulations: Dict[str, SimulationContext] = {}
        self._running = False
    
    async def start(self): ...
    async def stop(self): ...
    async def stop_when_idle(self): ...
    
    def register_simulation(self, ctx: SimulationContext): ...
    def get_simulation(self, id: str) -> SimulationContext: ...
```

---

## Part 2: Orchestration Patterns

### 2.1 base.py

Abstract base for all orchestration patterns.

```python
from abc import ABC, abstractmethod
from typing import List, Callable, Optional, Any

class BaseOrchestration(ABC):
    """Base class for orchestration patterns."""
    
    def __init__(
        self,
        members: List[Any],  # Agents or other orchestrations
        name: Optional[str] = None,
        pre_invoke: Optional[Callable] = None,
        post_invoke: Optional[Callable] = None,
        result_transform: Optional[Callable] = None
    ):
        self.members = members
        self.name = name or self.__class__.__name__
        self.pre_invoke = pre_invoke
        self.post_invoke = post_invoke
        self.result_transform = result_transform
    
    @abstractmethod
    async def invoke(
        self,
        task: str,
        runtime: SimulationRuntime,
        context: Optional[SimulationContext] = None
    ) -> OrchestrationResult:
        """Execute the orchestration pattern."""
        pass
    
    async def _call_agent(self, agent: Any, task: dict) -> dict:
        """Call an agent's execute method."""
        if hasattr(agent, 'execute'):
            return await agent.execute(task)
        return {"success": False, "error": "Agent has no execute method"}
```

### 2.2 sequential.py

Pipeline execution - each agent's output feeds the next.

```python
class SequentialOrchestration(BaseOrchestration):
    """Execute agents in sequence, passing output forward."""
    
    async def invoke(self, task: str, runtime: SimulationRuntime, context=None) -> OrchestrationResult:
        messages = []
        current_input = task
        
        for i, agent in enumerate(self.members):
            result = await self._call_agent(agent, {"input": current_input, "context": context})
            messages.append({"agent": agent.name, "output": result})
            
            if not result.get("success", False):
                return OrchestrationResult(success=False, output=None, messages=messages)
            
            current_input = result.get("output", current_input)
        
        return OrchestrationResult(success=True, output=current_input, messages=messages)
```

### 2.3 concurrent.py

Parallel execution with result aggregation.

```python
class ConcurrentOrchestration(BaseOrchestration):
    """Execute agents in parallel, aggregate results."""
    
    def __init__(self, members, max_concurrent: int = 10, **kwargs):
        super().__init__(members, **kwargs)
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    async def invoke(self, task: str, runtime: SimulationRuntime, context=None) -> OrchestrationResult:
        async def run_with_semaphore(agent):
            async with self.semaphore:
                return await self._call_agent(agent, {"input": task, "context": context})
        
        results = await asyncio.gather(
            *[run_with_semaphore(agent) for agent in self.members],
            return_exceptions=True
        )
        
        messages = []
        outputs = []
        for agent, result in zip(self.members, results):
            if isinstance(result, Exception):
                messages.append({"agent": agent.name, "error": str(result)})
            else:
                messages.append({"agent": agent.name, "output": result})
                if result.get("success"):
                    outputs.append(result.get("output"))
        
        return OrchestrationResult(success=len(outputs) > 0, output=outputs, messages=messages)
```

### 2.4 group_chat.py

Manager-coordinated conversation with speaker selection.

```python
class GroupChatManager:
    """Selects next speaker in group chat."""
    
    def __init__(self, llm_client, members: List[Any], max_turns: int = 10):
        self.llm = llm_client
        self.members = members
        self.max_turns = max_turns
        self.turn_count = 0
    
    async def select_next_speaker(self, history: List[dict]) -> Any:
        """Use LLM to select next speaker, fallback to round-robin."""
        try:
            prompt = self._build_selection_prompt(history)
            response = await self.llm.generate(prompt)
            agent_name = self._parse_agent_name(response)
            return next((a for a in self.members if a.name == agent_name), None)
        except:
            # Fallback to round-robin
            return self.members[self.turn_count % len(self.members)]
    
    def should_terminate(self, history: List[dict]) -> bool:
        """Check if conversation should end."""
        return self.turn_count >= self.max_turns

class GroupChatOrchestration(BaseOrchestration):
    """Manager-coordinated multi-agent conversation."""
    
    def __init__(self, members, manager: GroupChatManager, human_input_hook=None, **kwargs):
        super().__init__(members, **kwargs)
        self.manager = manager
        self.human_input_hook = human_input_hook
    
    async def invoke(self, task: str, runtime: SimulationRuntime, context=None) -> OrchestrationResult:
        history = [{"role": "user", "content": task}]
        
        while not self.manager.should_terminate(history):
            speaker = await self.manager.select_next_speaker(history)
            if speaker is None:
                break
            
            result = await self._call_agent(speaker, {"input": task, "history": history, "context": context})
            history.append({"role": speaker.name, "content": result.get("output", "")})
            self.manager.turn_count += 1
            
            # Optional human input
            if self.human_input_hook and await self._needs_human_input(result):
                human_response = await self.human_input_hook(history)
                history.append({"role": "human", "content": human_response})
        
        return OrchestrationResult(success=True, output=history[-1]["content"], messages=history)
```

### 2.5 handoff.py

Dynamic agent-to-agent control transfer.

```python
class HandoffOrchestration(BaseOrchestration):
    """Agents can transfer control to other agents."""
    
    def __init__(self, members, entry_agent: Any, max_handoffs: int = 10, **kwargs):
        super().__init__(members, **kwargs)
        self.entry_agent = entry_agent
        self.max_handoffs = max_handoffs
        self.agent_map = {a.name: a for a in members}
    
    async def invoke(self, task: str, runtime: SimulationRuntime, context=None) -> OrchestrationResult:
        current_agent = self.entry_agent
        messages = []
        handoff_count = 0
        
        while current_agent and handoff_count < self.max_handoffs:
            result = await self._call_agent(current_agent, {"input": task, "context": context})
            messages.append({"agent": current_agent.name, "output": result})
            
            # Check for handoff
            handoff_to = result.get("metadata", {}).get("handoff_to")
            if handoff_to and handoff_to in self.agent_map:
                current_agent = self.agent_map[handoff_to]
                handoff_count += 1
            else:
                break
        
        final_output = messages[-1]["output"] if messages else None
        return OrchestrationResult(success=True, output=final_output, messages=messages)
```

### 2.6 magentic.py

MagenticOne pattern for complex open-ended tasks.

```python
class TaskLedger:
    """Tracks task progress and subtask assignments."""
    
    def __init__(self):
        self.facts: List[str] = []
        self.subtasks: List[dict] = []
        self.progress: Dict[str, str] = {}
    
    def add_fact(self, fact: str): ...
    def add_subtask(self, subtask: dict): ...
    def update_progress(self, subtask_id: str, status: str): ...
    def get_pending_subtasks(self) -> List[dict]: ...

class MagenticManager:
    """Central coordinator for Magentic pattern."""
    
    def __init__(self, llm_client, members: List[Any], max_rounds: int = 10):
        self.llm = llm_client
        self.members = members
        self.max_rounds = max_rounds
        self.ledger = TaskLedger()
    
    async def plan(self, task: str) -> List[dict]:
        """Generate initial subtask plan."""
        
    async def select_agent(self, subtask: dict) -> Any:
        """Select best agent for subtask."""
        
    async def should_continue(self) -> bool:
        """Check if more work needed."""

class MagenticOrchestration(BaseOrchestration):
    """MagenticOne pattern - manager coordinates specialized agents."""
    
    def __init__(self, members, manager: MagenticManager, **kwargs):
        super().__init__(members, **kwargs)
        self.manager = manager
    
    async def invoke(self, task: str, runtime: SimulationRuntime, context=None) -> OrchestrationResult:
        messages = []
        
        # Initial planning
        subtasks = await self.manager.plan(task)
        for st in subtasks:
            self.manager.ledger.add_subtask(st)
        
        round_count = 0
        while await self.manager.should_continue() and round_count < self.manager.max_rounds:
            pending = self.manager.ledger.get_pending_subtasks()
            if not pending:
                break
            
            for subtask in pending:
                agent = await self.manager.select_agent(subtask)
                result = await self._call_agent(agent, {"subtask": subtask, "context": context})
                messages.append({"agent": agent.name, "subtask": subtask["id"], "output": result})
                
                self.manager.ledger.update_progress(subtask["id"], "completed" if result.get("success") else "failed")
                if result.get("facts"):
                    for fact in result["facts"]:
                        self.manager.ledger.add_fact(fact)
            
            round_count += 1
        
        return OrchestrationResult(
            success=True,
            output={"facts": self.manager.ledger.facts, "subtasks": self.manager.ledger.subtasks},
            messages=messages
        )
```

---

## Part 3: Phase Executor

### 3.1 phase_executor.py

6-phase chained execution with checkpointing.

```python
class PhaseExecutor:
    """Executes simulation phases with checkpointing."""
    
    PHASES = [
        SimulationPhase.ANALYZE,
        SimulationPhase.PLAN,
        SimulationPhase.VALIDATE,
        SimulationPhase.EXECUTE,
        SimulationPhase.EVALUATE,
        SimulationPhase.CONSOLIDATE
    ]
    
    def __init__(self, context: SimulationContext, orchestration: BaseOrchestration):
        self.context = context
        self.orchestration = orchestration
        self.provenance = ProvenanceChain()
    
    async def run_full_cycle(self, task: str, runtime: SimulationRuntime) -> SimulationResult:
        """Run all 6 phases for one action cycle."""
        for phase in self.PHASES:
            self.context.current_phase = phase
            
            phase_result = await self._execute_phase(phase, task, runtime)
            
            checkpoint = await self.context.save_checkpoint(phase_result)
            self.provenance.add_hash(checkpoint.state_hash)
            
            if not phase_result.get("success", False):
                break
        
        return self.context.compute_metrics()
    
    async def _execute_phase(self, phase: SimulationPhase, task: str, runtime: SimulationRuntime) -> dict:
        """Execute a single phase."""
        handler = getattr(self, f"_phase_{phase.value}", None)
        if handler:
            return await handler(task, runtime)
        return {"success": True, "phase": phase.value}
    
    async def _phase_analyze(self, task: str, runtime: SimulationRuntime) -> dict:
        """Gather market data, assess conditions."""
        # Use concurrent orchestration to gather data from multiple sources
        
    async def _phase_plan(self, task: str, runtime: SimulationRuntime) -> dict:
        """Generate action plan."""
        # Use sequential orchestration: research -> strategy -> risk_check
        
    async def _phase_validate(self, task: str, runtime: SimulationRuntime) -> dict:
        """Risk validation, approval checks."""
        # Check risk limits, validate plan feasibility
        
    async def _phase_execute(self, task: str, runtime: SimulationRuntime) -> dict:
        """Run simulated actions via mock adapters."""
        # Execute trades via mock exchange
        
    async def _phase_evaluate(self, task: str, runtime: SimulationRuntime) -> dict:
        """Score results, compute metrics."""
        # Calculate returns, update statistics
        
    async def _phase_consolidate(self, task: str, runtime: SimulationRuntime) -> dict:
        """Update strategy state, finalize."""
        # Record phase return, update positions
```

### 3.2 provenance.py

Hash-chain for replay validation.

```python
import hashlib
import json

class ProvenanceChain:
    """Maintains hash chain for checkpoint provenance."""
    
    def __init__(self):
        self.hashes: List[str] = []
    
    @staticmethod
    def compute_hash(data: dict, parent_hash: Optional[str] = None) -> str:
        """Compute SHA256 hash of phase data with parent chain."""
        content = json.dumps(data, sort_keys=True, default=str)
        if parent_hash:
            content = parent_hash + content
        return hashlib.sha256(content.encode()).hexdigest()
    
    def add_hash(self, hash_value: str):
        """Add hash to chain."""
        self.hashes.append(hash_value)
    
    def get_chain(self) -> List[str]:
        """Get full hash chain."""
        return self.hashes.copy()
    
    @staticmethod
    def validate_replay(original: List[str], replay: List[str]) -> bool:
        """Strict comparison of hash chains."""
        if len(original) != len(replay):
            return False
        return all(o == r for o, r in zip(original, replay))

class ReplayValidator:
    """Validates simulation replays against original runs."""
    
    def __init__(self, checkpoint_store: CheckpointStore):
        self.store = checkpoint_store
    
    async def validate(self, simulation_id: str, replay_hashes: List[str]) -> dict:
        """Validate replay matches original."""
        original = await self.store.get_checkpoint_chain(simulation_id)
        is_valid = ProvenanceChain.validate_replay(original, replay_hashes)
        return {
            "valid": is_valid,
            "original_count": len(original),
            "replay_count": len(replay_hashes),
            "first_mismatch": self._find_first_mismatch(original, replay_hashes)
        }
    
    def _find_first_mismatch(self, original: List[str], replay: List[str]) -> Optional[int]:
        for i, (o, r) in enumerate(zip(original, replay)):
            if o != r:
                return i
        return None
```

### 3.3 mock_adapters.py

Simulated versions of real integrations.

```python
class MockStripeClient:
    """Simulated Stripe for testing."""
    
    def __init__(self, failure_rate: float = 0.05):
        self.failure_rate = failure_rate
        self.transactions: List[dict] = []
    
    async def create_payment_intent(self, amount: float, currency: str = "usd") -> dict:
        if random.random() < self.failure_rate:
            return {"success": False, "error": "Payment failed"}
        
        txn = {
            "id": f"pi_{uuid.uuid4().hex[:24]}",
            "amount": amount,
            "currency": currency,
            "status": "succeeded",
            "created": datetime.utcnow().isoformat()
        }
        self.transactions.append(txn)
        return {"success": True, "payment_intent": txn}
    
    async def get_balance(self) -> dict:
        total = sum(t["amount"] for t in self.transactions if t["status"] == "succeeded")
        return {"available": total, "pending": 0}

class MockPlaidClient:
    """Simulated Plaid for banking."""
    
    def __init__(self, initial_balance: float = 10000.0):
        self.balance = initial_balance
        self.transactions: List[dict] = []
    
    async def get_accounts(self) -> List[dict]:
        return [{"id": "mock_account_1", "name": "Checking", "balance": self.balance}]
    
    async def get_transactions(self, start_date: str, end_date: str) -> List[dict]:
        return self.transactions
    
    async def transfer(self, amount: float, direction: str = "outflow") -> dict:
        if direction == "outflow" and amount > self.balance:
            return {"success": False, "error": "Insufficient funds"}
        
        self.balance += amount if direction == "inflow" else -amount
        txn = {"id": f"txn_{uuid.uuid4().hex[:16]}", "amount": amount, "direction": direction}
        self.transactions.append(txn)
        return {"success": True, "transaction": txn}

class MockLegalValidator:
    """Simulated legal/compliance checks."""
    
    def __init__(self, approval_rate: float = 0.95):
        self.approval_rate = approval_rate
    
    async def validate_action(self, action: dict) -> dict:
        approved = random.random() < self.approval_rate
        return {
            "approved": approved,
            "reason": None if approved else "Compliance check failed",
            "checked_at": datetime.utcnow().isoformat()
        }

class MockExchange:
    """Simulated trading exchange."""
    
    def __init__(self):
        self.orders: List[dict] = []
        self.positions: Dict[str, float] = {}
        self.prices: Dict[str, float] = {}  # Loaded from market data
    
    async def place_order(self, symbol: str, side: str, quantity: float, order_type: str = "market") -> dict:
        price = self.prices.get(symbol, 100.0)
        order = {
            "id": f"ord_{uuid.uuid4().hex[:16]}",
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "price": price,
            "status": "filled"
        }
        self.orders.append(order)
        
        # Update position
        delta = quantity if side == "buy" else -quantity
        self.positions[symbol] = self.positions.get(symbol, 0) + delta
        
        return {"success": True, "order": order}
    
    def load_prices(self, prices: Dict[str, float]):
        self.prices = prices

class MockServiceRegistry:
    """Registry of mock services for simulation."""
    
    def __init__(self, config: SimulationConfig):
        self.services = {
            "stripe": MockStripeClient(),
            "plaid": MockPlaidClient(initial_balance=config.initial_capital),
            "legal": MockLegalValidator(),
            "exchange": MockExchange()
        }
    
    def get(self, name: str) -> Any:
        return self.services.get(name)
```

---

## Part 4: Tournament System

### 4.1 tournament_runner.py

Orchestrates full tournament with round-robin pairings.

```python
class TournamentRunner:
    """Runs strategy tournaments."""
    
    def __init__(
        self,
        checkpoint_store: CheckpointStore,
        runtime: SimulationRuntime,
        broadcast_manager: Optional[BroadcastManager] = None
    ):
        self.store = checkpoint_store
        self.runtime = runtime
        self.broadcast = broadcast_manager
        self.scorer = CompositeScorer()
    
    async def run_tournament(self, config: TournamentConfig) -> str:
        """Run complete tournament, return tournament_id."""
        tournament_id = await self.store.create_tournament(config)
        await self.store.update_tournament_status(tournament_id, "running")
        
        try:
            if config.format == TournamentFormat.ROUND_ROBIN:
                await self._run_round_robin(tournament_id, config)
            
            await self.store.update_tournament_status(tournament_id, "completed")
        except Exception as e:
            await self.store.update_tournament_status(tournament_id, "cancelled")
            raise
        
        return tournament_id
    
    async def _run_round_robin(self, tournament_id: str, config: TournamentConfig):
        """Run round-robin format."""
        strategies = [await self.store.get_strategy(sid) for sid in config.strategy_ids]
        pairings = list(itertools.combinations(strategies, 2))
        
        semaphore = asyncio.Semaphore(config.max_concurrent)
        
        async def run_match(s1: StrategyVersion, s2: StrategyVersion):
            async with semaphore:
                result1 = await self._run_simulation(tournament_id, s1, config.simulation_config)
                result2 = await self._run_simulation(tournament_id, s2, config.simulation_config)
                await self._update_standings(tournament_id, s1, s2, result1, result2)
        
        await asyncio.gather(*[run_match(s1, s2) for s1, s2 in pairings])
    
    async def _run_simulation(self, tournament_id: str, strategy: StrategyVersion, sim_config: SimulationConfig) -> SimulationResult:
        """Run single simulation for a strategy."""
        config = sim_config.copy(update={"strategy_id": strategy.id})
        simulation_id = await self.store.create_simulation(config)
        
        context = SimulationContext(
            simulation_id=simulation_id,
            config=config,
            checkpoint_store=self.store,
            mock_services=MockServiceRegistry(config).services
        )
        
        # Create strategy-specific orchestration
        orchestration = self._build_orchestration(strategy)
        executor = PhaseExecutor(context, orchestration)
        
        result = await executor.run_full_cycle(f"Execute {strategy.name} strategy", self.runtime)
        
        if self.broadcast:
            await self.broadcast.emit(tournament_id, "simulation_complete", result.dict())
        
        return result
    
    async def _update_standings(self, tournament_id: str, s1, s2, r1: SimulationResult, r2: SimulationResult):
        """Update standings based on simulation results."""
        score1 = self.scorer.compute(r1)
        score2 = self.scorer.compute(r2)
        
        if score1 > score2:
            await self._record_win(tournament_id, s1.id, r1)
            await self._record_loss(tournament_id, s2.id, r2)
        else:
            await self._record_win(tournament_id, s2.id, r2)
            await self._record_loss(tournament_id, s1.id, r1)
    
    def _build_orchestration(self, strategy: StrategyVersion) -> BaseOrchestration:
        """Build orchestration based on strategy type."""
        # Default to sequential, can be customized per strategy
        return SequentialOrchestration(members=[...])
```

### 4.2 scoring.py

Composite scoring with ROI and Sharpe.

```python
import statistics

class CompositeScorer:
    """Computes composite score from simulation results."""
    
    def __init__(self, roi_weight: float = 0.6, sharpe_weight: float = 0.4):
        self.roi_weight = roi_weight
        self.sharpe_weight = sharpe_weight
    
    def compute(self, result: SimulationResult) -> float:
        """Compute composite score."""
        roi = result.roi
        sharpe = self.compute_sharpe(result.phase_returns)
        
        # Normalize to 0-100 scale
        normalized_roi = min(max(roi * 100, -100), 100)  # Cap at ±100%
        normalized_sharpe = min(max(sharpe * 33.33, -100), 100)  # Sharpe of 3 = 100
        
        return (self.roi_weight * normalized_roi) + (self.sharpe_weight * normalized_sharpe)
    
    def compute_sharpe(self, returns: List[float], risk_free_rate: float = 0.0) -> float:
        """Compute Sharpe-like ratio from phase returns."""
        if len(returns) < 2:
            return 0.0
        
        excess_returns = [r - risk_free_rate for r in returns]
        mean_return = statistics.mean(excess_returns)
        std_return = statistics.stdev(excess_returns)
        
        if std_return == 0:
            return 0.0
        
        return mean_return / std_return
    
    def compute_max_drawdown(self, capital_history: List[float]) -> float:
        """Compute maximum drawdown from capital history."""
        if not capital_history:
            return 0.0
        
        peak = capital_history[0]
        max_dd = 0.0
        
        for value in capital_history:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_dd = max(max_dd, drawdown)
        
        return max_dd
```

### 4.3 strategy_registry.py

Strategy lifecycle management.

```python
import semver

class StrategyRegistry:
    """Manages strategy versions and evolution."""
    
    def __init__(self, checkpoint_store: CheckpointStore):
        self.store = checkpoint_store
    
    async def register(self, name: str, version: str, parameters: dict, description: str = None) -> StrategyVersion:
        """Register a new strategy version."""
        # Validate semver
        semver.VersionInfo.parse(version)
        
        strategy = StrategyVersion(
            name=name,
            version=version,
            parameters=parameters,
            description=description
        )
        strategy.id = await self.store.register_strategy(strategy)
        return strategy
    
    async def get_latest(self, name: str) -> Optional[StrategyVersion]:
        """Get latest version of a strategy."""
        versions = await self.store.list_strategies(name=name)
        if not versions:
            return None
        
        sorted_versions = sorted(versions, key=lambda s: semver.VersionInfo.parse(s.version), reverse=True)
        return sorted_versions[0]
    
    async def evolve(self, strategy_id: str, mutation: dict) -> StrategyVersion:
        """Create evolved version of strategy."""
        base = await self.store.get_strategy(strategy_id)
        
        # Increment version
        current = semver.VersionInfo.parse(base.version)
        if mutation.get("breaking"):
            new_version = str(current.bump_major())
        elif mutation.get("feature"):
            new_version = str(current.bump_minor())
        else:
            new_version = str(current.bump_patch())
        
        # Apply mutation to parameters
        new_params = {**base.parameters, **mutation.get("parameters", {})}
        
        return await self.register(base.name, new_version, new_params, f"Evolved from {base.version}")
    
    async def crossover(self, strategy_id_1: str, strategy_id_2: str) -> StrategyVersion:
        """Create hybrid strategy from two parents."""
        s1 = await self.store.get_strategy(strategy_id_1)
        s2 = await self.store.get_strategy(strategy_id_2)
        
        # Simple parameter averaging for numeric values
        new_params = {}
        all_keys = set(s1.parameters.keys()) | set(s2.parameters.keys())
        
        for key in all_keys:
            v1 = s1.parameters.get(key)
            v2 = s2.parameters.get(key)
            
            if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                new_params[key] = (v1 + v2) / 2
            else:
                new_params[key] = random.choice([v1, v2])
        
        hybrid_name = f"{s1.name}_x_{s2.name}"
        return await self.register(hybrid_name, "1.0.0", new_params, f"Crossover of {s1.name} and {s2.name}")
```

### 4.4 standings.py

Leaderboard with Elo-like ratings.

```python
class StandingsManager:
    """Manages tournament standings and ratings."""
    
    K_FACTOR = 32  # Elo K-factor
    
    def __init__(self, checkpoint_store: CheckpointStore):
        self.store = checkpoint_store
        self.ratings: Dict[str, float] = {}  # strategy_id -> rating
    
    async def record_match(self, tournament_id: str, winner_id: str, loser_id: str, winner_result: SimulationResult, loser_result: SimulationResult):
        """Record match result and update ratings."""
        await self.store.update_standings(tournament_id, winner_id, winner_result)
        await self.store.update_standings(tournament_id, loser_id, loser_result)
        
        # Update Elo ratings
        self._update_elo(winner_id, loser_id, won=True)
    
    def _update_elo(self, player_id: str, opponent_id: str, won: bool):
        """Update Elo ratings."""
        r_player = self.ratings.get(player_id, 1500)
        r_opponent = self.ratings.get(opponent_id, 1500)
        
        expected = 1 / (1 + 10 ** ((r_opponent - r_player) / 400))
        actual = 1.0 if won else 0.0
        
        self.ratings[player_id] = r_player + self.K_FACTOR * (actual - expected)
        self.ratings[opponent_id] = r_opponent + self.K_FACTOR * ((1 - actual) - (1 - expected))
    
    async def get_leaderboard(self, tournament_id: str) -> List[Standing]:
        """Get ranked standings."""
        standings = await self.store.get_standings(tournament_id)
        
        # Sort by composite score
        sorted_standings = sorted(standings, key=lambda s: s.composite_score, reverse=True)
        
        # Assign ranks
        for i, standing in enumerate(sorted_standings):
            standing.rank = i + 1
        
        return sorted_standings
    
    async def get_daily_rollup(self, tournament_id: str, date: datetime) -> dict:
        """Get aggregated stats for a day."""
        events = await self.store.get_events(tournament_id)
        day_events = [e for e in events if e.created_at.date() == date.date()]
        
        return {
            "date": date.date().isoformat(),
            "simulations_run": len([e for e in day_events if e.event_type == "simulation_complete"]),
            "total_trades": sum(e.data.get("total_trades", 0) for e in day_events if e.event_type == "simulation_complete"),
            "avg_roi": statistics.mean([e.data.get("roi", 0) for e in day_events if e.event_type == "simulation_complete"]) if day_events else 0
        }
```

---

## Part 5: Streaming Layer

### 5.1 broadcast_manager.py

Extended WebSocket broadcasting.

```python
class BroadcastManager:
    """Manages WebSocket broadcasts for simulations."""
    
    def __init__(self, connection_manager):
        self.connections = connection_manager
        self.subscriptions: Dict[str, Set[WebSocket]] = {}  # channel -> websockets
        self.reconnect_delays = [1, 2, 4, 8, 16, 32]  # Exponential backoff
    
    async def subscribe(self, websocket: WebSocket, channel: str):
        """Subscribe websocket to channel."""
        if channel not in self.subscriptions:
            self.subscriptions[channel] = set()
        self.subscriptions[channel].add(websocket)
    
    async def unsubscribe(self, websocket: WebSocket, channel: str):
        """Unsubscribe websocket from channel."""
        if channel in self.subscriptions:
            self.subscriptions[channel].discard(websocket)
    
    async def emit(self, channel: str, event_type: str, data: dict):
        """Emit event to channel subscribers."""
        if channel not in self.subscriptions:
            return
        
        message = {
            "channel": channel,
            "event": event_type,
            "data": data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        dead_connections = []
        for ws in self.subscriptions[channel]:
            try:
                await ws.send_json(message)
            except:
                dead_connections.append(ws)
        
        for ws in dead_connections:
            self.subscriptions[channel].discard(ws)
    
    async def broadcast_all(self, event_type: str, data: dict):
        """Broadcast to all channels."""
        for channel in self.subscriptions:
            await self.emit(channel, event_type, data)
```

### 5.2 event_log.py

Three-phase logging integration.

```python
class PhaseEventLogger:
    """Logs phase events with three-phase pattern."""
    
    def __init__(self, checkpoint_store: CheckpointStore, broadcast_manager: BroadcastManager):
        self.store = checkpoint_store
        self.broadcast = broadcast_manager
    
    async def phase_start(self, simulation_id: str, phase: SimulationPhase, context: dict):
        """Log phase start."""
        event = SimulationEvent(
            correlation_id=simulation_id,
            event_type="phase_start",
            source=phase.value,
            data={"phase": phase.value, "context": context}
        )
        await self.store.log_event(event)
        await self.broadcast.emit(f"simulation:{simulation_id}", "phase_start", event.dict())
    
    async def phase_progress(self, simulation_id: str, phase: SimulationPhase, progress: float, message: str = None):
        """Log phase progress."""
        event = SimulationEvent(
            correlation_id=simulation_id,
            event_type="phase_progress",
            source=phase.value,
            data={"phase": phase.value, "progress": progress, "message": message}
        )
        await self.store.log_event(event)
        await self.broadcast.emit(f"simulation:{simulation_id}", "phase_progress", event.dict())
    
    async def phase_complete(self, simulation_id: str, phase: SimulationPhase, result: dict):
        """Log phase completion."""
        event = SimulationEvent(
            correlation_id=simulation_id,
            event_type="phase_complete",
            source=phase.value,
            data={"phase": phase.value, "result": result}
        )
        await self.store.log_event(event)
        await self.broadcast.emit(f"simulation:{simulation_id}", "phase_complete", event.dict())
    
    async def error(self, simulation_id: str, phase: SimulationPhase, error: str, traceback: str = None):
        """Log error."""
        event = SimulationEvent(
            correlation_id=simulation_id,
            event_type="error",
            source=phase.value,
            data={"phase": phase.value, "error": error, "traceback": traceback}
        )
        await self.store.log_event(event)
        await self.broadcast.emit(f"simulation:{simulation_id}", "error", event.dict())
```

### 5.3 websocket_routes.py

FastAPI WebSocket endpoints.

```python
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter(prefix="/ws/simulation", tags=["simulation-ws"])

@router.websocket("/{simulation_id}")
async def simulation_websocket(websocket: WebSocket, simulation_id: str):
    """WebSocket for single simulation updates."""
    await websocket.accept()
    broadcast = get_broadcast_manager()
    channel = f"simulation:{simulation_id}"
    
    await broadcast.subscribe(websocket, channel)
    
    try:
        while True:
            data = await websocket.receive_text()
            # Handle client messages if needed
    except WebSocketDisconnect:
        await broadcast.unsubscribe(websocket, channel)

@router.websocket("/tournament/{tournament_id}")
async def tournament_websocket(websocket: WebSocket, tournament_id: str):
    """WebSocket for tournament-wide events."""
    await websocket.accept()
    broadcast = get_broadcast_manager()
    channel = f"tournament:{tournament_id}"
    
    await broadcast.subscribe(websocket, channel)
    
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        await broadcast.unsubscribe(websocket, channel)

@router.websocket("/standings")
async def standings_websocket(websocket: WebSocket):
    """WebSocket for live leaderboard updates."""
    await websocket.accept()
    broadcast = get_broadcast_manager()
    
    await broadcast.subscribe(websocket, "standings")
    
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        await broadcast.unsubscribe(websocket, "standings")
```

---

## Part 6: API Routes

### 6.1 tournaments.py

```python
from fastapi import APIRouter, HTTPException, BackgroundTasks

router = APIRouter(prefix="/tournaments", tags=["tournaments"])

@router.post("/")
async def create_tournament(config: TournamentConfig) -> dict:
    """Create a new tournament."""
    
@router.get("/{tournament_id}")
async def get_tournament(tournament_id: str) -> dict:
    """Get tournament status and standings."""
    
@router.post("/{tournament_id}/start")
async def start_tournament(tournament_id: str, background_tasks: BackgroundTasks) -> dict:
    """Start tournament in background."""
    
@router.post("/{tournament_id}/stop")
async def stop_tournament(tournament_id: str) -> dict:
    """Gracefully stop tournament."""
    
@router.get("/{tournament_id}/standings")
async def get_standings(tournament_id: str) -> List[Standing]:
    """Get current standings."""
```

### 6.2 simulations.py

```python
router = APIRouter(prefix="/simulations", tags=["simulations"])

@router.post("/")
async def create_simulation(config: SimulationConfig) -> dict:
    """Start a single simulation."""
    
@router.get("/{simulation_id}")
async def get_simulation(simulation_id: str) -> dict:
    """Get simulation status and metrics."""
    
@router.get("/{simulation_id}/checkpoints")
async def get_checkpoints(simulation_id: str) -> List[Checkpoint]:
    """Get all checkpoints for replay."""
    
@router.post("/{simulation_id}/replay")
async def replay_simulation(simulation_id: str, from_checkpoint: str = None) -> dict:
    """Replay simulation from checkpoint."""
```

### 6.3 strategies.py

```python
router = APIRouter(prefix="/strategies", tags=["strategies"])

@router.post("/")
async def register_strategy(strategy: StrategyVersion) -> dict:
    """Register new strategy version."""
    
@router.get("/{strategy_id}")
async def get_strategy(strategy_id: str) -> StrategyVersion:
    """Get strategy details."""
    
@router.get("/{strategy_id}/history")
async def get_strategy_history(strategy_id: str) -> List[SimulationResult]:
    """Get performance history."""
    
@router.post("/{strategy_id}/evolve")
async def evolve_strategy(strategy_id: str, mutation: dict) -> StrategyVersion:
    """Create evolved version."""
    
@router.post("/crossover")
async def crossover_strategies(strategy_id_1: str, strategy_id_2: str) -> StrategyVersion:
    """Create hybrid strategy."""
```

---

## Integration Checklist

### Config Updates (config/settings.py)
- [ ] Add `SimulationSettings` class
- [ ] Add `SIMULATION_DB_PATH` setting
- [ ] Add `MAX_CONCURRENT_SIMULATIONS` setting
- [ ] Add scoring weight defaults

### MasterAI Integration (src/master_ai/brain.py)
- [ ] Add "SIMULATION" intent classification
- [ ] Route simulation commands to tournament runner
- [ ] Add simulation status queries

### Route Registration (src/api/main.py)
- [ ] Import simulation routers
- [ ] Register tournament routes
- [ ] Register simulation routes
- [ ] Register strategy routes
- [ ] Register WebSocket routes

### Testing
- [ ] Unit tests for each module
- [ ] Integration tests for full tournament flow
- [ ] Replay validation tests
- [ ] WebSocket streaming tests

---

## Implementation Order

1. **Week 1:** Core - `checkpoint_store.py`, `models.py`, `context.py`
2. **Week 2:** Execution - `phase_executor.py`, `provenance.py`, `mock_adapters.py`
3. **Week 3:** Orchestration - All 5 patterns
4. **Week 4:** Tournament - `tournament_runner.py`, `scoring.py`, `strategy_registry.py`, `standings.py`
5. **Week 5:** Streaming - `broadcast_manager.py`, `event_log.py`, `websocket_routes.py`
6. **Week 6:** API + Integration - All routes, config updates, MasterAI integration, tests
