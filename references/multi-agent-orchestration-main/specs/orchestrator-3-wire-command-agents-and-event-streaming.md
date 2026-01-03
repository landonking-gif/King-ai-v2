# Plan: Wire Orchestrator to Command-Level Agents with Event Streaming

## Task Description

Implement the complete agent management layer in `apps/orchestrator_3_stream` by creating agent CRUD operations, registering 8 management tools with the orchestrator agent, implementing command-level agent execution in background threads, and setting up real-time event streaming to the frontend via WebSocket and HTTP endpoints. This builds on the completed orchestrator-to-user chat functionality and references the proven patterns from `apps/orchestrator_1_term`.

## Objective

Enable the orchestrator agent to create, manage, and command other Claude Code agents by:
- Registering 8 management tools (`create_agent`, `list_agents`, `command_agent`, `check_agent_status`, `delete_agent`, `interrupt_agent`, `read_system_logs`, `report_cost`) with the orchestrator
- Implementing agent lifecycle management (create → execute → monitor → cleanup)
- Running command-level agents in background threads with session continuity
- Capturing agent execution events via hooks (PreToolUse, PostToolUse, UserPromptSubmit, Stop, SubagentStop, PreCompact)
- Streaming agent events to the EventStream UI component in real-time
- Providing HTTP `GET /events` endpoint for event history retrieval
- Maintaining full database persistence for all agent operations and logs

## Problem Statement

The `apps/orchestrator_3_stream` application currently supports orchestrator-to-user chat (Phase 1 complete) but lacks the core agent management capabilities that make it an orchestration system. Specifically:

**Missing Backend Infrastructure:**
- No agent CRUD database operations (create_agent, get_agent, list_agents, update_agent_costs, update_agent_session, delete_agent)
- No AgentManager class for lifecycle management
- No tool registration system for orchestrator agent
- No hook implementation for event capture
- No background thread execution for command-level agents
- No agent log insertion/retrieval functions
- No event streaming integration with WebSocket

**Missing Frontend Integration:**
- EventStream component shows test data only
- No real-time agent log display
- No agent status updates in AgentList sidebar
- No event filtering or search capabilities

**Reference Implementation Available:**
`apps/orchestrator_1_term` has fully functional agent management with:
- 8 registered tools using `@tool` decorator pattern
- AgentManager with create, command, interrupt, status check capabilities
- 6 hook types capturing all execution events
- Session continuity via `resume=session_id` parameter
- Background async task spawning for AI summarization
- Comprehensive database logging to `agent_logs` table

## Solution Approach

Adapt the proven patterns from `orchestrator_1_term` to the streaming web architecture in `orchestrator_3_stream`:

**Five-Layer Architecture:**

```
Frontend (Vue 3 + EventStream Component)
    ↓ WebSocket + HTTP /events endpoint
Backend WebSocket Manager (Broadcast Events)
    ↓ Event hooks + Direct broadcasts
Agent Manager (Background threads, session management)
    ↓ Agent database operations + Hook registration
Database (agents, agent_logs, prompts tables)
```

**Key Patterns to Implement:**

1. **Tool Registration Pattern** (from orchestrator_1_term/orchestrator_agent.py):
   - Use `@tool` decorator from `claude-agent-sdk`
   - Each tool returns structured dict with `content` array
   - Tools registered via `create_sdk_mcp_server()`
   - Pass to ClaudeAgentOptions with `mcp_servers` parameter

2. **Agent Lifecycle Pattern** (from orchestrator_1_term/agent_manager.py):
   - **Create**: DB record → Hooks → Session init → Capture session_id
   - **Command**: Load agent → Resume session → Execute → Log events → Update session
   - **Interrupt**: Track active clients, call `client.stop()`
   - **Status**: Query agent_logs for recent events

3. **Hook Factory Pattern** (from orchestrator_1_term/hooks.py):
   - Factory function accepts `(agent_id, task_slug, entry_counter, logger)`
   - Returns async hook callable: `(input_data, tool_use_id, context) -> Dict`
   - Hook inserts to DB, spawns async summarization task
   - Entry counter ensures sequential ordering

4. **Event Streaming Pattern** (NEW for orchestrator_3_stream):
   - Hooks write to DB AND broadcast via WebSocket
   - Message processing emits events in real-time
   - Frontend EventStream component receives and displays
   - HTTP `/events` endpoint for history/filtering

5. **Background Thread Execution** (from orchestrator_1_term/agent_manager.py):
   - Command-level agents run in asyncio tasks
   - Track in `active_clients` dict with threading.Lock
   - Allows orchestrator to issue multiple commands without blocking
   - Interrupt support via client tracking

## Relevant Files

### Reference Implementation Files (Read for patterns)

#### Agent Management Reference

- **apps/orchestrator_1_term/modules/orchestrator_agent.py** (Lines: 134-735)
  - Why: Shows tool registration with `@tool` decorator, tool schemas, return formats
  - Pattern: `_create_management_tools()` method returning list of 8 tool callables
  - Pattern: MCP server creation with `create_sdk_mcp_server(name="mgmt", tools=...)`

- **apps/orchestrator_1_term/modules/agent_manager.py** (Lines: 80-1127)
  - Why: Complete agent lifecycle implementation (create, command, interrupt, status)
  - Pattern: `create_agent()` 8-step process (lines 128-272)
  - Pattern: `command_agent()` session resumption flow (lines 389-571)
  - Pattern: `_process_agent_messages()` message loop with logging (lines 678-1079)
  - Pattern: `_build_hooks_for_agent()` hook registration (lines 621-672)

#### Hook and Event Logging Reference

- **apps/orchestrator_1_term/modules/hooks.py** (Lines: 41-632)
  - Why: 6 hook factories with DB insertion and async summarization
  - Pattern: `create_pre_tool_hook()` - PreToolUse event capture (lines 41-127)
  - Pattern: `create_post_tool_hook()` - PostToolUse event capture (lines 130-220)
  - Pattern: `create_pre_compact_hook()` - Token reset logic (lines 475-580)
  - Pattern: `_summarize_and_update()` - Background AI summarization (lines 588-632)

- **apps/orchestrator_1_term/modules/database/agent_logs.py** (Lines: 15-375)
  - Why: Agent event logging functions to copy/adapt
  - Functions: `insert_hook_event()`, `insert_message_block()`, `update_log_summary()`
  - Functions: `get_tail_summaries()`, `get_tail_raw()`, `get_latest_task_slug()`

- **apps/orchestrator_1_term/modules/database/agents.py** (Lines: 15-250)
  - Why: Agent CRUD operations to copy/adapt
  - Functions: `create_agent()`, `get_agent()`, `get_agent_by_name()`, `list_agents()`
  - Functions: `update_agent_session()`, `update_agent_costs()`, `delete_agent()`

#### Database Models

- **apps/orchestrator_db/models.py** (Lines: 88-143, 187-230)
  - Agent model: id, name, model, system_prompt, working_dir, status, session_id, costs, metadata
  - AgentLog model: agent_id, task_slug, entry_index, event_category, event_type, content, payload, summary
  - Already synced to: `apps/orchestrator_3_stream/backend/modules/orch_database_models.py`

#### System Prompts

- **apps/orchestrator_1_term/prompts/orchestrator_agent_system_prompt.md**
  - Why: Documents all 8 tools with parameters and usage examples
  - Copy to: `apps/orchestrator_3_stream/backend/prompts/` (already exists, will need tool updates)

- **apps/orchestrator_1_term/prompts/managed_agent_system_prompt_template.md**
  - Why: Template for command-level agent system prompts
  - Variables: {agent_name}, {working_dir}, {model}, {session_id}, {capabilities_section}

### Backend Files to Modify

#### Core Service Layer

- **apps/orchestrator_3_stream/backend/modules/database.py** (Lines: 310+)
  - Why: Add agent CRUD and event logging functions
  - Add functions: `create_agent()`, `get_agent()`, `get_agent_by_name()`, `list_agents()`
  - Add functions: `update_agent_session()`, `update_agent_costs()`, `update_agent_status()`, `delete_agent()`
  - Add functions: `insert_agent_log()`, `insert_hook_event()`, `insert_message_block()`
  - Add functions: `get_agent_logs()`, `get_tail_summaries()`, `get_tail_raw()`, `update_log_summary()`
  - Add functions: `insert_prompt()` for prompt tracking
  - Pattern: Follow existing `get_connection()` and asyncpg query patterns

- **apps/orchestrator_3_stream/backend/modules/orchestrator_service.py** (Lines: 172+)
  - Why: Needs agent_manager integration and tool availability
  - Modify: `__init__()` to accept `agent_manager` parameter
  - Modify: `_create_claude_agent_options()` to include MCP tools server
  - Add: `self.management_tools` from agent_manager
  - Integration: Orchestrator execution now has access to agent management tools

#### WebSocket Enhancements

- **apps/orchestrator_3_stream/backend/modules/websocket_manager.py** (Lines: 190+)
  - Why: Already has agent event broadcast methods, ensure they're used
  - Existing methods: `broadcast_agent_created()`, `broadcast_agent_updated()`, `broadcast_agent_status_change()`, `broadcast_agent_log()`
  - Verify: Methods match expected frontend event structure
  - No changes needed unless event formats need adjustment

#### Configuration

- **apps/orchestrator_3_stream/backend/modules/config.py** (Lines: 137+)
  - Why: Add agent-specific defaults
  - Add: `DEFAULT_AGENT_MODEL` (default "claude-sonnet-4-5-20250929")
  - Add: `AGENT_SYSTEM_PROMPT_TEMPLATE_PATH` (default "./prompts/managed_agent_system_prompt_template.md")
  - Add: `MAX_AGENT_TURNS` (default 500)

#### API Endpoints

- **apps/orchestrator_3_stream/backend/main.py** (Lines: 264+)
  - Why: Add agent management HTTP endpoints and event retrieval
  - Add routes: `POST /create_agent`, `GET /list_agents`, `POST /command_agent/{agent_id}`, `GET /agent_status/{agent_id}`, `DELETE /agent/{agent_id}`, `POST /interrupt_agent/{agent_id}`
  - Add route: `GET /events` - retrieve agent logs with filtering (agent_id, task_slug, limit, offset)
  - Add route: `GET /agent_logs/{agent_id}` - agent-specific event log retrieval
  - Modify: `lifespan()` to instantiate AgentManager and pass to OrchestratorService

### Backend Files to Create

#### Agent Management Core

- **apps/orchestrator_3_stream/backend/modules/agent_manager.py** (NEW)
  - Why: Centralize agent lifecycle management, tool registration, background execution
  - Class: `AgentManager`
    - `__init__(db, ws_manager, logger, config)` - Store dependencies
    - `active_clients: Dict[str, ClaudeSDKClient]` - Track running agents for interrupt
    - `active_clients_lock: threading.Lock` - Thread safety
    - `create_management_tools()` - Register 8 tools with @tool decorator
    - `create_agent(name, system_prompt, model, allowed_tools, disallowed_tools)` - Agent creation flow
    - `list_agents(archived=False)` - Query database for agents
    - `command_agent(agent_id, command, task_slug=None)` - Execute task on agent
    - `check_agent_status(agent_id, tail_count=10, verbose_logs=False)` - Get agent status + recent logs
    - `delete_agent(agent_id)` - Mark agent as archived
    - `interrupt_agent(agent_id)` - Stop running agent
    - `report_cost(session_id=None)` - Get token usage and costs
    - `_build_hooks_for_agent(agent_id, task_slug, entry_counter)` - Create hook dict
    - `_process_agent_messages(client, agent_id, task_slug, entry_counter)` - Message processing loop
    - `_summarize_and_update_block(log_id, block_type, block_data)` - Background AI summarization
  - Pattern: Copy structure from `apps/orchestrator_1_term/modules/agent_manager.py`
  - Integration: Tools returned from `create_management_tools()` passed to orchestrator

#### Hook System

- **apps/orchestrator_3_stream/backend/modules/hooks.py** (NEW)
  - Why: Capture all agent execution events and emit to WebSocket
  - Functions: `create_pre_tool_hook(agent_id, task_slug, entry_counter, logger, ws_manager)`
  - Functions: `create_post_tool_hook(...)`, `create_user_prompt_hook(...)`, `create_stop_hook(...)`
  - Functions: `create_subagent_stop_hook(...)`, `create_pre_compact_hook(...)`
  - Helper: `_summarize_and_update(log_id, event_type, event_data, logger)`
  - Pattern: Each hook:
    1. Extract event data from input_data
    2. Get entry_index from counter, increment
    3. Insert to database via `insert_hook_event()`
    4. Broadcast via `ws_manager.broadcast_agent_log()`
    5. Spawn background task for AI summarization
    6. Return empty dict `{}`
  - NEW: WebSocket integration in each hook for real-time streaming

#### Event Summarization

- **apps/orchestrator_3_stream/backend/modules/event_summarizer.py** (NEW)
  - Why: Generate AI summaries for agent events (lightweight log viewing)
  - Function: `async def summarize_event(event_data: Dict, event_type: str) -> str`
  - Uses: Claude SDK with Haiku model for fast, cheap summarization
  - Pattern: Similar to `apps/orchestrator_1_term/modules/single_agent_prompt.py`
  - Returns: 1-sentence summary (max 100 characters)
  - Fallback: Template-based summaries if AI fails

#### System Prompts

- **apps/orchestrator_3_stream/backend/prompts/managed_agent_system_prompt_template.md** (NEW)
  - Why: Template for command-level agents
  - Copy from: `apps/orchestrator_1_term/prompts/managed_agent_system_prompt_template.md`
  - Variables: {agent_name}, {working_dir}, {model}, {session_id}

### Frontend Files to Modify

#### Event Stream Component

- **apps/orchestrator_3_stream/frontend/src/components/EventStream.vue** (Lines: 1-300)
  - Why: Replace test data with real WebSocket events
  - Modify: `events` data to consume from Pinia store
  - Add: Real-time event append on WebSocket message
  - Add: Event filtering by level (DEBUG, INFO, WARNING, ERROR)
  - Add: Search functionality for event content
  - Integration: Listen to `ws_manager.broadcast_agent_log()` events

#### Store Integration

- **apps/orchestrator_3_stream/frontend/src/stores/orchestratorStore.ts** (Lines: 200+)
  - Why: Add agent event handling to store
  - Add: `agentEvents: Ref<EventStreamEntry[]>` - store all agent logs
  - Add: `addAgentEvent(event)` - append new event, trigger auto-scroll if enabled
  - Modify: `handleWebSocketMessage()` to route `agent_log` messages to `addAgentEvent()`
  - Add: `loadAgentEvents(agentId?, limit?)` - HTTP GET to `/events` for history

#### Agent List Component

- **apps/orchestrator_3_stream/frontend/src/components/AgentList.vue** (Lines: 1-200)
  - Why: Display real agents from database
  - Modify: `agents` data to load from API
  - Add: HTTP GET to `/list_agents` on mount
  - Add: WebSocket listener for `agent_created`, `agent_updated`, `agent_deleted`, `agent_status_changed`
  - Display: Agent name, status, model, cost

#### Type Definitions

- **apps/orchestrator_3_stream/frontend/src/types.d.ts** (Lines: 220+)
  - Add: `CreateAgentRequest { name, system_prompt, model, allowed_tools?, disallowed_tools? }`
  - Add: `ListAgentsResponse { agents: Agent[] }`
  - Add: `CommandAgentRequest { command, task_slug? }`
  - Add: `AgentStatusResponse { agent, recent_logs }`
  - Add: `EventLogEntry { id, agent_id, event_type, event_category, content, payload, timestamp }`
  - Add: `GetEventsRequest { agent_id?, task_slug?, limit?, offset? }`
  - Add: `GetEventsResponse { events: EventLogEntry[], total }`

### Frontend Files to Create

#### Agent Service

- **apps/orchestrator_3_stream/frontend/src/services/agentService.ts** (NEW)
  - Why: Centralize agent management API calls
  - Function: `createAgent(name, systemPrompt, model)` - POST to `/create_agent`
  - Function: `listAgents()` - GET from `/list_agents`
  - Function: `commandAgent(agentId, command)` - POST to `/command_agent/{agentId}`
  - Function: `getAgentStatus(agentId)` - GET from `/agent_status/{agentId}`
  - Function: `deleteAgent(agentId)` - DELETE to `/agent/{agentId}`
  - Function: `interruptAgent(agentId)` - POST to `/interrupt_agent/{agentId}`
  - Function: `getEvents(filters)` - GET from `/events` with query params

### Test Files to Create

#### Backend Tests

- **apps/orchestrator_3_stream/backend/tests/test_agent_manager.py** (NEW)
  - Test: `test_create_agent_with_real_database()` - Create agent, verify DB record
  - Test: `test_command_agent_with_real_sdk()` - Send command, verify execution
  - Test: `test_agent_session_continuity()` - Send 2 commands, verify session maintained
  - Test: `test_interrupt_agent()` - Start long task, interrupt, verify stopped
  - IMPORTANT: NO MOCKING - Use real database and Claude SDK

- **apps/orchestrator_3_stream/backend/tests/test_hooks.py** (NEW)
  - Test: `test_pre_tool_hook_logs_to_database()` - Trigger hook, verify DB insertion
  - Test: `test_hooks_broadcast_websocket_events()` - Verify WebSocket emission
  - Test: `test_entry_counter_sequential_ordering()` - Verify entry_index increments

- **apps/orchestrator_3_stream/backend/tests/test_event_endpoints.py** (NEW)
  - Test: `test_get_events_endpoint()` - Query events, verify filtering
  - Test: `test_get_agent_logs_endpoint()` - Agent-specific log retrieval

#### Frontend Tests

- **apps/orchestrator_3_stream/frontend/src/components/__tests__/EventStream.spec.ts** (NEW)
  - Test: WebSocket event reception and display
  - Test: Event filtering by level
  - Test: Search functionality

## Implementation Phases

### Phase 1: Database Layer Extension (Agent CRUD + Event Logging)

Extend `database.py` with all agent and event logging operations. This unblocks all backend development.

**Goal**: Complete database module with 20+ functions for agent management and event logging.

**Estimated Time**: 4-6 hours

### Phase 2: Agent Manager & Tool Registration

Implement `AgentManager` class with lifecycle management and create 8 management tools using `@tool` decorator.

**Goal**: Orchestrator can create, list, command, and manage agents via tools.

**Estimated Time**: 6-8 hours

### Phase 3: Hook System & Event Streaming

Implement 6 hook factories and integrate WebSocket broadcasting for real-time event streaming.

**Goal**: All agent execution events captured and streamed to frontend.

**Estimated Time**: 5-7 hours

### Phase 4: API Endpoints & Service Integration

Add HTTP endpoints for agent operations and event retrieval. Integrate AgentManager with OrchestratorService.

**Goal**: Frontend can create agents, issue commands, and retrieve event logs via API.

**Estimated Time**: 3-4 hours

### Phase 5: Frontend Integration

Update EventStream and AgentList components with real data, implement agent service, and connect WebSocket events.

**Goal**: UI displays real-time agent events and agent status updates.

**Estimated Time**: 4-5 hours

## Step by Step Tasks

IMPORTANT: Execute every step in order, top to bottom.

### 1. Extend Database Module with Agent Operations

- Edit `apps/orchestrator_3_stream/backend/modules/database.py` (after line 310)
- Add agent CRUD operations:
  - `async def create_agent(name, model, system_prompt, working_dir, metadata)` - INSERT agent, return UUID
    - Include fields: git_worktree, allowed_tools/disallowed_tools in metadata
    - Set initial status='idle', archived=False
  - `async def get_agent(agent_id: UUID)` - SELECT agent by ID, return dict or None
  - `async def get_agent_by_name(name: str)` - SELECT agent by name (unique constraint)
  - `async def list_agents(archived: bool = False)` - SELECT all agents WHERE archived=?, return list
  - `async def update_agent_session(agent_id, session_id)` - UPDATE session_id
  - `async def update_agent_status(agent_id, status)` - UPDATE status field
  - `async def update_agent_costs(agent_id, input_tokens, output_tokens, cost_usd)` - Incremental update
  - `async def delete_agent(agent_id)` - UPDATE archived=True (soft delete)
  - `async def reset_agent_tokens(agent_id)` - SET input_tokens=0, output_tokens=0 (for PreCompact hook)
- Copy patterns from `apps/orchestrator_1_term/modules/database/agents.py`
- Use `get_connection()` context manager, asyncpg queries, proper error handling

### 2. Add Agent Event Logging Functions

- Continue editing `apps/orchestrator_3_stream/backend/modules/database.py`
- Add event logging operations:
  - `async def insert_hook_event(agent_id, task_slug, entry_index, event_type, payload, session_id=None, adw_id=None, adw_step=None)` - INSERT to agent_logs with event_category='hook', return log UUID
  - `async def insert_message_block(agent_id, task_slug, entry_index, block_type, content, payload, session_id=None, adw_id=None, adw_step=None)` - INSERT to agent_logs with event_category='response', return log UUID
  - `async def update_log_summary(log_id: UUID, summary: str)` - UPDATE agent_logs SET summary=? WHERE id=?
  - `async def get_agent_logs(agent_id, task_slug=None, limit=50, offset=0)` - SELECT from agent_logs with filtering
  - `async def get_tail_summaries(agent_id, task_slug, count=10, offset=0)` - SELECT last N events WHERE summary IS NOT NULL
  - `async def get_tail_raw(agent_id, task_slug, count=10, offset=0)` - SELECT last N events with full content + payload
  - `async def get_latest_task_slug(agent_id)` - SELECT most recent task_slug for agent
  - `async def insert_prompt(agent_id, task_slug, author, prompt_text, session_id=None)` - INSERT to prompts table
- Copy patterns from `apps/orchestrator_1_term/modules/database/agent_logs.py` and `prompts.py`
- Ensure proper indexing on agent_id, task_slug, entry_index for tail reading performance

### 3. Create Event Summarizer Module

- Create `apps/orchestrator_3_stream/backend/modules/event_summarizer.py`
- Import Claude SDK: `from claude_code_sdk import ClaudeSDKClient, ClaudeAgentOptions`
- Implement `async def summarize_event(event_data: Dict[str, Any], event_type: str) -> str`:
  - Build prompt: "Summarize this {event_type} event in 1 sentence (max 100 chars): {event_data}"
  - Use Claude Haiku for fast, cheap summarization
  - Return summary string
  - On error: Fall back to template-based summary (e.g., "Tool {tool_name} executed")
- Implement template-based fallback: `_generate_simple_summary(event_type, event_data)`:
  - PreToolUse: "Preparing {tool_name} with {input_preview}"
  - PostToolUse: "{tool_name} completed" or "{tool_name} failed"
  - UserPromptSubmit: "User: {prompt_preview}"
  - Stop: "Agent stopped after {num_turns} turns"
- Pattern: Similar to `apps/orchestrator_1_term/modules/single_agent_prompt.py`
- Import in hooks module for async summarization calls

### 4. Create Hook System Module

- Create `apps/orchestrator_3_stream/backend/modules/hooks.py`
- Import dependencies: `asyncio`, `uuid`, `datetime`, event_summarizer, database functions, websocket_manager
- Implement hook factory for PreToolUse:
  - `def create_pre_tool_hook(agent_id, task_slug, entry_counter, logger, ws_manager) -> HookCallback`
  - Return async hook callable: `async def hook(input_data, tool_use_id, context) -> Dict`
  - Hook implementation:
    1. Extract tool_name, tool_input from input_data
    2. Build payload dict with tool_name, tool_input, tool_use_id, timestamp
    3. Get entry_index from counter["count"], increment counter
    4. Log to file: `logger.info(f"[Hook:PreToolUse] Agent={agent_id} Entry={entry_index} Tool={tool_name}")`
    5. Insert to database: `log_id = await insert_hook_event(agent_id, task_slug, entry_index, "PreToolUse", payload)`
    6. Broadcast WebSocket event: `await ws_manager.broadcast_agent_log({"id": str(log_id), "agent_id": str(agent_id), "event_type": "PreToolUse", "entry_index": entry_index, "payload": payload})`
    7. Spawn background summarization: `asyncio.create_task(_summarize_and_update(log_id, "PreToolUse", payload, logger))`
    8. Return {}
- Implement remaining 5 hook factories following same pattern:
  - `create_post_tool_hook()` - Captures tool results, errors
  - `create_user_prompt_hook()` - Captures user prompts (truncate to 1000 chars)
  - `create_stop_hook()` - Captures session stops with reason, num_turns, duration_ms
  - `create_subagent_stop_hook()` - Captures subagent completions
  - `create_pre_compact_hook()` - Captures compaction events AND resets agent token counters
- Implement helper function:
  - `async def _summarize_and_update(log_id, event_type, event_data, logger)`:
    - Call `summary = await summarize_event(event_data, event_type)`
    - Update database: `await update_log_summary(log_id, summary)`
    - Log result: `logger.debug(f"[Hook:Summary] Generated summary for log_id={log_id}: {summary}")`
    - Handle errors gracefully (don't crash if summarization fails)
- Pattern: Copy structure from `apps/orchestrator_1_term/modules/hooks.py`
- NEW: WebSocket integration in each hook for real-time frontend updates

### 5. Create Agent Manager Module

- Create `apps/orchestrator_3_stream/backend/modules/agent_manager.py`
- Import dependencies: `threading`, `asyncio`, `uuid`, `datetime`, Claude SDK, database functions, hooks module, websocket_manager, logger
- Implement `AgentManager` class:
  - `__init__(self, db, ws_manager, logger, config)`:
    - Store dependencies: self.db, self.ws_manager, self.logger, self.config
    - Initialize: `self.active_clients = {}`, `self.active_clients_lock = threading.Lock()`
  - `create_management_tools(self) -> List[Callable]`:
    - Use `@tool` decorator from `from claude_code_sdk.tools import tool`
    - Create 8 tool callables:
      1. `create_agent` - name, system_prompt, model (optional), allowed_tools (optional), disallowed_tools (optional)
      2. `list_agents` - no parameters
      3. `command_agent` - agent_name, command, task_slug (optional)
      4. `check_agent_status` - agent_name, tail_count (default 10), offset (default 0), verbose_logs (default False)
      5. `delete_agent` - agent_name
      6. `interrupt_agent` - agent_name
      7. `read_system_logs` - offset (default 0), limit (default 20), desc (default True), agent_id (optional), message_contains (optional), level (optional)
      8. `report_cost` - session_id (optional)
    - Each tool returns dict with `content` array: `{"content": [{"type": "text", "text": "result message"}]}`
    - On error: `{"content": [...], "is_error": True}`
    - Pattern: Copy from `apps/orchestrator_1_term/modules/orchestrator_agent.py` lines 134-735
  - `async def create_agent(self, name, system_prompt, model=None, allowed_tools=None, disallowed_tools=None)`:
    - Validate name uniqueness via `get_agent_by_name()`
    - Create agent record: `agent_id = await create_agent(name, model, system_prompt, working_dir, metadata)`
    - Build hooks: `hooks_dict = self._build_hooks_for_agent(agent_id, f"{name}-greeting", {"count": 0})`
    - Build options: `options = ClaudeAgentOptions(system_prompt=system_prompt, model=model, hooks=hooks_dict, ...)`
    - Initialize agent with greeting: `async with ClaudeSDKClient(options) as client: await client.query("Ready. Awaiting instructions.")`
    - Process messages to capture session_id: `session_id = await self._process_agent_messages(client, agent_id, ...)`
    - Update session in DB: `await update_agent_session(agent_id, session_id)`
    - Broadcast creation: `await self.ws_manager.broadcast_agent_created({"id": str(agent_id), "name": name, ...})`
    - Return success response: `{"ok": True, "agent_id": str(agent_id), "session_id": session_id}`
    - Pattern: Copy from `apps/orchestrator_1_term/modules/agent_manager.py` lines 128-272
  - `async def command_agent(self, agent_id_or_name, command, task_slug=None)`:
    - Load agent from DB by ID or name
    - Generate task_slug if not provided: `f"{slugify(command[:50])}-{timestamp}"`
    - Build hooks with task_slug
    - Build options with `resume=agent.session_id`
    - Execute in background (or await): `async with ClaudeSDKClient(options) as client: await client.query(command); session_id = await self._process_agent_messages(...)`
    - Track in active_clients for interrupt support
    - Update session and costs in DB
    - Broadcast status change: `await self.ws_manager.broadcast_agent_status_change(agent_id, "idle", "executing")`
    - Return: `{"ok": True, "task_slug": task_slug}`
    - Pattern: Copy from `apps/orchestrator_1_term/modules/agent_manager.py` lines 389-571
  - `async def check_agent_status(self, agent_id_or_name, tail_count=10, offset=0, verbose_logs=False)`:
    - Load agent from DB
    - Get latest task_slug: `task_slug = await get_latest_task_slug(agent_id)`
    - Retrieve logs: `logs = await get_tail_summaries(agent_id, task_slug, tail_count, offset)` if not verbose, else `get_tail_raw(...)`
    - Format response with agent status + recent activity
    - Return: `{"ok": True, "agent": {...}, "recent_logs": [...]}`
  - `async def delete_agent(self, agent_id_or_name)`:
    - Soft delete: `await delete_agent(agent_id)`
    - Broadcast deletion: `await self.ws_manager.broadcast_agent_deleted(agent_id)`
    - Return: `{"ok": True}`
  - `async def interrupt_agent(self, agent_id_or_name)`:
    - Load agent, get name
    - Check active_clients: `with self.active_clients_lock: client = self.active_clients.get(name)`
    - If found: `await client.stop()`, remove from active_clients
    - Update status: `await update_agent_status(agent_id, "blocked")`
    - Return: `{"ok": True, "interrupted": True}` or `{"ok": False, "error": "Agent not running"}`
  - `async def report_cost(self, session_id=None)`:
    - If session_id: get orchestrator by session, report costs
    - Else: get all agents, sum costs
    - Format: token counts, USD cost, context window usage
    - Return formatted report
  - `def _build_hooks_for_agent(self, agent_id, task_slug, entry_counter)`:
    - Create hooks dict: `{"PreToolUse": [{"hooks": [create_pre_tool_hook(...)]}], ...}`
    - Include all 6 hook types
    - Return hooks dict
    - Pattern: Copy from `apps/orchestrator_1_term/modules/agent_manager.py` lines 621-672
  - `async def _process_agent_messages(self, client, agent_id, agent_name, task_slug, entry_counter)`:
    - Iterate: `async for message in client.receive_response():`
    - For AssistantMessage: iterate blocks (TextBlock, ThinkingBlock, ToolUseBlock, ToolResultBlock)
      - Get entry_index, increment counter
      - Insert message block to DB
      - Broadcast via WebSocket
      - Spawn async summarization
      - Display in console (Rich panel)
    - For ResultMessage: capture session_id, update costs
    - Return session_id
    - Pattern: Copy from `apps/orchestrator_1_term/modules/agent_manager.py` lines 678-1079
  - `async def _summarize_and_update_block(self, log_id, block_type, block_data)`:
    - Call event_summarizer.summarize_event()
    - Update DB with summary
    - Handle errors gracefully
- Pattern: Overall structure from `apps/orchestrator_1_term/modules/agent_manager.py`

### 6. Integrate Agent Manager with Orchestrator Service

- Edit `apps/orchestrator_3_stream/backend/modules/orchestrator_service.py`
- Modify `__init__()` method:
  - Add parameter: `agent_manager: AgentManager`
  - Store: `self.agent_manager = agent_manager`
  - Get management tools: `self.management_tools = agent_manager.create_management_tools()`
- Modify `_create_claude_agent_options()` method (around line 98):
  - Import: `from claude_code_sdk.tools import create_sdk_mcp_server`
  - Create MCP server: `mcp_server = create_sdk_mcp_server(name="mgmt", version="1.0.0", tools=self.management_tools)`
  - Add to ClaudeAgentOptions: `mcp_servers={"mgmt": mcp_server}`
  - Update allowed_tools list: `["mcp__mgmt__create_agent", "mcp__mgmt__list_agents", "mcp__mgmt__command_agent", "mcp__mgmt__check_agent_status", "mcp__mgmt__delete_agent", "mcp__mgmt__interrupt_agent", "mcp__mgmt__read_system_logs", "mcp__mgmt__report_cost", "Bash"]`
- Pattern: Tool registration from `apps/orchestrator_1_term/modules/orchestrator_agent.py` lines 126-128, 773-800

### 7. Update Main Application Lifespan with Agent Manager

- Edit `apps/orchestrator_3_stream/backend/main.py`
- Import: `from modules.agent_manager import AgentManager`
- Modify `lifespan()` function startup section (around line 34-49):
  - After database initialization
  - Instantiate: `agent_manager = AgentManager(db=None, ws_manager=ws_manager, logger=logger, config=config)`
    - Note: db=None because agent_manager will use database module functions directly
  - Pass to orchestrator service: `orchestrator_service = OrchestratorService(db_manager=None, ws_manager=ws_manager, logger=logger, agent_manager=agent_manager, session_id=..., working_dir=...)`
  - Store in app.state: `app.state.agent_manager = agent_manager`
  - Log initialization: `logger.info("Agent Manager initialized with 8 management tools")`
- No changes needed in shutdown section (agent_manager cleanup handled by close_pool)

### 8. Add Agent Management HTTP Endpoints

- Edit `apps/orchestrator_3_stream/backend/main.py` (after line 264, after /send_chat endpoint)
- Add Pydantic request models:
  - `class CreateAgentRequest(BaseModel): name: str, system_prompt: str, model: Optional[str] = None, allowed_tools: Optional[List[str]] = None, disallowed_tools: Optional[List[str]] = None`
  - `class CommandAgentRequest(BaseModel): command: str, task_slug: Optional[str] = None`
  - `class GetEventsRequest(BaseModel): agent_id: Optional[str] = None, task_slug: Optional[str] = None, limit: int = 50, offset: int = 0`
- Add endpoints:
  - `@app.post("/create_agent")`:
    - Accept CreateAgentRequest
    - Call `agent_manager = app.state.agent_manager; result = await agent_manager.create_agent(...)`
    - Return: `{"status": "success", "agent": result}` or HTTP 500 on error
  - `@app.get("/list_agents")`:
    - Call `agents = await list_agents(archived=False)` from database module
    - Return: `{"status": "success", "agents": agents}`
  - `@app.post("/command_agent/{agent_id}")`:
    - Accept agent_id path param, CommandAgentRequest body
    - Call `result = await agent_manager.command_agent(agent_id, request.command, request.task_slug)`
    - Return: `{"status": "success", "task_slug": result["task_slug"]}`
  - `@app.get("/agent_status/{agent_id}")`:
    - Accept agent_id path param, query params: tail_count, offset, verbose_logs
    - Call `result = await agent_manager.check_agent_status(agent_id, tail_count, offset, verbose_logs)`
    - Return: `{"status": "success", "agent": result["agent"], "recent_logs": result["recent_logs"]}`
  - `@app.delete("/agent/{agent_id}")`:
    - Accept agent_id path param
    - Call `await agent_manager.delete_agent(agent_id)`
    - Return: `{"status": "success", "deleted": True}`
  - `@app.post("/interrupt_agent/{agent_id}")`:
    - Accept agent_id path param
    - Call `result = await agent_manager.interrupt_agent(agent_id)`
    - Return result dict
  - `@app.get("/events")`:
    - Accept query params: agent_id, task_slug, limit, offset
    - Build filters from query params
    - Call `events = await get_agent_logs(agent_id, task_slug, limit, offset)` from database module
    - Return: `{"status": "success", "events": events, "count": len(events)}`
  - `@app.get("/agent_logs/{agent_id}")`:
    - Accept agent_id path param, query params: limit, offset, task_slug
    - Call `logs = await get_agent_logs(agent_id, task_slug, limit, offset)`
    - Return: `{"status": "success", "logs": logs}`
- Add error handling: try/except with HTTP 500 response, log errors
- Log all requests: `logger.http_request(method, path, status_code)`

### 9. Update Backend Configuration

- Edit `apps/orchestrator_3_stream/backend/modules/config.py`
- Add after line 137 (after ORCHESTRATOR_WORKING_DIR):
  ```python
  # ============================================================================
  # AGENT CONFIGURATION
  # ============================================================================

  # Default model for managed agents
  DEFAULT_AGENT_MODEL = os.getenv("DEFAULT_AGENT_MODEL", DEFAULT_MODEL)

  # Agent system prompt template path
  AGENT_SYSTEM_PROMPT_TEMPLATE_PATH = os.getenv(
      "AGENT_SYSTEM_PROMPT_TEMPLATE_PATH",
      str(BACKEND_DIR / "prompts" / "managed_agent_system_prompt_template.md")
  )

  # Maximum turns for agent execution
  MAX_AGENT_TURNS = int(os.getenv("MAX_AGENT_TURNS", "500"))
  ```
- Update `.env.sample` with new variables

### 10. Create Managed Agent System Prompt Template

- Create `apps/orchestrator_3_stream/backend/prompts/managed_agent_system_prompt_template.md`
- Copy content from `apps/orchestrator_1_term/prompts/managed_agent_system_prompt_template.md`
- Template variables: {agent_name}, {working_dir}, {model}, {session_id}, {capabilities_section}
- Content should emphasize:
  - Agent identity and specialization
  - Working directory and session info
  - Task focus and autonomy
  - Communication of blockers and clarifications
  - Task completion protocol (summarize, note issues, suggest next steps)

### 11. Create Frontend Agent Service

- Create `apps/orchestrator_3_stream/frontend/src/services/agentService.ts`
- Import: `import { apiClient } from './api'`
- Import types from `../types.d.ts`
- Implement functions:
  - `export async function createAgent(name: string, systemPrompt: string, model?: string)`:
    - POST to `/create_agent` with body: `{name, system_prompt: systemPrompt, model}`
    - Return response.data
  - `export async function listAgents()`:
    - GET from `/list_agents`
    - Return response.data.agents
  - `export async function commandAgent(agentId: string, command: string, taskSlug?: string)`:
    - POST to `/command_agent/${agentId}` with body: `{command, task_slug: taskSlug}`
    - Return response.data
  - `export async function getAgentStatus(agentId: string, tailCount = 10, offset = 0, verboseLogs = false)`:
    - GET from `/agent_status/${agentId}?tail_count=${tailCount}&offset=${offset}&verbose_logs=${verboseLogs}`
    - Return response.data
  - `export async function deleteAgent(agentId: string)`:
    - DELETE to `/agent/${agentId}`
    - Return response.data
  - `export async function interruptAgent(agentId: string)`:
    - POST to `/interrupt_agent/${agentId}`
    - Return response.data
  - `export async function getEvents(filters: GetEventsRequest)`:
    - GET from `/events` with query params
    - Return response.data.events
  - `export async function getAgentLogs(agentId: string, limit = 50, offset = 0, taskSlug?: string)`:
    - GET from `/agent_logs/${agentId}?limit=${limit}&offset=${offset}&task_slug=${taskSlug || ''}`
    - Return response.data.logs
- Add error handling: try/catch with error logging

### 12. Update Frontend Type Definitions

- Edit `apps/orchestrator_3_stream/frontend/src/types.d.ts`
- Add after ChatStreamMessage (around line 220):
  ```typescript
  // Agent Management Types
  export interface Agent {
    id: string
    name: string
    model: string
    system_prompt?: string
    working_dir?: string
    status: 'idle' | 'executing' | 'waiting' | 'blocked' | 'complete'
    session_id?: string
    input_tokens: number
    output_tokens: number
    total_cost: number
    archived: boolean
    metadata: Record<string, any>
    created_at: string
    updated_at: string
  }

  export interface CreateAgentRequest {
    name: string
    system_prompt: string
    model?: string
    allowed_tools?: string[]
    disallowed_tools?: string[]
  }

  export interface CommandAgentRequest {
    command: string
    task_slug?: string
  }

  export interface AgentStatusResponse {
    status: string
    agent: Agent
    recent_logs: EventLogEntry[]
  }

  export interface EventLogEntry {
    id: string
    agent_id: string
    task_slug?: string
    entry_index?: number
    event_category: 'hook' | 'response'
    event_type: string
    content?: string
    payload: Record<string, any>
    summary?: string
    timestamp: string
  }

  export interface GetEventsRequest {
    agent_id?: string
    task_slug?: string
    limit?: number
    offset?: number
  }

  export interface GetEventsResponse {
    status: string
    events: EventLogEntry[]
    count: number
  }
  ```

### 13. Update Orchestrator Store with Agent Events

- Edit `apps/orchestrator_3_stream/frontend/src/stores/orchestratorStore.ts`
- Add imports: `import * as agentService from '../services/agentService'`
- Add state after chatMessages (around line 140):
  ```typescript
  const agents = ref<Agent[]>([])
  const agentEvents = ref<EventLogEntry[]>([])
  const selectedAgentId = ref<string | null>(null)
  ```
- Add actions:
  - `async function loadAgents()`:
    - Call `agents.value = await agentService.listAgents()`
    - Handle errors
  - `async function loadAgentEvents(agentId?: string, limit = 50, offset = 0)`:
    - Call `events = await agentService.getEvents({agent_id: agentId, limit, offset})`
    - Append to `agentEvents.value`
    - Handle errors
  - `function addAgentEvent(event: EventLogEntry)`:
    - Push to `agentEvents.value`
    - Trigger auto-scroll if EventStream has auto-follow enabled
- Modify `handleWebSocketMessage()` to route `agent_log` messages:
  ```typescript
  case 'agent_log':
    addAgentEvent(data.log)
    break
  case 'agent_created':
    agents.value.push(data.agent)
    break
  case 'agent_updated':
    const idx = agents.value.findIndex(a => a.id === data.agent_id)
    if (idx >= 0) agents.value[idx] = data.agent
    break
  case 'agent_deleted':
    agents.value = agents.value.filter(a => a.id !== data.agent_id)
    break
  case 'agent_status_changed':
    const agent = agents.value.find(a => a.id === data.agent_id)
    if (agent) agent.status = data.new_status
    break
  ```
- Update `initialize()` to load agents: `await loadAgents()`
- Export new state and actions: `return { ..., agents, agentEvents, selectedAgentId, loadAgents, loadAgentEvents, addAgentEvent }`

### 14. Update EventStream Component with Real Data

- Edit `apps/orchestrator_3_stream/frontend/src/components/EventStream.vue`
- Replace test data with store data:
  - `const store = useOrchestratorStore()`
  - `const events = computed(() => store.agentEvents.map((e, idx) => ({ ...e, lineNumber: idx + 1 })))`
- Load events on mount: `onMounted(() => { store.loadAgentEvents() })`
- Add computed properties for filtering:
  - `const filteredEvents = computed(() => { /* filter by level, search term */ })`
- Add watch for new events to trigger auto-scroll:
  ```typescript
  watch(() => store.agentEvents.length, () => {
    if (autoFollowEnabled.value) {
      nextTick(() => scrollToBottom())
    }
  })
  ```
- Keep existing UI (tabs, filters, search, auto-follow toggle)
- Map EventLogEntry to EventStreamEntry format:
  ```typescript
  const mappedEvents = computed(() => store.agentEvents.map((e, idx) => ({
    id: e.id,
    lineNumber: idx + 1,
    level: inferLevelFromEventType(e.event_type), // 'INFO', 'DEBUG', etc.
    agentId: e.agent_id,
    agentName: getAgentName(e.agent_id), // lookup from store.agents
    content: e.summary || e.content || `${e.event_type} event`,
    tokens: e.payload?.tokens,
    timestamp: e.timestamp,
    eventType: e.event_type,
    eventCategory: e.event_category
  })))
  ```

### 15. Update AgentList Component with Real Data

- Edit `apps/orchestrator_3_stream/frontend/src/components/AgentList.vue`
- Replace test data with store data:
  - `const store = useOrchestratorStore()`
  - `const agents = computed(() => store.agents)`
- Load agents on mount: `onMounted(() => { store.loadAgents() })`
- Display real agent data:
  - Name, status badge, model
  - Token counts: `${agent.input_tokens + agent.output_tokens} tokens`
  - Cost: `$${agent.total_cost.toFixed(4)}`
- Add click handler to filter events by agent:
  ```typescript
  function selectAgent(agentId: string) {
    store.selectedAgentId = agentId
    store.loadAgentEvents(agentId)
  }
  ```
- Update styling: status badge colors (idle=gray, executing=blue, complete=green, blocked=red)

### 16. Test Backend Agent Management

- Start backend: `cd apps/orchestrator_3_stream && ./start_be.sh`
- Test sequence using curl or Postman:
  1. Create agent: `curl -X POST http://127.0.0.1:9403/create_agent -H "Content-Type: application/json" -d '{"name": "tester", "system_prompt": "You are a test agent. Respond concisely."}'`
  2. Verify response has agent_id and session_id
  3. List agents: `curl http://127.0.0.1:9403/list_agents`
  4. Verify tester agent appears
  5. Send command: `curl -X POST http://127.0.0.1:9403/command_agent/<agent_id> -H "Content-Type: application/json" -d '{"command": "Say hello"}'`
  6. Check agent status: `curl http://127.0.0.1:9403/agent_status/<agent_id>?tail_count=5`
  7. Verify recent_logs array has events
  8. Get events: `curl http://127.0.0.1:9403/events?agent_id=<agent_id>&limit=10`
  9. Delete agent: `curl -X DELETE http://127.0.0.1:9403/agent/<agent_id>`
  10. List agents again: verify tester is gone (archived)
- Check database:
  - `SELECT * FROM agents WHERE name='tester'` - verify archived=true
  - `SELECT * FROM agent_logs WHERE agent_id='<agent_id>' ORDER BY entry_index` - verify events logged
  - `SELECT * FROM prompts WHERE agent_id='<agent_id>'` - verify prompts tracked
- Check backend logs for all operations

### 17. Test End-to-End with Frontend

- Start backend: `cd apps/orchestrator_3_stream && ./start_be.sh`
- Start frontend: `cd apps/orchestrator_3_stream && ./start_fe.sh`
- Open browser to `http://127.0.0.1:5175`
- Test orchestrator interaction:
  1. Send message to orchestrator: "Create a test agent named 'builder' for Python development"
  2. Verify orchestrator responds with agent creation confirmation
  3. Check AgentList sidebar: verify 'builder' agent appears with idle status
  4. Check EventStream: verify agent creation events appear
  5. Send message: "Tell builder to list the files in the current directory"
  6. Verify orchestrator dispatches command to builder
  7. Check EventStream: verify builder execution events stream in real-time
  8. Check AgentList: verify builder status changes (idle → executing → idle)
  9. Send message: "What is builder's status?"
  10. Verify orchestrator reports builder's recent activity
  11. Send message: "Delete builder"
  12. Verify agent disappears from AgentList
- Check console logs for WebSocket messages
- Check Network tab for HTTP requests and WebSocket frames
- Verify no JavaScript errors in console

### 18. Test Hook Event Capture

- With backend running, create a test agent
- Send command that uses multiple tools (e.g., "Read file README.md and summarize it")
- Monitor backend logs for hook events:
  - PreToolUse: Before Read tool executes
  - PostToolUse: After Read tool completes
  - UserPromptSubmit: When command submitted
  - Stop: When agent finishes
- Query database: `SELECT event_type, entry_index, summary FROM agent_logs WHERE agent_id='<agent_id>' ORDER BY entry_index`
- Verify:
  - All hook types captured
  - Entry indexes are sequential (0, 1, 2, 3...)
  - Summaries generated (may be NULL initially, then populated by background task)
- Check WebSocket in browser DevTools: verify agent_log messages arriving in real-time

### 19. Test Session Continuity

- Create test agent via API or orchestrator
- Send first command: "Remember this number: 42"
- Verify agent responds
- Query database for session_id: `SELECT session_id FROM agents WHERE name='test-agent'`
- Send second command: "What number did I tell you to remember?"
- Verify agent responds with "42" (proves session continuity)
- Query database again: verify session_id updated (sessions evolve with each interaction)
- Check agent_logs: verify both commands logged with same agent_id, different task_slugs

### 20. Test Interrupt Functionality

- Create test agent
- Send long-running command: "Count from 1 to 1000, showing each number"
- While executing, call interrupt endpoint: `curl -X POST http://127.0.0.1:9403/interrupt_agent/<agent_id>`
- Verify agent stops mid-execution
- Check database: verify agent status updated to 'blocked'
- Check AgentList in UI: verify status badge shows blocked
- Send new command to same agent: verify it can resume execution

### 21. Validate Database Schema Integrity

- Run queries to verify all relationships:
  - `SELECT COUNT(*) FROM agents` - verify agents created
  - `SELECT COUNT(*) FROM agent_logs` - verify events logged
  - `SELECT COUNT(*) FROM prompts` - verify prompts tracked
  - `SELECT a.name, COUNT(al.id) as event_count FROM agents a LEFT JOIN agent_logs al ON a.id = al.agent_id GROUP BY a.name` - verify agent-to-logs relationship
- Verify foreign key cascades:
  - Delete agent: verify agent_logs and prompts for that agent also deleted (if using CASCADE) or handle manually
- Verify indexes exist:
  - `\d agent_logs` in psql - check indexes on agent_id, task_slug, entry_index, timestamp
- Verify constraints:
  - Try to create agent with duplicate name: should fail with unique constraint violation

## Testing Strategy

**CRITICAL: NO MOCKING - Use Real Integrations**

This project follows a strict no-mocking policy:
- ✅ Use **real database connections** (PostgreSQL via asyncpg)
- ✅ Use **real Claude Agent SDK agents** (actual API calls)
- ❌ **NO mocks, stubs, or fakes** for database or Claude SDK
- ✅ Tests must be **ephemeral** - database starts and ends in the same state
- ✅ **Every module must have a corresponding test file**

### Test Organization

```
apps/orchestrator_3_stream/backend/
├── modules/
│   ├── agent_manager.py
│   ├── hooks.py
│   ├── event_summarizer.py
│   └── database.py (extended)
└── tests/
    ├── test_agent_manager.py          # NEW - Agent lifecycle tests
    ├── test_hooks.py                   # NEW - Hook event capture tests
    ├── test_event_summarizer.py        # NEW - AI summarization tests
    ├── test_database_agents.py         # NEW - Agent CRUD tests
    ├── test_database_events.py         # NEW - Event logging tests
    ├── test_agent_endpoints.py         # NEW - HTTP endpoint tests
    ├── test_event_streaming.py         # NEW - WebSocket streaming tests
    └── conftest.py                     # Extend with agent fixtures
```

### Ephemeral Testing Pattern

**Setup-Execute-Teardown for Agent Tests:**

```python
@pytest.fixture(scope="function")
async def test_agent():
    """Create test agent and cleanup after test"""
    await init_pool()

    # Create test orchestrator (needed for agent creation)
    orch_id = await get_or_create_orchestrator("Test Orch", "/tmp")

    # Create test agent
    agent_id = await create_agent(
        name=f"test-agent-{uuid4()}",
        model="claude-sonnet-4-5-20250929",
        system_prompt="You are a test agent. Respond concisely.",
        working_dir="/tmp",
        metadata={}
    )

    yield agent_id

    # Cleanup: Delete agent (cascades to agent_logs, prompts)
    await delete_agent(agent_id)
    await delete_orchestrator(orch_id)
    await close_pool()

async def test_command_agent_with_real_sdk(test_agent):
    """Test agent execution with REAL Claude SDK"""
    agent_id = test_agent

    # Execute command (makes actual Claude API call)
    agent_manager = AgentManager(None, ws_manager, logger, config)
    result = await agent_manager.command_agent(
        agent_id,
        command="Say hello in one word"
    )

    # Verify execution
    assert result["ok"] is True
    assert "task_slug" in result

    # Verify events logged to database
    logs = await get_agent_logs(agent_id, limit=10)
    assert len(logs) > 0

    # Verify at least one hook event captured
    hook_events = [l for l in logs if l["event_category"] == "hook"]
    assert len(hook_events) > 0

    # Cleanup happens in fixture
```

### Unit Tests (Real Database + Real SDK)

**Backend Module: test_agent_manager.py**
- ✅ `test_create_agent_with_real_database()` - Create agent, verify DB record, cleanup
- ✅ `test_list_agents_excludes_archived()` - Create 2 agents, archive 1, verify list shows only active
- ✅ `test_command_agent_with_real_sdk()` - Send simple command, verify execution and logging
- ✅ `test_agent_session_continuity()` - Send 2 commands, verify session_id maintained
- ✅ `test_interrupt_agent()` - Start long task, interrupt, verify stopped
- ✅ `test_check_agent_status_returns_logs()` - Verify status includes recent events
- ✅ `test_delete_agent_soft_delete()` - Delete agent, verify archived=true
- ❌ NO database or SDK mocks

**Backend Module: test_hooks.py**
- ✅ `test_pre_tool_hook_logs_to_database()` - Trigger PreToolUse, verify DB insertion
- ✅ `test_post_tool_hook_captures_result()` - Trigger PostToolUse, verify result logged
- ✅ `test_pre_compact_hook_resets_tokens()` - Trigger PreCompact, verify tokens reset to 0
- ✅ `test_hooks_broadcast_websocket_events()` - Mock WebSocket manager, verify broadcast calls
- ✅ `test_entry_counter_sequential_ordering()` - Multiple hooks, verify entry_index increments
- ✅ `test_async_summarization_spawned()` - Verify asyncio.create_task called for summarization

**Backend Module: test_event_summarizer.py**
- ✅ `test_summarize_event_with_real_sdk()` - Generate summary with Claude Haiku
- ✅ `test_fallback_summary_on_error()` - Fail AI call, verify template fallback
- ✅ `test_summary_under_100_chars()` - Verify length constraint

**Backend Module: test_database_agents.py**
- ✅ `test_create_agent_inserts_record()` - Create, verify fields match
- ✅ `test_get_agent_by_name()` - Create agent, fetch by name
- ✅ `test_update_agent_session()` - Update session_id, verify change
- ✅ `test_update_agent_costs_incremental()` - Update costs twice, verify cumulative
- ✅ `test_delete_agent_sets_archived()` - Delete, verify archived=true

**Backend Module: test_database_events.py**
- ✅ `test_insert_hook_event()` - Insert hook event, verify fields
- ✅ `test_insert_message_block()` - Insert response block, verify content
- ✅ `test_get_tail_summaries_returns_chronological()` - Insert 5 events, verify order
- ✅ `test_update_log_summary()` - Insert event, update summary, verify

**Backend Module: test_agent_endpoints.py**
- ✅ `test_create_agent_endpoint()` - POST to /create_agent, verify response
- ✅ `test_list_agents_endpoint()` - GET /list_agents, verify array returned
- ✅ `test_command_agent_endpoint()` - POST to /command_agent/{id}, verify execution
- ✅ `test_get_events_endpoint_with_filters()` - GET /events with agent_id filter
- ✅ `test_agent_status_endpoint()` - GET /agent_status/{id}, verify logs included

**Backend Module: test_event_streaming.py**
- ✅ `test_websocket_broadcasts_agent_log()` - Create agent, send command, capture WebSocket messages
- ✅ `test_event_stream_chronological_order()` - Verify events arrive in order
- ✅ `test_multiple_agents_events_separate()` - Create 2 agents, verify events tagged correctly

### Integration Tests (Real Everything)

**Backend Integration:**
- ✅ Test full agent lifecycle: Create → Command → Monitor → Delete
  - Create via POST /create_agent
  - Command via POST /command_agent/{id}
  - Monitor via GET /events and WebSocket
  - Delete via DELETE /agent/{id}
  - Verify database cleaned up
- ✅ Test orchestrator commanding agents:
  - Send orchestrator message: "Create an agent called builder"
  - Verify agent created via database query
  - Send: "Tell builder to list files"
  - Verify command dispatched and executed
- ✅ Test event streaming end-to-end:
  - Connect WebSocket client
  - Create agent and send command via API
  - Verify agent_log messages arrive in real-time
  - Compare WebSocket events with database events
  - Verify timestamps and entry_index match

**Frontend Integration:**
- ✅ Test Pinia store with real backend:
  - Call loadAgents(), verify agents array populated
  - Call loadAgentEvents(), verify events array populated
  - Listen to WebSocket, verify addAgentEvent() called
- ✅ Test EventStream component displays real data:
  - Mount component with store
  - Trigger backend events
  - Verify DOM updates with new event entries
- ✅ Test AgentList component updates on status changes:
  - Create agent via orchestrator
  - Verify agent appears in list
  - Send command to agent
  - Verify status badge changes (idle → executing → idle)

### End-to-End Tests (Full Stack)

**Playwright E2E Tests:**
- ✅ Test orchestrator creates and commands agent:
  - Navigate to frontend
  - Type in chat: "Create an agent called builder for Python development"
  - Wait for orchestrator response
  - Verify builder appears in AgentList sidebar
  - Type: "Tell builder to list the current directory files"
  - Wait for response
  - Verify events appear in EventStream component
  - Verify builder status updates in AgentList
  - Query database to confirm all events persisted
  - Cleanup: Delete test agent from database
- ✅ Test event streaming displays in real-time:
  - Create agent via API
  - Send long command via API
  - Watch EventStream component
  - Verify events appear as they're generated (not after completion)
  - Verify auto-scroll works
- ✅ Test session persistence across page refresh:
  - Create agent and send command
  - Refresh browser
  - Verify AgentList still shows agent
  - Verify EventStream loads previous events
  - Send new command
  - Verify session continuity

### Test Execution Commands

```bash
# Run all backend tests with real database and Claude SDK
cd apps/orchestrator_3_stream/backend
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/test_agent_manager.py -v

# Run tests matching pattern
uv run pytest tests/ -v -k "agent"

# Run with coverage
uv run pytest tests/ --cov=modules --cov-report=html

# Run E2E tests (requires backend + frontend running)
cd apps/orchestrator_3_stream
./start_be.sh  # Terminal 1
./start_fe.sh  # Terminal 2
uv run pytest tests/test_e2e_agents.py  # Terminal 3
```

## Acceptance Criteria

**Agent Management:**
- [ ] Orchestrator can create agents via "Create an agent named X" command
- [ ] Orchestrator can list agents via "List all agents" command
- [ ] Orchestrator can command agents via "Tell X to do Y" command
- [ ] Orchestrator can check agent status via "How is X doing?" command
- [ ] Orchestrator can delete agents via "Delete agent X" command
- [ ] Orchestrator can interrupt agents via "Stop agent X" command
- [ ] All agent operations reflected in database (agents, agent_logs, prompts tables)

**Event Streaming:**
- [ ] Agent execution events stream to EventStream component in real-time
- [ ] EventStream displays all event types (hooks + responses)
- [ ] EventStream shows entry numbers, timestamps, agent names
- [ ] EventStream filtering works (by level: DEBUG, INFO, WARNING, ERROR)
- [ ] EventStream search works (by content text)
- [ ] Auto-scroll follows new events when enabled
- [ ] HTTP GET /events endpoint returns event history with filtering

**Agent Lifecycle:**
- [ ] Created agents have unique names and session IDs
- [ ] Commands maintain session continuity (agents remember context)
- [ ] Agent status updates correctly (idle → executing → idle/blocked/complete)
- [ ] Token counts and costs track accurately per agent
- [ ] Deleted agents are soft-deleted (archived=true)
- [ ] Interrupted agents stop immediately

**Frontend Integration:**
- [ ] AgentList sidebar displays all active agents
- [ ] AgentList shows agent status with color-coded badges
- [ ] AgentList updates in real-time via WebSocket
- [ ] EventStream shows real agent logs (not test data)
- [ ] Clicking agent in AgentList filters EventStream to that agent
- [ ] All components handle WebSocket disconnection gracefully

**System Integration:**
- [ ] Orchestrator tools registered correctly (8 tools available)
- [ ] Hooks capture all execution events (6 hook types)
- [ ] Background summarization generates AI summaries for events
- [ ] WebSocket broadcasts reach all connected clients
- [ ] Database maintains referential integrity (cascading deletes)

## Validation Commands

Execute these commands to validate the task is complete:

### Backend Validation

- `cd apps/orchestrator_3_stream/backend && uv run python -m py_compile modules/*.py` - Verify all Python files compile
- `cd apps/orchestrator_3_stream/backend && uv run python -c "from modules.agent_manager import AgentManager; print('AgentManager imports successfully')"` - Test import
- `cd apps/orchestrator_3_stream/backend && uv run python -c "from modules.hooks import create_pre_tool_hook; print('Hooks module imports successfully')"` - Test import
- `cd apps/orchestrator_3_stream/backend && uv run pytest tests/test_agent_manager.py -v` - Run agent manager tests
- `cd apps/orchestrator_3_stream/backend && uv run pytest tests/test_hooks.py -v` - Run hooks tests
- `cd apps/orchestrator_3_stream/backend && uv run pytest tests/ -v` - Run all backend tests

### Database Validation

- `psql $DATABASE_URL -c "SELECT COUNT(*) FROM agents WHERE archived=false"` - Count active agents
- `psql $DATABASE_URL -c "SELECT COUNT(*) FROM agent_logs"` - Count event logs
- `psql $DATABASE_URL -c "SELECT event_type, COUNT(*) FROM agent_logs GROUP BY event_type"` - Event type distribution
- `psql $DATABASE_URL -c "SELECT a.name, COUNT(al.id) as events FROM agents a LEFT JOIN agent_logs al ON a.id = al.agent_id GROUP BY a.name"` - Events per agent

### API Validation

- `./start_be.sh` - Start backend (check for errors)
- `curl http://127.0.0.1:9403/health` - Health check
- `curl http://127.0.0.1:9403/list_agents` - List agents endpoint
- `curl -X POST http://127.0.0.1:9403/create_agent -H "Content-Type: application/json" -d '{"name": "test-validation", "system_prompt": "Test agent"}'` - Create agent
- `curl http://127.0.0.1:9403/events?limit=10` - Get recent events

### Frontend Validation

- `cd apps/orchestrator_3_stream/frontend && npm run build` - Build frontend
- `./start_fe.sh` - Start frontend (check console for errors)
- Open `http://127.0.0.1:5175` - Verify UI loads
- Check browser console: no JavaScript errors
- Check Network tab: WebSocket connection established
- Send orchestrator message: "Create a test agent" - Verify agent appears in sidebar
- Check EventStream: Verify events appear

### End-to-End Validation

**Manual Test Flow:**
1. Start backend and frontend
2. Send orchestrator message: "Create an agent named builder with a system prompt: You are a Python developer. Help with code tasks."
3. Verify: Builder appears in AgentList with idle status
4. Verify: Agent creation events appear in EventStream
5. Send: "Tell builder to list the files in the current directory"
6. Verify: Builder status changes to executing
7. Verify: Command execution events stream to EventStream in real-time
8. Verify: Builder completes and status returns to idle
9. Send: "What is builder's status?"
10. Verify: Orchestrator reports builder's recent activity
11. Send: "Delete builder"
12. Verify: Builder disappears from AgentList
13. Database check: `psql $DATABASE_URL -c "SELECT * FROM agents WHERE name='builder'"` - Should show archived=true

**Automated Playwright Validation:**
```bash
# Use playwright-validator agent
agent-playwright-validator "
Navigate to http://127.0.0.1:5175 and test the multi-agent orchestration system.
1. Send orchestrator message: 'Create an agent named test-agent'
2. Verify test-agent appears in the AgentList sidebar
3. Send: 'Tell test-agent to say hello'
4. Verify EventStream shows agent execution events
5. Verify test-agent status updates during execution
6. Take screenshots of AgentList and EventStream
7. Send: 'Delete test-agent'
8. Verify test-agent removed from AgentList
Report success if all steps complete without errors.
"
```

## Notes

### Architecture Decision: Background Thread Execution

Command-level agents run in asyncio tasks (not OS threads) but the pattern is still "background" from the orchestrator's perspective:
- Orchestrator issues command → Returns immediately -> make sure this is clear in the response from the tool call.
- Agent executes asynchronously in background
- Events stream via WebSocket as they occur
- Orchestrator can issue additional commands while agent works

This enables true multi-agent orchestration where one orchestrator manages multiple concurrently-executing agents.

### Hook Integration with WebSocket

Each hook function receives `ws_manager` as a parameter and broadcasts events in real-time:
```python
# In hook implementation
await ws_manager.broadcast_agent_log({
    "id": str(log_id),
    "agent_id": str(agent_id),
    "event_type": event_type,
    "entry_index": entry_index,
    "payload": payload,
    "timestamp": datetime.utcnow().isoformat()
})
```

This is NEW compared to orchestrator_1_term (which only logged to DB). orchestrator_3_stream adds real-time streaming on top of database persistence.

### Event Summarization Performance

AI summarization runs in background (fire-and-forget) to avoid blocking agent execution:
- Event inserted to DB immediately (synchronous)
- Summarization task spawned: `asyncio.create_task(_summarize_and_update(...))`
- Summary written to DB when complete (asynchronous)
- Frontend shows summaries when available, falls back to event_type if null

This keeps agent execution fast while still providing human-readable event summaries.

### Session Continuity Critical Path

Session ID must be:
1. Captured from ResultMessage after agent initialization
2. Stored in database (agents.session_id column)
3. Passed to ClaudeAgentOptions with `resume=session_id` on next command
4. Updated in database after each command (session evolves)

Breaking this chain loses conversation context and agents won't remember previous interactions.

### Tool Registration via MCP Server

Tools are registered using the Claude SDK's MCP (Model Context Protocol) pattern:
```python
from claude_code_sdk.tools import tool, create_sdk_mcp_server

@tool("create_agent", "Create a new agent", {"name": str, "system_prompt": str})
async def create_agent_tool(args):
    return {"content": [{"type": "text", "text": "Agent created"}]}

mcp_server = create_sdk_mcp_server(name="mgmt", version="1.0.0", tools=[create_agent_tool, ...])

options = ClaudeAgentOptions(mcp_servers={"mgmt": mcp_server}, allowed_tools=["mcp__mgmt__create_agent", ...])
```

Tool names are prefixed with `mcp__<server_name>__<tool_name>` in allowed_tools list.

### Database Indexes for Performance

The agent_logs table will grow large. Critical indexes:
- `(agent_id, task_slug, entry_index)` - For tail reading specific task events
- `(agent_id, timestamp DESC)` - For recent events across all tasks
- `(event_category, event_type)` - For filtering by event type
- `(task_slug)` - For task-level queries

Without these, queries will slow down significantly after 10,000+ events.

### Frontend Event Stream Performance

For large event lists (1000+ events), consider:
- Virtual scrolling (only render visible events)
- Pagination (load events in batches)
- Event expiry (only keep last N events in memory)
- Database-side filtering (don't load all events to frontend)

Current implementation loads all events - fine for demo, may need optimization for production.

### Cost Tracking and Token Limits

Each agent tracks tokens independently:
- `input_tokens` - Cumulative input across all commands
- `output_tokens` - Cumulative output across all commands
- `total_cost` - Cumulative USD cost

Context window limit (200K tokens for Sonnet 4):
- Monitor via `input_tokens + output_tokens`
- When agent reaches 80%, suggest `/compact` or reset
- PreCompact hook resets counters to show new compressed size

### Error Handling Strategy

**Database Errors:**
- Log full error with stack trace
- Return HTTP 500 with generic message to user
- Don't expose SQL errors to frontend

**Claude SDK Errors:**
- Catch API errors (rate limits, invalid requests)
- Log full error details
- Return user-friendly message
- For rate limits: implement exponential backoff

**WebSocket Errors:**
- Handle disconnections gracefully
- Implement automatic reconnection with backoff
- Show connection status in UI
- Queue events during disconnect, replay on reconnect

**Agent Execution Errors:**
- Capture in agent_logs with is_error flag
- Display error events in EventStream with red badge
- Don't crash orchestrator on agent failures
- Allow agent recovery via status check and restart

### Testing Strategy Notes

**Why No Mocking:**
- Real database tests catch SQL errors, constraint violations, migration issues
- Real Claude SDK tests catch API changes, invalid parameters, auth issues
- Mocks create false confidence ("works in test, fails in production")

**Managing Test Costs:**
- Use simple prompts: "Say hello" instead of "Write a 1000-line essay"
- Use Haiku for tests where model doesn't matter
- Limit test runs during development (only full suite before commits)
- Consider test database separate from production

**Ephemeral Test Data:**
- Every test creates unique agent names: `f"test-agent-{uuid4()}"`
- Fixtures handle cleanup: yield → delete test data
- Database must be identical before/after tests
- Parallel test support via unique identifiers

### Future Enhancements (Out of Scope)

- Multi-orchestrator support (currently singleton)
- Agent-to-agent direct communication (currently via orchestrator)
- Agent capability negotiation (currently static tool sets)
- Distributed agent execution (currently single backend)
- Agent memory/RAG integration (currently session-based only)
- Advanced event analytics (currently basic filtering)
- Agent performance metrics (latency, success rate, etc.)
- Cost budgets and alerts per agent
- Agent templates and presets
- Agent cloning and forking

### Dependencies

**New Backend Dependencies:**
- `claude-agent-sdk` - Already included (for Claude SDK)
- No additional packages needed

**New Frontend Dependencies:**
- No additional packages needed (axios already used)

**Development Dependencies:**
- `pytest` - For backend testing (already included)
- `pytest-asyncio` - For async test support (likely already included)

If new libraries needed:
- Backend: `cd apps/orchestrator_3_stream/backend && uv add <package>`
- Frontend: `cd apps/orchestrator_3_stream/frontend && npm install <package>`

### Reference Implementations

**Primary References (orchestrator_1_term):**
- `modules/orchestrator_agent.py` - Tool registration pattern, execution loop
- `modules/agent_manager.py` - Complete agent lifecycle management
- `modules/hooks.py` - Hook factories and event capture
- `modules/database/agents.py` - Agent CRUD operations
- `modules/database/agent_logs.py` - Event logging operations

**Copy these files as starting points:**
1. Copy `orchestrator_1_term/modules/agent_manager.py` → `orchestrator_3_stream/backend/modules/agent_manager.py`
2. Copy `orchestrator_1_term/modules/hooks.py` → `orchestrator_3_stream/backend/modules/hooks.py`
3. Adapt both files to add WebSocket integration
4. Copy database functions from `orchestrator_1_term/modules/database/*.py` to `orchestrator_3_stream/backend/modules/database.py`

**Key Adaptations for orchestrator_3_stream:**
- Add `ws_manager` parameter to all relevant classes/functions
- Add WebSocket broadcasts in hooks (NEW)
- Add HTTP endpoints for agent operations (NEW)
- Integrate with existing OrchestratorService (NEW)

### Prompt Templates

**Orchestrator System Prompt** (backend/prompts/orchestrator_agent_system_prompt.md):
- Already exists, likely complete
- Verify it documents all 8 tools with current parameters
- Update if tool signatures changed

**Managed Agent System Prompt Template** (backend/prompts/managed_agent_system_prompt_template.md):
- Copy from orchestrator_1_term
- Variables: {agent_name}, {working_dir}, {model}, {session_id}
- Used when creating agents to customize their behavior

### Configuration Variables

Add to `.env` and `.env.sample`:
```bash
# Agent Configuration
DEFAULT_AGENT_MODEL=claude-sonnet-4-5-20250929
AGENT_SYSTEM_PROMPT_TEMPLATE_PATH=./backend/prompts/managed_agent_system_prompt_template.md
MAX_AGENT_TURNS=500

# Frontend Configuration (already exists)
VITE_API_BASE_URL=http://127.0.0.1:9403
VITE_WS_URL=ws://127.0.0.1:9403/ws
```
