# Plan: Implement Orchestrator + User Chat End-to-End in orchestrator_3_stream

## Task Description

Implement complete orchestrator chat functionality in `apps/orchestrator_3_stream` by creating HTTP endpoints (`/load_chat`, `/send_chat`), WebSocket streaming for orchestrator responses, and integrating the frontend with the backend using the proven three-phase logging pattern from `apps/orchestrator_1_term` as a reference.

## Objective

Enable users to interact with the orchestrator agent through the web interface by:
- Loading existing chat history from the database
- Sending messages to the orchestrator agent
- Receiving streamed responses in real-time via WebSocket
- Persisting all messages to the `orchestrator_chat` table
- Maintaining session continuity across interactions
- Supporting CLI parameters `--session` and `--cwd` for session resumption and working directory configuration

## Problem Statement

The `apps/orchestrator_3_stream` application currently has a complete UI framework with a 3-column layout (AgentList, EventStream, OrchestratorChat) but the chat functionality is entirely disconnected from the backend. The frontend uses simulated responses with test data, and the backend lacks:
- HTTP endpoints for chat operations
- Database integration for message persistence
- Orchestrator agent execution logic
- WebSocket streaming for real-time responses

Meanwhile, `apps/orchestrator_1_term` has a fully functional orchestrator chat system using a three-phase logging pattern (pre-execution, execution, post-execution) that maintains perfect message history in the `orchestrator_chat` table.

## Solution Approach

Adapt the proven pattern from `orchestrator_1_term` to a web streaming architecture:

**Three-Phase Logging Pattern:**
1. **Pre-Execution**: Log user message to `orchestrator_chat` table immediately upon receipt
2. **Execution**: Execute orchestrator agent using Claude SDK, streaming responses to frontend via WebSocket
3. **Post-Execution**: Log complete orchestrator response to `orchestrator_chat` table

**Architecture Layers:**
```
Frontend (Vue 3 + Pinia)
    ↓ chatService.ts (HTTP + WebSocket)
Backend FastAPI
    ↓ orchestrator_service.py (business logic)
    ↓ database.py (connection pool + queries)
PostgreSQL (orchestrator_chat table)
```

**Key Patterns to Follow:**
- Use `insert_chat_message()` with directional types (sender_type, receiver_type)
- Derive turn count from total message count (no manual tracking)
- Maintain session continuity via session_id in `orchestrator_agents` table
- Stream response chunks via WebSocket while accumulating for final database save
- Update token counts and costs after each interaction

## Relevant Files

Use these files to complete the task:

### Backend Files - To Modify

- **apps/orchestrator_3_stream/backend/main.py**
  - Why: Add HTTP POST `/load_chat` and `/send_chat` endpoints; enhance WebSocket handler to process chat messages
  - Changes: Add route handlers, integrate orchestrator_service, parse incoming WebSocket messages

- **apps/orchestrator_3_stream/backend/modules/websocket_manager.py**
  - Why: Add specialized methods for streaming chat responses chunk-by-chunk
  - Changes: Add `broadcast_chat_stream()` for response chunks, `set_typing_indicator()` for status

- **apps/orchestrator_3_stream/backend/modules/config.py**
  - Why: Add orchestrator-specific configuration (system prompt, model, working directory)
  - Changes: Add `ORCHESTRATOR_AGENT_ID`, `ORCHESTRATOR_MODEL`, `ORCHESTRATOR_SYSTEM_PROMPT` env vars

- **apps/orchestrator_3_stream/backend/modules/logger.py**
  - Why: Add chat-specific logging method for consistent event tracking
  - Changes: Add `chat_event()` method for logging chat interactions

### Backend Files - New Files to Create

- **apps/orchestrator_3_stream/backend/modules/database.py**
  - Why: Create database connection pool and provide query functions for chat/agent operations
  - Functions needed:
    - `init_pool()` - Initialize asyncpg connection pool
    - `get_connection()` - Context manager for database queries
    - `close_pool()` - Cleanup connections on shutdown
    - `get_orchestrator()` - Fetch singleton orchestrator agent
    - `get_orchestrator_by_session(session_id)` - Validate and fetch orchestrator by session_id (for --session resumption)
    - `get_or_create_orchestrator(system_prompt, working_dir)` - Get existing or create new orchestrator singleton
    - `insert_chat_message()` - Log chat messages (copy from orchestrator_1_term)
    - `get_chat_history()` - Load past messages (copy from orchestrator_1_term)
    - `get_turn_count()` - Count messages for turn display (copy from orchestrator_1_term)
    - `update_orchestrator_session()` - Update session_id after interaction
    - `update_orchestrator_costs()` - Update token counts and costs

- **apps/orchestrator_3_stream/backend/modules/orchestrator_service.py**
  - Why: Implement orchestrator agent business logic - execution, streaming, database persistence
  - Methods needed:
    - `__init__(db, ws_manager, logger, session_id=None, working_dir=None)` - Dependency injection with optional session resumption
    - `load_chat_history(orchestrator_agent_id)` - Fetch and return chat messages
    - `process_user_message(message, orchestrator_agent_id)` - Execute agent, stream responses
    - `stream_orchestrator_response()` - Stream response chunks via WebSocket
    - `_create_claude_agent_options()` - Initialize Claude SDK client options with session resumption and working directory

### Frontend Files - To Modify

- **apps/orchestrator_3_stream/frontend/src/stores/orchestratorStore.ts**
  - Why: Replace simulated chat logic with real API calls using chatService
  - Changes:
    - Import chatService
    - Update `sendUserMessage()` to call `chatService.sendMessage()`
    - Update `connectWebSocket()` to use `chatService.connectWebSocket()`
    - Implement `handleWebSocketMessage()` to process incoming messages
    - Add `loadChatHistory()` action to fetch past messages
    - Remove `simulateOrchestratorResponse()` function

- **apps/orchestrator_3_stream/frontend/src/types.d.ts**
  - Why: Add type definitions for API request/response payloads and WebSocket messages
  - Changes: Add interfaces:
    - `LoadChatRequest` / `LoadChatResponse`
    - `SendChatRequest` / `SendChatResponse`
    - `ChatStreamMessage` (WebSocket payload structure)

- **apps/orchestrator_3_stream/frontend/src/components/OrchestratorChat.vue**
  - Why: Minor enhancements for streaming text display
  - Changes: Add logic to display incremental response chunks as they arrive (optional enhancement)

### Frontend Files - New Files to Create

- **apps/orchestrator_3_stream/frontend/src/services/chatService.ts**
  - Why: Centralize all HTTP and WebSocket communication for chat
  - Functions needed:
    - `loadChatHistory(orchestratorAgentId)` - HTTP GET to `/load_chat`
    - `sendMessage(message, orchestratorAgentId)` - HTTP POST to `/send_chat`
    - `connectWebSocket(url)` - Establish WebSocket connection
    - `onChatStream(callback)` - Register callback for streaming chunks
    - `disconnect()` - Cleanup WebSocket connection

- **apps/orchestrator_3_stream/frontend/src/services/api.ts**
  - Why: Configure Axios instance with base URL and interceptors
  - Configuration: Set baseURL from env vars, add error handling interceptors

### Reference Files (Read-Only - for understanding patterns)

- **apps/orchestrator_1_term/modules/orchestrator_agent.py**
  - Pattern: Three-phase logging (lines 807-825, 831-1010, 1085-1097)
  - Pattern: Agent execution loop with Claude SDK (lines 831-1010)
  - Pattern: Session and cost tracking (lines 1016-1049)

- **apps/orchestrator_1_term/modules/database/orchestrator_chat_db.py**
  - Functions to copy/adapt: `insert_chat_message()`, `get_chat_history()`, `get_turn_count()`
  - Pattern: Directional message types (sender_type, receiver_type)

- **apps/orchestrator_1_term/modules/database/orchestrator_agents_db.py**
  - Functions to adapt: `get_or_create_orchestrator()`, `update_orchestrator_session()`, `update_orchestrator_costs()`

- **apps/orchestrator_db/models.py**
  - Model to use: `OrchestratorChat` (lines 282-320) - already has proper UUID/datetime handling

## Implementation Phases

### Phase 1: Database Layer Foundation
Set up database connection pool and core query functions for chat operations. This unblocks all subsequent backend work.

**Goal**: Have working database.py module with connection pooling and all required query functions.

### Phase 2: Backend API & Service Layer
Implement HTTP endpoints, WebSocket handlers, and orchestrator service with Claude SDK integration.

**Goal**: Backend can receive messages, execute orchestrator agent, stream responses, and persist to database.

### Phase 3: Frontend Integration
Create chatService, update store with real API calls, connect WebSocket streaming, and update UI.

**Goal**: Frontend can load history, send messages, display streamed responses, and maintain conversation state.

## Step by Step Tasks

IMPORTANT: Execute every step in order, top to bottom.

### 1. Sync Database Models to Backend

**IMPORTANT**: Before creating database.py, sync the Pydantic models:

- Run `uv run python apps/orchestrator_db/sync_models.py` from project root
- This copies `apps/orchestrator_db/models.py` to `apps/orchestrator_3_stream/backend/modules/orch_database_models.py`
- Verify the sync completed successfully (should show ✅ confirmation)
- The models include: `OrchestratorAgent`, `Agent`, `OrchestratorChat`, `AgentLog`, `SystemLog`
- **Do NOT manually create or edit orch_database_models.py** - it's automatically generated

### 2. Create Database Module (Backend Foundation)

- Create `apps/orchestrator_3_stream/backend/modules/database.py`
- Import models: `from .orch_database_models import OrchestratorAgent, OrchestratorChat`
- Implement asyncpg connection pool initialization with `init_pool()` and `close_pool()`
- Implement `get_connection()` context manager for database queries
- Copy and adapt `insert_chat_message()` from `apps/orchestrator_1_term/modules/database/orchestrator_chat_db.py`
  - Use parameters: `orchestrator_agent_id`, `sender_type`, `receiver_type`, `message`, `agent_id`, `metadata`
  - Sender/receiver types: 'user', 'orchestrator', 'agent' (Literal types)
  - Returns message UUID
- Copy and adapt `get_chat_history()` from same source
  - Returns list of dicts with fields: `id`, `sender_type`, `receiver_type`, `message`, `agent_id`, `metadata`, `created_at`
- Copy and adapt `get_turn_count()` from same source
  - Returns count of all messages for orchestrator_agent_id
- Implement `get_orchestrator()` by copying from `apps/orchestrator_1_term/modules/database/orchestrator_agents_db.py`
- Implement `get_orchestrator_by_session(session_id)` from same source - validates session exists, returns orchestrator data
- Implement `get_or_create_orchestrator(system_prompt, working_dir)` from same source - gets or creates singleton orchestrator
- Implement `update_orchestrator_session()` from same source
- Implement `update_orchestrator_costs()` from same source
- Add proper error handling and logging to all database operations

### 3. Update Backend Configuration

- Edit `apps/orchestrator_3_stream/backend/modules/config.py`
- Add `ORCHESTRATOR_AGENT_ID` environment variable (with default value)
- Add `ORCHESTRATOR_MODEL` environment variable (default to configured model)
- Add `ORCHESTRATOR_SYSTEM_PROMPT` environment variable or load from file path
- Add `ORCHESTRATOR_WORKING_DIR` environment variable
- Add method `set_working_dir(path)` to dynamically override working directory
- Document these new env vars in `.env.sample`

### 4. Add CLI Argument Parsing to Backend

- Edit `apps/orchestrator_3_stream/backend/main.py`
- Import `argparse` for command-line argument parsing
- Add CLI argument parser before app creation:
  - `--session <session_id>` - Resume existing orchestrator session (optional)
  - `--cwd <path>` - Set working directory for orchestrator and agents (optional)
- Parse arguments and store in global variables or app.state
- Priority for working directory: `--cwd flag > current working directory (os.getcwd()) > .env default`
- If `--session` provided, validate it exists by calling `get_orchestrator_by_session(session_id)`
  - If invalid, log error and exit with error code
  - If valid, store session_id and orchestrator_data for service initialization
- If no `--session`, call `get_or_create_orchestrator()` to get/create singleton
- Log launch configuration showing:
  - Session status (new or resuming with session_id)
  - Working directory
  - Model configuration
  - If resuming: current token counts and cost
- Reference pattern from `apps/orchestrator_1_term/modules/cli_interface.py` lines 401-527

### 5. Enhance WebSocket Manager

- Edit `apps/orchestrator_3_stream/backend/modules/websocket_manager.py`
- Add `broadcast_chat_stream(orchestrator_agent_id: str, chunk: str, is_complete: bool = False)` method
  - Broadcast JSON message with type="chat_stream"
  - Include chunk text, is_complete flag, orchestrator_agent_id
- Add `set_typing_indicator(orchestrator_agent_id: str, is_typing: bool)` method
  - Broadcast JSON message with type="chat_typing"
  - Include is_typing boolean state
- Follow existing broadcast pattern from `broadcast_chat_message()` (line 156-158)

### 6. Enhance Logger Module

- Edit `apps/orchestrator_3_stream/backend/modules/logger.py`
- Add `chat_event(orchestrator_id: str, message: str, sender: str = "orchestrator")` method
- Format: `self.info(f"💬 Chat [{orchestrator_id}] {sender.upper()}: {message[:100]}")`
- Follow existing patterns from `agent_event()` and `websocket_event()` methods

### 7. Create Orchestrator Service Module

- Create `apps/orchestrator_3_stream/backend/modules/orchestrator_service.py`
- Implement `OrchestratorService` class with:
  - `__init__(db_manager, ws_manager, logger, session_id=None, working_dir=None)` - Store dependencies and session parameters
  - Store `self.session_id` for Claude SDK session resumption
  - Store `self.working_dir` for agent working directory configuration
  - `load_chat_history(orchestrator_agent_id)` - Call `db_manager.get_chat_history()` and return results
  - `process_user_message(user_message, orchestrator_agent_id)` - Main execution method:
    - Log user message to database (pre-execution phase)
    - Create/resume Claude SDK client with orchestrator's session_id
    - Send user message to agent
    - Iterate through response messages (async for loop)
    - Stream response chunks via `ws_manager.broadcast_chat_stream()`
    - Accumulate full response text
    - Log orchestrator response to database (post-execution phase)
    - Update session_id and costs in database
    - Return complete response
  - `_create_claude_agent_options()` - Configure Claude SDK client options
- Use patterns from `apps/orchestrator_1_term/modules/orchestrator_agent.py` `run_interaction()` method (lines 741-1116)
- Import Claude SDK types: `ClaudeSDKClient`, `ClaudeAgentOptions`, `AssistantMessage`, `TextBlock`, `ResultMessage`, etc.

### 8. Initialize Database in Main Application

- Edit `apps/orchestrator_3_stream/backend/main.py`
- Import database module: `from modules.database import init_pool, close_pool`
- Import orchestrator service: `from modules.orchestrator_service import OrchestratorService`
- In `lifespan()` function startup section (line 34-49):
  - Call `await init_pool()` with DATABASE_URL from config
  - Instantiate `OrchestratorService(db, ws_manager, logger, session_id=parsed_session_id, working_dir=parsed_working_dir)` and store in app.state
  - Pass session_id from CLI args (if --session was provided)
  - Pass working_dir from CLI args or config (priority: --cwd > os.getcwd() > .env default)
  - Log successful initialization
- In `lifespan()` function shutdown section (line 52-54):
  - Call `await close_pool()`
  - Log shutdown

### 9. Add HTTP Chat Endpoints

- Edit `apps/orchestrator_3_stream/backend/main.py` after health check endpoint (line 80)
- Add Pydantic models for request/response:
  - `LoadChatRequest(BaseModel)` with `orchestrator_agent_id: str, limit: int = 50`
  - `SendChatRequest(BaseModel)` with `message: str, orchestrator_agent_id: str`
- Add `@app.post("/load_chat")` endpoint:
  - Accept `LoadChatRequest` body
  - Call `orchestrator_service.load_chat_history()`
  - Return `{"status": "success", "messages": [...], "turn_count": N}`
  - Handle exceptions with HTTP 500 and error message
- Add `@app.post("/send_chat")` endpoint:
  - Accept `SendChatRequest` body
  - Call `orchestrator_service.process_user_message()`
  - Return `{"status": "success", "response": "...", "session_id": "..."}`
  - Handle exceptions with HTTP 500 and error message
- Log all HTTP requests using `logger.http_request()`

### 10. Enhance WebSocket Message Handler

- Edit `apps/orchestrator_3_stream/backend/main.py` WebSocket endpoint (lines 82-101)
- Parse incoming WebSocket messages as JSON
- Add message type routing:
  - If `message.type == "chat_message"`:
    - Extract user message and orchestrator_agent_id
    - Call `orchestrator_service.process_user_message()` asynchronously
    - Response streaming handled by service via ws_manager
  - Handle unknown message types gracefully
- Keep existing error handling for WebSocketDisconnect

### 11. Create Frontend API Configuration

- Create `apps/orchestrator_3_stream/frontend/src/services/api.ts`
- Import axios
- Create axios instance with:
  - `baseURL`: Load from import.meta.env (Vite environment variable)
  - Timeout: 30000ms
  - Headers: `Content-Type: application/json`
- Add response interceptor for error handling
- Export configured instance as `apiClient`

### 12. Create Frontend Chat Service

- Create `apps/orchestrator_3_stream/frontend/src/services/chatService.ts`
- Import `apiClient` from `./api.ts`
- Implement `loadChatHistory(orchestratorAgentId: string)`:
  - POST to `/load_chat` with orchestrator_agent_id
  - Return response data (messages array + turn_count)
- Implement `sendMessage(message: string, orchestratorAgentId: string)`:
  - POST to `/send_chat` with message and orchestrator_agent_id
  - Return response data
- Implement `connectWebSocket(url: string, callbacks: { onChatStream, onTyping, onError })`:
  - Create WebSocket connection
  - Parse incoming JSON messages
  - Route by message.type:
    - `chat_stream`: Call `callbacks.onChatStream(chunk, is_complete)`
    - `chat_typing`: Call `callbacks.onTyping(is_typing)`
    - `error`: Call `callbacks.onError(error)`
  - Return WebSocket instance for cleanup
- Implement `disconnect(ws: WebSocket)`:
  - Close WebSocket connection gracefully
- Add proper error handling and reconnection logic

### 13. Update Frontend Type Definitions

- Edit `apps/orchestrator_3_stream/frontend/src/types.d.ts`
- Add after existing types:
  - `LoadChatRequest { orchestrator_agent_id: string, limit?: number }`
  - `LoadChatResponse { status: string, messages: ChatMessage[], turn_count: number }`
  - `SendChatRequest { message: string, orchestrator_agent_id: string }`
  - `SendChatResponse { status: string, response: string, session_id: string }`
  - `ChatStreamMessage { type: string, orchestrator_agent_id: string, chunk?: string, is_complete?: boolean, is_typing?: boolean }`
- Ensure `ChatMessage` interface is already defined (it is, from line 128-132)

### 14. Update Pinia Store with Real API Integration

- Edit `apps/orchestrator_3_stream/frontend/src/stores/orchestratorStore.ts`
- Import `chatService` from `../services/chatService`
- Remove `simulateOrchestratorResponse()` function (lines 167-175)
- Update `sendUserMessage(content: string)` function (lines 148-165):
  - Keep user message addition to store
  - Remove setTimeout simulation
  - Call `await chatService.sendMessage(content, orchestratorAgentId)` (use orchestrator ID from state or config)
  - Handle response - message will stream via WebSocket, so no need to add orchestrator message here
  - Handle errors with try/catch
- Update `connectWebSocket()` function (lines 182-186):
  - Import WEBSOCKET_URL from config or env
  - Call `chatService.connectWebSocket(url, { onChatStream, onTyping, onError })`
  - Store WebSocket instance for later cleanup
  - Set `isConnected.value = true` on success
- Update `handleWebSocketMessage()` function (lines 194-197):
  - Remove (replaced by callbacks in chatService.connectWebSocket)
- Add `loadChatHistory()` action:
  - Call `await chatService.loadChatHistory(orchestratorAgentId)`
  - Replace `chatMessages.value` with loaded messages
  - Return turn_count for display
- Add callback handlers:
  - `onChatStream(chunk: string, is_complete: boolean)`:
    - If not complete: append chunk to last orchestrator message (or create new one)
    - If complete: mark message as finalized
    - Set `isTyping.value = false` when complete
  - `onTyping(is_typing: boolean)`:
    - Set `isTyping.value = is_typing`
  - `onError(error: any)`:
    - Log error, show notification to user
- Update `initialize()` function (lines 200-203):
  - Call `loadChatHistory()` after WebSocket connects
  - Handle errors gracefully

### 15. Update Environment Configuration

- Edit `apps/orchestrator_3_stream/.env.sample`
- Add backend API base URL configuration:
  - `VITE_API_BASE_URL=http://127.0.0.1:9403`
- Document orchestrator-specific env vars:
  - `ORCHESTRATOR_AGENT_ID=default-orchestrator`
  - `ORCHESTRATOR_MODEL=claude-sonnet-4-5-20250929`
  - `ORCHESTRATOR_SYSTEM_PROMPT_PATH=./prompts/orchestrator_agent_system_prompt.md`
  - `ORCHESTRATOR_WORKING_DIR=/path/to/working/dir`
- Copy updated `.env.sample` to `.env` if needed

### 16. Test End-to-End Flow

- Start backend: `cd apps/orchestrator_3_stream && ./start_be.sh`
- Start frontend: `cd apps/orchestrator_3_stream && ./start_fe.sh`
- Open browser to frontend URL
- Test sequence:
  1. Verify WebSocket connects (check console logs)
  2. Verify chat history loads (should be empty initially)
  3. Send a test message: "Hello orchestrator"
  4. Verify user message appears immediately in UI
  5. Verify typing indicator shows
  6. Verify orchestrator response streams in
  7. Verify turn counter increments
  8. Send second message to test session continuity
  9. Refresh page and verify chat history persists
- Check database:
  - Verify messages in `orchestrator_chat` table
  - Verify session_id updated in `orchestrator_agents` table
  - Verify token counts and costs updated

### 17. Validate Database Integration

- Query `orchestrator_chat` table:
  - `SELECT * FROM orchestrator_chat ORDER BY created_at DESC LIMIT 10`
  - Verify user and orchestrator messages logged correctly
  - Verify sender_type and receiver_type are correct
  - Verify timestamps are sequential
- Query `orchestrator_agents` table:
  - `SELECT * FROM orchestrator_agents WHERE archived = false`
  - Verify session_id is updated after interaction
  - Verify input_tokens, output_tokens, total_cost are incrementing
- Check turn count:
  - `SELECT COUNT(*) FROM orchestrator_chat WHERE orchestrator_agent_id = '<id>'`
  - Compare with turn counter displayed in UI

## Testing Strategy

**CRITICAL: NO MOCKING - Use Real Integrations**

This project follows a strict no-mocking policy:
- ✅ Use **real database connections** (PostgreSQL via asyncpg)
- ✅ Use **real Claude Agent SDK agents** (actual API calls)
- ❌ **NO mocks, stubs, or fakes** for database or Claude SDK
- ✅ Tests must be **ephemeral** - database starts and ends in the same state
- ✅ **Every module must have a corresponding test file**

### Test Organization

Create test files for each module:
```
apps/orchestrator_3_stream/backend/
├── modules/
│   ├── database.py
│   ├── orchestrator_service.py
│   ├── websocket_manager.py
│   ├── config.py
│   └── logger.py
└── tests/
    ├── test_database.py           # Tests for database.py
    ├── test_orchestrator_service.py  # Tests for orchestrator_service.py
    ├── test_websocket_manager.py     # Tests for websocket_manager.py
    ├── test_http_endpoints.py        # Tests for HTTP routes in main.py
    ├── test_websocket_endpoint.py    # Tests for WebSocket in main.py
    └── conftest.py                   # Pytest fixtures for database setup/teardown
```

### Ephemeral Testing Pattern

**Setup-Execute-Teardown Pattern:**

```python
import pytest
import asyncio
from modules.database import init_pool, close_pool, insert_chat_message, get_orchestrator

@pytest.fixture(scope="function")
async def db_test_data():
    """
    Fixture that creates test data and cleans it up after test.
    Database MUST be in same state before and after test.
    """
    # Setup: Initialize database connection
    await init_pool()

    # Create test orchestrator
    test_orch_id = await create_test_orchestrator()

    yield test_orch_id  # Provide to test

    # Teardown: Clean up ALL test data
    await delete_test_orchestrator(test_orch_id)
    await close_pool()

async def test_insert_chat_message(db_test_data):
    """Test chat message insertion with REAL database"""
    orch_id = db_test_data

    # Execute: Insert message
    msg_id = await insert_chat_message(
        orchestrator_agent_id=orch_id,
        sender_type="user",
        receiver_type="orchestrator",
        message="Test message"
    )

    # Assert: Verify insertion
    assert msg_id is not None

    # Cleanup happens in fixture teardown
```

### Unit Tests (Real Database + Real SDK)

**Backend Module: `test_database.py`**
- ✅ Test `insert_chat_message()` with all sender/receiver combinations
  - Create test orchestrator, insert message, verify in database, clean up
- ✅ Test `get_chat_history()` returns chronological order
  - Create test orchestrator, insert 5 messages, verify order, clean up
- ✅ Test `get_turn_count()` returns accurate count
  - Create test orchestrator, insert N messages, verify count equals N, clean up
- ✅ Test `get_orchestrator()` fetches singleton correctly
  - Create test orchestrator, fetch, verify fields, clean up
- ✅ Test `get_orchestrator_by_session()` validates session exists
  - Create test orchestrator with session_id, fetch by session, verify, clean up
- ✅ Test `update_orchestrator_session()` updates session_id
  - Create test orchestrator, update session, verify change, clean up
- ✅ Test `update_orchestrator_costs()` increments tokens/cost
  - Create test orchestrator, update costs, verify totals, clean up
- ❌ NO database mocks - use real PostgreSQL connection

**Backend Module: `test_orchestrator_service.py`**
- ✅ Test `process_user_message()` with REAL Claude SDK
  - Create test orchestrator, send simple message, verify response received
  - Test will make actual Claude API call (use inexpensive prompt)
  - Verify three-phase logging: user message logged, response logged, costs updated
- ✅ Test WebSocket streaming broadcasts chunks
  - Create test orchestrator, send message, capture WebSocket broadcasts
  - Verify chunks arrive in order and marked complete
- ✅ Test session continuity across multiple messages
  - Create test orchestrator, send 2 messages, verify session_id maintained
- ❌ NO Claude SDK mocks - use real agent execution

**Backend Module: `test_websocket_manager.py`**
- ✅ Test `broadcast_chat_stream()` sends messages to all connected clients
  - Create mock WebSocket connections, broadcast, verify all received
- ✅ Test `set_typing_indicator()` broadcasts typing state
  - Connect clients, send typing indicator, verify received

**Backend Module: `test_http_endpoints.py`**
- ✅ Test `/load_chat` endpoint with real database
  - Create test orchestrator and messages, call endpoint, verify response
- ✅ Test `/send_chat` endpoint with real Claude SDK
  - Create test orchestrator, POST message, verify response and database persistence
- ✅ Test error handling for invalid orchestrator_agent_id
  - Call endpoints with non-existent ID, verify 500 error

**Frontend Module: `test_chatService.ts`**
- ✅ Test `loadChatHistory()` makes HTTP request
  - Use real axios (can use MSW for HTTP mocking if needed)
- ✅ Test `sendMessage()` POSTs to backend
- ✅ Test `connectWebSocket()` establishes connection
- ✅ Test WebSocket message routing to callbacks

### Integration Tests (Real Database + Real SDK + Real HTTP)

**Backend Integration:**
- ✅ Test full chat flow: HTTP POST → Claude SDK execution → Database persistence
  - Create test orchestrator, POST to `/send_chat`, wait for completion
  - Verify message logged before execution (pre-phase)
  - Verify response logged after execution (post-phase)
  - Verify costs updated in database
  - Clean up test data
- ✅ Test WebSocket streaming end-to-end
  - Connect WebSocket client, send chat message via WS
  - Verify streaming chunks received
  - Verify final message persisted to database
- ✅ Test session resumption with `--session` flag
  - Create test orchestrator with session, restart service with `--session`
  - Verify session loaded correctly
- ❌ NO mocking - full integration with real services

**Frontend Integration:**
- ✅ Test Pinia store with real API calls
  - Use test backend instance or staging environment
- ✅ Test WebSocket connection and message handling
  - Connect to real WebSocket, send message, verify state updates

### End-to-End Tests (Real Everything)

**Playwright E2E Tests:**
- ✅ Test complete user workflow with real backend + real Claude SDK
  - Start backend with test database
  - Navigate to frontend
  - Type message, send, wait for response
  - Verify response appears in UI
  - Send follow-up message
  - Verify both messages in chat history
  - Query database to verify all messages persisted
  - Clean up test orchestrator and messages from database
- ✅ Test session persistence across page refresh
  - Send message, refresh page, verify chat history loads
- ❌ NO UI mocking - test against real running application

### Test Data Management

**Critical Rules for Ephemeral Tests:**

1. **Setup Phase**: Create all test data needed for the test
   ```python
   test_orch_id = await create_test_orchestrator(name="test-orch-123")
   test_msg_id = await insert_chat_message(...)
   ```

2. **Execute Phase**: Run the test operation
   ```python
   result = await get_chat_history(test_orch_id)
   assert len(result) == 1
   ```

3. **Teardown Phase**: Delete ALL test data created
   ```python
   await delete_chat_messages(test_orch_id)
   await delete_orchestrator(test_orch_id)
   ```

4. **Verification**: Database must be in identical state before/after test
   ```python
   # Before test: SELECT COUNT(*) FROM orchestrator_chat = N
   # Run test (creates and cleans up data)
   # After test: SELECT COUNT(*) FROM orchestrator_chat = N (same as before)
   ```

5. **Isolation**: Each test creates its own test data (unique IDs/names)
   - Use UUID or timestamp suffixes: `test-orch-{uuid4()}`
   - Prevents conflicts when tests run in parallel

### Test Execution Commands

```bash
# Run all backend tests with real database and Claude SDK
cd apps/orchestrator_3_stream/backend
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/test_database.py -v

# Run with coverage
uv run pytest tests/ --cov=modules --cov-report=html

# Run E2E tests (requires backend + frontend running)
cd apps/orchestrator_3_stream
./start_be.sh  # Terminal 1
./start_fe.sh  # Terminal 2
uv run pytest tests/test_e2e.py  # Terminal 3
```

### Test Environment Setup

**Environment Variables for Testing:**
```bash
# Use separate test database (same schema, different data)
TEST_DATABASE_URL=postgresql://user:pass@localhost:5432/orchestrator_test

# Use cheaper Claude model for tests (optional)
TEST_CLAUDE_MODEL=claude-sonnet-4-5-20250929

# Disable expensive features during tests (if needed)
TEST_MODE=true
```

**Database Reset Between Test Runs:**
```bash
# Optional: Reset test database to clean state
psql $TEST_DATABASE_URL -c "TRUNCATE orchestrator_chat, orchestrator_agents CASCADE;"
```

### Why No Mocking?

**Benefits of Real Integration Testing:**
1. **Catches real issues**: Database constraints, API errors, network problems
2. **Confidence**: If tests pass, production will work
3. **Accurate behavior**: No "works in mock, fails in production"
4. **Schema validation**: Database tests verify migrations are correct
5. **API compatibility**: Claude SDK tests verify API usage is current

**Managing Test Costs:**
- Use simple prompts for Claude SDK tests (e.g., "Say hello")
- Limit test runs during development (only run full suite before commits)
- Use test database separate from production
- Clean up test data immediately to avoid accumulation

### Test Coverage Goals

- ✅ Every module has a test file
- ✅ Every function is tested with real integrations
- ✅ All tests are ephemeral (setup → execute → teardown)
- ✅ Database is in same state before/after each test
- ✅ 80%+ code coverage with real integration tests
- ❌ 0% mocking 

## Acceptance Criteria

- [ ] User can load existing chat history on app mount (or empty state for new conversations)
- [ ] User can type and send messages to orchestrator
- [ ] User message appears immediately in chat UI
- [ ] Typing indicator shows while orchestrator is responding
- [ ] Orchestrator response streams in real-time via WebSocket
- [ ] Complete orchestrator response displays after streaming completes
- [ ] All messages (user + orchestrator) are persisted to `orchestrator_chat` table
- [ ] Turn counter accurately reflects total message count from database
- [ ] Session ID is maintained and updated after each interaction
- [ ] Token counts and costs are tracked and displayed
- [ ] Chat history persists across page refreshes
- [ ] WebSocket reconnects automatically on disconnect
- [ ] Error messages display gracefully for failed operations
- [ ] Backend logs all chat events for debugging

## Validation Commands

Execute these commands to validate the task is complete:

- `uv run python apps/orchestrator_db/sync_models.py` - **FIRST STEP**: Sync database models to backend (REQUIRED before any other steps)
- `cd apps/orchestrator_3_stream/backend && uv run python -m py_compile modules/*.py` - Verify backend Python syntax
- `cd apps/orchestrator_3_stream/frontend && npm run build` - Verify frontend builds without errors
- `cd apps/orchestrator_3_stream/backend && uv run python -c "from modules.database import init_pool, close_pool; import asyncio; asyncio.run(init_pool()); asyncio.run(close_pool()); print('Database connection test passed')"` - Test database connection
- `cd apps/orchestrator_3_stream && ./start_be.sh` - Start backend (check console for errors)
- `cd apps/orchestrator_3_stream && ./start_fe.sh` - Start frontend (check console for errors)
- `curl http://127.0.0.1:9403/health` - Verify backend health endpoint returns 200
- `psql $DATABASE_URL -c "SELECT COUNT(*) FROM orchestrator_chat"` - Verify chat table accessible
- Open browser to `http://127.0.0.1:5175` - Verify UI loads without console errors
- Send test message in UI - Verify message appears and orchestrator responds

### Automated End-to-End Validation with Playwright

Use the `playwright-validator` agent to automate testing of the chat interface:

**Task**: Create a playwright validation test that:
1. Navigates to `http://127.0.0.1:5175`
2. Locates the chat input field in the OrchestratorChat component
3. Types a test message: "Hello orchestrator, can you confirm you're working?"
4. Clicks the send button (or presses Enter)
5. Waits for orchestrator response to appear (up to 30 seconds)
6. Validates that a response was received
7. Sends a follow-up message: "What is 2+2?"
8. Waits for second response
9. Validates both messages and responses are visible in chat history
10. Takes screenshots of the final state
11. Reports success/failure with detailed logs

**Command to run**:
```bash
# Launch playwright-validator agent with test specification
claude-code agent playwright-validator --prompt "Navigate to http://127.0.0.1:5175 and test the orchestrator chat interface. First, type 'Hello orchestrator, can you confirm you're working?' into the chat input and send it. Wait for the orchestrator response to stream in. Then send a follow-up message: 'What is 2+2?' and wait for the second response. Take screenshots after each interaction. Validate that both user messages and orchestrator responses are visible in the chat history. Report success if all messages are displayed correctly and responses were received."
```

**Success Criteria**:
- Playwright agent successfully navigates to frontend
- Chat input is located and message is typed
- Message sends without errors
- Orchestrator response appears within 30 seconds
- Follow-up message also receives response
- Both message pairs visible in chat UI
- Screenshots show complete conversation
- No JavaScript console errors reported

**Troubleshooting**:
- If chat input not found: Verify frontend is running and chat component loaded
- If no response received: Check backend logs for orchestrator execution errors
- If WebSocket disconnected: Verify WebSocket connection in browser DevTools
- If messages not persisting: Check database for `orchestrator_chat` table entries

## Notes

### Database Schema
The `orchestrator_chat` table already exists from `apps/orchestrator_db/migrations/8_orchestrator_chat.sql`.

**OrchestratorChat Model** (from `apps/orchestrator_db/models.py`):
- `id` (UUID) - Message identifier
- `orchestrator_agent_id` (UUID FK) - Links to orchestrator agent
- `sender_type` (Literal['user', 'orchestrator', 'agent']) - Message sender
- `receiver_type` (Literal['user', 'orchestrator', 'agent']) - Message receiver
- `message` (str) - The actual message content (single field for all messages)
- `agent_id` (Optional[UUID]) - Required when sender/receiver is 'agent', None otherwise
- `metadata` (Dict[str, Any]) - Extra data like tools_used
- `created_at`, `updated_at` (datetime) - Automatic timestamps

**Three-Way Communication Pattern**:
1. User → Orchestrator: `sender_type='user'`, `receiver_type='orchestrator'`, `agent_id=None`
2. Orchestrator → User: `sender_type='orchestrator'`, `receiver_type='user'`, `agent_id=None`
3. Orchestrator → Agent: `sender_type='orchestrator'`, `receiver_type='agent'`, `agent_id=<agent_uuid>`
4. Agent → Orchestrator: `sender_type='agent'`, `receiver_type='orchestrator'`, `agent_id=<agent_uuid>`

**Model Location**: After running `sync_models.py`, the models are available at:
- `apps/orchestrator_3_stream/backend/modules/orch_database_models.py`
- Import: `from .orch_database_models import OrchestratorChat, OrchestratorAgent`

### Claude SDK Integration
The orchestrator service will use the Claude Agent SDK with:
- Model: `claude-sonnet-4-5-20250929` (or configured model)
- System prompt: Load from file or environment variable
- MCP tools: Management tools for agent operations (create, list, command, etc.)
- Streaming: Use async iteration over response messages

### Session Management
- Session ID is stored in `orchestrator_agents` table (singleton record)
- Each interaction resumes the session for conversation continuity
- Frontend should store orchestrator_agent_id (can be hardcoded or loaded from API)

### CLI Parameters Support
The backend supports command-line arguments for flexible configuration:

**--session <session_id>** (Optional)
- Resume an existing orchestrator session
- Validates session exists in database before starting
- Loads previous conversation context, token counts, and costs
- Example: `uv run python main.py --session "sess_abc123..."`
- If invalid session provided, logs error and exits

**--cwd <path>** (Optional)
- Set working directory for orchestrator and command-level agents
- Priority: `--cwd flag > os.getcwd() > .env ORCHESTRATOR_WORKING_DIR`
- All agents created by orchestrator will inherit this working directory
- Example: `uv run python main.py --cwd /Users/dan/projects/my-app`

**Usage Examples**:
```bash
# Start new session with default settings
uv run python main.py

# Start new session with custom working directory
uv run python main.py --cwd /path/to/project

# Resume existing session
uv run python main.py --session "sess_abc123..."

# Resume session with different working directory
uv run python main.py --session "sess_abc123..." --cwd /different/path
```

**Implementation Notes**:
- Parse CLI args using `argparse` before creating FastAPI app
- Store parsed values in global variables or app.state for access in lifespan
- Call `get_orchestrator_by_session()` to validate session before proceeding
- Call `get_or_create_orchestrator()` if no session specified
- Pass session_id and working_dir to OrchestratorService constructor
- Log launch configuration showing session status and working directory
- Reference pattern: `apps/orchestrator_1_term/modules/cli_interface.py` lines 401-527

### Streaming Implementation
- Backend streams response chunks via WebSocket as they arrive from Claude API
- Frontend accumulates chunks and displays them incrementally for better UX
- Full response is saved to database only after streaming completes

### Error Handling
- Database errors: Log and return HTTP 500 with error detail
- WebSocket errors: Attempt reconnection with exponential backoff
- API errors: Display user-friendly error messages in UI
- Claude SDK errors: Log full error, return generic message to user

### Performance Considerations
- Use connection pooling (asyncpg) to handle concurrent requests
- Limit chat history loading (default 50 messages, paginate if needed)
- Consider caching orchestrator_agent_id to avoid repeated queries

### Dependencies
Ensure these are installed:
- Backend: `fastapi`, `uvicorn`, `asyncpg`, `python-dotenv`, `claude-agent-sdk`, `rich`
- Frontend: `vue`, `pinia`, `axios` (or `fetch` API)

If new libraries are needed, add with:
- Backend: `cd apps/orchestrator_3_stream/backend && uv add <package>`
- Frontend: `cd apps/orchestrator_3_stream/frontend && npm install <package>`

### Reference Implementation
The `apps/orchestrator_1_term` implementation is production-ready and battle-tested. When in doubt, refer to:
- `orchestrator_agent.py` lines 741-1116 for execution flow
- `orchestrator_chat_db.py` for database operations
- `orchestrator_agents_db.py` for session/cost management

### Future Enhancements (Out of Scope)
- Message editing/deletion
- Chat history pagination for very long conversations
- Multi-orchestrator support (currently singleton only)
- Markdown rendering in chat messages
- Code syntax highlighting in responses
- Message reactions or threading
