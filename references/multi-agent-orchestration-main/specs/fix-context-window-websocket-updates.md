# Plan: Fix Context Window WebSocket Updates for AgentList.vue

## Task Description

The WebSocket events that should be updating the agents' context window (token usage) in AgentList.vue aren't updating the UI. When agents execute tasks and consume tokens, the backend updates the database with new token counts (input_tokens, output_tokens, total_cost), but the frontend never receives these updates, causing the context window display to remain static despite actual backend changes.

## Objective

Implement WebSocket broadcasting of agent token/cost updates so that the AgentList.vue component displays real-time context window usage as agents execute tasks. The frontend will reactively update the context window progress bars and token counts without requiring manual refresh.

## Problem Statement

The root cause is a **missing WebSocket broadcast step** in the agent execution workflow:

1. **Backend Flow (Currently):**
   - Agent executes task via `command_agent()` in `agent_manager.py`
   - `_process_agent_messages()` captures usage data from ResultMessage
   - `update_agent_costs()` saves token counts to database (line 1208-1210)
   - ❌ **MISSING**: No WebSocket broadcast after database update
   - Result: Database has correct tokens, but frontend never knows

2. **Frontend Readiness (Already Working):**
   - AgentList.vue calculates context window from `agent.input_tokens + agent.output_tokens`
   - orchestratorStore.ts has `handleAgentUpdated()` handler ready (lines 332-351)
   - chatService.ts routes `agent_updated` events to the store handler
   - Vue reactivity is properly configured with spread operators

3. **Infrastructure Exists (Not Used):**
   - websocket_manager.py has `broadcast_agent_updated()` method (lines 125-129)
   - This method is never called after cost updates
   - Other broadcasts work correctly (agent_created, agent_status_change, etc.)

## Solution Approach

Add a WebSocket broadcast immediately after the `update_agent_costs()` database operation in agent_manager.py. This follows the existing pattern used for agent status changes (lines 781-806) and ensures the frontend receives real-time token/cost updates. Fetch the updated agent data from the database to get cumulative totals, then broadcast the complete agent data via the existing `broadcast_agent_updated()` method.

## Relevant Files

### Backend Files

- **apps/orchestrator_3_stream/backend/modules/agent_manager.py** (lines 1200-1224)
  - Contains the `_process_agent_messages()` method where costs are updated
  - Missing WebSocket broadcast after `update_agent_costs()` call (line 1210)
  - Needs to fetch updated agent and call `broadcast_agent_updated()`

- **apps/orchestrator_3_stream/backend/modules/websocket_manager.py** (lines 125-129)
  - Already has `broadcast_agent_updated()` method implemented
  - Takes agent_id (str) and agent_data (dict) as parameters
  - Broadcasts `{"type": "agent_updated", "agent_id": ..., "agent": ...}`

- **apps/orchestrator_3_stream/backend/modules/database.py** (lines 757-785)
  - Contains `update_agent_costs()` function that updates database
  - Also needs `get_agent()` function to fetch updated agent data

### Frontend Files

- **apps/orchestrator_3_stream/frontend/src/stores/orchestratorStore.ts** (lines 332-351)
  - Already has `handleAgentUpdated()` handler properly wired
  - Merges updated agent data using spread operator: `agents.value[index] = {...old, ...new}`
  - WebSocket event routing is already configured (lines 252-256)

- **apps/orchestrator_3_stream/frontend/src/services/chatService.ts** (lines 121-122)
  - Already routes `agent_updated` events to `onAgentUpdated` callback
  - No changes needed - infrastructure is ready

- **apps/orchestrator_3_stream/frontend/src/components/AgentList.vue** (lines 253-267)
  - Correctly calculates context window from agent.input_tokens + agent.output_tokens
  - Reactive computed properties will automatically update when agent data changes
  - No changes needed - reactive chain is correct

### Database Models

- **apps/orchestrator_db/models.py** (lines 88-143, Agent model)
  - Agent model has input_tokens, output_tokens, total_cost fields
  - No schema changes required

## Step by Step Tasks

### 1. Add WebSocket Broadcast After Cost Update

**Location:** `apps/orchestrator_3_stream/backend/modules/agent_manager.py` (after line 1210)

- Import `get_agent` function from database module if not already imported
- After the `update_agent_costs()` call completes, fetch the updated agent from database
- Call `broadcast_agent_updated()` with the agent's updated token/cost data
- Include cumulative values (not incremental) to ensure frontend has accurate totals
- Add debug logging to trace the broadcast

**Code to add:**
```python
# After line 1210 (after update_agent_costs call)
if total_input_tokens or total_output_tokens:
    await update_agent_costs(
        agent_id, total_input_tokens, total_output_tokens, total_cost
    )

    # Fetch updated agent to get cumulative totals
    updated_agent = await get_agent(agent_id)

    if updated_agent:
        # Broadcast updated token/cost data to frontend
        await self.ws_manager.broadcast_agent_updated(
            agent_id=str(agent_id),
            agent_data={
                "input_tokens": updated_agent.input_tokens,
                "output_tokens": updated_agent.output_tokens,
                "total_cost": float(updated_agent.total_cost)
            }
        )

        self.logger.debug(
            f"Broadcast token update for agent {updated_agent.name} ({agent_id}): "
            f"in={updated_agent.input_tokens}, out={updated_agent.output_tokens}, "
            f"cost=${float(updated_agent.total_cost):.4f}"
        )
```

**Rationale:**
- Fetching the updated agent ensures we send cumulative totals, not incremental values
- This prevents sync issues if the frontend misses a broadcast or joins late
- Follows the same pattern as `broadcast_agent_status_change()` (lines 781-806)
- Includes only the fields that changed to minimize payload size
- Debug logging helps trace WebSocket event flow during development

### 2. Verify Database Import

**Location:** `apps/orchestrator_3_stream/backend/modules/agent_manager.py` (top of file)

- Check if `get_agent` is imported from `.database` module
- If not, add to imports: `from .database import get_agent, update_agent_costs, ...`
- Verify the function signature matches expected usage

**Expected import statement:**
```python
from .database import (
    get_agent,
    update_agent_costs,
    # ... other imports
)
```

### 3. Test WebSocket Event Flow

**Location:** Browser developer console + backend logs

- Start the backend server (`./start_be.sh`)
- Start the frontend server (`./start_fe.sh`)
- Open browser DevTools → Console
- Create a test agent
- Send the agent a command that will consume tokens
- Watch for console logs:
  - Frontend: "Agent updated:" log from orchestratorStore.ts line 333
  - Frontend: "✅ Updated agent <id> with new data:" log from store
  - Backend: "Broadcast token update for agent..." debug log
- Verify the context window progress bar in AgentList.vue updates in real-time
- Verify token counts (e.g., "1.2k / 200k") update correctly

**Validation checklist:**
- [ ] Backend logs show "Broadcast token update for agent..."
- [ ] Frontend console shows "Agent updated:" with payload
- [ ] Context window progress bar moves forward
- [ ] Token count text updates (e.g., "0 / 200k" → "1.5k / 200k")
- [ ] Multiple rapid updates are handled smoothly
- [ ] Updates persist after page refresh (database confirmation)

### 4. Test Edge Cases

**Location:** Browser + backend logs

- **Test Scenario 1: Multiple agents executing simultaneously**
  - Create 3 agents
  - Send commands to all 3 at once
  - Verify each agent's context window updates independently
  - Confirm no cross-agent data contamination

- **Test Scenario 2: WebSocket reconnection**
  - Start an agent task
  - Disconnect WebSocket (close browser DevTools → Network → WS)
  - Reconnect WebSocket
  - Verify agent data syncs correctly (cumulative totals are accurate)

- **Test Scenario 3: Zero token updates**
  - Send a command that results in 0 new tokens (edge case)
  - Verify no error occurs
  - Confirm broadcast still happens (or is skipped gracefully)

- **Test Scenario 4: Large token counts**
  - Simulate agent with >100k tokens consumed
  - Verify UI displays correctly (e.g., "125k / 200k")
  - Verify percentage calculation is accurate

### 5. Validate Reactive Patterns

**Location:** Vue DevTools + browser console

- Open Vue DevTools → Pinia Store
- Watch the `agents` array in orchestratorStore
- Trigger an agent task
- Verify:
  - Agent object in store updates (input_tokens, output_tokens, total_cost fields change)
  - AgentList.vue component re-renders automatically
  - Computed properties (`getTotalTokens`, `getContextPercentage`) recalculate
  - No manual refresh required

**Expected behavior:**
- Store mutation should be visible in Vue DevTools timeline
- Component should show "re-render" event
- No console errors related to reactivity

### 6. Review Existing Broadcast Patterns

**Location:** `apps/orchestrator_3_stream/backend/modules/agent_manager.py`

- Compare the new broadcast implementation with existing working broadcasts:
  - `broadcast_agent_status_change()` (lines 781-806)
  - `broadcast_agent_created()` (lines 670-678)
  - `broadcast_agent_summary_update()` (lines 1290-1293)
- Ensure consistency in:
  - UUID-to-string conversion: `str(agent_id)`
  - Float conversion for Decimal fields: `float(total_cost)`
  - Error handling patterns
  - Logging verbosity

**Consistency checklist:**
- [ ] UUIDs converted to strings before broadcast
- [ ] Decimal/float types properly serialized
- [ ] Follows same error handling pattern as other broadcasts
- [ ] Logging level matches other debug logs
- [ ] Method calls follow same async/await pattern

## Testing Strategy

### Unit Tests (Backend)

**File:** `apps/orchestrator_3_stream/backend/tests/test_agent_manager.py`

Test cases to add:
1. **Test broadcast after cost update:**
   - Mock `update_agent_costs()` and `get_agent()`
   - Mock `ws_manager.broadcast_agent_updated()`
   - Call `_process_agent_messages()` with mock ResultMessage containing usage data
   - Assert `broadcast_agent_updated()` was called with correct parameters

2. **Test cumulative token values:**
   - Mock agent with existing tokens (input=1000, output=500)
   - Update with new tokens (input=200, output=100)
   - Verify broadcast contains cumulative totals (input=1200, output=600)

3. **Test error handling:**
   - Mock `get_agent()` to return None
   - Verify broadcast is skipped gracefully (no exception)
   - Mock `broadcast_agent_updated()` to raise exception
   - Verify error is logged but doesn't crash agent execution

### Integration Tests (Backend + Frontend)

**File:** `apps/orchestrator_3_stream/backend/tests/test_websocket_integration.py`

Test cases:
1. **End-to-end WebSocket flow:**
   - Start WebSocket connection
   - Create agent and command it
   - Capture WebSocket messages
   - Assert `agent_updated` event received with token data

2. **Event ordering:**
   - Send command that generates multiple blocks (TextBlock, ToolUseBlock)
   - Verify `agent_log` events arrive before `agent_updated` event
   - Confirm cumulative tokens reflect all blocks

3. **Multiple concurrent agents:**
   - Create 3 agents, send commands to each
   - Verify each agent receives correct token updates
   - Confirm no cross-agent contamination

### Frontend Tests (Playwright)

**File:** `apps/orchestrator_3_stream/frontend/tests/agentlist.spec.ts`

Test cases:
1. **Context window updates visually:**
   - Navigate to orchestrator UI
   - Create agent
   - Send command
   - Wait for context window progress bar to update
   - Assert bar width increases
   - Assert token count text changes

2. **Reactive updates:**
   - Monitor agent card in AgentList
   - Trigger token update via WebSocket
   - Verify UI updates without manual refresh
   - Confirm no page flicker or full re-render

## Acceptance Criteria

1. **WebSocket Broadcast Implementation:**
   - [ ] `broadcast_agent_updated()` is called after `update_agent_costs()` completes
   - [ ] Broadcast includes cumulative token values (input_tokens, output_tokens, total_cost)
   - [ ] UUID is converted to string before broadcast
   - [ ] Decimal total_cost is converted to float before broadcast
   - [ ] Debug logging traces the broadcast with agent name and token counts

2. **Frontend Updates:**
   - [ ] Context window progress bar updates in real-time as agent executes
   - [ ] Token count text updates (e.g., "1.5k / 200k")
   - [ ] Context percentage calculation is accurate
   - [ ] No manual refresh required
   - [ ] Updates are smooth without UI flicker

3. **Reactivity:**
   - [ ] Store's `handleAgentUpdated()` receives WebSocket events
   - [ ] Agent object in store updates with new token values
   - [ ] AgentList.vue component re-renders automatically
   - [ ] Computed properties recalculate correctly

4. **Edge Cases:**
   - [ ] Multiple agents update independently without interference
   - [ ] Zero token updates don't cause errors
   - [ ] Large token counts (>100k) display correctly
   - [ ] WebSocket reconnection syncs agent data accurately
   - [ ] Rapid successive updates are handled smoothly

5. **Code Quality:**
   - [ ] Follows existing broadcast patterns in agent_manager.py
   - [ ] Consistent UUID/float conversion patterns
   - [ ] Error handling matches existing patterns
   - [ ] Debug logging is helpful but not verbose
   - [ ] No new dependencies or breaking changes

## Validation Commands

Execute these commands to validate the task is complete:

### Backend Validation
```bash
# Ensure backend code compiles
cd apps/orchestrator_3_stream/backend
uv run python -m py_compile modules/agent_manager.py

# Run backend tests (if they exist)
uv run pytest tests/test_agent_manager.py -v

# Start backend and check for import errors
./start_be.sh
# Look for successful startup without errors
```

### Frontend Validation
```bash
# Ensure frontend code compiles
cd apps/orchestrator_3_stream/frontend
npm run type-check

# Start frontend dev server
./start_fe.sh
# Verify no compilation errors
```

### Integration Testing
```bash
# Start both backend and frontend
cd apps/orchestrator_3_stream
./start_be.sh &
./start_fe.sh &

# Open browser to http://localhost:5175
# Create agent, send command, verify context window updates in real-time
```

### Manual Verification Checklist
- [ ] Open browser DevTools → Console
- [ ] Create test agent via UI
- [ ] Send command: "Read the package.json file"
- [ ] Observe backend logs for "Broadcast token update..."
- [ ] Observe frontend console for "Agent updated:" log
- [ ] Verify context window bar in AgentList moves from 0% to >0%
- [ ] Verify token count text updates (e.g., "0 / 200k" → "1.2k / 200k")
- [ ] Create second agent and verify independent updates
- [ ] Refresh page and verify token counts persist (database check)

## Notes

### Why Fetch Agent After Update?

We fetch the updated agent from the database (`get_agent()`) instead of broadcasting the incremental values because:
- Ensures cumulative totals are accurate (database is source of truth)
- Prevents sync issues if frontend misses a broadcast
- Makes the frontend stateless for token tracking
- Allows late-joining clients to get accurate data

### Why Not Broadcast Incremental Values?

Incremental broadcasting (e.g., "added 100 tokens") is problematic:
- Frontend state could drift out of sync
- WebSocket disconnections cause missing updates
- Multiple rapid updates complicate client-side aggregation
- Database and frontend could disagree on totals

By always broadcasting cumulative values, we eliminate these issues.

### Alternative Approaches Considered

1. **Broadcast incremental values:**
   - Simpler payload (just the delta)
   - ❌ Frontend state can drift
   - ❌ Requires client-side aggregation logic

2. **Poll backend periodically:**
   - No WebSocket changes needed
   - ❌ Delayed updates (not real-time)
   - ❌ Unnecessary server load

3. **Broadcast full agent object:**
   - Ensures all fields are up-to-date
   - ❌ Large payload size
   - ❌ Includes unchanged data

**Selected Approach (Partial Update with Cumulative Values):**
- ✅ Real-time updates
- ✅ Small payload (only token fields)
- ✅ Cumulative totals prevent drift
- ✅ Follows existing patterns in codebase

### Performance Considerations

- **Database Query Overhead:** Fetching the updated agent adds one database query per agent execution. This is acceptable because:
  - Agent executions are not extremely frequent (typically seconds apart)
  - The query is simple (primary key lookup)
  - Benefits (accurate data) outweigh minimal overhead

- **WebSocket Broadcast Overhead:** One additional broadcast per agent execution. Minimal impact:
  - Payload is small (~100 bytes)
  - WebSocket is already used extensively for other events
  - Modern WebSocket connections handle this easily

### Future Enhancements

If needed later, consider:
1. **Throttling broadcasts:** If updates are extremely frequent (>10/second), throttle to avoid overwhelming WebSocket
2. **Batch updates:** Aggregate multiple token updates before broadcasting (trade latency for efficiency)
3. **Model-specific context windows:** Replace hardcoded 200k limit with model-based lookup (Sonnet vs Haiku have different limits)
4. **Context warning events:** Broadcast specific events when context usage exceeds thresholds (80%, 90%)

These enhancements are not critical for the initial fix and can be implemented separately if needed.
