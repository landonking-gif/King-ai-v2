# Plan: Display System Information in GlobalCommandInput.vue

## Task Description

Enhance the GlobalCommandInput.vue component to display orchestrator system information below the input field. The information will be sourced from the orchestrator agent's SystemMessage data, which contains session metadata including session_id, cwd, tools, mcp_servers, model, slash_commands, and available agent templates.

The system will:
1. Capture SystemMessage data when the orchestrator boots up (after it's first message is sent)
2. Store this data in `orchestrator_agents.metadata` as `{system_message_info: {...}}`
3. Expose this data via the existing `/get_orchestrator` HTTP endpoint
4. Display 4 key pieces of information in GlobalCommandInput.vue:
   - Orchestrator Session ID
   - Current Working Directory (CWD)
   - Available Slash Commands (as tag UI)
   - Available Agent Templates (as list)
5. When we open the GlobalCommandInput.vue component, we'll fetch the data from the `/get_orchestrator` endpoint (IF it is not already in the state/store - check for specific variables)

To be completely clear we're operating inside the apps/orchestrator_3_stream directory - no where else. You may pull files from other directories as needed BUT we're only working inside the apps/orchestrator_3_stream directory (you should already have this as your CWD)

## Objective

By the end of this implementation:
- SystemMessage data will be captured from Claude SDK and persisted to the database
- The `/get_orchestrator` endpoint will return enriched data including session info, slash commands, and templates
- GlobalCommandInput.vue will display system information in a clean, informative layout below the input field
- Users will have better visibility into their orchestrator's capabilities and current state
- We want to overwrite the orchestrator_agents.metadata with new system data message info once every first message is sent to the orchestrator. So to be clear just store a local variable to keep track of this, so it resets every time the orchestrator boots up.

## Problem Statement

Currently, when the orchestrator agent boots up and receives a SystemMessage from Claude SDK containing valuable system metadata (session_id, cwd, available tools, slash commands, agent templates), this information is only logged to the console and not persisted or displayed to users. The GlobalCommandInput component is a simple textarea without any system information context, making it difficult for users to understand what capabilities are available or what the current working state is.

## Solution Approach

Implement a three-tier solution:

1. **Backend Data Capture Layer**: Intercept SystemMessage in `orchestrator_service.py` and store the data in `orchestrator_agents.metadata` using a new database update function
2. **Backend API Layer**: Enhance `/get_orchestrator` endpoint to return SystemMessage data along with basic orchestrator info; add helper functions to discover slash commands and templates
3. **Frontend Display Layer**: Modify GlobalCommandInput.vue to fetch and display the 4 key information items in a styled info panel below the input field

## Relevant Files

### Backend Files

- **backend/modules/orchestrator_service.py** (Lines 520-532)
  - Currently receives SystemMessage and logs it
  - MODIFY: Add logic to extract data and call database update function
  - MODIFY: Store SystemMessage data in structured format

- **backend/modules/database.py** (After line 380)
  - Currently has update functions for session, costs, and status
  - ADD: New `update_orchestrator_metadata()` function using JSONB merge operator
  - ADD: New `get_orchestrator_with_metadata()` helper to return full orchestrator info

- **backend/modules/subagent_loader.py** (Lines 98-221)
  - Already contains SubagentRegistry class that discovers templates
  - USE: Leverage existing `list_templates()` and `get_available_names()` methods
  - NO CHANGES: Reuse existing functionality

- **backend/main.py** (Lines 214-241)
  - Currently implements `/get_orchestrator` endpoint
  - MODIFY: Enhance response to include metadata, slash commands list, and templates list
  - ADD: Helper function to discover slash commands from `.claude/commands/`

### Frontend Files

- **frontend/src/components/GlobalCommandInput.vue** (Lines 1-278)
  - Currently displays only textarea and send button
  - MODIFY: Add system info display section below input field
  - MODIFY: Fetch orchestrator info on mount
  - MODIFY: Add styled UI for 4 info items (session ID, CWD, commands, templates)

- **frontend/src/types.d.ts** (After line 27)
  - Currently has OrchestratorAgent interface
  - ADD: SystemMessageInfo interface for metadata structure
  - ADD: SlashCommand interface for command metadata
  - ADD: SubagentTemplate interfaces (SubagentFrontmatter and SubagentTemplate)

- **frontend/src/services/chatService.ts** (Lines 14-17)
  - Already has `getOrchestratorInfo()` function
  - NO CHANGES: Function already exists and will return enriched data automatically

- **frontend/src/styles/global.css** (Lines 235-274)
  - Already has badge system and utility classes
  - USE: Leverage existing `.badge`, `.badge-info`, `.detail-label`, `.detail-value` classes
  - POSSIBLY ADD: New utility classes if needed for tag layout

### Database Files

- **apps/orchestrator_db/models.py** (Lines 31-81)
  - OrchestratorAgent model already has `metadata: Dict[str, Any]` field
  - NO CHANGES: Field already supports JSONB storage

### New Files

None - all changes are modifications to existing files.

## Implementation Phases

### Phase 1: Backend Data Capture (Lines of Code: ~150)
- Add `update_orchestrator_metadata()` function to database.py
- Modify SystemMessage handling in orchestrator_service.py to extract and store data
- Test metadata storage with real orchestrator execution

### Phase 2: Backend API Enhancement (Lines of Code: ~120)
- Create slash command discovery function
- Enhance `/get_orchestrator` endpoint to return metadata, commands, and templates
- Add helper functions for formatting data
- Test API response structure

### Phase 3: Frontend Display Implementation (Lines of Code: ~200)
- Add TypeScript interfaces for new data structures
- Modify GlobalCommandInput.vue to add info display section
- Implement fetching logic on component mount
- Style the info panel with existing CSS patterns
- Test UI display with real data

## Step by Step Tasks

IMPORTANT: Execute every step in order, top to bottom.

### 1. Add Database Metadata Update Function

**File**: `backend/modules/database.py` (after line 380)

- Add `update_orchestrator_metadata(metadata_updates: Dict[str, Any])` async function
- Use PostgreSQL JSONB concatenation operator `||` to merge new data with existing metadata
- Pattern follows existing `update_log_payload()` at lines 952-966
- Update `updated_at` timestamp automatically
- Test function works correctly with sample data

**Code Template**:
```python
async def update_orchestrator_metadata(metadata_updates: Dict[str, Any]) -> None:
    """
    Update orchestrator metadata fields using JSONB merge.

    Merges provided metadata with existing metadata without replacing it.

    Args:
        metadata_updates: Dictionary of metadata fields to add/update
    """
    async with get_connection() as conn:
        await conn.execute(
            """
            UPDATE orchestrator_agents
            SET metadata = metadata || $1::jsonb, updated_at = NOW()
            WHERE archived = false
            """,
            json.dumps(metadata_updates),
        )
```

### 2. Capture SystemMessage in Orchestrator Service

**File**: `backend/modules/orchestrator_service.py` (at line 531, after logging SystemMessage)

- Extract relevant fields from SystemMessage.data
- Structure data as `system_message_info` dict with keys: session_id, cwd, tools, mcp_servers, model, slash_commands (if available), agents (template names)
- Call `await update_orchestrator_metadata({"system_message_info": extracted_data})` to persist
- Log success/failure for debugging
- Handle errors gracefully (log but don't crash)

**Implementation Note**: SystemMessage contains raw data from Claude SDK. Extract only the fields we need for display.

**Code Structure**:
```python
# After line 531 in orchestrator_service.py
if isinstance(message, SystemMessage):
    subtype = getattr(message, "subtype", "unknown")
    data = getattr(message, "data", {})

    self.logger.warning(
        f"[OrchestratorService] SystemMessage received:\n"
        f"  Subtype: {subtype}\n"
        f"  Data: {data}\n"
    )

    # NEW: Store SystemMessage data in orchestrator metadata
    try:
        from .database import update_orchestrator_metadata

        # Extract relevant fields for display
        system_message_info = {
            "session_id": data.get("session_id"),
            "cwd": data.get("cwd"),
            "tools": data.get("tools", []),
            "model": data.get("model"),
            "subtype": subtype,
            "captured_at": datetime.now(timezone.utc).isoformat()
        }

        # Store in metadata
        await update_orchestrator_metadata({
            "system_message_info": system_message_info
        })

        self.logger.info(f"✅ Stored SystemMessage data in orchestrator metadata")
    except Exception as e:
        self.logger.error(f"Failed to store SystemMessage data: {e}")

    continue
```

### 3. Create Slash Command Discovery Function

**File**: `backend/main.py` (after line 261, after get_headers endpoint)

- Add helper function `discover_slash_commands()` that scans `.claude/commands/` directory
- Parse YAML frontmatter from each `.md` file to extract: name, description, argument-hint
- Return list of dicts: `[{"name": "scout", "description": "...", "arguments": "..."}, ...]`
- Handle missing directory gracefully (return empty list)
- Cache results to avoid repeated filesystem scans (use simple module-level cache)

**Code Structure**:
```python
from pathlib import Path
import yaml

# Cache for slash commands (module-level)
_slash_commands_cache = None

def discover_slash_commands(working_dir: str) -> List[Dict[str, Any]]:
    """
    Discover slash commands from .claude/commands/ directory.

    Returns:
        List of dicts with name, description, arguments
    """
    global _slash_commands_cache

    # Return cached results if available
    if _slash_commands_cache is not None:
        return _slash_commands_cache

    commands = []
    commands_dir = Path(working_dir) / ".claude" / "commands"

    if not commands_dir.exists():
        logger.warning(f"Slash commands directory not found: {commands_dir}")
        _slash_commands_cache = []
        return commands

    # Find all .md files
    for file_path in commands_dir.glob("*.md"):
        try:
            content = file_path.read_text()

            # Parse YAML frontmatter
            if content.startswith("---"):
                parts = content.split("---", 2)
                if len(parts) >= 3:
                    frontmatter = yaml.safe_load(parts[1])

                    commands.append({
                        "name": file_path.stem,
                        "description": frontmatter.get("description", ""),
                        "arguments": frontmatter.get("argument-hint", ""),
                        "model": frontmatter.get("model", "")
                    })
        except Exception as e:
            logger.warning(f"Failed to parse {file_path.name}: {e}")

    # Sort by name
    commands.sort(key=lambda x: x["name"])

    _slash_commands_cache = commands
    logger.info(f"Discovered {len(commands)} slash commands")

    return commands
```

### 4. Enhance /get_orchestrator Endpoint

**File**: `backend/main.py` (lines 214-241, modify existing endpoint)

- Keep existing orchestrator basic info
- Add `metadata` field to response (return full metadata dict)
- Add `slash_commands` field by calling `discover_slash_commands()`
- Add `agent_templates` field by using SubagentRegistry
- Maintain backward compatibility (all new fields are additive)

**Modified Response Structure**:
```python
@app.get("/get_orchestrator")
async def get_orchestrator_info():
    """
    Get orchestrator agent information including system metadata.

    Returns orchestrator ID, session, costs, metadata, slash commands, and templates.
    """
    try:
        logger.http_request("GET", "/get_orchestrator")

        orchestrator = app.state.orchestrator

        # Discover slash commands
        slash_commands = discover_slash_commands(config.get_working_dir())

        # Get agent templates from SubagentRegistry
        from modules.subagent_loader import SubagentRegistry
        registry = SubagentRegistry(config.get_working_dir(), logger)
        templates = registry.list_templates()

        logger.http_request("GET", "/get_orchestrator", 200)
        return {
            "status": "success",
            "orchestrator": {
                "id": str(orchestrator.id),
                "session_id": orchestrator.session_id,
                "status": orchestrator.status,
                "working_dir": orchestrator.working_dir,
                "input_tokens": orchestrator.input_tokens,
                "output_tokens": orchestrator.output_tokens,
                "total_cost": float(orchestrator.total_cost),
                "metadata": orchestrator.metadata,  # NEW: Include full metadata
            },
            "slash_commands": slash_commands,  # NEW: List of available commands
            "agent_templates": templates,      # NEW: List of available templates
        }
    except Exception as e:
        logger.error(f"Failed to get orchestrator info: {e}")
        raise HTTPException(status_code=500, detail=str(e))
```

### 5. Add TypeScript Type Definitions

**File**: `frontend/src/types.d.ts` (after line 27, after OrchestratorAgent interface)

- Add `SystemMessageInfo` interface matching backend structure
- Add `SlashCommand` interface for command metadata
- Add `SubagentTemplate` and `SubagentFrontmatter` interfaces
- Add these to `GetOrchestratorResponse` interface (new)

**Type Definitions**:
```typescript
// System Message Metadata
export interface SystemMessageInfo {
  session_id?: string | null
  cwd?: string | null
  tools?: string[]
  model?: string | null
  subtype?: string
  captured_at?: string
}

// Slash Command
export interface SlashCommand {
  name: string
  description: string
  arguments?: string
  model?: string
}

// Subagent Template
export interface SubagentFrontmatter {
  name: string
  description: string
  tools?: string[] | null
  model?: string | null
  color?: string | null
}

export interface SubagentTemplate {
  frontmatter: SubagentFrontmatter
  prompt_body: string
  file_path: string
}

// API Response for /get_orchestrator
export interface GetOrchestratorResponse {
  status: string
  orchestrator: OrchestratorAgent
  slash_commands: SlashCommand[]
  agent_templates: SubagentTemplate[]
}
```

### 6. Modify GlobalCommandInput Component Structure

**File**: `frontend/src/components/GlobalCommandInput.vue` (modify template section)

- Add new `<div class="system-info-panel">` below the command input wrapper
- Create 4 info rows using existing CSS patterns from AppHeader.vue
- Use conditional rendering (`v-if`) to only show when data is available
- Pattern: Label on left, value on right (matching AppHeader details-row)

**Template Structure**:
```vue
<template>
  <Transition name="command-input">
    <div v-show="visible" class="global-command-input-container">
      <div class="command-input-wrapper">
        <textarea
          ref="textareaRef"
          v-model="message"
          @keydown.enter.exact.prevent="sendMessage"
          @keydown.escape="handleEscape"
          class="command-input"
          placeholder="Type your command or message..."
          rows="4"
          :disabled="!isConnected"
          aria-label="Command input"
        />
        <div class="command-actions">
          <div class="shortcut-hint">
            Press <kbd>Esc</kbd> to close • <kbd>Enter</kbd> to send • <kbd>Shift+Enter</kbd> for new line
          </div>
          <button
            class="btn-send"
            @click="sendMessage"
            :disabled="!message.trim() || !isConnected"
          >
            <span class="send-icon">▶</span>
            Send
          </button>
        </div>
      </div>

      <!-- NEW: System Information Panel -->
      <div v-if="systemInfo" class="system-info-panel">
        <div class="info-row" v-if="systemInfo.session_id">
          <span class="info-label">Orchestrator Session ID:</span>
          <span class="info-value">{{ systemInfo.session_id }}</span>
        </div>

        <div class="info-row" v-if="systemInfo.cwd">
          <span class="info-label">CWD:</span>
          <span class="info-value">{{ systemInfo.cwd }}</span>
        </div>

        <div class="info-row" v-if="slashCommands.length > 0">
          <span class="info-label">Slash Commands:</span>
          <div class="info-tags">
            <span
              v-for="cmd in slashCommands"
              :key="cmd.name"
              class="badge badge-info"
              :title="cmd.description"
            >
              /{{ cmd.name }}
            </span>
          </div>
        </div>

        <div class="info-row" v-if="agentTemplates.length > 0">
          <span class="info-label">Agent Templates:</span>
          <div class="info-tags">
            <span
              v-for="template in agentTemplates"
              :key="template.frontmatter.name"
              class="badge badge-success"
              :title="template.frontmatter.description"
            >
              {{ template.frontmatter.name }}
            </span>
          </div>
        </div>
      </div>
    </div>
  </Transition>
</template>
```

### 7. Add Data Fetching Logic to GlobalCommandInput

**File**: `frontend/src/components/GlobalCommandInput.vue` (modify script section)

- Import types and chatService
- Add reactive refs for systemInfo, slashCommands, agentTemplates
- Create `fetchOrchestratorInfo()` function
- Call on component mount
- Handle errors gracefully

**Script Additions**:
```typescript
<script setup lang="ts">
import { ref, watch, nextTick, computed, onMounted, onUnmounted } from 'vue'
import { useOrchestratorStore } from '../stores/orchestratorStore'
import { getOrchestratorInfo } from '../services/chatService'
import type { SystemMessageInfo, SlashCommand, SubagentTemplate } from '../types'

// Store
const store = useOrchestratorStore()

// Props
const props = defineProps<{
  visible: boolean
}>()

// Emits
const emit = defineEmits<{
  send: [message: string]
}>()

// Refs
const textareaRef = ref<HTMLTextAreaElement>()
const message = ref('')

// NEW: System info refs
const systemInfo = ref<SystemMessageInfo | null>(null)
const slashCommands = ref<SlashCommand[]>([])
const agentTemplates = ref<SubagentTemplate[]>([])

// Computed
const isConnected = computed(() => store.isConnected)

// NEW: Fetch orchestrator info
const fetchOrchestratorInfo = async () => {
  try {
    const response = await getOrchestratorInfo()

    if (response.status === 'success') {
      // Extract system message info from metadata
      const metadata = response.orchestrator.metadata
      if (metadata?.system_message_info) {
        systemInfo.value = metadata.system_message_info
      }

      // Store slash commands and templates
      slashCommands.value = response.slash_commands || []
      agentTemplates.value = response.agent_templates || []

      console.log('[GlobalCommandInput] Loaded system info:', {
        sessionId: systemInfo.value?.session_id,
        commandCount: slashCommands.value.length,
        templateCount: agentTemplates.value.length
      })
    }
  } catch (error) {
    console.error('[GlobalCommandInput] Failed to fetch orchestrator info:', error)
  }
}

// Methods
const sendMessage = () => {
  if (!message.value.trim() || !isConnected.value) return

  emit('send', message.value.trim())
  message.value = ''

  // Close the command input after sending
  store.hideCommandInput()
}

const handleEscape = () => {
  store.hideCommandInput()
}

// Global escape key handler (works even when not focused on textarea)
const handleGlobalEscape = (event: KeyboardEvent) => {
  if (event.key === 'Escape' && props.visible) {
    handleEscape()
  }
}

// Watch for visibility changes to handle focus
watch(() => props.visible, async (newValue) => {
  if (newValue) {
    // Wait for DOM update then focus
    await nextTick()
    textareaRef.value?.focus()
  }
})

// Register global escape handler and fetch info on mount
onMounted(() => {
  document.addEventListener('keydown', handleGlobalEscape)
  fetchOrchestratorInfo()  // NEW: Fetch on mount
})

// Clean up event listener on unmount
onUnmounted(() => {
  document.removeEventListener('keydown', handleGlobalEscape)
})
</script>
```

### 8. Add Styles for System Info Panel

**File**: `frontend/src/components/GlobalCommandInput.vue` (add to scoped styles section)

- Add `.system-info-panel` styles matching AppHeader details-row pattern
- Add `.info-row` styles for label-value pairs
- Add `.info-label` and `.info-value` styles
- Add `.info-tags` for badge layout
- Use existing badge classes from global.css
- Ensure panel fits within command input container

**Style Additions**:
```vue
<style scoped>
/* Existing transition and container styles... */

.system-info-panel {
  padding: 1rem 1.5rem;
  background: rgba(6, 182, 212, 0.05);
  border-top: 1px solid rgba(6, 182, 212, 0.2);
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.info-row {
  display: flex;
  align-items: flex-start;
  gap: 1rem;
  font-size: 0.875rem;
}

.info-label {
  color: var(--text-secondary);
  font-weight: 500;
  min-width: 180px;
  flex-shrink: 0;
}

.info-value {
  color: var(--text-primary);
  font-family: 'Courier New', monospace;
  word-break: break-all;
  padding-left: 0.5rem;
  border-left: 2px solid rgba(6, 182, 212, 0.3);
}

.info-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  align-items: center;
}

.info-tags .badge {
  cursor: help;
  transition: transform 0.15s ease;
}

.info-tags .badge:hover {
  transform: translateY(-2px);
}

/* Responsive adjustments */
@media (max-width: 768px) {
  .info-row {
    flex-direction: column;
    gap: 0.25rem;
  }

  .info-label {
    min-width: unset;
  }

  .info-value {
    padding-left: 0;
    border-left: none;
    border-top: 2px solid rgba(6, 182, 212, 0.3);
    padding-top: 0.25rem;
  }
}
</style>
```

### 9. Test Backend Data Flow

**Validation Steps**:
- Start orchestrator backend: `cd backend && uv run python main.py`
- Send a test message via `/send_chat` endpoint to trigger orchestrator execution
- Check logs for "✅ Stored SystemMessage data in orchestrator metadata" message
- Query database to verify metadata was stored: `SELECT metadata FROM orchestrator_agents WHERE archived = false;`
- Test `/get_orchestrator` endpoint returns enhanced data with metadata, slash_commands, and agent_templates
- Verify slash commands list contains expected entries from `.claude/commands/`
- Verify agent templates list contains expected entries from `.claude/agents/`

### 10. Test Frontend UI Display

**Validation Steps**:
- Start frontend: `cd frontend && npm run dev`
- Open browser to frontend URL
- Press Cmd+K to open GlobalCommandInput
- Verify system info panel appears below input field
- Verify 4 info rows are displayed (session ID, CWD, slash commands, templates)
- Verify slash command badges are clickable and show tooltips with descriptions
- Verify agent template badges show names and have tooltips with descriptions
- Test responsive layout on mobile view (info rows should stack vertically)
- Verify styles match existing AppHeader pattern

### 11. Add YAML Parser Dependency (if needed)

**File**: `backend/pyproject.toml` (only if yaml not already installed)

- Check if `pyyaml` is already in dependencies
- If not, add: `uv add pyyaml`
- This is needed for parsing slash command frontmatter in step 3

**Validation**:
```bash
cd backend
uv pip list | grep -i yaml
# If not found:
uv add pyyaml
```

### 12. Update CLAUDE.md Documentation

**File**: `apps/orchestrator_3_stream/CLAUDE.md` (add to File-by-File Summary section)

- Document the new database function in backend/modules/database.py
- Document the enhanced /get_orchestrator endpoint in backend/main.py
- Document the modified GlobalCommandInput.vue component
- Document the new TypeScript types in frontend/src/types.d.ts

**Documentation Addition**:
```markdown
**Backend Files:**
- **backend/modules/database.py**: Added `update_orchestrator_metadata()` function for updating orchestrator metadata JSONB field using PostgreSQL merge operator.

**Frontend Files:**
- **frontend/src/components/GlobalCommandInput.vue**: Enhanced to display system information below input field including session ID, CWD, slash commands (as tags), and agent templates (as tags). Fetches data from `/get_orchestrator` on mount.
- **frontend/src/types.d.ts**: Added `SystemMessageInfo`, `SlashCommand`, `SubagentTemplate`, `SubagentFrontmatter`, and `GetOrchestratorResponse` interfaces for type safety.
```

## Testing Strategy

### Unit Tests

1. **Backend Database Function Test**:
   - File: `backend/tests/test_database.py` (create or extend)
   - Test `update_orchestrator_metadata()` successfully merges data
   - Test metadata persists correctly
   - Test JSONB merge doesn't overwrite existing keys

2. **Slash Command Discovery Test**:
   - File: `backend/tests/test_slash_command_discovery.py` (new)
   - Test `discover_slash_commands()` finds all `.md` files
   - Test YAML frontmatter parsing
   - Test handles missing directory gracefully
   - Test cache works correctly

3. **Frontend Type Validation**:
   - Run `npm run type-check` (or equivalent)
   - Verify all new interfaces compile without errors
   - Verify component props and refs use correct types

### Integration Tests

1. **End-to-End Flow Test**:
   - Start orchestrator backend
   - Send first message to trigger SystemMessage capture
   - Query `/get_orchestrator` and verify response structure
   - Start frontend and open GlobalCommandInput
   - Verify all 4 info items display correctly

2. **Edge Cases**:
   - Test with no slash commands directory (should show empty state gracefully)
   - Test with no agent templates (should show empty state gracefully)
   - Test with SystemMessage not yet received (info panel should not crash)
   - Test with very long session IDs (should truncate or wrap)
   - Test with many slash commands (should wrap in UI)

### Playwright Validation

Use playwright-validator agent or MCP server to:
1. Navigate to frontend URL
2. Press Cmd+K shortcut to open GlobalCommandInput
3. Take screenshot of system info panel
4. Verify 4 info rows are visible
5. Verify badges are rendered
6. Hover over badge to verify tooltip appears
7. Test responsive layout by resizing viewport

## Acceptance Criteria

- [ ] SystemMessage data is captured and stored in `orchestrator_agents.metadata` when orchestrator boots
- [ ] `/get_orchestrator` endpoint returns metadata, slash_commands, and agent_templates fields
- [ ] GlobalCommandInput.vue displays 4 info rows below the input field
- [ ] Session ID is displayed and matches the orchestrator's actual session ID
- [ ] CWD is displayed and matches the configured working directory
- [ ] Slash commands are displayed as cyan/teal badge tags with tooltips showing descriptions
- [ ] Agent templates are displayed as green badge tags with tooltips showing descriptions
- [ ] UI styles match existing AppHeader.vue patterns (labels, values, badges)
- [ ] Component is responsive and works on mobile viewports
- [ ] No console errors or warnings
- [ ] TypeScript types are correctly defined and used
- [ ] Backward compatibility maintained (existing API consumers not broken)
- [ ] CLAUDE.md documentation updated with new files and functionality

## Validation Commands

Execute these commands to validate the task is complete:

### Backend Validation

```bash
# 1. Check database function exists
cd backend
grep -n "update_orchestrator_metadata" modules/database.py
# Expected: Function definition found after line 380

# 2. Check SystemMessage handling updated
grep -A 10 "SystemMessage received" modules/orchestrator_service.py | grep "update_orchestrator_metadata"
# Expected: Call to update_orchestrator_metadata found

# 3. Check /get_orchestrator enhanced
grep -A 20 "get_orchestrator_info" main.py | grep -E "(slash_commands|agent_templates)"
# Expected: Both fields present in return statement

# 4. Test endpoint returns correct structure
curl http://localhost:9403/get_orchestrator | python -m json.tool
# Expected: JSON with orchestrator, slash_commands, and agent_templates keys

# 5. Verify slash command discovery
grep -n "discover_slash_commands" main.py
# Expected: Function definition found

# 6. Check YAML dependency
uv pip list | grep -i yaml
# Expected: pyyaml version shown
```

### Frontend Validation

```bash
# 1. Check TypeScript types added
cd frontend
grep -n "SystemMessageInfo\|SlashCommand\|SubagentTemplate" src/types.d.ts
# Expected: All 3 interfaces found

# 2. Check GlobalCommandInput modified
grep -n "system-info-panel" src/components/GlobalCommandInput.vue
# Expected: Class name found in template

# 3. Check data fetching added
grep -n "fetchOrchestratorInfo" src/components/GlobalCommandInput.vue
# Expected: Function definition and onMounted call found

# 4. TypeScript compilation check
npm run type-check
# Expected: No type errors

# 5. Build check
npm run build
# Expected: Build succeeds without errors
```

### Integration Validation

```bash
# 1. Start backend and send test message
# (In terminal 1)
cd backend && uv run python main.py

# (In terminal 2)
curl -X POST http://localhost:9403/send_chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello", "orchestrator_agent_id": "<orchestrator-uuid>"}'

# 2. Check logs for SystemMessage storage
# Look for: "✅ Stored SystemMessage data in orchestrator metadata"

# 3. Query orchestrator info
curl http://localhost:9403/get_orchestrator | python -m json.tool
# Expected: metadata.system_message_info present with session_id, cwd

# 4. Start frontend and test UI
cd frontend && npm run dev
# Open browser, press Cmd+K, verify 4 info rows display
```

### Playwright Validation Script

```typescript
// Save as: playwright-reports/validate-global-command-input.ts
import { test, expect } from '@playwright/test'

test('GlobalCommandInput displays system info', async ({ page }) => {
  await page.goto('http://localhost:5175')

  // Open command input with keyboard shortcut
  await page.keyboard.press('Meta+K')

  // Wait for system info panel
  await expect(page.locator('.system-info-panel')).toBeVisible()

  // Verify 4 info rows
  const infoRows = page.locator('.info-row')
  await expect(infoRows).toHaveCount(4)

  // Verify session ID row
  await expect(page.locator('.info-label').filter({ hasText: 'Orchestrator Session ID' })).toBeVisible()

  // Verify CWD row
  await expect(page.locator('.info-label').filter({ hasText: 'CWD' })).toBeVisible()

  // Verify slash commands row with badges
  await expect(page.locator('.info-label').filter({ hasText: 'Slash Commands' })).toBeVisible()
  const cmdBadges = page.locator('.badge-info')
  await expect(cmdBadges.first()).toBeVisible()

  // Verify agent templates row with badges
  await expect(page.locator('.info-label').filter({ hasText: 'Agent Templates' })).toBeVisible()
  const templateBadges = page.locator('.badge-success')
  await expect(templateBadges.first()).toBeVisible()

  // Take screenshot
  await page.screenshot({ path: 'playwright-reports/global-command-input-system-info.png' })
})
```

## Notes

### Dependencies

- **Backend**: Requires `pyyaml` for YAML frontmatter parsing (add with `uv add pyyaml` if not present)
- **Frontend**: No new dependencies needed (uses existing Vue 3, TypeScript, Pinia)

### Performance Considerations

- Slash command discovery is cached at module level to avoid repeated filesystem scans
- SubagentRegistry already implements caching for template discovery
- Frontend fetches orchestrator info once on mount, not on every visibility toggle
- Consider adding a refresh button if users need to reload system info without page refresh

### Edge Cases Handled

1. **SystemMessage not yet received**: Info panel won't crash, fields will be empty or hidden with `v-if`
2. **Missing directories**: Discovery functions return empty lists gracefully
3. **Malformed YAML**: Individual command files fail gracefully, others still load
4. **Long session IDs**: CSS word-break ensures they don't overflow
5. **Many commands/templates**: Flexbox wrap ensures UI doesn't break

### Future Enhancements (Out of Scope)

- Add "Copy to Clipboard" button for session ID (like AppHeader has)
- Add filtering/search for slash commands and templates
- Add click handler on badges to auto-insert into input field (e.g., clicking `/scout` inserts it)
- Add refresh button to reload system info without page reload
- Add WebSocket broadcast when SystemMessage changes (for real-time updates)
- Add expandable sections for each info row (collapsible to save space)

### Security Considerations

- Slash command discovery only reads from `.claude/commands/` directory (no arbitrary filesystem access)
- YAML parsing uses `yaml.safe_load()` to prevent code injection
- Frontend displays data as text (no innerHTML, prevents XSS)
- Session IDs are displayed but already public to authenticated users

### Backward Compatibility

- All changes to `/get_orchestrator` are additive (new fields added, existing fields unchanged)
- Frontend components that don't use new fields will continue to work
- Database metadata field is optional and non-breaking
- No changes to existing API contracts

---

**Implementation Priority**: **High**

This feature significantly improves user experience by providing visibility into orchestrator capabilities and current state. The 4 information items (session ID, CWD, commands, templates) are essential for users to understand what they can do and where they are working.

**Estimated Time**: 4-6 hours for full implementation and testing

**Risk Level**: **Low**
- All changes are additive and non-breaking
- Leverages existing patterns and components
- Well-defined scope with clear acceptance criteria
