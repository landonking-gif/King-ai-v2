# Plan: EventStream UI Implementation with Specialized Log Components

## Task Description

Implement a comprehensive EventStream UI for `apps/orchestrator_3_stream` that displays three distinct types of logs (agent_logs, system_logs, orchestrator_chat) with specialized UI components for each type. The implementation includes:

- Creating an event store (or extending orchestratorStore) to handle HTTP event retrieval via `/get_events` endpoint
- Connecting WebSocket real-time event streaming for live updates
- Building three specialized Vue components for rendering different log types
- Implementing filtering, search, and visual styling based on design guidelines
- Ensuring proper data flow from backend → WebSocket → store → UI components

## Objective

Create a production-ready, real-time event stream UI that elegantly displays all three log types (agent_logs, system_logs, orchestrator_chat) with distinct visual styles, filtering capabilities, and live WebSocket updates, following the design guidelines provided in the reference image.

## Problem Statement

The current EventStream.vue component has placeholder sample data and non-functional filters. The orchestrator system generates three distinct types of events stored in separate database tables (agent_logs, system_logs, orchestrator_chat), but there's no unified UI to display them with appropriate visual differentiation. Users need to see a real-time, color-coded, filterable log stream that clearly distinguishes between agent activities, system events, and orchestrator-to-agent communication.

## Solution Approach

We'll implement a **unified event stream architecture** where:

1. **Backend Layer**: Fix `/get_events` endpoint bug, ensure proper event retrieval and WebSocket broadcasting
2. **Store Layer**: Extend `orchestratorStore` with event stream management actions (fetch history, handle WebSocket events, filtering)
3. **Component Layer**: Create three specialized row components (AgentLogRow, SystemLogRow, OrchestratorChatRow) with unique styling
4. **Integration Layer**: Update EventStream.vue to use real data from store and render specialized components based on event type
5. **Styling Layer**: Implement color-coded visual design following the reference image guidelines

This approach maintains separation of concerns while providing a seamless user experience for monitoring all orchestration activities.

## Relevant Files

### Existing Files to Modify

- **`apps/orchestrator_3_stream/backend/main.py`** (Lines 1-417)
  - Fix missing `uuid` import for `/get_events` endpoint (Line 302 uses `uuid.UUID()`)
  - Current endpoint filters by agent_id and task_slug, returns agent_logs

- **`apps/orchestrator_3_stream/frontend/src/stores/orchestratorStore.ts`** (Lines 1-447)
  - Extend with event stream actions: `fetchEventHistory()`, `addAgentLogEvent()`, `addSystemLogEvent()`, `addOrchestratorChatEvent()`
  - Connect WebSocket callbacks for `agent_log` and `system_log` events
  - Already has `eventStreamEntries` state and `filteredEventStream` computed

- **`apps/orchestrator_3_stream/frontend/src/components/EventStream.vue`** (Lines 1-598)
  - Remove sample data (Lines 170-219)
  - Connect to orchestratorStore for real event data
  - Implement filtering logic (currently non-functional)
  - Use specialized row components based on event source type
  - Connect search functionality

- **`apps/orchestrator_3_stream/frontend/src/services/chatService.ts`** (Lines 1-142)
  - Update `WebSocketCallbacks` interface to include orchestrator_chat event handler
  - Add routing for orchestrator_chat WebSocket messages

- **`apps/orchestrator_3_stream/frontend/src/types.d.ts`** (Lines 1-220)
  - Add new types for event row component props
  - Extend EventStreamEntry if needed for source identification

- **`apps/orchestrator_3_stream/backend/modules/websocket_manager.py`** (Lines 1-243)
  - Verify all event types are broadcast correctly
  - May need to add `broadcast_orchestrator_chat()` method if not present

### New Files to Create

- **`apps/orchestrator_3_stream/frontend/src/components/event-rows/AgentLogRow.vue`**
  - Specialized component for agent_logs table entries
  - Display: line number, level badge, agent name, event content, tokens, timestamp
  - Color: Cyan/teal left border for hooks, green for successful responses

- **`apps/orchestrator_3_stream/frontend/src/components/event-rows/SystemLogRow.vue`**
  - Specialized component for system_logs table entries
  - Display: line number, level badge (DEBUG/INFO/WARNING/ERROR), message, metadata, timestamp
  - Color: Level-specific (purple for DEBUG, blue for INFO, amber for WARNING, red for ERROR)

- **`apps/orchestrator_3_stream/frontend/src/components/event-rows/OrchestratorChatRow.vue`**
  - Specialized component for orchestrator_chat table entries
  - Display: line number, sender/receiver indicators, message content, timestamp
  - Color: Distinct magenta/pink left border to differentiate from other logs
  - Special styling: Speech bubble or chat-like appearance

- **`apps/orchestrator_3_stream/frontend/src/services/eventService.ts`**
  - HTTP service for fetching event history from `/get_events`
  - Type-safe wrapper around apiClient for event operations

## Implementation Phases

### Phase 1: Backend Foundation (1-2 hours)
- Fix `/get_events` endpoint bug
- Verify WebSocket event broadcasting
- Test event retrieval endpoints

### Phase 2: Store Enhancement (2-3 hours)
- Extend orchestratorStore with event stream actions
- Implement HTTP event fetching
- Connect WebSocket event handlers
- Add event normalization logic (convert DB models to EventStreamEntry format)

### Phase 3: Specialized Row Components (4-6 hours)
- Create AgentLogRow.vue with styling
- Create SystemLogRow.vue with styling
- Create OrchestratorChatRow.vue with styling
- Implement shared styles and utilities

### Phase 4: EventStream Integration (2-3 hours)
- Update EventStream.vue to use real store data
- Implement dynamic component rendering based on event type
- Connect filtering and search functionality
- Remove sample data

### Phase 5: Polish & Testing (2-3 hours)
- Refine visual styling to match design guidelines
- Add loading states and error handling
- Implement smooth animations for new events
- Test real-time updates and filtering
- Verify performance with large event lists

## Step by Step Tasks

### 1. Fix Backend `/get_events` Endpoint Bug

**File**: `apps/orchestrator_3_stream/backend/main.py`

- Add `import uuid` to the imports section (after Line 18)
- Verify the endpoint correctly parses `agent_id` query parameter as UUID (Line 302)
- Test endpoint with sample agent_id: `curl "http://127.0.0.1:9403/get_events?agent_id=<uuid>"`

**Rationale**: The endpoint uses `uuid.UUID()` without importing the module, causing a `NameError`.

### 2. Create Event Service Module

**File**: `apps/orchestrator_3_stream/frontend/src/services/eventService.ts` (NEW)

- Import `apiClient` from `./api.ts`
- Define `EventsResponse` interface with `status`, `events`, `count` fields
- Implement `getEvents(params: { agent_id?, task_slug?, limit?, offset? })` function
- Export service functions

**Code Template**:
```typescript
import { apiClient } from './api'
import type { AgentLog, SystemLog } from '../types'

export interface EventsResponse {
  status: string
  events: AgentLog[]
  count: number
}

export interface GetEventsParams {
  agent_id?: string
  task_slug?: string
  limit?: number
  offset?: number
}

export async function getEvents(params: GetEventsParams = {}): Promise<EventsResponse> {
  const response = await apiClient.get<EventsResponse>('/get_events', { params })
  return response.data
}
```

### 3. Extend Type Definitions

**File**: `apps/orchestrator_3_stream/frontend/src/types.d.ts`

- Add `EventSourceType` literal type: `'agent_log' | 'system_log' | 'orchestrator_chat'`
- Update `EventStreamEntry` interface to include `sourceType: EventSourceType` field
- Add component prop interfaces:
  - `AgentLogRowProps`: `event: AgentLog`, `lineNumber: number`
  - `SystemLogRowProps`: `event: SystemLog`, `lineNumber: number`
  - `OrchestratorChatRowProps`: `event: OrchestratorChat`, `lineNumber: number`

**Code Addition**:
```typescript
// Event source identification
export type EventSourceType = 'agent_log' | 'system_log' | 'orchestrator_chat'

// Extended EventStreamEntry
export interface EventStreamEntry {
  id: string
  lineNumber: number
  sourceType: EventSourceType  // NEW: identifies which table/component to use
  level: LogLevel
  agentId?: string
  agentName?: string
  content: string
  tokens?: number
  timestamp: Date
  eventType?: string
  eventCategory?: EventCategory
  metadata?: Record<string, any>
}

// Component prop types
export interface AgentLogRowProps {
  event: AgentLog
  lineNumber: number
}

export interface SystemLogRowProps {
  event: SystemLog
  lineNumber: number
}

export interface OrchestratorChatRowProps {
  event: OrchestratorChat
  lineNumber: number
}
```

### 4. Extend Orchestrator Store with Event Stream Actions

**File**: `apps/orchestrator_3_stream/frontend/src/stores/orchestratorStore.ts`

**Location**: After existing actions (around Line 350)

**Add New Actions**:

```typescript
// ═══════════════════════════════════════════════════════════
// EVENT STREAM ACTIONS
// ═══════════════════════════════════════════════════════════

/**
 * Fetch event history from backend
 */
async function fetchEventHistory(params: {
  agent_id?: string
  task_slug?: string
  limit?: number
  offset?: number
} = {}) {
  try {
    const response = await eventService.getEvents(params)

    // Convert agent logs to EventStreamEntry format
    const entries: EventStreamEntry[] = response.events.map((log, index) => ({
      id: log.id,
      lineNumber: offset + index + 1,
      sourceType: 'agent_log' as EventSourceType,
      level: mapEventCategoryToLevel(log.event_category, log.event_type),
      agentId: log.agent_id,
      content: log.summary || log.content || log.event_type,
      tokens: extractTokensFromPayload(log.payload),
      timestamp: new Date(log.timestamp),
      eventType: log.event_type,
      eventCategory: log.event_category,
      metadata: log.payload
    }))

    // Replace or append based on offset
    if (params.offset === 0) {
      eventStreamEntries.value = entries
    } else {
      eventStreamEntries.value.push(...entries)
    }

    console.log(`Loaded ${entries.length} event history entries`)
  } catch (error) {
    console.error('Failed to fetch event history:', error)
    throw error
  }
}

/**
 * Add agent log event from WebSocket
 */
function addAgentLogEvent(log: AgentLog) {
  const lineNumber = eventStreamEntries.value.length + 1

  const entry: EventStreamEntry = {
    id: log.id,
    lineNumber,
    sourceType: 'agent_log',
    level: mapEventCategoryToLevel(log.event_category, log.event_type),
    agentId: log.agent_id,
    content: log.summary || log.content || log.event_type,
    tokens: extractTokensFromPayload(log.payload),
    timestamp: new Date(log.timestamp),
    eventType: log.event_type,
    eventCategory: log.event_category,
    metadata: log.payload
  }

  eventStreamEntries.value.push(entry)
  console.log(`Added agent log event: ${log.event_type}`)
}

/**
 * Add system log event from WebSocket
 */
function addSystemLogEvent(log: SystemLog) {
  const lineNumber = eventStreamEntries.value.length + 1

  const entry: EventStreamEntry = {
    id: log.id,
    lineNumber,
    sourceType: 'system_log',
    level: log.level,
    content: log.message,
    timestamp: new Date(log.timestamp),
    metadata: log.metadata
  }

  eventStreamEntries.value.push(entry)
  console.log(`Added system log event: ${log.level} - ${log.message}`)
}

/**
 * Add orchestrator chat event from WebSocket
 */
function addOrchestratorChatEvent(chat: OrchestratorChat) {
  const lineNumber = eventStreamEntries.value.length + 1

  const entry: EventStreamEntry = {
    id: chat.id,
    lineNumber,
    sourceType: 'orchestrator_chat',
    level: 'INFO', // Default level for chat messages
    content: chat.message,
    timestamp: new Date(chat.created_at),
    metadata: {
      sender_type: chat.sender_type,
      receiver_type: chat.receiver_type,
      agent_id: chat.agent_id,
      ...chat.metadata
    }
  }

  eventStreamEntries.value.push(entry)
  console.log(`Added orchestrator chat event: ${chat.sender_type} → ${chat.receiver_type}`)
}

// Helper functions
function mapEventCategoryToLevel(category: EventCategory, eventType: string): LogLevel {
  // Hook events are typically INFO unless they indicate errors
  if (category === 'hook') {
    if (eventType.toLowerCase().includes('error')) return 'ERROR'
    return 'INFO'
  }

  // Response events map based on event type
  if (eventType.toLowerCase().includes('error')) return 'ERROR'
  if (eventType.toLowerCase().includes('warn')) return 'WARNING'
  if (eventType.toLowerCase().includes('success')) return 'SUCCESS'
  if (eventType.toLowerCase().includes('debug')) return 'DEBUG'

  return 'INFO'
}

function extractTokensFromPayload(payload: Record<string, any>): number | undefined {
  // Extract token counts from payload if available
  const tokens = payload?.tokens || payload?.input_tokens || payload?.output_tokens
  return tokens ? Number(tokens) : undefined
}
```

**Update WebSocket Connection** (around Line 155):

```typescript
function connectWebSocket() {
  if (wsConnection && wsConnection.readyState === WebSocket.OPEN) {
    console.log('WebSocket already connected')
    return
  }

  const wsUrl = import.meta.env.VITE_WEBSOCKET_URL || 'ws://127.0.0.1:9403/ws'

  wsConnection = chatService.connectWebSocket(wsUrl, {
    onChatStream: handleChatStream,
    onTyping: handleTyping,
    onError: handleWebSocketError,
    onConnected: () => {
      isConnected.value = true
      console.log('✅ WebSocket connected')
    },
    onDisconnected: () => {
      isConnected.value = false
      console.log('❌ WebSocket disconnected')
    },

    // NEW: Event stream handlers
    onAgentLog: (message) => {
      if (message.log) {
        addAgentLogEvent(message.log)
      }
    },
    onSystemLog: (message) => {
      if (message.log) {
        addSystemLogEvent(message.log)
      }
    },
    onOrchestratorChat: (message) => {
      if (message.chat) {
        addOrchestratorChatEvent(message.chat)
      }
    },

    // Existing handlers for agent lifecycle
    onAgentCreated: (message) => {
      if (message.agent) {
        addAgent(message.agent)
      }
    },
    onAgentStatusChange: (message) => {
      if (message.agent_id) {
        updateAgent(message.agent_id, { status: message.new_status })
      }
    }
  })
}
```

**Export New Actions** (at bottom of file, around Line 440):

```typescript
return {
  // Existing exports...
  agents,
  selectedAgentId,
  orchestratorAgentId,
  eventStreamEntries,
  eventStreamFilter,
  chatMessages,
  isTyping,
  currentStreamingMessage,
  isConnected,
  activeAgents,
  runningAgents,
  idleAgents,
  selectedAgent,
  filteredEventStream,
  stats,

  // Existing actions...
  initialize,
  selectAgent,
  clearAgentSelection,
  loadAgents,
  loadChatHistory,
  sendUserMessage,
  addChatMessage,
  clearChat,
  addAgent,
  updateAgent,
  removeAgent,
  clearEventStream,
  setEventStreamFilter,
  exportEventStream,
  addEventStreamEntry,
  connectWebSocket,
  disconnectWebSocket,

  // NEW: Event stream actions
  fetchEventHistory,
  addAgentLogEvent,
  addSystemLogEvent,
  addOrchestratorChatEvent,
}
```

### 5. Update WebSocket Callbacks Interface

**File**: `apps/orchestrator_3_stream/frontend/src/services/chatService.ts`

**Location**: Lines 49-59

**Update Interface**:

```typescript
export interface WebSocketCallbacks {
  onChatStream: (chunk: string, isComplete: boolean) => void
  onTyping: (isTyping: boolean) => void
  onAgentLog?: (log: any) => void
  onSystemLog?: (log: any) => void           // NEW
  onOrchestratorChat?: (chat: any) => void   // NEW
  onAgentCreated?: (agent: any) => void
  onAgentStatusChange?: (data: any) => void
  onError: (error: any) => void
  onConnected?: () => void
  onDisconnected?: () => void
}
```

**Update Message Routing** (Lines 76-110):

```typescript
ws.onmessage = (event) => {
  try {
    const message = JSON.parse(event.data)

    // Route by message type
    switch (message.type) {
      case 'chat_stream':
        callbacks.onChatStream(
          message.chunk || '',
          message.is_complete || false
        )
        break

      case 'chat_typing':
        callbacks.onTyping(message.is_typing || false)
        break

      case 'agent_log':
        callbacks.onAgentLog?.(message)
        break

      case 'system_log':                      // NEW
        callbacks.onSystemLog?.(message)
        break

      case 'orchestrator_chat':               // NEW
        callbacks.onOrchestratorChat?.(message)
        break

      case 'agent_created':
        callbacks.onAgentCreated?.(message)
        break

      case 'agent_status_changed':
        callbacks.onAgentStatusChange?.(message)
        break

      case 'error':
        callbacks.onError(message)
        break

      default:
        console.log('Unknown message type:', message.type)
    }
  } catch (error) {
    console.error('Failed to parse WebSocket message:', error)
  }
}
```

### 6. Create AgentLogRow Component

**File**: `apps/orchestrator_3_stream/frontend/src/components/event-rows/AgentLogRow.vue` (NEW)

```vue
<template>
  <div
    class="event-row agent-log-row"
    :class="[
      `event-${level.toLowerCase()}`,
      `category-${event.event_category}`
    ]"
  >
    <div class="event-line-number">{{ lineNumber }}</div>

    <div class="event-badge" :class="`badge-${level.toLowerCase()}`">
      [{{ level }}]
    </div>

    <div class="event-category-badge" :class="`category-${event.event_category}`">
      {{ event.event_category === 'hook' ? '🪝' : '💬' }}
      {{ event.event_category.toUpperCase() }}
    </div>

    <div class="event-agent">
      <span class="agent-id">Agent-{{ formatAgentId(event.agent_id) }}</span>
      <span v-if="event.task_slug" class="task-slug">{{ event.task_slug }}</span>
    </div>

    <div class="event-content">
      <span class="event-type">{{ event.event_type }}</span>
      <span class="event-summary">{{ event.summary || event.content || '—' }}</span>
    </div>

    <div class="event-meta">
      <span v-if="tokens" class="event-tokens">🪙 {{ tokens }} tokens</span>
      <span class="event-time">{{ formatTime(event.timestamp) }}</span>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import type { AgentLog } from '../../types'

interface Props {
  event: AgentLog
  lineNumber: number
}

const props = defineProps<Props>()

const level = computed(() => {
  // Map event type to display level
  const type = props.event.event_type.toLowerCase()
  if (type.includes('error')) return 'ERROR'
  if (type.includes('warn')) return 'WARN'
  if (type.includes('success')) return 'SUCCESS'
  if (type.includes('debug')) return 'DEBUG'
  return 'INFO'
})

const tokens = computed(() => {
  // Extract tokens from payload
  const payload = props.event.payload
  return payload?.tokens || payload?.input_tokens || payload?.output_tokens
})

function formatAgentId(agentId: string): string {
  // Show last 4 characters of UUID
  return agentId.slice(-4).toUpperCase()
}

function formatTime(timestamp: Date | string): string {
  const date = typeof timestamp === 'string' ? new Date(timestamp) : timestamp
  return date.toLocaleTimeString('en-US', {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false
  })
}
</script>

<style scoped>
.agent-log-row {
  display: grid;
  grid-template-columns: 50px 80px 100px 120px 1fr 180px;
  gap: var(--spacing-md);
  align-items: baseline;
  padding: var(--spacing-sm) var(--spacing-md);
  background: var(--bg-secondary);
  border-left: 3px solid transparent;
  transition: all 0.15s ease;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.875rem;
}

.agent-log-row:hover {
  background: rgba(255, 255, 255, 0.03);
}

/* Line number */
.event-line-number {
  text-align: right;
  color: var(--text-muted);
  opacity: 0.5;
  font-size: 0.8rem;
}

/* Level badges */
.event-badge {
  font-weight: 600;
  text-transform: uppercase;
  font-size: 0.75rem;
}

.badge-info { color: var(--status-info); }
.badge-debug { color: var(--status-debug); }
.badge-success { color: var(--status-success); }
.badge-warn { color: var(--status-warning); }
.badge-error { color: var(--status-error); }

/* Category badge */
.event-category-badge {
  font-size: 0.7rem;
  font-weight: 600;
  padding: 2px 6px;
  border-radius: 4px;
  text-transform: uppercase;
}

.category-hook {
  background: rgba(6, 182, 212, 0.15);
  color: var(--accent);
  border: 1px solid rgba(6, 182, 212, 0.3);
}

.category-response {
  background: rgba(34, 197, 94, 0.15);
  color: var(--status-success);
  border: 1px solid rgba(34, 197, 94, 0.3);
}

/* Agent info */
.event-agent {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.agent-id {
  color: var(--accent-secondary);
  font-weight: 600;
}

.task-slug {
  color: var(--text-muted);
  font-size: 0.7rem;
  opacity: 0.7;
}

/* Content */
.event-content {
  display: flex;
  flex-direction: column;
  gap: 4px;
  min-width: 0;
}

.event-type {
  color: var(--text-secondary);
  font-size: 0.75rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.event-summary {
  color: var(--text-primary);
  word-wrap: break-word;
  overflow-wrap: break-word;
}

/* Metadata */
.event-meta {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  gap: 4px;
  color: var(--text-muted);
  font-size: 0.75rem;
}

.event-tokens {
  color: var(--status-warning);
}

.event-time {
  opacity: 0.7;
}

/* Border colors by level */
.event-info {
  border-left-color: var(--status-info);
}

.event-debug {
  border-left-color: var(--status-debug);
}

.event-success {
  border-left-color: var(--status-success);
}

.event-warn {
  border-left-color: var(--status-warning);
}

.event-error {
  border-left-color: var(--status-error);
  background: rgba(239, 68, 68, 0.05);
}

/* Animation */
@keyframes slideIn {
  from {
    opacity: 0;
    transform: translateX(-10px);
  }
  to {
    opacity: 1;
    transform: translateX(0);
  }
}

.agent-log-row {
  animation: slideIn 0.2s ease-out;
}
</style>
```

### 7. Create SystemLogRow Component

**File**: `apps/orchestrator_3_stream/frontend/src/components/event-rows/SystemLogRow.vue` (NEW)

```vue
<template>
  <div
    class="event-row system-log-row"
    :class="`event-${event.level.toLowerCase()}`"
  >
    <div class="event-line-number">{{ lineNumber }}</div>

    <div class="event-badge" :class="`badge-${event.level.toLowerCase()}`">
      [{{ event.level }}]
    </div>

    <div class="system-icon">
      <span class="icon">{{ getLevelIcon(event.level) }}</span>
      <span class="label">SYSTEM</span>
    </div>

    <div class="event-source" v-if="event.file_path">
      {{ formatFilePath(event.file_path) }}
    </div>

    <div class="event-content">
      {{ event.message }}
    </div>

    <div class="event-meta">
      <span v-if="hasMetadata" class="metadata-indicator" @click="toggleMetadata">
        📋 Metadata
      </span>
      <span class="event-time">{{ formatTime(event.timestamp) }}</span>
    </div>
  </div>

  <!-- Expandable metadata section -->
  <div v-if="showMetadata && hasMetadata" class="metadata-panel">
    <pre>{{ JSON.stringify(event.metadata, null, 2) }}</pre>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import type { SystemLog } from '../../types'

interface Props {
  event: SystemLog
  lineNumber: number
}

const props = defineProps<Props>()

const showMetadata = ref(false)

const hasMetadata = computed(() => {
  return props.event.metadata && Object.keys(props.event.metadata).length > 0
})

function getLevelIcon(level: string): string {
  switch (level) {
    case 'DEBUG': return '🔍'
    case 'INFO': return 'ℹ️'
    case 'WARNING': return '⚠️'
    case 'ERROR': return '❌'
    default: return '📝'
  }
}

function formatFilePath(path: string): string {
  // Show only the filename, not full path
  const parts = path.split('/')
  return parts[parts.length - 1]
}

function formatTime(timestamp: Date | string): string {
  const date = typeof timestamp === 'string' ? new Date(timestamp) : timestamp
  return date.toLocaleTimeString('en-US', {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false
  })
}

function toggleMetadata() {
  showMetadata.value = !showMetadata.value
}
</script>

<style scoped>
.system-log-row {
  display: grid;
  grid-template-columns: 50px 80px 100px 150px 1fr 180px;
  gap: var(--spacing-md);
  align-items: baseline;
  padding: var(--spacing-sm) var(--spacing-md);
  background: var(--bg-secondary);
  border-left: 3px solid transparent;
  transition: all 0.15s ease;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.875rem;
}

.system-log-row:hover {
  background: rgba(255, 255, 255, 0.03);
}

/* Line number */
.event-line-number {
  text-align: right;
  color: var(--text-muted);
  opacity: 0.5;
  font-size: 0.8rem;
}

/* Level badges */
.event-badge {
  font-weight: 600;
  text-transform: uppercase;
  font-size: 0.75rem;
}

.badge-debug { color: var(--status-debug); }
.badge-info { color: var(--status-info); }
.badge-warning { color: var(--status-warning); }
.badge-error { color: var(--status-error); }

/* System indicator */
.system-icon {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 0.7rem;
  font-weight: 600;
  padding: 2px 8px;
  border-radius: 4px;
  background: rgba(147, 51, 234, 0.15);
  color: #a855f7;
  border: 1px solid rgba(147, 51, 234, 0.3);
}

.icon {
  font-size: 1rem;
}

/* Source file */
.event-source {
  color: var(--text-muted);
  font-size: 0.75rem;
  font-style: italic;
  opacity: 0.8;
}

/* Content */
.event-content {
  color: var(--text-primary);
  word-wrap: break-word;
  overflow-wrap: break-word;
}

/* Metadata */
.event-meta {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  gap: 4px;
  color: var(--text-muted);
  font-size: 0.75rem;
}

.metadata-indicator {
  color: var(--accent);
  cursor: pointer;
  text-decoration: underline;
}

.metadata-indicator:hover {
  color: var(--accent-secondary);
}

.event-time {
  opacity: 0.7;
}

/* Metadata panel */
.metadata-panel {
  grid-column: 1 / -1;
  padding: var(--spacing-md);
  margin: 0 var(--spacing-md) var(--spacing-sm) var(--spacing-md);
  background: rgba(0, 0, 0, 0.3);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 4px;
  animation: slideDown 0.2s ease-out;
}

.metadata-panel pre {
  margin: 0;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.75rem;
  color: var(--text-secondary);
  overflow-x: auto;
}

/* Border colors by level */
.event-debug {
  border-left-color: var(--status-debug);
}

.event-info {
  border-left-color: var(--status-info);
}

.event-warning {
  border-left-color: var(--status-warning);
  background: rgba(245, 158, 11, 0.03);
}

.event-error {
  border-left-color: var(--status-error);
  background: rgba(239, 68, 68, 0.05);
}

/* Animations */
@keyframes slideIn {
  from {
    opacity: 0;
    transform: translateX(-10px);
  }
  to {
    opacity: 1;
    transform: translateX(0);
  }
}

@keyframes slideDown {
  from {
    opacity: 0;
    max-height: 0;
  }
  to {
    opacity: 1;
    max-height: 500px;
  }
}

.system-log-row {
  animation: slideIn 0.2s ease-out;
}
</style>
```

### 8. Create OrchestratorChatRow Component

**File**: `apps/orchestrator_3_stream/frontend/src/components/event-rows/OrchestratorChatRow.vue` (NEW)

```vue
<template>
  <div
    class="event-row orchestrator-chat-row"
    :class="`sender-${event.sender_type}`"
  >
    <div class="event-line-number">{{ lineNumber }}</div>

    <div class="communication-badge">
      <span class="icon">💬</span>
      COMM
    </div>

    <div class="communication-flow">
      <span class="sender" :class="`type-${event.sender_type}`">
        {{ formatParticipant(event.sender_type) }}
      </span>
      <span class="arrow">→</span>
      <span class="receiver" :class="`type-${event.receiver_type}`">
        {{ formatParticipant(event.receiver_type) }}
      </span>
    </div>

    <div class="event-content">
      <div class="message-bubble" :class="`from-${event.sender_type}`">
        {{ event.message }}
      </div>
    </div>

    <div class="event-meta">
      <span v-if="event.agent_id" class="agent-ref">
        Agent-{{ formatAgentId(event.agent_id) }}
      </span>
      <span class="event-time">{{ formatTime(event.created_at) }}</span>
    </div>
  </div>
</template>

<script setup lang="ts">
import type { OrchestratorChat } from '../../types'

interface Props {
  event: OrchestratorChat
  lineNumber: number
}

const props = defineProps<Props>()

function formatParticipant(type: 'user' | 'orchestrator' | 'agent'): string {
  switch (type) {
    case 'user': return '👤 User'
    case 'orchestrator': return '🤖 Orchestrator'
    case 'agent': return '⚙️ Agent'
    default: return type
  }
}

function formatAgentId(agentId: string): string {
  return agentId.slice(-4).toUpperCase()
}

function formatTime(timestamp: Date | string): string {
  const date = typeof timestamp === 'string' ? new Date(timestamp) : timestamp
  return date.toLocaleTimeString('en-US', {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false
  })
}
</script>

<style scoped>
.orchestrator-chat-row {
  display: grid;
  grid-template-columns: 50px 80px 200px 1fr 180px;
  gap: var(--spacing-md);
  align-items: center;
  padding: var(--spacing-md);
  background: var(--bg-secondary);
  border-left: 3px solid #d946ef; /* Magenta/pink for communication */
  transition: all 0.15s ease;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.875rem;
}

.orchestrator-chat-row:hover {
  background: rgba(217, 70, 239, 0.05);
}

/* Line number */
.event-line-number {
  text-align: right;
  color: var(--text-muted);
  opacity: 0.5;
  font-size: 0.8rem;
}

/* Communication badge */
.communication-badge {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 0.7rem;
  font-weight: 600;
  padding: 2px 8px;
  border-radius: 4px;
  background: rgba(217, 70, 239, 0.15);
  color: #d946ef;
  border: 1px solid rgba(217, 70, 239, 0.3);
}

.icon {
  font-size: 1rem;
}

/* Communication flow visualization */
.communication-flow {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
  font-size: 0.75rem;
  font-weight: 600;
}

.sender, .receiver {
  padding: 2px 6px;
  border-radius: 3px;
  white-space: nowrap;
}

.type-user {
  background: rgba(59, 130, 246, 0.15);
  color: #3b82f6;
}

.type-orchestrator {
  background: rgba(217, 70, 239, 0.15);
  color: #d946ef;
}

.type-agent {
  background: rgba(6, 182, 212, 0.15);
  color: #06b6d4;
}

.arrow {
  color: var(--text-muted);
  font-weight: normal;
}

/* Message content */
.event-content {
  min-width: 0;
}

.message-bubble {
  padding: var(--spacing-sm) var(--spacing-md);
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.03);
  border: 1px solid rgba(255, 255, 255, 0.1);
  color: var(--text-primary);
  word-wrap: break-word;
  overflow-wrap: break-word;
  line-height: 1.5;
}

.from-user {
  border-left: 3px solid #3b82f6;
}

.from-orchestrator {
  border-left: 3px solid #d946ef;
}

.from-agent {
  border-left: 3px solid #06b6d4;
}

/* Metadata */
.event-meta {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  gap: 4px;
  color: var(--text-muted);
  font-size: 0.75rem;
}

.agent-ref {
  color: var(--accent);
  font-weight: 600;
}

.event-time {
  opacity: 0.7;
}

/* Animation */
@keyframes slideIn {
  from {
    opacity: 0;
    transform: translateX(-10px);
  }
  to {
    opacity: 1;
    transform: translateX(0);
  }
}

.orchestrator-chat-row {
  animation: slideIn 0.2s ease-out;
}
</style>
```

### 9. Update EventStream.vue to Use Specialized Components

**File**: `apps/orchestrator_3_stream/frontend/src/components/EventStream.vue`

**Remove Sample Data** (Delete Lines 170-219):
- Remove the `sampleEvents` array
- Remove the fallback logic in the events computed property

**Import Specialized Components** (Add after Line 91):

```typescript
import { useOrchestratorStore } from '../stores/orchestratorStore'
import AgentLogRow from './event-rows/AgentLogRow.vue'
import SystemLogRow from './event-rows/SystemLogRow.vue'
import OrchestratorChatRow from './event-rows/OrchestratorChatRow.vue'
```

**Connect to Store** (Replace Lines 106-110 with):

```typescript
const store = useOrchestratorStore()
const events = computed(() => store.filteredEventStream)
```

**Update Template Event Rendering** (Replace Lines 62-84 with):

```vue
<div
  v-for="event in filteredEvents"
  :key="event.id"
  class="event-item"
>
  <!-- Render different components based on event source type -->
  <component
    :is="getEventComponent(event)"
    :event="getEventData(event)"
    :line-number="event.lineNumber"
  />
</div>
```

**Add Component Selection Logic** (Add after Line 220):

```typescript
// Filtered events based on search query and quick filters
const filteredEvents = computed(() => {
  let filtered = events.value

  // Apply quick filters
  if (activeQuickFilters.value.size > 0) {
    filtered = filtered.filter(event =>
      activeQuickFilters.value.has(event.level)
    )
  }

  // Apply search query
  if (searchQuery.value.trim()) {
    const query = searchQuery.value.toLowerCase()
    try {
      // Try as regex first
      const regex = new RegExp(query, 'i')
      filtered = filtered.filter(event =>
        regex.test(event.content) || regex.test(event.eventType || '')
      )
    } catch {
      // Fall back to simple string search
      filtered = filtered.filter(event =>
        event.content.toLowerCase().includes(query) ||
        (event.eventType && event.eventType.toLowerCase().includes(query))
      )
    }
  }

  return filtered
})

// Get appropriate component for event type
function getEventComponent(event: EventStreamEntry) {
  switch (event.sourceType) {
    case 'agent_log':
      return AgentLogRow
    case 'system_log':
      return SystemLogRow
    case 'orchestrator_chat':
      return OrchestratorChatRow
    default:
      return AgentLogRow // Fallback
  }
}

// Get event data in correct format for component
function getEventData(event: EventStreamEntry) {
  // EventStreamEntry already contains all necessary data
  // Just return the original event from metadata if available
  return event.metadata?.originalEvent || event
}
```

**Update Filter Tab Logic** (Replace setFilter function):

```typescript
function setFilter(value: string) {
  currentFilter.value = value
  store.setEventStreamFilter(value as EventStreamFilter)
}
```

**Add Load Initial Data** (Add after Line 238):

```typescript
// Load event history on mount
onMounted(async () => {
  try {
    await store.fetchEventHistory({ limit: 100 })
  } catch (error) {
    console.error('Failed to load event history:', error)
  }
})
```

**Add onMounted import** (Update Line 93):

```typescript
import { ref, computed, watch, nextTick, onMounted } from 'vue'
```

### 10. Add Comprehensive Filtering Logic

**File**: `apps/orchestrator_3_stream/frontend/src/stores/orchestratorStore.ts`

**Update filteredEventStream computed** (Lines 60-72):

```typescript
const filteredEventStream = computed(() => {
  switch (eventStreamFilter.value) {
    case 'errors':
      return eventStreamEntries.value.filter(e =>
        e.level === 'ERROR' || e.level === 'WARNING'
      )
    case 'hooks':
      return eventStreamEntries.value.filter(e =>
        e.eventCategory === 'hook'
      )
    case 'responses':
      return eventStreamEntries.value.filter(e =>
        e.eventCategory === 'response'
      )
    case 'all':
    default:
      return eventStreamEntries.value
  }
})
```

### 11. Initialize Event Stream on App Load

**File**: `apps/orchestrator_3_stream/frontend/src/stores/orchestratorStore.ts`

**Update initialize function** (Lines 370-402):

```typescript
async function initialize() {
  console.log('Initializing orchestrator store...')

  // Fetch orchestrator info first to get real UUID
  try {
    const response = await chatService.getOrchestratorInfo()
    orchestratorAgentId.value = response.orchestrator.id
    console.log('Orchestrator ID loaded:', orchestratorAgentId.value)
  } catch (error) {
    console.error('Failed to load orchestrator info:', error)
    return
  }

  // Connect WebSocket for real-time updates
  connectWebSocket()

  // Load agents from API
  try {
    await loadAgents()
  } catch (error) {
    console.error('Failed to load agents:', error)
  }

  // Load chat history
  try {
    await loadChatHistory()
  } catch (error) {
    console.error('Failed to load initial chat history:', error)
  }

  // NEW: Load event stream history
  try {
    await fetchEventHistory({ limit: 100 })
    console.log('Event stream history loaded')
  } catch (error) {
    console.error('Failed to load event stream history:', error)
  }

  console.log('Orchestrator store initialized')
}
```

### 12. Add Missing eventService Import

**File**: `apps/orchestrator_3_stream/frontend/src/stores/orchestratorStore.ts`

**Add import** (After Line 3):

```typescript
import { chatService } from '../services/chatService'
import { agentService } from '../services/agentService'
import { getEvents } from '../services/eventService'  // NEW
import type {
  Agent,
  AgentLog,
  SystemLog,
  OrchestratorChat,
  EventStreamEntry,
  EventStreamFilter,
  EventSourceType,
  ChatMessage,
  AppStats,
} from '../types'
```

### 13. Update Global Styles for Event Rows

**File**: `apps/orchestrator_3_stream/frontend/src/styles/global.css`

**Add at end of file** (after Line 400):

```css
/* ═══════════════════════════════════════════════════════════
   EVENT STREAM ROW STYLES
   ═══════════════════════════════════════════════════════════ */

.event-row {
  font-family: 'JetBrains Mono', monospace;
  border-radius: 4px;
  margin-bottom: 2px;
}

.event-row:last-child {
  margin-bottom: 0;
}

/* Shared event badge styles */
.event-badge {
  display: inline-block;
  font-family: 'JetBrains Mono', monospace;
}

/* Status-specific colors already defined above */
.badge-debug { color: var(--status-debug); }
.badge-info { color: var(--status-info); }
.badge-success { color: var(--status-success); }
.badge-warn { color: var(--status-warning); }
.badge-error { color: var(--status-error); }

/* Scrollbar for metadata panels */
.metadata-panel::-webkit-scrollbar {
  width: 8px;
  height: 8px;
}

.metadata-panel::-webkit-scrollbar-track {
  background: rgba(255, 255, 255, 0.05);
  border-radius: 4px;
}

.metadata-panel::-webkit-scrollbar-thumb {
  background: rgba(255, 255, 255, 0.2);
  border-radius: 4px;
}

.metadata-panel::-webkit-scrollbar-thumb:hover {
  background: rgba(255, 255, 255, 0.3);
}
```

### 14. Test Backend Event Retrieval

**Commands**:

```bash
# Test /get_events endpoint
curl "http://127.0.0.1:9403/get_events?limit=10"

# Test with agent_id filter
curl "http://127.0.0.1:9403/get_events?agent_id=<valid-uuid>&limit=10"

# Test with task_slug filter
curl "http://127.0.0.1:9403/get_events?task_slug=build-feature&limit=10"
```

### 15. Test Frontend Store Integration

**Browser Console**:

```javascript
// In browser dev tools
const store = useOrchestratorStore()

// Test fetch event history
await store.fetchEventHistory({ limit: 20 })

// Check events loaded
console.log(store.eventStreamEntries)

// Test filtering
store.setEventStreamFilter('errors')
console.log(store.filteredEventStream)
```

### 16. Add Empty State Handling

**File**: `apps/orchestrator_3_stream/frontend/src/components/EventStream.vue`

**Update empty state section** (Lines 55-60):

```vue
<div v-if="filteredEvents.length === 0" class="empty-state">
  <div class="empty-icon">📋</div>
  <div class="empty-message">
    {{ searchQuery ? 'No events match your search' : 'No events yet' }}
  </div>
  <div class="empty-hint" v-if="!searchQuery">
    Events will appear here as agents execute tasks
  </div>
</div>
```

### 17. Add Loading State

**File**: `apps/orchestrator_3_stream/frontend/src/components/EventStream.vue`

**Add loading ref** (After Line 138):

```typescript
const isLoading = ref(false)
```

**Add loading state to template** (After Line 52):

```vue
<div v-if="isLoading" class="loading-state">
  <div class="spinner"></div>
  <div>Loading events...</div>
</div>
```

**Update onMounted**:

```typescript
onMounted(async () => {
  isLoading.value = true
  try {
    await store.fetchEventHistory({ limit: 100 })
  } catch (error) {
    console.error('Failed to load event history:', error)
  } finally {
    isLoading.value = false
  }
})
```

**Add loading styles** (Add to EventStream.vue styles):

```css
.loading-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: var(--spacing-md);
  padding: var(--spacing-xl);
  color: var(--text-muted);
}

.spinner {
  width: 40px;
  height: 40px;
  border: 3px solid rgba(255, 255, 255, 0.1);
  border-top-color: var(--accent);
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}
```

### 18. Implement Export Functionality

**File**: `apps/orchestrator_3_stream/frontend/src/stores/orchestratorStore.ts`

**Update exportEventStream action** (Around Line 350):

```typescript
async function exportEventStream(format: 'json' | 'csv' | 'txt' = 'json') {
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-')
  const filename = `event-stream-${timestamp}.${format}`

  let content: string
  let mimeType: string

  switch (format) {
    case 'json':
      content = JSON.stringify(eventStreamEntries.value, null, 2)
      mimeType = 'application/json'
      break

    case 'csv':
      // CSV headers
      const headers = ['Line', 'Source', 'Level', 'Agent', 'Content', 'Timestamp']
      const rows = eventStreamEntries.value.map(entry => [
        entry.lineNumber,
        entry.sourceType,
        entry.level,
        entry.agentId || 'N/A',
        entry.content.replace(/"/g, '""'), // Escape quotes
        entry.timestamp.toISOString()
      ])
      content = [headers, ...rows]
        .map(row => row.map(cell => `"${cell}"`).join(','))
        .join('\n')
      mimeType = 'text/csv'
      break

    case 'txt':
      content = eventStreamEntries.value
        .map(entry => {
          const time = entry.timestamp.toISOString()
          const agent = entry.agentId ? `[${entry.agentId.slice(-4)}]` : '[SYS]'
          return `${entry.lineNumber} ${time} ${agent} [${entry.level}] ${entry.content}`
        })
        .join('\n')
      mimeType = 'text/plain'
      break
  }

  // Create download
  const blob = new Blob([content], { type: mimeType })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  link.click()
  URL.revokeObjectURL(url)

  console.log(`Exported ${eventStreamEntries.value.length} events as ${format}`)
}
```

### 19. Add Keyboard Shortcuts

**File**: `apps/orchestrator_3_stream/frontend/src/components/EventStream.vue`

**Add keyboard handler** (After Line 238):

```typescript
// Keyboard shortcuts
onMounted(() => {
  const handleKeydown = (e: KeyboardEvent) => {
    // Ctrl/Cmd + K: Clear search
    if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
      e.preventDefault()
      searchQuery.value = ''
    }

    // Ctrl/Cmd + E: Toggle error filter
    if ((e.ctrlKey || e.metaKey) && e.key === 'e') {
      e.preventDefault()
      toggleQuickFilter('ERROR')
    }

    // Ctrl/Cmd + D: Toggle debug filter
    if ((e.ctrlKey || e.metaKey) && e.key === 'd') {
      e.preventDefault()
      toggleQuickFilter('DEBUG')
    }

    // Ctrl/Cmd + L: Clear events
    if ((e.ctrlKey || e.metaKey) && e.key === 'l') {
      e.preventDefault()
      onClear()
    }
  }

  window.addEventListener('keydown', handleKeydown)

  onUnmounted(() => {
    window.removeEventListener('keydown', handleKeydown)
  })
})
```

**Add onUnmounted import** (Update Line 93):

```typescript
import { ref, computed, watch, nextTick, onMounted, onUnmounted } from 'vue'
```

### 20. Verify WebSocket Broadcasting

**File**: `apps/orchestrator_3_stream/backend/modules/websocket_manager.py`

**Check if all event types are being broadcast**:

- `broadcast_agent_log()` - ✓ Exists (Line 185)
- `broadcast_system_log()` - ✓ Exists (Line 190)
- `broadcast_orchestrator_chat()` - May need to add if missing

**Add if missing** (After Line 195):

```python
async def broadcast_orchestrator_chat(self, chat_data: dict):
    """Broadcast orchestrator chat event"""
    await self.broadcast({"type": "orchestrator_chat", "chat": chat_data})
```

### 21. Update Backend to Broadcast Chat Events

**File**: `apps/orchestrator_3_stream/backend/modules/orchestrator_service.py`

**Find where orchestrator_chat records are inserted and add broadcast**:

Look for database insert operations like:
```python
await database.insert_orchestrator_chat(...)
```

**Add WebSocket broadcast after insert**:

```python
# After inserting orchestrator chat
chat_data = {
    "id": str(chat_id),
    "orchestrator_agent_id": str(orchestrator_agent_id),
    "sender_type": sender_type,
    "receiver_type": receiver_type,
    "message": message,
    "created_at": datetime.now().isoformat()
}
await self.ws_manager.broadcast_orchestrator_chat(chat_data)
```

### 22. Extend /get_events Endpoint to Support All Event Types

**File**: `apps/orchestrator_3_stream/backend/main.py`

**Problem**: Current endpoint only retrieves agent_logs. We need to also fetch system_logs and orchestrator_chat.

**Update endpoint** (Lines 277-334):

```python
@app.get("/get_events")
async def get_events_endpoint(
    agent_id: Optional[str] = None,
    task_slug: Optional[str] = None,
    event_types: Optional[str] = "all",  # NEW: "all", "agent_logs", "system_logs", "orchestrator_chat"
    limit: int = 50,
    offset: int = 0
):
    """
    Get events from all sources for EventStream component.

    Query params:
        - agent_id: Optional filter by agent UUID
        - task_slug: Optional filter by task
        - event_types: Comma-separated list or "all" (default: "all")
        - limit: Max events to return (default 50)
        - offset: Pagination offset (default 0)

    Returns:
        - status: success/error
        - events: List of unified events with sourceType field
        - count: Total event count
    """
    try:
        logger.http_request("GET", "/get_events")

        # Parse event types
        requested_types = event_types.split(",") if event_types != "all" else ["agent_logs", "system_logs", "orchestrator_chat"]

        all_events = []

        # Fetch agent logs
        if "agent_logs" in requested_types:
            agent_uuid = uuid.UUID(agent_id) if agent_id else None
            if agent_uuid:
                agent_logs = await database.get_agent_logs(
                    agent_id=agent_uuid,
                    task_slug=task_slug,
                    limit=limit,
                    offset=offset
                )
            else:
                agent_logs = await database.list_agent_logs(limit=limit, offset=offset)

            # Add sourceType field
            for log in agent_logs:
                log['sourceType'] = 'agent_log'
                all_events.append(log)

        # Fetch system logs
        if "system_logs" in requested_types:
            system_logs = await database.list_system_logs(limit=limit, offset=offset)
            for log in system_logs:
                log['sourceType'] = 'system_log'
                all_events.append(log)

        # Fetch orchestrator chat
        if "orchestrator_chat" in requested_types:
            chat_logs = await database.list_orchestrator_chat(limit=limit, offset=offset)
            for log in chat_logs:
                log['sourceType'] = 'orchestrator_chat'
                all_events.append(log)

        # Sort by timestamp (most recent first)
        all_events.sort(key=lambda x: x.get('timestamp') or x.get('created_at'), reverse=True)

        # Apply limit after merging
        all_events = all_events[:limit]

        # Convert UUIDs to strings for JSON
        for event in all_events:
            for key, value in event.items():
                if isinstance(value, uuid.UUID):
                    event[key] = str(value)
                elif isinstance(value, datetime):
                    event[key] = value.isoformat()

        logger.http_request("GET", "/get_events", 200)
        return {
            "status": "success",
            "events": all_events,
            "count": len(all_events)
        }

    except Exception as e:
        logger.error(f"Failed to get events: {e}")
        raise HTTPException(status_code=500, detail=str(e))
```

### 23. Add Missing Database Functions

**File**: `apps/orchestrator_3_stream/backend/modules/database.py`

**Add these functions if missing**:

```python
async def list_agent_logs(limit: int = 50, offset: int = 0) -> List[dict]:
    """Get all agent logs across all agents"""
    query = """
        SELECT * FROM agent_logs
        ORDER BY timestamp DESC
        LIMIT $1 OFFSET $2
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(query, limit, offset)
        return [dict(row) for row in rows]

async def list_system_logs(limit: int = 50, offset: int = 0) -> List[dict]:
    """Get system logs"""
    query = """
        SELECT * FROM system_logs
        ORDER BY timestamp DESC
        LIMIT $1 OFFSET $2
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(query, limit, offset)
        return [dict(row) for row in rows]

async def list_orchestrator_chat(limit: int = 50, offset: int = 0) -> List[dict]:
    """Get orchestrator chat logs"""
    query = """
        SELECT * FROM orchestrator_chat
        ORDER BY created_at DESC
        LIMIT $1 OFFSET $2
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(query, limit, offset)
        return [dict(row) for row in rows]
```

### 24. Update Event Service to Handle Mixed Event Types

**File**: `apps/orchestrator_3_stream/frontend/src/services/eventService.ts`

**Update to handle unified response**:

```typescript
import { apiClient } from './api'
import type { AgentLog, SystemLog, OrchestratorChat } from '../types'

export interface UnifiedEvent {
  sourceType: 'agent_log' | 'system_log' | 'orchestrator_chat'
  [key: string]: any
}

export interface EventsResponse {
  status: string
  events: UnifiedEvent[]
  count: number
}

export interface GetEventsParams {
  agent_id?: string
  task_slug?: string
  event_types?: string  // "all" or "agent_logs,system_logs,orchestrator_chat"
  limit?: number
  offset?: number
}

export async function getEvents(params: GetEventsParams = {}): Promise<EventsResponse> {
  const response = await apiClient.get<EventsResponse>('/get_events', { params })
  return response.data
}
```

### 25. Update Store to Handle Mixed Event Types from API

**File**: `apps/orchestrator_3_stream/frontend/src/stores/orchestratorStore.ts`

**Update fetchEventHistory function** (Replace lines 229-265):

```typescript
async function fetchEventHistory(params: {
  agent_id?: string
  task_slug?: string
  event_types?: string
  limit?: number
  offset?: number
} = {}) {
  try {
    const response = await getEvents(params)

    // Convert mixed events to EventStreamEntry format
    const entries: EventStreamEntry[] = response.events.map((event, index) => {
      const baseEntry = {
        id: event.id,
        lineNumber: (params.offset || 0) + index + 1,
        sourceType: event.sourceType,
        timestamp: new Date(event.timestamp || event.created_at)
      }

      // Handle different event types
      switch (event.sourceType) {
        case 'agent_log':
          return {
            ...baseEntry,
            level: mapEventCategoryToLevel(event.event_category, event.event_type),
            agentId: event.agent_id,
            content: event.summary || event.content || event.event_type,
            tokens: extractTokensFromPayload(event.payload),
            eventType: event.event_type,
            eventCategory: event.event_category,
            metadata: { ...event.payload, originalEvent: event }
          } as EventStreamEntry

        case 'system_log':
          return {
            ...baseEntry,
            level: event.level,
            content: event.message,
            metadata: { ...event.metadata, originalEvent: event }
          } as EventStreamEntry

        case 'orchestrator_chat':
          return {
            ...baseEntry,
            level: 'INFO',
            content: event.message,
            metadata: {
              sender_type: event.sender_type,
              receiver_type: event.receiver_type,
              agent_id: event.agent_id,
              ...event.metadata,
              originalEvent: event
            }
          } as EventStreamEntry

        default:
          return baseEntry as EventStreamEntry
      }
    })

    // Replace or append based on offset
    if (params.offset === 0) {
      eventStreamEntries.value = entries
    } else {
      eventStreamEntries.value.push(...entries)
    }

    console.log(`Loaded ${entries.length} event history entries`)
  } catch (error) {
    console.error('Failed to fetch event history:', error)
    throw error
  }
}
```

### 26. Playwright MCP Testing Guide

**Use the Playwright MCP tool to validate the EventStream implementation with these natural language prompts:**

#### Test 1: Initial App Load and EventStream Display

**Prompt for Playwright MCP**:
```
Navigate to http://localhost:5173 and wait 2 seconds for the app to initialize.
Take a screenshot called "01-initial-load.png".
Verify that the EventStream component is visible in the center of the screen.
Count how many event items are displayed and report the number.
```

**Expected Result**: EventStream component is visible, may show 0 or more initial events.

---

#### Test 2: Send Command and Watch Events Stream In (CRITICAL TEST)

**Prompt for Playwright MCP**:
```
Navigate to http://localhost:5173 and wait 2 seconds for initialization.

Take a screenshot called "02-before-command.png".

In the orchestrator chat input on the right side, type this message:
"Create an agent called test-agent-001 for testing"

Click the Send button.

Wait for the typing indicator to appear, then wait for it to disappear (max 30 seconds).

Wait an additional 2 seconds for WebSocket events to stream in.

Take a screenshot called "03-after-command.png".

Count the following:
- Elements with class "agent-log-row" (should have cyan or teal left border)
- Elements with class "system-log-row" (should have purple SYSTEM badge)
- Elements with class "orchestrator-chat-row" (should have magenta left border)

Report the counts for each type and verify at least some events appeared.
```

**Expected Result**: Multiple event rows appear in the EventStream with distinct styling for each type.

---

#### Test 3: Filter Events by Error Level

**Prompt for Playwright MCP**:
```
Navigate to http://localhost:5173 and wait 2 seconds.

Count the total number of elements with class "event-row".

Take a screenshot called "04-before-filter.png".

Click on the quick filter button labeled "ERR" (in the EventStream header).

Wait 500ms for filtering to apply.

Take a screenshot called "05-after-filter.png".

Count how many elements with class "event-row" are still visible.

Verify that all visible events have class "event-error".

Report both counts and confirm filtering worked.
```

**Expected Result**: Only ERROR level events remain visible after clicking ERR filter.

---

#### Test 4: Search Events

**Prompt for Playwright MCP**:
```
Navigate to http://localhost:5173 and wait 2 seconds.

Find the search input in the EventStream component (class "search-input").

Type "agent" into the search box.

Wait 500ms for search to apply.

Take a screenshot called "06-search-results.png".

Count how many event rows are visible.

Get the text content of the first visible event row and verify it contains the word "agent".

Report the search results.
```

**Expected Result**: Only events containing "agent" in their content are displayed.

---

#### Test 5: Verify Three Distinct Event Row Types

**Prompt for Playwright MCP**:
```
Navigate to http://localhost:5173 and wait 2 seconds.

In the orchestrator chat, type: "List all agents and report their status"

Click Send and wait 5 seconds for events to stream in.

Take a screenshot called "07-all-event-types.png".

For each event type, verify the styling:

1. Agent log rows (.agent-log-row):
   - Should have a category badge showing "HOOK" or "RESPONSE"
   - Should show agent ID like "Agent-XXXX"
   - Left border should be blue/cyan/teal color

2. System log rows (.system-log-row):
   - Should have a purple "SYSTEM" badge
   - Should have an icon (🔍 🔧 ℹ️ ⚠️ or ❌)
   - Left border color based on level

3. Orchestrator chat rows (.orchestrator-chat-row):
   - Should have a "COMM" badge
   - Should show communication flow like "👤 User → 🤖 Orchestrator"
   - Should have magenta/pink left border (#d946ef)

Report if all three types are present and correctly styled.
```

**Expected Result**: All three event types visible with distinct visual styling.

---

#### Test 6: Auto-Scroll to New Events

**Prompt for Playwright MCP**:
```
Navigate to http://localhost:5173 and wait 2 seconds.

Use JavaScript to scroll the event stream container (.event-content-wrapper) to the top.

Get the current scroll position and save it.

In the orchestrator chat, type: "Report system status"

Click Send and wait 3 seconds.

Get the new scroll position.

Take a screenshot called "08-auto-scroll.png".

Verify that the scroll position increased (scrolled down toward new events).

Report both scroll positions.
```

**Expected Result**: Event stream automatically scrolls down when new events arrive.

---

#### Test 7: Filter Tabs

**Prompt for Playwright MCP**:
```
Navigate to http://localhost:5173 and wait 2 seconds.

Take a screenshot called "09-combined-stream.png" showing all events.

Click on the "Errors Only" filter tab.

Wait 500ms.

Take a screenshot called "10-errors-only.png".

Count how many events are visible and verify they all have ERROR or WARNING level.

Click on the "Combined Stream" tab to return to all events.

Report the difference in event counts.
```

**Expected Result**: "Errors Only" tab shows subset of events, only errors and warnings.

---

### 27. Playwright MCP Testing Summary

**To execute all tests, use the Playwright MCP tool with each prompt above in sequence.**

**Key Validation Points**:
- ✅ EventStream component renders and displays events
- ✅ Sending orchestrator commands generates events that stream into the UI
- ✅ All three event types (agent_log, system_log, orchestrator_chat) display with distinct styling
- ✅ Filtering by level works correctly
- ✅ Search functionality filters events
- ✅ Auto-scroll follows new events
- ✅ Filter tabs change displayed events

**Screenshot Outputs**:
1. `01-initial-load.png` - App on first load
2. `02-before-command.png` - Before sending command
3. `03-after-command.png` - After events stream in
4. `04-before-filter.png` - Before applying error filter
5. `05-after-filter.png` - After error filter applied
6. `06-search-results.png` - Search results
7. `07-all-event-types.png` - All three event row types visible
8. `08-auto-scroll.png` - Auto-scroll behavior
9. `09-combined-stream.png` - All events visible
10. `10-errors-only.png` - Only errors visible

## Testing Strategy

### Unit Tests

1. **Store Actions**:
   - Test `fetchEventHistory()` with various filter parameters
   - Test `addAgentLogEvent()`, `addSystemLogEvent()`, `addOrchestratorChatEvent()`
   - Test event normalization and line number assignment
   - Test filtering logic in `filteredEventStream` computed

2. **Components**:
   - Test AgentLogRow renders correct data and styles
   - Test SystemLogRow metadata expansion
   - Test OrchestratorChatRow participant formatting
   - Test EventStream component selection logic

3. **Services**:
   - Test eventService API calls
   - Test WebSocket message routing
   - Test error handling

### Integration Tests

1. **Backend to Frontend Flow**:
   - Start backend and frontend
   - Create test agent
   - Command agent to execute task
   - Verify agent_logs appear in EventStream UI in real-time
   - Verify system_logs appear for system events
   - Verify orchestrator_chat appears for orchestrator commands

2. **Filtering**:
   - Apply each filter tab (Combined Stream, Errors Only, Performance)
   - Toggle quick filters (DBG, INF, WARN, ERR, OK)
   - Enter search query
   - Verify filtered results are correct

3. **Real-time Updates**:
   - Open two browser tabs
   - Trigger events from one tab
   - Verify events appear in both tabs simultaneously

### Manual Testing Checklist

- [ ] Backend `/get_events` endpoint returns data without errors
- [ ] Frontend loads event history on mount
- [ ] WebSocket connects successfully
- [ ] Real-time events appear in UI as they occur
- [ ] AgentLogRow displays correctly with proper styling
- [ ] SystemLogRow displays correctly with metadata expansion
- [ ] OrchestratorChatRow displays correctly with participant badges
- [ ] Filter tabs work (all, errors, performance)
- [ ] Quick filters work (DBG, INF, WARN, ERR, OK)
- [ ] Search query filters events
- [ ] Regex search works
- [ ] Auto-scroll works for new events
- [ ] Clear button clears events
- [ ] Export button exports events (JSON/CSV/TXT)
- [ ] Keyboard shortcuts work (Ctrl+E, Ctrl+D, Ctrl+L, Ctrl+K)
- [ ] Loading state displays during initial load
- [ ] Empty state displays when no events
- [ ] Performance is acceptable with 1000+ events

## Acceptance Criteria

1. **Event Display**:
   - ✓ All three log types (agent_logs, system_logs, orchestrator_chat) display in unified stream
   - ✓ Each log type has visually distinct styling (color-coded borders, badges, layouts)
   - ✓ Line numbers increment correctly across all event types

2. **Data Flow**:
   - ✓ HTTP `/get_events` endpoint retrieves historical events
   - ✓ WebSocket broadcasts real-time events
   - ✓ Store manages event state and filtering
   - ✓ Components receive correct event data via props

3. **Filtering**:
   - ✓ Tab filters work (Combined Stream, Errors Only, Performance)
   - ✓ Quick filters work (DBG, INF, WARN, ERR, OK)
   - ✓ Search query filters by content and event type
   - ✓ Regex search supported

4. **User Experience**:
   - ✓ Real-time updates appear smoothly
   - ✓ Auto-scroll follows new events
   - ✓ Loading states prevent confusion
   - ✓ Empty states provide helpful messages
   - ✓ Export functionality works for all formats
   - ✓ Keyboard shortcuts improve efficiency

5. **Visual Design**:
   - ✓ Follows design guidelines from reference image
   - ✓ Color-coded by event level and type
   - ✓ Monospace font for technical content
   - ✓ Responsive layout adapts to different screen sizes
   - ✓ Smooth animations for new events

## Validation Commands

Execute these commands to validate the implementation:

### Backend Validation

```bash
# 1. Check backend starts without errors
cd apps/orchestrator_3_stream/backend
uv run python main.py

# 2. Test /get_events endpoint
curl "http://127.0.0.1:9403/get_events?limit=10" | jq

# 3. Test with filters
curl "http://127.0.0.1:9403/get_events?limit=5&offset=0" | jq '.events | length'

# Expected: Returns JSON with events array
```

### Frontend Validation

```bash
# 1. Check frontend builds without errors
cd apps/orchestrator_3_stream/frontend
npm run build

# Expected: Build completes successfully

# 2. Check TypeScript compilation
npm run type-check

# Expected: No type errors

# 3. Start dev server
npm run dev

# Expected: Server starts on http://localhost:5173
```

### Browser Validation

```javascript
// Open browser console at http://localhost:5173

// 1. Check store initialization
const store = useOrchestratorStore()
console.log('Events loaded:', store.eventStreamEntries.length)
// Expected: Number of events > 0

// 2. Check WebSocket connection
console.log('WebSocket connected:', store.isConnected)
// Expected: true

// 3. Test filtering
store.setEventStreamFilter('errors')
console.log('Filtered events:', store.filteredEventStream.length)
// Expected: Only ERROR/WARNING level events

// 4. Test event addition
store.addSystemLogEvent({
  id: crypto.randomUUID(),
  level: 'INFO',
  message: 'Test system log',
  timestamp: new Date(),
  metadata: {}
})
console.log('New event added:', store.eventStreamEntries[store.eventStreamEntries.length - 1])
// Expected: New event appears in list
```

### Integration Validation

```bash
# Start both backend and frontend
cd apps/orchestrator_3_stream

# Terminal 1: Start backend
./start_be.sh

# Terminal 2: Start frontend
./start_fe.sh

# Browser: Open http://localhost:5173
# Expected:
# - EventStream shows real events
# - Three different row component styles visible
# - Filters work correctly
# - New events appear in real-time via WebSocket
```

## Notes

### Import Additions Required

**orchestratorStore.ts**:
```typescript
import { getEvents } from '../services/eventService'
```

**EventStream.vue**:
```typescript
import { onMounted, onUnmounted } from 'vue'
import AgentLogRow from './event-rows/AgentLogRow.vue'
import SystemLogRow from './event-rows/SystemLogRow.vue'
import OrchestratorChatRow from './event-rows/OrchestratorChatRow.vue'
```

**main.py**:
```python
import uuid
```

### Color Variables Reference

Use these CSS variables for consistent styling:

```css
--accent: #06b6d4          /* Cyan - primary accent */
--accent-secondary: #14b8a6 /* Teal - secondary accent */
--status-info: #3b82f6     /* Blue */
--status-debug: #a855f7    /* Purple */
--status-success: #22c55e  /* Green */
--status-warning: #f59e0b  /* Amber */
--status-error: #ef4444    /* Red */
--text-primary: #f8fafc    /* Primary text */
--text-secondary: #cbd5e1  /* Secondary text */
--text-muted: #64748b      /* Muted text */
--bg-primary: #0a0a0a      /* Primary background */
--bg-secondary: #1a1a1a    /* Secondary background */
```
