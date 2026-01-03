# Plan: Regex Field Coverage Validation for Agent Logs

## Task Description

Validate that the regex searching functionality works correctly across all important fields for agent_logs in the frontend. Currently, the search only covers `content` and `eventType` fields. This plan will identify all searchable fields, document which fields should be searchable, extend the regex search to cover additional important fields, and validate the implementation.

## Objective

By completing this plan, we will:
1. **Map all available fields** in AgentLog (database) and EventStreamEntry (frontend)
2. **Identify searchable fields** - determine which fields are valuable for regex search
3. **Extend regex coverage** - expand search beyond just `content` and `eventType`
4. **Validate the implementation** - ensure regex search works across all important fields
5. **Document field mapping** - create reference showing database → frontend → searchable fields

## Problem Statement

The current regex search implementation only searches two fields:
- `event.content` - Log message/content
- `event.eventType` - Event type identifier

However, AgentLog has additional searchable fields in the database that are not currently searchable in the UI:
- `task_slug` - Task identifier (useful for filtering by task)
- `session_id` - Session identifier (useful for finding logs from specific session)
- `summary` - Summary text (useful for quick overview search)
- `payload.file_changes[].path` - File paths (useful for finding file-related logs)
- `payload.read_files[].path` - Read file paths (useful for finding file access logs)

This limits users' ability to find logs across all relevant fields.

## Solution Approach

We will extend the regex search to include all important text fields from AgentLog:

1. **Core searchable fields** (always searched):
   - `content` - Main log content
   - `eventType` - Event type identifier
   - `summary` - Event summary

2. **Secondary searchable fields** (search when available):
   - `task_slug` - Task identifier
   - `session_id` - Session identifier
   - File paths from `payload.file_changes[].path`
   - File paths from `payload.read_files[].path`

The new implementation will search across all these fields, allowing users to find logs by content, event type, task, session, or file paths.

## Relevant Files

### Existing Files
- **apps/orchestrator_db/models.py** - AgentLog database model (lines 189-231) - defines all available fields
- **apps/orchestrator_3_stream/frontend/src/types.d.ts** - EventStreamEntry frontend type (lines 174-187)
- **apps/orchestrator_3_stream/frontend/src/composables/useEventStreamFilter.ts** - Current regex filtering (lines 156-172)
- **apps/orchestrator_3_stream/frontend/src/components/FilterControls.vue** - Search UI component

### New Files to Create
- None required - implementation is modification only

## Step by Step Tasks

### 1. Document Current Field Mapping
- Read AgentLog model in models.py (lines 189-231)
- Read EventStreamEntry type in types.d.ts (lines 174-187)
- Create field mapping document showing:
  - Database field name
  - Field type
  - Frontend representation
  - Whether it's currently searchable
  - Recommendation for searchability

### 2. Identify Searchable Fields
- Analyze which fields contain useful text for searching:
  - `content` ✅ (already searched) - Main log message
  - `eventType` ✅ (already searched) - Event identifier
  - `summary` - Short text summary
  - `task_slug` - Useful for finding logs by task
  - `session_id` - Useful for finding logs by session
  - Payload file paths - Useful for finding file-related operations
- Exclude from search:
  - UUIDs (id, agent_id) - search agentName instead
  - Timestamps - use dedicated time filters
  - Numeric fields - use dedicated range filters
  - Booleans - use dedicated toggles

### 3. Update Regex Search Implementation
- Modify `useEventStreamFilter.ts` filteredEvents computed property (around line 162-163)
- Update the regex filtering to search across all identified fields:
  ```typescript
  // Current (2 fields):
  regex.test(event.content) || regex.test(event.eventType || '')

  // Updated (6+ fields):
  regex.test(event.content) ||
  regex.test(event.eventType || '') ||
  regex.test(event.metadata?.summary || '') ||
  // ... additional fields
  ```
- Include fallback for missing fields (null check)
- Maintain case-insensitive matching with 'i' flag

### 4. Add File Path Searching to Payload
- Extract and search file paths from metadata:
  - `payload.file_changes?.map(f => f.path).join(' ')` - create searchable string
  - `payload.read_files?.map(f => f.path).join(' ')` - create searchable string
- Update regex test to include these paths:
  ```typescript
  const filePaths = [
    ...(event.metadata?.file_changes?.map(f => f.path) || []),
    ...(event.metadata?.read_files?.map(f => f.path) || [])
  ].join(' ')
  regex.test(filePaths)
  ```

### 5. Handle Null and Optional Fields
- Ensure regex search safely handles:
  - `event.summary` - can be null
  - `event.task_slug` - can be null
  - `event.session_id` - can be null
  - `event.metadata.file_changes` - may not exist
  - `event.metadata.read_files` - may not exist
- Use optional chaining and provide empty string fallback

### 6. Update Search Placeholder/Documentation
- Update FilterControls.vue search input placeholder to reflect all searchable fields
- Change from: "Search logs (regex supported)"
- Change to: "Search content, type, task, session, files (regex supported)"
- Or: "Search logs by content, event, task, file path (regex supported)"

### 7. Validate Field Accessibility
- Verify all new fields are available in EventStreamEntry
- Check that EventStreamEntry.metadata contains AgentLog.payload
- Verify frontend receives all data from backend API
- Check that field names match database → frontend mapping

### 8. Test Edge Cases
- Empty/null fields should not break regex (covered by null checks)
- Very long file path lists should not cause performance issues
- Special regex characters in file paths should be handled
- Verify fallback to string search still works if regex is invalid

## Acceptance Criteria

✅ **Must Have:**
1. Regex search extends beyond just `content` and `eventType`
2. At minimum add: `summary`, `task_slug`, `session_id` to searchable fields
3. Implement file path searching from payload metadata
4. All null/optional fields handled safely without errors
5. Search results correctly match across all searchable fields
6. Fallback to string search still works for invalid regex

✅ **Should Have:**
1. Search placeholder updated to reflect searchable fields
2. Field mapping documentation created
3. Code comments explain which fields are searchable and why

## Validation Commands

Execute these commands to validate the task is complete:

```bash
# 1. Verify composable syntax is valid
grep -n "regex.test" apps/orchestrator_3_stream/frontend/src/composables/useEventStreamFilter.ts | head -20

# 2. Check that new fields are included in search
grep -n "event.summary\|event.task_slug\|event.session_id\|file_changes\|read_files" \
  apps/orchestrator_3_stream/frontend/src/composables/useEventStreamFilter.ts

# 3. Verify placeholder text is updated
grep -n "Search logs\|regex supported" apps/orchestrator_3_stream/frontend/src/components/FilterControls.vue

# 4. Check field types match between models.py and types.d.ts
diff <(grep -E "^\s*(content|event_type|summary|task_slug|session_id|payload):" \
  apps/orchestrator_db/models.py) \
  <(grep -E "^\s*(content|eventType|summary|task_slug|session_id|payload|metadata):" \
  apps/orchestrator_3_stream/frontend/src/types.d.ts) || echo "Field mapping verified"

# 5. Ensure no console errors during search with regex
# (Manual validation via Playwright or browser console)
```

## Field Coverage Matrix

| Database Field | Frontend Field | Type | Searchable | Priority |
|---|---|---|---|---|
| `id` | `id` | UUID | ❌ No | - |
| `agent_id` | `agentId` | UUID | ❌ No (use agentName filter) | - |
| `session_id` | `metadata.session_id` | string | ✅ **Add** | High |
| `task_slug` | `metadata.task_slug` | string | ✅ **Add** | High |
| `adw_id` | `metadata.adw_id` | string | ✅ **Add** | Medium |
| `adw_step` | `metadata.adw_step` | string | ✅ **Add** | Medium |
| `entry_index` | `metadata.entry_index` | number | ❌ No | - |
| `event_category` | `eventCategory` | enum | ❌ No (has dedicated filter) | - |
| `event_type` | `eventType` | string | ✅ **Already searchable** | High |
| `content` | `content` | string | ✅ **Already searchable** | High |
| `payload` | `metadata` | object | ✅ **Add** (file paths) | Medium |
| `summary` | `metadata.summary` | string | ✅ **Add** | Medium |
| `timestamp` | `timestamp` | datetime | ❌ No (future time filter) | - |

## Implementation Notes

### Current Search Logic (Lines 156-172)
```typescript
if (searchQuery.value.trim()) {
  const query = searchQuery.value.toLowerCase()
  try {
    const regex = new RegExp(query, 'i')
    filtered = filtered.filter(event =>
      regex.test(event.content) || regex.test(event.eventType || '')
    )
  } catch {
    filtered = filtered.filter(event =>
      event.content.toLowerCase().includes(query) ||
      (event.eventType && event.eventType.toLowerCase().includes(query))
    )
  }
}
```

### Proposed Updated Logic
```typescript
if (searchQuery.value.trim()) {
  const query = searchQuery.value.toLowerCase()
  try {
    const regex = new RegExp(query, 'i')
    filtered = filtered.filter(event => {
      // Core fields
      if (regex.test(event.content)) return true
      if (regex.test(event.eventType || '')) return true
      if (regex.test(event.metadata?.summary || '')) return true

      // Task and session fields
      if (regex.test(event.metadata?.task_slug || '')) return true
      if (regex.test(event.metadata?.session_id || '')) return true

      // ADW fields (optional)
      if (regex.test(event.metadata?.adw_id || '')) return true
      if (regex.test(event.metadata?.adw_step || '')) return true

      // File paths from metadata
      const filePaths = [
        ...(event.metadata?.file_changes?.map(f => f.path) || []),
        ...(event.metadata?.read_files?.map(f => f.path) || [])
      ].join(' ')
      if (regex.test(filePaths)) return true

      return false
    })
  } catch {
    // Fallback: string search on all fields
    const searchFields = [
      event.content.toLowerCase(),
      (event.eventType || '').toLowerCase(),
      (event.metadata?.summary || '').toLowerCase(),
      (event.metadata?.task_slug || '').toLowerCase(),
      (event.metadata?.session_id || '').toLowerCase(),
      [
        ...(event.metadata?.file_changes?.map(f => f.path) || []),
        ...(event.metadata?.read_files?.map(f => f.path) || [])
      ].join(' ').toLowerCase()
    ].join(' ')

    if (searchFields.includes(query)) return true
    return false
  }
}
```

### Data Flow
```
AgentLog (DB)
  ├─ content
  ├─ event_type
  ├─ summary
  ├─ task_slug
  ├─ session_id
  └─ payload {file_changes, read_files}
           ↓
EventStreamEntry (Frontend)
  ├─ content
  ├─ eventType
  ├─ metadata {
  │   ├─ summary
  │   ├─ task_slug
  │   ├─ session_id
  │   ├─ file_changes[]
  │   └─ read_files[]
  │ }
           ↓
useEventStreamFilter
  └─ Regex search across all fields above
```

## Notes

### Performance Considerations
- Searching multiple fields adds minimal overhead per event
- File path array flattening (`.join()`) only happens during filter pass
- With typical event size, negligible performance impact
- Could be optimized further with indexed search if performance becomes issue

### Backward Compatibility
- Current search behavior preserved (still searches content + eventType)
- Just adding additional fields to search
- Existing regex patterns will continue to work
- No breaking changes to API or UI

### Future Enhancements (Out of Scope)
- Elasticsearch or full-text search engine for large datasets
- Field-specific search syntax (e.g., `task:my-task content:error`)
- Search result highlighting
- Saved search queries
- Advanced filtering UI with field selection
