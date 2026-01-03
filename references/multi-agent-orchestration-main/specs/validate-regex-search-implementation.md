# Plan: Validate Regex Search Implementation for Agent Logs

## Task Description

Validate and test the regex searching functionality in the orchestrator frontend for agent_logs. The frontend currently implements regex search on the `content` and `eventType` fields through the `useEventStreamFilter` composable. This plan will comprehensively validate the implementation, create test cases, verify edge cases, and ensure everything works correctly across different scenarios.

## Objective

By completing this plan, we will:
1. **Understand** the current regex search implementation thoroughly
2. **Create comprehensive test cases** covering normal, edge case, and error scenarios
3. **Validate** that the regex search works correctly on all searchable fields
4. **Document** findings, limitations, and recommendations
5. **Implement** unit tests for the filtering logic
6. **Test** the UI with Playwright to ensure search functionality works end-to-end
7. **Identify** any bugs, security concerns (ReDoS), or performance issues

## Problem Statement

The frontend regex search implementation lacks formal test coverage and validation. There are no unit tests, no documented test cases, and no systematic validation of:
- Regex pattern handling (valid, invalid, edge cases)
- Searchable field coverage (only content + eventType)
- Fallback behavior when regex fails
- Performance with large datasets
- Security considerations (ReDoS attacks)
- UI integration and user feedback

This creates risk of undetected bugs and performance issues in production.

## Solution Approach

We will implement a three-phase validation strategy:

### Phase 1: Test Case Design & Documentation
- Create comprehensive test case matrix covering regex patterns, edge cases, and field coverage
- Document expected behavior for each test case
- Identify current limitations and security concerns

### Phase 2: Unit Test Implementation
- Create Jest/Vitest test file for `useEventStreamFilter.ts`
- Implement test cases for regex validation, fallback behavior, and filtering logic
- Test edge cases (invalid regex, empty queries, special characters, etc.)

### Phase 3: E2E Testing & UI Validation
- Create Playwright tests to validate UI search functionality
- Test end-to-end search workflow (user enters query → results filter → display)
- Validate error handling and fallback behavior in UI

## Relevant Files

### Existing Files to Review/Test
- **src/composables/useEventStreamFilter.ts** - Core regex filtering logic (lines 156-172 contain regex implementation)
- **src/components/FilterControls.vue** - Search input UI component
- **src/components/EventStream.vue** - Event stream display and filtering integration
- **src/types.d.ts** - TypeScript type definitions for EventStreamEntry
- **src/data/testData.ts** - Sample test data with agent logs
- **src/stores/orchestratorStore.ts** - Pinia store state management

### New Files to Create
- **frontend/src/composables/__tests__/useEventStreamFilter.test.ts** - Unit tests for filter composable
- **frontend/e2e/search-validation.spec.ts** - Playwright E2E tests for search UI
- **VALIDATION_REPORT.md** - Comprehensive validation report with findings and recommendations

## Implementation Phases

### Phase 1: Foundation & Test Case Design
- Extract and document the regex search implementation
- Create comprehensive test case matrix
- Identify all searchable fields and expected behavior
- Document edge cases and limitations
- Create test data sets for various scenarios

### Phase 2: Unit Testing
- Set up Jest/Vitest configuration (if not already present)
- Implement unit tests for regex filtering logic
- Test regex pattern validation and error handling
- Test field filtering (content vs eventType)
- Test fallback to substring search
- Validate filter chain integration

### Phase 3: UI & E2E Validation
- Create Playwright tests for search input UI
- Test regex pattern entry and submission
- Validate filtered results display correctly
- Test error handling and fallback behavior in UI
- Test integration with other filter types
- Validate performance with large datasets

## Step by Step Tasks
IMPORTANT: Execute every step in order, top to bottom.

### 1. Analyze Current Implementation & Document Findings
- Read and analyze `useEventStreamFilter.ts` to understand regex implementation
- Identify searchable fields: `content` and `eventType`
- Document the try-catch fallback mechanism
- List current limitations (no validation feedback, ReDoS risk, limited fields)
- Create comparison matrix of searchable vs non-searchable fields

### 2. Design Comprehensive Test Case Matrix
- Create test cases for valid regex patterns (string search, wildcards, character classes, alternation, quantifiers)
- Create test cases for invalid regex patterns (unclosed brackets, invalid groups, etc.)
- Create edge cases (empty query, whitespace, special characters, very long patterns)
- Create test cases for field coverage (content field, eventType field, combined)
- Create test cases for fallback behavior (invalid regex → string search)
- Document expected behavior and output for each test case

### 3. Create Structured Test Data
- Generate diverse sample logs covering different event types and content patterns
- Create test data sets for:
  - Error messages (with "error" keyword)
  - Tool use events (with event type names)
  - Database operations (with technical keywords)
  - Agent lifecycle events (initialization, completion)
  - Mixed case content (verify case-insensitive matching)
  - Content with special regex characters (escaped dots, pipes, etc.)

### 4. Set Up Unit Testing Infrastructure
- Check if Jest/Vitest is already configured in the frontend project
- If not, add testing dependencies: `uv add --group dev vitest @vue/test-utils`
- Create `frontend/src/composables/__tests__/` directory
- Create `useEventStreamFilter.test.ts` test file
- Set up test data imports and mock data structures

### 5. Implement Unit Tests for Regex Filtering
- Test 1: Simple string search (case-insensitive)
  - Input: "database", Content: "Failed to connect to database"
  - Expected: Should match
- Test 2: Invalid regex fallback
  - Input: "[abc" (unclosed bracket)
  - Expected: Should fall back to string search
- Test 3: Regex special characters
  - Input: "\.txt", Content: "reading file.txt"
  - Expected: Should match using regex escape
- Test 4: Field filtering - content only
  - Input: regex pattern, verify only content and eventType are searched
  - Expected: Should match only in those fields, not in other properties
- Test 5: Empty/whitespace query
  - Input: "", "   "
  - Expected: Should not filter (no search applied)
- Test 6: Case-insensitive matching
  - Input: "ERROR", Content: "An error occurred"
  - Expected: Should match despite case difference
- Test 7: Alternation patterns
  - Input: "initialize|complete", Content: "task_complete"
  - Expected: Should match using alternation

### 6. Implement Unit Tests for Edge Cases & Error Handling
- Test 8: Very long regex pattern
  - Input: Extremely long regex string
  - Expected: Should either match or gracefully handle without hanging
- Test 9: Multiline content
  - Input: regex with "." expecting newline matching
  - Expected: Should handle according to current limitation (. doesn't match \n)
- Test 10: Special regex characters in content
  - Input: User searching for literal special chars
  - Expected: Should handle escaping correctly
- Test 11: Multiple consecutive spaces in query
  - Input: "  db  "
  - Expected: Should trim and handle appropriately
- Test 12: Unicode and special characters
  - Input: regex with unicode patterns
  - Expected: Should match if supported by JavaScript RegExp

### 7. Implement Unit Tests for Integration with Other Filters
- Test 13: Agent filter + regex search
  - Apply agent filter AND search query
  - Expected: Should apply both filters in series
- Test 14: Category filter + regex search
  - Apply category filter AND search query
  - Expected: Should apply both filters in series
- Test 15: Level filter + regex search
  - Apply quick filter by level AND search query
  - Expected: Should apply both filters in series

### 8. Create Playwright E2E Test File
- Create `frontend/e2e/search-validation.spec.ts`
- Set up Playwright fixtures and page navigation
- Configure test data loading and component mounting
- Create test utilities for search input interaction

### 9. Implement Playwright E2E Tests - Basic Workflow
- Test 1: User enters simple search query
  - Navigate to EventStream component
  - Enter "database" in search input
  - Verify results filter to show only matching events
- Test 2: User enters regex pattern
  - Enter "error|fail" in search input
  - Verify results show events matching either pattern
- Test 3: User clears search
  - Enter query, then clear it
  - Verify all events display again

### 10. Implement Playwright E2E Tests - Error Handling
- Test 4: User enters invalid regex
  - Enter "[invalid" (unclosed bracket)
  - Verify component doesn't crash
  - Verify fallback to string search works (if "[invalid" matches any content)
- Test 5: User enters empty query
  - Enter empty string or spaces
  - Verify no filtering applied

### 11. Implement Playwright E2E Tests - UI Integration
- Test 6: Search combines with other filters
  - Apply agent filter
  - Apply search query
  - Verify both filters work together
- Test 7: Search results auto-scroll
  - Enter search that matches items
  - Verify auto-scroll enabled and follows filtered events
- Test 8: Filter controls display search hint
  - Verify input placeholder says "Search logs (regex supported)"
  - Verify UI clearly indicates regex is supported

### 12. Run All Tests & Collect Results
- Execute unit tests: `uv run vitest run`
- Collect test results (pass/fail counts, coverage)
- Execute Playwright tests: `uv run playwright test frontend/e2e/search-validation.spec.ts`
- Document any failures or unexpected behavior
- Capture performance metrics if possible

### 13. Create Validation Report
- Document test results summary (passed/failed/skipped)
- List all identified bugs or issues
- Document limitations found during testing
- Identify ReDoS vulnerabilities or performance concerns
- Create matrix of searchable vs non-searchable fields
- Document regex pattern coverage tested
- List recommendations for improvements

### 14. Validate Work & Create Recommendations Document
- Verify all test cases passed or documented as expected limitations
- Create list of bugs to fix (if any found)
- Create list of improvements for future work:
  - Add regex validation feedback to UI
  - Extend searchable fields to include agent names
  - Implement ReDoS protection
  - Add search history/suggestions
  - Add search result highlighting
- Document security considerations and mitigation strategies
- Final check: All searchable fields identified and tested

## Testing Strategy

### Unit Testing Approach
- **Framework**: Vitest (with Vue Test Utils if needed)
- **Coverage**: Regex pattern matching, error handling, field filtering, fallback behavior
- **Assertions**: Test expected filter results against known data sets
- **Edge Cases**: Invalid regex, empty queries, special characters, unicode, multiline content

### E2E Testing Approach
- **Framework**: Playwright MCP
- **Scenarios**: User search workflow, error handling in UI, integration with other filters
- **Validation**: Visual verification of filtered results, UI state changes, component behavior
- **Devices**: Test on desktop viewport (could extend to mobile if needed)

### Test Data Strategy
- Use structured test data from `testData.ts` as base
- Create additional test data sets for:
  - Various event types (error, tool_use, initialization, etc.)
  - Different content patterns (keywords, special characters, long strings)
  - Edge cases (empty content, very long content, unicode)

### Performance Validation
- Test with realistic dataset sizes (100, 1000+ events)
- Monitor filtering performance with complex regex patterns
- Identify any bottlenecks or performance regressions

### Security Validation
- Identify ReDoS (Regular Expression Denial of Service) vulnerabilities
- Test with potentially malicious regex patterns
- Document mitigation strategies

## Acceptance Criteria

✅ **Must Have:**
1. Comprehensive test case matrix documented covering at least 15 test scenarios
2. Unit tests implemented for regex filtering logic with >80% coverage of useEventStreamFilter.ts
3. All unit tests passing
4. Playwright E2E tests created for search workflow validation
5. All E2E tests passing
6. Validation report created documenting findings, limitations, and recommendations
7. All searchable fields (content, eventType) verified to work correctly
8. Fallback behavior (invalid regex → string search) validated

✅ **Should Have:**
1. Performance testing results documented (filtering speed with various dataset sizes)
2. ReDoS vulnerability assessment and mitigation recommendations
3. Security considerations documented
4. Identified bugs (if any) documented with reproduction steps
5. Future improvement recommendations documented

✅ **Nice to Have:**
1. Search result highlighting implementation
2. Regex validation feedback in UI
3. Test coverage report generated
4. Performance metrics baseline established

## Validation Commands

Execute these commands to validate the task is complete:

```bash
# 1. Run all unit tests
uv run vitest run frontend/src/composables/__tests__/useEventStreamFilter.test.ts

# 2. Check test coverage
uv run vitest run --coverage frontend/src/composables/__tests__/useEventStreamFilter.test.ts

# 3. Run Playwright E2E tests
uv run playwright test frontend/e2e/search-validation.spec.ts --headed

# 4. Verify test data exists and is properly structured
grep -n "testAgentLogs" frontend/src/data/testData.ts

# 5. Check that validation report was created
ls -la VALIDATION_REPORT.md

# 6. Verify composable filtering logic compiles
uv run python -m py_compile frontend/src/composables/useEventStreamFilter.ts 2>/dev/null && echo "TS syntax valid (basic check)"
```

## Notes

### Dependencies
- **Testing**: Vitest (Vue testing framework), @vue/test-utils
- **E2E**: Playwright MCP (should already be available)
- **Utilities**: No new external dependencies needed

### Key Implementation Details

1. **Regex Implementation Location**: `src/composables/useEventStreamFilter.ts` lines 156-172
   ```typescript
   try {
     const regex = new RegExp(query, 'i')
     filtered = filtered.filter(event =>
       regex.test(event.content) || regex.test(event.eventType || '')
     )
   } catch {
     // Fallback to string search
   }
   ```

2. **Searchable Fields**: Only `content` and `eventType` are searched via regex
   - Other fields (agentName, level, timestamp, tokens) use dedicated filters

3. **Important Limitation**: Case sensitivity is always case-insensitive (`'i'` flag hardcoded)

4. **Security Concern**: No protection against ReDoS attacks - user input directly used in RegExp constructor

5. **Test Data Available**: `src/data/testData.ts` contains sample logs that can be used for testing

### Future Improvements (Out of Scope)
- Add regex validation UI feedback
- Implement ReDoS protection (pattern length limits)
- Extend searchable fields to include metadata, agent names
- Add search history and suggestions
- Implement search result highlighting
- Add case-sensitive search option
- Add multiline regex support (DOTALL flag)

### References
- Current implementation: `src/composables/useEventStreamFilter.ts`
- Filter UI: `src/components/FilterControls.vue`
- Type definitions: `src/types.d.ts`
- Test data: `src/data/testData.ts`
