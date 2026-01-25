# Ralph Loop Memory Integration

## Overview

The Ralph autonomous coding agent now integrates with King AI v3's Memory Service to provide:

- **📔 Diary Entries**: Logs every attempt at implementing a story
- **🤔 Reflections**: Analyzes patterns after story completion
- **💡 Learning**: Uses past experiences to improve future attempts

This creates a **self-improving autonomous loop** that learns from its mistakes and applies successful patterns.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Ralph Autonomous Loop                    │
│                                                              │
│  ┌────────────┐      ┌────────────┐      ┌──────────────┐  │
│  │   Story    │──1──►│  Attempt   │──2──►│    Diary     │  │
│  │  Selection │      │Implementation│     │    Entry     │  │
│  └────────────┘      └────────────┘      └──────┬───────┘  │
│                             │                    │          │
│                             │ Success?           │          │
│                             ▼                    ▼          │
│                      ┌────────────┐      ┌──────────────┐  │
│                      │  Quality   │──3──►│   Memory     │  │
│                      │   Checks   │      │   Service    │  │
│                      └────────────┘      └──────┬───────┘  │
│                             │                    │          │
│                             │                    │          │
│                             ▼                    ▼          │
│                      ┌────────────┐      ┌──────────────┐  │
│                      │   Commit   │──4──►│  Reflection  │  │
│                      │   Changes  │      │   Analysis   │  │
│                      └────────────┘      └──────────────┘  │
│                                                              │
│  Next Story ◄────────────────┘                              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │  Memory Service   │
                    │  (PostgreSQL +    │
                    │   Vector DB)      │
                    └───────────────────┘
                              │
                   ┌──────────┴──────────┐
                   ▼                     ▼
            ┌──────────┐         ┌──────────────┐
            │ Diary    │         │ Reflections  │
            │ Entries  │         │ & Insights   │
            └──────────┘         └──────────────┘
```

## How It Works

### 1. Before Each Attempt: Query Past Learnings

Before trying to implement a story, Ralph queries the Memory Service for similar past experiences:

```python
past_learnings = await memory_client.query_past_learnings(
    story_title="Add user authentication",
    story_description="JWT tokens with refresh flow",
    top_k=3  # Get top 3 similar experiences
)
```

**Returns:**
- Similar stories that succeeded or failed
- Relevant patterns and insights
- Recommendations from past reflections

### 2. After Each Attempt: Write Diary Entry

After every attempt (success or failure), Ralph writes a diary entry:

```python
await memory_client.diary(
    story_id="US-123",
    story_title="Add user authentication",
    attempt_number=2,
    success=False,
    changes_made=3,
    error="Quality checks failed - Python syntax error",
    quality_checks=[("Python syntax", False)],
    context={"max_retries": 3}
)
```

**Captures:**
- What was tried (prompt, code generated)
- What happened (success/failure, files changed)
- Why it failed (error messages, quality checks)
- Learnings extracted automatically

### 3. After Story Completion: Reflect

When a story completes (or exceeds max retries), Ralph reflects on the entire journey:

```python
await memory_client.reflect(
    story_id="US-123",
    story_title="Add user authentication",
    total_attempts=3,
    final_success=True,
    all_attempts=[...],  # All attempt data
    files_changed=["src/auth.py", "src/middleware.py"],
    commit_sha="abc123"
)
```

**Analyzes:**
- Failure patterns (e.g., "Copilot didn't generate code in 2 attempts")
- Success factors (e.g., "Succeeded after making prompt more specific")
- Insights (e.g., "Include file paths in prompt for better results")
- Recommendations for future stories

## Memory Artifacts

### Diary Entry Structure

```json
{
  "entry_type": "diary",
  "timestamp": "2026-01-23T10:30:00Z",
  "story": {
    "id": "US-123",
    "title": "Add user authentication",
    "attempt_number": 2,
    "success": false
  },
  "execution": {
    "changes_made": 0,
    "quality_checks_passed": false,
    "error": "No code generated by Copilot CLI"
  },
  "learning": {
    "issue": "no_code_generated",
    "lesson": "Copilot may need more specific prompt or context"
  }
}
```

### Reflection Structure

```json
{
  "reflection_type": "story_completion",
  "story": {
    "id": "US-123",
    "total_attempts": 3,
    "final_success": true,
    "files_changed": ["src/auth.py"]
  },
  "analysis": {
    "failure_patterns": [
      "Copilot failed to generate code in 2 attempts"
    ],
    "success_factors": [
      "Applied 5 code changes",
      "All quality checks passed"
    ],
    "insights": [
      "Persistence paid off: succeeded after 3 attempts",
      "Modified 1 file: src/auth.py"
    ]
  },
  "learnings": {
    "what_worked": [
      "Copilot generated and applied code successfully"
    ],
    "what_didnt_work": [
      "Copilot didn't generate usable code"
    ],
    "recommendations": [
      "Make prompts more specific with file paths",
      "Continue using strategies that worked"
    ]
  }
}
```

## Usage

### Running Ralph with Memory Integration

```bash
# Default memory service (localhost)
python scripts/ralph/ralph.py

# Custom memory service URL
python scripts/ralph/ralph.py --memory-service http://3.236.144.91:8002

# With max retries
python scripts/ralph/ralph.py --max-retries 5 --memory-service http://3.236.144.91:8002
```

### From King AI v3 Orchestrator

The orchestrator automatically passes the memory service URL to Ralph:

```yaml
# orchestrator/manifests/ralph-code-implementation.yaml
- step: ralph_execution
  role: code
  skill: ralph_code_agent
  inputs:
    - name: prd
      source: previous_step
    - name: memory_service_url
      value: "http://memory-service:8002"
```

## Querying Ralph's Learnings

You can query Ralph's accumulated knowledge:

```python
# Via Memory Service API
POST http://localhost:8002/memory/query
{
  "query_text": "authentication JWT tokens",
  "filter_artifact_type": "reflection",
  "top_k": 5
}
```

Or via kautilya CLI:

```bash
kautilya memory query "authentication JWT" --type reflection
```

## Benefits

### 1. Self-Improvement
- Ralph learns from failures and doesn't repeat mistakes
- Successful patterns are identified and reused
- Each iteration gets smarter

### 2. Transparency
- Full audit trail of all attempts
- Clear understanding of what works vs what doesn't
- Provenance tracking for compliance

### 3. Knowledge Sharing
- Other agents can learn from Ralph's experiences
- Team members can review reflections for insights
- Continuous improvement across projects

### 4. Debugging
- Easy to diagnose why stories failed
- Can replay attempts with context
- Quality checks tracked per attempt

## Example: Learning in Action

### Iteration 1 (Failed)
```
Story: "Add user authentication with JWT"
Attempt: 1
Result: ❌ Failed
Reason: No code generated
Learning: "Copilot needs more specific prompt with file paths"
```

### Iteration 2 (Failed)
```
Story: "Add user authentication with JWT"
Attempt: 2
Result: ❌ Failed
Reason: Quality checks failed
Learning: "Generated code has syntax errors"
Past Learnings Applied: "Include file paths in prompt"
```

### Iteration 3 (Success!)
```
Story: "Add user authentication with JWT"
Attempt: 3
Result: ✅ Success
Files Changed: src/auth.py, src/middleware.py
Past Learnings Applied:
  - "Include file paths in prompt"
  - "Add syntax validation examples"
  
Reflection:
  - Persistence paid off after 3 attempts
  - File paths in prompt improved code generation
  - Quality checks passed with validation examples
  
Recommendations:
  - Always include file paths for focused code generation
  - Use validation examples in prompt for complex logic
```

## Configuration

### Environment Variables

```bash
# Memory service configuration
MEMORY_SERVICE_URL=http://localhost:8002

# Ralph configuration
RALPH_MAX_RETRIES=3
RALPH_ENABLE_REFLECTION=true
RALPH_ENABLE_DIARY=true
```

### Memory Service Requirements

Ralph requires the Memory Service to be running:

```bash
# Start Memory Service
cd king-ai-v3/agentic-framework-main/memory-service
python -m service.main

# Verify it's running
curl http://localhost:8002/health
```

## Troubleshooting

### Memory Service Unavailable

If the Memory Service is down, Ralph will continue but without memory features:

```
⚠️  Failed to commit diary entry: Connection refused
⚠️  Failed to reflect on story: Connection refused
```

Ralph won't crash - it just logs warnings and continues.

### Viewing Ralph's Memory

```bash
# View all Ralph diary entries
kautilya memory query "ralph_diary" --limit 50

# View Ralph reflections
kautilya memory query "ralph_reflection" --limit 20

# Get provenance chain
kautilya memory provenance <artifact_id>
```

### Clearing Ralph's Memory

```bash
# Clear specific session
kautilya memory compact --session ralph_20260123_103000

# Full reset (careful!)
kautilya memory reset --confirm
```

## Advanced Features

### Custom Learning Extractors

You can customize how Ralph extracts learnings:

```python
# In memory_integration.py
def _extract_learning(self, success, error, changes_made, quality_checks):
    # Add your custom logic here
    learning = {
        "outcome": "success" if success else "failure",
        "custom_metric": your_analysis(changes_made)
    }
    return learning
```

### Integration with Other Agents

Other agents can query Ralph's learnings:

```python
# From any subagent
past_code_patterns = await memory_client.query_past_learnings(
    "implement REST API endpoint",
    artifact_type="reflection",
    actor_id="ralph-autonomous-loop"
)
```

## Performance

- **Diary entry**: ~50ms per write
- **Reflection**: ~200ms per write (includes analysis)
- **Query learnings**: ~100ms for top-5 semantic search
- **Storage**: ~5KB per diary entry, ~15KB per reflection

No significant impact on Ralph execution time.

## Future Enhancements

- **Pattern Recognition**: Automatically detect recurring failure patterns
- **Prompt Templates**: Generate optimal prompts based on past successes
- **Success Prediction**: Estimate likelihood of success before attempting
- **Team Learning**: Share learnings across multiple Ralph instances
- **Visual Insights**: Dashboard showing Ralph's learning curve

## See Also

- [Memory Service README](../../king-ai-v3/agentic-framework-main/memory-service/README.md)
- [Ralph Code Agent README](../../king-ai-v3/agentic-framework-main/code-exec/skills/ralph_code_agent/README.md)
- [Provenance Tracking Guide](../../king-ai-v3/agentic-framework-main/docs/provenance.md)
