# Code Executor Service

Sandboxed execution service for deterministic skills with provenance tracking and validation.

## Overview

The Code Executor Service is a core component of the Enterprise Agentic Framework that provides:

- **Deterministic Skill Execution**: Sandboxed execution environment with timeout protection
- **Input/Output Validation**: JSON Schema-based validation for all skill inputs and outputs
- **Provenance Tracking**: Complete audit trail with SHA-256 hashing of inputs and outputs
- **Skill Registry**: Dynamic skill loading and management with safety flags
- **RESTful API**: FastAPI-based service with comprehensive endpoints

## Architecture

```
Code Executor Service
├── service/
│   ├── main.py           # FastAPI application
│   ├── executor.py       # Sandboxed execution engine
│   ├── registry.py       # Skill registry and loader
│   ├── models.py         # Pydantic models
│   └── config.py         # Configuration
├── skills/               # Skill definitions
│   ├── text_summarize/
│   ├── extract_entities/
│   ├── embed_text/
│   └── compact_memory/
└── tests/                # Test suite
```

## Features

### Skill Registry
- Auto-loads skills from directory structure
- Validates skill metadata and schemas
- Enforces safety flags and policy requirements
- Caches handler functions for performance

### Sandboxed Execution
- Input validation against JSON Schema
- Timeout protection (configurable)
- Output validation against JSON Schema
- Deterministic SHA-256 hashing for provenance
- Comprehensive execution logging

### Safety Flags
- `none`: No special safety considerations
- `pii_risk`: May handle personally identifiable information
- `external_call`: Makes external API calls
- `side_effect`: Has side effects (requires approval)
- `file_system`: Accesses file system
- `network_access`: Requires network access

## API Endpoints

### Health & Status
- `GET /` - Root endpoint
- `GET /health` - Health check

### Skills Management
- `GET /skills/list` - List all available skills
- `GET /skills/{skill_id}` - Get skill metadata
- `GET /skills/{skill_id}/schema` - Get skill I/O schemas
- `POST /skills/register` - Register new skill (planned)

### Execution
- `POST /skills/execute` - Execute a skill with validation

## Skill Structure

Each skill must have the following structure:

```
skills/skill_name/
├── skill.yaml        # Metadata
├── schema.json       # I/O schemas
└── handler.py        # Implementation
```

### skill.yaml
```yaml
name: skill_name
version: 1.0.0
description: What the skill does
safety_flags:
  - none
requires_approval: false
handler: handler.function_name
tags:
  - tag1
  - tag2
```

### schema.json
```json
{
  "input": {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "properties": { ... },
    "required": [ ... ]
  },
  "output": {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "properties": { ... },
    "required": [ ... ]
  }
}
```

### handler.py
```python
def function_name(arg1: type1, arg2: type2) -> dict:
    """Handler implementation."""
    # ... implementation
    return {
        "field1": value1,
        "field2": value2,
    }
```

## Built-in Skills

### text_summarize
Extractive summarization for reducing token count while preserving key information.

**Inputs:**
- `text`: Text to summarize
- `max_sentences`: Maximum sentences in summary (default: 5)
- `style`: Summary style - concise, detailed, or bullet-points

**Outputs:**
- `summary`: Summarized text
- `compression_ratio`: Token reduction ratio
- `original_length`: Original character count
- `summary_length`: Summary character count

### extract_entities
Named entity recognition for extracting persons, organizations, locations, dates, etc.

**Inputs:**
- `text`: Text to analyze
- `entity_types`: Types to extract (PERSON, ORGANIZATION, LOCATION, DATE, MONEY, EMAIL, PHONE, URL)
- `min_confidence`: Minimum confidence threshold (0.0-1.0)

**Outputs:**
- `entities`: Array of extracted entities with text, type, confidence, position
- `total_entities`: Total count
- `entity_counts`: Breakdown by entity type

### embed_text
Generate embeddings using sentence-transformers.

**Inputs:**
- `text`: Text to embed
- `model`: Model to use (all-MiniLM-L6-v2, all-mpnet-base-v2, paraphrase-MiniLM-L6-v2)
- `normalize`: Whether to normalize embeddings

**Outputs:**
- `embedding`: Vector representation
- `dimension`: Embedding dimensionality
- `model_used`: Model identifier
- `text_length`: Input text length

### compact_memory
Compact artifacts and conversation history for token reduction.

**Inputs:**
- `artifacts`: Array of artifacts to compact
- `strategy`: Compaction strategy (summarize, truncate, merge, intelligent)
- `max_tokens`: Maximum tokens in output
- `preserve_fields`: Fields to always keep

**Outputs:**
- `compacted_artifacts`: Compacted artifact array
- `compression_ratio`: Token reduction ratio
- `original_count` / `compacted_count`: Before/after counts
- `strategy_used`: Strategy applied

## Configuration

Environment variables (prefix: `CODE_EXEC_`):

```bash
CODE_EXEC_HOST=0.0.0.0
CODE_EXEC_PORT=8002
CODE_EXEC_DEBUG=false
CODE_EXEC_SKILLS_DIRECTORY=/path/to/skills
CODE_EXEC_MAX_EXECUTION_TIME=30
CODE_EXEC_LOG_LEVEL=INFO
CODE_EXEC_REDIS_URL=redis://localhost:6379/2
CODE_EXEC_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

## Running the Service

### Setup
```bash
# Create virtual environment
uv venv --python 3.11
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

### Start Service
```bash
# Development
uvicorn code_exec.service.main:app --reload --port 8002

# Production
uvicorn code_exec.service.main:app --host 0.0.0.0 --port 8002 --workers 4
```

### Run Tests
```bash
pytest code-exec/tests/ -v
pytest code-exec/tests/test_executor.py -k test_text_summarize_skill
```

## Usage Examples

### Python Client
```python
import httpx

async with httpx.AsyncClient() as client:
    response = await client.post(
        "http://localhost:8002/skills/execute",
        json={
            "skill": "text_summarize",
            "args": {
                "text": "Long text to summarize...",
                "max_sentences": 3,
            },
            "actor_id": "agent_001",
            "actor_type": "subagent",
        },
    )

    result = response.json()
    print(result["result"]["summary"])
    print(f"Provenance: {result['provenance']['inputs_hash']}")
```

### cURL
```bash
curl -X POST http://localhost:8002/skills/execute \
  -H "Content-Type: application/json" \
  -d '{
    "skill": "extract_entities",
    "args": {
      "text": "Contact John Doe at john@example.com",
      "entity_types": ["EMAIL", "PERSON"]
    }
  }'
```

## Provenance Record

Every execution produces a complete provenance record:

```json
{
  "execution_id": "uuid",
  "skill_name": "text_summarize",
  "skill_version": "1.0.0",
  "actor_id": "agent_001",
  "actor_type": "subagent",
  "inputs_hash": "sha256...",
  "outputs_hash": "sha256...",
  "tool_ids": [],
  "timestamp": "2025-12-28T12:00:00Z",
  "execution_time_ms": 45.2,
  "success": true
}
```

## Security

- All inputs validated against JSON Schema
- Skills with `side_effect` flag require policy approval
- Execution timeout prevents runaway processes
- Sandboxed environment limits skill capabilities
- Complete audit trail via provenance records

## Extending with New Skills

1. Create skill directory: `skills/my_skill/`
2. Add `skill.yaml` with metadata
3. Add `schema.json` with I/O schemas
4. Implement handler in `handler.py`
5. Restart service to auto-load

See existing skills for examples.

## Team Standards Compliance

- Python 3.11+ with type hints
- Black formatting (100-char line length)
- mypy strict mode
- Pydantic for validation
- anyio for async patterns
- 90%+ test coverage

## Integration

The Code Executor integrates with:

- **Orchestrator**: Receives execution requests from workflow engine
- **Memory Service**: Skills can store/retrieve artifacts
- **MCP Gateway**: Skills can call external tools (via policy)
- **Subagent Manager**: Subagents invoke skills for deterministic operations

## Troubleshooting

**Skills not loading:**
- Check `CODE_EXEC_SKILLS_DIRECTORY` path
- Verify skill.yaml and schema.json format
- Check handler.py for syntax errors

**Validation errors:**
- Verify input matches schema exactly
- Check for missing required fields
- Ensure data types match schema

**Execution timeout:**
- Increase `CODE_EXEC_MAX_EXECUTION_TIME`
- Optimize handler implementation
- Check for blocking I/O operations

## License

Internal - Enterprise Agentic Framework
