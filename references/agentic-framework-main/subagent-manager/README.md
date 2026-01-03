# Subagent Manager Service

Enterprise-grade service for managing isolated LLM contexts (subagents) with capability enforcement, schema validation, and bounded lifetimes.

## Overview

The Subagent Manager is a core component of the Agentic Framework that:

- **Spawns isolated LLM contexts** with role-specific capabilities
- **Enforces capability lists** to control available tools
- **Validates outputs** against registered JSON schemas
- **Manages bounded lifetimes** with automatic cleanup
- **Provides context isolation** - each subagent maintains separate conversation history
- **Supports multiple LLM providers** through adapter pattern (Anthropic, OpenAI, local models)

## Architecture

```
Lead Agent → Subagent Manager → Subagent Contexts → LLM Adapters → LLM Providers
                ↓
          Schema Validator
                ↓
          Typed Artifacts
```

### Key Components

1. **SubagentLifecycleManager** - Manages creation, execution, and cleanup
2. **SubagentContext** - Isolated context with conversation history
3. **SchemaValidator** - Validates outputs against JSON Schema
4. **LLM Adapters** - Provider-agnostic LLM integration

## API Endpoints

### Health Check
```bash
GET /health
```

### Spawn Subagent
```bash
POST /subagent/spawn
{
  "role": "research",
  "capabilities": ["web_search", "document_read"],
  "system_prompt": "You are a research assistant specializing in AI.",
  "timeout": 300,
  "max_iterations": 10,
  "metadata": {"project": "quantum-ai"}
}
```

**Response:**
```json
{
  "subagent_id": "research-a7b3c9d2",
  "role": "research",
  "status": "ready",
  "capabilities": ["web_search", "document_read"],
  "created_at": "2025-12-28T10:30:00Z",
  "timeout": 300
}
```

### Execute Task
```bash
POST /subagent/execute
{
  "subagent_id": "research-a7b3c9d2",
  "task": "Research the latest advances in quantum computing",
  "inputs": {
    "topic": "quantum computing",
    "max_sources": 5
  },
  "expected_output_schema": "research_snippet"
}
```

**Response:**
```json
{
  "subagent_id": "research-a7b3c9d2",
  "status": "completed",
  "output": {
    "id": "snippet-001",
    "text": "Recent advances in quantum computing include...",
    "summary": "Quantum computing breakthroughs in 2025",
    "created_at": "2025-12-28T10:31:00Z"
  },
  "raw_response": "Here's what I found...",
  "tokens_used": {
    "prompt_tokens": 150,
    "completion_tokens": 500,
    "total_tokens": 650
  },
  "execution_time_ms": 2500
}
```

### Get Status
```bash
GET /subagent/{subagent_id}/status
```

### Destroy Subagent
```bash
POST /subagent/destroy
{
  "subagent_id": "research-a7b3c9d2",
  "reason": "Task completed"
}
```

### List Subagents
```bash
GET /subagents
```

### Schema Management
```bash
GET /schemas                    # List all schemas
GET /schemas/{schema_name}      # Get specific schema
```

## Configuration

Environment variables (prefix: `SUBAGENT_`):

```bash
# Service
SUBAGENT_HOST=0.0.0.0
SUBAGENT_PORT=8001
SUBAGENT_LOG_LEVEL=INFO

# LLM Provider
SUBAGENT_LLM_PROVIDER=anthropic    # mock, anthropic, openai
SUBAGENT_LLM_MODEL=claude-sonnet-4-20250514
SUBAGENT_LLM_TEMPERATURE=0.7
SUBAGENT_LLM_MAX_TOKENS=4096

# API Keys
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...

# Lifecycle
SUBAGENT_DEFAULT_TIMEOUT=300       # seconds
SUBAGENT_MAX_LIFETIME=3600         # seconds
SUBAGENT_CLEANUP_INTERVAL=60       # seconds
SUBAGENT_MAX_CONCURRENT_SUBAGENTS=100

# Schema Registry
SUBAGENT_SCHEMA_REGISTRY_PATH=docs/schema_registry
```

## Subagent Roles

Pre-defined roles with specific capabilities:

- **research** - Web search, document reading, summarization
- **verify** - Fact checking, claim verification, evidence gathering
- **code** - Code reading, writing, testing, patching
- **synthesis** - Combining artifacts, generating reports
- **custom** - User-defined role

## Artifact Schemas

Built-in artifact schemas (in `docs/schema_registry/`):

### research_snippet
```json
{
  "id": "string",
  "source": {"url": "uri", "doc_id": "string"},
  "text": "string",
  "summary": "string",
  "tags": ["string"],
  "confidence": 0.95,
  "created_at": "datetime"
}
```

### claim_verification
```json
{
  "id": "string",
  "claim_text": "string",
  "verdict": "verified|refuted|inconclusive",
  "confidence": 0.85,
  "evidence_refs": ["string"],
  "created_at": "datetime"
}
```

### code_patch
```json
{
  "id": "string",
  "repo": "string",
  "files_changed": [{"path": "string", "diff": "string"}],
  "patch_summary": "string",
  "tests": ["string"],
  "merge_ready": true
}
```

## Setup & Development

### Install Dependencies
```bash
# Create virtual environment
uv venv --python 3.11
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

### Run Service
```bash
# Development mode (with reload)
python -m subagent_manager.service.main

# Production mode
uvicorn subagent_manager.service.main:app \
  --host 0.0.0.0 \
  --port 8001 \
  --workers 4
```

### Run Tests
```bash
# All tests
pytest subagent-manager/tests/

# Specific test
pytest subagent-manager/tests/test_subagent.py::TestSubagentLifecycle::test_spawn_subagent

# With coverage
pytest --cov=subagent_manager --cov-report=html
```

### Code Quality
```bash
# Format
black --line-length 100 subagent-manager/

# Type check
mypy --strict subagent-manager/service/

# Lint
ruff check subagent-manager/
```

## LLM Adapters

### Mock Adapter (Testing)
```python
from adapters.llm import MockLLMAdapter

adapter = MockLLMAdapter(
    model="test-model",
    response_template="Response: {prompt}",
    delay_ms=100
)
```

### Anthropic Claude
```python
from adapters.llm.anthropic import AnthropicAdapter

adapter = AnthropicAdapter(
    model="claude-sonnet-4-20250514",
    api_key=os.getenv("ANTHROPIC_API_KEY")
)
```

### OpenAI
```python
from adapters.llm.openai import OpenAIAdapter

adapter = OpenAIAdapter(
    model="gpt-4o",
    api_key=os.getenv("OPENAI_API_KEY")
)
```

## Usage Example

```python
import httpx
import anyio

async def main():
    async with httpx.AsyncClient(base_url="http://localhost:8001") as client:
        # Spawn research subagent
        spawn_response = await client.post("/subagent/spawn", json={
            "role": "research",
            "capabilities": ["web_search", "document_read"],
            "system_prompt": "You are a research assistant.",
            "timeout": 120
        })
        subagent = spawn_response.json()
        subagent_id = subagent["subagent_id"]

        # Execute task
        exec_response = await client.post("/subagent/execute", json={
            "subagent_id": subagent_id,
            "task": "Research quantum computing applications",
            "inputs": {"depth": "comprehensive"},
            "expected_output_schema": "research_snippet"
        })
        result = exec_response.json()

        print(f"Status: {result['status']}")
        if result["output"]:
            print(f"Summary: {result['output']['summary']}")

        # Cleanup
        await client.post("/subagent/destroy", json={
            "subagent_id": subagent_id
        })

anyio.run(main)
```

## Observability

### Prometheus Metrics
```bash
curl http://localhost:8001/metrics
```

Available metrics:
- `subagent_spawned_total` - Total subagents spawned by role
- `subagent_executed_total` - Total executions by status
- `subagent_execution_duration_seconds` - Execution duration histogram

### Logging
Structured logging with correlation IDs for tracing:
```
INFO: Subagent research-a7b3c9d2 spawned with 2 capabilities
INFO: Executing task for research-a7b3c9d2 (timeout: 120s)
INFO: Task completed in 2.5s (650 tokens)
```

## Security

- **Capability Enforcement** - Subagents can only access whitelisted tools
- **Schema Validation** - All outputs validated before returning
- **Timeout Protection** - Hard limits prevent runaway executions
- **Context Isolation** - Conversation histories never leak between subagents
- **PII Filtering** - Schema validation prevents sensitive data in artifacts

## Performance

- **Concurrent Subagents**: Up to 100 (configurable)
- **Average Execution**: 1-5 seconds (depends on LLM provider)
- **Memory per Subagent**: ~10-50MB (depends on context size)
- **Cleanup**: Automatic every 60 seconds

## Troubleshooting

### Subagent Timeout
```
Status: timeout
Error: Execution timed out after 300 seconds
```
**Solution**: Increase timeout or simplify task

### Schema Validation Failure
```
Status: failed
Error: Schema validation failed: 'id' is a required property
```
**Solution**: Adjust system prompt to guide LLM output format

### LLM Provider Error
```
Error: Anthropic API request failed: 401 Unauthorized
```
**Solution**: Check API key in environment variables

## License

MIT License - Part of the Agentic Framework
