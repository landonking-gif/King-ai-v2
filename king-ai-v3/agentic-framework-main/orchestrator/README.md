# Lead Agent/Orchestrator Service

The Lead Agent/Orchestrator is the central coordination service for the Agentic Framework. It manages workflow execution, spawns subagents, validates artifacts, and enforces policies.

## Overview

The Orchestrator service is responsible for:

- **Workflow Ingestion**: Load and validate YAML workflow manifests
- **Task Planning**: Decompose complex tasks into discrete steps
- **Subagent Spawning**: Create isolated subagent contexts with specific capabilities
- **Policy Enforcement**: Apply security policies and approval requirements
- **Artifact Validation**: Validate typed artifacts against JSON schemas
- **Commit Decision**: Decide when to commit artifacts to memory
- **Final Synthesis**: Coordinate synthesis of results from multiple subagents

## Architecture

```
┌─────────────────────────────────────────────────┐
│         Lead Agent/Orchestrator                 │
├─────────────────────────────────────────────────┤
│                                                 │
│  ┌──────────────┐      ┌──────────────┐       │
│  │   Workflow   │      │   Artifact   │       │
│  │    Engine    │──────│  Validator   │       │
│  └──────────────┘      └──────────────┘       │
│         │                     │                │
│         │                     │                │
│  ┌──────▼───────────────────▼────────┐        │
│  │      FastAPI Application           │        │
│  │  • POST /workflows/start           │        │
│  │  • POST /subagent/request          │        │
│  │  • POST /artifact/handle           │        │
│  └────────────────────────────────────┘        │
│                                                 │
└─────────────────────────────────────────────────┘
           │                │              │
           ▼                ▼              ▼
   Subagent Manager   Memory Service   MCP Gateway
```

## Quick Start

### Prerequisites

- Python 3.11+
- Redis (for session state)
- PostgreSQL (for structured data)
- Running instances of dependent services:
  - Subagent Manager
  - Memory Service
  - MCP Gateway

### Installation

```bash
# From project root, create and activate virtual environment
uv venv --python 3.11
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root with the following variables:

```bash
# Service Configuration
ORCHESTRATOR_HOST=0.0.0.0
ORCHESTRATOR_PORT=8000

# LLM Provider
DEFAULT_LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...

# Databases
POSTGRES_URL=postgresql://user:password@localhost:5432/agentic_framework
REDIS_URL=redis://localhost:6379/0

# Dependent Services
MCP_GATEWAY_URL=http://localhost:8080
SUBAGENT_MANAGER_URL=http://localhost:8001
MEMORY_SERVICE_URL=http://localhost:8002
CODE_EXECUTOR_URL=http://localhost:8003

# Security
JWT_SECRET_KEY=your-secret-key-change-in-production

# Logging
LOG_LEVEL=INFO
```

### Running the Service

```bash
# Run directly with Python
python -m orchestrator.service.main

# Or use the entry point function
python orchestrator/service/main.py

# Run with uvicorn for production
uvicorn orchestrator.service.main:app --host 0.0.0.0 --port 8000

# Run with auto-reload for development
uvicorn orchestrator.service.main:app --reload
```

The service will be available at `http://localhost:8000`.

API documentation is available at `http://localhost:8000/docs`.

## API Endpoints

### POST /workflows/start

Start a new workflow execution from a manifest.

**Request:**
```json
{
  "manifest_name": "customer-support-workflow",
  "user_input": {
    "ticket_id": "TICKET-123",
    "customer_context": "Premium customer, high priority"
  },
  "llm_provider": "anthropic"
}
```

**Response:**
```json
{
  "workflow_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "running",
  "message": "Workflow 'customer-support-workflow' started successfully",
  "estimated_duration_seconds": 720
}
```

### POST /subagent/request

Request execution of a subagent for a specific workflow step.

**Request:**
```json
{
  "workflow_id": "550e8400-e29b-41d4-a716-446655440000",
  "step_id": "research-customer-issue",
  "role": "research",
  "capabilities": ["ticket_read", "kb_search"],
  "inputs": {
    "ticket_id": "TICKET-123"
  },
  "timeout": 300
}
```

**Response:**
```json
{
  "subagent_id": "subagent-001",
  "workflow_id": "550e8400-e29b-41d4-a716-446655440000",
  "step_id": "research-customer-issue",
  "status": "success",
  "artifacts": ["artifact-001", "artifact-002"],
  "execution_time_seconds": 45.2,
  "token_usage": {
    "prompt_tokens": 1500,
    "completion_tokens": 800
  }
}
```

### POST /artifact/handle

Validate and optionally persist a typed artifact.

**Request:**
```json
{
  "artifact_data": {
    "id": "artifact-001",
    "artifact_type": "research_snippet",
    "created_by": "research-agent-001",
    "source": {
      "url": "https://example.com/kb/article-123"
    },
    "text": "Detailed research findings...",
    "summary": "Summary of findings",
    "confidence": 0.95,
    "provenance": { ... }
  },
  "artifact_type": "research_snippet",
  "validate_only": false,
  "workflow_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

**Response:**
```json
{
  "artifact_id": "artifact-001",
  "valid": true,
  "validation_errors": [],
  "persisted": true,
  "memory_ref": "memory://artifacts/artifact-001"
}
```

## Workflow Manifests

Workflows are defined in YAML manifests located in `orchestrator/manifests/`.

### Example Manifest

```yaml
name: "simple-research-workflow"
version: "1.0.0"
description: "Simple workflow for conducting research and generating a summary report"

steps:
  - id: "research-topic"
    role: "research"
    capabilities:
      - "web_search"
      - "document_read"
    inputs:
      - name: "topic"
        source: "user_input"
        required: true
    outputs:
      - name: "research_results"
        artifact_type: "research_snippet"
    timeout: 300

  - id: "generate-report"
    role: "synthesis"
    capabilities:
      - "summarize"
    inputs:
      - name: "research_results"
        source: "previous_step"
        required: true
    outputs:
      - name: "final_report"
        artifact_type: "synthesis_result"
    timeout: 180

memory:
  persist_on:
    - "on_complete"
  compaction:
    strategy: "summarize"
    max_tokens: 8000

tools:
  catalog_ids:
    - "web_search"
    - "document_reader"

policies:
  requires_human_approval: false
  max_tool_calls_per_step: 10
```

See `orchestrator/manifests/` for more examples.

## Testing

```bash
# Run all tests
pytest orchestrator/tests/

# Run with coverage
pytest orchestrator/tests/ --cov=orchestrator --cov-report=html

# Run specific test file
pytest orchestrator/tests/test_orchestrator.py

# Run with verbose output
pytest orchestrator/tests/ -v
```

## Code Quality

```bash
# Format code with Black
black --line-length 100 orchestrator/

# Type checking with mypy
mypy --strict orchestrator/

# Linting with ruff
ruff check orchestrator/
```

## Development

### Project Structure

```
orchestrator/
├── __init__.py                 # Package initialization
├── README.md                   # This file
├── service/                    # Service implementation
│   ├── __init__.py
│   ├── config.py              # Configuration management
│   ├── main.py                # FastAPI application
│   ├── models.py              # Pydantic models
│   └── workflow_engine.py     # Workflow execution engine
├── manifests/                  # YAML workflow manifests
│   ├── customer-support-workflow.yaml
│   └── simple-research-workflow.yaml
└── tests/                      # Test suite
    ├── __init__.py
    └── test_orchestrator.py   # Comprehensive tests
```

### Adding a New Workflow

1. Create a YAML manifest in `orchestrator/manifests/`
2. Define steps with roles, capabilities, inputs, and outputs
3. Configure memory persistence and compaction
4. Set tool catalog IDs and policies
5. Validate the manifest using the JSON schema

### Adding a New Artifact Type

1. Add enum value to `ArtifactType` in `models.py`
2. Create Pydantic model extending `ArtifactBase`
3. Update artifact validation logic in `workflow_engine.py`
4. Add tests for the new artifact type

## Monitoring

The Orchestrator service exposes metrics and health endpoints:

- **Health Check**: `GET /health` - Service and dependency health status
- **Metrics**: Prometheus metrics available (if enabled)
- **Logging**: Structured logs with configurable levels

## Security

- All artifacts are validated against JSON schemas
- PII detection enabled by default (configurable)
- Policy enforcement for human approval requirements
- JWT authentication for service-to-service communication
- Provenance tracking for all operations

## Ralph Code Agent Integration

The orchestrator now includes a **Ralph Code Agent** skill that enables autonomous code generation and modification by delegating to the Ralph loop on AWS.

### What It Does

When you ask the orchestrator to implement a code change (e.g., "Add user authentication to my API"), it will:

1. **Generate a detailed PRD** from your natural language request
2. **Delegate to the Ralph loop** running on your AWS server
3. **Execute Ralph** which autonomously writes, tests, and refines code
4. **Return the results** including files changed and implementation summary

### Usage

**Via Dashboard ("Talk to King AI" tab):**
```
"Implement a user profile page with avatar upload"
"Add password reset functionality with email verification"
"Create an API endpoint for file uploads with validation"
```

**Via kautilya CLI:**
```bash
kautilya workflow submit ralph-code-implementation \
  --input '{"task_description": "Add JWT authentication to FastAPI backend"}'
```

**Via API:**
```python
import requests

response = requests.post(
    "http://localhost:8000/workflows/start",
    json={
        "manifest_name": "ralph-code-implementation",
        "user_input": {
            "task_description": "Implement rate limiting on API endpoints"
        }
    }
)

workflow_id = response.json()["workflow_id"]
```

### Workflow Steps

The `ralph-code-implementation.yaml` manifest defines a 3-step workflow:

1. **PRD Generation** (`synthesis` role) - Creates structured requirements document
2. **Ralph Execution** (`code` role) - Runs Ralph on AWS with approval gate
3. **Verification** (`verify` role) - Validates and summarizes results

### Configuration

**Environment Variables:**
```bash
# AWS server for Ralph execution
RALPH_SERVER_HOST=54.167.201.176
RALPH_SSH_KEY_PATH=/path/to/king-ai-studio.pem
RALPH_SSH_USER=ubuntu

# Ralph timeout (seconds)
RALPH_EXECUTION_TIMEOUT=600
```

**Approval Requirements:**

By default, Ralph execution requires human approval before running on AWS. To disable:

```yaml
# In ralph-code-implementation.yaml
- step: ralph_execution
  approval_required: false  # Change to false for auto-execution
```

### Safety Features

- **Human Approval Gate**: All code changes require approval before execution
- **Sandboxed Execution**: Ralph runs in isolated environment on AWS
- **Timeout Protection**: Executions limited to 10 minutes
- **Full Provenance**: All changes tracked in PostgreSQL with timestamps
- **SSH Security**: Uses key-based authentication, no password exposure

### Files

- [code-exec/skills/ralph_code_agent/](../code-exec/skills/ralph_code_agent/) - Skill implementation
- [orchestrator/manifests/ralph-code-implementation.yaml](manifests/ralph-code-implementation.yaml) - Workflow manifest
- [examples/ralph-code-agent/](../examples/ralph-code-agent/) - Usage examples

### Testing

Run the test suite:
```bash
cd code-exec/skills/ralph_code_agent
python test_handler.py
```

### Troubleshooting

**SSH connection fails:**
- Verify AWS server IP (3.236.144.91) is reachable
- Check SSH key permissions: `chmod 600 king-ai-studio.pem`
- Test manual connection: `ssh -i king-ai-studio.pem ubuntu@3.236.144.91`

**Ralph execution times out:**
- Increase timeout in `RALPH_EXECUTION_TIMEOUT` environment variable
- Check Ralph logs on AWS: `ssh ubuntu@3.236.144.91 'tail -f /tmp/ralph_*.log'`

**PRD validation errors:**
- Ensure task description is detailed and specific
- Include context about your tech stack
- List specific files to modify if known

## Troubleshooting

### Service won't start

1. Check that all required environment variables are set
2. Verify dependent services (Redis, PostgreSQL) are running
3. Check logs for specific error messages
4. Ensure port 8000 is not already in use

### Workflow execution fails

1. Validate manifest using JSON schema
2. Check subagent manager service is available
3. Verify input data matches manifest requirements
4. Review logs for step-specific errors

### Artifact validation errors

1. Check artifact data matches the expected type schema
2. Review validation errors in the response
3. Ensure all required fields are present
4. Verify confidence scores are within 0.0-1.0 range

## Contributing

See the main project [CONTRIBUTING.md](../CONTRIBUTING.md) for development guidelines.

## License

[License details to be added]
