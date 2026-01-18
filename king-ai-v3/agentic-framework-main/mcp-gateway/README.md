# MCP Gateway

Enterprise MCP tool catalog and runtime proxy for the Agentic Framework.

## Overview

The MCP Gateway is a critical component of the agent framework that provides:

- **Tool Discovery**: Centralized catalog of available MCP servers and tools
- **Authentication**: JWT-based ephemeral tokens for scoped direct access
- **Rate Limiting**: Redis-backed rate limiting per tool and actor
- **PII Detection**: Automatic detection of sensitive data in tool invocations
- **Provenance Logging**: Complete audit trail of all tool invocations
- **Two Runtime Modes**:
  - **Orchestrated**: Lead Agent mediated access (default)
  - **Scoped Direct**: Ephemeral token-based direct access for subagents

## Architecture

```
┌─────────────────┐
│  Lead Agent /   │
│   Subagent      │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────────┐
│           MCP Gateway (Port 8080)                │
├─────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌──────────────┐             │
│  │   Catalog   │  │ Auth/Tokens  │             │
│  └─────────────┘  └──────────────┘             │
│  ┌─────────────┐  ┌──────────────┐             │
│  │ Rate Limit  │  │  PII Check   │             │
│  └─────────────┘  └──────────────┘             │
│  ┌─────────────────────────────────┐           │
│  │     Tool Invocation Proxy       │           │
│  └─────────────────────────────────┘           │
└──────────────┬──────────────────────────────────┘
               │
               ▼
    ┌─────────────────────┐
    │   MCP Servers       │
    │  (web_search,       │
    │   file_ops, etc.)   │
    └─────────────────────┘
```

## Features

### 1. Tool Catalog Management

Register, discover, and manage MCP servers:

```python
# Register a new MCP server
POST /catalog/register
{
  "tool_id": "web_search",
  "name": "Web Search",
  "version": "1.0.0",
  "owner": "platform-team",
  "contact": "platform@example.com",
  "tools": [...],
  "rate_limits": {"max_calls": 100, "window_seconds": 60}
}

# List all tools
GET /catalog/tools?enabled_only=true

# Get tool schema
GET /catalog/tools/{tool_id}/schema/{tool_name}
```

### 2. Authentication & Authorization

Mint ephemeral tokens for scoped direct access:

```python
# Mint token
POST /auth/token
{
  "scope": {
    "tool_ids": ["web_search", "text_processing"],
    "actor_id": "research_subagent",
    "actor_type": "subagent",
    "max_invocations": 10
  },
  "ttl_minutes": 15
}

# Response
{
  "token": "eyJ...",
  "expires_at": "2025-12-28T12:00:00Z",
  "scope": {...}
}
```

### 3. Tool Invocation

Two runtime modes supported:

#### Orchestrated Mode (Lead Agent mediated)
```python
POST /tools/invoke
{
  "tool_id": "web_search",
  "tool_name": "search",
  "arguments": {"query": "AI agents", "max_results": 10},
  "actor_id": "lead_agent",
  "actor_type": "lead_agent",
  "runtime_mode": "orchestrated"
}
```

#### Scoped Direct Mode (Token-based)
```python
POST /tools/invoke
{
  "tool_id": "web_search",
  "tool_name": "search",
  "arguments": {"query": "research topic"},
  "actor_id": "research_subagent",
  "actor_type": "subagent",
  "runtime_mode": "scoped_direct",
  "token": "eyJ..."
}
```

### 4. Rate Limiting

Automatic per-tool and per-actor rate limiting using Redis:

- Sliding window algorithm
- Configurable limits per tool
- Returns `retry_after` when limit exceeded

### 5. PII Detection

Automatic detection of sensitive data:

- Keyword-based detection (api_key, password, secret, etc.)
- Pattern matching (SSN, credit cards, emails)
- Flags invocations with `pii_detected: true`

### 6. Provenance Logging

Complete audit trail for all invocations:

```python
GET /provenance/logs?limit=100&tool_id=web_search

# Returns
[
  {
    "invocation_id": "abc123",
    "tool_id": "web_search",
    "tool_name": "search",
    "actor_id": "lead_agent",
    "actor_type": "lead_agent",
    "arguments_hash": "sha256...",
    "result_hash": "sha256...",
    "success": true,
    "execution_time_ms": 123.45,
    "pii_detected": false,
    "runtime_mode": "orchestrated",
    "timestamp": "2025-12-28T10:30:00Z"
  }
]
```

## Installation

### Prerequisites

- Python 3.11+
- Redis (for rate limiting)

### Setup

```bash
# Install dependencies (from project root)
uv pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
vim .env  # Add JWT_SECRET_KEY and REDIS_URL
```

### Environment Variables

Required:
- `JWT_SECRET_KEY`: Secret key for JWT token signing
- `REDIS_URL`: Redis connection URL (default: `redis://localhost:6379/0`)

Optional:
- `MCP_GATEWAY_HOST`: Host to bind to (default: `0.0.0.0`)
- `MCP_GATEWAY_PORT`: Port to listen on (default: `8080`)
- `ENABLE_PII_DETECTION`: Enable PII detection (default: `true`)
- `EPHEMERAL_TOKEN_TTL_MINUTES`: Default token TTL (default: `15`)

## Running the Service

### Development Mode

```bash
# From project root
python -m mcp_gateway.service.main

# Or with uvicorn directly
uvicorn mcp_gateway.service.main:app --reload --port 8080
```

### Production Mode

```bash
uvicorn mcp_gateway.service.main:app \
  --host 0.0.0.0 \
  --port 8080 \
  --workers 4 \
  --log-level info
```

## Testing

```bash
# Run all tests
pytest mcp-gateway/tests/

# Run with coverage
pytest mcp-gateway/tests/ --cov=mcp-gateway --cov-report=html

# Run specific test class
pytest mcp-gateway/tests/test_gateway.py::TestToolInvocation -v
```

## API Documentation

Once running, visit:
- Swagger UI: `http://localhost:8080/docs`
- ReDoc: `http://localhost:8080/redoc`
- OpenAPI JSON: `http://localhost:8080/openapi.json`

## Sample Tools (Sprint 0)

The gateway comes pre-configured with sample tools:

1. **web_search**: Web search with rate limiting
2. **file_operations**: File read/write operations
3. **database_query**: Database queries with PII risk classification
4. **text_processing**: Text summarization and entity extraction (safe)

All tools return mock data for Sprint 0. In production, these would proxy to actual MCP servers.

## Security

### Tool Classifications

Tools are classified for security:

- `SAFE`: No security concerns
- `PII_RISK`: May access sensitive data
- `EXTERNAL_CALL`: Makes external API calls
- `SIDE_EFFECT`: Modifies state (writes, deletes, etc.)
- `REQUIRES_APPROVAL`: Needs explicit human approval

### Rate Limiting

- Per-tool and per-actor limits
- Sliding window algorithm
- Redis-backed for distributed deployments

### PII Detection

- Automatic scanning of tool arguments
- Keyword and pattern-based detection
- Logged in provenance for audit

## Integration with Orchestrator

The gateway is designed to work with the Orchestrator:

1. **Orchestrated Mode**: Orchestrator validates requests and forwards to gateway
2. **Scoped Direct Mode**: Orchestrator mints tokens for subagents to call gateway directly

## Production Considerations

For production deployment:

1. **Database Migration**: Migrate catalog from in-memory to PostgreSQL
2. **Tool Execution**: Replace mock execution with actual HTTP calls to MCP servers
3. **Provenance Storage**: Write logs to PostgreSQL instead of in-memory
4. **Token Tracking**: Track token usage in Redis/DB instead of JWT payload
5. **Monitoring**: Add Prometheus metrics and OpenTelemetry traces
6. **Security**: Add TLS, API keys for registration endpoints
7. **Scaling**: Deploy multiple instances behind load balancer (Redis provides shared state)

## Code Quality

All code follows team standards:

- Type hints with mypy strict mode
- Black formatting (100 char line length)
- Pydantic for validation
- Async patterns with anyio
- 90%+ test coverage

## License

[License details to be added]
