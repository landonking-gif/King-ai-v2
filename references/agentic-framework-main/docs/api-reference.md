# API Reference

Complete API documentation for programmatic workflow execution and agent orchestration.

## Table of Contents

- [Overview](#overview)
- [Orchestrator API](#orchestrator-api)
- [Memory Service API](#memory-service-api)
- [Code Executor API](#code-executor-api)
- [MCP Gateway API](#mcp-gateway-api)
- [Authentication](#authentication)
- [Error Handling](#error-handling)
- [Python SDK](#python-sdk)
- [Examples](#examples)

## Overview

The Agentic Framework provides REST APIs and Python SDKs for:
- Executing workflows from manifests
- Spawning and managing subagents
- Storing and retrieving typed artifacts
- Executing skills in sandboxed environments
- Invoking MCP tools

### Base URLs

```
Orchestrator:    http://localhost:8000
Memory Service:  http://localhost:8001
Code Executor:   http://localhost:8002
MCP Gateway:     http://localhost:8080
```

### API Versioning

All APIs use semantic versioning in the URL path:

```
/api/v1/workflows/start
```

## Orchestrator API

The Orchestrator manages workflow execution and subagent coordination.

### Start Workflow

Execute a workflow from a YAML manifest.

**Endpoint:** `POST /api/v1/workflows/start`

**Request:**
```json
{
  "manifest_id": "research-workflow",
  "manifest_path": "manifests/research-workflow.yaml",
  "input_data": {
    "query": "multi-agent systems in finance",
    "max_results": 10
  },
  "session_id": "optional-session-id",
  "execution_mode": "async",
  "callbacks": {
    "on_step_complete": "https://webhook.site/...",
    "on_workflow_complete": "https://webhook.site/..."
  }
}
```

**Parameters:**
- `manifest_id` (string, required): Unique identifier for the workflow
- `manifest_path` (string, optional): Path to manifest file (if not using registry)
- `input_data` (object, required): Input parameters for the workflow
- `session_id` (string, optional): Session ID for memory persistence
- `execution_mode` (string, optional): `sync` or `async` (default: `async`)
- `callbacks` (object, optional): Webhook URLs for workflow events

**Response (Async Mode):**
```json
{
  "workflow_id": "wf_1a2b3c4d",
  "status": "running",
  "session_id": "sess_xyz123",
  "created_at": "2024-01-15T10:30:00Z",
  "estimated_completion": "2024-01-15T10:32:00Z",
  "steps_total": 3,
  "steps_completed": 0,
  "polling_url": "/api/v1/workflows/wf_1a2b3c4d/status"
}
```

**Response (Sync Mode):**
```json
{
  "workflow_id": "wf_1a2b3c4d",
  "status": "completed",
  "session_id": "sess_xyz123",
  "created_at": "2024-01-15T10:30:00Z",
  "completed_at": "2024-01-15T10:31:45Z",
  "steps_executed": 3,
  "final_output": {
    "artifact_id": "art_final_report_001",
    "artifact_type": "final_report",
    "data": {
      "title": "Multi-Agent Systems in Finance",
      "summary": "Comprehensive analysis of...",
      "confidence": 0.92
    }
  },
  "artifacts": [
    {
      "step_id": "research",
      "artifact_id": "art_research_001",
      "artifact_type": "research_snippet"
    },
    {
      "step_id": "verify",
      "artifact_id": "art_verify_001",
      "artifact_type": "claim_verification"
    }
  ],
  "execution_stats": {
    "total_duration_ms": 105000,
    "llm_tokens_used": 15234,
    "cost_usd": 0.23
  }
}
```

### Get Workflow Status

Poll workflow execution status.

**Endpoint:** `GET /api/v1/workflows/{workflow_id}/status`

**Response:**
```json
{
  "workflow_id": "wf_1a2b3c4d",
  "status": "running",
  "current_step": "verify",
  "steps_total": 3,
  "steps_completed": 1,
  "progress_percent": 33,
  "artifacts": [
    {
      "step_id": "research",
      "artifact_id": "art_research_001",
      "artifact_type": "research_snippet",
      "created_at": "2024-01-15T10:30:45Z"
    }
  ],
  "errors": []
}
```

### Cancel Workflow

Stop a running workflow.

**Endpoint:** `POST /api/v1/workflows/{workflow_id}/cancel`

**Response:**
```json
{
  "workflow_id": "wf_1a2b3c4d",
  "status": "cancelled",
  "cancelled_at": "2024-01-15T10:31:00Z",
  "partial_results": {
    "steps_completed": 1,
    "artifacts": ["art_research_001"]
  }
}
```

### List Workflows

Get all workflows for a session.

**Endpoint:** `GET /api/v1/workflows?session_id={session_id}&status={status}&limit={limit}`

**Query Parameters:**
- `session_id` (string, optional): Filter by session
- `status` (string, optional): Filter by status (running, completed, failed, cancelled)
- `limit` (integer, optional): Maximum results (default: 50)
- `offset` (integer, optional): Pagination offset

**Response:**
```json
{
  "workflows": [
    {
      "workflow_id": "wf_1a2b3c4d",
      "manifest_id": "research-workflow",
      "status": "completed",
      "created_at": "2024-01-15T10:30:00Z",
      "completed_at": "2024-01-15T10:31:45Z"
    }
  ],
  "total": 1,
  "limit": 50,
  "offset": 0
}
```

### Spawn Subagent

Create a new subagent with specific capabilities.

**Endpoint:** `POST /api/v1/subagents/spawn`

**Request:**
```json
{
  "role": "research",
  "capabilities": ["web_search", "document_read", "summarize"],
  "system_prompt": "You are a research agent specializing in financial technology.",
  "output_artifact_type": "research_snippet",
  "timeout": 60,
  "session_id": "sess_xyz123",
  "mcp_tools": ["web_search", "github.search_repos"],
  "context": {
    "previous_artifacts": ["art_query_001"]
  }
}
```

**Response:**
```json
{
  "subagent_id": "agent_research_001",
  "role": "research",
  "status": "ready",
  "capabilities": ["web_search", "document_read", "summarize"],
  "mcp_tools": ["web_search", "github.search_repos"],
  "created_at": "2024-01-15T10:30:00Z",
  "expires_at": "2024-01-15T10:31:00Z"
}
```

### Execute Subagent Task

Run a specific task with a spawned subagent.

**Endpoint:** `POST /api/v1/subagents/{subagent_id}/execute`

**Request:**
```json
{
  "task": "Research multi-agent systems in financial services",
  "inputs": {
    "query": "multi-agent systems finance",
    "max_results": 10
  },
  "execution_mode": "sync"
}
```

**Response:**
```json
{
  "subagent_id": "agent_research_001",
  "execution_id": "exec_001",
  "status": "completed",
  "output_artifact": {
    "artifact_id": "art_research_001",
    "artifact_type": "research_snippet",
    "data": {
      "summary": "Multi-agent systems are increasingly used...",
      "sources": [...],
      "confidence": 0.89
    }
  },
  "tools_used": ["web_search"],
  "execution_stats": {
    "duration_ms": 8500,
    "tokens_used": 2341,
    "cost_usd": 0.04
  }
}
```

### Terminate Subagent

Destroy a subagent and release resources.

**Endpoint:** `POST /api/v1/subagents/{subagent_id}/terminate`

**Response:**
```json
{
  "subagent_id": "agent_research_001",
  "status": "terminated",
  "terminated_at": "2024-01-15T10:31:00Z",
  "artifacts_produced": 1,
  "total_executions": 1
}
```

## Memory Service API

Manage typed artifacts and memory persistence.

### Store Artifact

Save a typed artifact with provenance.

**Endpoint:** `POST /api/v1/artifacts`

**Request:**
```json
{
  "artifact_type": "research_snippet",
  "data": {
    "id": "research_001",
    "source": {
      "url": "https://example.com/article",
      "doc_id": "doc_123"
    },
    "text": "Full article text...",
    "summary": "Brief summary...",
    "tags": ["finance", "ai", "multi-agent"],
    "confidence": 0.92
  },
  "provenance": {
    "actor_id": "agent_research_001",
    "actor_type": "subagent",
    "inputs_hash": "sha256:abc123...",
    "tool_ids": ["web_search"],
    "timestamp": "2024-01-15T10:30:45Z"
  },
  "session_id": "sess_xyz123",
  "persist_tiers": ["session", "vector", "structured"]
}
```

**Response:**
```json
{
  "artifact_id": "art_research_001",
  "artifact_type": "research_snippet",
  "stored_at": "2024-01-15T10:30:45Z",
  "storage_tiers": {
    "session": {
      "backend": "redis",
      "key": "sess:xyz123:art:research_001"
    },
    "vector": {
      "backend": "milvus",
      "collection": "artifacts",
      "embedding_id": "emb_001"
    },
    "structured": {
      "backend": "postgres",
      "table": "artifacts",
      "row_id": 1234
    }
  },
  "provenance_id": "prov_001"
}
```

### Retrieve Artifact

Get an artifact by ID.

**Endpoint:** `GET /api/v1/artifacts/{artifact_id}`

**Query Parameters:**
- `include_provenance` (boolean): Include provenance chain (default: false)

**Response:**
```json
{
  "artifact_id": "art_research_001",
  "artifact_type": "research_snippet",
  "data": {
    "id": "research_001",
    "summary": "Brief summary...",
    "confidence": 0.92
  },
  "created_at": "2024-01-15T10:30:45Z",
  "created_by": "agent_research_001",
  "provenance": {
    "actor_id": "agent_research_001",
    "actor_type": "subagent",
    "inputs_hash": "sha256:abc123...",
    "tool_ids": ["web_search"]
  }
}
```

### Query Artifacts (Semantic Search)

Search artifacts using vector similarity.

**Endpoint:** `POST /api/v1/artifacts/query`

**Request:**
```json
{
  "query": "financial regulations for AI systems",
  "artifact_types": ["research_snippet", "claim_verification"],
  "top_k": 5,
  "min_confidence": 0.7,
  "session_id": "sess_xyz123",
  "filters": {
    "tags": ["finance", "compliance"],
    "created_after": "2024-01-01T00:00:00Z"
  }
}
```

**Response:**
```json
{
  "results": [
    {
      "artifact_id": "art_research_001",
      "artifact_type": "research_snippet",
      "similarity_score": 0.94,
      "data": {
        "summary": "Financial regulations require...",
        "confidence": 0.89
      }
    },
    {
      "artifact_id": "art_research_005",
      "artifact_type": "research_snippet",
      "similarity_score": 0.87,
      "data": {
        "summary": "AI compliance standards...",
        "confidence": 0.82
      }
    }
  ],
  "total_matches": 2,
  "query_time_ms": 45
}
```

### Get Provenance Chain

Retrieve full provenance history for an artifact.

**Endpoint:** `GET /api/v1/artifacts/{artifact_id}/provenance`

**Response:**
```json
{
  "artifact_id": "art_final_report_001",
  "provenance_chain": [
    {
      "level": 0,
      "artifact_id": "art_final_report_001",
      "actor_id": "agent_synthesis_001",
      "actor_type": "subagent",
      "timestamp": "2024-01-15T10:31:30Z",
      "inputs": ["art_research_001", "art_verify_001"]
    },
    {
      "level": 1,
      "artifact_id": "art_verify_001",
      "actor_id": "agent_verify_001",
      "actor_type": "subagent",
      "timestamp": "2024-01-15T10:31:00Z",
      "inputs": ["art_research_001"],
      "tools_used": ["fact_check"]
    },
    {
      "level": 2,
      "artifact_id": "art_research_001",
      "actor_id": "agent_research_001",
      "actor_type": "subagent",
      "timestamp": "2024-01-15T10:30:45Z",
      "inputs": ["user_query"],
      "tools_used": ["web_search"]
    }
  ],
  "total_depth": 3
}
```

### Compact Memory

Trigger memory compaction for a session.

**Endpoint:** `POST /api/v1/memory/compact`

**Request:**
```json
{
  "session_id": "sess_xyz123",
  "strategy": "summarize",
  "max_tokens": 8000,
  "keep_recent": 5
}
```

**Response:**
```json
{
  "session_id": "sess_xyz123",
  "compacted_at": "2024-01-15T10:32:00Z",
  "before": {
    "artifacts": 25,
    "total_tokens": 45000
  },
  "after": {
    "artifacts": 10,
    "total_tokens": 7500
  },
  "strategy_used": "summarize",
  "compaction_summary": "Summarized 15 older artifacts into 5 compressed summaries"
}
```

## Code Executor API

Execute skills in sandboxed environments.

### Execute Skill

Run a registered skill with validated inputs.

**Endpoint:** `POST /api/v1/skills/execute`

**Request:**
```json
{
  "skill_name": "text_summarize",
  "inputs": {
    "text": "Long article text to summarize...",
    "style": "bullet-points",
    "max_sentences": 5
  },
  "execution_mode": "sandboxed",
  "timeout": 30,
  "session_id": "sess_xyz123"
}
```

**Response:**
```json
{
  "execution_id": "exec_skill_001",
  "skill_name": "text_summarize",
  "status": "completed",
  "output": {
    "summary": "• Key point 1\n• Key point 2\n• Key point 3",
    "word_count": 150,
    "compression_ratio": 0.15
  },
  "execution_stats": {
    "duration_ms": 1250,
    "memory_used_mb": 45,
    "cpu_percent": 12
  },
  "output_hash": "sha256:def456...",
  "logs": [
    {"level": "info", "message": "Loading text...", "timestamp": "2024-01-15T10:30:00.100Z"},
    {"level": "info", "message": "Summarization complete", "timestamp": "2024-01-15T10:30:01.350Z"}
  ]
}
```

### List Skills

Get all registered skills.

**Endpoint:** `GET /api/v1/skills`

**Query Parameters:**
- `tags` (string, optional): Filter by tags (comma-separated)
- `safety_flags` (string, optional): Filter by safety flags

**Response:**
```json
{
  "skills": [
    {
      "name": "text_summarize",
      "version": "1.0.0",
      "description": "Summarize text in various styles",
      "tags": ["nlp", "text-processing"],
      "safety_flags": ["none"],
      "requires_approval": false,
      "input_schema": "/api/v1/skills/text_summarize/schema/input",
      "output_schema": "/api/v1/skills/text_summarize/schema/output"
    },
    {
      "name": "sentiment_analyzer",
      "version": "1.0.0",
      "description": "Analyze sentiment of text",
      "tags": ["nlp", "sentiment"],
      "safety_flags": ["pii_risk"],
      "requires_approval": false
    }
  ],
  "total": 2
}
```

### Get Skill Schema

Retrieve JSON Schema for skill inputs/outputs.

**Endpoint:** `GET /api/v1/skills/{skill_name}/schema/{input|output}`

**Response:**
```json
{
  "skill_name": "text_summarize",
  "schema_type": "input",
  "schema": {
    "type": "object",
    "properties": {
      "text": {
        "type": "string",
        "description": "Text to summarize"
      },
      "style": {
        "type": "string",
        "enum": ["concise", "bullet-points", "paragraph"],
        "default": "concise"
      },
      "max_sentences": {
        "type": "integer",
        "minimum": 1,
        "maximum": 10,
        "default": 3
      }
    },
    "required": ["text"]
  }
}
```

## MCP Gateway API

Invoke external tools via Model Context Protocol.

### Invoke Tool

Call an MCP tool with parameters.

**Endpoint:** `POST /api/v1/mcp/tools/invoke`

**Request:**
```json
{
  "server_id": "github",
  "tool_name": "search_repos",
  "parameters": {
    "query": "agentic frameworks",
    "limit": 5,
    "sort": "stars"
  },
  "session_id": "sess_xyz123",
  "requester_id": "agent_research_001"
}
```

**Response:**
```json
{
  "invocation_id": "mcp_inv_001",
  "server_id": "github",
  "tool_name": "search_repos",
  "status": "success",
  "result": {
    "total_count": 127,
    "items": [
      {
        "name": "agentic-framework",
        "full_name": "example/agentic-framework",
        "stars": 1250,
        "url": "https://github.com/example/agentic-framework"
      }
    ]
  },
  "execution_stats": {
    "duration_ms": 850,
    "rate_limit_remaining": 59
  },
  "invoked_at": "2024-01-15T10:30:00Z"
}
```

### List MCP Servers

Get all registered MCP servers.

**Endpoint:** `GET /api/v1/mcp/servers`

**Response:**
```json
{
  "servers": [
    {
      "server_id": "github",
      "type": "external",
      "description": "GitHub API integration",
      "tools": ["search_repos", "create_issue", "list_prs"],
      "scopes": ["repo:read", "issues:write"],
      "rate_limit": 60,
      "status": "healthy"
    },
    {
      "server_id": "filesystem",
      "type": "builtin",
      "description": "Local file system operations",
      "tools": ["file_read", "file_write", "file_list"],
      "scopes": ["path:/tmp/*"],
      "status": "healthy"
    }
  ],
  "total": 2
}
```

### Get Tool Catalog

Browse available tools from MCP servers.

**Endpoint:** `GET /api/v1/mcp/catalog?server_id={server_id}`

**Response:**
```json
{
  "catalog": [
    {
      "server_id": "github",
      "tool_name": "search_repos",
      "description": "Search GitHub repositories",
      "parameters_schema": {
        "type": "object",
        "properties": {
          "query": {"type": "string"},
          "limit": {"type": "integer", "default": 10}
        },
        "required": ["query"]
      },
      "scopes_required": ["repo:read"]
    }
  ],
  "total": 1
}
```

## Authentication

All API requests require authentication via API key or JWT token.

### API Key Authentication

Include API key in request header:

```http
Authorization: Bearer sk-agent-abc123...
```

### JWT Token Authentication

For user-scoped requests:

```http
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

### Generate API Key

**Endpoint:** `POST /api/v1/auth/api-keys`

**Request:**
```json
{
  "name": "Production Workflow Executor",
  "scopes": ["workflows:execute", "artifacts:read"],
  "expires_in_days": 90
}
```

**Response:**
```json
{
  "api_key": "sk-agent-abc123def456...",
  "key_id": "key_001",
  "name": "Production Workflow Executor",
  "scopes": ["workflows:execute", "artifacts:read"],
  "created_at": "2024-01-15T10:00:00Z",
  "expires_at": "2024-04-15T10:00:00Z"
}
```

## Error Handling

### Error Response Format

All errors follow a consistent structure:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid input: 'max_results' must be between 1 and 100",
    "details": {
      "field": "max_results",
      "value": 500,
      "constraint": "maximum: 100"
    },
    "request_id": "req_xyz789",
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

### Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `VALIDATION_ERROR` | 400 | Invalid request parameters |
| `AUTHENTICATION_ERROR` | 401 | Missing or invalid credentials |
| `AUTHORIZATION_ERROR` | 403 | Insufficient permissions |
| `NOT_FOUND` | 404 | Resource not found |
| `TIMEOUT_ERROR` | 408 | Request timeout |
| `RATE_LIMIT_EXCEEDED` | 429 | Too many requests |
| `EXECUTION_ERROR` | 500 | Skill/workflow execution failed |
| `SERVICE_UNAVAILABLE` | 503 | Service temporarily unavailable |

### Retry Strategy

Use exponential backoff for transient errors:

```python
import time
import requests

def api_call_with_retry(url, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=data)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:  # Rate limit
                wait_time = 2 ** attempt  # Exponential backoff
                time.sleep(wait_time)
            elif e.response.status_code >= 500:  # Server error
                time.sleep(2 ** attempt)
            else:
                raise
    raise Exception("Max retries exceeded")
```

## Python SDK

High-level Python SDK for easier integration.

### Installation

```bash
pip install agentic-framework
```

### Orchestrator Client

```python
from agentic_framework import OrchestratorClient
import asyncio

async def main():
    # Initialize client
    client = OrchestratorClient(
        base_url="http://localhost:8000",
        api_key="sk-agent-abc123..."
    )

    # Start workflow
    workflow = await client.start_workflow(
        manifest_id="research-workflow",
        input_data={
            "query": "multi-agent systems",
            "max_results": 10
        }
    )

    print(f"Workflow started: {workflow.id}")

    # Poll for completion
    result = await workflow.wait_for_completion(timeout=120)

    print(f"Final output: {result.final_output}")
    print(f"Artifacts: {len(result.artifacts)}")

    # Get specific artifact
    artifact = await client.get_artifact(result.artifacts[0].id)
    print(f"Artifact data: {artifact.data}")

asyncio.run(main())
```

### Memory Client

```python
from agentic_framework import MemoryClient

async def main():
    client = MemoryClient(
        base_url="http://localhost:8001",
        api_key="sk-agent-abc123..."
    )

    # Store artifact
    artifact = await client.store_artifact(
        artifact_type="research_snippet",
        data={
            "summary": "AI adoption is accelerating...",
            "confidence": 0.89
        },
        provenance={
            "actor_id": "agent_001",
            "tools_used": ["web_search"]
        }
    )

    # Query similar artifacts
    results = await client.query_artifacts(
        query="AI adoption trends",
        top_k=5,
        min_confidence=0.7
    )

    for result in results:
        print(f"Similarity: {result.similarity_score}")
        print(f"Summary: {result.data['summary']}")

asyncio.run(main())
```

### Skill Executor Client

```python
from agentic_framework import SkillExecutorClient

async def main():
    client = SkillExecutorClient(
        base_url="http://localhost:8002",
        api_key="sk-agent-abc123..."
    )

    # Execute skill
    result = await client.execute_skill(
        skill_name="text_summarize",
        inputs={
            "text": "Long article...",
            "style": "bullet-points",
            "max_sentences": 5
        }
    )

    print(f"Summary: {result.output['summary']}")
    print(f"Duration: {result.stats.duration_ms}ms")

asyncio.run(main())
```

## Examples

### Example 1: Execute Workflow Programmatically

```python
from agentic_framework import OrchestratorClient
import asyncio

async def research_workflow_example():
    client = OrchestratorClient(
        base_url="http://localhost:8000",
        api_key="sk-agent-abc123..."
    )

    # Start workflow
    workflow = await client.start_workflow(
        manifest_path="manifests/research-workflow.yaml",
        input_data={
            "query": "enterprise AI adoption",
            "max_results": 20,
            "include_citations": True
        },
        execution_mode="async"
    )

    # Monitor progress with callbacks
    async for event in workflow.stream_events():
        if event.type == "step_complete":
            print(f"✓ Step {event.step_id} completed")
            print(f"  Artifact: {event.artifact_id}")
        elif event.type == "error":
            print(f"✗ Error in step {event.step_id}: {event.error}")

    # Get final result
    result = await workflow.get_result()

    return result.final_output

asyncio.run(research_workflow_example())
```

### Example 2: Custom Agent Orchestration

```python
from agentic_framework import OrchestratorClient
import asyncio

async def custom_orchestration():
    client = OrchestratorClient(base_url="http://localhost:8000")

    # Spawn research agent
    research_agent = await client.spawn_subagent(
        role="research",
        capabilities=["web_search", "document_read"],
        mcp_tools=["web_search", "github.search_repos"]
    )

    # Execute research task
    research_result = await research_agent.execute(
        task="Research multi-agent systems in healthcare",
        inputs={"max_results": 10}
    )

    # Spawn verification agent
    verify_agent = await client.spawn_subagent(
        role="verification",
        capabilities=["fact_check"],
        context={"previous_artifacts": [research_result.artifact_id]}
    )

    # Verify findings
    verify_result = await verify_agent.execute(
        task="Verify key claims from research",
        inputs={"research_data": research_result.output}
    )

    # Cleanup
    await research_agent.terminate()
    await verify_agent.terminate()

    return {
        "research": research_result.output,
        "verification": verify_result.output
    }

asyncio.run(custom_orchestration())
```

### Example 3: Skill Chaining

```python
from agentic_framework import SkillExecutorClient
import asyncio

async def skill_chain_example():
    client = SkillExecutorClient(base_url="http://localhost:8002")

    text = "Very long article about AI and machine learning..."

    # Step 1: Summarize
    summary_result = await client.execute_skill(
        skill_name="text_summarize",
        inputs={"text": text, "max_sentences": 5}
    )

    # Step 2: Extract entities from summary
    entities_result = await client.execute_skill(
        skill_name="extract_entities",
        inputs={"text": summary_result.output["summary"]}
    )

    # Step 3: Analyze sentiment
    sentiment_result = await client.execute_skill(
        skill_name="sentiment_analyzer",
        inputs={"text": summary_result.output["summary"]}
    )

    return {
        "summary": summary_result.output["summary"],
        "entities": entities_result.output["entities"],
        "sentiment": sentiment_result.output["sentiment"]
    }

asyncio.run(skill_chain_example())
```

### Example 4: Memory and Provenance Tracking

```python
from agentic_framework import MemoryClient
import asyncio

async def provenance_example():
    client = MemoryClient(base_url="http://localhost:8001")

    # Store initial research artifact
    research_artifact = await client.store_artifact(
        artifact_type="research_snippet",
        data={
            "summary": "AI adoption accelerating in healthcare",
            "confidence": 0.92
        },
        provenance={
            "actor_id": "agent_research_001",
            "tools_used": ["web_search"]
        }
    )

    # Store verification artifact (referencing research)
    verify_artifact = await client.store_artifact(
        artifact_type="claim_verification",
        data={
            "verdict": "verified",
            "confidence": 0.89
        },
        provenance={
            "actor_id": "agent_verify_001",
            "inputs": [research_artifact.id],
            "tools_used": ["fact_check"]
        }
    )

    # Get full provenance chain
    chain = await client.get_provenance_chain(verify_artifact.id)

    print("Provenance Chain:")
    for level, item in enumerate(chain):
        print(f"  Level {level}: {item.actor_id} → {item.artifact_id}")

    return chain

asyncio.run(provenance_example())
```

## Rate Limits

| Endpoint Pattern | Rate Limit | Window |
|-----------------|------------|--------|
| `/api/v1/workflows/*` | 100 requests | 1 minute |
| `/api/v1/subagents/*` | 200 requests | 1 minute |
| `/api/v1/artifacts/*` | 500 requests | 1 minute |
| `/api/v1/skills/*` | 300 requests | 1 minute |
| `/api/v1/mcp/*` | 100 requests | 1 minute |

Rate limit headers in response:
```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1705318260
```

## Webhooks

Configure webhooks for workflow events.

### Webhook Payload

```json
{
  "event": "workflow.step.completed",
  "workflow_id": "wf_1a2b3c4d",
  "step_id": "research",
  "timestamp": "2024-01-15T10:30:45Z",
  "data": {
    "artifact_id": "art_research_001",
    "artifact_type": "research_snippet",
    "duration_ms": 8500
  }
}
```

### Event Types

- `workflow.started`
- `workflow.step.started`
- `workflow.step.completed`
- `workflow.step.failed`
- `workflow.completed`
- `workflow.failed`
- `workflow.cancelled`

## See Also

- [Workflow Manifests](manifests.md) - YAML workflow configuration
- [Skills Guide](skills.md) - Creating custom skills
- [MCP Integration](mcp.md) - External tool integration
- [Examples](../examples/) - Working code examples
