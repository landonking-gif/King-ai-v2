# MCP Integration Guide

Complete guide to integrating external tools via the Model Context Protocol (MCP) into your agent workflows.

## Table of Contents

- [Overview](#overview)
- [MCP Architecture](#mcp-architecture)
- [Server Configuration](#server-configuration)
- [Tool Catalog](#tool-catalog)
- [Tool Invocation Patterns](#tool-invocation-patterns)
- [Security and Governance](#security-and-governance)
- [Available MCP Servers](#available-mcp-servers)
- [Custom MCP Servers](#custom-mcp-servers)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Overview

The Model Context Protocol (MCP) enables agents to access external tools and services in a standardized, secure way. Instead of hardcoding integrations, MCP provides:

- **Unified Interface**: Single protocol for all external tools
- **Dynamic Discovery**: Tools registered in central catalog
- **Scoped Permissions**: Fine-grained access control
- **Rate Limiting**: Prevent abuse and manage costs
- **Audit Logging**: Track all tool invocations

### Key Concepts

**MCP Server**: External service exposing tools via MCP protocol
**Tool**: Individual function provided by an MCP server (e.g., `github.search_repos`)
**Scope**: Permission boundary (e.g., `repo:read`, `path:/tmp/*`)
**Tool Catalog**: Central registry of available tools
**MCP Gateway**: Proxy layer for authentication, rate limiting, and policy enforcement

## MCP Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      Agent Workflow                          │
│  ┌────────────┐    ┌────────────┐    ┌────────────┐         │
│  │  Research  │───→│   Verify   │───→│ Synthesize │         │
│  │   Agent    │    │   Agent    │    │   Agent    │         │
│  └─────┬──────┘    └──────┬─────┘    └──────┬─────┘         │
│        │                  │                  │               │
└────────┼──────────────────┼──────────────────┼───────────────┘
         │                  │                  │
         ▼                  ▼                  ▼
┌──────────────────────────────────────────────────────────────┐
│                      MCP Gateway                             │
│  • Authentication   • Rate Limiting   • PII Filtering        │
│  • Scope Validation • Audit Logging  • Policy Enforcement   │
└─────┬────────┬─────────┬──────────┬──────────┬──────────────┘
      │        │         │          │          │
      ▼        ▼         ▼          ▼          ▼
┌──────────┐ ┌──────┐ ┌──────┐ ┌──────────┐ ┌──────────┐
│ GitHub   │ │ Slack│ │ Jira │ │Filesystem│ │Web Search│
│  Server  │ │Server│ │Server│ │  Server  │ │  Server  │
└──────────┘ └──────┘ └──────┘ └──────────┘ └──────────┘
```

### Invocation Modes

**1. Orchestrated Mode (Default)**

Agent → Lead Agent → MCP Gateway → MCP Server

- Lead agent validates and approves all tool calls
- Maximum security and audit trail
- Higher latency (2-hop)

**2. Scoped Direct Mode**

Agent → MCP Gateway → MCP Server (with ephemeral token)

- Lead agent mints short-lived token with scopes
- Agent calls gateway directly
- Lower latency, still secure

## Server Configuration

MCP servers are configured in `mcp_config.yaml` or via the MCP Gateway API.

### Configuration File Structure

```yaml
# mcp_config.yaml

mcp_servers:
  # Built-in servers (run in-process)
  filesystem:
    type: builtin
    description: Local file system operations
    tools:
      - file_read
      - file_write
      - file_list
      - file_delete
    scopes:
      - path:/tmp/*
      - path:/data/*
    rate_limit: 100  # requests per minute

  # External servers (HTTP/gRPC)
  github:
    type: external
    description: GitHub API integration
    endpoint: https://api.github.com
    protocol: rest  # or grpc
    tools:
      - search_repos
      - create_issue
      - list_prs
      - get_file_content
      - create_pr
    scopes:
      - repo:read
      - issues:write
      - pr:write
    auth:
      type: bearer_token
      env_var: GITHUB_TOKEN
    rate_limit: 60
    timeout_seconds: 30

  # Custom MCP server
  custom_analytics:
    type: external
    endpoint: http://localhost:9000
    protocol: grpc
    tools:
      - query_metrics
      - aggregate_data
    scopes:
      - analytics:read
    auth:
      type: api_key
      header: X-API-Key
      env_var: ANALYTICS_API_KEY
    rate_limit: 30

# Global policies
policies:
  pii_filtering: true
  require_approval_for:
    - file_delete
    - file_write
    - github.delete_repo
  max_retries: 3
  timeout_seconds: 30
  log_all_invocations: true

# Tool catalog discovery
catalog:
  auto_discover: true
  refresh_interval: 3600  # seconds
  sources:
    - type: registry
      url: https://mcp-registry.anthropic.com
    - type: local
      path: ./mcp-servers/
```

### Server Types

#### Built-In Servers

Run in-process for maximum performance:

```yaml
filesystem:
  type: builtin
  tools:
    - file_read
    - file_write
  scopes:
    - path:/allowed/directory/*
```

Available built-in servers:
- `filesystem`: Local file operations
- `environment`: Environment variable access
- `http_client`: Simple HTTP requests

#### External Servers

Connect to external services:

```yaml
github:
  type: external
  endpoint: https://api.github.com
  protocol: rest
  auth:
    type: bearer_token
    env_var: GITHUB_TOKEN
```

#### Custom Servers

Deploy your own MCP-compliant servers:

```yaml
my_service:
  type: external
  endpoint: http://my-service:8080
  protocol: grpc
  tls:
    enabled: true
    ca_cert: /path/to/ca.crt
```

## Tool Catalog

The tool catalog provides discovery and metadata for all available tools.

### Catalog Structure

```json
{
  "catalog_version": "1.0.0",
  "last_updated": "2024-01-15T10:00:00Z",
  "servers": [
    {
      "server_id": "github",
      "type": "external",
      "status": "healthy",
      "tools": [
        {
          "tool_id": "github.search_repos",
          "name": "search_repos",
          "description": "Search GitHub repositories by query",
          "category": "search",
          "parameters_schema": {
            "type": "object",
            "properties": {
              "query": {
                "type": "string",
                "description": "Search query"
              },
              "limit": {
                "type": "integer",
                "default": 10,
                "maximum": 100
              },
              "sort": {
                "type": "string",
                "enum": ["stars", "forks", "updated"],
                "default": "stars"
              }
            },
            "required": ["query"]
          },
          "output_schema": {
            "type": "object",
            "properties": {
              "total_count": {"type": "integer"},
              "items": {
                "type": "array",
                "items": {
                  "type": "object",
                  "properties": {
                    "name": {"type": "string"},
                    "full_name": {"type": "string"},
                    "stars": {"type": "integer"},
                    "url": {"type": "string"}
                  }
                }
              }
            }
          },
          "scopes_required": ["repo:read"],
          "rate_limit": 60,
          "estimated_latency_ms": 500
        }
      ]
    }
  ]
}
```

### Querying the Catalog

**List all servers:**
```bash
curl http://localhost:8080/api/v1/mcp/servers
```

**List tools from specific server:**
```bash
curl http://localhost:8080/api/v1/mcp/catalog?server_id=github
```

**Search tools by category:**
```bash
curl http://localhost:8080/api/v1/mcp/catalog/search?category=search&tags=git
```

### Catalog Discovery

Enable auto-discovery to fetch tools from registry:

```yaml
catalog:
  auto_discover: true
  refresh_interval: 3600
  sources:
    - type: registry
      url: https://mcp-registry.anthropic.com
```

## Tool Invocation Patterns

### Pattern 1: Direct Tool Call

Invoke a specific tool with parameters:

```python
from agentic_framework import MCPClient
import asyncio

async def main():
    client = MCPClient(base_url="http://localhost:8080")

    # Invoke tool
    result = await client.invoke_tool(
        server_id="github",
        tool_name="search_repos",
        parameters={
            "query": "agentic frameworks",
            "limit": 5,
            "sort": "stars"
        }
    )

    print(f"Found {result['total_count']} repositories")
    for repo in result['items']:
        print(f"  - {repo['full_name']} ({repo['stars']}⭐)")

asyncio.run(main())
```

### Pattern 2: Tool Binding in Manifests

Bind tools to agents in workflow manifests:

```yaml
# workflow.yaml
manifest_id: research-workflow
name: Research Workflow
version: "1.0.0"

steps:
  - id: research
    role: research
    capabilities: [summarize]  # Skills
    inputs:
      - name: query
        source: user_input
    outputs: [research_snippet]
    timeout: 60

# Tool binding
tools:
  catalog_ids:
    - github.search_repos    # Specific tool
    - web_search.*           # All tools from web_search server
    - filesystem             # All tools from filesystem server

  # Optional: Tool-specific configuration
  config:
    github:
      scopes: [repo:read]
      rate_limit: 30  # Override global limit

  # Optional: Filters
  filters:
    allowed_tools:
      - github.search_repos
      - github.list_prs
    blocked_tools:
      - github.delete_repo
```

### Pattern 3: Agent-Initiated Tool Call

Agent requests tool invocation during execution:

```python
# In agent execution loop
async def research_agent_logic(context: AgentContext):
    """Research agent with MCP tool access."""

    # Agent decides it needs GitHub search
    tool_result = await context.invoke_mcp_tool(
        server_id="github",
        tool_name="search_repos",
        parameters={"query": context.inputs["research_query"]}
    )

    # Process results
    repos = tool_result["items"]
    summaries = [
        f"{repo['name']}: {repo['description']}"
        for repo in repos[:5]
    ]

    return {
        "artifact_type": "research_snippet",
        "data": {
            "summary": "\n".join(summaries),
            "sources": [r["url"] for r in repos],
            "confidence": 0.85
        }
    }
```

### Pattern 4: Scoped Direct Access

Lead agent mints ephemeral token for subagent:

```python
async def spawn_agent_with_mcp_access():
    orchestrator = OrchestratorClient()

    # Mint scoped token (15-minute TTL)
    mcp_token = await orchestrator.mint_mcp_token(
        scopes=["repo:read", "issues:write"],
        servers=["github"],
        ttl_seconds=900
    )

    # Spawn subagent with direct access
    agent = await orchestrator.spawn_subagent(
        role="research",
        capabilities=["web_search"],
        mcp_token=mcp_token  # Agent can call gateway directly
    )

    # Agent uses token to call MCP Gateway
    result = await agent.execute_with_mcp(
        task="Research GitHub trends",
        mcp_calls=[
            {
                "server": "github",
                "tool": "search_repos",
                "params": {"query": "trending"}
            }
        ]
    )

    return result
```

## Security and Governance

### Authentication

MCP Gateway supports multiple authentication methods:

**1. Bearer Token (API Keys)**
```yaml
auth:
  type: bearer_token
  env_var: GITHUB_TOKEN
```

**2. OAuth 2.0**
```yaml
auth:
  type: oauth2
  client_id_env: OAUTH_CLIENT_ID
  client_secret_env: OAUTH_CLIENT_SECRET
  token_url: https://oauth.example.com/token
  scopes: [read, write]
```

**3. Mutual TLS**
```yaml
auth:
  type: mtls
  client_cert: /path/to/client.crt
  client_key: /path/to/client.key
  ca_cert: /path/to/ca.crt
```

### Scopes and Permissions

Define what resources tools can access:

**File System Scopes:**
```yaml
scopes:
  - path:/tmp/*              # Allow /tmp directory
  - path:/data/*.json        # Only JSON files in /data
  - path:!/etc/*             # Explicitly deny /etc
```

**API Scopes:**
```yaml
scopes:
  - repo:read                # Read repositories
  - issues:write             # Create/edit issues
  - pr:write                 # Create pull requests
```

**Wildcard Scopes:**
```yaml
scopes:
  - github:*                 # All GitHub operations
  - web:read                 # All read-only web operations
```

### Rate Limiting

Prevent abuse with multiple rate limit strategies:

**Per-Server Limits:**
```yaml
github:
  rate_limit: 60  # 60 requests per minute
```

**Per-Tool Limits:**
```yaml
tools:
  github.search_repos:
    rate_limit: 30  # More restrictive for expensive operation
```

**Per-Session Limits:**
```yaml
policies:
  session_limits:
    max_mcp_calls_per_workflow: 100
    max_cost_usd: 5.0
```

### PII Filtering

Automatically scrub sensitive data:

```yaml
policies:
  pii_filtering: true
  pii_patterns:
    - type: ssn
      pattern: '\d{3}-\d{2}-\d{4}'
      action: redact
    - type: email
      pattern: '[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
      action: hash
    - type: credit_card
      pattern: '\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}'
      action: reject  # Reject entire request
```

### Approval Workflows

Require human approval for dangerous operations:

```yaml
policies:
  require_approval_for:
    - file_delete
    - file_write
    - github.delete_repo
    - slack.send_message  # Prevent spam

  approval:
    timeout_seconds: 3600  # 1 hour
    on_timeout: reject
    approvers:
      - role: engineering_lead
      - email: security@company.com
    notification:
      channels: [slack, email]
```

**Approval Flow:**

1. Agent requests tool invocation
2. Gateway pauses execution, sends approval request
3. Approver receives notification with context
4. Approver reviews and approves/rejects
5. Execution continues or fails based on decision

### Audit Logging

Log all MCP invocations for compliance:

```yaml
policies:
  audit_logging:
    enabled: true
    backends:
      - type: postgres
        table: mcp_audit_log
      - type: s3
        bucket: audit-logs
        prefix: mcp/
    fields:
      - timestamp
      - server_id
      - tool_name
      - requester_id
      - parameters_hash
      - result_hash
      - duration_ms
      - status
```

**Audit Log Entry:**
```json
{
  "log_id": "log_001",
  "timestamp": "2024-01-15T10:30:45Z",
  "server_id": "github",
  "tool_name": "search_repos",
  "requester_id": "agent_research_001",
  "workflow_id": "wf_abc123",
  "session_id": "sess_xyz",
  "parameters": {
    "query": "agentic frameworks",
    "limit": 5
  },
  "parameters_hash": "sha256:abc123...",
  "result_summary": "Found 127 repositories",
  "result_hash": "sha256:def456...",
  "duration_ms": 850,
  "status": "success",
  "scopes_used": ["repo:read"],
  "rate_limit_remaining": 59
}
```

## Available MCP Servers

### Official Servers

**GitHub**
```yaml
github:
  tools: [search_repos, create_issue, list_prs, get_file_content, create_pr]
  scopes: [repo:read, issues:write, pr:write]
  install: npm install -g @modelcontextprotocol/server-github
```

**Slack**
```yaml
slack:
  tools: [send_message, list_channels, get_thread, create_channel]
  scopes: [channels:read, chat:write]
  install: npm install -g @modelcontextprotocol/server-slack
```

**Jira**
```yaml
jira:
  tools: [create_issue, update_issue, search_issues, get_project]
  scopes: [issues:read, issues:write]
  install: npm install -g @modelcontextprotocol/server-jira
```

**PostgreSQL**
```yaml
postgres:
  tools: [query, execute, get_schema]
  scopes: [db:read, db:write]
  install: npm install -g @modelcontextprotocol/server-postgres
```

**Firecrawl (Web Scraping)**
```yaml
firecrawl:
  tools: [scrape_url, crawl_site, extract_content]
  scopes: [web:read, web:extract]
  install: npm install -g @mendableai/firecrawl-mcp-server
```

### Community Servers

Browse community servers at: https://mcp-registry.anthropic.com

## Custom MCP Servers

Create your own MCP-compliant server:

### Server Implementation (Python)

```python
"""
Custom MCP Server Example
Exposes analytics tools via MCP protocol
"""

from typing import Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()


class ToolInvocation(BaseModel):
    tool_name: str
    parameters: Dict[str, Any]


class ToolResult(BaseModel):
    status: str
    result: Dict[str, Any]
    duration_ms: float


# Tool catalog
TOOLS = {
    "query_metrics": {
        "description": "Query analytics metrics",
        "parameters_schema": {
            "type": "object",
            "properties": {
                "metric_name": {"type": "string"},
                "start_date": {"type": "string"},
                "end_date": {"type": "string"}
            },
            "required": ["metric_name"]
        }
    },
    "aggregate_data": {
        "description": "Aggregate data by dimension",
        "parameters_schema": {
            "type": "object",
            "properties": {
                "dimension": {"type": "string"},
                "aggregation": {"type": "string", "enum": ["sum", "avg", "count"]}
            },
            "required": ["dimension", "aggregation"]
        }
    }
}


@app.get("/catalog")
async def get_catalog():
    """Return tool catalog."""
    return {
        "server_id": "custom_analytics",
        "server_version": "1.0.0",
        "tools": TOOLS
    }


@app.post("/invoke")
async def invoke_tool(invocation: ToolInvocation) -> ToolResult:
    """Invoke a tool."""
    if invocation.tool_name not in TOOLS:
        raise HTTPException(status_code=404, detail="Tool not found")

    # Route to appropriate handler
    if invocation.tool_name == "query_metrics":
        result = query_metrics(invocation.parameters)
    elif invocation.tool_name == "aggregate_data":
        result = aggregate_data(invocation.parameters)
    else:
        raise HTTPException(status_code=400, detail="Unknown tool")

    return ToolResult(
        status="success",
        result=result,
        duration_ms=123.45
    )


def query_metrics(params: Dict[str, Any]) -> Dict[str, Any]:
    """Query metrics implementation."""
    metric_name = params["metric_name"]

    # Mock implementation
    return {
        "metric_name": metric_name,
        "value": 12345,
        "unit": "requests",
        "timestamp": "2024-01-15T10:00:00Z"
    }


def aggregate_data(params: Dict[str, Any]) -> Dict[str, Any]:
    """Aggregate data implementation."""
    dimension = params["dimension"]
    aggregation = params["aggregation"]

    # Mock implementation
    return {
        "dimension": dimension,
        "aggregation": aggregation,
        "results": [
            {"key": "value1", "count": 100},
            {"key": "value2", "count": 200}
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)
```

### Registering Custom Server

```yaml
# Add to mcp_config.yaml
mcp_servers:
  custom_analytics:
    type: external
    endpoint: http://localhost:9000
    protocol: rest
    tools:
      - query_metrics
      - aggregate_data
    scopes:
      - analytics:read
```

## Best Practices

### 1. Scope Minimization

Grant minimum necessary scopes:

```yaml
# Bad: Too permissive
scopes:
  - github:*

# Good: Specific scopes
scopes:
  - repo:read
  - issues:write
```

### 2. Rate Limit Tuning

Set appropriate limits based on usage:

```yaml
# For expensive operations
github.search_repos:
  rate_limit: 30  # Lower limit

# For cheap operations
github.get_issue:
  rate_limit: 100  # Higher limit
```

### 3. Error Handling

Handle MCP errors gracefully:

```python
async def call_mcp_tool(server_id: str, tool_name: str, params: dict):
    try:
        result = await mcp_client.invoke_tool(server_id, tool_name, params)
        return result

    except RateLimitError as e:
        # Wait and retry
        await asyncio.sleep(e.retry_after)
        return await call_mcp_tool(server_id, tool_name, params)

    except AuthenticationError:
        # Refresh token
        await refresh_mcp_auth()
        return await call_mcp_tool(server_id, tool_name, params)

    except ToolNotFoundError:
        # Fallback to alternative
        return await call_alternative_tool(params)

    except TimeoutError:
        # Return partial result
        return {"status": "timeout", "partial_result": ...}
```

### 4. Caching

Cache MCP results when appropriate:

```python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=1000)
async def cached_mcp_call(server_id: str, tool_name: str, params_hash: str):
    """Cached MCP invocation."""
    # Actual params passed separately
    return await mcp_client.invoke_tool(server_id, tool_name, params)

async def call_with_cache(server_id: str, tool_name: str, params: dict):
    params_hash = hashlib.sha256(str(params).encode()).hexdigest()
    return await cached_mcp_call(server_id, tool_name, params_hash)
```

### 5. Monitoring

Track MCP usage metrics:

```python
from prometheus_client import Counter, Histogram

mcp_calls = Counter(
    'mcp_calls_total',
    'Total MCP tool invocations',
    ['server_id', 'tool_name', 'status']
)

mcp_duration = Histogram(
    'mcp_duration_seconds',
    'MCP call duration',
    ['server_id', 'tool_name']
)

async def monitored_mcp_call(server_id: str, tool_name: str, params: dict):
    with mcp_duration.labels(server_id=server_id, tool_name=tool_name).time():
        try:
            result = await mcp_client.invoke_tool(server_id, tool_name, params)
            mcp_calls.labels(server_id, tool_name, status="success").inc()
            return result
        except Exception as e:
            mcp_calls.labels(server_id, tool_name, status="error").inc()
            raise
```

## Troubleshooting

### Common Issues

**1. Authentication Failures**

```
Error: MCP_AUTH_ERROR: Invalid credentials for server 'github'
```

**Solution:**
- Verify environment variable is set: `echo $GITHUB_TOKEN`
- Check token has required scopes
- Regenerate token if expired

**2. Rate Limit Exceeded**

```
Error: MCP_RATE_LIMIT: Rate limit exceeded for 'github.search_repos' (60/min)
```

**Solution:**
- Implement exponential backoff
- Reduce request frequency
- Use caching for repeated queries
- Increase rate limit in config (if server allows)

**3. Scope Violations**

```
Error: MCP_SCOPE_VIOLATION: Tool 'file_write' requires scope 'path:/data/*' but agent has 'path:/tmp/*'
```

**Solution:**
- Add required scope to server config
- Request scoped token from orchestrator
- Use different tool with narrower scope

**4. Tool Not Found**

```
Error: MCP_TOOL_NOT_FOUND: Tool 'github.invalid_tool' not found in catalog
```

**Solution:**
- Check tool name spelling
- Refresh catalog: `kautilya mcp refresh-catalog`
- Verify server is running: `kautilya mcp test github`

**5. Timeout Errors**

```
Error: MCP_TIMEOUT: Tool invocation exceeded 30s timeout
```

**Solution:**
- Increase timeout in config
- Check network connectivity
- Use streaming for long-running operations

### Debugging

Enable debug logging:

```yaml
# mcp_config.yaml
logging:
  level: DEBUG
  log_requests: true
  log_responses: true
```

Test MCP connection:

```bash
# Test server connectivity
kautilya mcp test github

# List available tools
kautilya mcp list github

# Invoke tool manually
kautilya mcp invoke github search_repos '{"query": "test"}'
```

## See Also

- [Workflow Manifests](manifests.md) - Using MCP tools in workflows
- [API Reference](api-reference.md) - MCP Gateway API
- [Skills Guide](skills.md) - Creating custom skills
- [Examples](../examples/04-mcp-integration/) - MCP integration examples
- [MCP Specification](https://modelcontextprotocol.io) - Official MCP protocol documentation
