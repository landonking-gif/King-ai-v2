# MCP Integration Example

This example demonstrates how to integrate external tools via the Model Context Protocol (MCP) into your agent workflows.

## Overview

Shows how to:
- Configure MCP servers
- Define tool scopes and permissions
- Set rate limits
- Invoke tools from agents
- Apply security policies

## Prerequisites

```bash
# Install PyYAML for configuration parsing
pip install pyyaml

# No actual MCP servers required for this demo
```

## Running the Example

```bash
# From the repository root
cd examples/04-mcp-integration
python run.py
```

## What This Example Shows

1. **MCP Configuration**: How to configure multiple MCP servers
2. **Tool Scopes**: Fine-grained permission control
3. **Rate Limiting**: Throttling requests per tool/server
4. **Tool Invocation**: How agents call external tools
5. **Security Policies**: PII filtering, approvals, timeouts

## MCP Configuration

The `mcp_config.yaml` file defines MCP servers and their tools:

### Built-in Servers

```yaml
filesystem:
  type: builtin
  tools:
    - file_read
    - file_write
    - file_list
  scopes:
    - path:/tmp/*
    - path:/data/*
  rate_limit: 100
```

### External Servers

```yaml
github:
  type: external
  endpoint: https://api.github.com
  tools:
    - search_repos
    - create_issue
    - list_prs
  scopes:
    - repo:read
    - issues:write
  auth:
    type: bearer_token
    env_var: GITHUB_TOKEN
  rate_limit: 60
```

## Expected Output

```
============================================================
MCP Integration Example
============================================================

📋 Available MCP Servers:

   filesystem:
      Type: builtin
      Description: Local file system operations
      Tools: file_read, file_write, file_list, file_delete
      Rate Limit: 100/min

   github:
      Type: external
      Description: GitHub API integration
      Tools: search_repos, create_issue, list_prs, get_file_content
      Rate Limit: 60/min

   web_search:
      Type: external
      Description: Web search capability
      Tools: search, get_webpage
      Rate Limit: 30/min

   firecrawl_mcp:
      Type: external
      Description: Advanced web scraping
      Tools: scrape_url, crawl_site, extract_content
      Rate Limit: 20/min

============================================================
MCP Tool Usage Patterns
============================================================

[Pattern 1] GitHub Repository Search
   📡 Calling github.search_repos
      Parameters: {'query': 'agentic frameworks', 'limit': 5}
   ✓ Found 2 repositories
      - example/agentic-framework (1250⭐)
      - example/llm-orchestrator (890⭐)

[Pattern 2] File System Access
   📡 Calling filesystem.file_read
      Parameters: {'path': '/data/config.json'}
   ✓ Read 156 bytes from /data/config.json

[Pattern 3] Web Search
   📡 Calling web_search.search
      Parameters: {'query': 'multi-agent systems', 'max_results': 10}
   ✓ Found 2 results for 'multi-agent systems'

[Pattern 4] Web Scraping
   📡 Calling firecrawl_mcp.scrape_url
      Parameters: {'url': 'https://example.com/article'}
   ✓ Scraped 'Example Page Title' (1500 words)

============================================================
Security & Governance
============================================================

✓ PII Filtering: True
✓ Max Retries: 3
✓ Timeout: 30s
✓ Approval Required For:
   - file_delete
   - file_write
```

## Key Concepts

### 1. Tool Scopes

Scopes define what resources a tool can access:

```yaml
scopes:
  - repo:read       # Read repositories
  - issues:write    # Write issues
  - path:/tmp/*     # Access /tmp directory
  - web:read        # Read web content
```

### 2. Rate Limiting

Prevent abuse and manage costs:

```yaml
rate_limit: 60  # Maximum requests per minute
```

### 3. Authentication

Secure API access:

```yaml
auth:
  type: bearer_token
  env_var: GITHUB_TOKEN
```

### 4. Security Policies

```yaml
policies:
  pii_filtering: true  # Filter sensitive data
  require_approval_for:
    - file_delete
    - file_write
  max_retries: 3
  timeout_seconds: 30
```

## Tool Binding Patterns

### Pattern 1: Specific Tool

```yaml
# In manifest
tools:
  catalog_ids:
    - github.search_repos
```

### Pattern 2: Wildcard (All Tools from Server)

```yaml
tools:
  catalog_ids:
    - github:*
    - firecrawl_mcp:*
```

### Pattern 3: Multiple Servers

```yaml
tools:
  catalog_ids:
    - filesystem
    - github
    - web_search
```

## Real MCP Server Setup

### 1. Install MCP Servers

```bash
# GitHub MCP Server
npm install -g @modelcontextprotocol/server-github

# Firecrawl MCP Server
npm install -g @mendableai/firecrawl-mcp-server

# File System MCP Server (built-in)
# No installation needed
```

### 2. Configure Environment Variables

```bash
export GITHUB_TOKEN="ghp_..."
export FIRECRAWL_API_KEY="fc-..."
```

### 3. Start MCP Gateway

```bash
# Using Docker
docker-compose up mcp-gateway

# Or manually
cd mcp-gateway
python -m service.main
```

### 4. Use in Agent

```python
# Agent binds tools from MCP
agent = await orchestrator.spawn_agent(
    role="research",
    capabilities=[],  # Skills
    mcp_tools=[
        "github.search_repos",
        "web_search.*",
        "firecrawl_mcp.scrape_url"
    ]
)

# Agent can now use these tools
result = await agent.invoke_tool(
    "github.search_repos",
    {"query": "agentic frameworks"}
)
```

## Available MCP Servers

| Server | Tools | Use Case |
|--------|-------|----------|
| **filesystem** | file_read, file_write, file_list | Local file operations |
| **github** | search_repos, create_issue, list_prs | GitHub integration |
| **web_search** | search, get_webpage | Web search |
| **firecrawl** | scrape_url, crawl_site | Web scraping |
| **slack** | send_message, list_channels | Slack integration |
| **postgres** | query, execute | Database access |

## Security Best Practices

1. **Principle of Least Privilege**: Only grant necessary scopes
2. **Rate Limiting**: Prevent abuse and manage API costs
3. **PII Filtering**: Automatically scrub sensitive data
4. **Approval Workflows**: Require approval for dangerous operations
5. **Audit Logging**: Log all tool invocations for compliance

## Next Steps

- Deploy real MCP servers: See mcp-gateway/README.md
- Configure authentication: Set environment variables
- Browse MCP catalog: https://mcp-registry.example.com
- Read MCP specification: https://modelcontextprotocol.io
- See tools/kautilya/ for MCP management commands
