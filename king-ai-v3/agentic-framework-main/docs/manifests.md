# Workflow Manifests Guide

Complete guide to creating and managing YAML workflow manifests for multi-agent orchestration.

## Table of Contents

- [Overview](#overview)
- [Manifest Structure](#manifest-structure)
- [Step Configuration](#step-configuration)
- [Input/Output Routing](#inputoutput-routing)
- [Memory Configuration](#memory-configuration)
- [Tool Binding](#tool-binding)
- [Policies](#policies)
- [Examples](#examples)
- [Best Practices](#best-practices)

## Overview

Workflow manifests are YAML files that define how multiple agents collaborate to accomplish complex tasks. They provide a declarative, version-controlled way to orchestrate agent workflows.

### Key Benefits

- **Declarative**: Define what you want, not how to achieve it
- **Version Controlled**: Track changes in Git
- **Reusable**: Share workflows across projects
- **Testable**: Validate before execution
- **Auditable**: Full execution history

## Manifest Structure

### Minimal Manifest

```yaml
manifest_id: my-workflow
name: My Workflow
version: "1.0.0"

steps:
  - id: process
    role: processor
    capabilities: []
    inputs:
      - name: data
        source: user_input
    outputs: [result]
    timeout: 30
```

### Complete Manifest

```yaml
manifest_id: research-workflow
name: Research Workflow
version: "1.0.0"
description: Multi-step research with verification

metadata:
  author: John Doe
  created: 2024-01-01
  tags: [research, verification]

steps:
  - id: research
    role: research
    description: Gather information
    capabilities: [web_search, summarize]
    inputs:
      - name: query
        source: user_input
        description: Research query
        required: true
    outputs: [research_snippet]
    timeout: 60
    retries: 3

  - id: verify
    role: verification
    description: Verify facts
    capabilities: [fact_check]
    inputs:
      - name: research_data
        source: previous_step
        step: research
        artifact: research_snippet
    outputs: [claim_verification]
    timeout: 45

  - id: synthesize
    role: synthesis
    description: Create final report
    capabilities: [text_generation]
    inputs:
      - name: research
        source: step
        step: research
        artifact: research_snippet
      - name: verification
        source: previous_step
    outputs: [final_report]
    timeout: 30

memory:
  persist_on:
    - step_complete
    - workflow_complete
  compaction:
    strategy: summarize
    max_tokens: 8000
    interval: 300

tools:
  catalog_ids:
    - web_search
    - fact_check
    - github:*

policies:
  requires_human_approval: false
  max_retries: 3
  retry_delay: 5
  on_error: fail
  timeout_action: terminate
```

## Step Configuration

### Required Fields

```yaml
- id: unique_step_id          # Must be unique within manifest
  role: agent_role             # Agent role (research, verify, code, etc.)
  capabilities: []             # List of skills/tools
  inputs: []                   # Input definitions
  outputs: []                  # Output artifact types
  timeout: 30                  # Timeout in seconds
```

### Optional Fields

```yaml
- id: step_id
  role: role
  description: "Step description"        # Human-readable description
  capabilities: []
  inputs: []
  outputs: []
  timeout: 30
  retries: 3                             # Number of retry attempts
  retry_delay: 5                         # Delay between retries (seconds)
  on_error: fail                         # fail | skip | retry
  condition: "previous_step.confidence > 0.8"  # Conditional execution
```

### Agent Roles

Predefined roles with specific capabilities:

| Role | Purpose | Typical Capabilities |
|------|---------|---------------------|
| `research` | Information gathering | web_search, summarize, document_read |
| `verification` | Fact checking | fact_check, source_validation |
| `code` | Code generation/analysis | code_generation, code_review |
| `analysis` | Data analysis | data_processing, statistical_analysis |
| `synthesis` | Content creation | text_generation, summarize |
| `custom` | Custom role | User-defined |

### Capabilities

Skills and tools the agent can use:

```yaml
capabilities:
  # Skills (deterministic functions)
  - text_summarize
  - extract_entities
  - code_execution

  # MCP Tools (external services)
  - web_search
  - github.search_repos
  - filesystem.read
```

## Input/Output Routing

### Input Sources

#### 1. User Input

Direct input from workflow invocation:

```yaml
inputs:
  - name: query
    source: user_input
    description: "User's search query"
    required: true
    default: "AI agents"
```

#### 2. Previous Step

Output from immediately preceding step:

```yaml
inputs:
  - name: data
    source: previous_step
    artifact: research_snippet  # Optional: specific artifact type
```

#### 3. Specific Step

Output from any previous step by ID:

```yaml
inputs:
  - name: research
    source: step
    step: research              # Step ID
    artifact: research_snippet
```

#### 4. Environment Variable

```yaml
inputs:
  - name: api_key
    source: env
    env_var: OPENAI_API_KEY
```

#### 5. Constant

```yaml
inputs:
  - name: max_results
    source: constant
    value: 10
```

### Output Artifacts

Each step produces typed artifacts:

```yaml
outputs:
  - research_snippet    # Artifact type from schema registry
  - claim_verification
  - code_patch
```

Available artifact types (defined in `docs/schema_registry/`):
- `research_snippet`
- `claim_verification`
- `code_patch`
- `synthesis_result`
- `final_report`

## Memory Configuration

### Persistence Options

```yaml
memory:
  persist_on:
    - step_complete      # Save after each step
    - workflow_complete  # Save only at end
    - error             # Save on error
    - manual            # Explicit save calls only
```

### Compaction Strategies

Prevent context window overflow:

```yaml
memory:
  compaction:
    strategy: summarize     # summarize | truncate | none
    max_tokens: 8000       # Maximum context size
    interval: 300          # Compaction interval (seconds)
    keep_recent: 5         # Keep N most recent items
```

### Memory Tiers

Configure storage layers:

```yaml
memory:
  tiers:
    session:
      backend: redis
      ttl: 3600
    vector:
      backend: milvus
      collection: workflows
    structured:
      backend: postgres
      table: artifacts
    cold:
      backend: s3
      bucket: workflow-archives
```

## Tool Binding

### Catalog IDs

Reference tools from MCP catalog:

```yaml
tools:
  catalog_ids:
    - web_search           # Specific tool
    - github.search_repos  # Server.tool format
    - github:*             # All tools from server (wildcard)
    - filesystem           # All filesystem tools
```

### Tool Configuration

```yaml
tools:
  catalog_ids:
    - github:*

  config:
    github:
      scopes: [repo:read, issues:write]
      rate_limit: 60
      auth:
        type: bearer_token
        env_var: GITHUB_TOKEN
```

### Tool Filters

Restrict tool access:

```yaml
tools:
  catalog_ids:
    - github:*

  filters:
    allowed_tools:
      - github.search_repos
      - github.list_prs
    blocked_tools:
      - github.delete_repo
```

## Policies

### Approval Workflows

```yaml
policies:
  requires_human_approval: true

  approval:
    required_for:
      - step: code_execution
      - artifact_type: code_patch
      - tool: filesystem.write

    timeout: 3600          # Approval timeout (seconds)
    on_timeout: fail       # fail | skip | auto_approve

    approvers:
      - email: admin@company.com
      - role: engineering_lead
```

### Error Handling

```yaml
policies:
  on_error: fail          # fail | skip | retry | continue
  max_retries: 3
  retry_delay: 5          # seconds
  retry_backoff: 2.0      # Exponential backoff multiplier

  error_handlers:
    - error_type: ValidationError
      action: retry
      max_attempts: 5

    - error_type: TimeoutError
      action: skip
```

### Resource Limits

```yaml
policies:
  limits:
    max_execution_time: 3600    # seconds
    max_memory_mb: 1024
    max_tokens: 100000
    max_tool_calls: 50
    max_cost_usd: 5.0
```

### Safety Policies

```yaml
policies:
  safety:
    pii_filtering: true
    content_moderation: true

    blocked_domains:
      - malicious-site.com

    allowed_file_extensions:
      - .txt
      - .json
      - .yaml
```

## Examples

### Example 1: Simple Sequential Pipeline

```yaml
manifest_id: simple-pipeline
name: Simple Sequential Pipeline
version: "1.0.0"

steps:
  - id: fetch
    role: research
    capabilities: [web_search]
    inputs:
      - name: query
        source: user_input
    outputs: [research_snippet]
    timeout: 30

  - id: summarize
    role: synthesis
    capabilities: [text_summarize]
    inputs:
      - name: data
        source: previous_step
    outputs: [final_report]
    timeout: 20
```

### Example 2: Conditional Execution

```yaml
manifest_id: conditional-workflow
name: Conditional Workflow
version: "1.0.0"

steps:
  - id: analyze
    role: analysis
    capabilities: [data_analysis]
    inputs:
      - name: data
        source: user_input
    outputs: [analysis_result]
    timeout: 30

  - id: deep_dive
    role: research
    capabilities: [web_search, deep_research]
    condition: "analyze.confidence < 0.7"  # Only run if confidence low
    inputs:
      - name: topic
        source: step
        step: analyze
        artifact: analysis_result
    outputs: [research_snippet]
    timeout: 60

  - id: report
    role: synthesis
    capabilities: [text_generation]
    inputs:
      - name: analysis
        source: step
        step: analyze
      - name: research
        source: step
        step: deep_dive
        required: false  # May not exist if condition not met
    outputs: [final_report]
    timeout: 30
```

### Example 3: Parallel Execution

```yaml
manifest_id: parallel-workflow
name: Parallel Execution Workflow
version: "1.0.0"

steps:
  # Fan-out: Multiple agents work in parallel
  - id: research_a
    role: research
    capabilities: [web_search]
    parallel_group: gather
    inputs:
      - name: query
        source: user_input
    outputs: [research_snippet]
    timeout: 30

  - id: research_b
    role: research
    capabilities: [document_search]
    parallel_group: gather
    inputs:
      - name: query
        source: user_input
    outputs: [research_snippet]
    timeout: 30

  - id: research_c
    role: research
    capabilities: [academic_search]
    parallel_group: gather
    inputs:
      - name: query
        source: user_input
    outputs: [research_snippet]
    timeout: 30

  # Fan-in: Aggregate results
  - id: aggregate
    role: synthesis
    capabilities: [aggregation]
    inputs:
      - name: results
        source: parallel_group
        group: gather
    outputs: [final_report]
    timeout: 30
```

### Example 4: Human-in-the-Loop

```yaml
manifest_id: hitl-workflow
name: Human-in-the-Loop Workflow
version: "1.0.0"

steps:
  - id: generate_code
    role: code
    capabilities: [code_generation]
    inputs:
      - name: requirements
        source: user_input
    outputs: [code_patch]
    timeout: 60

  - id: review
    role: verification
    capabilities: [code_review]
    inputs:
      - name: code
        source: previous_step
    outputs: [review_result]
    timeout: 45

  - id: apply_changes
    role: code
    capabilities: [filesystem.write]
    inputs:
      - name: patch
        source: step
        step: generate_code
    outputs: [execution_result]
    timeout: 30
    requires_approval: true  # Wait for human approval

policies:
  approval:
    required_for:
      - step: apply_changes
    timeout: 3600
```

## Best Practices

### 1. Naming Conventions

```yaml
# Good
manifest_id: customer-support-workflow
step_id: verify_customer_data

# Bad
manifest_id: workflow1
step_id: step2
```

### 2. Timeouts

Set realistic timeouts based on step complexity:

```yaml
# Research/web operations: 30-60s
- id: research
  timeout: 60

# Code generation: 60-120s
- id: code_gen
  timeout: 120

# Simple transformations: 10-30s
- id: format
  timeout: 15
```

### 3. Error Handling

Always define error handling strategy:

```yaml
policies:
  on_error: fail          # Fail fast for critical workflows
  max_retries: 3          # Retry transient failures

  error_handlers:
    - error_type: RateLimitError
      action: retry
      retry_delay: 30
```

### 4. Artifact Validation

Use schema validation:

```yaml
steps:
  - id: process
    outputs: [research_snippet]  # Must match schema
    validation:
      strict: true
      schema_version: "1.0"
```

### 5. Documentation

```yaml
manifest_id: my-workflow
name: My Workflow
version: "1.0.0"
description: |
  This workflow performs X, Y, and Z.

  Use cases:
  - Scenario A
  - Scenario B

  Requirements:
  - API key for service X
  - Access to database Y

metadata:
  author: John Doe
  created: 2024-01-01
  updated: 2024-01-15
  tags: [production, customer-facing]
```

### 6. Version Control

```yaml
# Semantic versioning
version: "1.2.3"  # MAJOR.MINOR.PATCH

# Version history in metadata
metadata:
  changelog:
    - version: "1.2.3"
      date: 2024-01-15
      changes: "Added retry logic to research step"
    - version: "1.2.0"
      date: 2024-01-10
      changes: "Added verification step"
```

## Manifest Validation

### CLI Validation

```bash
# Validate manifest syntax
kautilya manifest validate workflow.yaml

# Test execution (dry run)
kautilya manifest test workflow.yaml --dry-run

# Lint for best practices
kautilya manifest lint workflow.yaml
```

### Programmatic Validation

```python
from orchestrator.service.models import WorkflowManifest
import yaml

# Load and validate
with open("workflow.yaml") as f:
    data = yaml.safe_load(f)

manifest = WorkflowManifest(**data)  # Validates with Pydantic

# Check for issues
issues = manifest.validate()
if issues:
    print("Validation errors:", issues)
```

## Schema Reference

Full JSON Schema available at: `docs/schemas/manifest_schema.json`

## See Also

- [API Reference](api-reference.md) - Programmatic workflow execution
- [Skills Guide](skills.md) - Creating custom capabilities
- [MCP Integration](mcp.md) - External tool integration
- [Examples](../examples/) - Working workflow examples
