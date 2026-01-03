# Multi-Step Workflow Example

This example demonstrates how to orchestrate multiple agents in a sequential pipeline using YAML manifests.

## Overview

Creates a 3-step workflow:
1. **Research Agent** - Gathers information on the topic
2. **Verification Agent** - Verifies facts and checks sources
3. **Synthesis Agent** - Combines findings into a final report

Each step produces a typed artifact that flows to the next step.

## Prerequisites

```bash
# Install PyYAML for manifest parsing
pip install pyyaml

# No other dependencies required for this demo
```

## Running the Example

```bash
# From the repository root
cd examples/02-multi-step-workflow
python run.py
```

## What This Example Shows

1. **YAML Workflow Manifests**: Declarative workflow definition
2. **Sequential Pipeline**: Agents executing in order
3. **Typed Artifacts**: Structured data passing between steps
4. **Configuration**: Timeouts, capabilities, memory settings

## Workflow Definition

The `workflow.yaml` file defines:

### Step 1: Research
```yaml
- id: research
  role: research
  capabilities: [web_search, summarize]
  inputs:
    - name: query
      source: user_input
  outputs: [research_snippet]
  timeout: 60
```

### Step 2: Verify
```yaml
- id: verify
  role: verification
  capabilities: [fact_check]
  inputs:
    - name: research_data
      source: previous_step
  outputs: [claim_verification]
  timeout: 45
```

### Step 3: Synthesize
```yaml
- id: synthesize
  role: synthesis
  capabilities: [text_generation, summarize]
  inputs:
    - name: research
      source: step
      step: research
    - name: verification
      source: previous_step
  outputs: [final_report]
  timeout: 30
```

## Expected Output

```
============================================================
Multi-Step Workflow Example
============================================================

📝 User Query: Multi-agent AI systems in enterprise applications

🔄 Workflow Execution:

============================================================
Executing Workflow: Multi-Step Research Workflow
============================================================

[Step 1/3] RESEARCH: Research the topic and gather information
  Role: research
  Capabilities: web_search, summarize
  Timeout: 60s
  ✓ Completed - Generated artifact: research_snippet
    Preview: Research findings on 'Multi-agent AI systems in enterpr...

[Step 2/3] VERIFY: Verify facts and check sources
  Role: verification
  Capabilities: fact_check
  Timeout: 45s
  ✓ Completed - Generated artifact: claim_verification
    Preview: Confidence: 92.0%

[Step 3/3] SYNTHESIZE: Synthesize findings into final report
  Role: synthesis
  Capabilities: text_generation, summarize
  Timeout: 30s
  ✓ Completed - Generated artifact: final_report
    Preview: This report synthesizes research findings and verificati...

============================================================
Workflow Complete!
============================================================

📊 Final Report: Research Report: Multi-agent AI systems in enterprise applications

This report synthesizes research findings and verification results...

Quality Score: 88.0%
```

## Files

- `workflow.yaml` - Workflow manifest definition
- `run.py` - Workflow simulator
- `README.md` - This file

## Artifact Flow

```
User Input (query)
    ↓
Research Agent → research_snippet
    ↓
Verification Agent → claim_verification
    ↓
Synthesis Agent → final_report
    ↓
Final Output
```

## Key Concepts

### 1. Typed Artifacts
Each step outputs a specific artifact type defined in the schema registry:
- `research_snippet`: Research findings with sources
- `claim_verification`: Fact-checking results
- `final_report`: Synthesized output

### 2. Input Sources
Steps can receive inputs from:
- `user_input`: Direct from user
- `previous_step`: From immediately preceding step
- `step`: From any specific step by ID

### 3. Memory Configuration
```yaml
memory:
  persist_on: [step_complete, workflow_complete]
  compaction:
    strategy: summarize
    max_tokens: 8000
```

### 4. Policy Configuration
```yaml
policies:
  requires_human_approval: false
  max_retries: 3
  on_error: fail
```

## Next Steps

- See `03-custom-skill/` for creating custom skills
- See `04-mcp-integration/` for external tool integration
- Read docs/manifests.md for full manifest schema
- Try the orchestrator API: `docs/api-reference.md`
