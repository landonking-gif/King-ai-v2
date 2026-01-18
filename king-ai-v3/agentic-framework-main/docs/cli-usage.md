# Kautilya CLI Usage Guide

Complete guide to using `kautilya` - the interactive CLI for building and managing agentic workflows.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Getting Started](#getting-started)
- [Basic Usage](#basic-usage)
- [Intermediate Usage](#intermediate-usage)
- [Advanced Usage](#advanced-usage)
- [Deep Research Platform](#deep-research-platform)
- [CLI Reference](#cli-reference)
- [Tips and Tricks](#tips-and-tricks)
- [Troubleshooting](#troubleshooting)

## Overview

`kautilya` is an interactive CLI that lets you:

- **Build agents** without writing code
- **Configure LLMs** from 6 different providers
- **Create skills** with guided scaffolding
- **Design workflows** using YAML manifests
- **Integrate tools** via MCP protocol
- **Run research** using multi-agent orchestration
- **Monitor execution** in real-time

### Why Use Kautilya?

**Interactive Development:**
```
Traditional Approach          Kautilya Approach
───────────────────          ─────────────────
Write Python code       →    Describe in CLI
Create YAML files       →    Guided prompts
Manual configuration    →    Interactive setup
Trial and error        →    Validation on save
```

**Rapid Prototyping:**
Build a complete multi-agent research workflow in minutes, not hours.

## Installation

### Prerequisites

```bash
# Python 3.11+
python --version

# Install uv (fast package manager - recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Install Kautilya

Kautilya is automatically installed with the Agentic Framework.

**Option 1: Install Full Framework (Recommended)**
```bash
git clone https://github.com/paragajg/agentic-framework.git
cd agentic-framework

# Create virtual environment and install
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e .

# Kautilya is now available!
kautilya --version
```

**Option 2: Install Kautilya Only**
```bash
git clone https://github.com/paragajg/agentic-framework.git
cd agentic-framework
uv pip install -e tools/kautilya/
```

**Verify Installation:**
```bash
kautilya --version
# Output: Kautilya v1.0.0
```

### Set Up LLM Provider

Create a `.env` file in your project root:
```bash
# Copy example and edit
cp .env.example .env

# Add your API key (choose one):
# For OpenAI:
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-4o-mini

# For Anthropic:
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

Test the connection:
```bash
kautilya llm test
# Output: ✓ Connection: Successfully connected!
```

## Getting Started

### Launching Interactive Mode

```bash
# Start interactive session
kautilya

# Or run single command
kautilya agent new my-agent --role research
```

### First Launch Experience

```
$ kautilya
┌─────────────────────────────────────────────────┐
│  Kautilya v1.0 - Agentic Framework CLI          │
│  Type /help for commands, or describe your task │
└─────────────────────────────────────────────────┘

Welcome to Kautilya! Let's get you started.

? Do you have an existing project? (y/N): N

Great! Let's create a new agent project.

? Project name: research-assistant
? Project directory: ./research-assistant
? LLM provider:
  ❯ anthropic      # Claude (recommended)
    openai         # GPT models
    azure          # Azure OpenAI
    gemini         # Google Gemini
    local          # Ollama (offline)
    vllm           # vLLM (optimized local)

? Enable MCP integration? (Y/n): Y
? Install common MCP servers? (Y/n): Y

⠋ Creating project structure...
⠋ Installing dependencies...
⠋ Configuring LLM adapter...
⠋ Setting up MCP servers...

✓ Project created: ./research-assistant
✓ LLM configured: anthropic (claude-sonnet-4)
✓ MCP servers installed: filesystem, web_search

Next steps:
  1. cd research-assistant
  2. kautilya
  3. Type /agent new to create your first agent

> _
```

### Project Structure

After initialization:

```
research-assistant/
├── .kautilya/
│   ├── config.yaml          # Project settings
│   └── llm.yaml             # LLM configurations
├── agents/                  # Agent definitions
├── skills/                  # Custom skills
├── manifests/               # Workflow definitions
├── schemas/                 # Custom artifact schemas
├── tests/                   # Test files
└── README.md
```

## Basic Usage

### Session 1: Creating Your First Agent

```
> /agent new research-agent

╭────────────────────────────────────────╮
│  Creating New Agent: research-agent    │
╰────────────────────────────────────────╯

? Agent role:
  ❯ research        # Information gathering
    verification    # Fact checking
    code           # Code generation/analysis
    analysis       # Data analysis
    synthesis      # Content creation
    custom         # Define your own

? Agent role: research

? Capabilities (comma-separated): web_search, document_read, summarize

? Output artifact type:
  ❯ research_snippet
    claim_verification
    code_patch
    analysis_result
    final_report
    custom

? Output artifact type: research_snippet

? System prompt file: (prompts/research.txt)
  Leave empty to use default

⠋ Generating agent structure...

✓ Created agent: agents/research-agent/
  ├── config.yaml              # Agent configuration
  ├── capabilities.json        # Skills and tools list
  └── prompts/
      └── system.txt           # System prompt

Agent Details:
  Role:         research
  Capabilities: web_search, document_read, summarize
  Output Type:  research_snippet
  Timeout:      60s (default)

> View configuration: cat agents/research-agent/config.yaml
> Edit capabilities: edit agents/research-agent/capabilities.json
> Test agent: /run agents/research-agent
```

**Generated Files:**

`agents/research-agent/config.yaml`:
```yaml
agent_id: research-agent
role: research
version: "1.0.0"

capabilities:
  skills:
    - web_search
    - document_read
    - summarize
  mcp_tools: []

output_artifact_type: research_snippet

settings:
  timeout: 60
  max_retries: 3
  temperature: 0.7

prompts:
  system: prompts/system.txt
```

### Session 2: Configuring LLM Provider

```
> /llm config

╭────────────────────────────────────────╮
│       LLM Provider Configuration       │
╰────────────────────────────────────────╯

Current Providers:
  ✓ anthropic (default)
  ○ openai
  ○ azure
  ○ gemini
  ○ local
  ○ vllm

? Select provider to configure: anthropic

? Select model:
  ❯ claude-sonnet-4-20250514    (Recommended)
    claude-opus-4-5-20251101     (Most capable)
    claude-haiku-4-20250514      (Fastest)

? API key source:
  ❯ environment variable
    vault (HashiCorp Vault)
    file

? Environment variable name: ANTHROPIC_API_KEY

? Set as default provider? (Y/n): Y

⠋ Testing connection...

✓ Connection successful
✓ Model: claude-sonnet-4-20250514
✓ Account: user@example.com
✓ Rate limit: 50 req/min

Configuration saved to .kautilya/llm.yaml

> Test LLM: /llm test
> List models: /llm list
```

### Session 3: Creating a Simple Skill

```
> /skill new extract-entities

╭────────────────────────────────────────╮
│    Creating New Skill: extract-entities│
╰────────────────────────────────────────╯

? Skill format:
  ❯ Hybrid (both formats, recommended)
    Native (skill.yaml + schema.json)
    Anthropic (SKILL.md only)

? Skill format: Hybrid

? Description: Extract named entities (people, organizations, locations) from text

? Input schema fields (name:type, comma-separated):
  text:string, entity_types:array

? Output schema fields:
  entities:array, confidence:number

? Safety flags:
  ☐ file_system
  ☐ network_access
  ☑ pii_risk              # Entities might be sensitive
  ☐ side_effect

? Requires approval? (y/N): N

⠋ Generating skill files...
⠋ Creating handler stub...
⠋ Generating tests...

✓ Created skill: skills/extract_entities/
  ├── SKILL.md             # Anthropic marketplace format
  ├── skill.yaml           # Native metadata
  ├── schema.json          # I/O validation
  ├── handler.py           # Implementation stub
  └── test_handler.py      # Test template

Next steps:
  1. Implement: edit skills/extract_entities/handler.py
  2. Test: pytest skills/extract_entities/test_handler.py
  3. Use in agent: Add "extract_entities" to capabilities

> Open in editor: code skills/extract_entities/
```

**Generated handler.py (stub):**

```python
"""
Extract Entities Skill

Extract named entities (people, organizations, locations) from text
"""

from typing import Dict, Any, List


def extract_entities(
    text: str,
    entity_types: List[str] = None
) -> Dict[str, Any]:
    """
    Extract named entities from text.

    Args:
        text: Input text to analyze
        entity_types: Optional filter for specific entity types

    Returns:
        Dictionary with entities and confidence score
    """
    # TODO: Implement entity extraction
    # Consider using spaCy, NLTK, or transformer models

    return {
        "entities": [
            {
                "text": "Example Entity",
                "type": "PERSON",
                "start": 0,
                "end": 14
            }
        ],
        "confidence": 0.85
    }
```

### Session 4: Running a Workflow

```
> /manifest run manifests/simple-research.yaml

╭────────────────────────────────────────╮
│  Running Workflow: simple-research     │
╰────────────────────────────────────────╯

? Input query: "What are the latest developments in multi-agent AI systems?"

⠋ Loading manifest...
✓ Manifest validated
✓ 3 steps detected

┌─────────────────────────────────────────────────┐
│ Step 1/3: research                              │
│ Agent: research-agent                           │
│ Action: Gathering information...               │
└─────────────────────────────────────────────────┘

  🔍 Invoking tool: web_search
     Query: "multi-agent AI systems 2024"
  ✓ Found 15 results

  📝 Invoking skill: summarize
  ✓ Summary generated (450 words)

  ✓ Artifact created: art_research_001
     Type: research_snippet
     Confidence: 0.89

┌─────────────────────────────────────────────────┐
│ Step 2/3: verify                                │
│ Agent: verify-agent                             │
│ Action: Fact-checking claims...                │
└─────────────────────────────────────────────────┘

  🔍 Invoking tool: fact_check
  ✓ 12 claims verified

  ✓ Artifact created: art_verify_001
     Type: claim_verification
     Verdict: 11/12 verified (91.7%)

┌─────────────────────────────────────────────────┐
│ Step 3/3: synthesize                            │
│ Agent: synthesis-agent                          │
│ Action: Creating final report...               │
└─────────────────────────────────────────────────┘

  ✓ Artifact created: art_final_001
     Type: final_report

╭────────────────────────────────────────────────╮
│              Workflow Complete                 │
├────────────────────────────────────────────────┤
│ Status:      Success                           │
│ Duration:    45.2s                             │
│ Steps:       3/3 completed                     │
│ Artifacts:   3 created                         │
│ LLM Tokens:  12,450                            │
│ Cost:        $0.19                             │
╰────────────────────────────────────────────────╯

Final Report:
───────────────────────────────────────────────

# Latest Developments in Multi-Agent AI Systems

Recent advancements in multi-agent systems show significant
progress in collaborative problem-solving and autonomous
coordination...

[Full report truncated for brevity]

> View artifacts: /artifacts list
> Save report: /export final_report.md
> Run again: /manifest run manifests/simple-research.yaml
```

## Intermediate Usage

### Session 5: Building a Multi-Step Research Workflow

```
> /manifest new

╭────────────────────────────────────────────────╮
│           Create New Workflow Manifest         │
╰────────────────────────────────────────────────╯

? Manifest name: comprehensive-research

? Description: Multi-step research with verification and synthesis

? Add steps (press Enter when done):

  Step 1:
  ? Step ID: initial_research
  ? Agent role: research
  ? Capabilities: web_search, document_read, summarize
  ? Input source: user_input
  ? Input name: research_query
  ? Output type: research_snippet
  ? Timeout (seconds): 60

  Step 2:
  ? Step ID: deep_dive
  ? Agent role: research
  ? Capabilities: web_search, academic_search, deep_research
  ? Input source: previous_step
  ? Output type: research_snippet
  ? Timeout (seconds): 120
  ? Conditional execution? (y/N): Y
  ? Condition: initial_research.confidence < 0.7

  Step 3:
  ? Step ID: verify_facts
  ? Agent role: verification
  ? Capabilities: fact_check, source_validation
  ? Input source: previous_step
  ? Output type: claim_verification
  ? Timeout (seconds): 60

  Step 4:
  ? Step ID: synthesize
  ? Agent role: synthesis
  ? Capabilities: text_generation, citation_formatting
  ? Multiple input sources? (y/N): Y
  ? Input 1 source: step
  ? Input 1 step: initial_research
  ? Input 2 source: step
  ? Input 2 step: deep_dive
  ? Input 2 required: false
  ? Input 3 source: previous_step
  ? Output type: final_report
  ? Timeout (seconds): 90

  Step 5:
  ? Step ID: (leave empty to finish)

? Memory persistence:
  ❯ on_complete       # Save only at end
    per_step          # Save after each step
    manual            # Explicit saves only

? Memory compaction strategy:
  ❯ summarize         # Summarize old artifacts
    truncate          # Keep only recent
    none              # No compaction

? Max context tokens: 8000

? Requires human approval? (y/N): N

? Add MCP tools:
  Available servers:
  ☑ web_search
  ☑ github
  ☐ slack
  ☐ filesystem

⠋ Generating manifest...
⠋ Validating workflow...

✓ Created: manifests/comprehensive-research.yaml
✓ Validation passed

Workflow Summary:
  Steps:      4 (1 conditional)
  Max time:   330s (5.5 min)
  MCP tools:  web_search, github
  Memory:     Compaction enabled (8K tokens)

> View manifest: cat manifests/comprehensive-research.yaml
> Test workflow: /manifest test manifests/comprehensive-research.yaml --dry-run
> Run workflow: /manifest run manifests/comprehensive-research.yaml
```

**Generated Manifest:**

```yaml
manifest_id: comprehensive-research
name: Comprehensive Research Workflow
version: "1.0.0"
description: Multi-step research with verification and synthesis

steps:
  - id: initial_research
    role: research
    capabilities: [web_search, document_read, summarize]
    inputs:
      - name: research_query
        source: user_input
        required: true
    outputs: [research_snippet]
    timeout: 60

  - id: deep_dive
    role: research
    capabilities: [web_search, academic_search, deep_research]
    condition: "initial_research.confidence < 0.7"
    inputs:
      - name: initial_findings
        source: previous_step
    outputs: [research_snippet]
    timeout: 120

  - id: verify_facts
    role: verification
    capabilities: [fact_check, source_validation]
    inputs:
      - name: research_data
        source: previous_step
    outputs: [claim_verification]
    timeout: 60

  - id: synthesize
    role: synthesis
    capabilities: [text_generation, citation_formatting]
    inputs:
      - name: initial_research
        source: step
        step: initial_research
      - name: deep_research
        source: step
        step: deep_dive
        required: false
      - name: verification
        source: previous_step
    outputs: [final_report]
    timeout: 90

memory:
  persist_on: [workflow_complete]
  compaction:
    strategy: summarize
    max_tokens: 8000
    interval: 300

tools:
  catalog_ids:
    - web_search.*
    - github.search_repos

policies:
  requires_human_approval: false
  max_retries: 3
  on_error: fail
```

### Session 6: MCP Server Integration

```
> /mcp add firecrawl

╭────────────────────────────────────────────────╮
│        Add MCP Server: firecrawl               │
╰────────────────────────────────────────────────╯

? Install from:
  ❯ Registry (recommended)
    NPM package
    Custom URL

? Install from: Registry

⠋ Searching registry...

Found: @mendableai/firecrawl-mcp-server
  Description: Advanced web scraping and crawling
  Version: 1.2.0
  Tools: scrape_url, crawl_site, extract_content, map_site
  Rating: ⭐⭐⭐⭐⭐ (4.8/5, 1.2k installs)

? Install this server? (Y/n): Y

⠋ Installing via npm...
✓ Installed: firecrawl-mcp-server v1.2.0

? Configuration:

? API Key source:
  ❯ environment variable
    prompt now
    vault

? Environment variable: FIRECRAWL_API_KEY

? Scopes needed:
  ☑ web:read
  ☑ web:extract
  ☐ web:write

? Rate limit (calls/min): 20

? Add to which manifests:
  ☑ comprehensive-research.yaml
  ☐ simple-research.yaml
  ☐ All manifests

⠋ Configuring server...
⠋ Updating manifests...

✓ Server configured: firecrawl_mcp
✓ Updated 1 manifest

Configuration saved to mcp_config.yaml

Test the server:
  > /mcp test firecrawl

Invoke a tool:
  > /mcp invoke firecrawl scrape_url '{"url": "https://example.com"}'
```

### Session 7: Importing Skills from Marketplace

```
> /skill import https://marketplace.anthropic.com/skills/pdf-form-filler.zip

╭────────────────────────────────────────────────╮
│         Import Skill from Marketplace          │
╰────────────────────────────────────────────────╯

⠋ Downloading from marketplace...
✓ Downloaded: pdf-form-filler.zip (2.3 MB)

⠋ Analyzing skill package...

Skill Information:
  Name:        pdf-form-filler
  Version:     1.2.0
  Format:      Anthropic (SKILL.md)
  Author:      Anthropic
  Downloads:   45.2k
  Rating:      ⭐⭐⭐⭐⭐ (4.9/5)

Description:
  Fill PDF forms with structured data. Supports AcroForms
  and XFA forms with intelligent field mapping.

Safety Review:
  ⚠️  Requires file system access (read + write)
  ⚠️  May process sensitive form data
  ✓  No network requests
  ✓  No code execution

? Import as:
  ❯ Hybrid (recommended)
    Anthropic only
    Convert to native

? Import as: Hybrid

? Assign safety flags:
  ☑ file_system
  ☑ pii_risk
  ☐ network_access

? Requires approval for use? (y/N): Y

⠋ Importing skill...
⠋ Converting to hybrid format...
⠋ Generating native schemas...

✓ Imported: skills/pdf_form_filler/
  ├── SKILL.md             # Original Anthropic format
  ├── skill.yaml           # Generated native metadata
  ├── schema.json          # Generated validation schemas
  └── handler.py           # Wrapper for marketplace skill

Skill ready to use!

Add to agent capabilities:
  > edit agents/my-agent/capabilities.json
  > Add "pdf_form_filler" to skills list

Test the skill:
  > /skill test pdf_form_filler
```

## Advanced Usage

### Session 8: Building a Parallel Research Workflow

```
> /manifest new parallel-research

╭────────────────────────────────────────────────╮
│      Parallel Research Workflow Builder        │
╰────────────────────────────────────────────────╯

? Use advanced features? (Y/n): Y

? Workflow type:
    Sequential (one step at a time)
  ❯ Parallel (fan-out/fan-in)
    Hierarchical (supervisor pattern)
    Dynamic (steps determined at runtime)

? Workflow type: Parallel

Let's define the fan-out phase:

? Number of parallel agents: 3

  Parallel Agent 1:
  ? ID: web_research
  ? Role: research
  ? Capabilities: web_search, summarize
  ? Parallel group name: gather

  Parallel Agent 2:
  ? ID: academic_research
  ? Role: research
  ? Capabilities: academic_search, pdf_extraction
  ? Parallel group: gather

  Parallel Agent 3:
  ? ID: github_research
  ? Role: research
  ? Capabilities: github.search_repos, code_analysis
  ? Parallel group: gather

All parallel agents will:
  ✓ Start simultaneously
  ✓ Share the same input (research_query)
  ✓ Run with independent timeouts
  ✓ Continue even if one fails

Now let's define the fan-in phase:

? Aggregation step ID: aggregate_results
? Aggregation role: synthesis
? Aggregation capabilities: multi_source_synthesis, deduplication
? How to handle partial results?
  ❯ Continue with available results
    Fail if any parallel step fails
    Require minimum N successes

? Minimum successes required: 2

⠋ Generating parallel workflow manifest...

✓ Created: manifests/parallel-research.yaml

Workflow Structure:
  ┌─────────────────┐
  │  User Input     │
  └────────┬────────┘
           │
    ┌──────┴──────────┬──────────┐
    ▼                 ▼          ▼
  ┌─────────┐   ┌─────────┐   ┌─────────┐
  │   Web   │   │Academic │   │ GitHub  │
  │Research │   │Research │   │Research │
  └────┬────┘   └────┬────┘   └────┬────┘
       │             │             │
       └─────────────┴─────────────┘
                     ▼
              ┌─────────────┐
              │  Aggregate  │
              │   Results   │
              └─────────────┘
                     ▼
              ┌─────────────┐
              │Final Report │
              └─────────────┘

Expected performance:
  Sequential time: ~180s
  Parallel time:   ~60s (3x faster!)
  Max concurrency: 3 agents
```

**Generated Parallel Manifest:**

```yaml
manifest_id: parallel-research
name: Parallel Research Workflow
version: "1.0.0"

steps:
  # Fan-out: Parallel execution
  - id: web_research
    role: research
    capabilities: [web_search, summarize]
    parallel_group: gather
    inputs:
      - name: query
        source: user_input
    outputs: [research_snippet]
    timeout: 60

  - id: academic_research
    role: research
    capabilities: [academic_search, pdf_extraction]
    parallel_group: gather
    inputs:
      - name: query
        source: user_input
    outputs: [research_snippet]
    timeout: 60

  - id: github_research
    role: research
    capabilities: [github.search_repos, code_analysis]
    parallel_group: gather
    inputs:
      - name: query
        source: user_input
    outputs: [research_snippet]
    timeout: 60

  # Fan-in: Aggregate results
  - id: aggregate_results
    role: synthesis
    capabilities: [multi_source_synthesis, deduplication]
    inputs:
      - name: results
        source: parallel_group
        group: gather
        min_results: 2  # Continue if at least 2 succeed
    outputs: [final_report]
    timeout: 90

policies:
  parallel:
    max_concurrent: 3
    on_partial_failure: continue
    min_success_count: 2
```

### Session 9: Interactive Deep Research Session

```
> /research --mode interactive

╭────────────────────────────────────────────────╮
│         Deep Research Platform Mode            │
╰────────────────────────────────────────────────╯

Welcome to interactive research mode!

This mode provides:
  • Multi-step research with refinement
  • Real-time source exploration
  • Iterative fact-checking
  • Progressive report building

Commands:
  /query <topic>         - Start new research
  /refine <aspect>       - Deep dive into aspect
  /verify <claim>        - Fact-check a claim
  /sources               - List all sources
  /export <format>       - Export final report
  /quit                  - Exit research mode

research> /query "How do multi-agent systems handle conflicting objectives?"

⠋ Initializing research session...
✓ Session ID: research_sess_001

┌─────────────────────────────────────────────────┐
│ Phase 1: Initial Research                       │
└─────────────────────────────────────────────────┘

⠋ Searching web sources... (0:03)
  Found: 47 relevant articles
  Top sources:
    • arxiv.org (12 papers)
    • medium.com (8 articles)
    • github.com (15 repos)

⠋ Extracting key information... (0:05)
  Extracted: 23 key concepts
  Identified: 5 major approaches

⠋ Generating initial summary... (0:02)

✓ Initial research complete (0:10 elapsed)

Summary (Confidence: 0.76 - Medium):
───────────────────────────────────────────

Multi-agent systems handle conflicting objectives through
several established approaches:

1. **Game-Theoretic Methods**
   - Nash equilibrium strategies
   - Pareto optimization
   - Sources: [1], [2], [4]

2. **Negotiation Protocols**
   - Contract Net Protocol
   - Auction-based coordination
   - Sources: [3], [5], [7]

3. **Hierarchical Arbitration**
   - Priority-based resolution
   - Supervisor agent patterns
   - Sources: [6], [8]

⚠️  Confidence is moderate. Consider deep dive.

research> What should we explore further?
  ❯ 1. Game-theoretic methods (seems most promising)
    2. Negotiation protocols (many recent papers)
    3. Hierarchical arbitration (practical examples)
    4. All of the above
    5. Something else (specify)

research> 1

┌─────────────────────────────────────────────────┐
│ Phase 2: Deep Dive - Game-Theoretic Methods     │
└─────────────────────────────────────────────────┘

⠋ Academic search... (0:08)
  Found: 156 academic papers
  Filtering: 2020-2024 only
  Top cited: 23 papers

⠋ Code repository analysis... (0:06)
  Found: 34 implementations
  Languages: Python (18), Julia (8), C++ (8)
  Stars total: 12.4k

⠋ Analyzing implementation patterns... (0:12)

✓ Deep dive complete (0:26 elapsed)

Detailed Findings (Confidence: 0.91 - High):
───────────────────────────────────────────

Game-theoretic approaches show three dominant patterns:

**1. Nash Equilibrium Strategies**
   Most widely used in practice. Key implementations:
   - Multi-agent-nash-py (2.3k ⭐)
   - equilibrium-solver (1.1k ⭐)

   Strengths:
   • Mathematically rigorous
   • Provable convergence properties
   • Well-studied optimization techniques

   Weaknesses:
   • Computationally expensive (O(n²) agents)
   • Assumes rational agents
   • May reach suboptimal equilibria

   Code Example: [Repo: multi-agent-nash-py]
   ```python
   def find_nash_equilibrium(agents, payoff_matrix):
       # Uses fictitious play algorithm
       ...
   ```

⚠️  Claims detected. Verify? (Y/n):

research> Y

┌─────────────────────────────────────────────────┐
│ Phase 3: Fact Verification                      │
└─────────────────────────────────────────────────┘

Verifying 12 factual claims...

⠋ Cross-referencing sources... (0:08)

Claim 1: "Nash equilibrium most widely used"
  ✓ Verified (8/10 sources agree)
  Sources: [1], [2], [4], [6], [7], [9], [11], [14]

Claim 2: "O(n²) computational complexity"
  ✓ Verified (6/7 technical sources)
  Exception: One paper claims O(n log n) with approximations

Claim 3: "2.3k GitHub stars for multi-agent-nash-py"
  ✓ Verified (live API check)
  Actual: 2,347 stars (as of 2024-01-15)

...

✓ Verification complete (12/12 claims checked)
  Verified: 11/12 (91.7%)
  Disputed: 1/12 (noted in report)

research> /sources

Sources Used (28 total):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Academic Papers (12):
  [1] "Multi-Agent Systems and Nash Equilibria"
      Authors: Johnson et al.
      Published: Nature Machine Intelligence, 2023
      Citations: 342
      Confidence: High ✓

  [2] "Game Theory in Distributed AI"
      Authors: Chen, Park, Lee
      Published: ICML 2023
      Citations: 156
      Confidence: High ✓

  [...additional sources...]

Code Repositories (8):
  [14] multi-agent-nash-py
       Language: Python
       Stars: 2,347
       Last updated: 2024-01-10
       License: MIT

  [...additional repos...]

Industry Articles (8):
  [22] "Practical Multi-Agent Coordination"
       Published: Medium
       Author: Sarah Mitchell
       Views: 12.4k
       Confidence: Medium

research> /refine "real-world applications in robotics"

┌─────────────────────────────────────────────────┐
│ Phase 4: Refinement - Robotics Applications     │
└─────────────────────────────────────────────────┘

⠋ Searching robotics-specific sources... (0:07)
⠋ Analyzing case studies... (0:11)

Found: 15 real-world implementations

Notable Applications:
  1. **Warehouse Robotics** (Amazon, Ocado)
     - 100+ robots coordinating
     - Nash equilibrium for path planning
     - 40% efficiency improvement

  2. **Autonomous Drone Swarms** (US Military, DJI)
     - Formation control using game theory
     - Conflict-free trajectory planning
     - Up to 50 drones simultaneously

  3. **Self-Driving Cars** (Waymo, Tesla FSD)
     - Multi-vehicle intersection coordination
     - Non-cooperative game theory
     - Safety-critical decisions

research> /export markdown

⠋ Generating final report... (0:05)

✓ Report generated: research_report_001.md

Final Report Statistics:
  Pages:        12
  Words:        4,850
  Sources:      28 (verified)
  Confidence:   0.91 (High)
  Time:         42 minutes
  Cost:         $0.87

Report includes:
  ✓ Executive summary
  ✓ Detailed findings
  ✓ Code examples
  ✓ Implementation patterns
  ✓ Real-world case studies
  ✓ Full bibliography
  ✓ Verification notes

research> /quit

Exiting research mode.

Session saved: research_sess_001
Resume later: /research --resume research_sess_001

> _
```

### Session 10: Custom Agent Orchestration

```
> /advanced orchestration

╭────────────────────────────────────────────────╮
│       Advanced Custom Orchestration            │
╰────────────────────────────────────────────────╯

This mode lets you build complex agent workflows using:
  • Dynamic agent spawning
  • Conditional routing
  • Custom approval workflows
  • Real-time monitoring

? Orchestration pattern:
    Supervisor (one agent manages others)
  ❯ Specialist Team (specialized agents for subtasks)
    Debate (agents discuss to reach consensus)
    Hierarchical (multi-level coordination)
    Custom (define your own)

? Pattern: Specialist Team

Let's build a specialist team:

? Team purpose: Comprehensive code review and improvement

? Specialists needed:

  Specialist 1:
  ? ID: security_expert
  ? Role: security_analysis
  ? Capabilities: code_security_scan, vulnerability_detection
  ? Triggers on: code_patch artifacts

  Specialist 2:
  ? ID: performance_expert
  ? Role: performance_analysis
  ? Capabilities: profiling, optimization_suggestions
  ? Triggers on: code_patch artifacts

  Specialist 3:
  ? ID: test_expert
  ? Role: testing
  ? Capabilities: test_generation, coverage_analysis
  ? Triggers on: code_patch artifacts

  Specialist 4:
  ? ID: documentation_expert
  ? Role: documentation
  ? Capabilities: docstring_generation, api_doc_creation
  ? Triggers on: code_patch artifacts

  Specialist 5:
  ? ID: (leave empty to finish)

? Coordinator agent:
  ? ID: coordinator
  ? Role: synthesis
  ? Capabilities: consensus_building, priority_ranking
  ? Gathers outputs from: all specialists

? Approval required before synthesis? (y/N): Y

? Approvers:
  engineering_lead, security_team

⠋ Generating orchestration manifest...

✓ Created: manifests/specialist-code-review.yaml

Team Structure:
                  ┌──────────────┐
                  │  Code Input  │
                  └──────┬───────┘
                         │
          ┌──────────────┼──────────────┐
          │              │              │
          ▼              ▼              ▼
    ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
    │ Security │   │Performance│   │  Test    │   │   Docs   │
    │  Expert  │   │  Expert   │   │  Expert  │   │  Expert  │
    └────┬─────┘   └────┬──────┘   └────┬─────┘   └────┬─────┘
         │              │              │              │
         └──────────────┴──────────────┴──────────────┘
                        │
                ┌───────▼────────┐
                │ Approval Gate  │
                └───────┬────────┘
                        │
                 ┌──────▼───────┐
                 │ Coordinator  │
                 │  (Synthesis) │
                 └──────────────┘

Test the orchestration:
  > /manifest run manifests/specialist-code-review.yaml
```

## Deep Research Platform

The Kautilya CLI can function as a powerful deep research platform, orchestrating multiple agents to conduct comprehensive research.

### Research Mode Features

1. **Multi-Source Aggregation**
   - Web search (Google, Bing, DuckDuckGo)
   - Academic databases (arXiv, PubMed, IEEE)
   - Code repositories (GitHub, GitLab)
   - Documentation sites
   - News aggregators

2. **Iterative Refinement**
   - Initial broad search
   - Confidence-based deep dives
   - Progressive narrowing
   - Quality assessment

3. **Fact Verification**
   - Cross-source validation
   - Citation checking
   - Claim verification
   - Confidence scoring

4. **Intelligent Synthesis**
   - Multi-source synthesis
   - Deduplication
   - Citation formatting
   - Progressive report building

### Research Workflow Example

**Use Case: Technology Landscape Analysis**

```bash
> /research --mode interactive

research> /query "State of large language model fine-tuning in 2024"

# Phase 1: Initial Research (1-2 minutes)
  - Searches 5-10 sources
  - Extracts key themes
  - Identifies knowledge gaps
  - Returns confidence score

# Phase 2: Deep Dive (5-10 minutes per topic)
  - Academic paper analysis
  - Code repository exploration
  - Industry report synthesis
  - Expert blog aggregation

# Phase 3: Verification (2-3 minutes)
  - Cross-source fact checking
  - Citation validation
  - Claim verification
  - Confidence updates

# Phase 4: Synthesis (3-5 minutes)
  - Multi-source integration
  - Deduplication
  - Progressive report building
  - Final quality check

# Total: 15-30 minutes for comprehensive report
```

### Research Templates

Create reusable research templates:

```bash
> /template create literature-review

Template: literature-review
───────────────────────────────────
Purpose: Academic literature review with citation analysis

Phases:
  1. Initial search (academic databases)
  2. Citation network analysis
  3. Key paper identification
  4. Detailed paper analysis
  5. Synthesis with citations

Outputs:
  - Annotated bibliography
  - Citation network graph
  - Key findings summary
  - Research gaps identified

Save template? (Y/n): Y

✓ Template saved: templates/literature-review.yaml

Use template:
  > /research --template literature-review --query "Your topic"
```

## CLI Reference

### Complete Command List

#### Project Management

```bash
/init                          # Initialize new project
/status                        # Show project status
/config                        # Edit project configuration
```

#### Agent Commands

```bash
/agent new <name>              # Create new agent
/agent list                    # List all agents
/agent edit <name>             # Edit agent configuration
/agent delete <name>           # Remove agent
/agent test <name>             # Test agent execution
```

#### Skill Commands

```bash
/skill new <name>              # Create new skill
/skill import <url|path>       # Import skill
/skill export <name>           # Export skill to ZIP
/skill convert <name>          # Convert skill format
/skill list                    # List all skills
/skill validate <name>         # Validate skill
/skill test <name>             # Test skill execution
/skill package <name>          # Package for distribution
```

#### LLM Commands

```bash
/llm config                    # Configure LLM provider
/llm list                      # List available providers
/llm test                      # Test LLM connection
/llm models                    # List available models
/llm switch <provider>         # Switch default provider
```

#### MCP Commands

```bash
/mcp add <server>              # Add MCP server
/mcp list                      # List registered servers
/mcp test <server>             # Test server connection
/mcp remove <server>           # Remove server
/mcp invoke <server> <tool>    # Invoke tool manually
/mcp refresh-catalog           # Refresh tool catalog
```

#### Manifest Commands

```bash
/manifest new                  # Create new workflow
/manifest validate <file>      # Validate manifest
/manifest test <file>          # Dry-run test
/manifest run <file>           # Execute workflow
/manifest list                 # List all manifests
```

#### Research Commands

```bash
/research                      # Start research mode
/research --mode interactive   # Interactive research
/research --template <name>    # Use research template
/research --resume <id>        # Resume session
```

#### Utility Commands

```bash
/run [file]                    # Run agent/workflow
/logs [agent]                  # View logs
/artifacts list                # List artifacts
/export <format>               # Export results
/help [command]                # Show help
/quit                          # Exit CLI
```

### Command Options

Most commands support additional options:

```bash
# With options
/agent new my-agent --role research --capabilities web_search,summarize

# Dry run (no actual execution)
/manifest run my-workflow.yaml --dry-run

# Verbose output
/skill test my-skill --verbose

# Output format
/artifacts list --format json

# Filter
/mcp list --type external

# Interactive mode
/skill new my-skill --interactive
```

## Tips and Tricks

### 1. Tab Completion

Press TAB for auto-completion:

```bash
> /ag[TAB]
> /agent [TAB]
  new  list  edit  delete  test

> /agent n[TAB]
> /agent new
```

### 2. Command History

Use arrow keys to navigate command history:

```bash
> ↑  # Previous command
> ↓  # Next command
```

### 3. Quick Edits

Open files directly from CLI:

```bash
> edit agents/my-agent/config.yaml
# Opens in $EDITOR (vim, nano, code, etc.)
```

### 4. Aliases

Create command aliases:

```bash
> /alias nr="/manifest run manifests/my-research.yaml"
> nr  # Runs the workflow
```

### 5. Session Persistence

Sessions auto-save and can be resumed:

```bash
# Session automatically saved on exit
> /quit

# Resume later
$ kautilya --resume last
# or
$ kautilya --resume session_abc123
```

### 6. Batch Mode

Run multiple commands from file:

```bash
$ kautilya --batch commands.txt

# commands.txt:
/agent new research-agent --role research
/skill new extract-entities
/manifest new my-workflow
```

### 7. Output Redirection

Save command output:

```bash
> /artifacts list > artifacts.json
> /manifest run workflow.yaml | tee execution.log
```

### 8. Environment Variables

Configure via environment:

```bash
export KAUTILYA_EDITOR=code
export KAUTILYA_LLM_PROVIDER=anthropic
export KAUTILYA_AUTO_SAVE=true

kautilya
```

## Troubleshooting

### Common Issues

**1. LLM Connection Fails**

```
Error: Failed to connect to LLM provider 'anthropic'
```

**Solution:**
```bash
# Check API key
> echo $ANTHROPIC_API_KEY

# Test connection
> /llm test

# Reconfigure if needed
> /llm config
```

**2. Skill Import Fails**

```
Error: Invalid skill package: missing SKILL.md
```

**Solution:**
```bash
# Validate package structure
> unzip -l skill.zip

# Import with format specification
> /skill import skill.zip --format native

# Or convert after import
> /skill convert my-skill --to hybrid
```

**3. Workflow Execution Timeout**

```
Error: Step 'research' timed out after 60s
```

**Solution:**
```bash
# Increase timeout in manifest
> edit manifests/my-workflow.yaml
# Change: timeout: 120

# Or use command option
> /manifest run my-workflow.yaml --timeout 120
```

**4. MCP Tool Not Found**

```
Error: Tool 'github.search_repos' not found in catalog
```

**Solution:**
```bash
# Refresh catalog
> /mcp refresh-catalog

# Test server
> /mcp test github

# List available tools
> /mcp list github
```

### Debug Mode

Enable debug logging:

```bash
# Set log level
> /config set log_level DEBUG

# View real-time logs
> /logs --follow

# Or run with debug flag
$ kautilya --debug
```

### Getting Help

```bash
# Command help
> /help [command]

# Show examples
> /help agent new --examples

# Interactive tutorial
> /tutorial

# Documentation
> /docs

# Report issue
> /issue "Description of problem"
# Opens GitHub issue with debug info
```

## Next Steps

### Beginner Path

1. ✓ Complete this guide
2. Try example workflows in `examples/`
3. Create your first custom skill
4. Build a simple research workflow
5. Share your workflow in discussions

### Intermediate Path

1. Build multi-agent workflows
2. Integrate MCP tools
3. Create skill marketplace submissions
4. Use research platform for real projects
5. Contribute skills to community

### Advanced Path

1. Design complex orchestration patterns
2. Build custom MCP servers
3. Create research templates
4. Optimize workflow performance
5. Contribute to framework core

## Additional Resources

- **Documentation**: `/docs/` directory
- **Examples**: `/examples/` directory
- **Templates**: `/templates/` directory
- **Community**: https://github.com/paragajg/agentic-framework/discussions
- **Issues**: https://github.com/paragajg/agentic-framework/issues

---

**Happy Building! 🚀**

For questions or feedback, reach out via GitHub Discussions or Issues.
