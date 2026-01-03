# Simple Agent Example

This example demonstrates how to use a skill directly through the SkillExecutor without the full orchestrator.

## Overview

Creates a simple agent that:
- Loads the `text_summarize` skill
- Executes the skill with sample text
- Shows different summarization styles (concise, bullet points)

## Prerequisites

```bash
# Ensure framework is installed
pip install -e ../..

# No additional dependencies required
```

## Running the Example

```bash
# From the repository root
cd examples/01-simple-agent
python run.py
```

## What This Example Shows

1. **Direct Skill Execution**: How to load and execute a skill without the orchestrator
2. **Skill Configuration**: How to configure the SkillExecutor
3. **Passing Inputs**: How to pass parameters to skills
4. **Handling Outputs**: How to process skill results

## Expected Output

```
============================================================
Simple Agent Example
============================================================

📝 Task: Summarize a research article about AI agents

🤖 Agent Configuration:
   - Role: Research Assistant
   - Skills: text_summarize
   - Execution: Direct skill invocation

🔄 Execution Flow:
   [1/3] Loading text_summarize skill...
   [2/3] Executing summarization...

✅ Summary (concise):
   Multi-agent systems enable specialized agents to collaborate
   on complex tasks, outperforming single-agent systems through
   coordination and diverse expertise.
   Word count: 18
   Execution time: 5ms

✅ Summary (bullet_points):
   • Multi-agent systems advance AI through agent collaboration
   • Systems excel in diverse expertise and parallel processing
   • Enterprise adoption growing in finance, healthcare sectors
   Word count: 22
   Execution time: 0ms

   [3/3] Task completed successfully!

============================================================
Example Complete!
============================================================
```

## Files

- `run.py` - Main script demonstrating skill execution
- `README.md` - This file

## Code Walkthrough

### 1. Initialize Skill Executor

```python
from code_exec.service.executor import SkillExecutor
from code_exec.service.config import CodeExecConfig

config = CodeExecConfig(
    skills_dir="../../code-exec/skills",
    sandbox_mode=False  # For demo purposes
)
executor = SkillExecutor(config)
```

### 2. Execute a Skill

```python
result = await executor.execute_skill(
    skill_name="text_summarize",
    inputs={
        "text": "Your text here...",
        "style": "concise",
        "max_sentences": 3
    }
)
```

### 3. Process Results

```python
if result.get("success"):
    summary = result["result"]["summary"]
    word_count = result["result"]["word_count"]
    exec_time = result["execution_time_ms"]
```

## Next Steps

- See `02-multi-step-workflow/` for multi-agent orchestration
- See `03-custom-skill/` for creating your own skills
- See `04-mcp-integration/` for external tool integration
