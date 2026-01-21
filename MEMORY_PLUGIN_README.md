# GitHub Copilot Memory Plugin

A comprehensive long-term memory system for AI-assisted development that learns from your coding sessions and continuously improves AI assistance.

## 🎯 Overview

The GitHub Copilot Memory Plugin implements a **Generative Agents-inspired memory architecture** with three key components:

1. **Diary**: Records detailed session information from coding activities
2. **Reflection**: Analyzes patterns across diary entries to extract insights
3. **Memory**: Stores learnings in long-term memory for future retrieval

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   VS Code       │    │   King AI       │    │   Memory        │
│   Extension     │    │   Agents        │    │   System        │
│                 │    │                 │    │                 │
│ • /diary cmd    │    │ • Auto-diary    │    │ • Diary entries │
│ • /reflect cmd  │    │ • Reflection    │    │ • Pattern       │
│ • COPILOT.md    │    │ • Long-term     │    │   analysis      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📦 Components

### 1. VS Code Extension (`copilot-memory-plugin/`)
- **Purpose**: Direct Copilot Chat integration
- **Features**: `/diary` and `/reflect` commands
- **Memory File**: `.copilot/memory/COPILOT.md`

### 2. King AI Integration (`src/memory/diary_reflection.py`)
- **Purpose**: Agent memory learning
- **Features**: Automatic diary generation, reflection analysis
- **Memory File**: `.king_ai/memory/diary/` and long-term storage

## 🚀 Quick Start

### Option A: VS Code Extension

1. **Open Extension Folder**:
   ```bash
   code copilot-memory-plugin/
   ```

2. **Install Dependencies**:
   ```bash
   npm install
   npm run compile
   ```

3. **Run Extension**:
   - Press `F5` in VS Code
   - Open GitHub Copilot Chat
   - Type `/diary` to generate a session diary

### Option B: King AI Integration

1. **Run Memory Test**:
   ```bash
   python test_memory_system.py
   ```

2. **Check Generated Files**:
   - Diary entries: `.king_ai/memory/diary/`
   - Processed log: `.king_ai/memory/processed.log`

## 📖 Usage Guide

### VS Code Extension Commands

#### `/diary` Command
Generates a diary entry for the current coding session.

**What it captures**:
- Current workspace files
- Git changes
- Session context
- User preferences

**Example Output**:
```markdown
# Diary Entry: 2026-01-19 - Session abc123

## Task Summary
Implemented user authentication system

## Work Done
- Modified 3 files
- Added JWT token handling

## Design Decisions
- Used bcrypt for password hashing
- Implemented refresh tokens

## User Preferences
- Prefers functional components
- Uses TypeScript strict mode
```

#### `/reflect` Command
Analyzes diary entries to identify patterns and update memory.

**What it does**:
- Scans unprocessed diary entries
- Identifies recurring patterns
- Updates `COPILOT.md` with learnings
- Marks entries as processed

### King AI Agent Integration

#### Automatic Diary Generation
Agents automatically create diary entries after completing tasks:

```python
from src.agents.reflective_loop import ReflectiveAgentLoop
from src.memory.manager import MemoryManager

# Initialize with memory
memory_manager = MemoryManager()
agent = ReflectiveAgentLoop(
    project_id="my_project",
    memory_manager=memory_manager
)

# Run task (diary created automatically)
result = await agent.run("Implement user dashboard")
```

#### Manual Reflection
Trigger reflection analysis:

```python
from src.memory.diary_reflection import ReflectionEngine

reflection = ReflectionEngine("my_project", memory_manager)
result = await reflection.perform_reflection()
print(f"Analyzed {result['entries_analyzed']} entries")
```

## 📁 File Structure

### VS Code Extension
```
copilot-memory-plugin/
├── src/extension.ts          # Main extension code
├── package.json              # Extension manifest
├── tsconfig.json             # TypeScript config
└── README.md                 # Extension docs

.copilot/memory/
├── diary/                    # Diary entries
├── processed.log             # Processed entries
└── COPILOT.md               # Long-term memory
```

### King AI Integration
```
.king_ai/memory/
├── diary/                    # Agent diary entries
├── processed.log             # Reflection tracking
└── [long-term storage]       # Vector embeddings

src/memory/
├── diary_reflection.py       # Diary & reflection engine
├── manager.py               # Memory manager
├── tier1_recent.py          # Recent context
├── tier2_summaries.py       # Session summaries
└── tier3_longterm.py        # Long-term memory
```

## 🎛️ Configuration

### VS Code Extension Settings
Add to `settings.json`:
```json
{
  "copilot-memory": {
    "autoDiary": true,
    "reflectionInterval": 10,
    "memoryPath": ".copilot/memory"
  }
}
```

### King AI Memory Settings
Configure in your agent initialization:
```python
agent_loop = ReflectiveAgentLoop(
    project_id="my_project",
    memory_manager=memory_manager,
    max_iterations=3,
    min_quality_score=0.7
)
```

## 📊 Memory Categories

### Diary Sections
- **Task Summary**: What was accomplished
- **Work Done**: Actions taken and results
- **Design Decisions**: Architectural choices
- **User Preferences**: Coding style preferences
- **Code Review Feedback**: Quality feedback
- **Challenges**: Problems encountered
- **Solutions**: How problems were solved
- **Code Patterns**: Recurring implementation patterns
- **Learnings**: Key insights gained

### Long-term Memory Categories
- **DECISION**: Important decisions made
- **FINDING**: Research findings
- **PREFERENCE**: User preferences
- **PROCESS**: Workflow information
- **ENTITY**: Important entities
- **CODE**: Code snippets
- **ERROR**: Errors and solutions
- **OTHER**: Miscellaneous

## 🔍 Examples

### Example Diary Entry
```markdown
# Diary Entry: 2026-01-19 - Session abc123

## Agent Information
- Agent ID: reflective_loop
- Project: authentication-system

## Task Summary
Implement secure user authentication with JWT tokens

## Work Done
- **planning**: Created plan with 4 steps ✓
- **execution**: Executed 4 steps ✓
- **validation**: Validation score: 0.95 ✓

## Design Decisions
- Used bcrypt for password hashing
- Implemented refresh token rotation
- Added rate limiting for login attempts

## Challenges
- Token expiration handling
- Password reset flow security

## Solutions
- Implemented sliding session windows
- Used secure random for token generation

## Learnings
- Always validate tokens on each request
- Implement proper logout cleanup
```

### Example Reflection Output
```
Reflection Analysis

Patterns Identified:
- Consistent use of bcrypt for password hashing
- Preference for JWT with refresh tokens
- Security-first approach to authentication

Updated COPILOT.md with:
- Use bcrypt for password hashing
- Implement refresh token rotation
- Add rate limiting to auth endpoints
```

## 🐛 Troubleshooting

### VS Code Extension Issues

**Extension not loading**:
- Check `npm run compile` completed successfully
- Verify VS Code version supports chat API
- Check debug console for errors

**Commands not appearing**:
- Ensure Copilot Chat is open
- Try reloading VS Code window (Ctrl+Shift+P → "Reload Window")
- Check extension is activated

### King AI Memory Issues

**Diary not created**:
```python
# Check diary directory exists
import os
print(os.path.exists(".king_ai/memory/diary"))
```

**Reflection not working**:
```python
# Check for unprocessed entries
from src.memory.diary_reflection import AgentDiary
diary = AgentDiary("project_id")
entries = diary.get_unprocessed_entries()
print(f"Unprocessed: {len(entries)}")
```

**Memory storage errors**:
- Ensure proper permissions on memory directories
- Check LLM/embedding clients are configured
- Verify Redis connection if using persistence

### Common Issues

**"await outside async function"**:
- Ensure all memory operations are in async contexts
- Check function signatures match async/await patterns

**Memory not persisting**:
- Verify storage paths exist
- Check file permissions
- Ensure embedding client is available

## 🔧 Advanced Features

### Custom Reflection Rules
Extend reflection analysis:

```python
class CustomReflectionEngine(ReflectionEngine):
    def _synthesize_insights(self, patterns):
        insights = super()._synthesize_insights(patterns)
        # Add custom pattern detection
        if "typescript" in patterns.get("preferences", []):
            insights.append("Strong preference for TypeScript")
        return insights
```

### Memory Retrieval
Access stored memories:

```python
# Search by query
memories = await memory_manager.search_memories(
    project_id="my_project",
    query="authentication patterns",
    limit=5
)

# Get decisions
decisions = await memory_manager.get_decisions(
    project_id="my_project",
    limit=10
)
```

### Integration with Ralph
The autonomous Ralph agent can use memory:

```bash
# Run Ralph with memory-enabled agents
cd scripts/ralph
./ralph.sh 5  # Process 5 user stories with memory
```

## 📈 Performance & Scaling

### Memory Limits
- **Diary Entries**: Unlimited (compressed over time)
- **Long-term Memory**: Vector search optimized
- **Reflection Frequency**: Configurable intervals

### Optimization Tips
- Run reflection periodically, not after every session
- Use project-specific memory namespaces
- Implement memory cleanup for old entries
- Cache frequent memory queries

## 🤝 Contributing

### Adding New Memory Categories
1. Update `MemoryCategory` enum in `tier3_longterm.py`
2. Add extraction logic in reflection engine
3. Update diary templates

### Extending Diary Sections
1. Modify `AgentDiary.generate_diary_entry()` method
2. Update analysis patterns in `ReflectionEngine`
3. Add new sections to diary templates

### Testing
```bash
# Run memory tests
python test_memory_system.py

# Test with different scenarios
python -m pytest tests/test_memory*.py -v
```

## � Documentation

### Quick Start Guides
- **[WELCOME.md](copilot-memory-plugin/WELCOME.md)** - Quick start guide for new users
- **[TROUBLESHOOTING.md](copilot-memory-plugin/TROUBLESHOOTING.md)** - Common issues and solutions

### Examples
- **[sample-diary-entry.md](copilot-memory-plugin/examples/sample-diary-entry.md)** - Example diary entry
- **[sample-reflection.md](copilot-memory-plugin/examples/sample-reflection.md)** - Example reflection analysis

### Project Files
- **[CHANGELOG.md](copilot-memory-plugin/CHANGELOG.md)** - Version history and changes
- **[LICENSE](copilot-memory-plugin/LICENSE)** - MIT License

## 🙏 Acknowledgments

Inspired by:
- **Claude Diary** by rlancemartin
- **Generative Agents** paper
- **Anthropic's** internal memory practices
- **Geoffrey Huntley's** Ralph pattern

---

**Happy coding with memory-enhanced AI assistance! 🚀**