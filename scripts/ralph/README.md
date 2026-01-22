# Ralph: Autonomous AI Agent Loop

Ralph is an autonomous AI coding assistant that iteratively implements Product Requirement Documents (PRDs) using AI code generation. Based on Geoffrey Huntley's Ralph pattern and Ryan Carson's implementation, it maintains fresh context across iterations while accumulating learnings.

**Ralph is a standalone tool** that can work with any codebase. While it was originally developed for King AI v2, it operates independently and can be used in any software project.

## Key Features

- **Autonomous Code Editing**: Ralph can now edit existing code, not just create new files
- **Multiple Edit Patterns**: Supports full file replacement, search/replace edits, and code insertion
- **Fuzzy Matching**: Attempts to find approximate matches when exact text isn't found (85% similarity)
- **GitHub CLI Integration**: Automatically creates/updates PRs and adds progress comments
- **Context-Aware**: Reads existing files to provide better context to AI for editing
- **Smart File Application**: Multiple patterns for applying code changes (filepath, edit, insert)
- **PRD-Driven Development**: Uses prd.json to track and implement user stories systematically

## Quick Start

1. **Check authentication**: `python check_auth.py`
2. **Set up GitHub token** (if needed): `$env:GITHUB_TOKEN = 'your_token'`
3. **Run Ralph**: `python ralph.py`

## Requirements

- Python 3.8+
- GitHub Copilot CLI (`npm install -g @github/copilot-cli`)
- A `prd.json` file in the project root
- Optional: `progress.txt` for tracking progress across runs

## Overview

Ralph implements user stories from a PRD by:
1. Reading the next incomplete story from `prd.json`
2. Using GitHub Copilot CLI to generate implementation code
3. Applying code changes to the codebase
4. Running quality checks and tests
5. Committing changes and marking stories complete
6. Repeating until all stories are implemented or max retries reached

## Key Features

- **Cross-platform**: Python implementation works on Windows, Mac, and Linux
- **AI-powered**: Uses GitHub Copilot CLI for code generation
- **Autonomous Code Editing**: Can edit existing code, not just create new files
- **Multiple Edit Patterns**: Supports full file replacement, search/replace, and targeted insertion
- **Fuzzy Matching**: Attempts approximate matches when exact text isn't found (85% similarity)
- **GitHub CLI Integration**: Automatically creates/updates PRs and tracks progress
- **Context-Aware Generation**: Reads existing files to provide better AI context
- **Smart retry logic**: Max 3 attempts per story before moving on
- **Quality gates**: Automated testing, linting, and type checking
- **Git integration**: Automatic branching, committing, pushing, and PR management
- **Error recovery**: Continues working even if individual stories fail
- **Failed story tracking**: Marks stories as failed after max retries
- **Windows asyncio fixes**: Proper event loop handling for Windows

## Files

- `ralph.py` - Main Python implementation with autonomous editing capabilities
- `check_auth.py` - Authentication checker and setup helper
- `generate_code.py` - AI code generation wrapper
- `prompt.md` - Enhanced AI prompt template with edit pattern documentation
- `ralph.bat` - Windows batch wrapper (legacy)
- `ralph.sh` - Bash implementation (legacy)

## Code Editing Capabilities

Ralph now supports three patterns for modifying code:

### 1. Full File Creation/Replacement
Use when creating new files or completely rewriting existing ones:
```
```filepath: path/to/file.py
[Complete file content]
```
```

### 2. Search/Replace Editing
Use for modifying specific sections of existing files:
```
```edit: path/to/file.py
SEARCH:
[Exact text to find]

REPLACE:
[New text to use]
```
```

Features:
- Exact text matching for precise edits
- Fuzzy matching fallback (85% similarity threshold)
- Only replaces first occurrence for safety
- Shows similarity percentage when using fuzzy match

### 3. Targeted Code Insertion
Use for adding code before or after specific markers:
```
```insert: path/to/file.py after "marker text"
[Code to insert]
```
```

### GitHub CLI Integration

Ralph now automatically:
- **Pushes commits** to remote repository after each completed story
- **Creates Pull Requests** for the feature branch if one doesn't exist
- **Updates PR description** with completed and remaining stories
- **Adds PR comments** documenting each completed story
- **Tracks progress** via GitHub's PR interface

This integration requires:
- GitHub CLI (`gh`) installed and authenticated
- Push access to the repository
- Network connectivity

If GitHub operations fail, Ralph continues with local commits only.

## Architecture

```
PRD JSON ──┐
           ├── Ralph Loop ──┤
Progress.txt─┘             ├── Copilot CLI ─── Code Edits ─── Quality Checks ─── Commit/Push ─── PR Update
File Context ───────────────┘                      ↓
                                              (filepath/edit/insert)
```

### Components
- **ralph.py**: Main orchestration with enhanced editing logic
- **prompt.md**: Template with edit pattern documentation
- **prd.json**: User stories with completion status
- **progress.txt**: Learnings and context from iterations
- **GitHub CLI**: PR and issue management integration

## Prerequisites

### Required
- **GitHub Copilot Subscription**: Active Copilot plan
- **GitHub Copilot CLI**: Installed and authenticated
- **Python 3.8+**: For Ralph script
- **Git**: Version control

### Optional
- **GitHub CLI (gh)**: For easier authentication
- **Quality Tools**: pytest, ruff, mypy, bandit (configured in project)

## Installation

### 1. Install GitHub Copilot CLI
```powershell
# Windows (PowerShell)
npm install -g @githubnext/github-copilot-cli

# Or with winget
winget install GitHub.Copilot
```

### 2. Set up Authentication

**Option A: GitHub CLI (Recommended)**
```powershell
gh auth login
```

**Option B: Personal Access Token**
```powershell
# Create token at: https://github.com/settings/tokens
# Required scopes: read:user, user:email
$env:GITHUB_TOKEN = 'your_token_here'
```

### 3. Verify Setup
```powershell
python scripts/ralph/check_auth.py
```

## Usage

### Basic Usage
```powershell
# Run until all stories complete or max retries
python scripts/ralph/ralph.py

# Run for specific number of iterations
python scripts/ralph/ralph.py 5

# Set custom max retries per story
python scripts/ralph/ralph.py --max-retries 5

# Reset all stories to incomplete
python scripts/ralph/ralph.py --reset
```

### Command Line Arguments
```
usage: ralph.py [-h] [--max-retries MAX_RETRIES] [--reset] [max_iterations]

positional arguments:
  max_iterations        Maximum number of iterations (default: unlimited)

optional arguments:
  -h, --help           Show this help message and exit
  --max-retries N      Maximum retries per story before marking as failed (default: 3)
  --reset              Reset all stories to incomplete
```

### Workflow

1. **Create PRD**: Edit `prd.json` with your user stories
2. **Check Auth**: Run `python scripts/ralph/check_auth.py`
3. **Run Ralph**: Run `python scripts/ralph/ralph.py`
4. **Monitor**: Check `progress.txt` and git commits

### Understanding Story Status

Stories can have three states:
- **Incomplete** (`passes: false`): Not yet implemented
- **Complete** (`passes: true`): Successfully implemented
- **Failed** (`metadata.failed: true`): Exceeded max retries, skipped

### Retry Logic

- Each story gets up to 3 attempts by default (configurable with `--max-retries`)
- After max retries, story is marked as failed and Ralph moves to next story
- Failed stories are tracked in `prd.json` metadata
- Use `--reset` to clear all completion and failure status

## Configuration

### Customizing Prompts
Edit `scripts/ralph/prompt.md` to include:
- Project-specific conventions
- Additional quality checks
- Codebase patterns
- Team preferences

### Quality Checks
Quality checks run automatically in `ralph.py`:
- Python: Syntax check with `py_compile`
- TypeScript: Type checking with `npm run type-check`
- Extensible: Add more checks in `_run_quality_checks()`

### Project Context
Update the prompt template with:
- Architecture details
- Framework preferences
- Coding standards
- Common gotchas

## Usage

### Basic Usage
```bash
# Run with default 10 iterations
./scripts/ralph/ralph.sh

# Run specific number of iterations
./scripts/ralph/ralph.sh 5

# Run until completion
./scripts/ralph/ralph.sh 100
```

### Monitoring
```bash
# Check current status
cat prd.json | python3 -c "import json, sys; data=json.load(sys.stdin); print(f\"Completed: {sum(1 for s in data['userStories'] if s['passes'])}/{len(data['userStories'])}\")"

# View recent progress
tail -20 progress.txt

# Check git history
git log --oneline -10
```

### Manual Intervention
If needed, you can:
- Edit `prd.json` to modify stories
- Update `progress.txt` with additional context
- Manually commit changes
- Skip problematic stories by marking as `passes: true`

## PRD Format

### JSON Structure
```json
{
  "branchName": "feature/my-feature",
  "userStories": [
    {
      "id": "story-1",
      "title": "Add user authentication",
      "description": "Implement login/logout functionality",
      "acceptanceCriteria": "Users can register, login, logout. Passwords hashed.",
      "passes": false,
      "priority": 1
    }
  ]
}
```

### Story Guidelines
- **Small Scope**: Each story should be implementable in one context window
- **Clear Acceptance**: Specific, testable criteria
- **Independent**: Minimal dependencies on other stories
- **Valuable**: Provides business value when complete

### Conversion from Markdown
The Ralph skill converts markdown PRDs with this structure:
```markdown
# Feature: User Authentication

## User Story 1: Login Form
**As a** user
**I want** to login
**So that** I can access my account

**Acceptance Criteria:**
- Form validates email/password
- Shows error messages
- Redirects on success
```

## Quality Assurance

### Automated Checks
Ralph runs these checks after each implementation:
- **pytest**: Unit and integration tests
- **ruff**: Code linting and formatting
- **mypy**: Type checking
- **bandit**: Security scanning

### Manual Verification
For UI stories, include in acceptance criteria:
- "Verify in browser using dev-tools"
- Ralph will prompt for manual verification

### Failure Handling
If checks fail:
- Changes are not committed
- Failure is logged to `progress.txt`
- Next iteration continues with next story
- No iteration limit reached, Ralph continues

## Troubleshooting

### Common Issues

#### Copilot Not Launching
```bash
# Check installation
copilot --version

# Re-authenticate
copilot
/login
```

#### JSON Parsing Errors
```bash
# Validate prd.json
python3 -c "import json; json.load(open('prd.json')); print('Valid JSON')"
```

#### Quality Checks Failing
```bash
# Run checks manually
pytest
ruff check
mypy .
bandit -r .
```

#### Permission Issues
```bash
# Make script executable
chmod +x scripts/ralph/ralph.sh
```

### Debug Mode
Add debug output to `ralph.sh`:
```bash
set -x  # Enable debug
./scripts/ralph/ralph.sh 1
```

### Recovery
If Ralph gets stuck:
1. Check `prd.json` for malformed data
2. Review `progress.txt` for error patterns
3. Manually complete problematic stories
4. Restart Ralph

## Advanced Features

### Custom Quality Checks
Add project-specific checks to the script:
```bash
# In ralph.sh, after "# Run quality checks"
npm run lint
docker-compose build
./custom-check.sh
```

### Integration with CI/CD
Ralph can be integrated into CI pipelines:
```yaml
# .github/workflows/ralph.yml
name: Ralph Implementation
on: [push]
jobs:
  ralph:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run Ralph
        run: ./scripts/ralph/ralph.sh 1
```

### Multi-Branch Strategy
For complex features:
- Use feature branches per story
- Merge completed stories to main
- Ralph creates `feature/story-{id}` branches

### Learning Enhancement
Add to `AGENTS.md` after each iteration:
- Discovered patterns
- Gotchas encountered
- Useful conventions
- Integration points

### Parallel Execution
For independent stories, run multiple Ralph instances:
```bash
./scripts/ralph/ralph.sh 1 &  # Instance 1
./scripts/ralph/ralph.sh 1 &  # Instance 2
wait
```

## Contributing

### Extending Ralph
- Add new quality checks
- Customize prompts for specific languages/frameworks
- Integrate additional AI assistants
- Add support for different PRD formats

### Skills Development
- Enhance PRD generation prompts
- Improve markdown-to-JSON conversion
- Add validation for PRD completeness

## License

This implementation is based on the Ralph pattern by Geoffrey Huntley and Ryan Carson's work.

## Support

For issues with Ralph:
1. Check this documentation
2. Review `progress.txt` for patterns
3. Test components individually
4. Open issues with debug output

---

**Ralph v1.0** - Autonomous AI Agent Loop for Product Development