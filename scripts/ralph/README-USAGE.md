# Ralph Usage Guide - VS Code Copilot CLI Integration

## Overview
Ralph is now configured to use the **VS Code Copilot CLI** directly to make actual code changes based on your PRD.

## How It Works

### 1. Prerequisites
- **VS Code** with **GitHub Copilot extension** installed
- Run Ralph from a **VS Code terminal** (where `copilot` command is available)
- A valid `prd.json` in your project root

### 2. Verify Copilot CLI Access
In your VS Code terminal, type:
```bash
copilot
```
If this works, you're ready to use Ralph!

## Running Ralph

### From VS Code Terminal (PowerShell)
```powershell
# Run with unlimited iterations
python scripts/ralph/ralph.py

# Run for 10 iterations max
python scripts/ralph/ralph.py 10

# With custom retry limit (5 attempts per story)
python scripts/ralph/ralph.py --max-retries 5

# Reset all stories to incomplete
python scripts/ralph/ralph.py --reset
```

## What Ralph Does

1. **Reads PRD**: Finds the next incomplete user story from `prd.json`
2. **Creates Prompt**: Builds a comprehensive prompt with:
   - Story details (title, description, acceptance criteria)
   - Full PRD context (all stories and their status)
   - Project architecture and conventions
   - Previous progress from `progress.txt`
3. **Calls Copilot CLI**: Sends the prompt to `copilot` command
4. **Parses Response**: Extracts code blocks using these formats:
   - ````filepath: path/to/file.py` - Creates/updates complete files
   - ````edit: path/to/file.py` - Edits existing files with SEARCH/REPLACE
5. **Applies Changes**: Writes code to your codebase
6. **Quality Checks**: Runs Python syntax checks, TypeScript checks
7. **Git Commit**: Stages and commits changes with message: `Ralph: Complete story {id} - {title}`
8. **Updates PRD**: Marks story as `"passes": true`
9. **Repeats**: Moves to next story

## Code Output Formats

Ralph expects Copilot to respond with code in these specific formats:

### Create or Update Complete File
```markdown
```filepath: src/api/routes/health.py
from fastapi import APIRouter

router = APIRouter()

@router.get("/health")
async def health_check():
    return {"status": "healthy"}
```
```

### Edit Existing File (Search & Replace)
```markdown
```edit: src/main.py
SEARCH:
app = FastAPI()

REPLACE:
app = FastAPI(
    title="King AI API",
    version="1.0.0"
)
```
```

## PRD Format

Your `prd.json` should look like:
```json
{
  "branchName": "feature/your-feature",
  "userStories": [
    {
      "id": "story-1",
      "title": "Implement health check endpoint",
      "description": "Create a /health endpoint that returns system status",
      "acceptanceCriteria": "GET /health returns 200 with status JSON",
      "passes": false,
      "priority": 1
    },
    {
      "id": "story-2",
      "title": "Add authentication",
      "description": "Implement JWT-based authentication",
      "acceptanceCriteria": "Login endpoint issues JWT tokens",
      "passes": false,
      "priority": 2
    }
  ]
}
```

## Story Selection Logic

Ralph selects stories based on:
1. **Incomplete status**: `"passes": false`
2. **Not failed**: Stories that haven't exceeded max retries
3. **Priority order**: Lower priority number = higher priority

## Retry Behavior

- **Default**: 3 attempts per story
- After max retries, story is marked as **failed** and skipped
- Failed stories are flagged in PRD with metadata:
  ```json
  "metadata": {
    "failed": true,
    "failedAt": "2026-01-22T10:30:00"
  }
  ```

## Progress Tracking

Ralph maintains `progress.txt` with logs like:
```
# Ralph Progress Log
Started: 2026-01-22T10:00:00
Project: king-ai-v2

[2026-01-22T10:05:00] Iteration 1: SUCCESS - story-1
  Story: Implement health check endpoint

[2026-01-22T10:10:00] Iteration 2: FAILED - story-2
  Story: Add authentication
  Error: Implementation failed (Attempt 1/3)
```

## Git Integration

Ralph automatically:
- Checks out the branch specified in PRD (`branchName`)
- Creates branch if it doesn't exist
- Stages all changes after successful implementation
- Commits with descriptive message
- Leaves branch ready for review/push

## Troubleshooting

### "Copilot CLI not found"
**Cause**: Not running from VS Code terminal or Copilot not installed

**Fix**:
1. Open project in VS Code
2. Open integrated terminal (Ctrl+` or View > Terminal)
3. Type `copilot` to verify
4. If not working, install GitHub Copilot extension

### "No file changes detected"
**Cause**: Copilot response didn't include proper code blocks

**Why this happens**:
- Story might be too vague
- Copilot provided explanation instead of code
- Story is already complete

**Fix**:
- Review the Copilot response preview in terminal output
- Make story description more specific
- Check if files already exist
- Manually implement and mark story complete in PRD

### "Quality checks failed"
**Cause**: Generated code has syntax errors or type issues

**Fix**:
1. Check the git diff: `git diff`
2. Fix syntax errors manually
3. Run Ralph again (it will retry the same story)
4. Or manually complete and mark story as passed

### Story fails repeatedly
**Solutions**:
1. **Break it down**: Split into smaller, more focused stories
2. **Add details**: Make acceptance criteria more specific
3. **Manual implementation**: Implement manually and update PRD
4. **Check dependencies**: Ensure prerequisite stories are complete

## Best Practices

### 1. Write Clear Stories
```json
// ❌ Bad - Too vague
{
  "title": "Add features",
  "description": "Make the app better"
}

// ✅ Good - Specific and actionable
{
  "title": "Add user registration endpoint",
  "description": "Create POST /api/auth/register endpoint that accepts email/password, validates input, hashes password with bcrypt, stores user in database, returns 201 with user ID",
  "acceptanceCriteria": "POST /api/auth/register with valid email/password returns 201 with user ID, invalid input returns 400 with error details, duplicate email returns 409"
}
```

### 2. Use Priority Correctly
- Priority 1: Foundation/core features
- Priority 2: Main features depending on Priority 1
- Priority 3: Nice-to-have features

### 3. Keep Stories Atomic
Each story should:
- Accomplish ONE thing
- Be testable independently
- Take < 15 minutes to implement

### 4. Monitor Progress
```powershell
# Watch Ralph in one terminal
python scripts/ralph/ralph.py

# In another terminal, monitor git commits
git log --oneline

# Check progress file
cat progress.txt
```

### 5. Review Before Pushing
```powershell
# See what Ralph changed
git log --oneline

# Review specific commit
git show HEAD

# Check current diff
git diff main
```

## Advanced Configuration

### Custom Timeout
Edit `ralph.py` line ~239:
```python
timeout=600  # 10 minutes (default)
timeout=1200  # 20 minutes (for complex stories)
```

### Custom Prompt Template
Edit `scripts/ralph/prompt.md` to adjust how Ralph communicates with Copilot.

### Disable Quality Checks
Edit `ralph.py` method `_run_quality_checks()` to skip checks (not recommended).

## Examples

### Example Session
```powershell
PS> python scripts/ralph/ralph.py 5

Starting Ralph Autonomous Agent Loop
Working directory: C:\Users\...\king-ai-v2
Max iterations: 5
Checked out branch: feature/master-control-panel

==================================================
Ralph Iteration 1
==================================================
Working on: Set up backend with FastAPI
Story ID: story-1

🤖 Generating code using GitHub Copilot CLI...
Invoking Copilot CLI with prompt...
✅ Generated code from Copilot CLI

============================================================
APPLYING CODE CHANGES
============================================================

Found 3 file(s) to create/update:

  ✅ Created/updated: src/api/main.py
  ✅ Created/updated: src/api/routes/health.py
  ✅ Created/updated: requirements.txt

============================================================
✅ SUCCESSFULLY APPLIED 3 CHANGE(S)
============================================================

Running quality checks...
Quality check results:
  ✅ Python syntax
✅ Successfully completed story: story-1
Committed changes: Ralph: Complete story story-1 - Set up backend with FastAPI

==================================================
Ralph Iteration 2
==================================================
...
```

## Integration with VS Code Tasks

You can create a VS Code task to run Ralph:

`.vscode/tasks.json`:
```json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "Run Ralph (10 iterations)",
      "type": "shell",
      "command": "python scripts/ralph/ralph.py 10",
      "problemMatcher": []
    },
    {
      "label": "Run Ralph (unlimited)",
      "type": "shell",
      "command": "python scripts/ralph/ralph.py",
      "problemMatcher": []
    },
    {
      "label": "Reset Ralph Stories",
      "type": "shell",
      "command": "python scripts/ralph/ralph.py --reset",
      "problemMatcher": []
    }
  ]
}
```

Then run with: Ctrl+Shift+P > "Tasks: Run Task" > "Run Ralph (10 iterations)"

## Safety & Rollback

Ralph commits after each successful story, so you can always rollback:
```powershell
# Undo last Ralph commit
git reset --hard HEAD~1

# Undo last 5 Ralph commits
git reset --hard HEAD~5

# Or use git revert for safe rollback
git revert HEAD
```

## Summary

Ralph is now configured to:
✅ Use VS Code Copilot CLI directly
✅ Send comprehensive prompts with full PRD context
✅ Parse code blocks and apply changes automatically
✅ Track progress and handle retries intelligently
✅ Commit changes with descriptive messages

Run it from your VS Code terminal and watch it implement your PRD! 🚀
