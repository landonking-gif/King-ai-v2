# Ralph Enhanced Capabilities - Example Usage

This document demonstrates Ralph's new autonomous code editing capabilities.

## Overview

Ralph can now:
1. **Edit existing code** using search/replace patterns
2. **Insert code** at specific locations
3. **Use fuzzy matching** to find approximate text matches
4. **Create/update PRs** automatically using GitHub CLI
5. **Provide file context** to AI for better code generation

## Example 1: Search/Replace Editing

Given an existing file `src/api/routes.py`:
```python
from fastapi import APIRouter

router = APIRouter()

@router.get("/health")
async def health_check():
    return {"status": "healthy"}
```

Ralph can edit it using:
```markdown
```edit: src/api/routes.py
SEARCH:
@router.get("/health")
async def health_check():
    return {"status": "healthy"}

REPLACE:
@router.get("/health")
async def health_check():
    """Enhanced health check with version info"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }
```
```

## Example 2: Code Insertion

To add an import at the top of the file:
```markdown
```insert: src/api/routes.py after "from fastapi import APIRouter"
from datetime import datetime
```
```

Or to add code before a specific line:
```markdown
```insert: src/utils/logger.py before "class Logger:"
# Logger configuration
import logging
```
```

## Example 3: Fuzzy Matching

Ralph will attempt to find approximate matches when exact text isn't found.

If the file has:
```python
def  calculate_total(items):  # Extra space
    return sum(items)
```

And you search for:
```python
def calculate_total(items):
    return sum(items)
```

Ralph will find it with **~95% similarity** and apply the change.

## Example 4: Creating New Files

```markdown
```filepath: src/models/user.py
from pydantic import BaseModel
from typing import Optional

class User(BaseModel):
    id: int
    name: str
    email: str
    is_active: bool = True
```
```

## Example 5: GitHub CLI Integration

Ralph automatically:

1. **Pushes commits** after each completed story
2. **Creates PRs** if one doesn't exist for the branch
3. **Updates PR description** with progress:
   ```markdown
   ## Completed User Stories
   - [x] **Add user authentication** (story-1)
   - [x] **Create user profile page** (story-2)
   
   ## Remaining Stories
   - [ ] **Add password reset** (story-3)
   ```
4. **Adds PR comments** for each completed story:
   ```
   ✅ Completed story `story-1`: **Add user authentication**
   
   Commit: Ralph: Complete story story-1 - Add user authentication
   ```

## Example 6: Context-Aware Generation

Ralph reads existing files mentioned in prompts to provide better context to the AI:

```markdown
User Story: Update the login route to include rate limiting

Ralph automatically:
1. Reads src/api/routes.py
2. Adds file content to AI prompt
3. AI generates edit with full context
4. Ralph applies the changes
```

## Best Practices

### 1. Use Appropriate Patterns

- **filepath**: For new files or complete rewrites
- **edit**: For modifying specific sections (preferred for existing files)
- **insert**: For adding code at specific locations

### 2. Provide Enough Context in SEARCH

Bad:
```
SEARCH:
return result
```

Good:
```
SEARCH:
async def process_data():
    result = compute()
    return result
```

### 3. Test in Isolation First

Before running Ralph on your entire PRD:
1. Test with a single story
2. Verify the edit patterns work
3. Check the PR integration
4. Then run on all stories

### 4. Monitor Progress

```bash
# Check completed stories
python3 -c "import json; data=json.load(open('prd.json')); print(f\"{sum(1 for s in data['userStories'] if s['passes'])}/{len(data['userStories'])} stories completed\")"

# View recent activity
tail -20 progress.txt

# Check PR status
gh pr view
```

## Troubleshooting

### Edit Pattern Not Working

If Ralph can't find the text to replace:
1. Check for whitespace differences (tabs vs spaces)
2. Verify the text exists in the file
3. Ralph will attempt fuzzy matching (85% threshold)
4. Check the output for similarity percentage

### GitHub CLI Issues

If PR creation fails:
1. Ensure `gh` is installed and authenticated
2. Check repository permissions
3. Verify network connectivity
4. Ralph continues with local commits if PR fails

### Quality Checks Failing

Ralph runs quality checks after each story:
1. Python syntax check
2. TypeScript type check (if applicable)
3. Custom checks (extensible)

If checks fail, changes are not committed.

## Advanced Usage

### Custom GitHub Actions

Create `.github/workflows/ralph.yml`:
```yaml
name: Ralph Auto-Implementation
on:
  push:
    paths:
      - 'prd.json'
jobs:
  ralph:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup Ralph
        run: |
          pip install -e .
          npm install -g @githubnext/github-copilot-cli
      - name: Run Ralph
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: python scripts/ralph/ralph.py 1
```

### Integration with Issue Tracking

Ralph can be extended to:
- Create issues for failed stories
- Link commits to issues
- Update issue status automatically
- Generate implementation reports

## Summary

Ralph's enhanced capabilities enable true autonomous code editing:
- ✅ Edit existing code precisely
- ✅ Insert code at specific locations
- ✅ Handle whitespace differences with fuzzy matching
- ✅ Automatically manage PRs and track progress
- ✅ Provide context-aware code generation
- ✅ Continue working despite failures
- ✅ Track story completion and retries

This makes Ralph a powerful tool for implementing PRDs autonomously while maintaining code quality and tracking progress through GitHub.
