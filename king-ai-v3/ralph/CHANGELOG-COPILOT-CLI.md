# Ralph Loop - Copilot CLI Integration Update

## Summary of Changes

This update fixes the Ralph loop to properly use the VS Code Copilot CLI and PRD to make actual code changes and edits.

## Files Modified

### 1. `scripts/ralph/ralph.py`
**Changes:**
- **`_generate_code()` method**: Completely rewritten to use VS Code Copilot CLI directly
  - Now invokes `copilot` command (available in VS Code terminal)
  - Sends comprehensive prompts with full PRD context
  - Increased timeout to 10 minutes for complex tasks
  - Better error handling and troubleshooting messages
  
- **`_create_prompt()` method**: Enhanced to include story ID in the prompt
  - Extracts story ID from PRD to match story with title
  - Passes story ID to prompt template for better context
  
- **`_apply_code_changes()` method**: Enhanced with better logging and feedback
  - Detailed output showing number of files found
  - Clear visual separators for code application phase
  - Preview of Copilot response when no changes detected
  - Better error messages and troubleshooting hints

### 2. `scripts/ralph/prompt.md`
**Changes:**
- Complete rewrite of prompt template for better results
- Added `{{STORY_ID}}` placeholder
- More explicit instructions for Copilot to provide actual code
- Clearer output format requirements with examples
- Emphasis on making actual changes, not suggestions
- Better structured sections with critical instructions highlighted
- Comprehensive examples of both file creation and editing formats

### 3. `scripts/ralph/README-USAGE.md` (NEW)
**Purpose:** Complete usage guide for the updated Ralph with Copilot CLI

**Contents:**
- Prerequisites and setup instructions
- How Ralph works (step-by-step)
- Code output format requirements
- PRD format specifications
- Story selection logic
- Retry behavior explanation
- Progress tracking details
- Git integration information
- Comprehensive troubleshooting guide
- Best practices for writing stories
- Advanced configuration options
- Example usage sessions
- VS Code tasks integration
- Safety and rollback procedures

### 4. `scripts/ralph/test_copilot.py` (NEW)
**Purpose:** Test script to verify Copilot CLI integration

**Features:**
- Tests if `copilot` command is accessible
- Sends test prompt to verify CLI communication
- Validates PRD file structure
- Checks prompt template for required placeholders
- Provides clear success/failure feedback

## Key Improvements

### 1. Direct Copilot CLI Integration
- **Before**: Attempted to use `gh copilot suggest` which may not be available
- **After**: Uses `copilot` command directly (available in VS Code terminal)
- **Benefit**: Works seamlessly in VS Code environment

### 2. Better Prompt Engineering
- **Before**: Basic prompt with minimal instructions
- **After**: Comprehensive prompt with:
  - Full PRD context (all stories and their status)
  - Story ID, title, description, and acceptance criteria
  - Project architecture and conventions
  - Previous progress context
  - Explicit output format requirements
  - Examples of correct code block formats
- **Benefit**: Copilot generates more accurate, actionable code

### 3. Enhanced Error Handling
- **Before**: Generic error messages
- **After**: Specific troubleshooting guidance including:
  - How to verify Copilot CLI access
  - Preview of Copilot response when parsing fails
  - Hints for common issues
  - Step-by-step fixes
- **Benefit**: Easier to diagnose and fix issues

### 4. Improved Logging
- **Before**: Simple emoji-based status updates
- **After**: Structured output with:
  - Visual separators
  - Counts of files/edits found
  - Detailed success/failure messages
  - Response previews for debugging
- **Benefit**: Better visibility into what Ralph is doing

### 5. Comprehensive Documentation
- **Before**: Basic README with setup instructions
- **After**: Complete usage guide covering:
  - How Ralph works internally
  - Best practices for writing stories
  - Troubleshooting common issues
  - Advanced configuration
  - Integration examples
- **Benefit**: Users can effectively use and customize Ralph

## How to Use the Updated Ralph

### Quick Start
1. Open project in VS Code
2. Open integrated terminal (Ctrl+`)
3. Verify Copilot CLI: `copilot --help`
4. Test setup: `python scripts/ralph/test_copilot.py`
5. Run Ralph: `python scripts/ralph/ralph.py`

### Verification Steps
```powershell
# 1. Test Copilot CLI access
copilot

# 2. Run integration test
python scripts/ralph/test_copilot.py

# 3. Run Ralph for 1 iteration (test)
python scripts/ralph/ralph.py 1

# 4. Check results
git log --oneline
cat progress.txt
```

### Expected Behavior
1. Ralph reads next incomplete story from PRD
2. Builds comprehensive prompt with full context
3. Sends prompt to Copilot CLI
4. Parses response for code blocks (filepath/edit formats)
5. Applies changes to codebase
6. Runs quality checks
7. Commits changes with descriptive message
8. Updates PRD to mark story complete
9. Moves to next story

## PRD Format

Ralph expects `prd.json` in this format:
```json
{
  "branchName": "feature/your-feature-name",
  "userStories": [
    {
      "id": "story-1",
      "title": "Story title",
      "description": "Detailed description of what to build",
      "acceptanceCriteria": "How to verify it's complete",
      "passes": false,
      "priority": 1
    }
  ]
}
```

## Code Block Formats

Ralph parses these formats from Copilot responses:

### Complete File Creation/Update
````markdown
```filepath: src/api/routes.py
from fastapi import APIRouter

router = APIRouter()

@router.get("/health")
async def health_check():
    return {"status": "healthy"}
```
````

### Targeted File Edit
````markdown
```edit: src/main.py
SEARCH:
old_code_here

REPLACE:
new_code_here
```
````

## Troubleshooting

### Issue: "Copilot CLI not found"
**Solution**: Run from VS Code terminal where `copilot` command is available

### Issue: "No file changes detected"
**Cause**: Copilot didn't use correct code block format
**Solution**: 
1. Check the response preview in output
2. Refine story description to be more specific
3. Update prompt template if needed

### Issue: Story fails repeatedly
**Solution**:
1. Break story into smaller pieces
2. Add more specific acceptance criteria
3. Manually implement and mark complete in PRD

## Testing the Changes

### Run the test script:
```powershell
python scripts/ralph/test_copilot.py
```

Expected output:
```
Testing Copilot CLI integration...
============================================================

1. Checking if 'copilot' command is available...
   ✅ Copilot CLI is available!

2. Testing simple prompt with Copilot CLI...
   ✅ Successfully received response from Copilot CLI!

3. Checking for prd.json file...
   ✅ Found prd.json at C:\...\prd.json
   ✅ PRD branch: feature/master-control-panel
   ✅ Total stories: 15
   ✅ Incomplete stories: 8

4. Checking prompt template...
   ✅ Found prompt template at scripts/ralph/prompt.md
   ✅ All required placeholders present

============================================================
✅ All tests passed! Ralph is ready to use.
============================================================
```

## Benefits of This Update

1. ✅ **Actually uses Copilot CLI** - No more fake/stubbed AI calls
2. ✅ **Better prompts** - Copilot gets full PRD context for better decisions
3. ✅ **Makes real changes** - Code is actually written to files
4. ✅ **Better error handling** - Clear troubleshooting guidance
5. ✅ **Comprehensive docs** - Users know how to use and troubleshoot
6. ✅ **Testable** - Includes test script to verify setup
7. ✅ **Production-ready** - Handles retries, errors, and edge cases

## Next Steps

1. Test the integration: `python scripts/ralph/test_copilot.py`
2. Review your PRD stories for clarity and specificity
3. Run Ralph on a test story: `python scripts/ralph/ralph.py 1`
4. Review the changes: `git diff`
5. If successful, run unlimited: `python scripts/ralph/ralph.py`

## Notes

- Ralph must be run from a **VS Code terminal** where the `copilot` command is available
- Each story should be **small and focused** (< 15 minutes to implement)
- Ralph will **retry failed stories** up to 3 times before marking them as failed
- All changes are **committed to git** so you can always rollback
- **Progress is tracked** in `progress.txt` for debugging

## Support

For issues:
1. Run test script: `python scripts/ralph/test_copilot.py`
2. Check progress log: `cat progress.txt`
3. Review git history: `git log --oneline`
4. Read troubleshooting in `README-USAGE.md`
