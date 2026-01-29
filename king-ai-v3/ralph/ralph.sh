                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              #!/bin/bash

# Ralph: Autonomous AI Agent Loop
# Based on Geoffrey Huntley's Ralph pattern and Ryan Carson's implementation

set -e

# Add Node.js to PATH for Copilot CLI (WSL/Windows compatibility)
export PATH="$PATH:/mnt/c/Program\ Files/nodejs"

# Configuration
MAX_ITERATIONS=${1:-100}  # Default higher for completion
PRD_FILE="prd.json"
PROGRESS_FILE="progress.txt"
# Support both v2 and v3 paths
if [ -f "ralph/prompt.md" ]; then
    PROMPT_FILE="ralph/prompt.md"
else
    PROMPT_FILE="scripts/ralph/prompt.md"
fi

# Check prerequisites
if ! command -v python >/dev/null 2>&1; then
    echo "python is required but not installed. Aborting."
    exit 1
fi
command -v git >/dev/null 2>&1 || { echo "git is required but not installed. Aborting."; exit 1; }
# Assuming copilot CLI is available, check if needed

# Check if prd.json exists
if [ ! -f "$PRD_FILE" ]; then
    echo "prd.json not found. Please create it first using the PRD skill."
    exit 1
fi

# Get branch name from prd.json
BRANCH_NAME=$(python -c "import json; print(json.load(open('$PRD_FILE'))['branchName'])" 2>/dev/null)
if [ "$BRANCH_NAME" = "null" ] || [ -z "$BRANCH_NAME" ]; then
    echo "branchName not found in prd.json"
    exit 1
fi

# Create and checkout feature branch
if ! git show-ref --verify --quiet refs/heads/"$BRANCH_NAME"; then
    git checkout -b "$BRANCH_NAME"
else
    git checkout "$BRANCH_NAME"
fi

# Initialize progress.txt if it doesn't exist
if [ ! -f "$PROGRESS_FILE" ]; then
    echo "# Ralph Progress Log" > "$PROGRESS_FILE"
    echo "Started: $(date)" >> "$PROGRESS_FILE"
    echo "" >> "$PROGRESS_FILE"
fi

iteration=1
while [ $iteration -le $MAX_ITERATIONS ]; do
    echo "=== Ralph Iteration $iteration ==="

    # Find highest priority incomplete story
    STORY=$(python -c "
import json, base64
data = json.load(open('$PRD_FILE'))
incomplete_stories = [s for s in data['userStories'] if not s['passes']]
if not incomplete_stories:
    print('')
else:
    print(base64.b64encode(json.dumps(incomplete_stories[0]).encode()).decode())
" 2>/dev/null)

    if [ -z "$STORY" ]; then
        echo "<promise>COMPLETE</promise>"
        echo "All stories completed!"
        break
    fi

    # Decode story
    STORY_JSON=$(echo "$STORY" | base64 -d)
    STORY_ID=$(python -c "import json, base64; story = json.loads(base64.b64decode('$STORY').decode()); print(story['id'])" 2>/dev/null)
    STORY_TITLE=$(python -c "import json, base64; story = json.loads(base64.b64decode('$STORY').decode()); print(story['title'])" 2>/dev/null)
    STORY_DESCRIPTION=$(python -c "import json, base64; story = json.loads(base64.b64decode('$STORY').decode()); print(story['description'])" 2>/dev/null)
    STORY_ACCEPTANCE=$(python -c "import json, base64; story = json.loads(base64.b64decode('$STORY').decode()); print(story['acceptanceCriteria'])" 2>/dev/null)

    echo "Working on story: $STORY_TITLE"

    # Create prompt for this iteration
    ITERATION_PROMPT=$(cat "$PROMPT_FILE")
    ITERATION_PROMPT="${ITERATION_PROMPT//\{\{STORY_TITLE\}\}/$STORY_TITLE}"
    ITERATION_PROMPT="${ITERATION_PROMPT//\{\{STORY_DESCRIPTION\}\}/$STORY_DESCRIPTION}"
    ITERATION_PROMPT="${ITERATION_PROMPT//\{\{STORY_ACCEPTANCE\}\}/$STORY_ACCEPTANCE}"

    # Add progress context
    if [ -f "$PROGRESS_FILE" ]; then
        PROGRESS_CONTEXT=$(tail -20 "$PROGRESS_FILE")
        ITERATION_PROMPT="$ITERATION_PROMPT

Previous learnings:
$PROGRESS_CONTEXT"
    fi

    # Save iteration prompt to temp file
    TEMP_PROMPT="/tmp/ralph_prompt_$iteration.md"
    echo "$ITERATION_PROMPT" > "$TEMP_PROMPT"

    # Run AI code generation
    echo "Running AI code generation for story $STORY_ID..."
    GENERATED_CODE=$(echo "$ITERATION_PROMPT" | python scripts/ralph/generate_code.py)

    # Check if generation succeeded
    if [ $? -eq 0 ] && [ -n "$GENERATED_CODE" ]; then
        SUCCESS=true
        echo "Code generation successful"

        # Apply the generated code
        echo "Applying generated code..."
        echo "$GENERATED_CODE" > "/tmp/ralph_code_$iteration.txt"

        # Parse and apply code changes (simple implementation - could be enhanced)
        python -c "
import re
import os

with open('/tmp/ralph_code_$iteration.txt', 'r') as f:
    content = f.read()

# Simple file creation/modification logic
files = re.findall(r'```\s*([^`\n]+)\s*\n```', content, re.MULTILINE | re.DOTALL)
for file_match in re.finditer(r'```\s*([^`\n]+)\s*\n(.*?)\n```', content, re.MULTILINE | re.DOTALL):
    file_path, code = file_match.groups()
    file_path = file_path.strip()
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Write the code
    with open(file_path, 'w') as f:
        f.write(code.strip())
    
    print(f'Created/updated file: {file_path}')
" 2>/dev/null

    else
        SUCCESS=false
        echo "Code generation failed"
    fi

    if [ "$SUCCESS" = true ]; then
        # Run quality checks
        echo "Running quality checks..."
        # Add your quality check commands here
        # e.g., npm run typecheck, npm test, etc.

        # Assuming checks pass
        CHECKS_PASS=true

        if [ "$CHECKS_PASS" = true ]; then
            # Commit changes
            git add .
            git commit -m "Ralph: Complete story $STORY_ID - $STORY_TITLE"

            # Update prd.json
            python -c "
import json
data = json.load(open('$PRD_FILE'))
for story in data['userStories']:
    if story['id'] == '$STORY_ID':
        story['passes'] = True
        break
with open('${PRD_FILE}.tmp', 'w') as f:
    json.dump(data, f, indent=2)
" 2>/dev/null
            mv "${PRD_FILE}.tmp" "$PRD_FILE"

            # Append to progress
            echo "Iteration $iteration: Completed story $STORY_ID" >> "$PROGRESS_FILE"
            echo "Success: $STORY_TITLE" >> "$PROGRESS_FILE"
            echo "" >> "$PROGRESS_FILE"
        else
            echo "Quality checks failed for story $STORY_ID" >> "$PROGRESS_FILE"
            echo "Failed: $STORY_TITLE" >> "$PROGRESS_FILE"
            echo "" >> "$PROGRESS_FILE"
        fi
    else
        echo "Implementation failed for story $STORY_ID" >> "$PROGRESS_FILE"
        echo "Failed: $STORY_TITLE" >> "$PROGRESS_FILE"
        echo "" >> "$PROGRESS_FILE"
    fi

    # Clean up temp file
    rm -f "$TEMP_PROMPT"

    iteration=$((iteration + 1))
done

if [ $iteration -gt $MAX_ITERATIONS ]; then
    echo "Reached maximum iterations ($MAX_ITERATIONS) without completing all stories."
fi