#!/bin/bash

# Auto Ralph: Fully Automated Feature Implementation
# Takes a feature description and handles PRD generation, conversion, and execution

set -e

# Add Node.js to PATH
export PATH="$PATH:/mnt/c/Program Files/nodejs"

# Check prerequisites
command -v copilot >/dev/null 2>&1 || { echo "GitHub Copilot CLI is required. Install from https://github.com/github/copilot-cli"; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "python3 is required."; exit 1; }
command -v git >/dev/null 2>&1 || { echo "git is required."; exit 1; }

if [ $# -eq 0 ]; then
    echo "Usage: $0 \"feature description\" or $0 file.txt"
    exit 1
fi

if [ -f "$1" ]; then
    FEATURE_DESCRIPTION=$(cat "$1")
else
    FEATURE_DESCRIPTION="$1"
fi
TIMESTAMP=$(date +%s)
PRD_MD_FILE="tasks/prd-${TIMESTAMP}.md"
PRD_JSON_FILE="prd.json"

echo "=== Auto Ralph: Starting with feature: $FEATURE_DESCRIPTION ==="

# Ensure tasks directory exists
mkdir -p tasks

# Step 1: Generate PRD using Copilot
echo "Generating PRD..."
PRD_PROMPT="Generate a Product Requirement Document (PRD) for the following feature description. Output in markdown format with these sections:
- Title
- Overview
- Goals and Objectives
- Target Audience
- User Stories (with acceptance criteria)
- Technical Requirements
- Timeline
- Success Metrics

Feature: $FEATURE_DESCRIPTION"

echo "$PRD_PROMPT" | copilot --allow-all-tools > "$PRD_MD_FILE" 2>/dev/null || {
    echo "Failed to generate PRD. Please check Copilot CLI setup."
    exit 1
}

echo "PRD generated: $PRD_MD_FILE"

# Step 2: Convert PRD to JSON using Copilot
echo "Converting PRD to JSON..."
CONVERT_PROMPT="Convert the following markdown PRD to JSON format suitable for Ralph automation. Use this structure:
{
  \"branchName\": \"feature/[feature-name]\",
  \"userStories\": [
    {
      \"id\": \"story-1\",
      \"title\": \"string\",
      \"description\": \"string\",
      \"acceptanceCriteria\": \"string\",
      \"passes\": false,
      \"priority\": 1
    }
  ]
}

PRD Content:
$(cat "$PRD_MD_FILE")"

echo "$CONVERT_PROMPT" | copilot --allow-all-tools > "$PRD_JSON_FILE" 2>/dev/null || {
    echo "Failed to convert PRD to JSON. Please check the generated PRD."
    exit 1
}

# Validate JSON
python3 -c "import json; json.load(open('$PRD_JSON_FILE'))" || {
    echo "Generated JSON is invalid. Please check $PRD_JSON_FILE"
    exit 1
}

echo "PRD converted to JSON: $PRD_JSON_FILE"

# Step 3: Run Ralph until completion
echo "Running Ralph until completion..."
./scripts/ralph/ralph.sh 1000  # High number to ensure completion

echo "=== Auto Ralph: Complete ==="