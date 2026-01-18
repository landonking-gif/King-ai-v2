#!/bin/bash
# Update MCP Server from YAML file
# Usage: ./scripts/update_mcp_server.sh <yaml_file>

set -e

YAML_FILE="$1"

if [ -z "$YAML_FILE" ]; then
    echo "Usage: $0 <yaml_file>"
    echo ""
    echo "Example:"
    echo "  $0 examples/firecrawl_mcp_server.yaml"
    exit 1
fi

if [ ! -f "$YAML_FILE" ]; then
    echo "Error: File not found: $YAML_FILE"
    exit 1
fi

# Extract tool_id from YAML
TOOL_ID=$(grep "^tool_id:" "$YAML_FILE" | cut -d: -f2 | tr -d ' ')

if [ -z "$TOOL_ID" ]; then
    echo "Error: Could not extract tool_id from $YAML_FILE"
    exit 1
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  MCP Server Update Workflow"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Tool ID: $TOOL_ID"
echo "Config File: $YAML_FILE"
echo ""

# Show current registration
echo "Current registration:"
kautilya mcp list | grep -A 1 "$TOOL_ID" || echo "  (not registered)"
echo ""

# Ask for confirmation
read -p "Proceed with update? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Update cancelled."
    exit 0
fi

# Unregister
echo ""
echo "Step 1: Unregistering $TOOL_ID..."
kautilya mcp unregister "$TOOL_ID" 2>/dev/null || echo "  (server not registered, continuing...)"

# Re-import
echo ""
echo "Step 2: Importing from $YAML_FILE..."
kautilya mcp import "$YAML_FILE" --format yaml

# Verify
echo ""
echo "Step 3: Verifying update..."
kautilya mcp list | grep -A 1 "$TOOL_ID"

echo ""
echo "✓ Update complete!"
echo ""
