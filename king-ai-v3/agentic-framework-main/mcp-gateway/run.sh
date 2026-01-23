#!/bin/bash
# Startup script for MCP Gateway service

set -e

echo "Starting MCP Gateway script"

# Activate virtual environment if exists
if [ -d "/home/ubuntu/agentic-framework-main/.venv" ]; then
    echo "Activating venv..."
    source /home/ubuntu/agentic-framework-main/.venv/bin/activate
    echo "Venv activated, python: $(which python)"
else
    echo "Venv not found"
    exit 1
fi

# Set PYTHONPATH to include the project root (parent directory)
export PYTHONPATH="$PWD/..:$PYTHONPATH"
echo "PYTHONPATH: $PYTHONPATH"

# Check if Redis is running
if ! redis-cli ping > /dev/null 2>&1; then
    echo "WARNING: Redis is not running. Rate limiting will fail."
    echo "Start Redis with: redis-server"
fi

# Run the service
echo "Starting MCP Gateway on port 8080..."
python -m mcp_gateway.service.main

