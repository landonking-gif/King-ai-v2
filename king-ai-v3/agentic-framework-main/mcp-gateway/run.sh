#!/bin/bash
# Startup script for MCP Gateway service

set -e

echo "Starting MCP Gateway script"

# Change to script directory
cd "$(dirname "$0")"

# Activate virtual environment if exists
if [ -d "/home/ubuntu/agentic-framework-main/.venv" ]; then
    echo "Activating venv..."
    source /home/ubuntu/agentic-framework-main/.venv/bin/activate
    echo "Venv activated, python: $(which python)"
else
    echo "Venv not found, exiting"
    exit 1
fi

# Set PYTHONPATH to include current and parent directory
export PYTHONPATH="$PWD:$PWD/..:$PYTHONPATH"
echo "PYTHONPATH: $PYTHONPATH"

# Check if Redis is running
if ! redis-cli ping > /dev/null 2>&1; then
    echo "WARNING: Redis is not running. Rate limiting will fail."
    echo "Start Redis with: redis-server"
fi

# Run the service
echo "Starting MCP Gateway on port 8080..."
python -m service.main

