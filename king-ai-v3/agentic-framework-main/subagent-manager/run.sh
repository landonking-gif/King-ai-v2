#!/bin/bash
# Startup script for Subagent Manager service

set -e

echo "Starting Subagent Manager script"

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

# Run the service
echo "Starting Subagent Manager on port 8001..."
uvicorn service.main:app --host 0.0.0.0 --port 8001
